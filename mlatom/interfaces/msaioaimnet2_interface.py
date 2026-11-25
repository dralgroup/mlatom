import os
import json
import uuid
import shutil
import tempfile
import logging
from typing import Any, Dict, List, Union
import numpy as np
import torch
try:
    import aimnetex
except ImportError as error:
    raise ImportError('MS-AIOAIMNet2 models, including OMNI-P3x, need the aimnetex package: '
                      'pip install "git+https://github.com/mmartyka/aimnetex"') from error

import mlatom as ml
from mlatom import model_cls
from mlatom.model_cls import ml_model, torch_model, hyperparameter
from mlatom import data


def _quiet_train_run(*args, **kwargs):
    import logging
    from aimnetex.train.train import run as train_run
    logging.disable(logging.CRITICAL)
    try:
        train_run(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)


def _frozen_oscillator_output(**kwargs):
    from aimnetex.modules import Output

    # Freeze before the trainer wraps the model in DistributedDataParallel.
    return Output(**kwargs).requires_grad_(False)


def _is_distributed_launcher():
    '''
    True when this process was started by an external distributed launcher
    (torchrun / torchelastic), i.e. it is already one rank of a process group
    that may span several nodes. In that case the interface must NOT spawn its
    own workers; it joins the launcher's process group instead. The launcher
    sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT in the env.
    '''
    env = os.environ
    return ('LOCAL_RANK' in env and 'WORLD_SIZE' in env
            and ('RANK' in env or 'TORCHELASTIC_RUN_ID' in env))


class msaioaimnet2(ml_model, torch_model):
    '''
    MLatom interface for the MS-AIOAIMNet2 multi-state neural network model.

    Supports training, prediction, saving, and loading of models that predict
    multi-state energies, energy gradients, hessians, dipole moments and, when trained
    with ``train_osc=True``, S0->S_n oscillator strengths (a rotation-invariant scalar head).

    Arguments:
        model_file (str, optional): Path to a saved model directory or .jpt file.
        device (str, optional): 'cpu' or 'cuda'. Auto-detected if not specified.
        hyperparameters (dict, optional): Override default hyperparameters.
        verbose (int, optional): 0=silent, 1=normal, 2=debug.
        model_config (str, optional): Path to model architecture YAML.
        train_config (str, optional): Path to training configuration YAML.
    '''

    hyperparameters = model_cls.hyperparameters({
        # Training
        'batch_size':                   hyperparameter(value=64, minval=8, maxval=512, optimization_space='linear', dtype=int),
        'max_epochs':                   hyperparameter(value=1000000, minval=100, maxval=1000000, optimization_space='log', dtype=int),
        'learning_rate':                hyperparameter(value=3e-4, minval=1e-5, maxval=0.01, optimization_space='log'),
        'early_stopping_learning_rate': hyperparameter(value=1.0e-5, minval=1e-7, maxval=1e-4, optimization_space='log'),
        'lr_reduce_patience':           hyperparameter(value=10, minval=5, maxval=50, optimization_space='linear', dtype=int),
        'lr_reduce_factor':             hyperparameter(value=0.5, minval=0.1, maxval=0.9, optimization_space='linear'),
        'warmup_epochs':                hyperparameter(value=0, minval=0, maxval=100, optimization_space='linear', dtype=int),
        'weight_decay':                 hyperparameter(value=1e-5),
        'val_fraction':                 hyperparameter(value=0.1),
        # Loss weights
        'energy_weight':                hyperparameter(value=1.0, minval=0.1, maxval=10.0, optimization_space='linear'),
        'force_weight':                 hyperparameter(value=0.3, minval=0.01, maxval=5.0, optimization_space='linear'),
        'dipole_weight':                hyperparameter(value=0.1, minval=0.0, maxval=5.0, optimization_space='linear'),
        'osc_weight':                   hyperparameter(value=0.5, minval=0.0, maxval=5.0, optimization_space='linear'),
        # per-atom charge targets (train(charges_property=...)); plain per-atom MSE in e^2
        'charges_weight':               hyperparameter(value=1.0, minval=0.0, maxval=100.0, optimization_space='linear'),
        # Loss functions
        'energy_loss_fn':               hyperparameter(value='aimnetex.train.loss.energy_loss_fn'),
        'force_loss_fn':                hyperparameter(value='aimnetex.train.loss.peratom_loss_fn'),
        'dipole_loss_fn':               hyperparameter(value='aimnetex.train.loss.dipole_loss_fn'),
        'charges_loss_fn':              hyperparameter(value='aimnetex.train.loss.charges_loss_fn'),
        # Keyword arguments forwarded to the energy loss function, e.g. {'delta': 0.25}
        # for aimnetex.train.loss.huber_energy_loss_fn (the other losses take fixed kwargs)
        'energy_loss_kwargs':           hyperparameter(value={}),
        # Model architecture
        'nfeature':                     hyperparameter(value=16, minval=4, maxval=128, optimization_space='linear', dtype=int),
        'LoT_dim':                      hyperparameter(value=1, minval=1, maxval=16, optimization_space='linear', dtype=int),
        'd2features':                   hyperparameter(value=True),
        'ncomb_v':                      hyperparameter(value=12, minval=4, maxval=32, optimization_space='linear', dtype=int),
        'num_charge_channels':          hyperparameter(value=1),
        'hidden':                       hyperparameter(value=[[512, 380], [512, 380], [512, 380, 380]]),
        'aim_size':                     hyperparameter(value=256, minval=64, maxval=1024, optimization_space='linear', dtype=int),
        'rc_s':                         hyperparameter(value=5.0, minval=3.0, maxval=10.0, optimization_space='linear'),
        'lrcoulomb_rc':                 hyperparameter(value=5.0, minval=3.0, maxval=10.0, optimization_space='linear'),
        'nshifts_s':                    hyperparameter(value=16, minval=8, maxval=32, optimization_space='linear', dtype=int),
        'energy_mlp_hidden':            hyperparameter(value=[128, 128]),
        'osc_mlp_hidden':               hyperparameter(value=[128, 128]),
        # SAE mode: 'linreg' (single per-element table) or 'linreg_per_lot'
        # (one per-element table per unique lot_descriptor row in the training set).
        'sae_mode':                     hyperparameter(value='linreg'),
        # Periodic saving
        'save_every_n_epochs':          hyperparameter(value=1, minval=0, maxval=1000, dtype=int),
    })

    model_file = None
    program = 'AIMNet2'
    meta_data = {'genre': 'neural network'}
    verbose = 1

    _aimnet_dir = os.path.dirname(os.path.abspath(aimnetex.__file__))

    def __init__(
        self,
        model_file: str = None,
        device: str = None,
        hyperparameters: Union[Dict[str, Any], model_cls.hyperparameters] = {},
        verbose: int = 1,
        model_config: str = None,
        train_config: str = None,
        nstates: int = None,
        pre_sharded: bool = False,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.hyperparameters = self.__class__.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)

        self.verbose = verbose

        # When True (under a multi-node launcher), each rank's train() receives
        # only its own data shard; the interface does NOT re-shard, and the SAE
        # is computed globally via all-reduce. See _train_distributed.
        self.pre_sharded = pre_sharded

        self.model_config_path = model_config or os.path.join(self._aimnet_dir, 'model_config.yaml')
        self.train_config_path = train_config or os.path.join(self._aimnet_dir, 'train_config.yaml')

        self._weights_path = None
        self._jpt_path = None
        self._sae_path = None
        self._model_cfg = None
        self._calc = None

        if model_file:
            if os.path.exists(model_file):
                self.load(model_file)
            else:
                if self.verbose >= 1:
                    print(f'The trained MS-AIOAIMNet2 model will be saved in {model_file}')
            self.model_file = model_file

    @property
    def calc(self):
        '''Lazy-load the AIMNet2Calculator for inference.'''
        if self._calc is None:
            if self._jpt_path is None or not os.path.exists(self._jpt_path):
                raise RuntimeError(
                    'No compiled model (.jpt) available. '
                    'Call save() after training, or load() a saved model.'
                )
            from aimnetex.calculators import AIMNet2Calculator
            self._calc = AIMNet2Calculator(model=self._jpt_path)
        return self._calc

    def __getstate__(self):
        '''Exclude the JIT calculator from pickling; it will be lazy-loaded again.'''
        state = self.__dict__.copy()
        state['_calc'] = None
        return state

    def _compute_sae(self, h5_path, output_path, max_samples=100000):
            '''Compute self-atomic energies from HDF5 dataset (replicates calc_sae.py logic).'''
            from aimnetex.data import SizeGroupedDataset

            ds = SizeGroupedDataset(h5_path, keys=['numbers', 'energy'])
            # Record all unique atom types before subsampling (memory-efficient)
            all_atom_types = set()
            for g in ds.groups:
                all_atom_types.update(np.unique(g["numbers"]).tolist())

            if max_samples > 0 and len(ds) > max_samples:
                ds = ds.random_split(max_samples / len(ds))[0]

            sae_initial = ds.apply_peratom_shift('energy', '_energy')

            # Remove 2nd-98th percentile outliers and recompute
            energy = ds.concatenate('_energy')
            pct1, pct2 = np.percentile(energy, [2, 98])
            for _n, g in list(ds.items()):
                mask = (g['_energy'] > pct1) & (g['_energy'] < pct2)
                g = g.sample(mask)
                if not len(g):
                    ds._data.pop(_n)
                else:
                    ds[_n] = g
            if len(ds) > 0:
                sae = ds.apply_peratom_shift('energy', '_energy')
            else:
                sae = sae_initial

            # Fill in atom types lost during outlier removal using pre-outlier values
            for z in all_atom_types:
                if z not in sae and z in sae_initial:
                    sae[z] = sae_initial[z]
                    logging.warning(
                        f"SAE: atom type Z={z} lost during outlier removal, using pre-outlier value"
                    )

            # Fill in atom types lost during subsampling by recomputing from full dataset
            missing = all_atom_types - set(sae.keys())
            if missing:
                logging.warning(
                    f"SAE: atom types {missing} lost during subsampling, recomputing from full dataset"
                )
                ds_full = SizeGroupedDataset(h5_path, keys=['numbers', 'energy'])
                sae_full = ds_full.apply_peratom_shift('energy', '_energy')
                for z in missing:
                    sae[z] = sae_full[z]
                del ds_full

            with open(output_path, 'w') as f:
                for k, v in sae.items():
                    f.write(f'{k}: {v}\n')

            return sae

    def _compute_sae_per_lot(self, h5_path, output_path, max_samples=100000, num_types=64,
                              fixed_lot_descriptors=None):
        '''Compute per-(LoT, element) SAE: one linreg per unique lot_descriptor row.

        If ``fixed_lot_descriptors`` is given (list[list[int]]), the returned
        table uses exactly that layout: each row in the fixed list becomes a
        row in the SAE table in that order. Descriptors that appear in the
        training data but not in ``fixed_lot_descriptors`` raise an error.
        Descriptors in the fixed list that are absent from the training data
        get an all-zeros row (the model's checkpoint values for them are
        preserved through load_state_dict; they just aren't refit). This is
        used to keep AL fine-tuning compatible with a pretrained checkpoint.

        Writes a yaml of the form:

            lots:
              - descriptor: [...]
                shifts: {z: shift, ...}
              - descriptor: [...]
                shifts: {z: shift, ...}

        Returns a dict with:
            mode            : 'linreg_per_lot'
            lot_descriptors : list[list[int]]   length n_lots
            sap_table       : list[list[float]] [n_lots, num_types]   (zero-padded)
            num_lots, num_types
        '''
        import yaml as _yaml
        from aimnetex.data import SizeGroupedDataset

        ds = SizeGroupedDataset(h5_path, keys=['numbers', 'energy', 'lot_descriptor'])

        # Unique LoT descriptors across the whole dataset
        all_lots = np.concatenate(
            [np.asarray(g['lot_descriptor']) for g in ds.groups], axis=0
        )
        data_unique_lots = np.unique(all_lots, axis=0)

        if fixed_lot_descriptors is not None:
            unique_lots = np.asarray(fixed_lot_descriptors, dtype=all_lots.dtype)
            # All data descriptors must be present in the fixed layout
            fixed_set = {tuple(int(x) for x in row) for row in unique_lots}
            missing = [tuple(int(x) for x in row) for row in data_unique_lots
                       if tuple(int(x) for x in row) not in fixed_set]
            if missing:
                raise ValueError(
                    f"Training data contains lot_descriptor(s) {missing} not in "
                    f"the pretrained model's table {sorted(fixed_set)}. "
                    "Fine-tuning cannot introduce new LoTs without breaking the "
                    "checkpoint's LoTAtomicShift shape."
                )
        else:
            unique_lots = data_unique_lots
        n_lots = len(unique_lots)

        # Atom types present anywhere
        all_atom_types = set()
        for g in ds.groups:
            all_atom_types.update(np.unique(g['numbers']).tolist())
        ntyp = max(all_atom_types) + 1
        if ntyp > num_types:
            raise ValueError(
                f"Dataset contains atomic number {ntyp - 1} but num_types={num_types}"
            )

        sap_table = np.zeros((n_lots, num_types), dtype=np.float64)
        lots_entries = []

        for i_lot, lot_desc in enumerate(unique_lots):
            # Mask each group to the molecules with this descriptor
            ds_lot = SizeGroupedDataset()
            for n, g in ds.items():
                mask = np.all(np.asarray(g['lot_descriptor']) == lot_desc, axis=-1)
                if mask.any():
                    ds_lot[n] = g.sample(mask)
            if len(ds_lot) == 0:
                # Pretrained LoT with no fine-tuning data: keep the row as
                # zeros so the table layout still matches the checkpoint, and
                # emit an empty entry in the yaml.
                lots_entries.append({
                    'descriptor': [int(x) for x in lot_desc],
                    'shifts': {},
                })
                if self.verbose >= 1:
                    logging.warning(
                        f"SAE[lot={lot_desc.tolist()}]: no training data for this "
                        "descriptor; keeping zero shifts (pretrained checkpoint values "
                        "will load via load_state_dict)."
                    )
                continue

            present_atom_types = set()
            for g in ds_lot.groups:
                present_atom_types.update(np.unique(g['numbers']).tolist())

            if max_samples > 0 and len(ds_lot) > max_samples:
                ds_lot = ds_lot.random_split(max_samples / len(ds_lot))[0]

            sae_initial = ds_lot.apply_peratom_shift('energy', '_energy')

            # Percentile trim and re-fit
            energy = ds_lot.concatenate('_energy')
            pct1, pct2 = np.percentile(energy, [2, 98])
            for n, g in list(ds_lot.items()):
                m = (g['_energy'] > pct1) & (g['_energy'] < pct2)
                g = g.sample(m)
                if not len(g):
                    ds_lot._data.pop(n)
                else:
                    ds_lot[n] = g

            if len(ds_lot) > 0:
                sae_lot = ds_lot.apply_peratom_shift('energy', '_energy')
            else:
                sae_lot = sae_initial

            # Fill atom types lost during outlier removal from pre-outlier values
            for z in present_atom_types:
                if z not in sae_lot and z in sae_initial:
                    sae_lot[z] = sae_initial[z]
                    logging.warning(
                        f"SAE[lot={lot_desc.tolist()}]: atom type Z={z} lost during outlier "
                        "removal, using pre-outlier value"
                    )

            for z, v in sae_lot.items():
                sap_table[i_lot, int(z)] = float(v)

            lots_entries.append({
                'descriptor': [int(x) for x in lot_desc],
                'shifts': {int(z): float(v) for z, v in sae_lot.items()},
            })

        with open(output_path, 'w') as f:
            _yaml.safe_dump({'lots': lots_entries}, f, sort_keys=False)

        return {
            'mode': 'linreg_per_lot',
            'lot_descriptors': [[int(x) for x in row] for row in unique_lots],
            'sap_table': sap_table.tolist(),
            'num_lots': n_lots,
            'num_types': num_types,
        }

    # ------------------------------------------------------------------
    # Distributed SAE (used only in pre_sharded multi-node training).
    #
    # The per-element self-atomic-energy fit is ordinary least squares
    # ``F @ s ~= E`` (F[i,z] = count of element z in molecule i), the same fit
    # aimnet's SizeGroupedDataset.apply_peratom_shift performs locally. Its
    # sufficient statistics G = FᵀF and b = FᵀE are sums over molecules, so
    # summing them across ranks (all-reduce) yields the exact global fit without
    # any rank holding the full dataset. The 2nd–98th percentile residual trim is
    # reproduced globally via an all-reduced histogram. reduce_sum=None makes
    # every routine fall back to a plain single-process fit.
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_for_sae(ds, num_types, with_lot=False):
        '''Flatten a SizeGroupedDataset into per-molecule count matrix F
        [N, num_types], energies E [N], and (optionally) lot_descriptor rows
        [N, dlen] — all in one consistent molecule ordering.'''
        eye = np.eye(num_types, dtype=np.float64)
        F_blocks, E_blocks, lot_blocks = [], [], []
        for g in ds.values():
            nums = np.asarray(g['numbers'])
            if nums.size and int(nums.max()) >= num_types:
                raise ValueError(
                    f"Dataset contains atomic number {int(nums.max())} but num_types={num_types}"
                )
            F_blocks.append(eye[nums].sum(-2))
            E_blocks.append(np.asarray(g['energy'], dtype=np.float64))
            if with_lot:
                lot_blocks.append(np.asarray(g['lot_descriptor']))
        if F_blocks:
            F = np.concatenate(F_blocks, axis=0)
            E = np.concatenate(E_blocks, axis=0)
            lot = np.concatenate(lot_blocks, axis=0) if with_lot else None
        else:
            F = np.zeros((0, num_types), dtype=np.float64)
            E = np.zeros((0,), dtype=np.float64)
            lot = np.zeros((0, 0), dtype=np.int64) if with_lot else None
        return F, E, lot

    @staticmethod
    def _fit_sae_lstsq(F, E, reduce_sum=None):
        '''Least-squares per-element shifts from sufficient statistics.

        Returns (sap [ntyp], present_idx). ``reduce_sum(arr)`` sums ``arr`` across
        ranks (identity when None), so the same code gives a single-process fit or
        an exact global fit. Restricting the solve to present elements makes it
        equivalent to ``np.linalg.lstsq(F, E)`` on the columns that actually occur.'''
        ntyp = F.shape[1]
        G = F.T @ F                       # [ntyp, ntyp]
        b = F.T @ E                       # [ntyp]
        p = F.sum(0)                      # [ntyp] total atoms of each element
        if reduce_sum is not None:
            G = reduce_sum(G)
            b = reduce_sum(b)
            p = reduce_sum(p)
        sap = np.zeros(ntyp, dtype=np.float64)
        present = np.nonzero(p > 0)[0]
        if len(present):
            sub = np.linalg.lstsq(G[np.ix_(present, present)], b[present], rcond=None)[0]
            sap[present] = sub
        return sap, present

    @staticmethod
    def _global_trim_mask(residuals, lo_pct, hi_pct, reducers, nbins=1000, passes=3):
        '''Boolean mask keeping local ``residuals`` within the GLOBAL
        [lo_pct, hi_pct] percentile band.

        The cut points are located by an all-reduced histogram refined over
        ``passes`` windows: pass 1 spans the full global range; each later pass
        zooms into a window bracketing the cuts (carrying a global count-below
        offset) so the bulk — where the percentiles sit — gets fine resolution
        even when far outliers stretch the range. Converges to np.percentile to
        ~1e-7 in 3 passes. All early-exit tests use GLOBALLY reduced values, so
        every rank issues the identical sequence of collectives.'''
        _sum, _min, _max = reducers['sum'], reducers['min'], reducers['max']
        n = len(residuals)
        total = float(_sum(np.array([float(n)]))[0])
        if total <= 0:
            return np.ones(n, dtype=bool)
        gmin = float(_min(np.array([residuals.min() if n else np.inf]))[0])
        gmax = float(_max(np.array([residuals.max() if n else -np.inf]))[0])
        if not (np.isfinite(gmin) and np.isfinite(gmax)) or gmax <= gmin:
            return np.ones(n, dtype=bool)

        def cuts(win_lo, win_hi):
            below = float(_sum(np.array([float((residuals < win_lo).sum()) if n else 0.0]))[0])
            edges = np.linspace(win_lo, win_hi, nbins + 1)
            h = _sum(np.histogram(residuals, bins=edges)[0].astype(np.float64)
                     if n else np.zeros(nbins, dtype=np.float64))
            frac = (below + np.concatenate([[0.0], np.cumsum(h)])) / total   # global CDF at edges

            def q(pct):
                tf = pct / 100.0
                i = min(max(int(np.searchsorted(frac, tf, side='left')), 1), nbins)
                c0, c1 = frac[i - 1], frac[i]
                e0, e1 = edges[i - 1], edges[i]
                return e0 + (tf - c0) / (c1 - c0) * (e1 - e0) if c1 > c0 else e1

            return q(lo_pct), q(hi_pct)

        win_lo, win_hi = gmin, gmax
        lo_cut, hi_cut = gmin, gmax
        for _ in range(passes):
            lo_cut, hi_cut = cuts(win_lo, win_hi)
            span = max(hi_cut - lo_cut, 1e-12)
            win_lo, win_hi = lo_cut - 0.5 * span, hi_cut + 0.5 * span
        return (residuals > lo_cut) & (residuals < hi_cut)

    def _compute_sae_distributed(self, h5_path, output_path, reducers, num_types=64):
        '''Flat linreg SAE across ranks via all-reduced sufficient statistics.
        Each rank reads its own shard h5; the emitted sae.yaml is identical on all
        ranks. Mirrors _compute_sae (initial fit -> 2/98 residual trim -> refit,
        lost elements filled from the pre-trim fit).'''
        from aimnetex.data import SizeGroupedDataset
        ds = SizeGroupedDataset(h5_path, keys=['numbers', 'energy'])
        F, E, _ = self._flatten_for_sae(ds, num_types)

        sap0, present0 = self._fit_sae_lstsq(F, E, reduce_sum=reducers['sum'])
        resid = E - F @ sap0
        mask = self._global_trim_mask(resid, 2, 98, reducers)
        sap, present = self._fit_sae_lstsq(F[mask], E[mask], reduce_sum=reducers['sum'])

        final = {int(z): float(sap[z]) for z in present}
        for z in present0.tolist():                 # fill types lost in the trim
            final.setdefault(int(z), float(sap0[z]))
        with open(output_path, 'w') as f:
            for z in sorted(final):
                f.write(f'{z}: {final[z]}\n')
        return final

    def _compute_sae_per_lot_distributed(self, h5_path, output_path, reducers,
                                         all_gather_obj, num_types=64,
                                         fixed_lot_descriptors=None):
        '''Per-(LoT, element) SAE across ranks. Same math as _compute_sae_per_lot,
        but each per-LoT fit is summed across ranks and the LoT layout is a global
        union (or the pinned pretrained layout). Emits an identical sae.yaml and an
        identical sae_resolved dict on every rank.'''
        import yaml as _yaml
        from aimnetex.data import SizeGroupedDataset
        ds = SizeGroupedDataset(h5_path, keys=['numbers', 'energy', 'lot_descriptor'])
        F, E, lot = self._flatten_for_sae(ds, num_types, with_lot=True)

        # local unique LoT descriptor rows
        local_lots = np.unique(lot, axis=0) if len(lot) else np.zeros((0, 0), dtype=np.int64)

        if fixed_lot_descriptors is not None:
            unique_lots = np.asarray(fixed_lot_descriptors, dtype=np.int64)
            fixed_set = {tuple(int(x) for x in row) for row in unique_lots}
            local_set = {tuple(int(x) for x in row) for row in local_lots}
            local_missing = sorted(local_set - fixed_set)
            # Agree across ranks (one collective) so a bad shard raises on EVERY
            # rank rather than deadlocking the others in the per-LoT loop below.
            any_missing = float(reducers['max'](np.array([1.0 if local_missing else 0.0]))[0]) > 0
            if any_missing:
                raise ValueError(
                    f"A training shard contains lot_descriptor(s) not in the pretrained "
                    f"model's table {sorted(fixed_set)} (this rank: {local_missing})."
                )
        else:
            # global union of LoT rows across ranks, deterministically ordered
            gathered = all_gather_obj([[int(x) for x in row] for row in local_lots])
            all_rows = [tuple(r) for rank_rows in gathered for r in rank_rows]
            uniq = sorted(set(all_rows))
            if not uniq:
                raise ValueError("No lot_descriptor rows found in any shard.")
            unique_lots = np.asarray(uniq, dtype=np.int64)
        n_lots = len(unique_lots)

        sap_table = np.zeros((n_lots, num_types), dtype=np.float64)
        lots_entries = []
        dlen = unique_lots.shape[1]
        for i_lot, lot_desc in enumerate(unique_lots):
            if len(lot) and lot.shape[1] == dlen:
                m = np.all(lot == lot_desc, axis=-1)
            else:
                m = np.zeros(len(E), dtype=bool)     # this rank has none of this LoT
            F_l, E_l = F[m], E[m]

            sap0, present0 = self._fit_sae_lstsq(F_l, E_l, reduce_sum=reducers['sum'])
            resid = E_l - F_l @ sap0
            trim = self._global_trim_mask(resid, 2, 98, reducers)
            sap, present = self._fit_sae_lstsq(F_l[trim], E_l[trim], reduce_sum=reducers['sum'])

            if len(present0) == 0:                    # LoT with no data on any rank
                lots_entries.append({'descriptor': [int(x) for x in lot_desc], 'shifts': {}})
                continue
            shifts = {int(z): float(sap[z]) for z in present}
            for z in present0.tolist():
                shifts.setdefault(int(z), float(sap0[z]))
            for z, v in shifts.items():
                sap_table[i_lot, int(z)] = float(v)
            lots_entries.append({
                'descriptor': [int(x) for x in lot_desc],
                'shifts': {int(z): float(v) for z, v in sorted(shifts.items())},
            })

        with open(output_path, 'w') as f:
            _yaml.safe_dump({'lots': lots_entries}, f, sort_keys=False)
        return {
            'mode': 'linreg_per_lot',
            'lot_descriptors': [[int(x) for x in row] for row in unique_lots],
            'sap_table': sap_table.tolist(),
            'num_lots': n_lots,
            'num_types': num_types,
        }

    def train(
        self,
        molecular_database: data.molecular_database = None,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        hyperparameters: Union[Dict[str, Any], model_cls.hyperparameters] = {},
        spliting_ratio: float = 0.9,
        save_model: bool = True,
        dipole_property: str = None,
        train_osc: bool = False,
        reset_energy_shifter: bool = True,
        charges_property: str = None,
        drop_osc: bool = False,
    ) -> None:
        '''
        Train the MS-AIOAIMNet2 model.

        Arguments:
            molecular_database: Training data with electronic states.
            property_to_learn: Property name for energies (default 'energy').
            xyz_derivative_property_to_learn: Property name for gradients (e.g. 'energy_gradients').
            validation_molecular_database: Validation data or 'sample_from_molecular_database'.
            hyperparameters: Override hyperparameters for this training run.
            spliting_ratio: Fraction of data for training (rest for validation).
            save_model: Whether to save the model after training.
            dipole_property: Property name for dipole moments (e.g. 'dipole_moment'), or None to skip.
            train_osc: Also train the oscillator-strength head on the molecules'
                ``oscillator_strengths`` (S0->S_n, one entry per excited state), weighted by
                the ``osc_weight`` hyperparameter. When False, the weights of any existing
                head remain fixed, but its predictions are no longer fitted and can become much
                less accurate as the shared network learns; a warning is printed.
            drop_osc: Remove the oscillator-strength head. Requires ``train_osc=False``.
            charges_property: Name of a per-atom charge array on every electronic state (e.g.
                'charges'), trained directly with a per-atom MSE weighted by ``charges_weight``;
                states/molecules without it are masked out. None (default) leaves the charges
                constrained by the dipole loss only.
            reset_energy_shifter: Refit the self-atomic energies (SAE) on the training data
                (default); levels of theory and elements absent from the data are left without SAE
                in the saved model. False keeps the SAE of the starting model (``model_file``).

        Distributed training is selected automatically:
            - Single GPU: trains in-process.
            - Single node, multiple GPUs: ignite spawns one worker per local GPU.
            - Multiple nodes: launch the training script under a distributed
              launcher (e.g. ``srun torchrun --nnodes N --nproc_per_node G ...``);
              this method detects the launcher, prepares the shared data once on
              global rank 0, and trains across all ranks.
        '''
        if train_osc and drop_osc:
            raise ValueError('drop_osc=True requires train_osc=False')
        self.set_num_threads()

        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        # --- Multi-node launcher mode (torchrun / torchelastic) ---
        # When launched under `srun torchrun ... script.py`, every process runs
        # this whole method. We must NOT spawn our own workers; each process
        # joins the launcher's process group, global rank 0 prepares the shared
        # data once, and all ranks train together.
        if _is_distributed_launcher():
            self._train_distributed(
                molecular_database=molecular_database,
                property_to_learn=property_to_learn,
                xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
                validation_molecular_database=validation_molecular_database,
                spliting_ratio=spliting_ratio,
                save_model=save_model,
                dipole_property=dipole_property,
                train_osc=train_osc,
                charges_property=charges_property,
                reset_energy_shifter=reset_energy_shifter,
                drop_osc=drop_osc,
            )
            return

        # --- Single-node paths (unchanged launch model) ---
        from aimnetex.train.train import run as train_run
        _run_fn = train_run if self.verbose >= 1 else _quiet_train_run

        num_gpus = torch.cuda.device_count()
        work_dir = tempfile.mkdtemp(prefix='aimnet2_train_')
        prep = self._prepare_training_data(
            molecular_database=molecular_database,
            property_to_learn=property_to_learn,
            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
            validation_molecular_database=validation_molecular_database,
            spliting_ratio=spliting_ratio,
            dipole_property=dipole_property,
            train_osc=train_osc,
            work_dir=work_dir,
            world_size=max(num_gpus, 1),
            charges_property=charges_property,
            reset_energy_shifter=reset_energy_shifter,
            drop_osc=drop_osc,
        )

        if self.verbose >= 1:
            print(f'Starting training in {work_dir}')

        if num_gpus > 1:
            # Single-node multi-GPU: spawn one worker per local GPU via ignite.
            if self.verbose >= 1:
                print(f'Using DDP training with {num_gpus} GPUs')
            # When ntasks-per-node > num_gpus, ignite's SLURM auto-detection
            # sets WORLD_SIZE from SLURM_NTASKS and conflicts with nproc_per_node.
            # Hide all SLURM env vars so ignite uses plain spawn-based DDP.
            _saved_slurm = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith('SLURM_')}
            try:
                from ignite import distributed as idist
                with idist.Parallel(backend='nccl', nproc_per_node=num_gpus) as parallel:
                    parallel.run(
                        _run_fn, num_gpus,
                        prep['model_cfg_yaml'], prep['train_cfg_yaml'],
                        prep['load_weights'], prep['weights_path'],
                    )
            finally:
                os.environ.update(_saved_slurm)
        else:
            _run_fn(
                local_rank=0,
                world_size=1,
                model_cfg=prep['model_cfg_yaml'],
                train_cfg=prep['train_cfg_yaml'],
                load=prep['load_weights'],
                save=prep['weights_path'],
            )

        self._finalize_training(prep, save_model)

    def _prepare_training_data(
        self,
        molecular_database,
        property_to_learn,
        xyz_derivative_property_to_learn,
        validation_molecular_database,
        spliting_ratio,
        dipole_property,
        train_osc,
        work_dir,
        world_size,
        pre_sharded=False,
        reducers=None,
        all_gather_obj=None,
        rank=0,
        charges_property=None,
        reset_energy_shifter=True,
        drop_osc=False,
    ):
        '''
        Build HDF5 data, the SAE table, model/train configs and checkpoint dirs
        for a training run inside ``work_dir`` and return them as a dict.
        ``world_size`` is the number of DDP ranks the training set is sharded
        across (global world size under a multi-node launcher, else the local GPU
        count); the training set is truncated to a multiple of it so per-rank
        shards stay equal.

        When ``pre_sharded`` is True, ``molecular_database`` is already this rank's
        own shard: it is NOT re-truncated, ``work_dir`` is rank-local, the SAE is
        computed globally via all-reduce (using ``reducers``/``all_gather_obj``),
        aimnet is told not to re-shard (``ddp_load_full_dataset``), and
        ``batches_per_epoch`` is pinned to the global minimum to avoid DDP
        deadlock on uneven shards. ``rank`` is the global rank; shared model-dir
        writes are gated to rank 0.
        '''
        from omegaconf import OmegaConf
        from aimnetex.data2mlatom import convert_mlatom_to_aimnet2

        os.makedirs(work_dir, exist_ok=True)

        # --- Split data ---
        if isinstance(validation_molecular_database, str) and validation_molecular_database == 'sample_from_molecular_database':
            n_train = int(len(molecular_database) * spliting_ratio)
            indices = np.random.permutation(len(molecular_database))
            train_mols = data.molecular_database([molecular_database[i] for i in indices[:n_train]])
            val_mols = data.molecular_database([molecular_database[i] for i in indices[n_train:]])
        elif validation_molecular_database is not None:
            train_mols = molecular_database
            val_mols = validation_molecular_database
        else:
            train_mols = molecular_database
            val_mols = None

        # --- Ensure even DDP sharding to prevent NCCL deadlock ---
        # Truncate the training set to a multiple of the effective world size so
        # every rank gets the same number of samples (uneven shards deadlock the
        # DDP collectives). In pre_sharded mode each rank already holds only its
        # own shard (which must NOT be re-truncated); the deadlock is instead
        # avoided by pinning batches_per_epoch to the global minimum below.
        if world_size > 1 and not pre_sharded:
            remainder = len(train_mols) % world_size
            if remainder != 0:
                train_mols = train_mols[:-remainder]

        # --- Convert to HDF5 and free molecular databases ---
        train_h5 = os.path.join(work_dir, 'train.h5')
        lot_dim = int(self.hyperparameters['LoT_dim'].value)
        osc_property = 'oscillator_strengths' if train_osc else None
        n_osc = convert_mlatom_to_aimnet2(
            train_mols, train_h5,
            energy_property=property_to_learn,
            forces_property=xyz_derivative_property_to_learn or 'energy_gradients',
            dipole_property=dipole_property,
            osc_property=osc_property,
            LoT_dim=lot_dim,
            charges_property=charges_property,
        )
        del train_mols
        if train_osc and not pre_sharded and n_osc == 0:
            raise ValueError('train_osc=True, but no molecule in the training set has oscillator_strengths')

        val_h5 = None
        if val_mols is not None and len(val_mols) > 0:
            val_h5 = os.path.join(work_dir, 'val.h5')
            convert_mlatom_to_aimnet2(
                val_mols, val_h5,
                energy_property=property_to_learn,
                forces_property=xyz_derivative_property_to_learn or 'energy_gradients',
                dipole_property=dipole_property,
                osc_property=osc_property,
                LoT_dim=lot_dim,
                charges_property=charges_property,
            )
        del val_mols

        import gc
        gc.collect()

        # --- Compute SAE ---
        sae_file = os.path.join(work_dir, 'sae.yaml')
        sae_mode = str(self.hyperparameters['sae_mode'].value)

        # If continuing from a pretrained model (e.g. AL fine-tune), pin the
        # LoT layout (num_lots + lot_descriptors order) to the pretrained
        # checkpoint so load_state_dict matches; the SAE *values* are recomputed
        # from the current training data (reset_energy_shifter=True) or kept.
        pretrained_lot_descriptors = None
        pretrained_sae = None
        if self.model_file:
            candidate = os.path.join(self.model_file, 'sae.yaml')
            if os.path.exists(candidate):
                pretrained_sae = candidate
                from aimnetex.config import load_yaml
                existing = load_yaml(candidate)
                if isinstance(existing, dict) and 'lots' in existing:
                    pretrained_lot_descriptors = [
                        list(entry['descriptor']) for entry in existing['lots']
                    ]
                    sae_mode = 'linreg_per_lot'
                    if self.verbose >= 1 and reset_energy_shifter:
                        print(f'Pretrained model has {len(pretrained_lot_descriptors)} '
                              f'LoTs; pinning new SAE table to that layout')

        if not reset_energy_shifter and pretrained_sae:
            # Keep the starting model's SAE: it shifts the training targets and is exported
            # unchanged, so LoTs and elements without new data keep their SAE.
            shutil.copy2(pretrained_sae, sae_file)
            if pretrained_lot_descriptors is not None:
                sae_resolved = {
                    'mode': 'linreg_per_lot',
                    'lot_descriptors': pretrained_lot_descriptors,
                    'num_lots': len(pretrained_lot_descriptors),
                    'num_types': 64,
                }
            else:
                sae_mode = 'linreg'
                sae_resolved = {'mode': 'linreg'}
            if self.verbose >= 1 and rank == 0:
                print(f'Keeping the SAE of the starting model ({pretrained_sae})')
        elif sae_mode == 'linreg_per_lot':
            if pre_sharded:
                # SAE fit over the union of all ranks' shards via all-reduce.
                sae_resolved = self._compute_sae_per_lot_distributed(
                    train_h5, sae_file, reducers, all_gather_obj,
                    fixed_lot_descriptors=pretrained_lot_descriptors,
                )
            else:
                sae_resolved = self._compute_sae_per_lot(
                    train_h5, sae_file,
                    fixed_lot_descriptors=pretrained_lot_descriptors,
                )
        elif sae_mode == 'linreg':
            if pre_sharded:
                self._compute_sae_distributed(train_h5, sae_file, reducers)
            else:
                self._compute_sae(train_h5, sae_file)
            sae_resolved = {'mode': 'linreg'}
        else:
            raise ValueError(f"Unknown sae_mode {sae_mode!r}; expected 'linreg' or 'linreg_per_lot'")

        # --- Build model config with hyperparameter overrides ---
        model_cfg = OmegaConf.load(self.model_config_path)
        outputs_overrides: Dict[str, Any] = {
            'energy_mlp': {
                'kwargs': {
                    'n_in': int(self.hyperparameters['aim_size'].value),
                    'mlp': {'hidden': self.hyperparameters['energy_mlp_hidden'].value},
                }
            },
            'lrcoulomb': {
                'kwargs': {'rc': float(self.hyperparameters['lrcoulomb_rc'].value)},
            },
        }
        pretrained_outputs = (self._model_cfg.get('kwargs', {}).get('outputs', {})
                              if self._model_cfg is not None else {})
        if train_osc:
            # oscillator-strength head: per-atom MLP on the final AIM features (state-aware,
            # like the energy head) + sum over atoms
            outputs_overrides['osc_mlp'] = {
                'class': 'aimnetex.modules.Output',
                'kwargs': {
                    'n_in': int(self.hyperparameters['aim_size'].value),
                    'n_out': 1,
                    'key_in': 'aim',
                    'key_out': 'oscillator_strength',
                    'mlp': {'activation_fn': 'torch.nn.GELU', 'last_linear': True,
                            'hidden': self.hyperparameters['osc_mlp_hidden'].value},
                },
            }
            outputs_overrides['osc_sum'] = {
                'class': 'aimnetex.modules.AtomicSum',
                'kwargs': {'key_in': 'oscillator_strength', 'key_out': 'oscillator_strength'},
            }
        elif not drop_osc and 'osc_mlp' in pretrained_outputs:
            if not self._weights_path or not os.path.isfile(self._weights_path):
                raise FileNotFoundError('Keeping the oscillator-strength head requires the starting model weights')
            for name in ('osc_mlp', 'osc_sum'):
                outputs_overrides[name] = OmegaConf.to_container(pretrained_outputs[name], resolve=True)
            outputs_overrides['osc_mlp']['class'] = __name__ + '._frozen_oscillator_output'
            if rank == 0:
                print(f'WARNING: the starting model {self.model_file} has an oscillator-strength head and '
                      f'train_osc=False: the head keeps its weights frozen while the shared network is trained, '
                      f'so its oscillator strengths are no longer fitted and can become much less accurate. '
                      f'Pass train_osc=True with oscillator-strength labels to train the head, or drop_osc=True '
                      f'to remove it.', flush=True)
        else:
            outputs = model_cfg.get('kwargs', {}).get('outputs', {})
            outputs.pop('osc_mlp', None)
            outputs.pop('osc_sum', None)
        if sae_mode == 'linreg_per_lot':
            # NOTE: do NOT pass init_shifts here — during training, targets are
            # already SAE-shifted by the data loader, so the model only needs to
            # learn the residual. SAE is added back into shifts at export time
            # by aimnetex.train.pt2jpt.add_sae_to_shifts (per-LoT branch).
            outputs_overrides['atomic_shift'] = {
                'class': 'aimnetex.modules.LoTAtomicShift',
                'kwargs': {
                    'key_in': 'energy',
                    'key_out': 'energy',
                    'num_types': int(sae_resolved['num_types']),
                    'num_lots': int(sae_resolved['num_lots']),
                    'lot_descriptors': sae_resolved['lot_descriptors'],
                },
            }
        model_overrides = {
            'kwargs': {
                'nfeature': int(self.hyperparameters['nfeature'].value),
                'LoT_dim': int(self.hyperparameters['LoT_dim'].value),
                'd2features': bool(self.hyperparameters['d2features'].value),
                'ncomb_v': int(self.hyperparameters['ncomb_v'].value),
                'num_charge_channels': int(self.hyperparameters['num_charge_channels'].value),
                'hidden': self.hyperparameters['hidden'].value,
                'aim_size': int(self.hyperparameters['aim_size'].value),
                'aev': {
                    'rc_s': float(self.hyperparameters['rc_s'].value),
                    'nshifts_s': int(self.hyperparameters['nshifts_s'].value),
                },
                'outputs': outputs_overrides,
            }
        }
        model_cfg = OmegaConf.merge(model_cfg, OmegaConf.create(model_overrides))

        # --- Build train config ---
        train_cfg = OmegaConf.load(self.train_config_path)

        overrides = {
            'data': {
                'train': train_h5,
                'val': val_h5,
                'sae': {
                    'energy': {
                        'file': sae_file,
                        'mode': sae_mode,
                    }
                },
                'val_fraction': float(self.hyperparameters['val_fraction'].value),
                'separate_val': True,
                'x': ['coord', 'numbers', 'charge', 'state', 'lot_descriptor'],
                'samplers': {
                    'train': {
                        'kwargs': {
                            'batch_size': int(self.hyperparameters['batch_size'].value),
                        }
                    }
                },
            },
            'optimizer': {
                'kwargs': {
                    'lr': float(self.hyperparameters['learning_rate'].value),
                    'weight_decay': float(self.hyperparameters['weight_decay'].value),
                }
            },
            'scheduler': {
                'kwargs': {
                    'factor': float(self.hyperparameters['lr_reduce_factor'].value),
                    'patience': int(self.hyperparameters['lr_reduce_patience'].value),
                },
                'terminate_on_low_lr': float(self.hyperparameters['early_stopping_learning_rate'].value),
                # sibling key consumed by aimnet build_engine (build_module ignores it);
                # 0 => no warmup. Linearly ramps LR to target over the first N epochs.
                'warmup_epochs': int(self.hyperparameters['warmup_epochs'].value),
            },
            'trainer': {
                'epochs': int(self.hyperparameters['max_epochs'].value),
            },
            'wandb': None,
        }

        # Build data.y, loss, and metrics dynamically based on requested targets
        data_y = [property_to_learn]
        loss_components = {
            property_to_learn: {
                'fn': str(self.hyperparameters['energy_loss_fn'].value),
                'weight': float(self.hyperparameters['energy_weight'].value),
                'kwargs': dict(self.hyperparameters['energy_loss_kwargs'].value or {}),
            }
        }
        metrics_cfg = {
            property_to_learn: {'abbr': 'E', 'scale': 23.06},
        }

        if xyz_derivative_property_to_learn:
            data_y.append('forces')
            loss_components['forces'] = {
                'fn': str(self.hyperparameters['force_loss_fn'].value),
                'weight': float(self.hyperparameters['force_weight'].value),
                'kwargs': {'key_true': 'forces', 'key_pred': 'forces'},
            }
            metrics_cfg['forces'] = {'abbr': 'F', 'peratom': True, 'mult': 3, 'scale': 23.06}

        if train_osc:
            data_y.append('oscillator_strength')
            loss_components['oscillator_strength'] = {
                'fn': 'aimnetex.train.loss.oscillator_strength_loss_fn',
                'weight': float(self.hyperparameters['osc_weight'].value),
                'kwargs': {'key_pred': 'oscillator_strength', 'key_true': 'oscillator_strength'},
            }

        if dipole_property:
            data_y.append('dipole_moment')
            loss_components['dipole_moment'] = {
                'fn': str(self.hyperparameters['dipole_loss_fn'].value),
                'weight': float(self.hyperparameters['dipole_weight'].value),
                'kwargs': {
                    'key_pred_charges': 'charges',
                    'key_pred_coords': 'coord',
                    'key_true': 'dipole_moment',
                },
            }

        if charges_property:
            # per-atom reference charges: dataset 'charges' (n_rows, n_atoms), NaN rows masked
            data_y.append('charges')
            loss_components['charges'] = {
                'fn': str(self.hyperparameters['charges_loss_fn'].value),
                'weight': float(self.hyperparameters['charges_weight'].value),
                'kwargs': {'key_pred': 'charges', 'key_true': 'charges'},
            }
            metrics_cfg['charges'] = {'abbr': 'q', 'peratom': True, 'mult': 1, 'scale': 1.0}

        # Delete template sections that must be fully replaced (not deep-merged)
        del train_cfg['data']['y']
        del train_cfg['loss']
        del train_cfg['metrics']

        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.create(overrides))

        # Set the dynamically-built sections
        train_cfg.data.y = data_y
        train_cfg.loss = OmegaConf.create({
            'class': 'aimnetex.train.loss.MTLoss',
            'kwargs': {'components': loss_components},
        })
        train_cfg.metrics = OmegaConf.create({
            'class': 'aimnetex.train.metrics.RegMultiMetric',
            'kwargs': {'cfg': metrics_cfg},
        })

        # --- Pre-sharded: don't re-shard, and equalize per-epoch batch count ---
        # With ddp_load_full_dataset=True aimnet loads each rank's file whole
        # (no strided slicing). That bypasses the tail-truncation that normally
        # keeps per-rank batch counts equal, so we pin batches_per_epoch to the
        # global minimum: every rank then runs the same number of forwards, and
        # neither the gradient all-reduce (train) nor the DDP buffer broadcast
        # (val too) can deadlock on uneven shards.
        if pre_sharded:
            import h5py
            batch_size = int(self.hyperparameters['batch_size'].value)
            try:
                batch_mode = str(train_cfg.data.samplers.train.kwargs.get('batch_mode', 'molecules'))
            except Exception:
                batch_mode = 'molecules'

            def _local_batch_count(h5_path):
                # The sampler's TRUE epoch length = sum over size-groups of
                # ceil(units/bs), evaluated AFTER aimnet merges groups smaller
                # than 8*batch_size (exactly what get_loaders does before building
                # the sampler). Matching this precisely means the pinned count
                # equals what a non-pinned run naturally uses -> for equal shards
                # there is NO truncation and NO duplicate padding (identical to
                # the centralized run). units = h5 rows ('molecules' mode; one row
                # per electronic state) or rows*natoms ('atoms').
                if not h5_path or not os.path.exists(h5_path):
                    return 0

                def _sum_ceils(groups, getlen, getnatoms):
                    tot = 0
                    for _g in groups:
                        rows = getlen(_g)
                        units = rows * getnatoms(_g) if batch_mode == 'atoms' else rows
                        tot += -(-units // batch_size)   # ceil division
                    return tot

                try:  # exact: replicate merge_groups then count
                    from aimnetex.data import SizeGroupedDataset
                    _ds = SizeGroupedDataset(h5_path, keys=['numbers'])
                    _ds.merge_groups(min_size=8 * batch_size, mode_atoms=(batch_mode == 'atoms'))
                    total = _sum_ceils(
                        _ds.groups,
                        lambda g: len(g),
                        lambda g: int(np.asarray(g['numbers']).shape[1]),
                    )
                    del _ds
                    return total
                except Exception:  # fallback: unmerged per-group ceil-sum from h5 metadata
                    with h5py.File(h5_path, 'r') as _f:
                        return _sum_ceils(
                            [g for g in _f.values() if isinstance(g, h5py.Group) and 'numbers' in g],
                            lambda g: int(g['numbers'].shape[0]),
                            lambda g: int(g['numbers'].shape[1]) if g['numbers'].ndim > 1 else 1,
                        )

            def _global_min_batches(h5_path):
                # Pin to the GLOBAL MIN of the per-rank natural batch count: every
                # rank runs the same number of forwards (no deadlock on uneven
                # shards) while matching the batches an unsharded/centralized run
                # would use on the same data.
                local = max(1, _local_batch_count(h5_path))
                if reducers is not None:
                    return max(1, int(reducers['min'](np.array([float(local)]))[0]))
                return max(1, local)

            train_bpe = _global_min_batches(train_h5)
            OmegaConf.update(train_cfg, 'data.ddp_load_full_dataset', True, force_add=True)
            OmegaConf.update(train_cfg, 'data.samplers.train.kwargs.batches_per_epoch', train_bpe, force_add=True)
            val_bpe = None
            _samplers = train_cfg.data.get('samplers', None)
            if _samplers is not None and 'val' in _samplers:   # config-driven -> same on all ranks
                val_bpe = _global_min_batches(val_h5)
                OmegaConf.update(train_cfg, 'data.samplers.val.kwargs.batches_per_epoch', val_bpe, force_add=True)
            if self.verbose >= 1 and rank == 0:
                print(f'[pre_sharded] ddp_load_full_dataset=True, batches_per_epoch '
                      f'train={train_bpe}'
                      + (f' val={val_bpe}' if val_bpe is not None else '')
                      + f' (global min over ranks, batch_mode={batch_mode})')

        # --- Checkpoints and periodic saving ---
        save_interval = int(self.hyperparameters['save_every_n_epochs'].value)
        if self.model_file:
            # All checkpoints go inside the model directory on shared storage
            model_dir = os.path.abspath(self.model_file)
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_dir = model_dir
            best_checkpoint_dir = model_dir

            # Pre-write config, SAE, meta so the dir is a loadable partial save.
            # In pre_sharded mode every rank runs prep; only global rank 0 writes
            # to the shared model dir to avoid a write race (content is identical).
            if rank == 0:
                OmegaConf.save(model_cfg, os.path.join(model_dir, 'model_config.yaml'))
                shutil.copy2(sae_file, os.path.join(model_dir, 'sae.yaml'))
                meta = {
                    'hyperparameters': {k: v.value for k, v in self.hyperparameters.items()},
                    'program': self.program,
                }
                with open(os.path.join(model_dir, 'meta.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

            if save_interval > 0:
                train_cfg = OmegaConf.merge(train_cfg, OmegaConf.create({
                    'full_save': {
                        'dirname': model_dir,
                        'interval': save_interval,
                    }
                }))
                if self.verbose >= 1:
                    print(f'Periodic save every {save_interval} epoch(s) to {model_dir}')
        else:
            checkpoint_dir = os.path.join(work_dir, 'checkpoints')
            best_checkpoint_dir = os.path.join(work_dir, 'best_checkpoint')
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(best_checkpoint_dir, exist_ok=True)

        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.create({
            'checkpoint': {
                'dirname': checkpoint_dir,
                'filename_prefix': 'checkpoint',
                'kwargs': {'n_saved': 1, 'require_empty': False},
            },
            'best_checkpoint': {
                'dirname': best_checkpoint_dir,
                'filename_prefix': 'best',
                'n_saved': 1,
            },
        }))

        # --- Resolve config YAML and persist to work_dir ---
        weights_path = os.path.join(work_dir, 'model_weights.pt')

        model_cfg_yaml = OmegaConf.to_yaml(model_cfg)
        train_cfg_yaml = OmegaConf.to_yaml(train_cfg)

        if self.verbose >= 2:
            print('--- Model config YAML ---')
            print(model_cfg_yaml)
            print('--- Train config YAML ---')
            print(train_cfg_yaml)
            print('--- End configs ---')

        # Persist the fully-resolved configs into work_dir so that, under a
        # multi-node launcher, every rank can read identical configs that already
        # point at the shared h5/sae paths in work_dir.
        OmegaConf.save(model_cfg, os.path.join(work_dir, 'model_config.yaml'))
        OmegaConf.save(train_cfg, os.path.join(work_dir, 'train_config.yaml'))

        # Load existing weights if continuing training
        load_weights = self._weights_path if self._weights_path and os.path.exists(self._weights_path) else None

        return {
            'work_dir': work_dir,
            'model_cfg': model_cfg,
            'train_cfg': train_cfg,
            'model_cfg_yaml': model_cfg_yaml,
            'train_cfg_yaml': train_cfg_yaml,
            'sae_file': sae_file,
            'weights_path': weights_path,
            'best_checkpoint_dir': best_checkpoint_dir,
            'load_weights': load_weights,
        }

    def _finalize_training(self, prep, save_model):
        '''
        Post-training: promote the best checkpoint to the final weights file,
        record paths on the instance, and (optionally) export the saved model.
        Runs once per training (in the driver for single-node, or on global
        rank 0 under a multi-node launcher).
        '''
        import glob

        weights_path = prep['weights_path']

        # --- Use best checkpoint if available, otherwise final weights ---
        best_files = glob.glob(os.path.join(prep['best_checkpoint_dir'], 'best_model_*.pt'))
        if best_files:
            best_file = max(best_files, key=os.path.getmtime)
            shutil.copy2(best_file, weights_path)
            if self.verbose >= 1:
                print(f'Using best model checkpoint: {os.path.basename(best_file)}')

        self._weights_path = weights_path
        self._sae_path = prep['sae_file']
        self._model_cfg = prep['model_cfg']
        self._work_dir = prep['work_dir']

        if save_model and self.model_file:
            self.save(self.model_file)

        if self.verbose >= 1:
            print('Training complete.')

    def _train_distributed(
        self,
        molecular_database,
        property_to_learn,
        xyz_derivative_property_to_learn,
        validation_molecular_database,
        spliting_ratio,
        save_model,
        dipole_property,
        train_osc,
        charges_property=None,
        reset_energy_shifter=True,
        drop_osc=False,
    ):
        '''
        Multi-node DDP training under an external launcher (e.g. `srun torchrun`).
        Every process runs this method. Two data paths:

        - Centralized (default): global rank 0 prepares the shared data/configs
          once into a deterministic shared work dir; all ranks read those configs
          and train on strided slices of one shared h5.
        - Pre-sharded (`self.pre_sharded`): each rank's `molecular_database` is
          already its own shard. Every rank converts its shard to a node-local h5
          and the SAE is fit GLOBALLY via all-reduce, so no rank ever holds the
          full dataset. aimnet is told not to re-shard.

        Either way, only global rank 0 writes checkpoints / exports the model.
        '''
        from ignite import distributed as idist
        from aimnetex.train.train import run as train_run

        _run_fn = train_run if self.verbose >= 1 else _quiet_train_run

        # Deterministic shared work dir: the SLURM job is expected to run from
        # shared scratch (e.g. `cd $SCRATCH/$SLURM_JOB_ID`), so a cwd-relative
        # path resolves to the same shared location on every node and rank.
        job_id = (os.environ.get('SLURM_JOB_ID')
                  or os.environ.get('TORCHELASTIC_RUN_ID')
                  or 'dist')
        work_dir = os.path.join(os.getcwd(), f'aimnet2_train_{job_id}')
        sentinel = os.path.join(work_dir, '.prep_done')

        # Use the launcher's native RANK / LOCAL_RANK / WORLD_SIZE / MASTER_*
        # rendezvous (set by torchrun), not SLURM's task layout (which may differ,
        # e.g. ntasks-per-node=1). Hiding SLURM_* makes ignite use the torchrun env.
        _saved_slurm = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith('SLURM_')}

        def _worker(local_rank):
            # Bind this process to its local GPU before any collective/CUDA op.
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            rank = idist.get_rank()
            world_size = idist.get_world_size()

            if self.pre_sharded:
                # --- Pre-sharded: each rank prepares ONLY its own shard ---
                # No rank-0 gating / no sentinel: every rank converts its own
                # shard to a node-local h5 and the SAE is fit GLOBALLY via
                # all-reduce (so no rank needs the full dataset). The collectives
                # inside _prepare_training_data run in identical order on every
                # rank (the LoT layout is a global union / pinned), so they match.
                import tempfile
                dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

                def _reduce(arr, op):
                    t = torch.as_tensor(np.asarray(arr, dtype=np.float64), device=dev)
                    t = idist.all_reduce(t, op=op)
                    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)

                reducers = {
                    'sum': lambda a: _reduce(a, 'SUM'),
                    'min': lambda a: _reduce(a, 'MIN'),
                    'max': lambda a: _reduce(a, 'MAX'),
                }

                def _all_gather_obj(obj):
                    import torch.distributed as dist
                    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
                        return [obj]
                    out = [None] * dist.get_world_size()
                    dist.all_gather_object(out, obj)
                    return out

                shard_dir = tempfile.mkdtemp(prefix=f'aimnet2_shard{rank}_')
                if self.verbose >= 1:
                    print(f'[rank {rank}] pre_sharded: preparing local shard in {shard_dir}',
                          flush=True)
                prep = self._prepare_training_data(
                    molecular_database=molecular_database,
                    property_to_learn=property_to_learn,
                    xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
                    validation_molecular_database=validation_molecular_database,
                    spliting_ratio=spliting_ratio,
                    dipole_property=dipole_property,
                    train_osc=train_osc,
                    work_dir=shard_dir,
                    world_size=world_size,
                    pre_sharded=True,
                    reducers=reducers,
                    all_gather_obj=_all_gather_obj,
                    rank=rank,
                    charges_property=charges_property,
                    reset_energy_shifter=reset_energy_shifter,
                    drop_osc=drop_osc,
                )
                _run_fn(
                    local_rank=rank,
                    world_size=world_size,
                    model_cfg=prep['model_cfg_yaml'],
                    train_cfg=prep['train_cfg_yaml'],
                    load=prep['load_weights'],
                    save=prep['weights_path'],
                )
                idist.barrier()
                if rank == 0:
                    self._finalize_training(prep, save_model)
                return

            prep = None
            if rank == 0:
                if os.path.exists(sentinel):
                    os.remove(sentinel)
                if self.verbose >= 1:
                    print(f'[rank 0] Preparing shared training data in {work_dir} '
                          f'(world_size={world_size})', flush=True)
                prep = self._prepare_training_data(
                    molecular_database=molecular_database,
                    property_to_learn=property_to_learn,
                    xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
                    validation_molecular_database=validation_molecular_database,
                    spliting_ratio=spliting_ratio,
                    dipole_property=dipole_property,
                    train_osc=train_osc,
                    work_dir=work_dir,
                    world_size=world_size,
                    charges_property=charges_property,
                    reset_energy_shifter=reset_energy_shifter,
                    drop_osc=drop_osc,
                )
                # Signal the other ranks that the shared inputs are ready.
                with open(sentinel, 'w') as f:
                    f.write('ok')
            else:
                # Wait WITHOUT a collective so an arbitrarily long data prep on
                # rank 0 cannot trip the NCCL barrier timeout.
                import time
                waited = 0.0
                while not os.path.exists(sentinel):
                    time.sleep(2.0)
                    waited += 2.0
                    if waited > 6 * 3600:
                        raise RuntimeError(
                            f'Timed out waiting for rank 0 data prep in {work_dir}')

            # Every rank reads the rank-0-written configs (which embed the shared
            # h5/sae paths) so all ranks build an identical model and data view.
            with open(os.path.join(work_dir, 'model_config.yaml')) as f:
                model_cfg_yaml = f.read()
            with open(os.path.join(work_dir, 'train_config.yaml')) as f:
                train_cfg_yaml = f.read()
            weights_path = os.path.join(work_dir, 'model_weights.pt')
            load_weights = (self._weights_path
                            if self._weights_path and os.path.exists(self._weights_path)
                            else None)

            _run_fn(
                local_rank=rank,
                world_size=world_size,
                model_cfg=model_cfg_yaml,
                train_cfg=train_cfg_yaml,
                load=load_weights,
                save=weights_path,
            )

            # Ensure all ranks (and the rank-0 weight save) are done before rank 0
            # promotes the best checkpoint / exports the model.
            idist.barrier()
            if rank == 0:
                self._finalize_training(prep, save_model)

        try:
            with idist.Parallel(backend='nccl') as parallel:
                parallel.run(_worker)
        finally:
            os.environ.update(_saved_slurm)

    def predict(
        self,
        molecular_database: data.molecular_database = None,
        molecule: data.molecule = None,
        nstates: int = 1,
        current_state: int = 0,
        calculate_energy: bool = False,
        calculate_energy_gradients: bool = False,
        calculate_hessian: bool = False,
        calculate_dipole_moment: bool = False,
        calculate_dipole_derivatives: bool = False,
        lot_descriptor: list = None,
        batch_size: int = 65536,
        property_to_predict: Union[str, None] = 'estimated_y',
        xyz_derivative_property_to_predict: Union[str, None] = None,
        hessian_to_predict: Union[str, None] = None,
    ) -> None:
        '''
        Predict properties for molecules using the trained model.

        Arguments:
            molecular_database: Database of molecules.
            molecule: Single molecule.
            nstates: Number of electronic states to predict.
            current_state: State to assign to top-level mol properties.
            calculate_energy: Compute energies.
            calculate_energy_gradients: Compute energy gradients: True for all states, or one bool per
                state (e.g. the active-state mask of namd); states without gradients get NaN gradients.
            calculate_hessian: Compute Hessian matrices.
            calculate_dipole_derivatives: Compute dipole moment derivatives w.r.t. coordinates (for IR spectra).
            lot_descriptor: LoT descriptor vector. If None, uses one-hot default based on LoT_dim.
            batch_size: Batch size for inference.

        For nstates > 1, a model with an oscillator-strength head also sets
        mol.oscillator_strengths (S0->S_n, n = 1..nstates-1), as the QM interfaces do; a model
        without the head removes any oscillator_strengths the molecule already had.
            property_to_predict: Label for predicted scalar property.
            xyz_derivative_property_to_predict: Label for predicted gradient property.
            hessian_to_predict: Label for predicted hessian property.
        '''
        # Resolve property names via parent class (maps calculate_energy -> 'energy', etc.)
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(
                molecular_database=molecular_database,
                molecule=molecule,
                calculate_energy=calculate_energy,
                calculate_energy_gradients=calculate_energy_gradients,
                calculate_hessian=calculate_hessian,
                property_to_predict=property_to_predict,
                xyz_derivative_property_to_predict=xyz_derivative_property_to_predict,
                hessian_to_predict=hessian_to_predict,
            )

        # States whose gradients are computed (one backward pass each): a list selects them per state
        if isinstance(calculate_energy_gradients, list):
            if len(calculate_energy_gradients) != nstates:
                raise ValueError(f'calculate_energy_gradients has {len(calculate_energy_gradients)} entries for {nstates} states')
            gradient_states = [bool(g) for g in calculate_energy_gradients]
        else:
            gradient_states = [bool(xyz_derivative_property_to_predict)] * nstates

        # Default lot_descriptor: one-hot with 1 at last position, length = LoT_dim
        if lot_descriptor is None:
            lot_dim = int(self.hyperparameters['LoT_dim'].value)
            lot_descriptor = [0] * (lot_dim - 1) + [1]
        lot_tensor = torch.tensor(lot_descriptor, dtype=torch.float32)

        # Hessian path in the underlying calculator does not support batched input.
        effective_batch_size = 1 if calculate_hessian else batch_size
        for batch in molDB.batches(effective_batch_size):
            B = len(batch)

            # Partition the batch by atom count so each sub-batch passed to
            # the calculator is homogeneous (np.array stacking requires it).
            groups = {}
            for i, mol in enumerate(batch):
                groups.setdefault(len(mol.xyz_coordinates), []).append(i)

            # Per-state, per-molecule output slots indexed by original batch position.
            state_energies           = [[None]*B for _ in range(nstates)]
            state_charges            = [[None]*B for _ in range(nstates)]
            state_gradients          = [[None]*B for _ in range(nstates)]
            state_hessians           = [[None]*B for _ in range(nstates)]
            state_dipoles            = [[None]*B for _ in range(nstates)]
            state_dipole_derivatives = [[None]*B for _ in range(nstates)]
            state_oscs               = [[None]*B for _ in range(nstates)]

            for state_idx in range(nstates):
                for n_atoms, idxs in groups.items():
                    sub = [batch[i] for i in idxs]
                    coords = torch.tensor(
                        np.array([m.xyz_coordinates for m in sub]),
                        dtype=torch.float32
                    )
                    numbers = torch.tensor(
                        np.array([m.atomic_numbers for m in sub]),
                        dtype=torch.int32
                    )
                    charges = torch.tensor(
                        np.array([getattr(m, 'charge', 0) for m in sub]),
                        dtype=torch.float32
                    )
                    states = torch.full((len(sub),), state_idx, dtype=torch.int32)
                    lot_descriptors = lot_tensor.unsqueeze(0).expand(len(sub), -1)

                    input_data = {
                        'coord': coords,
                        'numbers': numbers,
                        'charge': charges,
                        'state': states,
                        'lot_descriptor': lot_descriptors,
                    }

                    result = self.calc.eval(
                        input_data,
                        forces=gradient_states[state_idx],
                        hessian=bool(hessian_to_predict),
                        dipole_derivatives=calculate_dipole_derivatives,
                    )

                    # Energy: eV -> Hartree
                    energies_g = result['energy'].detach().cpu().numpy() * ml.constants.eV2Hartree
                    charges_g  = result['charges'].detach().cpu().numpy()

                    if calculate_dipole_moment:
                        coords_np = coords.detach().cpu().numpy()
                        # charges_g: [G, N], coords_np: [G, N, 3] -> dipole: [G, 3]
                        dipoles_g = np.einsum('bn,bnd->bd', charges_g, coords_np) / ml.constants.Debye
                    if calculate_dipole_derivatives and 'dipole_derivatives' in result:
                        dd_g = result['dipole_derivatives'].detach().cpu().numpy() / ml.constants.Debye
                    if 'oscillator_strength' in result:
                        osc_g = result['oscillator_strength'].detach().cpu().numpy().reshape(-1)
                    if gradient_states[state_idx]:
                        grads_g = -result['forces'].detach().cpu().numpy() * ml.constants.eV2Hartree
                    elif xyz_derivative_property_to_predict:
                        grads_g = np.full((len(sub), n_atoms, 3), np.nan)   # not requested for this state
                    if hessian_to_predict and 'hessian' in result:
                        hess_g = result['hessian'].detach().cpu().numpy() * ml.constants.eV2Hartree

                    for j, i in enumerate(idxs):
                        state_energies[state_idx][i] = energies_g[j]
                        state_charges[state_idx][i]  = charges_g[j]
                        if calculate_dipole_moment:
                            state_dipoles[state_idx][i] = dipoles_g[j]
                        if calculate_dipole_derivatives and 'dipole_derivatives' in result:
                            state_dipole_derivatives[state_idx][i] = dd_g[j]
                        if 'oscillator_strength' in result:
                            state_oscs[state_idx][i] = float(osc_g[j])
                        if xyz_derivative_property_to_predict:
                            state_gradients[state_idx][i] = grads_g[j]
                        if hessian_to_predict and 'hessian' in result:
                            # calculator returns hessian with shape (N, 3, N, 3), no batch dim
                            # (hessian path is single-molecule only; effective_batch_size is 1 above)
                            state_hessians[state_idx][i] = hess_g

                # After all groups for this state: write batch-level DB properties.
                if property_to_predict:
                    name = property_to_predict + ('_state' + str(state_idx) if nstates > 1 else '')
                    batch.add_scalar_properties(state_energies[state_idx], name)
                if xyz_derivative_property_to_predict:
                    name = xyz_derivative_property_to_predict + ('_state' + str(state_idx) if nstates > 1 else '')
                    for i, mol in enumerate(batch):
                        mol.add_xyz_vectorial_property(state_gradients[state_idx][i], xyz_vectorial_property=name)

            # --- Per-molecule: populate electronic_states ---
            for idx, mol in enumerate(batch):
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())

                for state_idx in range(nstates):
                    # Energy (named per property_to_predict: 'energy' when calculate_energy=True,
                    # 'estimated_y' by default)
                    if property_to_predict:
                        setattr(mol.electronic_states[state_idx], property_to_predict, state_energies[state_idx][idx])

                    # Charges
                    mol.electronic_states[state_idx].AIMNet_charges = state_charges[state_idx][idx]

                    # Gradients
                    if xyz_derivative_property_to_predict:
                        mol.electronic_states[state_idx].add_xyz_derivative_property(
                            np.array(state_gradients[state_idx][idx]).astype(float),
                            'energy', 'energy_gradients'
                        )

                    # Dipole moment
                    if calculate_dipole_moment:
                        mol.electronic_states[state_idx].dipole_moment = state_dipoles[state_idx][idx]

                    # Dipole derivatives
                    if calculate_dipole_derivatives and state_dipole_derivatives[state_idx][idx] is not None:
                        mol.electronic_states[state_idx].dipole_derivatives = state_dipole_derivatives[state_idx][idx]

                    # Hessian
                    if hessian_to_predict and state_hessians[state_idx][idx] is not None:
                        hess = state_hessians[state_idx][idx]  # shape (N, 3, N, 3)
                        n3 = hess.shape[0] * 3
                        mol.electronic_states[state_idx].hessian = hess.reshape(n3, n3)

                # Oscillator strengths S0->S_n from the oscillator-strength head (clipped at 0).
                # Without the head, any oscillator_strengths the molecule came with are removed:
                # they belong to the excited states that were just replaced.
                if nstates > 1:
                    if state_oscs[1][idx] is not None:
                        mol.oscillator_strengths = np.array([max(state_oscs[si][idx], 0.0) for si in range(1, nstates)])
                    else:
                        mol.__dict__.pop('oscillator_strengths', None)

                # Assign current state to top-level properties
                if property_to_predict:
                    mol.__dict__[property_to_predict] = state_energies[current_state][idx]

                mol.AIMNet_charges = mol.electronic_states[current_state].AIMNet_charges

                if calculate_dipole_moment:
                    mol.dipole_moment = mol.electronic_states[current_state].dipole_moment

                if calculate_dipole_derivatives and state_dipole_derivatives[current_state][idx] is not None:
                    mol.dipole_derivatives = mol.electronic_states[current_state].dipole_derivatives

                if xyz_derivative_property_to_predict:
                    mol.add_xyz_derivative_property(
                        np.array(state_gradients[current_state][idx]).astype(float),
                        'energy', 'energy_gradients'
                    )

                if hessian_to_predict and state_hessians[current_state][idx] is not None:
                    hess = state_hessians[current_state][idx]  # shape (N, 3, N, 3)
                    n3 = hess.shape[0] * 3
                    mol.hessian = hess.reshape(n3, n3)

    def save(self, model_file: str = '') -> None:
        '''
        Save the model to a directory containing weights, config, SAE, and JIT model.

        Arguments:
            model_file (str): Directory path to save the model into.
        '''
        from omegaconf import OmegaConf
        from aimnetex.train.pt2jpt import set_eval, add_cutoff, add_sae_to_shifts, mask_not_implemented_species
        from aimnetex.config import build_module, load_yaml

        if not model_file:
            model_file = f'msaioaimnet2_{uuid.uuid4()}'
        model_file = os.path.abspath(model_file)
        self.model_file = model_file

        os.makedirs(model_file, exist_ok=True)

        # 1. Save model weights
        weights_dst = os.path.join(model_file, 'model_weights.pt')
        if self._weights_path and os.path.exists(self._weights_path):
            if os.path.abspath(self._weights_path) != os.path.abspath(weights_dst):
                shutil.copy2(self._weights_path, weights_dst)
        else:
            raise RuntimeError('No model weights available. Train the model first.')

        # 2. Save model config
        cfg_dst = os.path.join(model_file, 'model_config.yaml')
        if self._model_cfg is not None:
            OmegaConf.save(self._model_cfg, cfg_dst)
        elif os.path.exists(self.model_config_path):
            if os.path.abspath(self.model_config_path) != os.path.abspath(cfg_dst):
                shutil.copy2(self.model_config_path, cfg_dst)

        # 3. Save SAE
        sae_dst = os.path.join(model_file, 'sae.yaml')
        if self._sae_path and os.path.exists(self._sae_path):
            if os.path.abspath(self._sae_path) != os.path.abspath(sae_dst):
                shutil.copy2(self._sae_path, sae_dst)

        # 4. Compile to JIT (.jpt)
        jpt_path = os.path.join(model_file, 'model.jpt')
        model_cfg = OmegaConf.load(cfg_dst)
        model_nn = build_module(OmegaConf.to_container(model_cfg))
        model_nn = set_eval(model_nn)
        model_nn = add_cutoff(model_nn)

        sd = torch.load(weights_dst, map_location='cpu', weights_only=True)
        model_nn.load_state_dict(sd, strict=False)

        if os.path.exists(sae_dst):
            model_nn = add_sae_to_shifts(model_nn, sae_dst)
            sae_data = load_yaml(sae_dst)
            if sae_data and isinstance(sae_data, dict):
                if 'lots' in sae_data:
                    species_set: set = set()
                    for entry in sae_data['lots']:
                        species_set.update(int(z) for z in entry['shifts'].keys())
                    species = sorted(species_set)
                else:
                    species = [int(z) for z in sae_data.keys()]
                model_nn = mask_not_implemented_species(model_nn, species)
                model_nn.register_buffer('implemented_species', torch.tensor(species, dtype=torch.int64))

        model_jit = torch.jit.script(model_nn)
        model_jit.save(jpt_path)
        self._jpt_path = jpt_path

        # 5. Save metadata
        meta = {
            'hyperparameters': {k: v.value for k, v in self.hyperparameters.items()},
            'program': self.program,
        }
        with open(os.path.join(model_file, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Reset calculator to pick up new .jpt
        self._calc = None

        if self.verbose >= 1:
            print(f'Model saved in {model_file}')

    def load(self, model_file: str = '') -> None:
        '''
        Load a saved model.

        Arguments:
            model_file (str): Path to model directory or .jpt file.
        '''
        model_file = os.path.abspath(model_file)
        self.model_file = model_file

        if os.path.isdir(model_file):
            # Our directory save format
            self._jpt_path = os.path.join(model_file, 'model.jpt')
            self._weights_path = os.path.join(model_file, 'model_weights.pt')
            self._sae_path = os.path.join(model_file, 'sae.yaml')

            cfg_path = os.path.join(model_file, 'model_config.yaml')
            if os.path.exists(cfg_path):
                from omegaconf import OmegaConf
                self._model_cfg = OmegaConf.load(cfg_path)

            meta_path = os.path.join(model_file, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                saved_hp = meta.get('hyperparameters', {})
                for k, v in saved_hp.items():
                    if k in self.hyperparameters:
                        self.hyperparameters[k].value = v

        elif model_file.endswith('.jpt'):
            # Single JIT file (inference only)
            self._jpt_path = os.path.abspath(model_file)

        else:
            raise ValueError(f'Unrecognized model format: {model_file}. '
                             f'Expected a directory or .jpt file.')

        # Reset calculator so it lazy-loads
        self._calc = None

        if self.verbose >= 1:
            print(f'Model loaded from {model_file}')

    def generate_model_dict(self):
        '''Generate model dictionary for JSON serialization.'''
        model_dict = {
            'type': 'ml_model',
            'ml_model_type': 'msaioaimnet2',
            'kwargs': {
                'model_file': os.path.abspath(self.model_file) if self.model_file else None,
            },
            'hyperparameters': {k: v.value for k, v in self.hyperparameters.items()},
            'nthreads': self.nthreads,
        }
        return model_dict

    def reset(self):
        '''Reset the model and delete saved files.'''
        if self.model_file and os.path.isdir(self.model_file):
            shutil.rmtree(self.model_file)
        elif self.model_file and os.path.exists(self.model_file):
            os.remove(self.model_file)
        self._weights_path = None
        self._jpt_path = None
        self._sae_path = None
        self._model_cfg = None
        self._calc = None
