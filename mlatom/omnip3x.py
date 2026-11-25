'''
.. code-block::

  !---------------------------------------------------------------------------!
  !                                                                           !
  ! omnip3x: OMNI-P3x universal neural network potential for large-scale      !
  !          computational photochemistry workflows                           !
  ! Implementation by: Mikolaj Martyka                                        !
  ! Modifications by Pavlo O. Dral                                            !
  !   Oscillator fine-tuning and ensemble spread                              !
  !---------------------------------------------------------------------------!
'''

import os
import shutil
import uuid
import numpy as np
from . import models, data
from .model_cls import method_model, downloadable_model, torch_model


class omnip3x(method_model, downloadable_model, torch_model):
    '''
    OMNI-P3x: universal neural network potential for the ground and singlet excited states of
    neutral and charged molecules containing H, B, C, N, O, F, Si, P, S, Cl, Br and I.

    OMNI-P3x is an ensemble of four MS-AIOAIMNet2 networks; the predictions are the means of the
    four members: state energies, energy gradients, Hessians, permanent dipole moments, neural
    atomic charges (``AIMNet_charges``) and S0->Sn oscillator strengths. The level of theory is
    chosen with the ``level`` argument of ``predict``:

        - ``'TDA-wB97X-D/def2-SVP'`` (default): ground and excited states.
        - ``'wB97M-D3(BJ)/def2-TZVPP'``: ground state. The network predicts wB97M/def2-TZVPP and
          the D3(BJ) dispersion with the parameters of AIMNet2 is added.
        - ``'CCSD(T)*/CBS'``: ground state of molecules containing H, C, N and O.

    ``energy_standard_deviation`` gives the population standard deviation across members
    for each state and for the selected state on the molecule. It is zero for a single member.

    Arguments:
        method (str, optional): ``'OMNI-P3x'``.
        model_index (int, optional): Use only ensemble member ``model_index`` (0-3) instead of the
            mean of all four; ``train`` then fine-tunes only this member.
        model_file (str, optional): Directory of a fine-tuned OMNI-P3x (members cv0 to cv3), loaded
            if it exists. Otherwise the pretrained OMNI-P3x is loaded and ``train`` saves the
            fine-tuned members there.
        nthreads (int, optional): Number of CPU threads. Defaults to the current PyTorch setting.

    .. code-block:: python

        omnip3x = ml.methods(method='OMNI-P3x')
        # S0 and ten singlet excited states; gradients of all states
        omnip3x.predict(molecule=mol, nstates=11, current_state=0,
                        calculate_energy=True, calculate_energy_gradients=True)
        mol.excitation_energies     # Hartree
        mol.oscillator_strengths
        # ground state at CCSD(T)*/CBS
        omnip3x.predict(molecule=mol, level='CCSD(T)*/CBS', calculate_energy_gradients=True)

        # fine-tuning on TDA-wB97X-D/def2-SVP data; ml.methods(method='OMNI-P3x', model_file='omnip3x_ft')
        # loads the fine-tuned OMNI-P3x again
        omnip3x = ml.methods(method='OMNI-P3x', model_file='omnip3x_ft')
        omnip3x.train(molecular_database=db, xyz_derivative_property_to_learn='energy_gradients')
    '''

    supported_methods = ['OMNI-P3x', 'OMNIP3x']
    model_dir = 'omnip3x_model'
    member_dirs = ['cv0', 'cv1', 'cv2', 'cv3']
    download_links = [
        'https://aitomistic.xyz/model/omnip3x_model.zip',
        'https://zenodo.org/records/24344108/files/omnip3x_model.zip?download=1',  # doi:10.5281/zenodo.24344108
    ]
    elements = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    levels = {  # level of theory: (level-of-theory descriptor, supported elements)
        'TDA-wB97X-D/def2-SVP':    ([0, 0, 1], elements),
        'wB97M-D3(BJ)/def2-TZVPP': ([0, 1, 0], elements),
        'CCSD(T)*/CBS':            ([1, 0, 0], {1, 6, 7, 8}),
    }

    def __init__(self, method: str = 'OMNI-P3x', model_index: int = None, model_file: str = None, nthreads: int = None):
        self.method = method
        self.model_index = model_index
        self.model_file = os.path.abspath(model_file) if model_file else None
        # rebuilds the object in models.load (dump keeps only str and dict attributes)
        self.init_kwargs = {'method': method, 'model_index': model_index, 'model_file': self.model_file}
        self.load()
        if nthreads:
            self.set_num_threads(nthreads)

    def load(self):
        import yaml
        members = self.member_dirs if self.model_index is None else [f'cv{self.model_index}']
        member_files = ['model.jpt', 'sae.yaml']
        if self.model_file and os.path.exists(self.model_file):
            model_dir = self.model_file
        else:
            if self.model_file:
                print(f'{self.model_file} does not exist: the pretrained OMNI-P3x is loaded and train() saves the fine-tuned OMNI-P3x there')
            model_dir, to_download = self.check_model_path(self.model_dir, [os.path.join(m, f) for m in members for f in member_files])
            if to_download:
                for link in self.download_links:
                    try:
                        self.download(link, model_dir, extract=True, flatten=False)
                        break
                    except Exception:
                        continue
        missing = [m for m in members if not all(os.path.exists(os.path.join(model_dir, m, f)) for f in member_files)]
        if missing:
            raise FileNotFoundError(
                f'OMNI-P3x model files not found: {model_dir} lacks {", ".join(missing)} (directories with model.jpt and sae.yaml). '
                f'Unzip omnip3x_model.zip from https://aitomistic.xyz/model/omnip3x_model.zip '
                f'(doi:10.5281/zenodo.24344108) into {self.model_dir} in ~/.mlatom/models or in a directory '
                f'listed in $MODELSPATH.')
        self.members = [models.msaioaimnet2(model_file=os.path.join(model_dir, member), verbose=0) for member in members]
        # Elements with self-atomic energies (SAE) at each level of theory, in every member. Fine-tuning with
        # refitted SAE leaves the levels and elements absent from its data without them; predict refuses those.
        sae = {}
        for member in members:
            with open(os.path.join(model_dir, member, 'sae.yaml')) as f:
                for entry in yaml.safe_load(f)['lots']:
                    lot, elements = tuple(entry['descriptor']), {int(z) for z in entry['shifts']}
                    sae[lot] = sae.get(lot, elements) & elements
        # level name -> atomic numbers (plain types: model.dump writes dict attributes to JSON)
        self.sae_elements = {name: sorted(sae.get(tuple(descriptor), set())) for name, (descriptor, _) in self.levels.items()}

    def predict(
        self,
        molecular_database=None,
        molecule=None,
        calculate_energy=True,
        calculate_energy_gradients=False,
        calculate_hessian=False,
        calculate_dipole_moment=False,
        nstates=1,
        current_state=0,
        level='TDA-wB97X-D/def2-SVP',
    ):
        '''
        Predict the properties of the nstates lowest singlet states at the level of theory ``level``
        (see the class documentation). Excited states are available only at TDA-wB97X-D/def2-SVP;
        its training data cover S0-S10, i.e., nstates <= 11. The energies are always calculated;
        ``molecule.electronic_states`` is replaced by the nstates predicted states and the
        molecule-level properties are those of ``current_state``. For nstates > 1, the molecule
        gets the S0->Sn ``oscillator_strengths`` if the model has an oscillator-strength head.
        ``calculate_energy_gradients`` is True for all states or one bool per state
        (states without gradients get NaN gradients). Levels of
        theory and elements without self-atomic energies (left out of the data of a fine-tuning)
        are refused.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        name = {key.casefold(): key for key in self.levels}.get(level.casefold())
        if name is None:
            raise ValueError(f'Unknown level {level!r}; the OMNI-P3x levels are {", ".join(self.levels)}')
        descriptor, level_elements = self.levels[name]
        if nstates > 1 and name != 'TDA-wB97X-D/def2-SVP':
            raise ValueError(f'Excited states are available only at TDA-wB97X-D/def2-SVP; {name} is a ground-state level')
        supported = level_elements & set(self.sae_elements[name])
        if not supported:
            raise ValueError(f'This OMNI-P3x has no self-atomic energies at {name}: its fine-tuning data did not include this level')
        for mol in molDB.molecules:
            if not set(mol.atomic_numbers) <= supported:
                raise ValueError(f'OMNI-P3x at {name} supports only molecules containing '
                                 + ', '.join(data.atomic_number2element_symbol[z] for z in sorted(supported)))

        # each member predicts on its own copy (coordinates, elements, charge) of the molecules
        member_dbs = []
        for member in self.members:
            member_db = molDB.copy(atomic_labels=['xyz_coordinates'], molecular_labels=[])
            member.predict(molecular_database=member_db, nstates=nstates, current_state=current_state,
                           calculate_energy=True, calculate_energy_gradients=calculate_energy_gradients,
                           calculate_hessian=calculate_hessian, calculate_dipole_moment=calculate_dipole_moment,
                           lot_descriptor=descriptor)
            member_dbs.append(member_db)

        for imol, mol in enumerate(molDB.molecules):
            member_mols = [member_db.molecules[imol] for member_db in member_dbs]
            mol.electronic_states = []
            for istate in range(nstates):
                member_states = [member_mol.electronic_states[istate] for member_mol in member_mols]
                state = mol.copy(atomic_labels=['xyz_coordinates'], molecular_labels=[])
                member_energies = [s.energy for s in member_states]
                state.energy = np.mean(member_energies)
                state.energy_standard_deviation = np.std(member_energies)
                state.AIMNet_charges = np.mean([s.AIMNet_charges for s in member_states], axis=0)
                if calculate_energy_gradients:
                    state.energy_gradients = np.mean([s.energy_gradients for s in member_states], axis=0)
                if calculate_hessian:
                    state.hessian = np.mean([s.hessian for s in member_states], axis=0)
                if calculate_dipole_moment:
                    state.dipole_moment = np.mean([s.dipole_moment for s in member_states], axis=0)
                mol.electronic_states.append(state)
            if name == 'wB97M-D3(BJ)/def2-TZVPP':
                self.add_dispersion(mol.electronic_states[0], calculate_energy_gradients, calculate_hessian)
            # A model can be saved without an oscillator-strength head.
            if nstates > 1 and 'oscillator_strengths' in member_mols[0].__dict__:
                mol.oscillator_strengths = np.mean([member_mol.oscillator_strengths for member_mol in member_mols], axis=0)
            else:
                mol.__dict__.pop('oscillator_strengths', None)

            current = mol.electronic_states[current_state]
            mol.energy = current.energy
            mol.energy_standard_deviation = current.energy_standard_deviation
            mol.AIMNet_charges = current.AIMNet_charges
            if calculate_energy_gradients:
                mol.energy_gradients = current.energy_gradients
            if calculate_hessian:
                mol.hessian = current.hessian
            if calculate_dipole_moment:
                mol.dipole_moment = current.dipole_moment

    def add_dispersion(self, state, calculate_energy_gradients=False, calculate_hessian=False):
        '''
        Add the D3(BJ) dispersion of the wB97M-D3(BJ)/def2-TZVPP level to a state: aimnet's DFTD3
        with the parameters of the AIMNet2 wB97M-D3 model (s6 1, s8 0.3908, a1 0.5660, a2 3.1280;
        C6 and C8 terms, no three-body term), derivatives by automatic differentiation.
        '''
        import torch
        from aimnetex import constants, nbops, ops
        from aimnetex.modules import DFTD3
        if not hasattr(self, '_dftd3'):
            self._dftd3 = DFTD3(s8=0.3908, a1=0.5660, a2=3.1280).double()
        numbers = torch.tensor(state.atomic_numbers, dtype=torch.long)[None]

        def energy(coord):  # Hartree
            d3_input = nbops.calc_masks(nbops.set_nb_mode({'coord': coord[None], 'numbers': numbers}))
            d3_input['d_ij'] = ops.calc_distances(d3_input)[0]
            return self._dftd3(d3_input)['energy'].sum() / constants.Hartree

        coord = torch.tensor(state.xyz_coordinates, dtype=torch.float64, requires_grad=True)
        dispersion = energy(coord)
        state.energy += dispersion.item()
        if calculate_energy_gradients:
            state.energy_gradients = state.energy_gradients + torch.autograd.grad(dispersion, coord)[0].numpy()
        if calculate_hessian:
            hessian = torch.autograd.functional.hessian(energy, coord.detach())
            state.hessian = state.hessian + hessian.reshape(coord.numel(), coord.numel()).numpy()

    def train(
        self,
        molecular_database=None,
        property_to_learn='energy',
        xyz_derivative_property_to_learn=None,
        validation_molecular_database='sample_from_molecular_database',
        hyperparameters={},
        spliting_ratio=0.9,
        dipole_property=None,
        train_osc=True,
        charges_property=None,
        reset_energy_shifter=True,
        drop_osc=False,
    ):
        '''
        Fine-tune the ensemble members (all four, or member ``model_index``), each on its own random
        training/validation split, and predict with them afterwards. The members are saved in
        ``model_file`` (``omnip3x_<uuid>`` if it is not set) as cv0 to cv3, from where
        ``ml.methods(method='OMNI-P3x', model_file=...)`` loads them.

        The arguments are those of ``msaioaimnet2.train``. The training data are taken at
        TDA-wB97X-D/def2-SVP unless a molecule carries the ``descriptor`` of another level
        (e.g. ``mol.descriptor = [1, 0, 0]`` for CCSD(T)*/CBS). The default hyperparameters follow the
        fine-tuning protocol of the OMNI-P3x paper: learning rate 5e-5 after a 5-epoch warm-up,
        multiplied by 0.75 after 10 epochs without improvement until it falls below 5e-6; loss
        weights 1 (energies), 1 (forces), 0.5 (oscillator strengths) and 0.1 (dipole moments).

        Arguments:
            train_osc (bool, optional): Train the oscillator-strength head on the molecules'
                ``oscillator_strengths`` (default). False keeps an existing head's weights fixed
                while training the shared network; its oscillator strengths are then no longer
                fitted and can become much less accurate (a warning is printed).
            drop_osc (bool, optional): Remove the oscillator-strength head. Requires
                ``train_osc=False``.
            reset_energy_shifter (bool, optional): Refit the self-atomic energies on the training data
                (default). The data must then cover every element and level of theory the fine-tuned
                OMNI-P3x is used for; predict refuses the others. False keeps the pretrained
                self-atomic energies.
        '''
        if train_osc and drop_osc:
            raise ValueError('drop_osc=True requires train_osc=False')
        self.set_num_threads()
        if not self.model_file:
            self.model_file = os.path.abspath(f'omnip3x_{uuid.uuid4()}')
            self.init_kwargs['model_file'] = self.model_file
        hyperparameters = {'learning_rate': 5e-5, 'early_stopping_learning_rate': 5e-6, 'force_weight': 1.0,
                           **hyperparameters}
        for member in self.members:
            member_file = os.path.join(self.model_file, os.path.basename(member.model_file))   # cv<index>
            if os.path.abspath(member.model_file) != member_file:
                shutil.copytree(member.model_file, member_file, dirs_exist_ok=True)
            models.msaioaimnet2(model_file=member_file).train(
                molecular_database=molecular_database,
                property_to_learn=property_to_learn,
                xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
                validation_molecular_database=validation_molecular_database,
                hyperparameters=hyperparameters,
                spliting_ratio=spliting_ratio,
                dipole_property=dipole_property,
                train_osc=train_osc,
                drop_osc=drop_osc,
                charges_property=charges_property,
                reset_energy_shifter=reset_energy_shifter,
            )
        self.load()   # the fine-tuned members and their self-atomic energies, as saved
