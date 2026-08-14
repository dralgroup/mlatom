#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! aiqm1: Artificial intelligence quantum-mechanical method 1                ! 
  ! Implementations by: Peikung Zheng & Pavlo O. Dral                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os
import sys
from collections import OrderedDict

from . import data
from . import dispersion as dispersion_utils
from . import delta_learning
from .model_cls import model, torchani_model, method_model, model_tree_node, downloadable_model
from .interfaces.torchani_interface import ani


class aiqm1_ani_wrapper(ani):
    """
    Rebuilds an original AIQM1 ensemble member as a fully trainable
    :class:`ani` object, matching its AEV and network architecture.

    That is all this class does; the prediction keywords AIQM1 passes down and a
    network has no answer for are handled by :attr:`ani.ignorable_prediction_kwargs`,
    not here. Defined at module level so saved model trees can be reloaded.
    """

    @classmethod
    def from_pretrained(cls, member, device=None):
        """
        Convert an :class:`ani_nns_in_aiqm1` member into a trainable
        :class:`aiqm1_ani_wrapper` instance.

        Parameters
        ----------
        member : ani_nns_in_aiqm1
            Original AIQM1 ensemble member.
        device : torch.device or str, optional
            Target device. Defaults to ``member.device``.

        Returns
        -------
        aiqm1_ani_wrapper
        """
        import torch
        import torchani

        if device is None:
            device = member.device
        else:
            device = torch.device(device)

        species_order = [data.atomic_number2element_symbol[z] for z in member.species_order]

        hyperparameters = {
            'Rcr': float(member.aev_computer.Rcr),
            'Rca': float(member.aev_computer.Rca),
            'EtaR': member.aev_computer.EtaR.reshape(-1).tolist(),
            'ShfR': member.aev_computer.ShfR.reshape(-1).tolist(),
            'Zeta': member.aev_computer.Zeta.reshape(-1).tolist(),
            'ShfZ': member.aev_computer.ShfZ.reshape(-1).tolist(),
            'EtaA': member.aev_computer.EtaA.reshape(-1).tolist(),
            'ShfA': member.aev_computer.ShfA.reshape(-1).tolist(),
            'activation_function': 'GELU',
        }

        instance = cls(hyperparameters=hyperparameters, device=str(device), verbose=0)
        instance.species_order = species_order
        instance.argsdict['species_order'] = species_order
        instance.aev_computer = member.aev_computer.to(device)

        networkdict = OrderedDict()
        state_dict = OrderedDict()
        for i, specie in enumerate(species_order):
            networkdict[specie] = member.nn[str(i)]
            for key, value in member.nn[str(i)].state_dict().items():
                state_dict[f'{specie}.{key}'] = value

        instance.networkdict = networkdict
        instance.nn = torchani.ANIModel(networkdict)
        instance.nn.load_state_dict(state_dict)
        instance.neurons = [
            [layer.out_features for layer in networkdict[specie] if isinstance(layer, torch.nn.Linear)]
            for specie in species_order
        ]

        instance.energy_shifter = torchani.utils.EnergyShifter([0.0] * len(species_order))
        instance.energy_shifter.to(device)

        instance.model = torchani.nn.Sequential(instance.aev_computer, instance.nn).to(device).double()
        instance.model.eval()
        instance.optimizer_setup(**instance.hyperparameters)

        return instance


class aiqm1(torchani_model, method_model, downloadable_model):
    """
    The Artificial intelligence-quantum mechanical method as in the `AIQM1 paper`_.

    Arguments:
        method (str, optional): AIQM method used. Currently supports AIQM1, AIQM1\@DFT*, and AIQM1\@DFT. Default value: AIQM1.
        qm_program (str): The QM program used in the calculation of ODM2* part. Currently supports MNDO and Sparrow program. 
        qm_program_kwargs (dictionary, optional): Keywords passed to QM program.
        baseline_kwargs (dictionary, optional): Keywords passed to the ODM2* baseline method.
        dispersion_kwargs (dictionary, optional): Keywords passed to the D4 dispersion method.
        model_index (int or list, optional): Ensemble member(s) to fine-tune. Defaults to all 8 members.
        device (str or torch.device, optional): Device used for the ANI NN part.
        working_directory (str, optional): Working directory for baseline/dispersion calculations.
        nthreads (int, optional): Number of threads. Default: 1.

    .. _AIQM1 Paper:
        https://doi.org/10.1038/s41467-021-27340-2

    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run AIQM1 calculation
        aiqm1 = ml.methods(method='AIQM1', qm_program='MNDO')
        aiqm1.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True)
        # Get energy, gradient, and prediction uncertainty of AIQM1
        energy = mol.energy
        gradient = mol.gradient
        std = mol.aiqm1_nn.energy_standard_deviation


    """

    supported_methods = ['AIQM1', 'AIQM1@DFT', 'AIQM1@DFT*']
    atomic_energies = {'AIQM1': {1:-0.50088038, 6:-37.79221710, 7:-54.53360298, 8:-75.00986203},
                       'AIQM1@DFT': {1:-0.50139362, 6:-37.84623117, 7:-54.59175573, 8:-75.07674376}}
    atomic_energies['AIQM1@DFT*'] = atomic_energies['AIQM1@DFT']

    _tl = True
    verbose = 1

    def __init__(self,
                 method='AIQM1',
                 qm_program=None,
                 qm_program_kwargs=None,
                 dftd4_kwargs=None,
                 baseline_kwargs=None,
                 dispersion_kwargs=None,
                 working_directory=None,
                 nthreads=1,
                 model_index=None,
                 device=None,
                 **kwargs):

        import torch

        self.method = method.upper()
        self.model_name = self.method.lower().replace('*', 'star').replace('@', 'at')
        self.qm_program = qm_program
        if self.qm_program is None and 'program' in kwargs:
            self.qm_program = kwargs.pop('program')

        if qm_program_kwargs is None:
            self.qm_program_kwargs = {}
        else:
            self.qm_program_kwargs = dict(qm_program_kwargs)
        if dftd4_kwargs is None:
            self.dftd4_kwargs = {}
        else:
            self.dftd4_kwargs = dict(dftd4_kwargs)

        if baseline_kwargs is None:
            self.baseline_kwargs = dict(self.qm_program_kwargs)
            if qm_program is not None and 'program' not in self.baseline_kwargs:
                self.baseline_kwargs['program'] = qm_program
        else:
            self.baseline_kwargs = dict(baseline_kwargs)

        # ``None`` -> the model's own dispersion term, D4(wB97X) for AIQM1 and
        # none for AIQM1@DFT*.  ``False`` switches it off; ``{}`` raises.  The
        # stored spec identifies the term only; where it runs is decided when the
        # node is built, so an unrelated argument can never turn it on or off.
        if dispersion_kwargs is None:
            if self.dftd4_kwargs:
                self.dispersion_kwargs = dispersion_utils.normalize(
                    dict(self.dftd4_kwargs, method=self.dftd4_kwargs.get('method', 'd4')))
            else:
                self.dispersion_kwargs = dispersion_utils.native_dispersion_kwargs(self.method)
        else:
            self.dispersion_kwargs = dispersion_utils.normalize(dispersion_kwargs)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if model_index is None:
            self.model_index = [ii for ii in range(8)]
        elif isinstance(model_index, int):
            self.model_index = [model_index]
        elif isinstance(model_index, list):
            self.model_index = model_index
        else:
            raise ValueError(f"Unrecognized model_index type: {type(model_index)}. Please provide int, list or None.")

        if working_directory is not None:
            self.set_working_directory(working_directory)
        else:
            self._working_directory = None

        self.element_symbols_available = ['H', 'C', 'N', 'O']
        self.nthreads = nthreads
        self.use_atomic_shift = True
        self.load_model()

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        if hasattr(self, 'aiqm1_model'):
            self.aiqm1_model.nthreads = self._nthreads

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value):
        self._working_directory = value
        self.set_working_directory(value)
        if hasattr(self, 'nn'):
            self.reload_atomic_shift()
            self.reload_baseline()
            self.reload_dispersion()
            self.reload()
        else:
            self.load_model()

    def set_working_directory(self, working_directory):
        self._working_directory = os.path.abspath(working_directory)
        self.qm_program_kwargs['working_directory'] = self._working_directory
        self.dftd4_kwargs['working_directory'] = self._working_directory
        self.baseline_kwargs['working_directory'] = self._working_directory

    def dispersion_working_directory(self):
        return self._working_directory

    def load_model(self):
        from .models import methods
        modelname = self.model_name
        ani_nn_children = []

        for ii in range(8):
            nn_i = model_tree_node(name=f'{modelname}_nn{ii}', operator='predict',
                                   model=ani_nns_in_aiqm1(method=self.method, model_index=ii))
            ani_nn_children.append(nn_i)
        ani_nns = model_tree_node(name=f'{modelname}_nn', children=ani_nn_children, operator='average')
        self.nn = ani_nns

        self.reload_atomic_shift()
        self.reload_baseline()
        self.reload_dispersion()

        aiqm1_children = [ani_nns, self.atomic_shift_node, self.baseline]
        if self.d4 is not None:
            aiqm1_children.append(self.d4)
        self.aiqm1_model = model_tree_node(name=modelname, children=aiqm1_children, operator='sum')

    def reload_atomic_shift(self):
        if self.use_atomic_shift:
            self.atomic_shift_node = model_tree_node(
                name=f'{self.model_name}_atomic_energy_shift',
                operator='predict',
                model=atomic_energy_shift(method=self.method)
            )
        else:
            self.atomic_shift_node = None

    def reload_baseline(self):
        from .models import methods
        self.baseline = model_tree_node(
            name='odm2star',
            operator='predict',
            model=methods(method='ODM2*', **self.baseline_kwargs)
        )

    def reload_dispersion(self):
        self.d4 = dispersion_utils.build_node(
            self.dispersion_kwargs, name='d4_wb97x',
            working_directory=self.dispersion_working_directory())

    def reload(self):
        aiqm1_children = [self.nn]
        if self.use_atomic_shift and self.atomic_shift_node is not None:
            aiqm1_children.append(self.atomic_shift_node)
        aiqm1_children.append(self.baseline)
        if self.d4 is not None:
            aiqm1_children.append(self.d4)
        self.aiqm1_model = model_tree_node(name=self.model_name, children=aiqm1_children, operator='sum')

    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False,
                nstates=1, current_state=0, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        if 'nthreads' in self.__dict__: self.aiqm1_model.nthreads = self.nthreads
        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol,
                                      calculate_energy=calculate_energy,
                                      calculate_energy_gradients=calculate_energy_gradients,
                                      calculate_hessian=calculate_hessian,
                                      nstates=nstates,
                                      current_state=current_state,
                                      **kwargs)

    def predict_for_molecule(self, molecule=None,
                             calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False,
                             nstates=1, current_state=0, **kwargs):

        for atom in molecule.atoms:
            if not atom.atomic_number in [1, 6, 7, 8]:
                errmsg = ' * Warning * Molecule contains elements other than CHNO, no calculations performed'
                raise ValueError(errmsg)

        if nstates > 1:
            mol_copy = molecule.copy()
            mol_copy.electronic_states = []
            for _ in range(nstates - len(molecule.electronic_states)):
                molecule.electronic_states.append(mol_copy.copy())

        if self.use_atomic_shift and len(molecule.atoms) == 1:
            molecule.energy = self.atomic_energies[self.method][molecule.atoms[0].atomic_number]
            standard_atom = data.atom(atomic_number=molecule.atoms[0].atomic_number)
            if molecule.charge != 0 or molecule.multiplicity != standard_atom.multiplicity:
                from .models import methods
                odm2model = methods(method='ODM2*', program=self.qm_program, working_directory=self.working_directory)
                mol_odm2 = molecule.copy()
                odm2model.predict(molecule=mol_odm2, nstates=nstates, **kwargs)
                mol_standard_odm2 = molecule.copy()
                mol_standard_odm2.charge = 0
                mol_standard_odm2.multiplicity = standard_atom.multiplicity
                odm2model.predict(molecule=mol_standard_odm2, nstates=nstates, **kwargs)
                molecule.energy = molecule.energy + mol_odm2.energy - mol_standard_odm2.energy
        else:
            if nstates > 1 and isinstance(calculate_energy_gradients, list):
                if any(calculate_energy_gradients):
                    calculate_energy_gradients = [True] * nstates
            self.aiqm1_model.predict(molecule=molecule,
                                     calculate_energy=calculate_energy,
                                     calculate_energy_gradients=calculate_energy_gradients,
                                     calculate_hessian=calculate_hessian,
                                     nstates=nstates,
                                     current_state=current_state,
                                     **kwargs)

            properties = []
            atomic_properties = []

            calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
            calculate_hessian = bool(np.array(calculate_hessian).any())
            if calculate_energy: properties.append('energy')
            if calculate_energy_gradients: atomic_properties.append('energy_gradients')
            if calculate_hessian: properties.append('hessian')
            modelname = self.model_name

            if nstates > 1:
                for mol_el_st in molecule.electronic_states:
                    mol_el_st.__dict__[f'{modelname}_nn'].standard_deviation(properties=properties + atomic_properties)
            else:
                molecule.__dict__[f'{modelname}_nn'].standard_deviation(properties=properties + atomic_properties)

    def _load_pretrained_ani_members(self):
        members = []
        for idx in self.model_index:
            member = ani_nns_in_aiqm1(method=self.method, model_index=idx)
            wrapper = aiqm1_ani_wrapper.from_pretrained(member, device=self.device)
            members.append(wrapper)
        return members

    def train(self, **kwargs):
        kwargs['save_model'] = True
        kwargs['reset_parameters'] = False
        # Rebuild the network around the pretrained weights: fine-tuning data may
        # bring elements the original member never saw. Stated here rather than
        # left to train()'s default, which is False so that ani.train() on an
        # existing model keeps continuing it, as it always has.
        kwargs['reset_network'] = True
        kwargs['reset_aev'] = True
        kwargs['reset_optimizer'] = True

        file_to_save_model = kwargs.pop('file_to_save_model', None)
        verbose = kwargs.get('verbose', 0)

        # Transfer-learning defaults: subtract only the ODM2* baseline.  The
        # fixed atomic-energy shift is *not* removed; instead the ANI SAE is
        # fitted from the delta labels.  A dispersion correction can be requested
        # explicitly with ``dispersion_kwargs`` (e.g. ``{'functional': 'b3lyp'}``).
        # It is subtracted from the reference labels AND added back at prediction -
        # the same term on both sides.  No default where the model has a native
        # term; the declaration comes from dispersion_kwargs= or from a prepared
        # delta_db='s provenance stamp, never neither and never two that disagree.
        tl_dispersion_kwargs = kwargs.pop('dispersion_kwargs', 'not given')

        property_to_learn = kwargs.get('property_to_learn', 'energy')
        xyz_derivative_property_to_learn = kwargs.get('xyz_derivative_property_to_learn', None)

        resolved = delta_learning.resolve_training_labels(
            model_method=self.method,
            molecular_database=kwargs.get('molecular_database', None),
            delta_db=kwargs.pop('delta_db', None),
            # Reusing an already-computed baseline is what makes trying a second
            # dispersion choice on the same geometries free; it accepts a prepared
            # database and takes the component out of it by name.
            baseline_db=kwargs.pop('baseline_db', None),
            dispersion_db=kwargs.pop('dispersion_db', None),
            dispersion_kwargs=tl_dispersion_kwargs,
            baseline_method='ODM2*',
            baseline_kwargs=self.baseline_kwargs,
            qm_program=self.qm_program,
            property_to_learn=property_to_learn,
            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
            working_directory=self.working_directory,
            subtract_atomic_shift=False,
            verbose=verbose,
        )
        self.dispersion_kwargs = resolved['dispersion_kwargs']
        self._training_database = resolved['database']
        self._training_delta_source = resolved['delta_source']
        self._training_stamp = resolved['stamp']
        delta_db = resolved['database']

        kwargs['molecular_database'] = delta_db
        kwargs['property_to_learn'] = 'delta_energy'
        if xyz_derivative_property_to_learn:
            kwargs['xyz_derivative_property_to_learn'] = 'delta_energy_gradients'

        hyperparameters = kwargs.get('hyperparameters', {})
        if 'fixed_layers' not in hyperparameters:
            hyperparameters['fixed_layers'] = [[0,4]]
        if 'loss_type' not in hyperparameters:
            hyperparameters['loss_type'] = 'geometric'
        if 'max_epochs' not in hyperparameters:
            hyperparameters['max_epochs'] = 100
        kwargs['hyperparameters'] = hyperparameters

        self.use_atomic_shift = False

        if hasattr(self, 'nn'):
            del self.nn
        if hasattr(self, 'aiqm1_model'):
            del self.aiqm1_model

        pretrained_models = self._load_pretrained_ani_members()

        # Reload AEV parameters from the first member into hyperparameters so that
        # training keeps the AIQM1-specific AEV setup.
        first_member = pretrained_models[0]
        for aev_param in ['Rcr', 'Rca', 'EtaR', 'ShfR', 'Zeta', 'ShfZ', 'EtaA', 'ShfA']:
            if aev_param not in hyperparameters:
                try:
                    hyperparameters[aev_param] = first_member.aev_computer._buffers[aev_param].reshape(-1,)
                except Exception:
                    hyperparameters[aev_param] = getattr(first_member.aev_computer, aev_param)
        kwargs['hyperparameters'] = hyperparameters

        retrained_models = []
        modelname = self.model_name
        save_dir = file_to_save_model if file_to_save_model else f'{modelname}_retrained'
        os.makedirs(save_dir, exist_ok=True)

        import torchani
        for ii, pmodel in enumerate(pretrained_models):
            print(f'\nStart retraining on model {self.model_index[ii]}...')
            sys.stdout.flush()
            kwargs['file_to_save_model'] = os.path.join(save_dir, f'cv{self.model_index[ii]}.pt')
            kwargs['reset_energy_shifter'] = True
            pmodel.element_symbols_available = self.element_symbols_available
            pmodel.train(**kwargs)
            retrained_models.append(pmodel)

        self.nn = model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn{self.model_index[ii]}',
                    model=retrained_models[ii],
                    operator='predict'
                ) for ii in range(len(self.model_index))
            ],
            operator='average'
        )

        self.reload_atomic_shift()
        self.reload_baseline()
        self.reload_dispersion()
        self.reload()

        self._training_databases = delta_learning.write_training_database(
            self._training_database, save_dir,
            delta_source=self._training_delta_source, verbose=verbose)
        self.energy_expression = delta_learning.report_model_composition(
            baseline='ODM2*', neural_network='dNN',
            dispersion_kwargs=self.dispersion_kwargs, verbose=verbose)

        # Leave a directory that can be loaded back, rather than one holding
        # weights that only become a model after a separate save() call.
        self.save(save_dir)

        self.element_symbols_available = retrained_models[0].species_order

    def _iter_ani_leaves(self, node=None):
        """
        Iterate over the ``aiqm1_ani_wrapper`` leaf models inside the AIQM1 tree.
        """
        if node is None:
            node = self.aiqm1_model
        if node.model is not None and isinstance(node.model, aiqm1_ani_wrapper):
            try:
                cv_index = int(node.name.split('_nn')[-1])
            except (ValueError, IndexError):
                cv_index = 0
            yield node, cv_index
        if node.children:
            for child in node.children:
                yield from self._iter_ani_leaves(child)

    @staticmethod
    def _resolve_model_paths(model_dict, base_dir):
        """Recursively resolve relative ``model_file`` paths."""
        if not isinstance(model_dict, dict):
            return
        if model_dict.get('type') == 'ml_model' and 'kwargs' in model_dict:
            kwargs = model_dict['kwargs']
            if 'model_file' in kwargs:
                model_file = kwargs['model_file']
                if not os.path.isabs(model_file):
                    kwargs['model_file'] = os.path.normpath(os.path.join(base_dir, model_file))
        for child in model_dict.get('children') or []:
            aiqm1._resolve_model_paths(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm1._resolve_model_paths(model, base_dir)

    @staticmethod
    def _make_paths_relative(model_dict, base_dir):
        """Recursively make ``model_file`` paths relative to ``base_dir``."""
        if not isinstance(model_dict, dict):
            return
        if model_dict.get('type') == 'ml_model' and 'kwargs' in model_dict:
            kwargs = model_dict['kwargs']
            if 'model_file' in kwargs:
                model_file = kwargs['model_file']
                if os.path.isabs(model_file):
                    kwargs['model_file'] = os.path.relpath(model_file, base_dir)
        for child in model_dict.get('children') or []:
            aiqm1._make_paths_relative(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm1._make_paths_relative(model, base_dir)

    def save(self, model_file=''):
        """
        Save the transfer-learned AIQM1 model to a directory.

        The directory will contain:

        - ``tree.json``: model metadata and tree structure
        - ``cv*.pt``:    TorchANI weights for each ensemble member
        """
        import json
        if not model_file:
            model_file = f'{self.model_name}_tl_model'
        os.makedirs(model_file, exist_ok=True)
        base_dir = os.path.abspath(model_file)

        for node, cv_index in self._iter_ani_leaves():
            pt_path = os.path.join(base_dir, f'cv{cv_index}.pt')
            node.model.save(pt_path)

        model_tree_dict = self.aiqm1_model.dump(format='dict')
        self._make_paths_relative(model_tree_dict, base_dir)

        tree = {
            'type': 'aiqm1',
            'module': {
                'name': self.__module__,
                'path': sys.modules[self.__module__].__spec__.origin,
            },
            'tl': True,
            'method': self.method,
            'model_index': self.model_index,
            'device': str(self.device),
            'element_symbols_available': self.element_symbols_available,
            'baseline_kwargs': self.baseline_kwargs,
            'dispersion_kwargs': self.dispersion_kwargs,
            'energy_expression': getattr(self, 'energy_expression', None),
            'training_labels': getattr(self, '_training_databases', None),
            'provenance': getattr(self, '_training_stamp', None),
            'qm_program': self.qm_program,
            'working_directory': self.working_directory,
            'model_tree': model_tree_dict,
        }
        tree_path = os.path.join(base_dir, 'tree.json')
        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=4)
        if self.verbose:
            print(f'AIQM1 TL model saved in {base_dir}')

    @classmethod
    def load(cls, model_file, device=None):
        """
        Load a transfer-learned AIQM1 model from a directory or ``tree.json``.
        """
        import json
        if os.path.isdir(model_file):
            tree_file = os.path.join(model_file, 'tree.json')
        else:
            tree_file = model_file
        if not os.path.isfile(tree_file):
            raise FileNotFoundError(f'Cannot find tree.json at {tree_file}')

        with open(tree_file) as f:
            model_dict = json.load(f)
        base_dir = os.path.dirname(os.path.abspath(tree_file))
        if device is None and 'device' in model_dict:
            device = model_dict['device']
        return cls.from_dict(model_dict, device=device, base_dir=base_dir)

    @classmethod
    def from_dict(cls, model_dict, device=None, base_dir=None):
        """Reconstruct an ``aiqm1`` instance from a dictionary produced by :meth:`save`."""
        import torch
        instance = cls.__new__(cls)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        instance.device = torch.device(device)
        instance.method = model_dict['method']
        instance.model_name = instance.method.lower().replace('*', 'star').replace('@', 'at')
        instance.model_index = model_dict.get('model_index', [ii for ii in range(8)])
        instance._tl = True
        instance.verbose = 1
        instance.element_symbols_available = model_dict.get('element_symbols_available', ['H', 'C', 'N', 'O'])
        instance.baseline_kwargs = dict(model_dict.get('baseline_kwargs', {}))
        instance.dispersion_kwargs = model_dict.get('dispersion_kwargs', None)
        instance.qm_program = model_dict.get('qm_program', None)
        instance.qm_program_kwargs = dict(instance.baseline_kwargs)
        instance.dftd4_kwargs = dict(instance.dispersion_kwargs or {})
        instance.energy_expression = model_dict.get('energy_expression', None)
        instance._training_databases = model_dict.get('training_labels', None)
        instance._training_stamp = model_dict.get('provenance', None)
        instance._working_directory = None
        wd = model_dict.get('working_directory', None)
        if wd is not None:
            instance._working_directory = wd
            instance.baseline_kwargs['working_directory'] = wd
            instance.qm_program_kwargs['working_directory'] = wd
            instance.dftd4_kwargs['working_directory'] = wd

        if base_dir is not None:
            cls._resolve_model_paths(model_dict['model_tree'], base_dir)
        from .models import load_dict
        instance.aiqm1_model = load_dict(model_dict['model_tree'], base_dir=base_dir)

        instance.nn = None
        instance.atomic_shift_node = None
        instance.baseline = None
        instance.d4 = None
        instance.use_atomic_shift = False
        for child in instance.aiqm1_model.children:
            if child.name == f'{instance.model_name}_nn':
                instance.nn = child
            elif 'atomic_energy_shift' in child.name:
                instance.atomic_shift_node = child
                instance.use_atomic_shift = True
            elif child.name == 'odm2star':
                instance.baseline = child
            elif child.name == 'dispersion':
                instance.d4 = child

        instance.nthreads = 1
        return instance


class atomic_energy_shift(model):
    atomic_energy_shifts = {'AIQM1': {1: -4.29365862e-02, 6: -3.34329586e+01, 7: -4.69301173e+01, 8: -6.29634763e+01},
                            'AIQM1@DFT': {1: -4.27888067e-02, 6: -3.34869833e+01, 7: -4.69896148e+01, 8: -6.30294433e+01}}
    atomic_energy_shifts['AIQM1@DFT*'] = atomic_energy_shifts['AIQM1@DFT']

    def __init__(self, method='AIQM1'):
        self.method = method

    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
        calculate_hessian = bool(np.array(calculate_hessian).any())

        for mol in molDB.molecules:
            molecules = [mol]

            if nstates > 1:
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())
                molecules = mol.electronic_states

            for mol in molecules:
                if calculate_energy:
                    sae = 0.0
                    for atom in mol.atoms:
                        sae += self.atomic_energy_shifts[self.method][atom.atomic_number]
                    mol.energy = sae
                if calculate_energy_gradients:
                    for atom in mol.atoms:
                        atom.energy_gradients = np.zeros(3)
                if calculate_hessian:
                    ndim = len(mol.atoms) * 3
                    mol.hessian = np.zeros(ndim * ndim).reshape(ndim, ndim)


class ani_nns_in_aiqm1(torchani_model, downloadable_model):
    species_order = [1, 6, 7, 8]

    def __init__(self, method='AIQM1', model_index=0):
        import torch
        if method == 'AIQM1':
            self.level = 'cc'
        elif method in ['AIQM1@DFT', 'AIQM1@DFT*']:
            self.level = 'dft'
        self.method = method
        self.model_index = model_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.define_aev()
        self.load_model()

    def define_aev(self):
        import torch
        Rcr = 5.2000e+00
        Rca = 4.0000e+00
        EtaR = torch.tensor([1.6000000e+01], device=self.device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00,
                             2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00,
                             3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00,
                             4.9312500e+00], device=self.device)
        Zeta = torch.tensor([3.2000000e+01], device=self.device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00,
                             2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=self.device)
        EtaA = torch.tensor([8.0000000e+00], device=self.device)
        ShfA = torch.tensor([9.0000000e-01, 1.6750000e+00, 2.4499998e+00, 3.2250000e+00], device=self.device)
        num_species = len(self.species_order)
        import torchani
        aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        self.aev_computer = aev_computer

    def load_model(self):
        import torch
        method = 'aiqm1_' + self.level
        self.define_nn()

        if method == 'aiqm1_cc':
            download_links = [
                'https://zenodo.org/records/15383390/files/aiqm1_cc_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_odm2star_cc_20211202.zip']
            model_dir = 'aiqm1_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]

        elif method == 'aiqm1_dft':
            download_links = [
                'https://zenodo.org/records/15383390/files/aiqm1_dft_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_odm2star_dft_20211202.zip']
            model_dir = 'aiqm1_dft_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        if to_download: self.download(download_links, mlatom_model_dir)

        checkpoint = torch.load(os.path.join(mlatom_model_dir, f'cv{self.model_index}.pt'),
                                map_location=self.device, weights_only=False)
        self.nn.load_state_dict(checkpoint['nn'])
        import torchani
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).double()

    def define_nn(self):
        import torch
        aev_dim = self.aev_computer.aev_length
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 144),
            torch.nn.GELU(),
            torch.nn.Linear(144, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )

        import torchani
        nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
        self.nn = nn

    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        from torchani.utils import ChemicalSymbolsToInts

        calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
        calculate_hessian = bool(np.array(calculate_hessian).any())
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)

        for mol in molDB.molecules:
            molecules = [mol]

            if nstates > 1:
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())
                molecules = mol.electronic_states

            import torch
            for mol in molecules:
                atomic_numbers = np.array([atom.atomic_number for atom in mol.atoms])
                xyz_coordinates = torch.tensor(np.array(mol.xyz_coordinates).astype('float')).to(self.device).requires_grad_(
                    calculate_energy_gradients or calculate_hessian)
                xyz_coordinates = xyz_coordinates.unsqueeze(0)
                species = species_to_tensor(atomic_numbers).to(self.device).unsqueeze(0)
                ANI_NN_energy = self.model((species, xyz_coordinates)).energies
                if calculate_energy: mol.energy = float(ANI_NN_energy)
                if calculate_energy_gradients or calculate_hessian:
                    ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energy.sum(), xyz_coordinates,
                                                                    create_graph=True, retain_graph=True)[0]
                    if calculate_energy_gradients:
                        grads = ANI_NN_energy_gradients[0].detach().cpu().numpy()
                        for iatom in range(len(mol.atoms)):
                            mol.atoms[iatom].energy_gradients = grads[iatom]
                if calculate_hessian:
                    import torchani
                    ANI_NN_hessian = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energy)
                    mol.hessian = ANI_NN_hessian[0].detach().cpu().numpy()


if __name__ == '__main__':
    pass
