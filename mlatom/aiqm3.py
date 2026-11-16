'''
.. code-block::

  !---------------------------------------------------------------------------!
  ! aiqm3: Artificial Intelligence-Enhanced Quantum Mechanical Method 3       !
  ! Implementations by: Yuxinxin Chen and                                     !
  !                     Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

The AIQM3 weights are fetched at first use from Zenodo record 15876694; their
own terms apply. Cite Y. Chen, Y.-F. Hou, R. Zubatyuk, O. Isayev, P.O. Dral,
J. Chem. Theory Comput. 2026, 22, 2232.
'''
import os
import sys
import json
from . import dispersion as dispersion_utils
from . import delta_learning
from .model_cls import method_model, model_tree_node, downloadable_model
from .interfaces.torchani_interface import ani

class aiqm3(method_model, downloadable_model):

    """
    GFN2-xTB based artificial intelligence quantum-mechanical method 3 (AIQM3)

    Arguments:

        method (str, optional): Currently supports AIQM3, AIQM3@DFT, AIQM3@DFT*. Default: AIQM3
        working_directory (str, optional): The path to save temporary calculation file. By default, current directory will be choosed.
        baseline_kwargs (dict, optional): Keywords passed to GFN2-xTB. Default: None
        dispersion_kwargs (dict, optional): keywords passed to dftd3. Default: None
        nthreads (int, optional): The number of threads to be used. Default: None

    .. code-block::

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run AIQM3 calculation
        aiqm3 = ml.methods(method='aiqm3')
        aiqm3.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and uncertainty of AIQM3
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian
        std = mol.aiqm3_nn.energy_standard_deviation

    Molecules containing elements outside the set the model was trained on cannot
    be treated by AIQM3 itself. Models covering more elements include the Aitomic
    add-ons (for example AIQM4@DFT) and AIQM3@UAIQM, the UAIQM extension of this
    model, which can be used out of the box for them; only elements that GFN2-xTB
    itself cannot describe are out of reach. The add-ons, including the UAIQM
    models, are installed separately (http://MLatom.com/aitomic). Specify the
    model version explicitly:

    .. code-block:: python

        uaiqm = ml.models.methods(method='uaiqm_gfn2xtbstar@cc', version='20250115')

    .. code-block:: bash

        mlatom sp uaiqm_gfn2xtbstar@cc uversion=20250115 xyzfile=mol.xyz

    """

    supported_methods = ['AIQM3', 'AIQM3@DFT', 'AIQM3@DFT*']
    _tl = True
    verbose = 1

    def __init__(
        self,
        method: str = 'AIQM3',
        working_directory: str = None,
        baseline_kwargs: dict = None,
        dispersion_kwargs: dict = None,
        nthreads: int = 1,
        model_index = None,
        device = None,
    ):

        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.method = method.lower()
        self.model_name = self.method.replace('@','at')

        if model_index is None:
            self.model_index = [ii for ii in range(8)]
        elif isinstance(model_index, list):
            self.model_index = model_index
        elif isinstance(model_index, int):
            self.model_index = [model_index]
        else:
            raise ValueError(f"Unrecognized model_index type: {type(model_index)}. Please provide int, list or None.")

        if baseline_kwargs is None: self.baseline_kwargs = {}
        else: self.baseline_kwargs = baseline_kwargs

        # ``None`` -> the model's own dispersion term, D3(BJ)/b973c for AIQM3 and
        # none for AIQM3@DFT*.  ``False`` switches it off; ``{}`` raises.  The
        # stored spec identifies the term only; where it runs is decided when the
        # node is built, so an unrelated argument can never turn it on or off.
        if dispersion_kwargs is None:
            self.dispersion_kwargs = dispersion_utils.native_dispersion_kwargs(self.method)
        else:
            self.dispersion_kwargs = dispersion_utils.normalize(dispersion_kwargs)

        self._working_directory = None
        if working_directory is not None:
            self.set_working_directory(working_directory)
        self._load_pretrained_models()
        self.nthreads = nthreads

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.aiqm3_model.nthreads = self._nthreads # nthreads can be directly assigned to each tree node

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value):
        self._working_directory = value
        self.set_working_directory(value)
        self.reload_baseline(); self.reload_dispersion()
        self.reload()

    def set_working_directory(self, working_directory):
        self._working_directory = working_directory
        self.baseline_kwargs["working_directory"] = os.path.join(os.path.abspath(working_directory), "_baseline")

    def dispersion_working_directory(self):
        if self._working_directory is None:
            return None
        return os.path.join(os.path.abspath(self._working_directory), "_dispersion")

    def reload_baseline(self):
        from .models import methods
        self.baseline = model_tree_node(
            name='gfn2xtbstar',
            model=methods(method='GFN2-xTB*', **self.baseline_kwargs),
            operator='predict'
        )

    def reload_dispersion(self):
        self.d3 = dispersion_utils.build_node(
            self.dispersion_kwargs, name='d3b973c',
            working_directory=self.dispersion_working_directory())

    def reload(self):
        children = [self.baseline, self.nn]
        if self.d3 is not None:
            children.append(self.d3)
        self.aiqm3_model = model_tree_node(
            name=self.model_name,
            children=children,
            operator='sum'
        )

    def predict(
        self,
        molecular_database=None,
        molecule=None,
        calculate_energy=True,
        calculate_energy_gradients=False,
        calculate_hessian=False,
        calculate_dipole_derivatives=False,
    ):

        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol,
                                calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian,
                                calculate_dipole_derivatives=calculate_dipole_derivatives)

    def predict_for_molecule( # no specific treatment to atomic energies currently
        self,
        molecule=None,
        calculate_energy=True,
        calculate_energy_gradients=False,
        calculate_hessian=False,
        calculate_dipole_derivatives=False,
    ):

        supported_elements = self._supported_element_symbols()
        from . import data
        unsupported = sorted({
            data.atomic_number2element_symbol.get(atom.atomic_number, str(atom.atomic_number))
            for atom in molecule.atoms
        } - set(supported_elements))
        if unsupported:
            errmsg = (
                ' * Error * Molecule contains elements outside the set this AIQM3 model was '
                'trained on: %s. Supported elements: %s. No calculations performed.\n'
                ' To treat these elements, use a model that covers them:\n'
                '   - the Aitomic add-ons may contain models trained on more elements\n'
                '     (for example AIQM4@DFT);\n'
                '   - AIQM3@UAIQM, the UAIQM extension of this model, can be used out of\n'
                '     the box for these elements; only elements that GFN2-xTB cannot\n'
                '     describe are out of reach.\n'
                ' The add-ons, including the UAIQM models, are installed separately; see\n'
                ' http://MLatom.com/aitomic. Choose the model version explicitly\n'
                ' (\'newest\' is the default); e.g.:\n'
                '   Python: ml.models.methods(method=\'uaiqm_gfn2xtbstar@cc\', version=\'20250115\')\n'
                '   CLI:    mlatom sp uaiqm_gfn2xtbstar@cc uversion=20250115 xyzfile=mol.xyz'
                % (', '.join(unsupported), ', '.join(supported_elements)))
            raise ValueError(errmsg)

        self.aiqm3_model.predict(
            molecule=molecule,
            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian,
            calculate_dipole_derivatives=calculate_dipole_derivatives,
        )

        molecule.__dict__[f'{self.model_name}_nn'].standard_deviation(properties=['energy'])

    def _supported_element_symbols(self):
        """The elements this model can treat: its weights' species, or the published set."""
        symbols = getattr(self, 'element_symbols_available', None)
        if symbols:
            return list(symbols)
        return ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']

    def _load_pretrained_models(self):

        model_paths = self.get_model_paths()

        self.reload_baseline()
        self.reload_dispersion()

        self.nn = model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn{self.model_index[ii]}',
                    model=ani(
                        model_file=model_paths[self.model_index[ii]],
                        device=self.device, verbose=0),
                    operator='predict'
                ) for ii in range(len(self.model_index))
            ],
            operator='average'
        )
        self.element_symbols_available = self.nn.children[0].model.species_order

        self.aiqm3_model = model_tree_node(
            name=self.model_name,
            children=[self.baseline, self.nn, self.d3] if self.d3 is not None else [self.baseline, self.nn],
            operator='sum'
        )

    def get_model_paths(self):
        if self.model_name.lower() == 'aiqm3':
            download_links = [
                'https://zenodo.org/records/15876694/files/aiqm3_cc_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_gfn2xtbstar_cc_20250115.zip']
            model_dir = 'aiqm3_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]

        elif self.model_name.lower() in ['aiqm3atdft', 'aiqm3atdft*']:
            download_links = [
                'https://zenodo.org/records/15876694/files/aiqm3_dft_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_gfn2xtbstar_dft_20240619.zip']
            model_dir = 'aiqm3_dft_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]
        else:
            raise ValueError(f'Unsupported AIQM3 method: {self.model_name}')

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        if to_download: self.download(download_links, mlatom_model_dir)

        model_paths = [os.path.join(mlatom_model_dir, model_files[ii]) for ii in range(8)]
        return model_paths

    # ------------------------------------------------------------------
    # Transfer-learning API
    # ------------------------------------------------------------------
    def _prepare_delta_database(
        self,
        target_db,
        property_to_learn,
        xyz_derivative_property_to_learn=None,
        verbose=0,
    ):
        from . import data

        delta_db = data.molecular_database()
        for mol in target_db.molecules:
            delta_mol = mol.copy(atomic_labels=['xyz_coordinates'])
            delta_mol.__dict__['target_energy'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn:
                delta_mol.__dict__['target_grad'] = mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            delta_db.append(delta_mol)

        self.baseline.predict(
            molecular_database=delta_db,
            calculate_energy=True,
            calculate_energy_gradients=bool(xyz_derivative_property_to_learn),
        )

        for mol in delta_db.molecules:
            baseline_energy = mol.gfn2xtbstar.energy
            mol.delta_energy = mol.target_energy - baseline_energy
            if xyz_derivative_property_to_learn:
                baseline_grad = mol.gfn2xtbstar.energy_gradients
                delta_grad = mol.target_grad - baseline_grad
                mol.add_xyz_vectorial_property(delta_grad, 'delta_energy_gradients')

        return delta_db

    def train(self, **kwargs):

        # default settings
        kwargs['save_model'] = True
        kwargs['reset_parameters'] = False
        kwargs['reset_network'] = True     # rebuild around them; the data may add elements
        kwargs['reset_aev'] = True
        kwargs['reset_optimizer'] = True
        kwargs['reset_energy_shifter'] = True

        file_to_save_model = kwargs.pop('file_to_save_model', None)
        verbose = kwargs.get('verbose', 0)

        # The dispersion term named here is subtracted from the reference labels
        # AND added back at prediction - the same term on both sides.  No default
        # where the model has a native term; the declaration comes from
        # dispersion_kwargs= or from a prepared delta_db='s provenance stamp.
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
            baseline_method='GFN2-xTB*',
            baseline_kwargs=self.baseline_kwargs,
            property_to_learn=property_to_learn,
            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
            working_directory=self.working_directory,
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

        if 'hyperparameters' in kwargs:
            _hyperparameters = kwargs['hyperparameters']
        else:
            _hyperparameters = {}

        # default hyperparameters
        if 'fixed_layers' not in _hyperparameters:
            _hyperparameters['fixed_layers'] = [[0,4]]
        if 'loss_type' not in _hyperparameters:
            _hyperparameters['loss_type'] = 'geometric'
        if 'max_epochs' not in _hyperparameters:
            _hyperparameters['max_epochs'] = 100

        # Remove loaded model
        del self.nn
        del self.aiqm3_model

        # Load pretrained models
        model_paths = self.get_model_paths()
        pretrained_models = [ani(model_file=model_paths[self.model_index[ii]], device=self.device, verbose=verbose) for ii in range(len(self.model_index))]

        # reload aev parameters
        for aev_param in ['Rcr','Rca','EtaR','ShfR','Zeta','ShfZ','EtaA','ShfA']:
            if aev_param not in _hyperparameters:
                try:
                    _hyperparameters[aev_param] = pretrained_models[0].aev_computer._buffers[aev_param].reshape(-1,)
                except:
                    _hyperparameters[aev_param] = pretrained_models[0].aev_computer.__dict__[aev_param]
        kwargs['hyperparameters'] = _hyperparameters

        retrained_models = []
        modelname = self.model_name
        save_dir = file_to_save_model if file_to_save_model else f'{modelname}_retrained'
        os.makedirs(save_dir, exist_ok=True)

        for ii, pmodel in enumerate(pretrained_models):
            print(f'\nStart retraining on model {self.model_index[ii]}...')
            sys.stdout.flush()
            kwargs['file_to_save_model'] = os.path.join(save_dir, f'cv{self.model_index[ii]}.pt')
            pmodel.element_symbols_available = self.element_symbols_available
            pmodel.train(**kwargs)
            retrained_models.append(pmodel)

        self.nn = model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn{self.model_index[ii]}',
                    model=pretrained_models[ii],
                    operator='predict'
                ) for ii in range(len(self.model_index))
            ],
            operator='average'
        )

        self.reload_baseline()
        self.reload_dispersion()
        self.reload()
        self.element_symbols_available = retrained_models[0].species_order

        self._training_databases = delta_learning.write_training_database(
            self._training_database, save_dir,
            delta_source=self._training_delta_source, verbose=verbose)
        self.energy_expression = delta_learning.report_model_composition(
            baseline='GFN2-xTB*', neural_network='dNN',
            dispersion_kwargs=self.dispersion_kwargs, verbose=verbose)

        # Leave a directory that can be loaded back, rather than one holding
        # weights that only become a model after a separate save() call.
        self.save(save_dir)

    def _iter_ani_leaves(self, node=None):
        """
        Iterate over the ``ani`` leaf models inside the AIQM3 model tree.
        """
        if node is None:
            node = self.aiqm3_model
        if node.model is not None and isinstance(node.model, ani):
            try:
                cv_index = int(node.name.split('nn')[-1])
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
            aiqm3._resolve_model_paths(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm3._resolve_model_paths(model, base_dir)

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
            aiqm3._make_paths_relative(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm3._make_paths_relative(model, base_dir)

    def save(self, model_file: str = '') -> None:
        """Save the transfer-learned AIQM3 model to a directory."""
        if not model_file:
            model_file = f'{self.model_name}_tl_model'
        os.makedirs(model_file, exist_ok=True)
        base_dir = os.path.abspath(model_file)

        # Save each ani leaf as cv{i}.pt
        for node, cv_index in self._iter_ani_leaves():
            pt_path = os.path.join(base_dir, f'cv{cv_index}.pt')
            node.model.save(pt_path)
            node.model.model_file = pt_path

        # Build a model tree dict with relative paths
        model_tree_dict = self.aiqm3_model.dump(format='dict')
        self._make_paths_relative(model_tree_dict, base_dir)

        tree = {
            'type': 'aiqm3',
            'module': {
                'name': self.__module__,
                'path': sys.modules[self.__module__].__spec__.origin,
            },
            'tl': True,
            'method': self.method,
            'model_index': self.model_index,
            'device': str(self.device) if hasattr(self, 'device') else 'cpu',
            'element_symbols_available': self.element_symbols_available,
            'baseline_kwargs': self.baseline_kwargs,
            'dispersion_kwargs': self.dispersion_kwargs,
            'energy_expression': getattr(self, 'energy_expression', None),
            'training_labels': getattr(self, '_training_databases', None),
            'provenance': getattr(self, '_training_stamp', None),
            'working_directory': self.working_directory if hasattr(self, '_working_directory') else None,
            'model_tree': model_tree_dict,
        }
        tree_path = os.path.join(base_dir, 'tree.json')
        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=4)
        if self.verbose:
            print(f'AIQM3 TL model saved in {base_dir}')

    @classmethod
    def load(cls, model_file: str, device=None):
        """Load a transfer-learned AIQM3 model from a directory or ``tree.json``."""
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
        """Reconstruct an ``aiqm3`` instance from a dictionary produced by :meth:`save`."""
        import torch
        instance = cls.__new__(cls)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        instance.device = torch.device(device)
        instance.method = model_dict['method']
        instance.model_name = instance.method.lower().replace('@', 'at')
        instance.model_index = model_dict.get('model_index', [ii for ii in range(8)])
        instance._tl = True
        instance.verbose = 1
        instance.element_symbols_available = model_dict.get('element_symbols_available', [])
        instance.baseline_kwargs = dict(model_dict.get('baseline_kwargs', {}))
        instance.dispersion_kwargs = model_dict.get('dispersion_kwargs', None)
        instance._working_directory = None
        if model_dict.get('working_directory') is not None:
            instance._working_directory = model_dict['working_directory']
            instance.baseline_kwargs['working_directory'] = os.path.join(os.path.abspath(instance._working_directory), '_baseline')
            instance.baseline_kwargs['working_directory'] = os.path.join(
                os.path.abspath(instance._working_directory), '_baseline')
        if base_dir is not None:
            cls._resolve_model_paths(model_dict['model_tree'], base_dir)
        for child in model_dict['model_tree']['children'][1]['children']:
            child['model']['kwargs']['device'] = str(instance.device)
        from .models import load_dict
        instance.aiqm3_model = load_dict(model_dict['model_tree'], base_dir=base_dir)
        instance.baseline = instance.aiqm3_model.children[0]
        instance.nn = instance.aiqm3_model.children[1]
        instance.energy_expression = model_dict.get('energy_expression', None)
        instance._training_databases = model_dict.get('training_labels', None)
        instance._training_stamp = model_dict.get('provenance', None)
        # Find the dispersion node by name, not by position.
        instance.d3 = None
        for child in instance.aiqm3_model.children:
            if getattr(child, 'name', '') == 'dispersion':
                instance.d3 = child
        instance.nthreads = 1
        return instance
