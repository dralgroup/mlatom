import os
import sys
import numpy as np
from . import data, models, constants
from . import dispersion as dispersion_utils
from . import delta_learning
from .model_cls import method_model, model_tree_node, downloadable_model
from .interfaces.torchani_interface import ani

class aiqm2(method_model, downloadable_model):

    """ 
    GFN2-xTB based artificial intelligence quantum-mechanical method 2 (AIQM2)

    Arguments:

        method (str, optional): Currently supports AIQM2, AIQM2@DFT
        working_directory (str, optional): The path to save temporary calculation file
        qm_program_kwargs (dict, optional): Keywords passed to GFN2-xTB

    .. code-block:: 

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run AIQM2 calculation
        aiqm2 = ml.methods(method='aiqm2')
        aiqm2.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and uncertainty of AIQM2 
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian
        std = mol.aiqm2_nn.energy_standard_deviation

    """ 
    
    supported_methods = ['AIQM2', 'AIQM2@DFT', 'AIQM2@DFT*']
    _tl = True
    verbose = 1

    def __init__(
        self,
        method: str = 'AIQM2',
        working_directory: str = None,
        baseline_kwargs: dict = None,
        dispersion_kwargs: dict = None,
        nthreads: int = 1,
        model_index = None,
        device = None,
    ):
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.method = method.lower()
        self.model_name = self.method.replace('@','at')

        if model_index is None:
            self.model_index = [ii for ii in range(8)]
        elif isinstance(model_index,list):
            self.model_index = model_index
        elif isinstance(model_index,int):
            self.model_index = [model_index]
        else:
            raise ValueError(f"Unrecognized model_index type: {type(model_index)}. Please provide int, list or None.")

        if baseline_kwargs is None: self.baseline_kwargs = {}
        else: self.baseline_kwargs = baseline_kwargs
        # ``None`` -> the model's own dispersion term, i.e. D4(wB97X) for AIQM2
        # and none for AIQM2@DFT*.  ``False`` switches it off; ``{}`` raises.
        # The stored spec identifies the term only; where it runs is decided when
        # the node is built, so an unrelated argument can never turn it on or off.
        if dispersion_kwargs is None:
            self.dispersion_kwargs = dispersion_utils.native_dispersion_kwargs(self.method)
        else:
            self.dispersion_kwargs = dispersion_utils.normalize(dispersion_kwargs)
        self._working_directory = None
        if working_directory is not None:
            self.set_working_directory(working_directory)
        self.load_pretrained_model()
        self.nthreads = nthreads

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.aiqm2_model.nthreads = self._nthreads # nthreads can be directly assigned to each tree node
    
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
        self.d4 = dispersion_utils.build_node(
            self.dispersion_kwargs, name='d4wb97x',
            working_directory=self.dispersion_working_directory())

    def reload(self):
        children = [self.baseline, self.nn]
        if self.d4 is not None:
            children.append(self.d4)
        self.aiqm2_model = model_tree_node(
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
        calculate_polarizability_derivatives=False,
    ):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol,
                                calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian,
                                calculate_dipole_derivatives=calculate_dipole_derivatives,calculate_polarizability_derivatives=calculate_polarizability_derivatives,)

    def predict_for_molecule( # no specific treatment to atomic energies currently
        self,
        molecule=None,
        calculate_energy=True, 
        calculate_energy_gradients=False, 
        calculate_hessian=False,
        calculate_dipole_derivatives=False,
        calculate_polarizability_derivatives=False,
    ):

        for element_symbol in np.unique(molecule.element_symbols):
            if element_symbol not in self.element_symbols_available:
                errmsg = f' * Warning * Molecule contains elements \'{element_symbol}\' other than {self.element_symbols_available}, no calculations performed'
                raise ValueError(errmsg)

        self.aiqm2_model.predict(
            molecule=molecule,
            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, 
            calculate_dipole_derivatives=calculate_dipole_derivatives,calculate_polarizability_derivatives=calculate_polarizability_derivatives,
        )

        molecule.__dict__[f'{self.model_name}_nn'].standard_deviation(properties=['energy'])

    def load_pretrained_model(self):

        # models = self.load_model_part(self.model_index)
        

        model_paths = self.get_model_paths()

        self.reload_baseline()
        self.reload_dispersion()

        from .interfaces.torchani_interface import ani
        # use module-level wrapper so saved trees can be reloaded
        ani_wrapper = ani
        self.nn = model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn{self.model_index[ii]}',
                    # model=ani(
                    #     model_file=model_paths[ii],
                    #     verbose=0),
                    model=ani_wrapper(
                        model_file=model_paths[self.model_index[ii]],
                        verbose=0),
                    operator='predict'
                ) for ii in range(len(self.model_index))
            ],
            operator='average'
        )
        self.element_symbols_available = self.nn.children[0].model.species_order

        self.reload()

    def load_model_part(self,model_index):
        ani_wrapper = ani
        model_name, model_path, download = self.check_model_path(self.method)
        if download: self.download(model_name, model_path)
        
        model_paths = [os.path.join(model_path, f'cv{ii}.pt') for ii in model_index]
        
        models = [ani_wrapper(model_file=each,verbose=0,device=self.device) for each in model_paths]

        return models
    
    def get_model_paths(self):
        if self.model_name.lower() == 'aiqm2':
            download_links = [
                'https://zenodo.org/records/15383333/files/aiqm2_cc_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_gfn2xtbstar_cc_20240106.zip']
            model_dir = 'aiqm2_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]
        elif self.model_name.lower() in ['aiqm2atdft', 'aiqm2atdft*']:
            download_links = [
                'https://zenodo.org/records/15383333/files/aiqm2_dft_model.zip?download=1',
                'https://aitomistic.xyz/model/uaiqm_gfn2xtbstar_dft_20240106.zip']
            model_dir = 'aiqm2_dft_model'
            model_files = [f'cv{ii}.pt' for ii in range(8)]

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        if to_download: self.download(download_links, mlatom_model_dir)
        
        model_paths = [os.path.join(mlatom_model_dir, model_files[ii]) for ii in range(8)]
        return model_paths

    def train(self,**kwargs):

        # default settings 
        kwargs['save_model'] = True # force saving model
        kwargs['reset_energy_shifter'] = True # force resetting energy shifter for backup old energy shifter
        kwargs['reset_parameters'] = False # force keeping parameters
        kwargs['reset_network'] = True     # rebuild around them; the data may add elements
        kwargs['reset_aev'] = True
        kwargs['reset_optimizer'] = True

        if 'file_to_save_model' in kwargs:
            file_to_save_model = kwargs['file_to_save_model']
        else:
            file_to_save_model = None

        if 'verbose' not in kwargs:
            verbose = 0
        else:
            verbose = kwargs['verbose']

        # The dispersion term named here is subtracted from the reference labels
        # AND added back at prediction - the same term on both sides.  There is no
        # default where the model has a native term: the declaration comes either
        # from dispersion_kwargs= or from the provenance stamp of a prepared
        # delta_db=, never from neither and never from two that disagree.
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
            dispersion_kwargs=kwargs.pop('dispersion_kwargs', 'not given'),
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
            _hyperparameters = kwargs['hyperparameters'].copy()   # leave the caller's dict without the defaults below
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
        del self.aiqm2_model

        # Load pretrained models
        # pretrained_models = self.load_model_part(self.model_index)
        model_paths = self.get_model_paths()
        from .interfaces.torchani_interface import ani
        # use module-level wrapper so saved trees can be reloaded
        ani_wrapper = ani
        pretrained_models = [ani_wrapper(model_file=model_paths[self.model_index[ii]],verbose=verbose) for ii in range(len(self.model_index))]

        # reload aev parameters
        for aev_param in ['Rcr','Rca','EtaR','ShfR','Zeta','ShfZ','EtaA','ShfA']:
            if aev_param not in _hyperparameters:
                try: _hyperparameters[aev_param] = pretrained_models[0].aev_computer._buffers[aev_param].reshape(-1,)
                except: _hyperparameters[aev_param] = pretrained_models[0].aev_computer.__dict__[aev_param]
        kwargs['hyperparameters'] = _hyperparameters

        retrained_models = []
        modelname = self.method.lower().replace('-','')
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
            name = f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn{self.model_index[ii]}',
                    model=pretrained_models[ii],
                    operator='predict'
                ) for ii in range(len(self.model_index))
            ],
            operator='average'
        )
        from .models import methods
        # baseline = model_tree_node(
        #     name='gfn2xtbstar',
        #     model=methods(method='GFN2-xTB*', **self.qm_program_kwargs),
        #     operator='predict'
        # )
        # d4 = model_tree_node(
        #     name='d4wb97x',
        #     model=methods(method='D4', functional='wb97x', working_directory=self.working_directory),
        #     operator='predict'
        # )
        self.reload_baseline()
        self.reload_dispersion()
        self.reload()
        # Update available elements
        self.element_symbols_available = retrained_models[0].species_order

        # The labels this model was fitted to are kept next to its weights: the
        # baseline is the expensive part of preparation, and a user who forgets to
        # save it pays for it again on the next run.
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
        Iterate over the ``ani`` leaf models inside the AIQM2 model tree.

        Yields tuples ``(node, cv_index)`` where ``node`` is a
        :class:`model_tree_node` whose ``.model`` is an :class:`ani` instance,
        and ``cv_index`` is parsed from the node name.
        """
        from .interfaces.torchani_interface import ani
        if node is None:
            node = self.aiqm2_model
        if node.model is not None and isinstance(node.model, ani):
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
            aiqm2._resolve_model_paths(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm2._resolve_model_paths(model, base_dir)

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
            aiqm2._make_paths_relative(child, base_dir)
        model = model_dict.get('model')
        if model:
            aiqm2._make_paths_relative(model, base_dir)

    def save(self, model_file: str = '') -> None:
        """
        Save the transfer-learned AIQM2 model to a directory.

        The directory will contain:

        - ``tree.json``: model metadata and tree structure
        - ``cv*.pt``:    TorchANI weights for each ensemble member
        """
        import json
        if not model_file:
            model_file = f'{self.model_name}_tl_model'
        os.makedirs(model_file, exist_ok=True)
        base_dir = os.path.abspath(model_file)

        # Save each ani leaf as cv{i}.pt
        for node, cv_index in self._iter_ani_leaves():
            pt_path = os.path.join(base_dir, f'cv{cv_index}.pt')
            node.model.save(pt_path)

        # Build a model tree dict with relative paths
        model_tree_dict = self.aiqm2_model.dump(format='dict')
        self._make_paths_relative(model_tree_dict, base_dir)

        tree = {
            'type': 'aiqm2',
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
            'working_directory': self.working_directory,
            # What this model is, and what its labels were: readable without
            # loading the model, because every defect in this area is otherwise
            # invisible at the call site.
            'energy_expression': getattr(self, 'energy_expression', None),
            'training_labels': getattr(self, '_training_databases', None),
            'provenance': getattr(self, '_training_stamp', None),
            'model_tree': model_tree_dict,
        }
        tree_path = os.path.join(base_dir, 'tree.json')
        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=4)
        if self.verbose:
            print(f'AIQM2 TL model saved in {base_dir}')

    @classmethod
    def load(cls, model_file: str, device=None):
        """
        Load a transfer-learned AIQM2 model from a directory or ``tree.json``.
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
        """Reconstruct an ``aiqm2`` instance from a dictionary produced by :meth:`save`."""
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
        instance.energy_expression = model_dict.get('energy_expression', None)
        instance._training_databases = model_dict.get('training_labels', None)
        instance._training_stamp = model_dict.get('provenance', None)
        # Set the private attribute rather than the property: the property setter
        # rebuilds the tree, which needs self.nn - assigned only further down.
        instance._working_directory = model_dict.get('working_directory', None)
        if instance._working_directory is not None:
            instance.baseline_kwargs['working_directory'] = os.path.join(
                os.path.abspath(instance._working_directory), '_baseline')
        if base_dir is not None:
            cls._resolve_model_paths(model_dict['model_tree'], base_dir)
        from .models import load_dict
        instance.aiqm2_model = load_dict(model_dict['model_tree'], base_dir=base_dir)
        instance.baseline = instance.aiqm2_model.children[0]
        instance.nn = instance.aiqm2_model.children[1]
        # Find the dispersion node by name, not by position.
        instance.d4 = None
        for child in instance.aiqm2_model.children:
            if getattr(child, 'name', '') == 'dispersion':
                instance.d4 = child
        instance.nthreads = 1
        return instance
