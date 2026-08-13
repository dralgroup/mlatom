import os 
import sys
import json
from . import dispersion as dispersion_utils
from . import delta_learning
from .model_cls import method_model, downloadable_model, model_tree_node

class omnip1(method_model, downloadable_model):

    """ 
    The universal interatomic potential capable of simultaneously learning and making predictions at different QC levels
    DOI: https://doi.org/10.1021/acs.jctc.5c00858

    Arguments:

        method (str, optional): Currently only OMNI-P1 is available. Default: `cc`. 
        level (str or int, optional): The label source to predict at - `cc`, `dft`, or the integer the model encodes one as. Default: `cc`.
        working_directory (str, optional): The path to save temporary calculation file. Default: `None`, i.e., nothing will be dumped.
        dispersion_kwargs (dict, optional): The keywords used for D4 dispersion correction. Default: `None`
        nthreads (int, optional): The number of threads to be used. Default: `1`

    .. code-block:: 

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')

        # Run single-point calculation at coupled cluster level
        omnip1 = ml.methods(method='omnip1', level='cc')
        omnip1.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        energy = mol.energy
        gradient = mol.energy_gradients
        hess = mol.hessian

        # Run single-point calculation at dft level
        omnip1 = ml.methods(method='omnip1', level='dft')
        omnip1.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
    """ 

    supported_methods = ['OMNI-P1', 'OMNIP1']

    def __init__(self, method:str='OMNI-P1', level='cc', nthreads:int=1, dispersion_kwargs:dict=None,working_directory:str = None):
        self.method = method.lower().replace('-', '').replace('_', '')
        # 'cc', 'dft', or the integer the model encodes a source as.
        self.level = level.lower() if isinstance(level, str) else level
        # ``None`` -> the model's own D4(wB97X) term.  ``False`` switches it off;
        # ``{}`` raises.  The stored spec identifies the term only; where it runs
        # is decided when the node is built, so passing working_directory can no
        # longer turn dispersion back on by making an empty dict truthy.
        if dispersion_kwargs is None:
            self.dispersion_kwargs = dispersion_utils.native_dispersion_kwargs('omni-p1')
        else:
            self.dispersion_kwargs = dispersion_utils.normalize(dispersion_kwargs)
        self._working_directory = None
        if working_directory is not None:
            self.set_working_directory(working_directory)
        self.load_nn()
        self.reload_dispersion()
        self.reload()
        self.nthreads = nthreads

    def load_nn(self):
        from .omnip1_nn import omnip1_NN
        download_links = ['https://aitomistic.xyz/model/omnip1_model.zip']
        model_dir = 'omnip1_model'
        model_file = 'model.pt'

        mlatom_model_dir, to_download = self.check_model_path(model_dir, [model_file])
        if to_download: self.download(download_links, mlatom_model_dir)
        
        model_path = os.path.join(mlatom_model_dir, model_file)
        self.nn = model_tree_node(
            name='omnip1_nn',
            model=omnip1_NN(model_path=model_path, level=self.level),
            operator='predict')

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.omnip1_model.nthreads = self._nthreads
    
    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, value):
        self._working_directory = value
        self.set_working_directory(value)
        self.reload_dispersion()
        self.reload()

    def set_working_directory(self, working_directory):
        self._working_directory = working_directory

    def dispersion_working_directory(self):
        if self._working_directory is None:
            return None
        return os.path.join(os.path.abspath(self._working_directory), "_dispersion")
    
    def reload_dispersion(self):
        self.d4 = dispersion_utils.build_node(
            self.dispersion_kwargs, name='d4wb97x',
            working_directory=self.dispersion_working_directory())

    def reload(self):
        children = [self.nn]
        if self.d4 is not None:
            children.append(self.d4)
        self.omnip1_model = model_tree_node(
            name='omnip1',
            children=children,
            operator='sum'
        )

    def predict(self, 
                molecular_database = None, molecule = None,
                calculate_energy = False, calculate_energy_gradients = False, calculate_hessian = False, **kwargs):
        
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB:
            self.omnip1_model.predict(
                molecule=mol, 
                calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)

    # ------------------------------------------------------------------
    # Transfer-learning API
    # ------------------------------------------------------------------
    def train(
        self,
        molecular_database=None,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database = 'sample_from_molecular_database',
        hyperparameters: dict = None,
        spliting_ratio: float = 0.8,
        save_model: bool = True,
        file_to_save_model: str = None,
        use_last_model: bool = False,
        dispersion_kwargs = 'not given',
        delta_db = None,
        verbose: int = 1,
    ) -> None:
        """
        Fine-tune the OMNI-P1 model on a molecular database.

        ``dispersion_kwargs`` names the dispersion term that is subtracted from
        the reference labels and added back at prediction.  OMNI-P1 has a native
        D4(wB97X) term, so a declaration is required: previously the node was kept
        while nothing was subtracted, and the fine-tuned model returned
        ``E_target + D4(wB97X)`` - a double count.
        """
        if hyperparameters is None:
            hyperparameters = {}
        hyperparameters = dict(hyperparameters)

        # TL defaults
        if 'max_epochs' not in hyperparameters:
            hyperparameters['max_epochs'] = 100
        if 'batch_size' not in hyperparameters:
            hyperparameters['batch_size'] = 8
        if 'fixed_layers' not in hyperparameters:
            hyperparameters['fixed_layers'] = [[0,4]]
        if 'loss_type' not in hyperparameters:
            hyperparameters['loss_type'] = 'weighted'

        # Whatever dispersion is named is subtracted here and added back by the
        # rebuilt tree below, so the two sides cannot disagree.
        resolved = delta_learning.resolve_training_labels(
            model_method='omni-p1',
            molecular_database=molecular_database,
            delta_db=delta_db,
            dispersion_kwargs=dispersion_kwargs,
            baseline_method=None,
            property_to_learn=property_to_learn,
            xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,
            working_directory=self._working_directory,
            verbose=verbose,
        )
        self.dispersion_kwargs = resolved['dispersion_kwargs']
        self._training_database = resolved['database']
        self._training_delta_source = resolved['delta_source']
        self._training_stamp = resolved['stamp']

        self.nn.model.train(
            molecular_database=resolved['database'],
            property_to_learn='delta_energy',
            xyz_derivative_property_to_learn=(
                'delta_energy_gradients' if xyz_derivative_property_to_learn else None),
            validation_molecular_database=validation_molecular_database,
            hyperparameters=hyperparameters,
            spliting_ratio=spliting_ratio,
            save_model=save_model,
            file_to_save_model=file_to_save_model,
            use_last_model=use_last_model,
            reset_energy_shifter=True,
            verbose=verbose,
        )

        # Rebuild the tree with the retrained NN and the dispersion node that
        # matches what was subtracted.
        self.reload_dispersion()
        self.reload()

        save_dir = delta_learning.model_directory(file_to_save_model, 'omnip1_retrained')
        self._training_databases = delta_learning.write_training_database(
            self._training_database, save_dir,
            delta_source=self._training_delta_source, verbose=verbose)
        self.energy_expression = delta_learning.report_model_composition(
            baseline=None, neural_network='NN',
            dispersion_kwargs=self.dispersion_kwargs, verbose=verbose)

        # Leave a directory that can be loaded back, rather than one holding
        # weights that only become a model after a separate save() call.
        self.save(save_dir)

    def save(self, model_file: str = '') -> None:
        """Save the transfer-learned OMNI-P1 model to a directory."""
        if not model_file:
            model_file = 'omnip1_tl_model'
        os.makedirs(model_file, exist_ok=True)
        base_dir = os.path.abspath(model_file)

        pt_path = os.path.join(base_dir, 'model.pt')
        self.nn.model.save(pt_path)

        tree = {
            'type': 'omnip1',
            'module': {
                'name': self.__module__,
                'path': sys.modules[self.__module__].__spec__.origin,
            },
            'tl': True,
            'method': self.method,
            'level': self.level,
            'dispersion_kwargs': self.dispersion_kwargs,
            'energy_expression': getattr(self, 'energy_expression', None),
            'training_labels': getattr(self, '_training_databases', None),
            'provenance': getattr(self, '_training_stamp', None),
            'working_directory': getattr(self, '_working_directory', None),
            'model_file': 'model.pt',
            'nthreads': self.nthreads if hasattr(self, '_nthreads') else 1,
        }
        tree_path = os.path.join(base_dir, 'tree.json')
        with open(tree_path, 'w') as f:
            json.dump(tree, f, indent=4)

    @classmethod
    def load(cls, model_file: str):
        """Load a transfer-learned OMNI-P1 model from a directory or tree.json."""
        if os.path.isdir(model_file):
            tree_file = os.path.join(model_file, 'tree.json')
        else:
            tree_file = model_file
        if not os.path.isfile(tree_file):
            raise FileNotFoundError(f'Cannot find tree.json at {tree_file}')

        with open(tree_file) as f:
            model_dict = json.load(f)
        base_dir = os.path.dirname(os.path.abspath(tree_file))
        return cls.from_dict(model_dict, base_dir=base_dir)

    @classmethod
    def from_dict(cls, model_dict, base_dir=None):
        """Reconstruct an ``omnip1`` instance from a dictionary produced by :meth:`save`."""
        instance = cls.__new__(cls)
        instance.method = model_dict.get('method', 'omnip1')
        instance.level = model_dict.get('level', 'cc')
        disp = model_dict.get('dispersion_kwargs', None)
        instance.dispersion_kwargs = None if disp is None else dict(disp)
        instance.energy_expression = model_dict.get('energy_expression', None)
        instance._training_databases = model_dict.get('training_labels', None)
        instance._training_stamp = model_dict.get('provenance', None)
        instance._working_directory = None
        if model_dict.get('working_directory') is not None:
            instance._working_directory = model_dict['working_directory']
            instance.set_working_directory(instance._working_directory)

        model_file = model_dict.get('model_file', 'model.pt')
        if base_dir is not None and not os.path.isabs(model_file):
            model_file = os.path.normpath(os.path.join(base_dir, model_file))

        from .omnip1_nn import omnip1_NN
        instance.nn = model_tree_node(
            name='omnip1_nn',
            model=omnip1_NN(model_path=model_file, level=instance.level),
            operator='predict')

        instance.reload_dispersion()
        instance.reload()
        instance.nthreads = model_dict.get('nthreads', 1)
        return instance
