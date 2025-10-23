import os 
from .model_cls import method_model, downloadable_model, model_tree_node

class omnip1(method_model, downloadable_model):

    """ 
    The universal interatomic potential capable of simultaneously learning and making predictions at different QC levels
    DOI: https://doi.org/10.1021/acs.jctc.5c00858

    Arguments:

        method (str, optional): Currently only OMNI-P1 is available. Default: `cc`. 
        level (str, optional): The level of theory to be predicted. Avaialbe options are `dft` and `cc`. Default: `cc`.
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
        self.level = level.lower()
        if dispersion_kwargs is None: self.dispersion_kwargs = {}
        else: self.dispersion_kwargs = dispersion_kwargs
        if working_directory is not None:
            self.set_working_directory(working_directory)
        self.load_nn()
        self.reload_dispersion()
        self.reload()
        self.nthread = nthreads

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
        self.dispersion_kwargs["working_directory"] = os.path.join(os.path.abspath(working_directory), "_dispersion")
    
    def reload_dispersion(self):
        from .models import methods
        self.d4 = model_tree_node(
            name='d4wb97x',
            model=methods(method='D4', functional='wb97x',**self.dispersion_kwargs),
            operator='predict'
        )
    
    def reload(self):
        self.omnip1_model = model_tree_node(
            name='omnip1',
            children=[self.nn, self.d4],
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
