from . import data, models, constants
from multiprocessing import cpu_count
from .interfaces.torchani_interface import ani
import os 

class aiqm2(models.model, metaclass=models.meta_method):

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
        aiqm2 = ml.models.methods(method='aiqm2')
        aiqm2.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and uncertainty of AIQM2 
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian
        std = mol.aiqm2_model.energy_standard_deviation

    """ 

    mlatomdir=os.path.dirname(__file__)
    dirname = os.path.join(mlatomdir, 'aiqm2_model')

    def __init__(
        self,
        method: str = 'AIQM2',
        working_directory: str = '.',
        qm_program_kwargs: dict = {}
    ):

        self.method = method.lower()
        self.model_name = self.method.replace('@','at')
        self.working_directory = working_directory
        self.qm_program_kwargs = qm_program_kwargs
        self.load()

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

        for atom in molecule.atoms:
            if not atom.atomic_number in [1, 6, 7, 8]:
                errmsg = ' * Warning * Molecule contains elements other than CHNO, no calculations performed'
                raise ValueError(errmsg)

        self.aiqm2_model.predict(
            molecule=molecule,
            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, 
            calculate_dipole_derivatives=calculate_dipole_derivatives,
        )

        molecule.__dict__[f'{self.model_name}_nn'].standard_deviation(properties=['energy'])

    def load(self):
        if not os.path.exists(os.path.join(self.dirname, f'{self.model_name}_cv0.pt')):
            self.download_models()
        
        model_paths = [os.path.join(self.dirname, f'{self.model_name}_cv{ii}.pt') for ii in range(8)]

        baseline = models.model_tree_node(
            name='gfn2xtbstar',
            model=models.methods(method='GFN2-xTB*', **self.qm_program_kwargs),
            operator='predict'
        )
        d4 = models.model_tree_node(
            name='d4wb97x',
            model=models.methods(method='D4', functional='wb97x', working_directory=self.working_directory),
            operator='predict'
        )
        nn = models.model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                models.model_tree_node(
                    name=f'{self.model_name}_nn_{ii}',
                    # model=models.ani(
                    #     model_file=model_paths[ii],
                    #     verbose=0),
                    model=ani_wrapper(
                        model_file=model_paths[ii],
                        verbose=0),
                    operator='predict'
                ) for ii in range(8)
            ],
            operator='average'
        )

        self.aiqm2_model = models.model_tree_node(
            name=self.model_name,
            children=[baseline, nn, d4],
            operator='sum'
        )

    def download_models(self):
        import requests
        urls = [f"https://github.com/dralgroup/mlatom/raw/refs/heads/main/mlatom/aiqm2_model/{self.model_name}_cv{ii}.pt" for ii in range(8)]
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname, exist_ok=True)
        print(f'Downloading aiqm2 model parameters ...')
        for ii in range(8):
            resource_res = requests.get(urls[ii])
            with open(f'{self.dirname}/{self.model_name}_cv{ii}.pt','wb') as f:
                f.write(resource_res.content)

class ani_wrapper(ani):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def predict(self,**kwargs):
        if 'calculate_dipole_derivatives' in kwargs.keys():
            del kwargs['calculate_dipole_derivatives']
        super().predict(**kwargs)
