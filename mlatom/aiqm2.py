import os
from . import data, models, constants
from .model_cls import method_model, model_tree_node, downloadable_model

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
        std = mol.aiqm2_model.energy_standard_deviation

    """ 
    
    supported_methods = ['AIQM2', 'AIQM2@DFT', 'AIQM2@DFT*']
    # mlatomdir=os.path.dirname(__file__)
    # dirname = os.path.join(mlatomdir, 'aiqm2_model')

    def __init__(
        self,
        method: str = 'AIQM2',
        working_directory: str = '.',
        qm_program_kwargs: dict = {},
        nthreads: int = 1
    ):

        self.method = method.lower()
        self.model_name = self.method.replace('@','at')
        self.working_directory = working_directory
        self.qm_program_kwargs = qm_program_kwargs
        self.load()
        self.nthreads = nthreads

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.aiqm2_model.nthreads = self._nthreads

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
        from .models import methods

        model_name, model_path, download = self.check_model_path(self.method)
        if download: self.download(model_name, model_path)
        
        model_paths = [os.path.join(model_path, f'cv{ii}.pt') for ii in range(8)]

        baseline = model_tree_node(
            name='gfn2xtbstar',
            model=methods(method='GFN2-xTB*', **self.qm_program_kwargs),
            operator='predict'
        )
        d4 = model_tree_node(
            name='d4wb97x',
            model=methods(method='D4', functional='wb97x', working_directory=self.working_directory),
            operator='predict'
        )
        from .interfaces.torchani_interface import ani
        class ani_wrapper(ani):
            def __init__(self,**kwargs):
                super().__init__(**kwargs)

            def predict(self,**kwargs):
                if 'calculate_dipole_derivatives' in kwargs.keys():
                    del kwargs['calculate_dipole_derivatives']
                super().predict(**kwargs)
        nn = model_tree_node(
            name=f'{self.model_name}_nn',
            children=[
                model_tree_node(
                    name=f'{self.model_name}_nn_{ii}',
                    # model=ani(
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

        self.aiqm2_model = model_tree_node(
            name=self.model_name,
            children=[baseline, nn, d4],
            operator='sum'
        )



