import os
from . import data, models, constants
from .model_cls import method_model, model_tree_node, downloadable_model

class aiqm3(method_model, downloadable_model):

    """ 
    GFN2-xTB based artificial intelligence quantum-mechanical method 3 (AIQM3)

    Arguments:

        method (str, optional): Currently supports AIQM3, AIQM3@DFT, AIQM3@DFT*
        working_directory (str, optional): The path to save temporary calculation file. By default, current directory will be choosed.
        baseline_kwargs (dict, optional): Keywords passed to GFN2-xTB
        dispersion_kwargs (dict, optional): 

    .. code-block:: 

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run AIQM2 calculation
        aiqm3 = ml.methods(method='aiqm3')
        aiqm3.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and uncertainty of AIQM2 
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian
        std = mol.aiqm3_model_nn.energy_standard_deviation

    """ 
    
    supported_methods = ['AIQM3', 'AIQM3@DFT', 'AIQM3@DFT*']

    def __init__(
        self,
        method: str = 'AIQM3',
        working_directory: str = None,
        baseline_kwargs: dict = None,
        dispersion_kwargs: dict = None,
        nthreads: int = 1
    ):

        self.method = method.lower()
        self.model_name = self.method.replace('@','at')
        self.working_directory = working_directory
        if baseline_kwargs is None: self.baseline_kwargs = {}
        else: self.baseline_kwargs = baseline_kwargs
        if dispersion_kwargs is None: self.dispersion_kwargs = {}
        else: self.dispersion_kwargs = dispersion_kwargs
        
        self.load()
        self.nthreads = nthreads

    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.aiqm3_model.nthreads = self._nthreads

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
            if not atom.atomic_number in [1, 6, 7, 8, 9, 16, 17 ]:
                errmsg = ' * Warning * Molecule contains elements other than H, C, N, O, F, S, Cl, no calculations performed'
                raise ValueError(errmsg)

        self.aiqm3_model.predict(
            molecule=molecule,
            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, 
            calculate_dipole_derivatives=calculate_dipole_derivatives,
        )

        molecule.__dict__[f'{self.model_name}_nn'].standard_deviation(properties=['energy'])

    def load(self):
        from .models import methods

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

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        if to_download: self.download(download_links, mlatom_model_dir)
        
        model_paths = [os.path.join(mlatom_model_dir, model_files[ii]) for ii in range(8)]

        baseline = model_tree_node(
            name='gfn2xtbstar',
            model=methods(method='GFN2-xTB*', working_directory=self.working_directory, **self.baseline_kwargs),
            operator='predict'
        )
        d3 = model_tree_node(
            name='d3b973c',
            model=methods(method='d3bj', functional='b973c', working_directory=self.working_directory, **self.dispersion_kwargs),
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
                    model=ani_wrapper(
                        model_file=model_paths[ii],
                        verbose=0),
                    operator='predict'
                ) for ii in range(8)
            ],
            operator='average'
        )

        self.aiqm3_model = model_tree_node(
            name=self.model_name,
            children=[baseline, nn, d3],
            operator='sum'
        )



