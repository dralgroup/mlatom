import os
from ... import data, models, constants
from ...model_cls import method_model, model_tree_node, downloadable_model

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
        mol = ml.molecule.from_xyz_file(filename='ethanol.xyz')
        # Run AIQM3 calculation
        aiqm3 = ml.methods(method='aiqm3')
        aiqm3.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and uncertainty of AIQM3 
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian
        std = mol.aiqm3_model_nn.energy_standard_deviation

    """ 
    
    supported_methods = ['AIQM3', 'AIQM3@DFT']

    def __new__(cls, *args, **kwargs):
        # When the aitomic_addons package is installed it provides the real
        # AIQM3; resolve to it here so this stub transparently becomes the real
        # implementation. Imported at construction time, not at module load:
        # the add-on imports mlatom, so a load-time import here would be circular.
        if cls is aiqm3:
            try:
                from aitomic_addons import aiqm3 as _aiqm3
            except ImportError:
                _aiqm3 = None
            if _aiqm3 is not None:
                return _aiqm3(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        method: str = 'AIQM3',
        working_directory: str = None,
        baseline_kwargs: dict = None,
        dispersion_kwargs: dict = None,
        nthreads: int = 1
    ):

        if self.is_method_supported(method=method):
            raise ValueError(f'The requested method "{method}" is provided by the Aitomic Add-Ons for MLatom. Install them with:  pip install aitomic-addons  (more information: https://aitomistic.com/mlatom/addons.html).')
        else:
            self.raise_unsupported_method_error(method)

