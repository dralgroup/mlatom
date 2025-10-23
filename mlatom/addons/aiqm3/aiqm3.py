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

    def __init__(
        self,
        method: str = 'AIQM3',
        working_directory: str = None,
        baseline_kwargs: dict = None,
        dispersion_kwargs: dict = None,
        nthreads: int = 1
    ):

        if self.is_method_supported(method=method):
            raise ValueError(f'The requested method "{method}" is supported as an add-on in Aitomic, please refer to http://MLatom.com/aitomic for the instructions on how to obtain it.')
        else:
            self.raise_unsupported_method_error(method)

