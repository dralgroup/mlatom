#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Add-on mock implementation for:                                           !
  !                                                                           !
  ! uaiqm: Universal and Updatable Artificial Intelligence-Enhanced           !
  !        Quantum Mechanical Methods                                         ! 
  ! Implementations by: Yuxinxin Chen and Pavlo O. Dral                       ! 
  !---------------------------------------------------------------------------! 
'''

from ...model_cls import method_model

class uaiqm(method_model):
    """
    The Universal and Updatable Artificial Intelligence-Quantum Mechanical methods.

    Arguments:
        method (str, optional): UAIQM method used. 
        version (str, optional): The version of each UAIQM method. Default is the the newest.
        uaiqm_kwargs (dictionary, optional): Keywords passed to calculation for each part in uaiqm.
        selector (uaiqm selector, optional): If no customized selector is provided, default selection scheme will be used.
        verbose (bool, optional): Whether to print information during automatic selection
    
    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run UAIQM calculation
        uaiqm = ml.models.methods(method='uaiqm_odm2star@cc', version='newest')
        uaiqm.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        # Get energy, gradient, and prediction uncertainty of UAIQM method 
        energy = mol.energy
        gradient = mol.gradient

    """
    
    supported_methods = ['uaiqm_optimal', 'uaiqm_nobaseline@dft', 'uaiqm_nobaseline@cc', 'uaiqm_odm2star@dft', 'uaiqm_odm2star@cc', 'uaiqm_gfn2xtbstar@dft', 'uaiqm_gfn2xtbstar@cc', 'uaiqm_wb97x631gp@cc', 'uaiqm_wb97xdef2tzvpp@cc']
    
    def __init__(
        self, 
        # <<< define methods <<<
        method: str = 'uaiqm_gfn2xtbstar@cc', 
        version: str = 'newest',
        # <<< solvent - only support xtb based method now <<< 
        solvent: str = None,
        # <<< keywords for each componnet <<<
        baseline_kwargs: dict = {},
        dispersion_kwargs: dict = {},
        # <<< set number of threads (default 1)<<<
        nthreads: int = 1,
        # <<< unused keyword <<<
        selector = None,
        # <<< verbose setting <<<
        verbose: bool = False,
        warning: bool = True,
        # <<< file saving <<<
        working_directory: str = None,
        save_files_in_current_directory: bool = True
        ):
        
        if self.is_method_supported(method=method):
            raise ValueError(f'The requested method "{method}" is supported as an add-on in Aitomic, please refer to http://MLatom.com/aitomic for the instructions on how to obtain it.')
        else:
            self.raise_unsupported_method_error(method)
    
if __name__ == '__main__':
    pass
