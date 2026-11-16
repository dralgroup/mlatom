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


def _real_uaiqm():
    """The add-on's implementation, or None when it is not installed."""
    try:
        from aitomic_addons import uaiqm as _uaiqm
    except ImportError:
        return None
    return _uaiqm


class _uaiqm_meta(type):
    """Forward class-level attributes to the add-on implementation.

    Construction is redirected in ``__new__``; a classmethod such as
    ``uaiqm.load(...)`` never constructs, so it needs this forwarder.
    """

    def __getattr__(cls, name):
        real = _real_uaiqm()
        if real is not None and hasattr(real, name):
            return getattr(real, name)
        raise AttributeError(
            "'%s' is provided by the Aitomic Add-Ons for MLatom, which are not "
            "installed. Install them with:  pip install aitomic-addons"
            % name)


class uaiqm(method_model, metaclass=_uaiqm_meta):
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
            raise ValueError(f'The requested method "{method}" is provided by the Aitomic Add-Ons for MLatom. Install them with:  pip install aitomic-addons  (more information: https://aitomistic.com/mlatom/addons.html).')
        else:
            self.raise_unsupported_method_error(method)

    def __new__(cls, *args, **kwargs):
        # Imported at construction time, not module load: the add-on imports
        # mlatom, so a load-time import here would be circular.
        if cls is uaiqm:
            real = _real_uaiqm()
            if real is not None:
                return real(*args, **kwargs)
        return super().__new__(cls)
    
if __name__ == '__main__':
    pass
