'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Add-on mock implementation for:                                           !
  !                                                                           !
  ! A model behind OMNI-P2x                                                   !
  ! Implemented by: Mikolaj Martyka                                           ! 
  ! Mocked up by Pavlo O. Dral                                                !
  !---------------------------------------------------------------------------! 
'''

from typing import Any, Union, Dict
from ... import model_cls
from ...model_cls import ml_model, torchani_model
class vecmsani(ml_model, torchani_model):
    '''
    Create an `MS-ANI', an extension of the ANI NN model for multi-state learning. First described in <10.26434/chemrxiv-2024-dtc1w>. 
    
    Interfaces to `TorchANI <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        device (str, optional): Indicate which device the calculation will be run on. i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''

    def __init__(self, model_file: str = None, device: str = None, hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters]={}, verbose=1, nstates=1,validate_train=True):
        
        raise ValueError(f'The requested model type is supported as an add-on in Aitomic, please refer to http://MLatom.com/aitomic for the instructions on how to obtain it.')
    
if __name__ == '__main__':
    pass
