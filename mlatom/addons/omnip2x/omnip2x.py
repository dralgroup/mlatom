#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Add-on mock implementation for:                                           !
  !                                                                           !
  ! omnip2x: OMNI-P2x: A Universal Neural Network Potential for               !
  !          Excited-State Simulations                                        !
  ! Implementations by: Mikolaj Martyka and Pavlo O. Dral                     ! 
  !---------------------------------------------------------------------------! 
'''

from ...model_cls import method_model

class omnip2x(method_model):
    supported_methods = ['omni-p2x','omnip2x' ]
    
    def __init__(self,         
                 method: str = 'OMNI-P2x',
                working_directory: str = '.'
                ):
        
        if self.is_method_supported(method=method):
            raise ValueError(f'The requested method "{method}" is supported as an add-on in Aitomic, please refer to http://MLatom.com/aitomic for the instructions on how to obtain it.')
        else:
            self.raise_unsupported_method_error(method)

if __name__ == '__main__':
    pass
                
        
        
            
        



        