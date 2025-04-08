'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Module with parts of torchani_interface which take long time to load      !
  ! Copied and adapted from torchani_interface.py by Pavlo O. Dral            !
  ! Implemented by: Mikolaj Martyka                                           ! 
  !---------------------------------------------------------------------------! 
'''


import torch

class StateInputNet(torch.nn.Module):
    def __init__(self, AEV_comp, net):
        super(StateInputNet,self).__init__()
        self.AEV_computer = AEV_comp
        self.network = net
    def forward(self, species, coordinates, state):
        species_, aev = self.AEV_computer((species,coordinates))
        state_tensor = torch.transpose(state.expand(species.size()[1],-1),0,1)  
        aev_with_state = torch.cat((aev,state_tensor.unsqueeze(2)),2)
        out = self.network((species_, aev_with_state))
        return out
