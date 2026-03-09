'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Module with parts of torchani_interface which take long time to load      !
  ! Copied and adapted from torchani_interface.py by Pavlo O. Dral            !
  ! Implemented by: Mikolaj Martyka                                           ! 
  !---------------------------------------------------------------------------! 
'''


import torch

class ReStateInputNet(torch.nn.Module):
    def __init__(self, AEV_comp, net):
        super(ReStateInputNet,self).__init__()
        self.AEV_computer = AEV_comp
        self.network = net
    def forward(self, species, coordinates, state, descriptor):
        re_vector = descriptor
        species_, aev = self.AEV_computer((species,coordinates))
        state_tensor = torch.transpose(state.expand(species.size()[1],-1),0,1)
        re_vector = re_vector.unsqueeze(2)

        re_tensor = torch.transpose(re_vector.expand(-1,-1,species.size()[1]),1,2)

        aev_with_state = torch.cat((aev,re_tensor, state_tensor.unsqueeze(2)),2)

        out = self.network((species_, aev_with_state))
        return out
