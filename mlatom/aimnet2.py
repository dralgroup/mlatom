'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! aimnet2: Universal AIMnet2 models                                         ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''
from .model_cls import torchani_model, method_model
from .decorators import doc_inherit

class aimnet2_methods(torchani_model, method_model):
    '''
    Universal ML methods with AIMNet2: https://doi.org/10.26434/chemrxiv-2023-296ch

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'AIMNet2@b973c'`` and ``'AIMNet2@wb97m-d3'``.
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.

    '''
    
    supported_methods = ['AIMNet2@b973c', 'AIMNet2@wb97m-d3']
    element_symbols_available = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']

    def __init__(self, method: str = 'AIMNet2@b973c', device = None):
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.model_path = self.parse_aimnet2_resources(method)
        self.model = torch.jit.load(self.model_path)
        
    @doc_inherit
    def predict(
            self, 
            molecular_database = None, 
            molecule = None,
            calculate_energy: bool = False,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
            batch_size: int = 2**16,
        ) -> None:
        import numpy as np

        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for element_symbol in np.unique(np.concatenate(molDB.element_symbols)):
            if element_symbol not in self.element_symbols_available:
                print(f' * Warning * Molecule contains elements \'{element_symbol}\', which is not supported by method \'{self.method}\' that only supports {self.element_symbols_available}, no calculations performed')
                return

        for mol in molDB:
            self.predict_for_molecule(
                molecule=mol, 
                calculate_energy=calculate_energy, 
                calculate_energy_gradients=calculate_energy_gradients, 
                calculate_hessian=calculate_hessian)

    def predict_for_molecule(
        self,
        molecule, 
        calculate_energy: bool = False, 
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False):
        
        import torch

        coord = torch.as_tensor(molecule.xyz_coordinates).to(torch.float).to(self.device).unsqueeze(0)
        numbers = torch.as_tensor(molecule.atomic_numbers).to(torch.long).to(self.device).unsqueeze(0)
        charge = torch.tensor([molecule.charge], dtype=torch.float, device=self.device)
        nninput = dict(coord=coord, numbers=numbers, charge=charge)
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(calculate_energy_gradients)
        if calculate_energy_gradients:
            nninput['coord'].requires_grad_(True)
        nnoutput = self.model(nninput)
        if calculate_energy:
            molecule.energy = nnoutput['energy'].item()
        if calculate_energy_gradients:
            if 'forces' in nnoutput:
                f = nnoutput['forces'][0]
            else:
                f = - torch.autograd.grad(nnoutput['energy'], nninput['coord'])[0][0]
            forces = f.detach().cpu().numpy()
            molecule.add_xyz_vectorial_property(forces, 'energy_gradients')

        if calculate_hessian:
            print('Hessian not available yet')
            molecule.hessian = np.zeros((len(molecule),)*2)
        
        torch._C._set_grad_enabled(prev)

    @staticmethod
    def parse_aimnet2_resources(method):
        import requests, os
        local_dir = os.path.expanduser('~/.local/AIMNet2/')
        repo_name = "AIMNet2"
        tag_name = method.lower().replace('@', '_')
        url = "https://github.com/isayevlab/{}/raw/old/models/{}_ens.jpt".format(repo_name, tag_name)
        if not os.path.exists(local_dir+f'{tag_name}_ens.jpt'):
            os.makedirs(local_dir, exist_ok=True)
            print(f'Downloading {method} model parameters ...')
            resource_res = requests.get(url)
            with open(local_dir+f'{tag_name}_ens.jpt', 'wb') as f:
                f.write(resource_res.content)
        return local_dir + f'{tag_name}_ens.jpt'
