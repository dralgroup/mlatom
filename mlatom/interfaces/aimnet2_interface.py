'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! aimnet2: Universal AIMnet2 models                                         ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''
from ..model_cls import torchani_model, method_model, model_tree_node, downloadable_model
from ..decorators import doc_inherit

class aimnet2_methods(torchani_model, method_model):

    '''
    Universal ML methods with AIMNet2: https://doi.org/10.26434/chemrxiv-2023-296ch. Model files can be downloaded from https://github.com/zubatyuk/aimnet-model-zoo. For installation of AIMNet2 calculator, please refer to https://github.com/isayevlab/AIMNet2.

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'AIMNet2@b973c'`` and ``'AIMNet2@wb97m'``.
        model_index (int): the index of models
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.

    '''
    
    supported_methods = ['AIMNet2@b973c', 'AIMNet2@wb97m']
    element_symbols_available = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']

    def __init__(self, method: str = 'AIMNet2@b973c', model_index=None, device=None):
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if isinstance(model_index, int):
            self.model = aimnet2_methods_single(method, model_index, device=self.device)
        else:
            self.model = aimnet2_methods_ensemble(method)

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
            self.model.predict(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)

class aimnet2_methods_single(torchani_model, method_model,downloadable_model):

    def __init__(self, method, model_index=None, device=None):

        self.device = device
        self.model_index = model_index 
        self.method = method 
        self.model = self.load(self.method, self.model_index)
        
    def predict(
        self,
        molecule, 
        calculate_energy: bool = False, 
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False):
        
        import torch
        import numpy as np 

        coord = torch.as_tensor(molecule.xyz_coordinates).to(torch.float).to(self.device).unsqueeze(0)
        numbers = torch.as_tensor(molecule.atomic_numbers).to(torch.long).to(self.device).unsqueeze(0)
        charge = torch.tensor([molecule.charge], dtype=torch.float, device=self.device)
        nninput = dict(coord=coord, numbers=numbers, charge=charge)
        nnoutput = self.model.eval(nninput, forces=calculate_energy_gradients, hessian=calculate_hessian)
        if calculate_energy:
            molecule.energy = nnoutput['energy'].item()
        if calculate_energy_gradients:
            gradients = -nnoutput['forces'].detach().cpu().numpy()[0]
            molecule.add_xyz_vectorial_property(gradients, 'energy_gradients')
        if calculate_hessian:
            hessian = nnoutput['hessian'].detach().cpu().numpy()
            hessian = hessian.reshape(hessian.shape[0]*3, hessian.shape[0]*3)
            molecule.hessian = hessian 

    def download(self, model_path, model_index):

        method = self.method.lower().replace('@','_')
        link = f"https://github.com/zubatyuk/aimnet-model-zoo/raw/refs/heads/main/aimnet2/{method}_{model_index}.jpt"

        downloaded_file = self._download(link, None, model_path=model_path, target_name=f'{method}_{model_index}.jpt')

        if not downloaded_file:
            raise ValueError(f'Failed to download required model files. Possible solutions:\n 1. Check your internet connection.\n 2. Download from links below:\n{link}\nThe model .pt files should be placed under {model_path}'); sys.stdout.flush()

    def load(self, method, model_index=None):

        method = method.lower().replace('@','_')
        self.model_downloadable_files[f'{method}_model'] = [f'{method}_{model_index}.jpt']
        model_name, model_path, download = self.check_model_path(self.method)
        if download: self.download(model_path, model_index)
        
        import os, torch
        model_path = os.path.join(model_path, f'{method}_{model_index}.jpt')
        model = torch.jit.load(model_path, map_location=self.device)

        from aimnet2calc.calculator import AIMNet2Calculator
        return AIMNet2Calculator(model)
    
def aimnet2_methods_ensemble(method):

    method_name = method.lower().replace('@', '_').replace('-','_')
    models = []
    for ii in range(4):
        models.append(model_tree_node(
            name=f'{method_name}_{ii}',
            model=aimnet2_methods_single(method, model_index=ii),
            operator='predict'
        ))
    return model_tree_node(
        name=method_name,
        children=models,
        operator='average'
    )


