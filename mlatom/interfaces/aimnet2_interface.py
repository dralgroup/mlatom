'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! aimnet2: Universal AIMnet2 models                                         ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''
from ..model_cls import torchani_model, method_model, model_tree_node, downloadable_model
from ..decorators import doc_inherit
from .. import constants
import os, sys 

class aimnet2_methods(torchani_model, method_model):

    '''
    Universal ML methods with AIMNet2: https://doi.org/10.26434/chemrxiv-2023-296ch. Model files can be downloaded according to https://github.com/isayevlab/aimnetcentral/blob/main/aimnet/calculators/model_registry.yaml. For installation of AIMNet2, please refer to https://github.com/isayevlab/aimnetcentral.

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'AIMNet2@b973c'`` and ``'AIMNet2@wb97m'``.
        model_index (int): the index of models
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.

    '''
    
    supported_methods = ['AIMNet2@b973c', 'AIMNet2@b973c-d3', 'AIMNet2@wb97m', 'AIMNet2@wb97m-d3']
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

        coord = torch.as_tensor(molecule.xyz_coordinates).to(torch.float).to(self.device)
        numbers = torch.as_tensor(molecule.atomic_numbers).to(torch.long).to(self.device)
        charge = torch.tensor(molecule.charge, dtype=torch.float, device=self.device)
        mult = torch.tensor(molecule.multiplicity,dtype=torch.float, device=self.device)
        nninput = dict(coord=coord, numbers=numbers, charge=charge, mult=mult)
        nnoutput = self.model(nninput, forces=calculate_energy_gradients, hessian=calculate_hessian)
        if calculate_energy:
            molecule.energy = nnoutput['energy'].item() * constants.eV2Hartree
        if calculate_energy_gradients:
            gradients = -nnoutput['forces'].detach().cpu().numpy() * constants.eV2Hartree
            molecule.add_xyz_vectorial_property(gradients, 'energy_gradients')
        if calculate_hessian:
            hessian = nnoutput['hessian'].detach().cpu().numpy() * constants.eV2Hartree
            hessian = hessian.reshape(hessian.shape[0]*3, hessian.shape[0]*3)
            molecule.hessian = hessian 

    def load(self, method, model_index=None):

        import os, torch

        method = method.lower().replace('@','_').replace('-d3','')
        download_links = [f"https://storage.googleapis.com/aimnetcentral/AIMNet2/{method}_d3_{model_index}.jpt"]
        model_dir = f"{method}_model"
        model_files = f"{method}_d3_{model_index}.jpt"

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        mlatom_model_path = os.path.join(mlatom_model_dir, f'{method}_d3_{model_index}.jpt')
        if to_download: self.download(download_links, mlatom_model_path, extract=False, flatten=False)
        
        model = torch.jit.load(mlatom_model_path, map_location=self.device)

        from aimnet.calculators import AIMNet2Calculator
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


