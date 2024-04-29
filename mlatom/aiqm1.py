#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! aiqm1: Artificial intelligence quantum-mechanical method 1                ! 
  ! Implementations by: Peikung Zheng & Pavlo O. Dral                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os
from . import data, models, stopper

import torch
import torchani
from torchani.utils import ChemicalSymbolsToInts

class aiqm1(models.torchani_model):
    """
    The Artificial intelligence-quantum mechanical method as in the `AIQM1 paper`_.

    Arguments:
        method (str, optional): AIQM method used. Currently supports AIQM1, AIQM1\@DFT*, and AIQM1\@DFT. Default value: AIQM1.
        qm_program (str): The QM program used in the calculation of ODM2* part. Currently supports MNDO and Sparrow program. 
        qm_program_kwargs (dictionary, optional): Keywords passed to QM program.
 
    .. _AIQM1 Paper:
        https://doi.org/10.1038/s41467-021-27340-2
    
    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run AIQM1 calculation
        aiqm1 = ml.models.methods(method='AIQM1', qm_program='MNDO')
        aiqm1.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True)
        # Get energy, gradient, and prediction uncertainty of AIQM1 
        energy = mol.energy
        gradient = mol.gradient
        std = mol.aiqm1_nn.energy_standard_deviation


    """
    available_methods = models.methods.methods_map['aiqm1']
    atomic_energies = {'AIQM1': {1:-0.50088038, 6:-37.79221710, 7:-54.53360298, 8:-75.00986203},
                       'AIQM1@DFT': {1:-0.50139362, 6:-37.84623117, 7:-54.59175573, 8:-75.07674376}}
    atomic_energies['AIQM1@DFT*'] = atomic_energies['AIQM1@DFT']
    
    def __init__(self, method='AIQM1', qm_program=None, qm_program_kwargs={}, dftd4_kwargs={}, **kwargs):
        self.method = method.upper()
        self.qm_program = qm_program
        self.qm_program_kwargs = qm_program_kwargs
        modelname = self.method.lower().replace('*','star').replace('@','at')
        ani_nn_children = []
        for ii in range(8):
            nn_i = models.model_tree_node(name=f'{modelname}_nn{ii}', operator='predict', model=ani_nns_in_aiqm1(method=self.method, model_index=ii))
            ani_nn_children.append(nn_i)
        ani_nns = models.model_tree_node(name=f'{modelname}_nn', children=ani_nn_children, operator='average')
        shift = models.model_tree_node(name=f'{modelname}_atomic_energy_shift', operator='predict', model=atomic_energy_shift(method=self.method))
        odm2star = models.model_tree_node(name='odm2star', operator='predict', model=models.methods(method='ODM2*', program=qm_program, **qm_program_kwargs))
        aiqm1_children = [ani_nns, shift, odm2star]
        if self.method != 'AIQM1@DFT*':
            d4 = models.model_tree_node(name='d4_wb97x', operator='predict', model=models.methods(method='D4', functional='wb97x', **dftd4_kwargs))
            aiqm1_children.append(d4)
        self.aiqm1_model = models.model_tree_node(name=modelname, children=aiqm1_children, operator='sum')
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, current_state=0, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        if 'nthreads' in self.__dict__: self.aiqm1_model.nthreads = self.nthreads
        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol,
                                    calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, nstates=nstates, 
                                    current_state=current_state,
                                    **kwargs)
        
    def predict_for_molecule(self, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, current_state=0, **kwargs):
        
        for atom in molecule.atoms:
            if not atom.atomic_number in [1, 6, 7, 8]:
                errmsg = ' * Warning * Molecule contains elements other than CHNO, no calculations performed'
                # print(errmsg)
                raise ValueError(errmsg)
        
        if nstates >1:
            mol_copy = molecule.copy()
            mol_copy.electronic_states = []
            for _ in range(nstates - len(molecule.electronic_states)):
                molecule.electronic_states.append(mol_copy.copy())
        
        # for molecule in molecules:
        if len(molecule.atoms) == 1:
            molecule.energy = self.atomic_energies[self.method][molecule.atoms[0].atomic_number]
            standard_atom = data.atom(atomic_number=molecule.atoms[0].atomic_number)
            if molecule.charge != 0 or molecule.multiplicity != standard_atom.multiplicity:
                odm2model = models.methods(method='ODM2*', program=self.qm_program)
                mol_odm2 = molecule.copy()
                odm2model.predict(molecule=mol_odm2, nstates=nstates, **kwargs)
                mol_standard_odm2 = molecule.copy() ; mol_standard_odm2.charge = 0; mol_standard_odm2.multiplicity=standard_atom.multiplicity
                odm2model.predict(molecule=mol_standard_odm2, nstates=nstates, **kwargs)
                molecule.energy = molecule.energy + mol_odm2.energy - mol_standard_odm2.energy
        else:
            if nstates > 1 and isinstance(calculate_energy_gradients, list): 
                if any(calculate_energy_gradients):
                    calculate_energy_gradients = [True] * nstates
            self.aiqm1_model.predict(molecule=molecule,
                                    calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, nstates=nstates, 
                                    current_state=current_state,
                                    **kwargs)
            
            properties = [] ; atomic_properties = []
            
            calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
            calculate_hessian = bool(np.array(calculate_hessian).any())
            if calculate_energy: properties.append('energy')
            if calculate_energy_gradients: atomic_properties.append('energy_gradients')
            if calculate_hessian: properties.append('hessian')
            modelname = self.method.lower().replace('*','star').replace('@','at')

            if nstates >1:
                for mol_el_st in molecule.electronic_states:
                    mol_el_st.__dict__[f'{modelname}_nn'].standard_deviation(properties=properties+atomic_properties)
            else:
                molecule.__dict__[f'{modelname}_nn'].standard_deviation(properties=properties+atomic_properties)
            

class atomic_energy_shift(models.model):
    atomic_energy_shifts = {'AIQM1': {1: -4.29365862e-02, 6: -3.34329586e+01, 7: -4.69301173e+01, 8: -6.29634763e+01},
                            'AIQM1@DFT': {1: -4.27888067e-02, 6: -3.34869833e+01, 7: -4.69896148e+01, 8: -6.30294433e+01}}
    atomic_energy_shifts['AIQM1@DFT*'] = atomic_energy_shifts['AIQM1@DFT']
    
    def __init__(self, method = 'AIQM1'):
        self.method = method
        
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
        calculate_hessian = bool(np.array(calculate_hessian).any())
         
        for mol in molDB.molecules:
            molecules = [mol]

            if nstates >1:
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())
                molecules = mol.electronic_states
            
            for mol in molecules:
                if calculate_energy:
                    sae = 0.0
                    for atom in mol.atoms:
                        sae += self.atomic_energy_shifts[self.method][atom.atomic_number]
                    mol.energy = sae
                if calculate_energy_gradients:
                    for atom in mol.atoms:
                        atom.energy_gradients = np.zeros(3)
                if calculate_hessian:
                    ndim = len(mol.atoms) * 3
                    mol.hessian = np.zeros(ndim*ndim).reshape(ndim,ndim)

class ani_nns_in_aiqm1(models.torchani_model):
    species_order = [1, 6, 7, 8]
    
    def __init__(self, method='AIQM1', model_index = 0):
        if method == 'AIQM1':
            self.level = 'cc'
        elif method in ['AIQM1@DFT', 'AIQM1@DFT*']:
            self.level = 'dft'
        self.model_index = model_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.define_aev()
        self.load_model()
    
    def define_aev(self):
        Rcr = 5.2000e+00
        Rca = 4.0000e+00
        EtaR = torch.tensor([1.6000000e+01], device=self.device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=self.device)
        Zeta = torch.tensor([3.2000000e+01], device=self.device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=self.device)
        EtaA = torch.tensor([8.0000000e+00], device=self.device)
        ShfA = torch.tensor([9.0000000e-01, 1.6750000e+00,  2.4499998e+00, 3.2250000e+00], device=self.device)
        num_species = len(self.species_order)
        aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        self.aev_computer = aev_computer

    def load_model(self):
        mlatomdir=os.path.dirname(__file__)
        dirname = os.path.join(mlatomdir, 'aiqm1_model')
        method = 'aiqm1_' + self.level
        self.define_nn()
        checkpoint = torch.load(os.path.join(dirname, f'{method}_cv{self.model_index}.pt'), map_location=self.device)
        self.nn.load_state_dict(checkpoint['nn'])
        self.model  = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).double()

    def define_nn(self):
        aev_dim = self.aev_computer.aev_length
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 144),
            torch.nn.GELU(),
            torch.nn.Linear(144, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
        self.nn = nn
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        calculate_energy_gradients = bool(np.array(calculate_energy_gradients).any())
        calculate_hessian = bool(np.array(calculate_hessian).any())
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)
        
        for mol in molDB.molecules:
            molecules = [mol]

            if nstates >1:
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())
                molecules = mol.electronic_states
            
            for mol in molecules:
                atomic_numbers = np.array([atom.atomic_number for atom in mol.atoms])
                xyz_coordinates = torch.tensor(np.array(mol.xyz_coordinates).astype('float')).to(self.device).requires_grad_(calculate_energy_gradients or calculate_hessian)
                xyz_coordinates = xyz_coordinates.unsqueeze(0)
                species = species_to_tensor(atomic_numbers).to(self.device).unsqueeze(0)
                ANI_NN_energy = self.model((species, xyz_coordinates)).energies
                if calculate_energy: mol.energy = float(ANI_NN_energy)
                if calculate_energy_gradients or calculate_hessian:
                    ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energy.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                    if calculate_energy_gradients:
                        grads = ANI_NN_energy_gradients[0].detach().cpu().numpy()
                        for iatom in range(len(mol.atoms)):
                            mol.atoms[iatom].energy_gradients = grads[iatom]
                if calculate_hessian:
                    ANI_NN_hessian = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energy)
                    mol.hessian = ANI_NN_hessian[0].detach().cpu().numpy()

if __name__ == '__main__':
    pass
