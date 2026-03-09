'''
.. code-block::
  !---------------------------------------------------------------------------! 
  !                                                                           !                                                                          
  ! omnip2x: OMNI-P2x: A Universal Neural Network Potential for               !
  !          Excited-State Simulations                                        !
  ! Implementations by: Mikolaj Martyka and Pavlo O. Dral                     ! 
  !---------------------------------------------------------------------------! 
'''

from ... import models, data
from multiprocessing import cpu_count
import os,sys
from ...model_cls import method_model, downloadable_model, model_tree_node
import numpy as np
class omnip2x(method_model, downloadable_model):
    supported_methods = ['omni-p2x','omnip2x' ]
    

    def __init__(self,         
                 method: str = 'OMNI-P2x',
                 working_directory: str = '.',
                 nthreads=1,
                 warnings = True,
                ):
        
        self.method = method.lower()
        self.working_directory = working_directory
        self.load()
        self.nthreads = nthreads
        self.warnings = warnings
    @property
    def nthreads(self):
        return self._nthreads

    @nthreads.setter
    def nthreads(self, value):
        self._nthreads = value
        self.emodel.nthreads = self._nthreads
        self.osc_model.nthreads = self._nthreads
    
    def load(self):

        edownload_links = ['https://github.com/dralgroup/omni-p2x/raw/refs/heads/main/model_weights/OMNIP2x_CV_1.pt',
                           'https://github.com/dralgroup/omni-p2x/raw/refs/heads/main/model_weights/OMNIP2x_CV_2.pt',
                           'https://github.com/dralgroup/omni-p2x/raw/refs/heads/main/model_weights/OMNIP2x_CV_3.pt']
        fdownload_link = ['https://github.com/dralgroup/omni-p2x/raw/refs/heads/main/model_weights/OMNIP2x_oscillator_V2.pt']

        download_link_1 = "https://aitomistic.xyz/model/omnip2x_model.zip"
        download_link_2 = edownload_links+fdownload_link

        model_dir = 'omnip2x_model'
        ffile = ["OMNIP2x_oscillator.pt"]; efiles = ["OMNIP2x_CV_1.pt", "OMNIP2x_CV_2.pt", "OMNIP2x_CV_3.pt"]
        # for dl, ff in zip(download_links, efiles+ffile):
        mlatom_model_dir, to_download = self.check_model_path(model_dir, efiles+ffile)
        # mlatom_model_path = os.path.join(mlatom_model_dir, ff)
        if to_download:
            download_emsg = None
            for ii, llink in enumerate([download_link_1, download_link_2]):
                try:
                    if ii == 0:
                        self.download(llink, mlatom_model_dir, extract=True, flatten=False)
                        break
                    else:
                        for dl, ff in zip(download_link_2, efiles+ffile):
                            mlatom_model_path = os.path.join(mlatom_model_dir, ff)
                            self.download(dl, mlatom_model_path, extract=False, flatten=False)
                except Exception as e:
                    download_emsg = e 
            if download_emsg is not None:
                raise download_emsg        
        self.mlatom_model_dir = mlatom_model_dir
        self.osc_model = models.msani(model_file=os.path.join(mlatom_model_dir, "OMNIP2x_oscillator.pt"), verbose=0)
        en_models = []
        for i in range(0,3):
            model = models.vecmsani(model_file=mlatom_model_dir+"/OMNIP2x_CV_{}.pt".format(i+1), verbose=0)
            en_models.append(model)
        children = [model_tree_node(
                        name=f'OMNI_en_nn{ii}', 
                        model=animodel, 
                        operator='predict') for ii, animodel in enumerate(en_models)]
        model_ensemble = model_tree_node(
                    name='OMNI_en_nns', 
                    children=children, 
                    operator='average')
        self.emodel = model_ensemble
        
        self.uqs = np.array([4.333759266353241942e-03,
                            5.523547643637850661e-03,
                            4.863181308056573580e-03,
                            4.718907877053602103e-03,
                            4.634083754076488530e-03,
                            4.515167937575360717e-03,
                            4.491026248046649585e-03,
                            4.473054320221912100e-03,
                            4.439498560961521213e-03,
                            4.468134731761315946e-03,
                            4.649729310286577696e-03])
    def predict(
        self, 
        molecular_database=None, 
        molecule=None,
        calculate_energy=True, 
        calculate_energy_gradients=False,
        nstates = 1,
        current_state=0,
        use_fragment_correction=False
    ):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        for mol in molDB.molecules:
            if use_fragment_correction and nstates>1:
                frag_db = check_fragments(mol)
                if frag_db:
                    ee_list = []
                    f_list = []
                    uq_list = []
                    
                    for fragmol in frag_db:
                        self.predict_for_molecule(molecule=fragmol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, nstates=nstates, current_state =current_state)
                        for ee in fragmol.excitation_energies:
                            ee_list.append(ee)
                        for f in fragmol.oscillator_strengths:
                            f_list.append(f)
                        for istate in fragmol.electronic_states:
                            uq_list.append(istate.energy_standard_deviation)
                            
                    self.predict_for_molecule(molecule=mol, calculate_energy=True, calculate_energy_gradients=False, nstates=nstates, current_state =0)
                    if np.min(ee_list) < np.min(mol.excitation_energies):
                        print("Splitting the system into locally excited chromophores")
                        mol.electronic_states = []
                        mol_copy = mol.copy()
                        mol_copy.electronic_states = []
                        for _ in range(len(frag_db)*(nstates-1)+1 - len(mol.electronic_states)):
                            mol.electronic_states.append(mol_copy.copy())
                        sorted_indices = np.argsort(ee_list)
                        mol.electronic_states[0].energy = mol.energy
                        for istate, idx in enumerate(sorted_indices):
                            mol.electronic_states[istate+1].energy  = mol.energy+ee_list[idx]
                        mol.oscillator_strengths = []
                        for idx in sorted_indices:
                            mol.oscillator_strengths.append(f_list[idx])
                        for idx in sorted_indices:
                            mol.electronic_states[idx].energy_standard_deviation = uq_list[idx]
                        mol.energy_standard_deviation = mol.electronic_states[current_state].energy_standard_deviation
                        
                            
                        
                    else:
                        print("Excitation energy of total system lower than of fragments, treating system as exciton.")
                        pass
                        
                else:
                    self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, nstates=nstates, current_state =current_state)
            else:
                self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, nstates=nstates, current_state =current_state)
                    
            
    def predict_for_molecule(
    self,
    molecule=None,
    calculate_energy=True, 
    calculate_energy_gradients=False,
    nstates=1,
    current_state = 0
):

        for atom in molecule.atoms:
            if not atom.atomic_number in [1, 6, 7, 8, 9, 16, 17]:
                errmsg = ' * Warning * Molecule contains elements other than CHNOFSCl, no calculations performed'
                raise ValueError(errmsg)
        
        self.emodel.predict(
            molecule=molecule,
            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, current_state = current_state, nstates=nstates
        )
        
        
        if nstates > 1:
            for i in range(nstates):
                molecule.electronic_states[i].__dict__['OMNI_en_nns'].standard_deviation(properties=['energy'])
            for i in range(nstates):
                molecule.electronic_states[i].energy_standard_deviation = molecule.electronic_states[i].__dict__['OMNI_en_nns'].energy_standard_deviation
                try:
                    if (self.warnings == True and molecule.electronic_states[i].energy_standard_deviation > self.uqs[i]):
                        print(f"Uncertainty of state {i} is high, please check the result")
                except:
                    pass
            molecule.energy_standard_deviation = molecule.electronic_states[current_state].__dict__['OMNI_en_nns'].energy_standard_deviation
        else:
            molecule.__dict__['OMNI_en_nns'].standard_deviation(properties=['energy'])
            molecule.energy_standard_deviation = molecule.__dict__['OMNI_en_nns'].energy_standard_deviation
            if self.warnings and molecule.energy_standard_deviation > self.uqs[0]:
                print(f"Uncertainty of state 0 is high, please check the result")
        
        
        if nstates> 1:
            self.osc_model.predict(molecule=molecule, calculate_energy=False, calculate_energy_gradients=False, nstates=nstates, property_to_predict= "f")
            molecule.oscillator_strengths = []
            for istate in range(nstates):
                if istate == 0:
                    pass
                else:
                    exec("molecule.oscillator_strengths.append(molecule.f_state{})".format(istate))
            for idx, val in enumerate(molecule.oscillator_strengths):
                if val<0:
                    molecule.oscillator_strengths[idx] =0.00

    
    
    
    def train(self,molecular_database=None, reset_energy_shifter=True,train_osc = False,xyz_derivative_property_to_learn = None, spliting_ratio=0.8, en_model_filename="OMNI-P2x_ft_emodel", osc_model_filename="OMNI-P2x_ft_osc_model", nstates=2, verbose=1, hyperparameters = {}):
        if hyperparameters:
            _hyperparameters = hyperparameters
        else:
            _hyperparameters = {}
        if 'max_epochs' not in _hyperparameters:
            _hyperparameters['max_epochs'] = 100
        if 'gap_coefficient' not in _hyperparameters:
            _hyperparameters['gap_coefficient'] = 1
        if 'fixed_layers' not in _hyperparameters:
            _hyperparameters['fixed_layers'] = [[0,4]]
        new_models = []

        model_dir = 'omnip2x_model'
        ffile = ["OMNIP2x_oscillator.pt"]; efiles = ["OMNIP2x_CV_1.pt", "OMNIP2x_CV_2.pt", "OMNIP2x_CV_3.pt"]
        mlatom_model_dir, to_download = self.check_model_path(model_dir, efiles+ffile)

        for i in range(0,3):
            print(f"Training the energy model no.{i+1}")
            cvmodel = models.vecmsani(model_file=mlatom_model_dir+"/OMNIP2x_CV_{}.pt".format(i+1).format(i+1), verbose=verbose)
            cvmodel.save(en_model_filename+f"ft_{i+1}.pt")
            model = models.vecmsani(model_file=en_model_filename+f"ft_{i+1}.pt", nstates=nstates, verbose=verbose)
            model.train(molecular_database,reset_energy_shifter=reset_energy_shifter, hyperparameters =_hyperparameters, property_to_learn='energy', xyz_derivative_property_to_learn=xyz_derivative_property_to_learn, spliting_ratio=spliting_ratio, reset_optimizer = True)
            new_models.append(model)
        if train_osc:
            print("Training the oscillator strength model")
            _hyperparameters['gap_coefficient'] = 0
            for mol in molecular_database:
                for istate in range(len(mol.electronic_states)):
                    if istate ==0:
                        mol.electronic_states[istate].f = 0
                    else:
                        mol.electronic_states[istate].f = mol.oscillator_strengths[istate-1]
            self.osc_model.save(osc_model_filename)
            f_model = models.msani(model_file=osc_model_filename+".pt", nstates=nstates, verbose=verbose)
            f_model.train(molecular_database,reset_energy_shifter=True, hyperparameters =_hyperparameters, property_to_learn='f', reset_optimizer = True, )
            self.osc_model = f_model

        children = [model_tree_node(
                    name=f'OMNI_en_nn{ii}', 
                    model=animodel, 
                    operator='predict') for ii, animodel in enumerate(new_models)]
        new_model_ensemble = model_tree_node(
                    name='OMNI_en_nns', 
                    children=children, 
                    operator='average')
        self.emodel = new_model_ensemble
        self.warnings = False

COVALENT_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Cl': 0.99,
    'Br': 1.14,
    'I': 1.33,
    'S': 1.03,
}

def bonded(atom1, atom2, coord1, coord2, tolerance=0.7):
    r1 = COVALENT_RADII.get(atom1, 0.1)
    r2 = COVALENT_RADII.get(atom2, 0.1)
    max_dist = r1 + r2 + tolerance
    dist = np.linalg.norm(coord1 - coord2)
    return dist < max_dist

def get_fragments(mol):
    import networkx as nx
    atoms = mol.get_element_symbols()
    coords = mol.xyz_coordinates
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if bonded(atoms[i], atoms[j], coords[i], coords[j]):
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    
    fragments = []
    for comp in components:
        indices = sorted(comp)
        fragment_atoms = [atoms[i] for i in indices]
        fragment_coords = coords[indices]
        fragments.append(indices)
    rtrn_db = data.molecular_database()
    for frag in fragments:
        fragment_mol = data.molecule()
        for idx in frag:
            fragment_mol.atoms.append(mol.atoms[idx])
        rtrn_db.append(fragment_mol)
    
    return rtrn_db

def check_fragments(molecule=None):
    db = get_fragments(molecule)
    if len(db) !=1:
        print("Non-covalent molecular complex detected, applying fragment correction!")
        return db
    else:
        return None

def increase_input_dim(model, n):
    import torch
    import torch.nn as nn
    for name, submodel in model.named_children():
        if isinstance(submodel, nn.Sequential):
            old_layer = submodel[0]
            assert isinstance(old_layer, nn.Linear), f"First layer of {name} is not Linear"

            # create new layer with +1 input dim
            new_layer = nn.Linear(old_layer.in_features + n, old_layer.out_features, bias=old_layer.bias is not None)

            # Copy existing weights
            with torch.no_grad():
                # copy old weights to the first part of the new weight matrix
                new_layer.weight[:, :-n] = old_layer.weight
                new_layer.weight[:, -1].normal_(0, 1e-3)  
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)

            # Replace the old layer in the sequential
            submodel[0] = new_layer

    return model        
if __name__ == '__main__':
    pass
                
        
        
            
        



        