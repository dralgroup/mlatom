#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! columbus: interface to the COLUMBUS program                               ! 
  ! Implementations by: Jakub Martinka, Lina Zhang and Pavlo O. Dral          !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
from .. import constants
from ..model_cls import method_model
class columbus_methods(method_model):
    bin_env_name = 'COLUMBUS'
    
    def __init__(self, command_line_arguments=None, save_files_in_current_directory=True, working_directory=None, directory_with_input_files='', **kwargs):
        self.command_line_arguments = command_line_arguments
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.directory_with_input_files = directory_with_input_files
        if self.directory_with_input_files != '':
            self.directory_with_input_files = os.path.abspath(directory_with_input_files)
        self.progbin = self.get_bin_env_var()
        if self.progbin is not None:
            self.progbin += '/runc'
        else:
            raise ValueError('Cannot find the COLUMBUS program, please set the environment variable: export COLUMBUS=...')
        
    def predict(self, 
                molecular_database=None, 
                molecule=None, 
                nstates=1,
                nsinglets=1,
                ntriplets=0,
                current_state=0, 
                calculate_energy=True, 
                calculate_energy_gradients=False, 
                calculate_hessian=False,
                calculate_nacv=False,
                calculate_soc=False,
                level_of_theory='CASSCF'):

        if ntriplets == 0:
            nsinglets = nstates
        if nstates != nsinglets + ntriplets:
            raise ValueError(f'Total number of states does not correspond to the number of singlet and triplet states: (nstates: {nstates}, nsinglets: {nsinglets}, ntriplets: {ntriplets}')

        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        if not isinstance(calculate_energy_gradients, list):
            if calculate_energy_gradients:
                calculate_energy_gradients = [False] * nstates
                calculate_energy_gradients[current_state] = True
            else:
                calculate_energy_gradients = [False] * nstates

        if not isinstance(calculate_hessian, list):
            if calculate_hessian:
                calculate_hessian = [False] * nstates
                calculate_hessian[current_state] = True
            else:
                calculate_hessian = [False] * nstates
        
        import tempfile, subprocess
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory:
                    tmpdirname = '.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                    if not os.path.exists(tmpdirname):
                        os.makedirs(tmpdirname)
                    tmpdirname = os.path.abspath(tmpdirname)
                if self.directory_with_input_files != '':
                    import shutil, os
                    filenames=os.listdir(self.directory_with_input_files)
                    for filename in filenames:
                        shutil.copy2(os.path.join(self.directory_with_input_files, filename), tmpdirname)                  

                xyzfilename = f'{tmpdirname}/geom'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename, format='COLUMBUS')
                
                progargs = [self.progbin] + self.command_line_arguments
                proc = subprocess.Popen(progargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                outs,errs = proc.communicate()
                with open(f'{tmpdirname}/runls', 'w') as ff:
                    ff.write(outs)

                columbus_successful = True if 'timings' in outs+errs else False
                mol.columbus_successful = columbus_successful

                if mol.columbus_successful:
                    # Get properties
                    # Get energies
                    if calculate_energy:
                        if level_of_theory == 'CASSCF':
                            energiespath = f'{tmpdirname}/LISTINGS/mcscfsm.sp'
                            if os.path.exists(energiespath):
                                found_states = False
                                with open(energiespath, 'r') as ff:
                                    for line in ff:
                                        if 'Individual total energies for all states' in line.strip():
                                            state_energies_singlets = []
                                            state_energies_triplets = []
                                            found_states = True
                                            continue
                                        if found_states:
                                            if '---' in line: break
                                            if ntriplets == 0:
                                                state_energies_singlets.append(float(line.split('=')[1].strip().split(',')[0]))
                                            else: 
                                                if 'DRT #1' in line:
                                                    state_energies_singlets.append(float(line.split('=')[1].strip().split(',')[0]))
                                                if 'DRT #2' in line:
                                                    state_energies_triplets.append(float(line.split('=')[1].strip().split(',')[0]))
                            else:
                                print('There is no CASSCF energy file. (LISTINGS/mcscfsm.sp)')

                        if level_of_theory == 'MRCI':
                            if ntriplets > 0:
                                energiespath = f'{tmpdirname}/LISTINGS/ciudgsm.drt1.sp'
                            else:
                                energiespath = f'{tmpdirname}/LISTINGS/ciudgsm.sp'
                            if os.path.exists(energiespath):
                                state_energies_singlets = []
                                with open(energiespath, 'r') as ff:
                                    for line in ff:
                                        if 'eci       =' in line.strip():
                                            state_energies_singlets.append(float(line.strip().split()[2]))
                            else:
                                print('There is no MRCI singlet energy file. (LISTINGS/ciudgsm.sp or LISTINGS/ciudgsm.drt1.sp)')
                            if ntriplets > 0:
                                energiespath = f'{tmpdirname}/LISTINGS/ciudgsm.drt2.sp'
                                if os.path.exists(energiespath):
                                    state_energies_triplets = []
                                    with open(energiespath, 'r') as ff:
                                        for line in ff:
                                            if 'eci       =' in line.strip():
                                                state_energies_triplets.append(float(line.strip().split()[2]))
                                else:
                                    print('There is no MRCI triplet energy file. (LISTINGS/ciudgsm.drt2.sp)')

                        mol_copy = mol.copy()
                        mol_copy.electronic_states = []
                        for _ in range(nstates - len(mol.electronic_states)):
                            mol.electronic_states.append(mol_copy.copy())
                        for i in range(nsinglets):
                            mol.electronic_states[i].energy = state_energies_singlets[i]
                        for i in range(ntriplets):
                            mol.electronic_states[nsinglets+i].energy = state_energies_triplets[i]
                            mol.electronic_states[nsinglets+i].multiplicity = 3
                        mol.energy = mol.electronic_states[current_state].energy # triplets are listed after singlets (e.g. S0,S1,T1)
                    
                    # Get energy gradients (CASSCF/MRCI gradients are saved in the same file)
                    if any(calculate_energy_gradients):
                        singlet_states = []; triplet_states = [];
                        for filename in os.listdir(tmpdirname+'/GRADIENTS'):
                            if filename.startswith('cartgrd.drt1.state') and filename.endswith('.sp'):
                                istate = int(filename.split('state')[1].split('.')[0])
                                singlet_states.append(istate-1)
                            if filename.startswith('cartgrd.drt2.state') and filename.endswith('.sp'):
                                istate = int(filename.split('state')[1].split('.')[0])
                                triplet_states.append(istate-1)
                        if len(singlet_states) == 0 and len(triplet_states) == 0:
                            print('There is no gradient file.')
                        else:
                            singlet_state_gradients = {}; triplet_state_gradients = {}
                            for istate in singlet_states:
                                gradpath = f'{tmpdirname}/GRADIENTS/cartgrd.drt1.state{istate+1}.sp'
                                energy_gradient = []
                                with open(gradpath) as ff:
                                    iatom = -1
                                    for line in ff:
                                        iatom += 1
                                        energy_gradient.append(np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float))
                                singlet_state_gradients[istate] = energy_gradient
                            for istate in triplet_states:
                                gradpath = f'{tmpdirname}/GRADIENTS/cartgrd.drt2.state{istate+1}.sp'
                                energy_gradient = []
                                with open(gradpath) as ff:
                                    iatom = -1
                                    for line in ff:
                                        iatom += 1
                                        energy_gradient.append(np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float))
                                triplet_state_gradients[istate] = energy_gradient
                            if not mol.electronic_states:
                                    mol.electronic_states.extend([mol.copy() for _ in range(nstates)])
                            for index, value in enumerate(calculate_energy_gradients):
                                if value and index < nsinglets:
                                    mol.electronic_states[index].add_xyz_derivative_property(np.array(singlet_state_gradients[index]).astype(float), 'energy', 'energy_gradients')
                                if value and index >= nsinglets:
                                    mol.electronic_states[index].add_xyz_derivative_property(np.array(triplet_state_gradients[index-nsinglets]).astype(float), 'energy', 'energy_gradients')

                            if current_state < nsinglets:
                                mol.add_xyz_derivative_property(np.array(singlet_state_gradients[current_state]).astype(float), 'energy', 'energy_gradients')
                            else:
                                mol.add_xyz_derivative_property(np.array(triplet_state_gradients[current_state-nsinglets]).astype(float), 'energy', 'energy_gradients')
                    
                    # Get nonadiabatic coupling vectors
                    if calculate_nacv:
                        from itertools import combinations
                        state_comb = list(combinations(range(1, nsinglets+1), 2))
                        mol.nacv = np.zeros((nsinglets,nsinglets,*mol.xyz_coordinates.shape))
                        for initial_state, final_state in state_comb:
                            nacpath = f'{tmpdirname}/GRADIENTS/cartgrd.nad.drt1.state{initial_state}.drt1.state{final_state}.sp'
                            if os.path.exists(nacpath):
                                with open(nacpath) as ff:
                                    iatom = -1
                                    for line in ff:
                                        iatom += 1
                                        nac_per_atom = np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float)
                                        mol.nacv[initial_state-1][final_state-1][iatom] = nac_per_atom
                                        mol.nacv[final_state-1][initial_state-1][iatom] = -nac_per_atom
                        if ntriplets > 1:
                            mol.nonadiabatic_coupling_vectors_triplets = np.zeros((ntriplets,ntriplets,*mol.xyz_coordinates.shape))
                            state_comb = list(combinations(range(1, ntriplets+1), 2))
                            for initial_state, final_state in state_comb:
                                nacpath = f'{tmpdirname}/GRADIENTS/cartgrd.nad.drt2.state{initial_state}.drt2.state{final_state}.sp'
                                if os.path.exists(nacpath):
                                    with open(nacpath) as ff:
                                        iatom = -1
                                        for line in ff:
                                            iatom += 1
                                            nac_per_atom = np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float)
                                            mol.nonadiabatic_coupling_vectors_triplets[initial_state-1][final_state-1][iatom] = nac_per_atom
                                            mol.nonadiabatic_coupling_vectors_triplets[final_state-1][initial_state-1][iatom] = -nac_per_atom
                    
                    # Get spin-orbit couplings
                    if calculate_soc:
                        from itertools import product
                        state_comb = list(product(range(nsinglets), range(ntriplets)))
                        mol.spin_orbit_coupling_vector = [[0.0, 0.0, 0.0] for _ in range(nsinglets*ntriplets)]
                        runls = f'{tmpdirname}/runls'
                        found_socs = False
                        if os.path.exists(runls):
                            ht_matrix = []
                            with open(runls) as ff:
                                for line in ff:
                                    if 'HT matrix in AU' in line.strip():
                                        found_socs = True
                                        singlet_lines = 0
                                        continue
                                    if found_socs:
                                        if '---' in line: break
                                        ht_matrix.append([float(xx) / constants.Hartree2cm for xx in line.split()])
                            for isinglet_state, itriplet_state in state_comb:
                                row = nsinglets + itriplet_state
                                col = isinglet_state
                                socvec = [ht_matrix[row][col], ht_matrix[row+1][col], ht_matrix[row+2][col]]
                                mol.spin_orbit_coupling_vector[isinglet_state*ntriplets+itriplet_state] = socvec
                        
                else:
                    print('Calculation failed!')
                    continue
if __name__ == '__main__':
    pass