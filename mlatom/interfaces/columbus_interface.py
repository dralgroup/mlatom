#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! columbus: interface to the COLUMBUS program                               ! 
  ! Implementations by: Lina Zhang and Pavlo O. Dral                          !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
from .. import constants, models

class columbus_methods(models.model):
    def __init__(self, command_line_arguments=None, save_files_in_current_directory=True, working_directory=None, directory_with_input_files='', **kwargs):
        self.command_line_arguments = command_line_arguments
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.directory_with_input_files = directory_with_input_files
        if self.directory_with_input_files != '':
            self.directory_with_input_files = os.path.abspath(directory_with_input_files)
        try:
            self.progbin = os.environ['COLUMBUS'] + '/runc'
        except:
            msg = 'Cannot find the COLUMBUS program, please set the environment variable: export COLUMBUS=...'
            raise ValueError(msg)
        
    def predict(self, 
                molecular_database=None, 
                molecule=None, 
                nstates=1, 
                current_state=0, 
                calculate_energy=True, 
                calculate_energy_gradients=False, 
                calculate_hessian=False,
                calculate_nacv=False):

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
                
                columbus_successful = False
                if 'timings' in outs+errs:
                    columbus_successful = True
                
                mol.columbus_successful = columbus_successful

                if mol.columbus_successful:
                    # Get properties
                    # Get energies
                    if calculate_energy:
                        energiespath = f'{tmpdirname}/LISTINGS/mcscfsm.sp'
                        if os.path.exists(energiespath):
                            found_states = False
                            with open(energiespath) as ff:
                                for line in ff:
                                    if 'Individual total energies for all states' in line.strip():
                                        state_energies = []
                                        found_states = True
                                        continue
                                    if found_states:
                                        if '---' in line: break
                                        state_energies.append(float(line.split('=')[1].strip().split(',')[0]))
                        else:
                            print('There is no energy file.')                   


                        mol_copy = mol.copy()
                        mol_copy.electronic_states = []
                        for _ in range(nstates - len(mol.electronic_states)):
                            mol.electronic_states.append(mol_copy.copy())
                        for i in range(nstates):
                            mol.electronic_states[i].energy = state_energies[i]
                        mol.energy = mol.electronic_states[current_state].energy
                    
                    # Get energy gradients
                    if any(calculate_energy_gradients):
                        states = []
                        for filename in os.listdir(tmpdirname+'/GRADIENTS'):
                            if filename.startswith('cartgrd.drt1.state') and filename.endswith('.sp'):
                                istate = int(filename.split('state')[1].split('.')[0])
                                states.append(istate-1)
                        if len(states) == 0:
                            print('There is no gradient file.')
                        else:
                            state_gradients = {}
                            for istate in states:
                                gradpath = f'{tmpdirname}/GRADIENTS/cartgrd.drt1.state{istate+1}.sp'
                                energy_gradient = []
                                with open(gradpath) as ff:
                                    iatom = -1
                                    for line in ff:
                                        iatom += 1
                                        energy_gradient.append(np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float))
                                state_gradients[istate] = energy_gradient 
                            if not mol.electronic_states:
                                    mol.electronic_states.extend([mol.copy() for _ in range(nstates)])
                            for index, value in enumerate(calculate_energy_gradients):
                                if value:
                                    mol.electronic_states[index].add_xyz_derivative_property(np.array(state_gradients[index]).astype(float), 'energy', 'energy_gradients')
                            mol.add_xyz_derivative_property(np.array(state_gradients[current_state]).astype(float), 'energy', 'energy_gradients')
                    
                    # Get nonadiabatic coupling vectors
                    if calculate_nacv:
                        from itertools import combinations
                        state_comb = list(combinations(range(1, nstates+1), 2))
                        for atom in mol.atoms:
                            atom.nonadiabatic_coupling_vectors = [[np.zeros(3) for ii in range(nstates)] for jj in range(nstates)]
                        for final_state, initial_state in state_comb:
                            nacpath = f'{tmpdirname}/GRADIENTS/cartgrd.nad.drt1.state{initial_state}.drt1.state{final_state}.sp'
                            if os.path.exists(nacpath):
                                with open(nacpath) as ff:
                                    iatom = -1
                                    for line in ff:
                                        iatom += 1
                                        nacvec = np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float)
                                        mol.atoms[iatom].nonadiabatic_coupling_vectors[initial_state-1][final_state-1] = nacvec
                                        mol.atoms[iatom].nonadiabatic_coupling_vectors[final_state-1][initial_state-1] = -nacvec
                else:
                    print('Calculation failed!')
                    continue
if __name__ == '__main__':
    pass