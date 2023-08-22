#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! columbus: interface to the COLUMBUS program                               ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
pythonpackage = True
try: from .. import constants, data, stopper
except:
    import constants, data, stopper
    pythonpackage = False

class columbus_methods():
    def __init__(self, command_line_arguments=None, save_files_in_current_directory=True, directory_with_input_files=None):
        self.command_line_arguments = command_line_arguments
        self.save_files_in_current_directory = save_files_in_current_directory
        self.directory_with_input_files = directory_with_input_files
        try:
            self.progbin = os.environ['COLUMBUS'] + '/runc'
        except:
            msg = 'Cannot find the COLUMBUS program, please set the environment variable: export COLUMBUS=...'
            if pythonpackage: raise ValueError(msg)
            else: stopper.stopMLatom(msg) 
        
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)
        
        import tempfile, subprocess
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory:
                    tmpdirname = '.'
                if self.directory_with_input_files != None:
                    import shutil, os
                    filenames=os.listdir(self.directory_with_input_files)
                    for filename in filenames:
                        shutil.copy2(os.path.join(self.directory_with_input_files, filename), tmpdirname)                  
            
                xyzfilename = f'{tmpdirname}/geom'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename, format='COLUMBUS')
                
                progargs = [self.progbin] + self.command_line_arguments
                proc = subprocess.Popen(progargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                proc.wait()
                # Get energies
                energiespath = f'{tmpdirname}/LISTINGS/energy'
                if os.path.exists(energiespath):
                    with open(energiespath) as ff:
                        for line in ff:
                            if '**** ROOTS ****' in line:
                                mol.electronic_state_energies = []
                                continue
                            if '***' in line: break
                            mol.electronic_state_energies.append(float(line.split()[-1]))
                
                # Get energy gradients
                nstates = len(mol.electronic_state_energies)
                for atom in mol.atoms:
                    atom.electronic_state_energy_gradients = []
                for istate in range(1, nstates+1):
                    gradpath = f'{tmpdirname}/GRADIENTS/cartgrd.drt1.state{istate}.sp'
                    if os.path.exists(gradpath):
                        with open(gradpath) as ff:
                            iatom = -1
                            for line in ff:
                                iatom += 1
                                mol.atoms[iatom].electronic_state_energy_gradients.append(np.array([float(xx) / constants.Bohr2Angstrom for xx in line.replace('D','e').split()]).astype(float))
                
                
                # Get nonadiabatic coupling vectors
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

if __name__ == '__main__':
    pass