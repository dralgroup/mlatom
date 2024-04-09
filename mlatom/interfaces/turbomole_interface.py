#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! Turbomole: interface to the Turbomole program package                     ! 
  ! Implementations by: Sebastian V. Pios and Lina Zhang                      !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
import tempfile
import subprocess
import re
import time
import shutil
from .. import constants, data, models
from .. import utils as file_utils

class turbomole_methods(models.model):
    def __init__(self, save_files_in_current_directory=True, working_directory=None, directory_with_input_files='', **kwargs):
        super().__init__()
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.directory_with_input_files = directory_with_input_files
        if self.directory_with_input_files != '':
            self.directory_with_input_files = os.path.abspath(directory_with_input_files)
        if 'TURBODIR' not in os.environ:
            msg = 'Cannot find the TURBOMOLE program package, please set the environment variable: export TURBODIR=...'
            raise ValueError(msg)

    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        os.environ["PARA_ARCH"] = 'SMP'
        if self.nthreads:
            os.environ["PARNODES"] = str(self.nthreads)
        else:
            os.environ["PARNODES"] = str(os.cpu_count())
        os.environ["PATH"] = f'{os.environ["TURBODIR"]}/bin/{subprocess.getoutput("sysname")}/:{os.environ["PATH"]}'
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "SSE4_2"

    def predict(self, 
                method='ADC(2)',
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

        if method == 'ADC(2)':
            module = ['dscf','ricc2']
            gs_model = 'mp2'
            outfilename = 'ricc2.out'

        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory:
                    tmpdirname = '.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                    if not os.path.exists(tmpdirname):
                        os.makedirs(tmpdirname)
                    else:
                        os.system(f'rm -rf {tmpdirname}')
                        os.makedirs(tmpdirname)
                    tmpdirname = os.path.abspath(tmpdirname)
                if self.directory_with_input_files != '':
                    filenames=os.listdir(self.directory_with_input_files)
                    for filename in filenames:
                        shutil.copy2(os.path.join(self.directory_with_input_files, filename), tmpdirname)

                xyzfilename = f'{tmpdirname}/coord'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename, format='TURBOMOLE')

                turbomole_successful = False

                if method == 'ADC(2)':
                    run_module = subprocess.Popen(f'{module[0]} > {module[0]}.out', stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                    run_module.communicate()

                    check_status = subprocess.Popen('actual', stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                    stdout_data, stderr_data = check_status.communicate()

                    if stdout_data.startswith('fine, '):
                        run_module = subprocess.Popen(f'{module[1]} > {module[1]}.out', stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                        run_module.communicate()

                        check_status = subprocess.Popen('actual', stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                        stdout_data, stderr_data = check_status.communicate()

                        if stdout_data.startswith('fine, '):
                            turbomole_successful = True
                            mol.turbomole_successful = turbomole_successful
                        else:
                            print('Please check the result!')
                            return
                    else:
                        print('Please check the result!')
                        return
                
                if mol.turbomole_successful:
                    # Get properties
                    # Get energies
                    if calculate_energy:
                        state_energies = ricc2_energy(f'{tmpdirname}/{outfilename}', gs_model)

                        mol_copy = mol.copy()
                        mol_copy.electronic_states = []
                        for _ in range(nstates - len(mol.electronic_states)):
                            mol.electronic_states.append(mol_copy.copy())
                        for i in range(nstates):
                            mol.electronic_states[i].energy = state_energies[i]
                        mol.energy = mol.electronic_states[current_state].energy

                    # Get energy gradients
                    if any(calculate_energy_gradients):
                        state_gradients = ricc2_gradient(fname=f'{tmpdirname}/{outfilename}')

                        if not mol.electronic_states:
                            mol.electronic_states.extend([mol.copy() for _ in range(nstates)])
                        for index, value in enumerate(calculate_energy_gradients):
                            if value:
                                mol.electronic_states[index].add_xyz_derivative_property(np.array(state_gradients[index+1]).astype(float), 'energy', 'energy_gradients')
                        mol.add_xyz_derivative_property(np.array(state_gradients[current_state+1]).astype(float), 'energy', 'energy_gradients')
                else:
                    print('Calculation failed!') 
                    continue   
    
def ricc2_gradient(fname):
    grads = dict()
    # Try to get ground state gradient.
    try:
        cfile = file_utils.go_to_keyword(fname, "GROUND STATE FIRST-ORDER PROPERTIES")[0]
        grads[1] = get_grad_from_stdout(cfile)
    except ValueError:
        pass
    # Try to get excited state gradients.
    try:
        cfile = file_utils.go_to_keyword(fname, "EXCITED STATE PROPERTIES")[0]
    except ValueError:
        return grads
    while True:
        try:
            line = file_utils.search_file(cfile, 
                    "Excited state reached by transition:", 
                    max_res=1, 
                    close=False, 
                    after=3)
            cstate = int(line[0].split()[4]) + 1
        except ValueError:
            cfile.close()
            break
        try:
            file_utils.search_file(cfile, 
                    "cartesian gradient of the energy",
                    max_res=1,
                    stop_at=r"\+={73}\+",
                    close=False)
            grads[cstate] = get_grad_from_stdout(cfile)
        except ValueError:
            pass
    return grads

def ricc2_energy(fname, model):
    search_string = "Final "+re.escape(model.upper())+" energy"
    gs_energy = file_utils.search_file(fname, search_string)
    file_utils.split_columns(gs_energy, col=5, convert=np.float64)
    ex_energy = file_utils.search_file(fname, "Energy:")
    ex_energy = file_utils.split_columns(ex_energy, 1, convert=np.float64)
    energy = np.repeat(gs_energy, len(ex_energy) + 1)
    energy[1:] = energy[1:] + ex_energy
    return energy

def get_grad_from_stdout(cfile):
    grad = file_utils.search_file(cfile, r"^  ATOM", after=3, stop_at=r"resulting FORCE", close=False)
    grad = [line[5:] for line in grad]
    grad = [' '.join(grad[0::3]), ' '.join(grad[1::3]), ' '.join(grad[2::3])]
    grad = [line.split() for line in grad]
    #grad = [list(map(file_utils.fortran_double, vals)) for vals in grad]
    grad = [[file_utils.fortran_double(val) / constants.Bohr2Angstrom for val in vals] for vals in grad]
    return np.array(grad).T

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

if __name__ == '__main__':
    pass
