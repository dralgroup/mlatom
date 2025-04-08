#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! sparrow: interface to the Sparrow program                                 ! 
  ! Implementations by: Pavlo O. Dral & Peikun Zheng                          !
  !---------------------------------------------------------------------------! 

  '''

import os
import numpy as np
from .. import constants, simulations
from ..model_cls import OMP_model, method_model
from ..decorators import doc_inherit 

class sparrow_methods(OMP_model, method_model):
    '''
    Sparrow interface

    Arguments:
        method (str): method to use
        read_keywords_from_file (str): keywords used in Sparrow
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'False'``
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``

    .. note::

        Methods supported:

        Energy: DFTB0, DFTB2, DFTB3
        MNDO, MNDO/d, AM1, PM3, PM6, RM1,
        OM2, OM3, ODM2*, ODM3* 
        AIQM1

        Gradients: DFTB0, DFTB2, DFTB3
        MNDO, MNDO/d, AM1, PM3, PM6, RM1
    
    '''
    bin_env_name = 'sparrowbin'
    supported_methods = ['DFTB0', 'DFTB2', 'DFTB3', 'MNDO', 'MNDO/d', 'AM1', 'RM1', 'PM3', 'PM6', 'OM2', 'OM3', 'ODM2*', 'ODM3*', 'AIQM1']
    availability_of_gradients_for_methods = {
        'DFTB0': True, 'DFTB2': True, 'DFTB3': True,
        'MNDO': True, 'MNDO/d': True, 'AM1': True, 'RM1': True, 'PM3': True, 'PM6': True,
        'OM2': False, 'OM3': False, 'ODM2*': False, 'ODM3*': False, 
        'AIQM1': False}
    
    def __init__(self, method='ODM2*', read_keywords_from_file='', save_files_in_current_directory=False, working_directory=None):
        self.method = method
        self.read_keywords_from_file = read_keywords_from_file
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.sparrowbin = self.get_bin_env_var()
        if self.sparrowbin is None:
            raise ValueError('Cannot find the Sparrow program, please set the environment variable: export sparrowbin=...')
        if method.casefold() == 'odm2*': print(' !WARNING! ODM2* calculations will be performed with Sparrow which has no implementation of analytical gradients and excited-state property calculations with this Hamiltonian. If you have the MNDO program you might want to use it for such calculations. Alternatively, choose a newer AIQM-series methods such as AIQM2 that is not based on ODM2* but on GFN2-xTB. MNDO is not available on the XACS cloud.')
    
    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False,
                **kwargs):
        allowed_kwargs = {'nstates': 1, 'current_state': 0,
                          'calculate_dipole_derivatives': False,
                        'calculate_nacv': False, 'read_density_matrix': False}
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs.keys():
                raise ValueError(f"keyworded argument '{kwarg}={kwargs[kwarg]}' is not allowed in Sparrow interface")
            elif kwargs[kwarg] != allowed_kwargs[kwarg]:
                raise ValueError(f"keyworded argument '{kwarg}={kwargs[kwarg]}' is not allowed in Sparrow interface, only '{kwarg}={allowed_kwargs[kwarg]}' is allowed. You might want to use the MNDO interface.")
            
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        # Not very good method naming in Sparrow...
        # ODM2 is not implemented, ODM2 is just ODM2*, same for ODM3/ODM3*
        if self.method == 'ODM2*': method_to_pass = 'ODM2'
        elif self.method == 'ODM3*': method_to_pass = 'ODM3'
        else: method_to_pass = self.method
        
        additional_sparrow_keywords = []
        if self.read_keywords_from_file != '':
            kw_file = self.read_keywords_from_file
            with open(kw_file, 'r') as fkw:
                for line in fkw:
                    additional_sparrow_keywords = line.split()
                    joined_args = ''.join(additional_sparrow_keywords)
                    if 'iop' in joined_args or 'job' in joined_args:
                        raise ValueError('Sparrow does not support mndo keywords. If you have the MNDO program you might want to use it for such calculations. Alternatively, choose a newer AIQM-series methods that are not based on ODM2* but on GFN2-xTB. MNDO is not available on the XACS cloud.')
            imol = -1
            for mol in molDB.molecules:
                imol += 1
                jmol = imol
                if len(additional_sparrow_keywords) < imol+1: jmol = -1
                mol.additional_sparrow_keywords = additional_sparrow_keywords[jmol]
        
        import tempfile, subprocess
        ii = 0
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:  
                if self.save_files_in_current_directory: tmpdirname = '.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                    if not os.path.exists(tmpdirname):
                        os.makedirs(tmpdirname)
                    tmpdirname = os.path.abspath(tmpdirname)
                ii += 1
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                sparrowargs = [self.sparrowbin,
                            '-x', xyzfilename,
                            '-c', '%d' % mol.charge,
                            '-s','%d' % mol.multiplicity,
                            '-M', method_to_pass,
                            '-o',]
                if mol.multiplicity != 1:
                    sparrowargs.append('-u')
                sparrowargs += additional_sparrow_keywords
                
                if calculate_energy_gradients and self.availability_of_gradients_for_methods[self.method]:
                    sparrowargs += ['-G']
                if calculate_hessian and self.availability_of_gradients_for_methods[self.method]:
                    sparrowargs += ['-H']
                
                #cmd = ' '.join(sparrowargs)
                proc = subprocess.Popen(sparrowargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                #proc.wait()
                outs,errs = proc.communicate()
                #os.system(cmd + " &> sparrow.out")
                mol.sparrow_scf_successful = False
                # for readable in proc.stdout:
                #     if 'SCF converged!' in readable:
                #         mol.sparrow_scf_successful = True
                if 'SCF converged!' in outs+errs:
                    mol.sparrow_scf_successful = True
                if not mol.sparrow_scf_successful:
                    return
                
                if calculate_energy:
                    mol.energy = np.loadtxt(f'{tmpdirname}/energy.dat', comments='#').tolist()
                if calculate_energy_gradients:
                    if self.availability_of_gradients_for_methods[self.method]:
                        mol.energy_gradients = np.loadtxt(f'{tmpdirname}/gradients.dat', comments='#') / constants.Bohr2Angstrom
                    else:
                        save_files_in_current_directory = self.save_files_in_current_directory
                        self.save_files_in_current_directory = False
                        working_directory = self.working_directory
                        self.working_directory = None
                        _ = simulations.numerical_gradients(mol, self, 1e-5, model_kwargs = {'calculate_energy_gradients': False, 'calculate_hessian': False})
                        self.save_files_in_current_directory = save_files_in_current_directory
                        self.working_directory = working_directory
                if calculate_hessian:
                    if self.availability_of_gradients_for_methods[self.method]:
                        mol.hessian = np.loadtxt(f'{tmpdirname}/hessian.dat', comments='#') / (constants.Bohr2Angstrom**2)
                    else:
                        save_files_in_current_directory = self.save_files_in_current_directory
                        self.save_files_in_current_directory = False
                        working_directory = self.working_directory
                        self.working_directory = None
                        _ = simulations.numerical_hessian(mol, self, 5.29167e-4, 1e-5, model_kwargs = {'calculate_energy_gradients': False, 'calculate_hessian': False})
                        self.save_files_in_current_directory = save_files_in_current_directory
                        self.working_directory = working_directory

if __name__ == '__main__':
    pass
