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
from .. import constants, simulations, models
from ..decorators import doc_inherit 

class sparrow_methods(models.OMP_model):
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
    availability_of_gradients_for_methods = {
        'DFTB0': True, 'DFTB2': True, 'DFTB3': True,
        'MNDO': True, 'MNDO/d': True, 'AM1': True, 'RM1': True, 'PM3': True, 'PM6': True,
        'OM2': False, 'OM3': False, 'ODM2*': False, 'ODM3*': False, 
        'AIQM1': False}
    available_methods = models.methods.methods_map['sparrow'] #need to sync with dict availability_of_gradients_for_methods somehow
    
    def __init__(self, method='ODM2*', read_keywords_from_file='', save_files_in_current_directory=False, working_directory=None, **kwargs):
        self.method = method
        self.read_keywords_from_file = read_keywords_from_file
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        try:
            self.sparrowbin = os.environ['sparrowbin']
        except:
            msg = 'Cannot find the Sparrow program, please set the environment variable: export sparrowbin=...'
            raise ValueError(msg)
    
    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
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
                            '-o']
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
                        gradients = np.loadtxt(f'{tmpdirname}/gradients.dat', comments='#') / constants.Bohr2Angstrom 
                    else:
                        save_files_in_current_directory = self.save_files_in_current_directory
                        self.save_files_in_current_directory = False
                        gradients = simulations.numerical_gradients(mol, self, 1e-5, kwargs_funtion_predict_energy = {'calculate_energy_gradients': False, 'calculate_hessian': False})
                        self.save_files_in_current_directory = save_files_in_current_directory
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].energy_gradients = gradients[iatom]
                if calculate_hessian:
                    if self.availability_of_gradients_for_methods[self.method]:
                        mol.hessian = np.loadtxt(f'{tmpdirname}/hessian.dat', comments='#') / (constants.Bohr2Angstrom**2)
                    else:
                        save_files_in_current_directory = self.save_files_in_current_directory
                        self.save_files_in_current_directory = False
                        mol.hessian = simulations.numerical_hessian(mol, self, 5.29167e-4, 1e-5, kwargs_funtion_predict_energy = {'calculate_energy_gradients': False, 'calculate_hessian': False})
                        self.save_files_in_current_directory = save_files_in_current_directory

if __name__ == '__main__':
    pass
