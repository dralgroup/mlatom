#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! sparrow: interface to the Sparrow program                                 ! 
  ! Implementations by: Pavlo O. Dral & Peikun Zheng                          !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
pythonpackage = True
try: from .. import constants, data, simulations, stopper
except:
    import constants, data, simulations, stopper
    pythonpackage = False        

class sparrow_methods():
    availability_of_gradients_for_methods = {
        'DFTB0': True, 'DFTB2': True, 'DFTB3': True,
        'MNDO': True, 'MNDO/d': True, 'AM1': True, 'RM1': True, 'PM3': True, 'PM6': True,
        'OM2': False, 'OM3': False, 'ODM2*': False, 'ODM3*': False, 
        'AIQM1': False}
    available_methods = list(availability_of_gradients_for_methods.keys())
    
    def __init__(self, method='ODM2*'):
        self.method = method
        try:
            self.sparrowbin = os.environ['sparrowbin']
        except:
            msg = 'Cannot find the Sparrow program, please set the environment variable: export sparrowbin=...'
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
            
        # Not very good method naming in Sparrow...
        # ODM2 is not implemented, ODM2 is just ODM2*, same for ODM3/ODM3*
        if self.method == 'ODM2*': method_to_pass = 'ODM2'
        elif self.method == 'ODM3*': method_to_pass = 'ODM3'
        else: method_to_pass = self.method
            
        import tempfile, subprocess
        ii = 0
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:  
                ii += 1
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                
                sparrowargs = [self.sparrowbin,
                            '-x', xyzfilename,
                            '-c', '%d' % mol.charge,
                            '-s','%d' % mol.multiplicity,
                            '-M', method_to_pass,
                            '-o']
                
                if calculate_energy_gradients and self.availability_of_gradients_for_methods[self.method]:
                    sparrowargs += ['-G']
                if calculate_hessian and self.availability_of_gradients_for_methods[self.method]:
                    sparrowargs += ['-H']
                
                #cmd = ' '.join(sparrowargs)
                proc = subprocess.Popen(sparrowargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                #proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                proc.wait()
                #os.system(cmd + " &> sparrow.out")
                mol.sparrow_scf_successful = False
                for readable in proc.stdout:
                    if 'SCF converged!' in readable:
                        mol.sparrow_scf_successful = True
                if not mol.sparrow_scf_successful:
                    return
                
                if calculate_energy:
                    mol.energy = np.loadtxt(f'{tmpdirname}/energy.dat', comments='#').tolist()
                if calculate_energy_gradients:
                    if self.availability_of_gradients_for_methods[self.method]:
                        gradients = np.loadtxt(f'{tmpdirname}/gradients.dat', comments='#') / constants.Bohr2Angstrom 
                    else:
                        gradients = simulations.numerical_grad(mol, self, 1e-5, kwargs_funtion_predict_energy = {'calculate_energy_gradients': False, 'calculate_hessian': False})
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].energy_gradients = gradients[iatom]
                if calculate_hessian:
                    if self.availability_of_gradients_for_methods[self.method]:
                        mol.hessian = np.loadtxt(f'{tmpdirname}/hessian.dat', comments='#') / (constants.Bohr2Angstrom**2)
                    else:
                        mol.hessian = simulations.numerical_hessian(mol, self, 5.29167e-4, 1e-5, kwargs_funtion_predict_energy = {'calculate_energy_gradients': False, 'calculate_hessian': False})

if __name__ == '__main__':
    pass