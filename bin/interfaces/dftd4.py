#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! mndo: interface to the MNDO program                                       ! 
  ! Implementations by: Pavlo O. Dral & Peikun Zheng                          !
  !---------------------------------------------------------------------------! 
'''
import json
import numpy as np

pythonpackage = True
try:
    from .. import data
    from .. import stopper
except:
    import data
    import stopper
    pythonpackage = False

class dftd4_methods():
    def __init__(self, functional=None):
        self.functional = functional
    
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
    
        import os
        try: dftd4bin = os.environ['dftd4bin']
        except:
            if pythonpackage: raise ValueError('Cannot find the dftd4bin program, please set the environment variable: export dftd4bin=...')
            else: stopper.stopMLatom('Cannot find the dftd4bin program, please set the environment variable: export dftd4bin=...')
        
        if calculate_energy_gradients or calculate_hessian:
            try:
                from .. import constants
            except:
                import constants
            
        import tempfile, subprocess        
        ii = 0
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:
                ii += 1
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                
                dftd4args = [dftd4bin, xyzfilename, '-f', '%s' % self.functional, '-c', '%d' % mol.charge]
                
                if calculate_hessian:
                    dftd4args += ['-s', '--grad', '--hess', '--orca', '--json']
                elif calculate_energy_gradients:
                    dftd4args += ['-s', '--grad', '--orca', '--json']
                elif calculate_energy:
                    dftd4args += ['-s', '--orca', '--json']
                
                proc = subprocess.Popen(dftd4args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                proc.wait()
                dftd4_successful = False
                for readable in proc.stderr:
                    if 'normal termination of dftd4' in readable:
                        dftd4_successful = True
                mol.dftd4_successful = dftd4_successful
                with open(f'{tmpdirname}/dftd4.json', 'r') as f:
                    d4_results = json.load(f)
                
                if calculate_energy:
                    energy = float(d4_results['energy'])
                    mol.energy = energy
                if calculate_energy_gradients:
                    grad = np.array(d4_results['gradient']) / constants.Bohr2Angstrom
                    grad = grad.reshape(-1, 3)
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].energy_gradients = grad[iatom]
                if calculate_hessian:
                    natoms = len(mol.atoms)
                    hess = np.array(d4_results['hessian']) / (constants.Bohr2Angstrom**2)
                    mol.hessian = hess.reshape(natoms*3,natoms*3)

if __name__ == '__main__':
    pass