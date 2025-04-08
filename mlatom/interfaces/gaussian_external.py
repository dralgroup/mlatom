#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gaussian_external: module for running Gaussian for external tasks         ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, sys
import numpy as np
import importlib.util
# ~POD, 2025.03.19
# the complicated import below is required to load the same instance of mlatom,
# which called this external script in the first place.
dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path2init = os.path.join(dir_path, '__init__.py')
dirname = os.path.basename(dir_path)
spec = importlib.util.spec_from_file_location(dirname, path2init)
mlatom4gaussian = importlib.util.module_from_spec(spec)
sys.modules[dirname] = mlatom4gaussian
spec.loader.exec_module(mlatom4gaussian)

def run_gaussian_external(EIn_file, EOu_file, model_predict_kwargs):
    # write new coordinate into 'xyz_temp.dat'
    derivs, molecule = read_gaussian_EIn(EIn_file)
    # calculate energy, gradients, hessian for new coordinates
    # import json
    # with open('model.json', 'r') as fjson:
    #     model_dict = json.load(fjson)
    # if 'method' in model_dict:
    #     kwargs = {}
    #     if 'kwargs' in model_dict:
    #         kwargs = model_dict['kwargs']
    #         del model_dict['kwargs']
    #     model = models.methods(**model_dict, **kwargs)
    model = mlatom4gaussian.models.load('model.json')
    calc_hessian = False
    if derivs == 2: calc_hessian = True
    if 'filename' in model_predict_kwargs:
        if model_predict_kwargs['print_properties'] is not None:
            printstrs = model._predict_geomopt(molecule=molecule, calculate_hessian=calc_hessian, **model_predict_kwargs)
            filename=model_predict_kwargs['filename']
            with open(f'{filename}_tmp_out.out', 'a') as ff:
                ff.writelines(printstrs)
        else:
            model._predict_geomopt(molecule=molecule, calculate_hessian=calc_hessian, **model_predict_kwargs)
    else:
        model.predict(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=calc_hessian, **model_predict_kwargs)
    if not 'energy' in molecule.__dict__:
        raise ValueError('model did not return any energy')
    write_gaussian_EOu(EOu_file, derivs, molecule)
    
    if os.path.exists('gaussian_freq_mol.json'):
        molecule.dump(filename='gaussian_freq_mol.json', format='json')

def read_gaussian_EIn(EIn_file):
    molecule = mlatom4gaussian.data.molecule()
    with open(EIn_file, 'r') as fEIn:
        lines = fEIn.readlines()
        line0 = lines[0].strip().split()
        natoms = int(line0[0]); derivs = int(line0[1])
        molecule.charge = int(line0[2]); molecule.multiplicity = int(line0[3])
        
        for i in range(1, natoms+1):
            xx = lines[i].strip().split()
            atom = mlatom4gaussian.data.atom(atomic_number=int(xx[0]),
                             xyz_coordinates=np.array(xx[1:-1]).astype('float')*mlatom4gaussian.constants.Bohr2Angstrom)
            molecule.atoms.append(atom)
    
    return derivs, molecule

def write_gaussian_EOu(EOu_file, derivs, molecule):
    import fortranformat
    with open(EOu_file, 'w') as fEOu:
        # energy, dipole-moment (xyz)   E, Dip(I), I=1,3
        if 'dipole_moment' in molecule.__dict__.keys():
            dp = molecule.dipole_moment
        else:
            dp = [0.0,0.0,0.0]
        writer = fortranformat.FortranRecordWriter('(4D20.12)')
        output = writer.write([molecule.energy, dp[0], dp[1], dp[2]])
        fEOu.write(output)
        fEOu.write('\n')
        
        writer = fortranformat.FortranRecordWriter('(3D20.12)')
        # gradient on atom (xyz)        FX(J,I), J=1,3; I=1,NAtoms
        output = writer.write(molecule.get_energy_gradients().flatten()*mlatom4gaussian.constants.Bohr2Angstrom)
        fEOu.write(output)
        fEOu.write('\n')
        if derivs == 2:
            natoms = len(molecule.atoms)
            # polarizability                Polar(I), I=1,6
            polor = np.zeros(6)
            output = writer.write(polor)
            fEOu.write(output)
            fEOu.write('\n')
            # dipole derivatives            DDip(I), I=1,9*NAtoms
            ddip = np.zeros(9*natoms)
            if 'dipole_derivatives' in molecule.__dict__.keys():
                ddip = molecule.dipole_derivatives * mlatom4gaussian.constants.Bohr2Angstrom * mlatom4gaussian.constants.Debye2au 
            output = writer.write(ddip)
            fEOu.write(output)
            fEOu.write('\n')
            # force constants               FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2
            output = writer.write(molecule.hessian[np.tril_indices(natoms*3)]*mlatom4gaussian.constants.Bohr2Angstrom**2)
            fEOu.write(output)

if __name__ == '__main__': 
    model_predict_kwargs_str_file, _, EIn_file, EOu_file, _, _, _ = sys.argv[1:]
    with open(model_predict_kwargs_str_file) as f:
        model_predict_kwargs =  eval(f.read())
    run_gaussian_external(EIn_file, EOu_file, model_predict_kwargs)