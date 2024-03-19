#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! xtb: interface to the xtb program                                         ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
from .. import constants, models
from ..decorators import doc_inherit

class xtb_methods(models.OMP_model):
    '''
    xTB interface

    Arguments:
        method (str): xTB methods
        read_keywords_from_file (str): keywords used in xTB

    .. note::

        Only GFN2-xTB is available. 

    Examples:

    .. code-block::

        from ml.interfaces.xtb import xtb_methods()

        # read molecule from xyz file
        mol = ml.data.molecule()
        mol.read_from_xyz_file('sp.xyz')

        # initialize xtb methods
        model = xtb_methods(method='GFN2-xTB)

        # calculate energy, gradients and hessian
        model.predict(molecule=mol, 
                    calculate_energy_gradients=True, 
                    calculate_hessian=True)
        print(mol.energy)

    '''
    available_methods = models.methods.methods_map['xtb']
    
    def __init__(self, method='GFN2-xTB', read_keywords_from_file='', **kwargs):
        self.method = method
        self.read_keywords_from_file = read_keywords_from_file
        try:
            self.xtbbin = os.environ['xtb']
        except:
            msg = 'Cannot find the xtb program, please set the environment variable: export xtb=...'
            raise ValueError(msg)
        
        if 'nthreads' in kwargs:
            self.nthreads = kwargs['nthreads']
        else:
            self.nthreads = 1
        if 'stacksize' in kwargs:
            os.environ["OMP_STACKSIZE"] = kwargs['stacksize']
        
    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        additional_xtb_keywords = []
        if self.read_keywords_from_file != '':
            kw_file = self.read_keywords_from_file
            with open(kw_file, 'r') as fxtbkw:
                for line in fxtbkw:
                    additional_xtb_keywords = line.split()
            imol = -1
            for mol in molDB.molecules:
                imol += 1
                jmol = imol
                if len(additional_xtb_keywords) < imol+1: jmol = -1
                mol.additional_xtb_keywords = additional_xtb_keywords[jmol]
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdirname:
            ii = 0
            for mol in molDB.molecules:
                ii += 1
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                
                xtbargs = [self.xtbbin, xyzfilename]
                if mol.charge != 0: xtbargs += ['-c', '%d' % mol.charge] # there is a bug in xtb - it does not read --charg
                number_of_unpaired_electrons = mol.multiplicity - 1
                xtbargs += ['-u', '%d' % number_of_unpaired_electrons] # there is a bug in xtb - it does not read --uhf
                # with open(f'{tmpdirname}/.CHRG', 'w') as fcharge, open(f'{tmpdirname}/.UHF', 'w') as fuhf:
                #     fcharge.writelines(f'{mol.charge}\n')
                #     fuhf.writelines(f'{number_of_unpaired_electrons}\n')
                outputs = []
                if calculate_energy and not calculate_energy_gradients:
                    proc = subprocess.Popen(xtbargs + additional_xtb_keywords, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                    outs,errs = proc.communicate()
                    stdout = outs.split('\n')
                    stderr = errs.split('\n')
                    xtb_scf_successful = False
                    for readable in stderr:
                        if 'normal termination of xtb' in readable:
                            xtb_scf_successful = True
                    mol.__dict__['xtb_scf_successful'] = xtb_scf_successful
                    if xtb_scf_successful:
                        for readable in stdout:
                            outputs.append(readable)
                            if 'TOTAL ENERGY' in readable:
                                energy = float(readable.split()[3])
                                mol.energy = energy
                            if 'dispersion' in readable:
                                dispersion = float(readable.split()[3])
                                mol.__dict__['D4_dispersion_from_GFN2xTB'] = dispersion

                        for iline in range(len(outputs)):
                            if 'molecular dipole:' in outputs[iline]:
                                sum_line = outputs[iline+3].strip().split()[1:]
                                dipole_moment = [eval(each) for each in sum_line]
                                mol.dipole_moment = dipole_moment

                
                if calculate_energy_gradients:
                    proc = subprocess.Popen(xtbargs + ['--grad'] + additional_xtb_keywords, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                    outs,errs = proc.communicate()
                    stdout = outs.split('\n')
                    stderr = errs.split('\n')
                    xtb_scf_successful = False
                    xtb_grad_successful = False
                    for readable in stderr:
                        if 'normal termination of xtb' in readable:
                            xtb_scf_successful = True
                            xtb_grad_successful = True
                    mol.__dict__['xtb_scf_successful'] = xtb_scf_successful
                    mol.__dict__['xtb_grad_successful'] = xtb_grad_successful
                    if xtb_grad_successful:
                        for readable in stdout:
                            outputs.append(readable)
                            if 'TOTAL ENERGY' in readable:
                                energy = float(readable.split()[3])
                                mol.energy = energy
                            if 'dispersion' in readable:
                                dispersion = float(readable.split()[3])
                                mol.__dict__['D4_dispersion_from_GFN2xTB'] = dispersion
                        with open(f'{tmpdirname}/gradient', 'r') as fout:
                            iatom = -1
                            for line in fout:
                                if len(line.split()) != 3: continue
                                iatom += 1
                                mol.atoms[iatom].energy_gradients = np.array([float(xx) / constants.Bohr2Angstrom for xx in line.split()]).astype(float)

                        for iline in range(len(outputs)):
                            if 'molecular dipole:' in outputs[iline]:
                                sum_line = outputs[iline+3].strip().split()[1:]
                                dipole_moment = [eval(each) for each in sum_line]
                                mol.dipole_moment = dipole_moment


                if calculate_hessian:
                    natoms = len(mol.atoms)
                    proc = subprocess.Popen(xtbargs + ['--hess'] + additional_xtb_keywords, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                    outs,errs = proc.communicate()
                    stdout = outs.split('\n')
                    stderr = errs.split('\n')
                    xtb_hessian_successful = False
                    for readable in stderr:
                        if 'normal termination of xtb' in readable:
                            xtb_hessian_successful = True
                    mol.__dict__['xtb_hessian_successful'] = xtb_hessian_successful
                    if xtb_hessian_successful or os.path.exists(f'{tmpdirname}/hessian'): # There is bug in xtb that it fails for H2, thus, it may end with error but still prints hessian.
                        with open(f'{tmpdirname}/hessian', 'r') as fout:
                            nlines = 0
                            hess = []
                            for line in fout:
                                nlines += 1
                                if nlines == 1: continue
                                for xx in line.split():
                                    hess.append(float(xx) / (constants.Bohr2Angstrom**2))
                            hess = np.array(hess).astype(float)
                            mol.hessian = hess.reshape(natoms*3,natoms*3)

if __name__ == '__main__':
    pass