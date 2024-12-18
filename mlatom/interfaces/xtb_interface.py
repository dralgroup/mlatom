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
import sys
import subprocess
from .. import constants, models
from ..decorators import doc_inherit

class xtb_methods(models.OMP_model, metaclass=models.meta_method):
    '''
    xTB interface

    Arguments:
        method (str): xTB methods
        read_keywords_from_file (str): keywords used in xTB

    .. note::

        GFN2-xTB and GFN2-xTB* (remove D4 from GFN2-xTB) are available. 

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
    
    def __init__(self, method='GFN2-xTB', read_keywords_from_file='', nthreads=None, **kwargs):
        self.method = method
        if self.method.lower() == 'GFN2-xTB*'.lower():
            self.without_d4 = True
        else:
            self.without_d4 = False
        self.read_keywords_from_file = read_keywords_from_file
        if 'solvent' in kwargs:
            self.solvent = kwargs['solvent']
        else:
            self.solvent = None

        try:
            self.xtbbin = os.environ['xtb']
        except:
            self.xtbbin = "%s/xtb" % os.path.dirname(__file__)

        if nthreads is None:
            from multiprocessing import cpu_count
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads
        if 'stacksize' in kwargs:
            os.environ["OMP_STACKSIZE"] = kwargs['stacksize']

        if 'save_files_in_current_directory' in kwargs:
            self.save_files_in_current_directory = kwargs['save_files_in_current_directory']
        else:
            self.save_files_in_current_directory = False
        
    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, calculate_dipole_derivatives=False,calculate_polarizability_derivatives=False):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        if calculate_dipole_derivatives or calculate_polarizability_derivatives or self.without_d4:
            self.xtbbin = "%s/xtb" % os.path.dirname(__file__)
        
        self.additional_xtb_keywords = []
        if self.read_keywords_from_file != '':
            kw_file = self.read_keywords_from_file
            with open(kw_file, 'r') as fxtbkw:
                for line in fxtbkw:
                    self.additional_xtb_keywords = line.split()
            imol = -1
            for mol in molDB.molecules:
                imol += 1
                jmol = imol
                if len(self.additional_xtb_keywords) < imol+1: jmol = -1
                mol.additional_xtb_keywords = self.additional_xtb_keywords[jmol]

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory:
                tmpdirname = '.'
            self.tmpdirname = os.path.abspath(tmpdirname)
            for ii, mol in enumerate(molDB.molecules):
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                
                self.xtbargs = [self.xtbbin, xyzfilename]
                if mol.charge != 0: self.xtbargs += ['-c', '%d' % mol.charge] # there is a bug in xtb - it does not read --charg
                number_of_unpaired_electrons = mol.multiplicity - 1
                self.xtbargs += ['-u', '%d' % number_of_unpaired_electrons] # there is a bug in xtb - it does not read --uhf
                # with open(f'{tmpdirname}/.CHRG', 'w') as fcharge, open(f'{tmpdirname}/.UHF', 'w') as fuhf:
                #     fcharge.writelines(f'{mol.charge}\n')
                #     fuhf.writelines(f'{number_of_unpaired_electrons}\n')
                if self.without_d4:
                    self.xtbargs += ['--withoutd4']

                if self.solvent:
                    self.xtbargs += ['--alpb', self.solvent]
                
                if calculate_energy and not calculate_energy_gradients and not calculate_hessian:
                    self.predict_energy(mol)

                if calculate_energy_gradients:
                   self.predict_energy_gradients(mol)

                if calculate_hessian:
                    self.predict_hessian(mol,calculate_polarizability_derivatives, calculate_dipole_derivatives)

    def predict_energy(
            self,
            mol,
    ):
        terminated = False
        outputs = []
        while not terminated:
            stdout, stderr = self.run_xtb_job(self.xtbargs+self.additional_xtb_keywords,self.tmpdirname)
            rerun, terminated = self.error_handle(stdout+stderr)
        xtb_scf_successful = not rerun
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
                if 'Gsolv' in readable:
                    solvation_free_energy = float(readable.split()[3])
                    mol.solvation_free_energy = solvation_free_energy

            for iline in range(len(outputs)):
                if 'molecular dipole:' in outputs[iline]:
                    sum_line = outputs[iline+3].strip().split()[1:]
                    dipole_moment = [eval(each) for each in sum_line]
                    mol.dipole_moment = dipole_moment
        else:
            print('xTB calculation failed.')
            sys.stdout.flush()

    def predict_energy_gradients(
            self,
            mol,
    ):
        terminated = False
        outputs = []
        while not terminated:
            stdout, stderr = self.run_xtb_job(self.xtbargs + ['--grad'] + self.additional_xtb_keywords, self.tmpdirname)
            rerun, terminated = self.error_handle(stdout+stderr)
        xtb_scf_successful = not rerun
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
                if 'Gsolv' in readable:
                    solvation_free_energy = float(readable.split()[3])
                    mol.solvation_free_energy = solvation_free_energy
            with open(f'{self.tmpdirname}/gradient', 'r') as fout:
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
        else:
            print('xTB calculation failed.')
            sys.stdout.flush()
                        
    def predict_hessian(
            self, 
            mol, 
            calculate_polarizability_derivatives,
            calculate_dipole_derivatives
    ):
        terminated = False
        outputs = []
        natoms = len(mol.atoms)
        while not terminated:
            if not calculate_polarizability_derivatives:
                stdout, stderr = self.run_xtb_job(self.xtbargs + ['--hess'] + self.additional_xtb_keywords, self.tmpdirname)
            else:
                stdout, stderr = self.run_xtb_job(self.xtbargs + ['--hess','--ptb','--alpha'] + self.additional_xtb_keywords, self.tmpdirname)
            rerun, terminated = self.error_handle(stdout+stderr)

        xtb_scf_successful = not rerun
        mol.__dict__['xtb_scf_successful'] = xtb_scf_successful

        # There is bug in xtb that it fails for H2, thus, it may end with error but still prints hessian.
        if xtb_scf_successful or os.path.exists(f'{self.tmpdirname}/hessian'): 
            for readable in stdout:
                outputs.append(readable)
                if 'TOTAL ENERGY' in readable:
                    energy = float(readable.split()[3])
                    mol.energy = energy
                if 'dispersion' in readable:
                    dispersion = float(readable.split()[3])
                    mol.__dict__['D4_dispersion_from_GFN2xTB'] = dispersion
                if 'Gsolv' in readable:
                    solvation_free_energy = float(readable.split()[3])
                    mol.solvation_free_energy = solvation_free_energy

            with open(f'{self.tmpdirname}/hessian', 'r') as fout:
                nlines = 0
                hess = []
                for line in fout:
                    nlines += 1
                    if nlines == 1: continue
                    for xx in line.split():
                        hess.append(float(xx) / (constants.Bohr2Angstrom**2))
                hess = np.array(hess).astype(float)
                mol.hessian = hess.reshape(natoms*3,natoms*3)
            # Read dipole derivatives
            if calculate_dipole_derivatives:
                if os.path.exists(f'{self.tmpdirname}/dipd'):
                    with open(f'{self.tmpdirname}/dipd', 'r') as fout:
                        dipd = []
                        for iline, line in enumerate(fout.readlines()):
                            if iline == 0: continue 
                            dipd += [float(each) for each in line.split()]
                        dipd = np.array(dipd).astype(float)
                        mol.dipole_derivatives = dipd / constants.Debye
            # Read polarizability derivatives 
            if calculate_polarizability_derivatives:
                if os.path.exists(f'{self.tmpdirname}/polard'):
                    with open(f'{self.tmpdirname}/polard','r') as fout:
                        polard = [] 
                        for iline,line in enumerate(fout.readlines()):
                            if iline == 0: continue 
                            polard += [float(each) for each in line.split()]
                            # for ii in range(6):
                            #     # polard += [float(each) for each in line.split()]
                            #     polard.append(float(line[5+15*ii:5+15*(ii+1)]))
                        polard = np.array(polard).astype(float)
                        mol.polarizability_derivatives = polard
        else:
            print('xTB calculation failed.')
            sys.stdout.flush()

    def run_xtb_job(self,xtbargs,tmpdirname):
        segmentation_error = False
        floating_point_error = False
        other_error = None
        stderr = []; stdout = []
        try:
            proc = subprocess.run(xtbargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, check=True)
            stdout = proc.stdout.split('\n')
            stderr = proc.stderr.split('\n')
        except subprocess.CalledProcessError as e:
            if e.returncode == -11: # kill signals: SIGSEGV
                segmentation_error = True
            elif e.returncode == -6:
                floating_point_error = True
            elif e.returncode == 1:
                # not errors of system signal
                proc = subprocess.Popen(xtbargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                stdout, stderr = proc.communicate()
                stdout = stdout.split('\n')
                stderr = stderr.split('\n')
            else:
                other_error = e.returncode

        if segmentation_error:
            stderr.append('segmentation fault')
        if floating_point_error:
            stderr.append('floating-point exceptions')
        if other_error:
            stderr.append(f'killed signal {other_error}')
        return stdout, stderr

    def error_handle(self,stds):
        terminated = True
        rerun = True
        for readable in stds:
            if 'normal termination of xtb' in readable:
                terminated = True
                rerun = False
                return rerun, terminated
            if 'segmentation fault' in readable.lower(): # memory problem
                if "OMP_STACKSIZE" not in os.environ:
                    print('OMP stacksize is too small for the system. Try to increase to 1G')
                    sys.stdout.flush()
                    os.environ["OMP_STACKSIZE"] = '1G'
                    rerun = True
                    terminated = False
                    return rerun, terminated 
                else:
                    raise ValueError('OMP stacksize is too small for the system.')
            if 'solv_model_loadInternalParam' in readable:
                raise ValueError(f'Please specify correct solvent name for xTB calculation.')
            if 'floating-point exceptions' in readable.lower():
                print('xTB calculation WARNING: Floating-point exceptions are signalling')
                sys.stdout.flush()
                continue
            if 'killed signal' in readable.lower():
                raise ValueError(f'xTB calculation errors terminated by the system with {readable}')
        
        # convergence error not implemented
        return rerun, terminated

if __name__ == '__main__':
    pass