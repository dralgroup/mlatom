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
from .. import constants
from ..model_cls import OMP_model, method_model
from ..decorators import doc_inherit

class xtb_methods(OMP_model, method_model):
    '''
    xTB interface

    Arguments:
        method (str): xTB methods
        read_keywords_from_file (str): the file that contain keywords used in xTB
        nthreads (int): number of threads used in xTB. It will be passed to OMP_NUM_THREADS.
        solvent (str): the implicit solvent model.
        stack_size (str): the stack size to be use for each threads, e.g., '100m', '1G'. It will be passed to OMP_STACKSIZE
        etemp (float): the electronic temperature. By default is 300K
        acc (float): the threshold for SCF convergence. By default is 1 (1e-5)
        unlimited_stack_size (bool): equal to `ulimit -c unlimited` in bash
        restart (bool): whether to reuse previous calculation files. By default turned off
        save_files_in_current_directory (bool): whether to save files in current directory. By default False
        working_directory (str): the path to save calculation files. By default is None.

    .. note::

        GFN2-xTB and GFN2-xTB* (remove D4 from GFN2-xTB) are available. 

    Examples:

    .. code-block::

        from mlatom import xtb_methods

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
    supported_methods = ['GFN2-xTB', 'GFN2-xTB*']
    
    def __init__(self, 
                 method: str = 'GFN2-xTB', 
                 read_keywords_from_file: str = '', 
                 nthreads: int = 1, 
                 solvent: str = None,
                 stack_size: str = None, 
                 etemp = None,
                 acc = None,
                 unlimited_stack_size: bool = False,
                 restart: bool = False,
                 save_files_in_current_directory: bool = False,
                 working_directory: str = None,
                 ):
        
        self.method = method
        if self.method.lower() == 'GFN2-xTB*'.lower():
            self.without_d4 = True
        else:
            self.without_d4 = False
        self.read_keywords_from_file = read_keywords_from_file
        self.solvent = solvent
        self.etemp = etemp
        self.acc = acc
        self.stack_size = stack_size
        self.unlimited_stack_size = unlimited_stack_size
        self.restart = restart
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.nthreads = nthreads

        #try:
        #    self.xtbbin = os.environ['xtb']
        #except:
        # we provide a pre-compiled xtb binary with more features
        self.xtbbin = "%s/xtb" % os.path.dirname(__file__)
        if self.stack_size:
            os.environ["OMP_STACKSIZE"] = self.stack_size

    @classmethod
    def is_program_found(cls):
        return True        

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

        # unlimit system stack for xtb
        if self.unlimited_stack_size:
            import resource
            stack_info = resource.getrlimit(resource.RLIMIT_STACK)
            if stack_info[0] != -1:
                print(f'WARNING: The current limit of stack size is {stack_info[0]} kbytes. Trying to unlimit the stack size ...')
                sys.stdout.flush()
                resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory: tmpdirname = '.'
            if self.working_directory is not None:
                tmpdirname = self.working_directory
                if not os.path.exists(tmpdirname):
                    os.makedirs(tmpdirname)
            self.tmpdirname = os.path.abspath(tmpdirname)
            for ii, mol in enumerate(molDB.molecules):
                xyzfilename = f'{self.tmpdirname}/predict{ii+1}.xyz'
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
                
                if self.etemp is not None:
                    self.xtbargs += ['--etemp', str(self.etemp)]
                
                if self.acc is not None:
                    self.xtbargs += ['--acc', str(self.acc)]

                if not self.restart:
                    self.xtbargs += ['--norestart']

                # rename all temporary files
                self.xtbargs += ['--namespace', f'molecule{ii}']
                
                if calculate_energy and not calculate_energy_gradients and not calculate_hessian:
                    self.predict_energy(mol)

                if calculate_energy_gradients:
                   self.predict_energy_gradients(mol, ii)

                if calculate_hessian:
                    self.predict_hessian(mol,ii,calculate_polarizability_derivatives, calculate_dipole_derivatives)

    def predict_energy(
            self,
            mol,
    ):
        xtb_scf_successful = False
        outputs = []
        stdout, stderr = self.run_xtb_job(self.xtbargs+self.additional_xtb_keywords,self.tmpdirname)
        try:
            xtb_scf_successful = self.error_handle(stdout+stderr)
        except:
            raise
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
                    for ii in range(3):
                        dipole_moment[ii] = dipole_moment[ii] / constants.Debye2au
                    mol.dipole_moment = dipole_moment
        else:
            print('xTB calculation failed with unhandled error in MLatom.')
            sys.stdout.flush()

    def predict_energy_gradients(
            self,
            mol,
            ii
    ):
        xtb_scf_successful = False
        outputs = []
        stdout, stderr = self.run_xtb_job(self.xtbargs + ['--grad'] + self.additional_xtb_keywords, self.tmpdirname)
        try:
            xtb_scf_successful = self.error_handle(stdout+stderr)
        except:
            raise
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
            with open(f'{self.tmpdirname}/molecule{ii}.gradient', 'r') as fout:
                iatom = -1
                for line in fout:
                    if len(line.split()) != 3: continue
                    iatom += 1
                    mol.atoms[iatom].energy_gradients = np.array([float(xx) / constants.Bohr2Angstrom for xx in line.split()]).astype(float)

            for iline in range(len(outputs)):
                if 'molecular dipole:' in outputs[iline]:
                    sum_line = outputs[iline+3].strip().split()[1:]
                    dipole_moment = [eval(each) for each in sum_line]
                    for ii in range(3):
                        dipole_moment[ii] = dipole_moment[ii] / constants.Debye2au
                    mol.dipole_moment = dipole_moment
        else:
            print('xTB calculation failed with unhandled error in MLatom.')
            sys.stdout.flush()
                        
    def predict_hessian(
            self, 
            mol, 
            ii,
            calculate_polarizability_derivatives,
            calculate_dipole_derivatives
    ):

        outputs = []
        natoms = len(mol.atoms)
        xtb_scf_successful = False
        if not calculate_polarizability_derivatives:
            stdout, stderr = self.run_xtb_job(self.xtbargs + ['--hess'] + self.additional_xtb_keywords, self.tmpdirname)
        else:
            stdout, stderr = self.run_xtb_job(self.xtbargs + ['--hess','--ptb','--alpha'] + self.additional_xtb_keywords, self.tmpdirname)

        try:
            xtb_scf_successful = self.error_handle(stdout+stderr)
        except:
            if not os.path.exists(f'{self.tmpdirname}/molecule{ii}.hessian'):
                raise

        mol.__dict__['xtb_scf_successful'] = xtb_scf_successful

        # There is bug in xtb that it fails for H2, thus, it may end with error but still prints hessian.
        # 6.7.0 will not raise error on hessian of H2 but 6.6.1 will raise segmentation error
        if xtb_scf_successful or os.path.exists(f'{self.tmpdirname}/molecule{ii}.hessian'): 
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

            with open(f'{self.tmpdirname}/molecule{ii}.hessian', 'r') as fout:
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
                if os.path.exists(f'{self.tmpdirname}/molecule{ii}.dipd'):
                    with open(f'{self.tmpdirname}/molecule{ii}.dipd', 'r') as fout:
                        dipd = []
                        for iline, line in enumerate(fout.readlines()):
                            if iline == 0: continue 
                            dipd += [float(each) for each in line.split()]
                        dipd = np.array(dipd).astype(float)
                        mol.dipole_derivatives = dipd / constants.Debye
            # Read polarizability derivatives 
            if calculate_polarizability_derivatives:
                if os.path.exists(f'{self.tmpdirname}/molecule{ii}.polard'):
                    with open(f'{self.tmpdirname}/molecule{ii}.polard','r') as fout:
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
            print('xTB calculation failed with unhandled error in MLatom.')
            sys.stdout.flush()

    def run_xtb_job(self,xtbargs,tmpdirname):
        segmentation_error = False
        program_error = False # error that will be in output
        other_error = None
        stderr = []; stdout = []
        try:
            proc = subprocess.run(xtbargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, check=True)
            stdout = proc.stdout.split('\n')
            stderr = proc.stderr.split('\n')
        except subprocess.CalledProcessError as e:
            kill_signal = e.returncode
            if kill_signal < 0:
                # get the signal name for numerical signal
                kill_proc = subprocess.run(
                    f'kill -l {-kill_signal}',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    text=True,
                )
                kill_signal_name = kill_proc.stdout.strip()
                # kill signal: SIGSEGV - segmentation error
                # ref: https://man7.org/linux/man-pages/man7/signal.7.html
                if kill_signal_name == "SEGV": 
                    segmentation_error = True
                # kill signal: ABRT - hard to handle. This will result in abnormal termination
                # ref: https://man7.org/linux/man-pages/man3/abort.3.html
                elif kill_signal_name == "ABRT":
                    program_error = True
                else:
                    other_error = kill_signal_name
            else:
                program_error = True

        if program_error:
            # program abnormally terminated with error in the output file generated by the program, not terminated from system signal
            proc = subprocess.Popen(xtbargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
            stdout, stderr = proc.communicate()
            stdout = stdout.split('\n')
            stderr = stderr.split('\n')

        if segmentation_error:
            stderr.append('segmentation fault')
        if program_error:
            stderr.append('program error')
        if other_error:
            stderr.append(f'killed signal {other_error}')
        return stdout, stderr

    def error_handle(self,stds):
        xtb_scf_successful = True

        if stds[-1] == 'segmentation fault':
            # if "OMP_STACKSIZE" not in os.environ:
            #     print('OMP stacksize is too small for the system. Try to increase to 1G')
            #     sys.stdout.flush()
            #     os.environ["OMP_STACKSIZE"] = '1G'
            #     rerun = True
            #     terminated = False
            #     return rerun, terminated 
            # else:
            raise ValueError('OMP stacksize is too small for the system. Try to increase accordint to offical manual https://xtb-docs.readthedocs.io/en/latest/setup.html#parallelisation.')
        
        if stds[-1] == 'program error':
            for ii, readable in enumerate(stds):

                if 'SIGABRT' in readable:
                    error_info = ''
                    for jj in range(ii):
                        if 'Calculation Setup' in stds[jj]:
                            error_info = '\n'.join(stds[jj-1:ii])
                            raise ValueError(f'xtb abnormal termination with following error: \n{error_info}')
                
                if '[ERROR]' in readable:
                    error_info = ''
                    for jj in range(ii, len(stds)):
                        if '###################################################' in stds[jj]:
                            error_info = '\n'.join(stds[ii:jj])
                            raise ValueError(f'xtb abnormal termination with following error: \n{error_info}')
        
        if 'killed signal' in stds[-1]:
            raise ValueError(f'xTB calculation errors terminated by the system with {stds[-1]}')

        for readable in stds:
            if 'normal termination of xtb' in readable:
                xtb_scf_successful = True
            
        # convergence error not implemented
        return xtb_scf_successful

if __name__ == '__main__':
    pass
