'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! dftd3: interface to the dftd3 program                                     ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''
# https://github.com/dftd3/simple-dftd3
# https://dftd3.readthedocs.io/en/latest/index.html
import json
import numpy as np
import sys
import os 
from .. import data, constants
from ..model_cls import method_model
from .. import stopper
from ..decorators import doc_inherit

class dftd3_methods(method_model):
    '''
    DFT-D3 interface

    Arguments:
        functional (str): functional to use
        method (str): 
            - "d3bj": Rational damping function
            - "d3zero": Zero damping function
            - "d3bjm": Modified damping parameters for the rational damping function
            - "d3zerom": Modified version of the zero damping function
            - "d3op": Optimized power damping function
        damping_function_params (list, optional): custom damping parameters in correct order. 
            - 'd3bj': [s6, s8, a1, a2]
            - 'd3zero': [s6, s8, rs6]
            - 'd3bjm': [s6, s8, a1, a2]
            - 'd3zerom': [s6, s8, rs6, bet]
            - 'd3op': [s6, s8, a1, a2, bet]

    .. note::
    
        Hessian in dftd3 uses numerical hessian.

    '''

    bin_env_name = 'dftd3bin'
    supported_methods = ['d3zero', 'd3bj', 'd3bjm', 'd3zerom', 'd3op']

    def __init__(self, functional='wb97x', method='d3bj', damping_function_params=None, save_files_in_current_directory=True, working_directory=None):
        self.functional = functional
        self.method = method.casefold()
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory 
        self.damping_function_params = damping_function_params
        if self.damping_function_params:
            self.damping_function_params = [str(ii) for ii in self.damping_function_params]
        self.dftd3bin = self.get_bin_env_var()
        if self.dftd3bin is None:
            raise ValueError('Cannot find the dftd3 program, please set the environment variable: export dftd3bin=...')

    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, **kwargs):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        import tempfile, subprocess        
        ii = 0
        for mol in molDB.molecules:
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory: tmpdirname='.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                    if not os.path.exists(tmpdirname):
                        os.makedirs(tmpdirname)
                    tmpdirname = os.path.abspath(tmpdirname)
                ii += 1
                xyzfilename = f'{tmpdirname}/predict{ii}.xyz'
                mol.write_file_with_xyz_coordinates(filename = xyzfilename)
                
                dftd3args = [self.dftd3bin, f"--{self.method.replace('d3','')}", '%s' % self.functional, '--noedisp']
                if self.damping_function_params:
                    dftd3args += [f"--{self.method.replace('d3','')}-param"]
                    dftd3args += self.damping_function_params
                if calculate_energy_gradients or calculate_hessian:
                    dftd3args += ['--json', 'dftd3.json', '--grad', 'grad.txt']
                elif calculate_energy:
                    dftd3args += ['--json', 'dftd3.json']
                dftd3args += [xyzfilename]
                proc = subprocess.Popen(dftd3args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True)
                outs,errs = proc.communicate()
                dftd3_successful = False
                if '[Error]' not in outs+errs:
                    dftd3_successful = True

                mol.dftd3_successful = dftd3_successful
                with open(f'{tmpdirname}/dftd3.json', 'r') as f:
                    d3_results = json.load(f)

                if calculate_energy:
                    energy = float(d3_results['energy'])
                    mol.energy = energy
                if calculate_energy_gradients:
                    grad = np.array(d3_results['gradient']) / constants.Bohr2Angstrom
                    grad = grad.reshape(-1, 3)
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].energy_gradients = grad[iatom]
                if calculate_hessian:
                    mol.hessian = self.numerical_hessian(molecule=mol.copy())

    def numerical_hessian(self, molecule, eps=5.29167e-4, epsgrad=1e-5):
        self.predict(molecule=molecule, calculate_energy_gradients=True)
        g1 = molecule.get_energy_gradients()
        coordinates1 = molecule.xyz_coordinates.reshape(-1)
        ndim = len(coordinates1)
        hess = np.zeros((ndim, ndim))
        coordinates2 = coordinates1
        for i in range(ndim):
            x0 = coordinates2[i]
            coordinates2[i] = coordinates1[i] + eps
            molecule2 = molecule.copy()
            molecule2.xyz_coordinates = coordinates2.reshape(len(molecule2.atoms),3)
            self.predict(molecule=molecule2, calculate_energy_gradients=True)
            g2 = molecule2.get_energy_gradients()
            hess[:, i] = (g2.reshape(-1) - g1.reshape(-1)) / eps
            coordinates2[i] = x0
        return hess 

if __name__ == '__main__':
    pass
