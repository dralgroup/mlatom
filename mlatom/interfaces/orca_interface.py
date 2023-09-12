#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! orca: interface to the ORCA program                                       ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
import tempfile, subprocess
from .. import constants, models
from ..utils import doc_inherit

class orca_methods(models.model):
    '''
    ORCA interface

    Arguments:
        method (str): method used the same as in ORCA, e.g. ``'B3LYP/6-31G*'`` (case insensitive)
        save_files_in_current_directory (bool): keep input and output files or not, default ``'False'``
        nthreads (int): equivalent to %pal nprocs in ORCA input file 
        runTypes (str): equivalent to calculation types in ORCA input file, default ``'SP'`` (case insensitive)

    '''

    orca_keywords = {
        'methods'                : 'wb97x', #functionals for DFT; HF, CCSD...
        'basis'                 : 'def2-TZVPP', 
        'auxiliary_basis'       : 'AUTOAUX', 
        'speedup_algorithm'     : 'RIJCOSX', 
        'runTypes'              : ['SP'], #OPT, FREQ(for hessian), SP(for energy), ENGRAD(energy and gradient)...
        'nthreads'              : 1
    }
    
    def __init__(self, method='', save_files_in_current_directory=False, **kwargs):
        if not "orcabin" in os.environ:
            raise ValueError('enviromental variable orcabin is not set')
        else:
            self.orcabin = os.environ['orcabin']
        #self.orca_keywords.update(orca_keywords)
        if method != '':
            self.orca_keywords['methods'] = method.split('/')[0]
            self.orca_keywords['basis'] = method.split('/')[1]

        if 'save_files_in_current_directory' in kwargs.keys():
            self.save_files_in_current_directory = kwargs['save_files_in_current_directory']
        else:
            self.save_files_in_current_directory = save_files_in_current_directory
        #if 'qm_kwargs' in kwargs.keys():
        for keyword, keyword_value in kwargs.items():
            self.orca_keywords[keyword] = keyword_value

    @doc_inherit 
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        self.calculate_energy = calculate_energy
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_hessian = calculate_hessian
        
        # decide calculation type
        if calculate_energy:
            if 'SP' not in [k.upper() for k in self.orca_keywords['runTypes']]:
                self.orca_keywords['runTypes'].append('SP')
        if calculate_energy_gradients:
            if 'ENGRAD' not in [k.upper() for k in self.orca_keywords['runTypes']]:
                self.orca_keywords['runTypes'].append('ENGRAD')
        if calculate_hessian:
            if 'FREQ' not in [k.upper() for k in self.orca_keywords['runTypes']]:
                self.orca_keywords['runTypes'].append('FREQ')
                    
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory: tmpdirname = '.'

            for imol in range(len(molDB.molecules)):
                imolecule = molDB.molecules[imol]
                self.inpfile = f'{tmpdirname}/molecule{imol}'
                # calculation
                generate_orca_inp(filename=f'{self.inpfile}.inp', molecule=imolecule, keywords=self.orca_keywords)
                # run orca
                orcaargs = [self.orcabin, f'{self.inpfile}.inp', '>', f'{self.inpfile}.out'] 
                proc = subprocess.Popen(' '.join(orcaargs), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                proc.communicate()
                # read output
                self.parse_orca_output(molecule=imolecule)

    # store energy as following format:(use tree node)
    # mol.energy.method_basis=energy
    # atom.gradient.method_basis=gradient
    # mol.hessian.method_basis=hessian (n*3 x n*3)
    def parse_orca_output(self, molecule):
        natom = len(molecule.atoms)
        
        if self.calculate_energy:
            if not self.calculate_energy_gradients:
                with open(f'{self.inpfile}_property.txt', 'r') as orcaout:
                    orcaout_lines = orcaout.readlines()
                    for ii in range(len(orcaout_lines)):
                        if 'Total Energy' in orcaout_lines[ii]:
                            en = float(orcaout_lines[ii].split()[-1])    
            else:
                with open(f'{self.inpfile}.engrad', 'r') as orcaout:
                    orcaout_lines = orcaout.readlines()
                    for ii in range(len(orcaout_lines)):
                        if 'total energy' in orcaout_lines[ii]:
                            en = float(orcaout_lines[ii+2])
            molecule.energy = en               
                    
        if self.calculate_energy_gradients:
            with open(f'{self.inpfile}.engrad', 'r') as orcaout:
                orcaout_lines = orcaout.readlines()
                for ii in range(len(orcaout_lines)):
                    if 'gradient' in orcaout_lines[ii]:
                        grad = orcaout_lines[ii+2: ii+2+natom*3]
                        grad = [float(g.split()[0])/constants.Bohr2Angstrom for g in grad]
            for ii in range(natom):
                a = molecule.atoms[ii]
                a.energy_gradients = grad[3*ii: 3*ii+3] 
        
        if self.calculate_hessian:            
            with open(f'{self.inpfile}.hess', 'r') as orcaout:
                orcaout_lines = orcaout.readlines()
                hessian_index = orcaout_lines.index('$hessian\n') # start from $hessian
                ncoordinate = int(orcaout_lines[hessian_index+1]) # second line is #coordinates
                ncircle = int((ncoordinate-0.5) / 5)+1 # blocks

                hessian_matrix = np.zeros((ncoordinate, ncoordinate))

                for ii in range(ncircle):
                    start_index = hessian_index+2+(ncoordinate+1)*ii
                    cols_index_list = orcaout_lines[start_index].split()
                    for jj in range(ncoordinate):
                        hessian_line = orcaout_lines[start_index+1+jj].split()
                        for kk in range(len(cols_index_list)):
                            col_index = cols_index_list[kk]
                            hessian_matrix[int(hessian_line[0])][int(col_index)] = float(hessian_line[kk+1])/constants.Bohr2Angstrom**2
            molecule.hessian = hessian_matrix
                            
def generate_orca_inp(filename, molecule, keywords, **kwargs):
    with open(filename, 'w') as forca:
        #keywords
        forca.writelines(f'! {keywords["methods"]} {keywords["basis"]} {keywords["auxiliary_basis"]} {keywords["speedup_algorithm"]}\n')
        runTypes = ' '.join(keywords["runTypes"])
        forca.writelines(f'! {runTypes}\n')
        forca.writelines(f'%pal nprocs {keywords["nthreads"]} end\n')
        forca.writelines(f'%maxcore 1000\n')
        #structure
        forca.writelines(f'* xyz {molecule.charge} {molecule.multiplicity}\n')
        for atom in molecule.atoms:
            line = '%s     %12.8f     %12.8f     %12.8f\n' % (
                    atom.element_symbol,
                    atom.xyz_coordinates[0],
                    atom.xyz_coordinates[1],
                    atom.xyz_coordinates[2])
            forca.writelines(line)
        forca.writelines('*')


if __name__ == '__main__':
    pass