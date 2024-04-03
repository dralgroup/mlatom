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
import math
import tempfile, subprocess
from .. import constants, models
from ..decorators import doc_inherit

class orca_methods(models.model):
    '''
    ORCA interface

    Arguments:
        method (str): method used the same as in ORCA, e.g. ``'B3LYP/6-31G*'`` (case insensitive). This is the first line in orca input and you can also store the whole first line here.
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'False'``
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``
        nthreads (int): equivalent to %pal nprocs in ORCA input file 
        nthreads_list (list): a list of number of nthreads used in CCSD(T)*/CBS method. The order should be [mp2_tz, mp2_qz, dlpno_normal_dz, dlpno_normal_tz, dlpno_tight_dz] 
        additional_keywords (list): list of keywords(str) to be added to orca input (first line)
        input_file (str): name of your orca input file. Default will use "[molecule number]_"
        output_keywords (list): list of keywords that you want to extract from output file. (Currently customized keywords are only supported in energy calculation and in property.txt file)        
    
    
    .. note::

        When using CCSD(T)*/CBS for calculation, please make sure the ``'nthreads'`` you used for each method will not cause memory exceeding. 
        We suggest using ``'nthreads_list'`` to properly set ``'nthreads'`` for each component method in the order: [MP2/cc-pVTZ, MP2/cc-pVQZ, DLPNO-CCSD(T)-normalPNO/cc-pVDZ, DLPNO-CCSD(T)-normalPNO/cc-pVTZ, DLPNO-CCSD(T)-tightPNO/cc-pVTZ]
        If only ``'nthreads'`` is set, all component methods would use the same number of threads.  
    
    '''
    
    def __init__(self, method='wb97x/6-31G*', **kwargs):
        if not "orcabin" in os.environ:
            raise ValueError('enviromental variable orcabin is not set')
        else:
            self.orcabin = os.environ['orcabin']
        
        self.method = method
        self.orca_successful = True

        if self.method == 'CCSD(T)*/CBS':
            self.cc_method = ccsdtstarcbs(**kwargs)
        if 'nthreads' in kwargs:
            self.nthreads = kwargs['nthreads']
        else:
            self.nthreads = 1 
        
        if 'additional_keywords' in kwargs:
            self.additional_keywords = kwargs['additional_keywords']
        else:
            self.additional_keywords = None

        if 'save_files_in_current_directory' in kwargs:
            self.save_files_in_current_directory = kwargs['save_files_in_current_directory']
        else:
            self.save_files_in_current_directory = False 

        if 'working_directory' in kwargs:
            self.working_directory = kwargs['working_directory']
        else:
            self.working_directory = None

        if 'input_file' in kwargs:
            self.input_file = kwargs['input_file']
        else:
            self.input_file = '' 

        if 'output_keywords' in kwargs:
            self.output_keywords = kwargs['output_keywords']
        else:
            self.output_keywords = None
        
    @doc_inherit 
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        if self.method == 'CCSD(T)*/CBS':
            self.cc_method.predict(molecular_database=molDB)
        else:
            self.calculate_energy = calculate_energy
            self.calculate_energy_gradients = calculate_energy_gradients
            self.calculate_hessian = calculate_hessian
                        
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory: tmpdirname = '.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                    if not os.path.exists(tmpdirname):
                        os.makedirs(tmpdirname)
                tmpdirname = os.path.abspath(tmpdirname)

                for imol in range(len(molDB.molecules)):
                    imolecule = molDB.molecules[imol]
                    self.inpfile = f'{tmpdirname}/molecule{imol}_' + self.input_file
                    # calculation
                    self.generate_orca_inp(molecule=imolecule)
                    # run orca
                    orcaargs = [self.orcabin, f'{self.inpfile}.inp', '>', f'{self.inpfile}.out'] 
                    proc = subprocess.Popen(' '.join(orcaargs), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                    proc.communicate()
                    # read output
                    self.parse_orca_output(molecule=imolecule)

    def parse_orca_output(self, molecule):
        natom = len(molecule.atoms)
        
        if self.calculate_energy:
            if not self.calculate_energy_gradients:
                with open(f'{self.inpfile}_property.txt', 'r') as orcaout:
                    orcaout_lines = orcaout.readlines()
                    for ii in range(len(orcaout_lines)):
                        if 'Total Energy' in orcaout_lines[ii]: # ? check SCf energy or total enenrgy
                            molecule.energy = float(orcaout_lines[ii].split()[-1]) 
            else:
                with open(f'{self.inpfile}.engrad', 'r') as orcaout:
                    orcaout_lines = orcaout.readlines()
                    for ii in range(len(orcaout_lines)):
                        if 'total energy' in orcaout_lines[ii]:
                            molecule.energy = float(orcaout_lines[ii+2])

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

        if not self.calculate_energy and self.output_keywords:
            for keyword in self.output_keywords:
                keyword_name = self.input_file+'_'+keyword.lower().replace(' ', '_')
                with open(f'{self.inpfile}_property.txt', 'r') as orcaout:
                    orcaout_lines = orcaout.readlines()
                    for ii in range(len(orcaout_lines)):
                        if keyword in orcaout_lines[ii]:
                            molecule.__dict__[keyword_name] = float(orcaout_lines[ii].split()[-1]) 

    def generate_orca_inp(self, molecule, **kwargs):
        with open(f'{self.inpfile}.inp', 'w') as forca:
            #keywords
            if '/' in self.method and len(self.method.split('/'))==2:
                method_to_write = ' '.join(self.method.split('/'))
            else:
                method_to_write = self.method
            if self.additional_keywords:
                method_to_write += ' '.join(self.additional_keywords)
            forca.writelines(f'! {method_to_write}\n')
            # decide calculation type
            if self.calculate_energy:
                calculation_type = []
                calculation_type += ['SP']
                if self.calculate_energy_gradients:
                    calculation_type += ['ENGRAD']
                if self.calculate_hessian:
                    calculation_type += ['FREQ']
                calculation_type = ' '.join(calculation_type)
                forca.writelines(f'! {calculation_type}\n')
            
            forca.writelines(f'%pal nprocs {self.nthreads} end\n')
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

class ccsdtstarcbs(models.model):

    def __init__(self, **kwargs):

        if 'nthreads_list' in kwargs:
            self.nthreads_list = kwargs['nthreads_list']
            if len(self.nthreads_list) != 5:
                print('Number of values to asign to each methods should be 5. Default nthreads will be used')
                self.nthreads_list = None  
        else:
            self.nthreads_list = None

        self.successful = True
        if self.nthreads_list:
            kwargs.update({'nthreads':self.nthreads_list[0], 'input_file':'mp2_tz', 'output_keywords':['SCF Energy', 'Correlation Energy']})
        else:
            kwargs.update({'input_file':'mp2_tz', 'output_keywords':['SCF Energy', 'Correlation Energy']})
        self.mp2_tz = orca_methods(method='''RIMP2 RIJK cc-pVTZ cc-pVTZ/JK cc-pVTZ/C\n! tightscf noautostart scfconvforced miniprint nopop''',
                                            **kwargs)

        if self.nthreads_list:
            kwargs.update({'nthreads':self.nthreads_list[1], 'input_file':'mp2_qz', 'output_keywords':['SCF Energy', 'Correlation Energy']})
        else:
            kwargs.update({'input_file':'mp2_qz', 'output_keywords':['SCF Energy', 'Correlation Energy']})
        self.mp2_qz = orca_methods(method='''RIMP2 RIJK cc-pVQZ cc-pVQZ/JK cc-pVQZ/C\n! tightscf noautostart scfconvforced miniprint nopop''', 
                                            **kwargs)

        if self.nthreads_list:
            kwargs.update({'nthreads':self.nthreads_list[2],'input_file':'dlpno_normal_dz', 'output_keywords':['Total Correlation Energy']})
        else:
            kwargs.update({'input_file':'dlpno_normal_dz', 'output_keywords':['Total Correlation Energy']})
        self.npno_dz = orca_methods(method='''DLPNO-CCSD(T) normalPNO RIJK cc-pVDZ cc-pVDZ/C cc-pvTZ/JK\n! tightscf noautostart scfconvforced miniprint nopop''',
                                            **kwargs)

        if self.nthreads_list:
            kwargs.update({'nthreads':self.nthreads_list[3], 'input_file':'dlpno_normal_tz','output_keywords':['Total Correlation Energy']})
        else:
            kwargs.update({'input_file':'dlpno_normal_tz','output_keywords':['Total Correlation Energy']})
        self.npno_tz = orca_methods(method='''DLPNO-CCSD(T) normalPNO RIJK cc-pVTZ cc-pVTZ/C cc-pVTZ/JK\n! tightscf noautostart scfconvforced miniprint nopop''', 
                                            **kwargs)

        if self.nthreads_list:
            kwargs.update({'nthreads':self.nthreads_list[4], 'input_file':'dlpno_tight_dz','output_keywords':['Total Correlation Energy']})
        else:
            kwargs.update({'input_file':'dlpno_tight_dz','output_keywords':['Total Correlation Energy']})
        self.tpno_dz = orca_methods(method='''DLPNO-CCSD(T) tightPNO RIJK cc-pVDZ cc-pVDZ/C cc-pVTZ/JK\n! tightscf noautostart scfconvforced miniprint nopop''',
                                            **kwargs)

        self.combination = [self.mp2_tz, self.mp2_qz, self.npno_dz, self.npno_tz, self.tpno_dz]

    def predict(self, molecular_database, **kwargs):
        for method in self.combination:
            method.predict(molecular_database=molecular_database, calculate_energy=False)
        for mol in molecular_database:
            alpha = 5.46; beta = 2.51
            hf_tz = mol.mp2_tz_scf_energy
            hf_qz = mol.mp2_qz_scf_energy
            mp2_tz = mol.mp2_tz_correlation_energy
            mp2_qz = mol.mp2_qz_correlation_energy
            npno_dz = mol.dlpno_normal_dz_total_correlation_energy
            npno_tz = mol.dlpno_normal_tz_total_correlation_energy
            tpno_dz = mol.dlpno_tight_dz_total_correlation_energy

            E_hf_xtrap = (math.exp(-alpha * 4**0.5) * hf_tz - math.exp(-alpha * 3**0.5) * hf_qz) \
                    / (math.exp(-alpha * 4**0.5) - math.exp(-alpha * 3**0.5))
            E_mp2_xtrap = (4**beta * mp2_qz - 3**beta * mp2_tz) / (4**beta - 3**beta)
            energy = E_hf_xtrap + E_mp2_xtrap - mp2_tz + npno_tz + tpno_dz - npno_dz

            mol.energy = energy 
