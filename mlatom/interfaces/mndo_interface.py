#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! mndo: interface to the MNDO program                                       ! 
  ! Implementations by: Pavlo O. Dral & Peikun Zheng                          !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
from requests.structures import CaseInsensitiveDict
from .. import stopper, simulations, models
from ..utils import doc_inherit

class mndo_methods(models.model):
    '''
    MNDO interface

    Arguments:
        method (str): method used in MNDO
        read_keywords_from_file (str): keywords used in MNDO
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'True'``
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``
    '''
    method_keywords = {'ODM2*': 'iop=-22 immdp=-1',
                    'ODM2': 'iop=-22',
                    'ODM3': 'iop=-23',
                    'OM3': 'iop=-8',
                    'OM2': 'iop=-6',
                    'OM1': 'iop=-5',
                    'PM3': 'iop=-7',
                    'AM1': 'iop=-2',
                    'MNDO/d': 'iop=-10',
                    'MNDOC': 'iop=-1',
                    'MNDO': 'iop=0',
                    'MINDO/3': 'iop=1',
                    'CNDO/2': 'iop=2',
                    'SCC-DFTB': 'iop=5',
                    'SCC-DFTB-heats': 'iop=6',
                    'MNDO/H': 'iop=-3',
                    'MNDO/dH': 'iop=-13'
                    }

    heats_scf_methods = [mm.casefold() for mm in ['OM1', 'OM2', 'OM3',
                        'PM3', 'AM1', 'MNDO/d', 'MNDOC', 'MNDO',
                        'MINDO/3', 'CNDO/2', 'SCC-DFTB', 'SCC-DFTB-heats',
                        'MNDO/H', 'MNDO/dH']]

    available_methods = models.methods.methods_map['mndo'] #need to sync with dict method_keywords somehow
    
    def __init__(self, method='ODM2*', read_keywords_from_file='', save_files_in_current_directory=True, working_directory=None, **kwargs):
        self.method = method
        self.read_keywords_from_file = read_keywords_from_file
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        try: self.mndobin = os.environ['mndobin']
        except:
            errmsg = 'Cannot find the MNDO program, please set the environment variable: export mndobin=...'
            raise ValueError(errmsg)

    @doc_inherit
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        try: from .. import constants, data, stopper
        except:
            import constants, data, stopper
            
        if self.method.casefold() in self.heats_scf_methods:
            energy_label = 'energy'
            molecule.scf_enthalpy_of_formation_at_298_K = True
        else: energy_label = 'energy'
            
        if calculate_hessian: import struct
        
        import tempfile, subprocess
        if self.read_keywords_from_file != '':
            mndokw_file = self.read_keywords_from_file
            with open(mndokw_file, 'r') as fmndokw:
                mndokeywords = fmndokw.read().strip('\n')
            mndokeywords = mndokeywords.split('\n\n')
            imol = -1
            for mol in molDB.molecules:
                imol += 1
                jmol = imol
                if len(mndokeywords) < imol+1: jmol = -1
                mol.mndo_keywords = mndokeywords[jmol]
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory: tmpdirname = '.'
            if self.working_directory is not None:
                tmpdirname = self.working_directory
                if not os.path.exists(tmpdirname):
                    os.makedirs(tmpdirname)
                tmpdirname = os.path.abspath(tmpdirname)
                    
            ii = 0
            for mol in molDB.molecules:
                if calculate_energy_gradients or calculate_hessian:
                    natoms = len(mol.atoms)
                ii += 1
                mndoinpfilename = f'{tmpdirname}/mndo{ii}.inp'
                with open(mndoinpfilename, 'w') as fmndo:
                    if 'mndo_keywords' in mol.__dict__.keys(): 
                        fmndo.writelines(mol.mndo_keywords + '\n\n\n')
                    else:
                        if calculate_hessian:
                            fmndo.writelines('jop=2 +\n')
                        elif calculate_energy_gradients:
                            fmndo.writelines('jop=-2 +\n')
                        else:
                            fmndo.writelines('jop=-1 +\n')
                        fmndo.writelines('%s igeom=1 iform=1 nsav15=3 +\n' % CaseInsensitiveDict(self.method_keywords)[self.method])
                        fmndo.writelines('icuts=-1 icutg=-1 kitscf=9999 iscf=9 iplscf=9 +\n')
                        fmndo.writelines('iprint=-1 kprint=-5 lprint=-2 mprint=0 jprint=-1 +\n')
                        kharge = 'kharge=%d' % mol.charge
                        imult = ''
                        if mol.multiplicity == 1:
                            imult = 'imult=0'
                        else:
                            imult = 'imult=%d' % mol.multiplicity
                        fmndo.writelines(f'{kharge} {imult} nprint=-1\n\n\n')
                    for atom in mol.atoms:
                        line = '%2d     %12.8f %3d     %12.8f %3d     %12.8f %3d\n' % (
                            atom.atomic_number,
                            atom.xyz_coordinates[0], 1,
                            atom.xyz_coordinates[1], 1,
                            atom.xyz_coordinates[2], 1)
                        fmndo.writelines(line)
                    fmndo.writelines('\n')
                
                mndooutfilename = f'{tmpdirname}/mndo{ii}.out'
                mndoargs = [self.mndobin, '<', mndoinpfilename, '>', mndooutfilename, '2>&1']
                cmd = ' '.join(mndoargs)
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                proc.communicate()
                # proc.wait()
                mndo_scf_successful = True
                outputs = []
                with open(mndooutfilename, 'r') as fout:
                    for readable in fout:
                        outputs.append(readable)
                        if 'UNABLE TO ACHIEVE SCF CONVERGENCE' in readable: mndo_scf_successful = False
                if not mndo_scf_successful: print(' * Warning * mndo calculations did not converge!')
                mol.mndo_scf_successful = mndo_scf_successful
                with open(f'{tmpdirname}/fort.15', 'r') as ffort15:
                    iatom = -1
                    nenergyline = 0
                    for line in ffort15:
                        if calculate_energy and 'ENERGY' in line:
                            nenergyline = 1
                            continue
                        if calculate_energy and nenergyline == 1:
                            nenergyline = -1
                            energy = line.split()[0]
                            energy = float(energy) / (27.21 * 23.061)
                            mol.__dict__[energy_label] = energy
                        if not calculate_energy_gradients: continue
                        if 'CARTESIAN GRADIENT' in line:
                            iatom = 0
                            continue
                        if iatom >= 0 and iatom <= natoms - 1:
                            mol.atoms[iatom].energy_gradients = np.array([float(xx) / (27.21 * 23.061) for xx in line.split()[-3:]]).astype(float)
                            iatom += 1
                if calculate_hessian:
                    lhess = natoms * natoms * 9
                    fhess = open(f'{tmpdirname}/fort.4', 'rb')
                    data = fhess.read()
                    dt = f'id{lhess}d'
                    dat_size = struct.calcsize(dt)
                    temp = struct.unpack(dt, data[:dat_size])
                    hess = np.array(temp[2:]).astype(float) / (constants.Bohr2Angstrom**2)
                    mol.hessian = hess.reshape(natoms*3,natoms*3)
                    
                # read dipole moment 
                for iline in range(len(outputs)):
                    if 'DIPOLE' in outputs[iline]:
                        sum_line = outputs[iline+4].rstrip()
                        sum_line = sum_line.split(' ')
                        sum_line = [each for each in sum_line if each != '' ][1:]
                        dipole_moment = np.array([eval(each) for each in sum_line])
                        mol.dipole_moment = dipole_moment                

if __name__ == '__main__':
    pass