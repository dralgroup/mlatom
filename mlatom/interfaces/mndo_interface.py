#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! mndo: interface to the MNDO program                                       ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, and Lina Zhang           !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np
from requests.structures import CaseInsensitiveDict
from .. import constants, data
from ..model_cls import method_model
from ..decorators import doc_inherit

mndo_method_keywords = {'ODM2*': 'iop=-22 immdp=-1',
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
class mndo_methods(method_model):
    '''
    MNDO interface

    Arguments:
        method (str): method used in MNDO
        read_keywords_from_file (str): keywords used in MNDO
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'True'``
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``
    '''
    bin_env_name = 'mndobin'
    method_keywords = mndo_method_keywords
    supported_methods = list(method_keywords.keys())
    heats_scf_methods = [mm.casefold() for mm in ['OM1', 'OM2', 'OM3',
                        'PM3', 'AM1', 'MNDO/d', 'MNDOC', 'MNDO',
                        'MINDO/3', 'CNDO/2', 'SCC-DFTB', 'SCC-DFTB-heats',
                        'MNDO/H', 'MNDO/dH']]
    
    def __init__(self, method='ODM2*', read_keywords_from_file='', save_files_in_current_directory=True, working_directory=None):
        self.method = method
        self.read_keywords_from_file = read_keywords_from_file
        if self.read_keywords_from_file != '':
            self.read_keywords_from_file = os.path.abspath(read_keywords_from_file)
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.mndobin = self.get_bin_env_var()
        if self.mndobin is None:
            raise ValueError('Cannot find the MNDO program, please set the environment variable: export mndobin=...')

    @doc_inherit
    def predict(self, 
                molecular_database=None, 
                molecule=None, 
                nstates=1, 
                current_state=0, 
                calculate_energy=True, 
                calculate_energy_gradients=False, 
                calculate_hessian=False,
                calculate_dipole_derivatives=False,
                calculate_nacv=False,
                read_density_matrix=False):
        
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        
        # it is not implemented, but some method's energies are not total energies, but HoFs.
        # if self.method.casefold() in self.heats_scf_methods:
        #     energy_label = 'energy'
        #     molecule.scf_enthalpy_of_formation_at_298_K = True
        # else: energy_label = 'energy'

        if not isinstance(calculate_energy_gradients, list):
            if calculate_energy_gradients:
                calculate_energy_gradients = [False] * nstates
                calculate_energy_gradients[current_state] = True
            else:
                calculate_energy_gradients = [False] * nstates

        if not isinstance(calculate_hessian, list):
            if calculate_hessian:
                calculate_hessian = [False] * nstates
                calculate_hessian[current_state] = True
            else:
                calculate_hessian = [False] * nstates

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
                ii += 1
                mndoinpfilename = f'{tmpdirname}/mndo{ii}.inp'
                write_mndo_input_file(filename=mndoinpfilename, molecule=mol, method=self.method, nstates=nstates,
                                      calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian,
                                      calculate_dipole_derivatives=calculate_dipole_derivatives, calculate_nacv=calculate_nacv,
                                      read_density_matrix=read_density_matrix)
                mndooutfilename = f'{tmpdirname}/mndo{ii}.out'
                mndoargs = [self.mndobin, '<', mndoinpfilename, '>', mndooutfilename, '2>&1']
                cmd = ' '.join(mndoargs)
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdirname, universal_newlines=True, shell=True)
                proc.communicate()
                parse_mndo_output(filename=mndooutfilename, molecule=mol)
                if nstates > 1:
                    if mol.electronic_states is not []:
                        if 'energy' in mol.electronic_states[current_state].__dict__.keys():
                            mol.energy = mol.electronic_states[current_state].energy
                        if 'energy_gradients' in mol.electronic_states[current_state].atoms[0].__dict__.keys():
                            mol.energy_gradients = mol.electronic_states[current_state].energy_gradients

def write_mndo_input_file(filename=None, molecule=None, method='ODM2*', nstates=1,
                          calculate_energy_gradients=False, calculate_hessian=False,
                          calculate_dipole_derivatives=False, calculate_nacv=False,
                          read_density_matrix=False):
    
    with open(filename, 'w') as fmndo:
        if 'mndo_keywords' in molecule.__dict__.keys(): 
            fmndo.writelines(molecule.mndo_keywords + '\n\n\n')
        else:
            if any(calculate_hessian):
                fmndo.writelines('jop=2 +\n')
            elif any(calculate_energy_gradients):
                fmndo.writelines('jop=-2 +\n')
            else:
                fmndo.writelines('jop=-1 +\n')

            fmndo.writelines('%s igeom=1 iform=1 nsav15=3 +\n' % CaseInsensitiveDict(mndo_method_keywords)[method])
            fmndo.writelines('icuts=-1 icutg=-1 kitscf=9999 iscf=9 iplscf=9 +\n')
            if calculate_dipole_derivatives:
                fmndo.writelines('iprint=-1 kprint=-5 lprint=0 mprint=0 jprint=-1 +\n')
            else:
                fmndo.writelines('iprint=-1 kprint=-5 lprint=-2 mprint=0 jprint=-1 +\n')
            if read_density_matrix:
                fmndo.writelines('ktrial=11 +\n')
            kharge = 'kharge=%d' % molecule.charge
            imult = ''
            if molecule.multiplicity == 1:
                imult = 'imult=0'
            else:
                imult = 'imult=%d' % molecule.multiplicity
            fmndo.writelines(f'{kharge} {imult} nprint=-1 ')

            if nstates > 1:
                fmndo.writelines('+\n')
                fmndo.writelines(f'kci=6 iroot={nstates} iuvcd=2 +\n')
                fmndo.writelines(f'ncisym=-1 ioutci=3 ipubo=1 ')
                if any(calculate_energy_gradients):
                    fmndo.writelines('+\n')
                    ncigrd = calculate_energy_gradients.count(True)
                    if not calculate_nacv:
                        fmndo.writelines(f'ncigrd={ncigrd} icross=1 ')
                    else:
                        fmndo.writelines(f'ncigrd={ncigrd} icross=7 inac=0 ')
            fmndo.writelines('\n\n\n')
        
        for atom in molecule.atoms:
            line = '%2d     %12.8f %3d     %12.8f %3d     %12.8f %3d\n' % (
                atom.atomic_number,
                atom.xyz_coordinates[0], 1,
                atom.xyz_coordinates[1], 1,
                atom.xyz_coordinates[2], 1)
            fmndo.writelines(line) 
        if (nstates > 1) and any(calculate_energy_gradients):
            fmndo.writelines(' 0       0.00000000   0       0.00000000   0       0.00000000   0\n')
            energy_gradients_true_states = [str(index+1) for index, value in enumerate(calculate_energy_gradients) if value]
            fmndo.writelines(' '.join(energy_gradients_true_states))
        # if ('mndo_keywords' in molecule.__dict__.keys()) and (molecule.mndo_keywords.find('ncigrd') != -1):
        #     fmndo.writelines(molecule.mndo_keywords[(molecule.mndo_keywords.find('extra lines for ncigrd:\n') + len('extra lines for ncigrd:\n')):])
        fmndo.writelines('\n')

def parse_mndo_output(filename=None, molecule=None):
    """
    Parse MNDO output file and return a molecule obj
    :param filename: Name of Gaussian output file (e.g. benzene.log or benzene.out)
    :return: molecule if molecule is not provided
    """
    
    if molecule is not None:
        mol = molecule
    else:
        mol = data.molecule()
    tmpdirname = os.path.dirname(os.path.realpath(filename))
    
    outputs = []
    nstates = 1
    current_state = None
    
    def extract_keyword_from_output_line(keyword, line):
        parts = line.split()
        # parts looks like ["IOP", "=", "-22", "JOP", "=", "-2", ...]
        for i in range(0, len(parts) - 2, 3):  # step by 3: keyword, =, value
            if parts[i] == keyword and parts[i+1] == "=":
                return int(parts[i+2])
         
    with open(filename, 'r') as fout:
        for readable in fout:
            outputs.append(readable)
            if 'IROOT' in readable:
                nstates = extract_keyword_from_output_line('IROOT', readable)
            if 'LROOT' in readable:
                current_state = extract_keyword_from_output_line('LROOT', readable) - 1
            if "KCI" in readable:
                kci = extract_keyword_from_output_line('KCI', readable) - 1
            else:
                kci=-1
    
    # find flags
    output_multi_ens_flag = 'E='
    ffort15_en_flag = ' ENERGY, CARTESIAN AND INTERNAL GRADIENT NORM'
    ffort15_ens_flag = ' STATES, ENERGIES'
    ffort15_grad_flag = ' CARTESIAN GRADIENT'
    ffort15_grads_flag = ' CARTESIAN GRADIENT FOR STATE '
    ffort15_coupling_flag = ' CARTESIAN INTERSTATE COUPLING GRADIENT FOR STATES'
    output_os_flag = 'f_r'
    found_ffort15_coord_flag = False
    found_output_multi_ens_flag = False
    found_ffort15_en_flag = False
    found_ffort15_ens_flag = False
    found_ffort15_grad_flag = False
    found_ffort15_grads_flag = False
    found_ffort15_coupling_flag = False
    found_output_os_flag = False
    output_charges_index = -1
    state_dipole_flag = 'State dipole moments'
    found_state_dipole_flag = False
    
    for i, line in enumerate(outputs):
        line = line.rstrip('\n')
        if output_multi_ens_flag in line:
            found_output_multi_ens_flag = True    
            try:
                output_ens_index.append(i)
            except NameError:
                output_ens_index = [i]
            continue
        if output_os_flag in line:
            found_output_os_flag = True
            output_os_index = i+1
            found_output_os_end_flag = False
            continue
        if found_output_os_flag and (not found_output_os_end_flag):
            if line.strip() == '':
                output_os_end_index = i-1
                found_output_os_end_flag = True
            continue
        if 'NET ATOMIC CHARGES AND ORBITAL POPULATIONS.' in line:
            output_charges_index = i+4
            continue
        if state_dipole_flag in line:
            found_state_dipole_flag = True
            output_dipole_index = i+3
            found_state_dipole_end_flag = False
        if found_state_dipole_flag and (not found_state_dipole_end_flag) and i>output_dipole_index:
            if line.strip() == '':
                output_dipole_end_index= i-1
                found_state_dipole_end_flag = True

    with open(f'{tmpdirname}/fort.15', 'r') as ffort15:
        ffort15_lines = ffort15.readlines()
    
    for i, line in enumerate(ffort15_lines):
        line = line.rstrip('\n')
        if 'CARTESIAN COORDINATES: NUMAT =' in line:
            natoms = int(line.split()[-1])
            found_ffort15_coord_flag = True
            ffort15_coord_index = i+1
            continue
        if line == ffort15_en_flag:
            found_ffort15_en_flag = True
            ffort15_en_index = i+1
            continue
        if ffort15_ens_flag in line:
            found_ffort15_ens_flag = True
            found_ffort15_ens_end_flag = False
            continue
        if found_ffort15_ens_flag and (not found_ffort15_ens_end_flag):
            if line.strip() == '':
                found_ffort15_ens_end_flag = True
            continue
        if line == ffort15_grad_flag:
            found_ffort15_grad_flag = True
            ffort15_grad_index = i+1
            continue
        if ffort15_grads_flag in line:
            found_ffort15_grads_flag = True
            try:
                ffort15_grads_index.append(i+1)
            except NameError:
                ffort15_grads_index = [i+1]
            continue
        if ffort15_coupling_flag in line:
            found_ffort15_coupling_flag = True
            try:
                ffort15_coupling_index.append(i+1)
            except NameError:
                ffort15_coupling_index = [i+1]
            continue
    
    # read atom coordinates
    if mol.atoms == []:
        if found_ffort15_coord_flag:
            energy_gradient = []
            for line in ffort15_lines[ffort15_coord_index: ffort15_coord_index+natoms]:
                xx = line.split()
                mol.atoms.append(data.atom(nuclear_charge=int(xx[1]), xyz_coordinates=np.array([float(yy) for yy in xx[2:]]).astype(float)))
    
    # read properties
    # read energy/energies
    if found_ffort15_en_flag:
        energy = ffort15_lines[ffort15_en_index].split()[0]
        energy = float(energy) / (27.21 * 23.061)
        # mol.__dict__[energy_label] = energy
        mol.energy = energy
    
    if nstates > 1:
        if found_output_multi_ens_flag:
            state_energies = []
            for idx in output_ens_index:
                parts = outputs[idx].split('E=')
                if len(parts) == 2:
                    energy = parts[1].split()[0]
                    energy = float(energy) / 27.21
                    state_energies.append(energy)
            mol_copy = mol.copy()
            mol_copy.electronic_states = []
            for _ in range(nstates - len(mol.electronic_states)):
                mol.electronic_states.append(mol_copy.copy())
            for i in range(nstates):
                mol.electronic_states[i].energy = state_energies[i]
            # mol.__dict__[energy_label] = mol.electronic_states[current_state].energy
            if current_state: mol.energy = mol.electronic_states[current_state].energy

    if found_output_os_flag:
        # read oscillator strengths
        oscillator_strengths = []
        for line in outputs[output_os_index: output_os_end_index+1]:
            oscillator_strength = float(line.split()[6])
            oscillator_strengths.append(oscillator_strength)
        mol.oscillator_strengths = oscillator_strengths
    if found_state_dipole_flag:
        state_dipole_moments = []
        for line in outputs[output_dipole_index: output_dipole_end_index+1]:
            splitline = line.split()
            state_dipole_moments.append(np.array([splitline[5], splitline[6], splitline[7]]))
        for idx, istate in enumerate(mol.electronic_states):
            if idx == 0 and kci == 6:
                istate.dipole_moment = np.nan # due to bug in kci=6 state dipole moment for s0
            else: 
                try:
                    istate.dipole_moment = state_dipole_moments[idx].astype('float64')
                except:
                    pass
    # read gradients
    energy_gradient = []
    if found_ffort15_grad_flag:
        for line in ffort15_lines[ffort15_grad_index: ffort15_grad_index+natoms]:
            energy_gradient_per_atom = [float(xx) / (27.21 * 23.061) for xx in line.split()[-3:]]
            energy_gradient.append(energy_gradient_per_atom)
        mol.add_xyz_derivative_property(np.array(energy_gradient).astype(float), 'energy', 'energy_gradients')

    state_gradients = {}
    if found_ffort15_grads_flag:
        for idx in ffort15_grads_index:
            energy_gradient = []
            for line in ffort15_lines[idx: idx+natoms]:
                energy_gradient_per_atom = [float(xx) / (27.21 * 23.061) for xx in line.split()[-3:]]
                energy_gradient.append(energy_gradient_per_atom)
            state_gradients[int(ffort15_lines[idx-1].split()[4])-1] = energy_gradient
        if not mol.electronic_states:
            mol.electronic_states.extend([mol.copy() for _ in range(nstates)])
        for istate in state_gradients.keys():
            mol.electronic_states[istate].add_xyz_derivative_property(np.array(state_gradients[istate]).astype(float), 'energy', 'energy_gradients')
        if current_state: mol.add_xyz_derivative_property(np.array(state_gradients[current_state]).astype(float), 'energy', 'energy_gradients')     
    
    if nstates > 1 and mol.electronic_states and energy_gradient and state_gradients == {} and current_state:
        mol.electronic_states[current_state].energy_gradients = mol.energy_gradients

    # read interstate coupling gradient(s) ⟨ψ_i|∂H/∂R|ψ_j⟩ and calculate nonadiabatic coupling vector(s) ⟨ψ_i|∂H/∂R|ψ_j⟩/(E_i-E_j)
    if found_ffort15_coupling_flag:
        nacv = {}
        for idx in ffort15_coupling_index:
            nonadiabatic_coupling_vector = []  
            initial_state = int(ffort15_lines[idx-1].split()[6])-1
            final_state = int(ffort15_lines[idx-1].split()[7])-1
            try:
                gap = mol.electronic_states[initial_state].energy - mol.electronic_states[final_state].energy
            except IndexError:
                print('Make sure that the energies of [%d]th and [%d]th roots are available!' % (initial_state+1, final_state+1))
            for line in ffort15_lines[idx: idx+natoms]:
                nonadiabatic_coupling_per_atom = [float(xx) / (27.21 * 23.061 * gap) for xx in line.split()[-3:]]
                nonadiabatic_coupling_vector.append(nonadiabatic_coupling_per_atom)
            nacv[(initial_state, final_state)] = nonadiabatic_coupling_vector
        
        state_comb = [(i, j) for i in range(1, len(output_ens_index)) for j in range(0, i)]
        mol.nacv = [[np.tile(np.zeros(3), (natoms, 1)) for ii in range(len(output_ens_index))] for jj in range(len(output_ens_index))]
        for index, (initial_state, final_state) in enumerate(state_comb):
            try:
                mol.nacv[initial_state][final_state] = np.array(nacv[(initial_state, final_state)]).astype(float)
                mol.nacv[final_state][initial_state] = -np.array(nacv[(initial_state, final_state)]).astype(float)
            except KeyError:
                mol.nacv[initial_state][final_state] = None
                mol.nacv[final_state][initial_state] = None
    
    if nstates == 1:
        # read hessian(s)
        hess_file = f'{tmpdirname}/fort.4'
        if os.path.exists(hess_file):
            import struct
            fhess = open(f'{tmpdirname}/fort.4', 'rb')
            lhess = natoms * natoms * 9
            hessdata = fhess.read()
            dt = f'id{lhess}d'
            dat_size = struct.calcsize(dt)
            temp = struct.unpack(dt, hessdata[:dat_size])
            hess = np.array(temp[2:]).astype(float) / (constants.Bohr2Angstrom**2)
            mol.hessian = hess.reshape(natoms*3,natoms*3)

    # read dipole moment
    if nstates == 1:
        for iline in range(len(outputs)):
            if 'DIPOLE      ' in outputs[iline]:
                sum_line = outputs[iline+4].rstrip()
                sum_line = sum_line.split(' ')
                sum_line = [each for each in sum_line if each != '' ][1:]
                dipole_moment = np.array([eval(each) for each in sum_line])
                mol.dipole_moment = dipole_moment    
            if 'DIPOLE DERIVATIVES' in outputs[iline]:
                if 'CARTESIAN COORDINATES' in outputs[iline+1]:    
                    dipole_derivatives = []
                    flag = iline+6
                    icount = 1
                    while icount <= 3*len(mol):
                        flag += 1
                        if outputs[flag].strip() == '':
                            pass 
                        else:
                            dipole_derivatives.append([eval(each) for each in outputs[flag].strip().split()[1:]])
                            icount += 1
                    ######################### old way to store data #########################
                    # excited states may have dipole derivatives
                    mol.dipole_derivatives = np.array(dipole_derivatives).reshape(9*len(mol))    
    
    if output_charges_index > -1:
        for ii in range(natoms):
            mol.atoms[ii].atomic_charge = float(outputs[output_charges_index + ii].split()[2])
    
    if molecule is None:
        return mol                   

if __name__ == '__main__':
    pass