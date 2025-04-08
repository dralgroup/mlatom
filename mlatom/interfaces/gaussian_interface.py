#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gaussian_interface: interface to the Gaussian program                     ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, sys, subprocess
import numpy as np

from .. import constants, data
from ..model_cls import method_model
from ..decorators import doc_inherit

class gaussian_methods(method_model):
    '''
    Gaussian interface

    Arguments:
        method (str): Method to use
        chkfilename (bool, str): request to create chk file defined by %chk in Gaussian (default: None). If True, assignes default filename.
        memory (str): equivalent to %mem in Gaussian input file, e.g., memory=16Gb (default: None).
        nthreads (int): equivalent to %proc in Gaussian input file.
        gaussian_keywords (str): any gaussian keywords in addition to method.
        additional_input (str): any additional input to be appended to the Gaussian input file.
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'False'``.
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``.

    .. note::

        The format of method should be the same as that in Gaussian, e.g., ``'B3LYP/6-31G*'``
        
    '''
    
    def __init__(self,
                 method='B3LYP/6-31G*',
                 gaussian_keywords=None, additional_input='',
                 chkfilename=None,
                 memory=None,
                 nthreads=None,save_files_in_current_directory=False,
                 working_directory=None,
                 ):
        if not "GAUSS_EXEDIR" in os.environ:
            raise ValueError('enviromental variable GAUSS_EXEDIR is not set')
        self.method = method
        self.gaussian_keywords = gaussian_keywords
        self.additional_input = additional_input
        self.chkfilename = chkfilename
        if nthreads is None:
            from multiprocessing import cpu_count
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads
        self.memory = memory
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        _, _ = check_gaussian()

    @classmethod
    def is_program_found(cls):
        try:
            _, _ = check_gaussian()
            return True
        except:
            return False

    @doc_inherit
    def predict(self,molecular_database=None,molecule=None,
                nstates=1, 
                current_state=0, 
                calculate_energy=True,
                calculate_energy_gradients=False,
                calculate_hessian=False,
                calculate_dipole_derivatives=False,
                gaussian_keywords='',):
        '''
            nstates (int):                 number of electronic structure states (default: 1, ground state)
            current_state (int):           default is the ground state (for nstates=1) or the first excited state (nstates > 1)
            gaussian_keywords (str): any gaussian keywords.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)           
        method = self.method
        #self.energy_to_read = energy_to_read
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_energy = calculate_energy
        self.calculate_hessian = calculate_hessian
        self.calculate_dipole_derivatives = calculate_dipole_derivatives
        self.gaussian_keywords = gaussian_keywords
        if nstates > 1:
            if 'TD-'.casefold() in method.casefold():
                import re
                method = re.sub('td-', '', method, flags=re.IGNORECASE)
                if current_state == 0: current_state = 1 # default in Gaussian
                method += f' TD(NStates={nstates-1},Root={current_state})'
        if calculate_energy_gradients:
            if not calculate_hessian:
                if 'force'.casefold() in method.casefold():
                    pass 
                else:
                    method += ' force(nostep)'
        if calculate_hessian:
            self.chkfilename = True
            if 'freq'.casefold() in method.casefold():
                pass 
            else:
                method += ' freq'     
            if calculate_dipole_derivatives:
                method += ' IOp(7/33=1)'

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory: tmpdirname = '.'
            if self.working_directory is not None:
                tmpdirname = self.working_directory
                if not os.path.exists(tmpdirname):
                    os.makedirs(tmpdirname)
                tmpdirname = os.path.abspath(tmpdirname)
            for imol in range(len(molDB.molecules)):
                imolecule = molDB.molecules[imol]
                filename_wo_extension = 'molecule'+str(imol)
                if self.chkfilename == True:
                    chkfilename=f'{filename_wo_extension}.chk'
                elif type(self.chkfilename) == str:
                    chkfilename = self.chkfilename
                else:
                    chkfilename=None
                # Run Gaussian job
                run_gaussian_job(filename=f'{filename_wo_extension}.com',molecule=imolecule,
                                 method=method,
                                 gaussian_keywords=self.gaussian_keywords,additional_input=self.additional_input,
                                 chkfilename=chkfilename,
                                 memory=self.memory,
                                 nthreads=self.nthreads,
                                 working_directory=tmpdirname,
                                 )

                # Read Gaussian output file
                parse_gaussian_output(filename=os.path.join(tmpdirname,f'{filename_wo_extension}.log'),molecule=imolecule)
            
def run_gaussian_job(filename=None,
                     molecule=None, reactants=None, products=None,
                     gaussian_keywords='',
                     nthreads=1, memory=None,
                     working_directory='.',
                     # The block below is for using Gaussian for SP calculations
                     method=None,
                     chkfilename=None,
                     # The block below is for using Gaussian as engine for jobs with external potential
                     external_task=None,
                     model_predict_kwargs={},
                     extra_keywords='', # string with extra keywords
                     additional_input='',
                     opt_keywords = ['nomicro'],
                     freq_keywords=[], # list with additional arguments such as ['NoRaman',] etc.
                     irc_keywords=['CalcFC']
                     ):
        
    if external_task is not None:
        pythonbin = sys.executable
        path_to_this_file=os.path.abspath(__file__)
        path_to_gaussian_external = os.path.join(os.path.dirname(path_to_this_file), 'gaussian_external.py')
        
        model_predict_kwargs_str = str(model_predict_kwargs)
        model_predict_kwargs_str_file = 'model_predict_kwargs'
        with open(os.path.join(working_directory, model_predict_kwargs_str_file), 'w') as f:
            f.write(model_predict_kwargs_str)
        external_command = f"external='{pythonbin} {path_to_gaussian_external} {model_predict_kwargs_str_file}'"
        if external_task.casefold() == 'opt':
            if len(opt_keywords) > 0:
                opt_str = ','.join(opt_keywords)
                if not 'nomicro' in opt_str.casefold():
                    print("Warning: please specify opt_keywords=['nomicro'] unless you know what you are doing.")
                gaussian_keywords = f"opt({opt_str})"
            else:
                gaussian_keywords = "opt"
        elif 'freq' in external_task.casefold():
            if len(freq_keywords) != 0:
                freq_str = ','.join(freq_keywords)
                if external_task.casefold() == 'freq(anharmonic)':
                    freq_str = f',{freq_str}'
                    gaussian_keywords = f"freq(anharmonic{freq_str})"
                elif external_task.casefold() == 'freq':
                    gaussian_keywords = f"freq({freq_str})"
            else:
                if external_task.casefold() == 'freq(anharmonic)':
                    gaussian_keywords = f"Freq(anharmonic)"
                elif external_task.casefold() == 'freq':
                    gaussian_keywords = f"Freq"
                else:
                    gaussian_keywords = f"{external_task}"
        elif external_task.casefold() in ['ts', 'qst2', 'qst3']:
            if len(opt_keywords) > 0:
                if not 'nomicro' in [kwd.casefold() for kwd in opt_keywords]:
                    print("Warning: please specify opt_keywords=['nomicro'] unless you know what you are doing.")
                if len(opt_keywords) == 1 and opt_keywords[0].casefold() == 'nomicro':
                    if external_task.casefold() == 'ts':
                        opt_keywords = ['CalcFC', 'noeigen', 'nomicro']
                    else:
                        opt_keywords = ['CalcFC', 'nomicro']
                if not external_task.casefold() in [kwd.casefold() for kwd in opt_keywords]:
                    opt_keywords = [external_task] + opt_keywords
            else:
                opt_keywords = ['TS', 'CalcFC', 'noeigen', 'nomicro']
            opt_str = ','.join(opt_keywords)
            gaussian_keywords = f"Opt({opt_str})"
        elif external_task.casefold() == 'irc':
            if len(irc_keywords) > 0:
                irc_str = ','.join(irc_keywords)
                gaussian_keywords = f"irc({irc_str})"
            else:
                gaussian_keywords = "irc"
        if len(extra_keywords) > 0:
            gaussian_keywords += f' {extra_keywords}'
        gaussian_keywords += f'\n{external_command}'
    else:
        if method is not None:
            gaussian_keywords += f'{method}'
     
    if not gaussian_keywords.strip()[:1] == '#':
        if external_task is not None or 'freq' in gaussian_keywords:
            gaussian_keywords = f'#p {gaussian_keywords}'
        else:
                gaussian_keywords = f'# {gaussian_keywords}'

    if memory is not None:
        gaussian_keywords = f"%mem={memory}\n" + gaussian_keywords
    gaussian_keywords = f'%nproc={nthreads}\n' + gaussian_keywords 
    if chkfilename is not None:
        gaussian_keywords = f'%chk={chkfilename}\n' + gaussian_keywords
    
    write_gaussian_input_file(filename=os.path.join(working_directory,filename),
                              molecule=molecule, reactants=reactants, products=products,
                              gaussian_keywords=gaussian_keywords,additional_input=additional_input)
    Gaussianbin, _ = check_gaussian()
    proc = subprocess.Popen([Gaussianbin, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_directory, universal_newlines=True)
    proc.communicate()
    
def check_gaussian():
    status = os.popen('echo $GAUSS_EXEDIR').read().strip()
    if len(status) != 0:
        Gaussianroot = status.split('bsd')[0]
        if 'g16' in Gaussianroot:
            version = 'g16'
        elif 'g09' in Gaussianroot:
            version = 'g09'
        Gaussianbin = os.path.join(Gaussianroot,version)
    else:
        raise ValueError('Cannot find Gaussian software in the environment, set $GAUSS_EXEDIR environmental variable')
    version = version.replace('g', '')
    return Gaussianbin, version

def write_gaussian_input_file(filename=None,
                              molecule=None, reactants=None, products=None,
                              gaussian_keywords='', additional_input=''):
    
    input_string = f'{gaussian_keywords}\n'
    
    def mol_str(molecule, title='molecule'):
        value = ''
        if 'comment' in molecule.__dict__:
            if molecule.comment != '':
                title = molecule.comment
        elif molecule.id != '':
            title = molecule.id
        value += f'\n{title}\n'
        value += f'\n{molecule.charge} {molecule.multiplicity}\n'
        value += molecule.get_xyz_string(only_coordinates=True)
        return value
    
    if reactants is not None:
        input_string += mol_str(reactants, title='reactants')
    if products is not None:
        input_string += mol_str(products, title='products')
    if molecule is not None:
        input_string += mol_str(molecule)
    
    if len(additional_input) > 0:
        input_string += f'\n{additional_input}'
    input_string += '\n\n'
    if filename is None:
        return input_string
    with open(filename, 'w') as f:
        f.writelines(input_string)

def read_Hessian_matrix_from_Gaussian_chkfile(filename,molecule):
    natoms = len(molecule.atoms)
    hessian = np.zeros((3*natoms,3*natoms))
    cmd = f'chkchk -p {filename}'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    outs,errs = proc.communicate()
    stdout = outs.split('\n')
    stderr = errs.split('\n')
    outputs = []
    for readable in stdout:
        outputs.append(readable)
    flag = None
    for iline in range(len(outputs)):
        if 'Input for Opt=FCCards' in outputs[iline]:
            flag = iline+1 
    # If there is only 1 atom, chk does not have hessian matrix
    if flag is None:
        molecule.hessian = hessian
    else:
        flag += 1 + int((3*natoms-0.5)//6 + 1)# Skip energy and forces
        hessian_len = int(((3*natoms)*(3*natoms+1)//2-0.5)//6 + 1)
        hessian_array = []
        for ii in range(hessian_len):
            hessian_array += [eval(each) for each in outputs[flag+ii].strip().split()]
        #print(hessian_array)
        for ii in range(3*natoms):
            hessian[:ii+1,ii] = hessian_array[(ii+1)*ii//2:(ii+2)*(ii+1)//2]
        for ii in range(3*natoms):
            for jj in range(ii+1,3*natoms):
                hessian[jj,ii] = hessian[ii,jj]
        #print(hessian[:,-2])
        hessian /= constants.Bohr2Angstrom**2
        molecule.hessian = hessian

def parse_gaussian_output(filename=None, molecule=None):
    """
    Parsing Gaussian output file and return a molecule obj
    :param filename: Name of Gaussian output file (e.g. benzene.log or benzene.out)
    :return: molecule if molecule is not provided
    """
    
    if molecule is not None:
        mol = molecule
    else:
        mol = data.molecule()
    
    #successful = False
    readArchive = False
    chkfilename = None
    archive = []
    forces_iatom = -4
    i_read_hessian = -1
    mulliken_charge_iatom = -3
    read_dipole_moment = False
    readNMs = False
    ireadS = -1
    es_mults = []
    es_contribs = []
    readES = False
    i_read_input_orientation = -1
    i_read_standard_orientation = -1
    xyz_input_orientation = []
    xyz_standard_orientation = []
    scf_energies = []
    energy_gradients = []
    Ucase = False
    readAlpha = False
    readOcc = False
    nOMOs = {'all': 0, 'A': 0, 'B': 0}
    # flags to check whether a property was never read before - by default overwrite everything
    read_already_freq = False
    read_already_redmass = False
    read_already_force_const = False
    read_already_ir_intens = False
    read_already_ES = False
    read_already_FB = False
    read_already_overtones = False
    read_already_comb_bands = False
    read_already_FBIR = False
    read_already_OvertonesIR = False
    read_already_CombinationBandsIR = False
    # start anharmonic properties
    ireadFB = -1
    ireadOvertones = -1
    ireadCombinationBands = -1
    i_anh_thermo = -1
    ireadFBIR = -1
    ireadOvertonesIR = -1
    ireadCombinationBandsIR = -1
    FBIRError = False
    OvertonesIRError = False 
    CombinationBandsIRError = False
    # end anharmonic properties

    with open(filename, 'r') as ff:
        for line in ff:
            #print(line)
            if ' 1\\1\\' in line or ' 1|1|' in line:
                readArchive = True
                archive = [line.rstrip()]
                continue
            if readArchive:
                if '\\\\@' in line or '||@' in line:
                    readArchive = False
                archive.append(line.rstrip())
                continue
            if '%chk=' in line:
                chkfilename = line.split('=')[-1].strip()
                if not '.chk' in chkfilename:
                    chkfilename += '.chk'
            if 'Input orientation:' in line:
                i_read_input_orientation = 0
                xyz_input_orientation.append([])
                continue
            if i_read_input_orientation > -1:
                i_read_input_orientation += 1
                if i_read_input_orientation >= 5:
                    if '-------------------------' in line:
                        i_read_input_orientation = -1
                        continue
                    xyz_input_orientation[-1].append([float(xx) for xx in line.split()[-3:]])
                    continue
            if 'Standard orientation:' in line:
                i_read_standard_orientation = 0
                xyz_standard_orientation.append([])
                continue
            if i_read_standard_orientation > -1:
                i_read_standard_orientation += 1
                if i_read_standard_orientation >= 5:
                    if '-------------------------' in line:
                        i_read_standard_orientation = -1
                        continue
                    xyz_standard_orientation[-1].append([float(xx) for xx in line.split()[-3:]])
                    continue
            if 'SCF Done:' in line:
                scf_energies.append(float(line.split()[4]))
            if 'Forces (Hartrees/Bohr)' in line:
                forces_iatom = -3
                energy_gradients.append([])
                continue
            if forces_iatom > -4:
                forces_iatom += 1
                if forces_iatom < 0: continue
                if '------------------' in line:
                    forces_iatom = -4
                    continue
                energy_gradients[-1].append(-data.array([float(each) for each in line.split()[2:]]) / constants.Bohr2Angstrom)
                continue
            if 'The second derivative matrix:' in line:
                i_read_hessian = 0
                natoms = len(mol.atoms)
                mol.hessian = np.zeros((3*natoms,3*natoms))
                continue
            if i_read_hessian > -1:
                i_read_hessian += 1
                nblocks = int((3*natoms-0.5)//5 + 1)
                icount = 0
                for iblock in range(nblocks):
                    for ii in range(3*natoms-5*iblock):
                        if i_read_hessian == 1+icount+ii+1:
                            temp = [float(each) for each in line.strip().split()[1:]]
                            for jj in range(min(5,3*natoms-5*iblock)):
                                if ii >= jj:
                                    mol.hessian[ii+5*iblock,jj+5*iblock] = temp[jj]
                    icount += 3*natoms-5*iblock+1
                if i_read_hessian >= icount:
                    i_read_hessian = -1
                    for ii in range(3*natoms):
                        for jj in range(ii+1,3*natoms):
                            mol.hessian[ii,jj] = mol.hessian[jj,ii]
                    mol.hessian = mol.hessian / constants.Bohr2Angstrom**2
                continue
            if 'Mulliken charges:' in line:
                mulliken_charge_iatom = -2
                continue
            if mulliken_charge_iatom > -3:
                if len(line.split()) > 3:
                    mulliken_charge_iatom = -3
                    continue
                mulliken_charge_iatom += 1
                if mulliken_charge_iatom < 0:
                    continue
                if len(mol.atoms) < mulliken_charge_iatom+1:
                    mol.atoms.append(data.atom(element_symbol=line.split()[1]))
                mol.atoms[mulliken_charge_iatom].mulliken_charge = float(line.split()[-1])
                continue
            if 'Dipole moment' in line:
                read_dipole_moment = True
                continue
            if read_dipole_moment:
                xx = line.split()
                try:
                    mol.dipole_moment = data.array([float(yy) for yy in [xx[1], xx[3], xx[5], xx[7]]])
                except:
                    pass
                read_dipole_moment = False
                continue
            if 'Frequencies --' in line:
                if not read_already_freq:
                    read_already_freq = True
                    mol.frequencies = []
                mol.frequencies += [float(xx) for xx in line.split()[2:]]
                continue
            if 'Red. masses --' in line:
                if not read_already_redmass:
                    read_already_redmass = True
                    mol.reduced_masses = []
                mol.reduced_masses += [float(xx) for xx in line.split()[3:]]
                continue
            if 'Frc consts  --' in line:
                if not read_already_force_const:
                    read_already_force_const = True
                    mol.force_constants = []
                mol.force_constants += [float(xx) for xx in line.split()[3:]]
                continue
            if 'IR Inten    --' in line:
                if not read_already_ir_intens:
                    read_already_ir_intens = True
                    mol.infrared_intensities = []
                mol.infrared_intensities += [float(xx) for xx in line.split()[3:]]
                continue
            if 'Atom  AN      X      Y      Z' in line:
                readNMs = True
                nm_iatom = -1
                continue
            elif readNMs:
                if len(line.split()) <= 3:
                    readNMs = False
                    continue
                nm_iatom += 1
                if len(mol.atoms) < nm_iatom+1:
                    mol.atoms.append(data.atom(atomic_number=int(line.split()[1])))
                if not 'normal_modes' in mol.atoms[nm_iatom].__dict__:
                    mol.atoms[nm_iatom].normal_modes = []
                xyzs = np.array([float(xx) for xx in line.split()[2:]])
                nxyzs = len(xyzs) / 3
                for xyz in np.array_split(xyzs, nxyzs):
                    mol.atoms[nm_iatom].normal_modes.append(list(xyz))
            if 'Zero-point correction=' in line:
                mol.ZPE = float(line.split()[2])
                continue
            if 'Thermal correction to Energy=' in line:
                mol.DeltaE2U = float(line.split()[-1])
                continue
            if 'Thermal correction to Enthalpy=' in line:
                mol.DeltaE2H = float(line.split()[-1])
                continue
            if 'Thermal correction to Gibbs Free Energy=' in line:
                mol.DeltaE2G = float(line.split()[-1])
                continue
            if 'Sum of electronic and zero-point Energies=' in line:
                mol.U0 = float(line.split()[-1])
                mol.H0 = mol.U0
                continue
            if 'Sum of electronic and thermal Energies=' in line:
                mol.U = float(line.split()[-1])
                continue
            if 'Sum of electronic and thermal Enthalpies=' in line:
                mol.H = float(line.split()[-1])
                continue
            if 'Sum of electronic and thermal Free Energies=' in line:
                mol.G = float(line.split()[-1])
                continue
            if 'E (Thermal)             CV                S' in line:
                ireadS = 0
                continue
            if ireadS > -1:
                ireadS +=1
                if ireadS == 2:
                    mol.S = float(line.split()[-1]) * constants.kcalpermol2Hartree / 1000.0
                    ireadS = -1
                    continue
            
            if 'Alpha Orbitals:' in line:
                Ucase = True
            if ' Occupied ' in line:
                readOcc = True
                nOMOs = {'all': 0, 'A': 0, 'B': 0}
            if readOcc:
                if ' Virtual ' in line:
                    readOcc = False
                    if Ucase and not readAlpha:
                        readAlpha = True
                        #continue
                    # to-do
                    #else:
                    #    break
                    continue
                if Ucase and not readAlpha:
                    nOMOs['A'] += len(line[17:].split())
                elif Ucase and readAlpha:
                    nOMOs['B'] += len(line[17:].split())
                else:
                    nOMOs['all'] += len(line[17:].split())
                continue
            
            if 'Excited State' in line:                
                xx = line.split()
                excitation_energy = float(xx[4]) * constants.eV2hartree
                ff = float(xx[8].split('=')[1])
                mult = xx[3]
                if not read_already_ES:
                    read_already_ES = True
                    mol.electronic_states = []
                    mol.excitation_energies = [excitation_energy]
                    mol.oscillator_strengths = [ff]
                else:
                    mol.excitation_energies += [excitation_energy]
                    mol.oscillator_strengths.append(float(xx[8].split('=')[1]))
                readES = True
                es_mults.append(xx[3])
                es_contribs.append([])
                continue
            if readES:
                if '->' in line:
                    es_contribs[-1].append({'from': int(line.split('->')[0]),
                                            'to':   int(line.split('->')[1].split()[0]),
                                            'coeff': float(line.split()[-1])})
                elif '<-' in line:
                    es_contribs[-1].append({'from': int(line.split('<-')[1].split()[0]),
                                            'to':   int(line.split('<-')[0]),
                                            'coeff': float(line.split()[-1])})
                else:
                    readES = False
                continue
            if 'Fundamental Bands' in line and not 'anharmonic_frequencies' in mol.__dict__:
                read_already_FB = True
                ireadFB += 1
                mol.anharmonic_frequencies = []
                mol.harmonic_frequencies = []
                continue
            elif 'Fundamental Bands' in line and not read_already_FB:
                read_already_FB = True
                ireadFB += 1
                mol.anharmonic_frequencies = []
                mol.harmonic_frequencies = []
                continue
            if ireadFB > -1:
                if len(line.strip()) == 0:
                    ireadFB = -1
                    continue
                ireadFB += 1
                if ireadFB >= 3:
                    mol.harmonic_frequencies.append(float(line[24:38]))
                    mol.anharmonic_frequencies.append(float(line[38:48]))
                    continue
            if 'Overtones' in line and not 'anharmonic_overtones' in mol.__dict__:
                read_already_overtones = True
                ireadOvertones += 1
                mol.anharmonic_overtones = []
                mol.harmonic_overtones = []
                continue
            elif 'Overtones' in line and not read_already_overtones:
                read_already_overtones = True
                ireadOvertones += 1
                mol.anharmonic_overtones = []
                mol.harmonic_overtones = []
                continue
            if ireadOvertones > -1:
                if len(line.strip()) == 0:
                    ireadOvertones = -1
                    continue
                ireadOvertones += 1
                if ireadOvertones >= 3:
                    mol.anharmonic_overtones.append(float(line[38:48]))
                    mol.harmonic_overtones.append(float(line[24:38].strip().split()[-1]))
                    continue
            if 'Combination Bands' in line and not 'anharmonic_combination_bands' in mol.__dict__:
                read_already_comb_bands = True
                ireadCombinationBands += 1
                mol.anharmonic_combination_bands = []
                mol.harmonic_combination_bands = []
                continue
            elif 'Combination Bands' in line and not read_already_comb_bands:
                read_already_comb_bands = True
                ireadCombinationBands += 1
                mol.anharmonic_combination_bands = []
                mol.harmonic_combination_bands = []
                continue
            if ireadCombinationBands > -1:
                if len(line.strip()) == 0:
                    ireadCombinationBands = -1
                    continue
                ireadCombinationBands += 1
                if ireadCombinationBands >= 3:
                    mol.anharmonic_combination_bands.append(float(line[38:48]))
                    mol.harmonic_combination_bands.append(float(line[24:38].strip().split()[-1]))
                    continue
            if 'ZPE(anh)' in line:
                mol.anharmonic_ZPE = float(line[48:61].replace('D','E')) * constants.kJpermol2Hartree
            if 'Input values of T(K) and P(atm):' in line:
                i_anh_thermo = 0
                mol.temperature = float(line.strip().split()[-2])
                E = mol.U - mol.DeltaE2U
                mol.anharmonic_U0 = E + mol.anharmonic_ZPE
                mol.anharmonic_H0 = E + mol.anharmonic_ZPE
                continue
            if i_anh_thermo > -1:
                i_anh_thermo += 1
                if i_anh_thermo == 4:
                    mol.anharmonic_DeltaE2U = float(line.strip().split()[-2].replace('D','E')) * constants.kJpermol2Hartree
                    mol.anharmonic_U = E + mol.anharmonic_DeltaE2U
                    continue
                if i_anh_thermo == 5:
                    mol.anharmonic_DeltaE2H = float(line.strip().split()[-2].replace('D','E')) * constants.kJpermol2Hartree
                    mol.anharmonic_H = E + mol.anharmonic_DeltaE2H
                    continue
                if i_anh_thermo == 6:
                    mol.anharmonic_S = float(line.strip().split()[-3].replace('D','E')) * constants.kJpermol2Hartree / 1000.0
                    mol.anharmonic_G = mol.anharmonic_H - mol.temperature * mol.anharmonic_S
                    mol.anharmonic_DeltaE2G = mol.anharmonic_G - E
                    i_anh_thermo = -1
                    continue
            
            # Read anharmonic infrared intensities
            if 'Fundamental Bands' in line and not 'anharmonic_infrared_intensities' in mol.__dict__:
                read_already_FBIR = True
                ireadFBIR += 1
                mol.anharmonic_infrared_intensities = []
                mol.harmonic_infrared_intensities = []
                continue
            elif 'Fundamental Bands' in line and not read_already_FBIR:
                read_already_FBIR = True
                ireadFBIR += 1
                mol.anharmonic_infrared_intensities = []
                mol.harmonic_infrared_intensities = []
                continue 
            if ireadFBIR > -1:
                if len(line.strip()) == 0:
                    ireadFBIR = -1
                    continue 
                ireadFBIR += 1
                if ireadFBIR == 2:
                    if not "I(anharm)" in line:
                        del mol.__dict__['anharmonic_infrared_intensities']
                        del mol.__dict__['harmonic_infrared_intensities']
                        ireadFBIR = -1
                    continue 
                if ireadFBIR >= 3:
                    templine = line.strip()
                    # Sometimes anharmonic IR intensities are very large due to bad PES. Gaussian will output ********** in such cases.
                    if "*" in templine:
                        templine = [each for each in templine if each != '*']
                        templine = templine.split()
                        try:
                            mol.harmonic_infrared_intensities.append(float(templine[-1]))
                            mol.anharmonic_infrared_intensities.append(np.nan)
                        except:
                            FBIRError = True 
                    else:
                        templine = templine.split()
                        try:
                            mol.harmonic_infrared_intensities.append(float(templine[-2]))
                            mol.anharmonic_infrared_intensities.append(float(templine[-1]))
                        except:
                            FBIRError = True
            if 'Overtones' in line and not 'anharmonic_overtones_infrared_intensities' in mol.__dict__:
                read_already_OvertonesIR = True
                ireadOvertonesIR += 1
                mol.anharmonic_overtones_infrared_intensities = []
                mol.harmonic_overtones_infrared_intensities = []
                continue
            elif 'Overtones' in line and not read_already_OvertonesIR:
                read_already_OvertonesIR = True
                ireadOvertonesIR += 1
                mol.anharmonic_overtones_infrared_intensities = []
                mol.harmonic_overtones_infrared_intensities = []
                continue 
            if ireadOvertonesIR > -1:
                if len(line.strip()) == 0:
                    ireadOvertonesIR = -1
                    continue 
                ireadOvertonesIR += 1
                if ireadOvertonesIR == 2:
                    if not "I(anharm)" in line:
                        del mol.__dict__['anharmonic_overtones_infrared_intensities']
                        del mol.__dict__['harmonic_overtones_infrared_intensities']
                        ireadOvertonesIR = -1
                    continue 
                if ireadOvertonesIR >= 3:
                    templine = line.strip()
                    # Sometimes anharmonic IR intensities are very large due to bad PES. Gaussian will output ********** in such cases.
                    if "*" in templine:
                        templine = [each for each in templine if each != '*']
                        templine = templine.split()
                        try:
                            mol.harmonic_overtones_infrared_intensities.append(float(templine[-1]))
                            mol.anharmonic_overtones_infrared_intensities.append(np.nan)
                        except:
                            OvertonesIRError = True 
                    else:
                        templine = templine.split()
                        try:
                            mol.harmonic_overtones_infrared_intensities.append(float(templine[-2]))
                            mol.anharmonic_overtones_infrared_intensities.append(float(templine[-1]))
                        except:
                            OvertonesIRError = True
            if 'Combination Bands' in line and not 'anharmonic_combination_bands_infrared_intensities' in mol.__dict__:
                read_already_CombinationBandsIR = True
                ireadCombinationBandsIR += 1
                mol.anharmonic_combination_bands_infrared_intensities = []
                mol.harmonic_combination_bands_infrared_intensities = []
                continue 
            elif 'Combination Bands' in line and not read_already_CombinationBandsIR:
                read_already_CombinationBandsIR = True
                ireadCombinationBandsIR += 1
                mol.anharmonic_combination_bands_infrared_intensities = []
                mol.harmonic_combination_bands_infrared_intensities = []
                continue 
            if ireadCombinationBandsIR > -1:
                if len(line.strip()) == 0:
                    ireadCombinationBandsIR = -1
                    continue 
                ireadCombinationBandsIR += 1
                if ireadCombinationBandsIR == 2:
                    if not "I(anharm)" in line:
                        del mol.__dict__['anharmonic_combination_bands_infrared_intensities']
                        del mol.__dict__['harmonic_combination_bands_infrared_intensities']
                        ireadCombinationBandsIR = -1
                    continue 
                if ireadCombinationBandsIR >= 3:
                    templine = line.strip()
                    # Sometimes anharmonic IR intensities are very large due to bad PES. Gaussian will output ********** in such cases.
                    if "*" in templine:
                        templine = [each for each in templine if each != '*']
                        templine = templine.split()
                        try:
                            mol.harmonic_combination_bands_infrared_intensities.append(float(templine[-1]))
                            mol.anharmonic_combination_bands_infrared_intensities.append(np.nan)
                        except:
                            CombinationBandsIRError = True 
                    else:
                        templine = templine.split()
                        try:
                            mol.harmonic_combination_bands_infrared_intensities.append(float(templine[-2]))
                            mol.anharmonic_combination_bands_infrared_intensities.append(float(templine[-1]))
                        except:
                            CombinationBandsIRError = True        

            #if 'Normal termination of Gaussian' in line:
            #    successful = True

    archiveText = r''
    Nlines = 0
    for line in archive:
        Nlines += 1
        archiveText += line.strip()
        if 'NImag=' in archiveText:
            if line[-len('NImag='):] != 'NImag=':
                break
        if '\\\\@' in archiveText:
            break
    optjob = False
    if archiveText != r'':
        method_names = ['HF', 'MP2', 'MP3', 'MP4D', 'MP4DQ', 'MP4SDQ', 'MP4SDTQ', 'MP5', 'CISD', 'QCISD', 'QCISD(T)', 'CCSD', 'CCSD(T)']
        method_energies = []
        archiveTextSplit = archiveText.split('\\\\') # here it splits with '\\' as a delimiter
        if 'opt'.casefold() in archiveTextSplit[0].casefold():
            optjob = True
        if 'freq'.casefold() in archiveTextSplit[1].casefold() and chkfilename is not None:
            dir_path = os.path.dirname(os.path.realpath(filename))
            if os.path.exists(f'{dir_path}/{chkfilename}'):
                read_Hessian_matrix_from_Gaussian_chkfile(f'{dir_path}/{chkfilename}', mol)
        coords = archiveTextSplit[3]
        coords = coords.split('\\') # here it splits with '\' as a delimiter
        charge, mult = (int(xx) for xx in coords[0].split(','))
        mol.charge = charge
        mol.multiplicity = mult
        coords = coords[1:]
        if mol.atoms == []:
            for coord in coords:
                xx = coord.split(',')
                if len(xx[1:]) > 3: del xx[1]
                mol.atoms.append(data.atom(element_symbol=xx[0], xyz_coordinates=data.array([float(yy) for yy in xx[1:]])))
        else:
            for idx, coord in enumerate(coords):
                xx = coord.split(',')
                mol.atoms[idx].element_symbol = xx[0]
                if len(xx[1:]) > 3: del xx[1]
                mol.atoms[idx].xyz_coordinates = data.array([float(yy) for yy in xx[1:]])
        for xx in archiveTextSplit[4].split('\\'):
            method_flag = False
            for name in method_names:
                if f'{name}=' in xx:
                    yy = xx.split('=')
                    method_energies.append((yy[0], float(yy[-1])))
                    method_flag = True
                    break
            if method_flag: continue
            if 'DipoleDeriv' in xx:
                yy = xx.split('=')[-1].strip()
                # to-do: reshape, make sure the same as in Yifan's implementation before with 'DipoleDeriv ' (note space!)
                mol.dipole_derivatives = data.array([float(zz) for zz in yy.split(',')]) / constants.Debye # do we need to divide by Debye?
                continue
    if len(method_energies) > 1:
        method_children = []
        for method in method_energies:
            method_children.append(data.properties_tree_node(name=method[0]))
            method_children[-1].energy = method[1]
            mol.__dict__[method[0]] = method_children[-1]
        mol.energy_levels = data.properties_tree_node(name='energy_levels',children=method_children)
        mol.energy = method_energies[-1][1]
        # in old implementation it was wrong: mol.HF = energy, now it is mol.HF.energy = energy
    elif len(method_energies) == 1:
        mol.energy = method_energies[0][1]
    
    if nOMOs['all'] > 0:
        mol.n_occ_mos = nOMOs['all']
    if len(mol.excitation_energies) > 0:
        mol.excitation_energies = data.array(mol.excitation_energies)
        mol.electronic_states = [mol.copy(atomic_labels=[], molecular_labels=[]) for ii in range(mol.nstates)]
        for istate, mol_state in enumerate(mol.electronic_states):
            if istate == 0:
                mol_state.energy = mol.energy
                continue
            mol_state.energy = mol.energy + mol.excitation_energies[istate-1]
            if 'Singlet' in es_mults[istate-1]:
                mol_state.multiplicity = 1
            elif 'Doublet' in es_mults[istate-1]:
                mol_state.multiplicity = 2
            elif 'Triplet' in es_mults[istate-1]:
                mol_state.multiplicity = 3
            elif 'Quartet' in es_mults[istate-1]:
                mol_state.multiplicity = 4
            mol_state.mo_contributions = es_contribs[istate-1]
    
    if 'oscillator_strengths' in mol.__dict__:
        if len(mol.oscillator_strengths) > 0:
            mol.oscillator_strengths = data.array(mol.oscillator_strengths)
        
    if 'anharmonic_frequencies' in mol.__dict__.keys():
        # by default, we should get the anharmonic properties if they are available
        order = np.argsort(mol.harmonic_frequencies)
        mol.harmonic_frequencies   = [mol.harmonic_frequencies[ii] for ii in order]
        mol.anharmonic_frequencies = [mol.anharmonic_frequencies[ii] for ii in order]
        mol.frequencies = mol.anharmonic_frequencies
        # for some reason Gaussian prints twice the same analysis of harmonic frequencies, so we need to delete the duplicated information
        freq_len = len(mol.frequencies)
        if 'anharmonic_infrared_intensities' in mol.__dict__.keys():
            mol.anharmonic_infrared_intensities = [mol.anharmonic_infrared_intensities[ii] for ii in order]
            mol.harmonic_infrared_intensities = np.copy(mol.infrared_intensities[:freq_len])
            mol.infrared_intensities = mol.anharmonic_infrared_intensities
        else:
            # If anharmonic IR intensities are not calculated, harmonic IR intensities will be read twice
            if len(mol.frequencies) != len(mol.infrared_intensities):
                mol.infrared_intensities = mol.infrared_intensities[:len(mol.frequencies)]
        #mol.frequencies = mol.frequencies[:freq_len]
        #mol.infrared_intensities = mol.infrared_intensities[:freq_len] # to-do check whether it is correct length now
        mol.force_constants = mol.force_constants[:freq_len]
        mol.reduced_masses = mol.reduced_masses[:freq_len]
        for iatom in range(len(mol.atoms)):
            mol.atoms[iatom].normal_modes = mol.atoms[iatom].normal_modes[:freq_len]
        thermochemistry_properties = ['ZPE','DeltaE2U','DeltaE2H','DeltaE2G','U0','H0','U','H','G','S']
        for each_property in thermochemistry_properties:
            mol.__dict__['harmonic_'+each_property] = mol.__dict__[each_property]
            mol.__dict__[each_property] = mol.__dict__['anharmonic_'+each_property]
    if FBIRError:
        print("Warning! An error occurred when reading anharmonic IR intensities.")
    if OvertonesIRError:
        print("Warning! An error occurred when reading anharmonic overtones IR intensities.")
    if CombinationBandsIRError:
        print("Warning! An error occurred when reading anharmonic combination bands IR intensities.")
    
    if 'force_constants' in mol.__dict__.keys():
        mol.force_constants = data.array(mol.force_constants)
    if 'reduced_masses' in mol.__dict__.keys():
        mol.reduced_masses = data.array(mol.reduced_masses)
    for atom in mol.atoms:
        if 'normal_modes' in atom.__dict__.keys():
            atom.normal_modes = data.array(atom.normal_modes)
    if 'infrared_intensities' in mol.__dict__.keys():
        if np.all(mol.infrared_intensities == 0):
            del mol.__dict__['infrared_intensities']
    
    # delete the last geom if there were fewer SCF energies than geometries
    # (can be due to failed SCF calculations or duplicated printing after successful geom opt)
    if len(scf_energies) < len(xyz_input_orientation) and len(xyz_input_orientation) > 1:
        if np.max(np.abs(data.array(xyz_input_orientation[-1]) - data.array(xyz_input_orientation[-2]))) < 1e-6:
            del xyz_input_orientation[-1]
    if len(scf_energies) < len(xyz_standard_orientation) and len(xyz_standard_orientation) > 1:
        if np.max(np.abs(data.array(xyz_standard_orientation[-1]) - data.array(xyz_standard_orientation[-2]))) < 1e-6:
            del xyz_standard_orientation[-1]
    # if it was opt freq job, then geometries and scf energies are duplicated at the end too
    if len(scf_energies) > 1:
        if len(scf_energies) == len(xyz_input_orientation) and len(scf_energies) == len(xyz_standard_orientation):
            if abs(scf_energies[-2] - scf_energies[-1]) < 1e-6 and np.max(np.abs(data.array(xyz_input_orientation[-1]) - data.array(xyz_input_orientation[-2]))) < 1e-6:
                del scf_energies[-1]
                del xyz_input_orientation[-1]
                del xyz_standard_orientation[-1]
            if len(energy_gradients) > len(scf_energies):
                if np.max(np.abs(data.array(energy_gradients[-1]) - data.array(energy_gradients[-2]))) < 1e-6:
                    del energy_gradients[-1]
        elif len(scf_energies) == len(xyz_input_orientation):
            if abs(scf_energies[-2] - scf_energies[-1]) < 1e-6 and np.max(np.abs(data.array(xyz_input_orientation[-1]) - data.array(xyz_input_orientation[-2]))) < 1e-6:
                del scf_energies[-1]
                del xyz_input_orientation[-1]
            if len(energy_gradients) > len(scf_energies):
                if np.max(np.abs(data.array(energy_gradients[-1]) - data.array(energy_gradients[-2]))) < 1e-6:
                    del energy_gradients[-1]
        elif len(scf_energies) == len(xyz_standard_orientation):
            if abs(scf_energies[-2] - scf_energies[-1]) < 1e-6 and np.max(np.abs(data.array(xyz_standard_orientation[-1]) - data.array(xyz_standard_orientation[-2]))) < 1e-6:
                del scf_energies[-1]
                del xyz_standard_orientation[-1]
            if len(energy_gradients) > len(scf_energies):
                if np.max(np.abs(data.array(energy_gradients[-1]) - data.array(energy_gradients[-2]))) < 1e-6:
                    del energy_gradients[-1]

    # get molecular database for multiple single-point calculations in the output file
    if len(scf_energies) > 1:
        db = data.molecular_database()
        db.molecules = [mol.copy(atomic_labels=[], molecular_labels=[]) for ii in range(len(scf_energies))]
        for iscf in range(len(scf_energies)):
            db.molecules[iscf].energy = scf_energies[iscf]
        if len(scf_energies) == len(xyz_input_orientation):
            for iscf in range(len(scf_energies)):
                db.molecules[iscf].xyz_coordinates = data.array(xyz_input_orientation[iscf])
        # to-do: reorient to input orientation using mol.xyz_coordinates which are already in input orientation for the last geometry
        #elif len(scf_energies) == len(xyz_standard_orientation)
        #    for iscf in range(len(scf_energies)):
        #            db.molecules[iscf].xyz_coordinates = data.array(xyz_standard_orientation[iscf])
        if len(scf_energies) == len(energy_gradients):
            for iscf in range(len(scf_energies)):
                db.molecules[iscf].energy_gradients = data.array(energy_gradients[iscf])
        mol.molecular_database = db
        if optjob:
            moltraj = data.molecular_trajectory()
            for istep, mol_step in enumerate(db):
                moltraj.steps.append(data.molecular_trajectory_step(step=istep, molecule=mol_step))
            mol.optimization_trajectory = moltraj

    # get energy gradients from the last place they are calculated
    if len(energy_gradients) > 0:
        mol.energy_gradients = data.array(energy_gradients[-1])
    
    if 'molecular_database' in mol.__dict__.keys():
        if np.max(np.abs(db[-1].xyz_coordinates - mol.xyz_coordinates)) < 1e-6:
            tmpmol = mol.copy()
            del tmpmol.__dict__['molecular_database']
            mol.molecular_database.molecules[-1] = tmpmol
            if 'optimization_trajectory' in tmpmol.__dict__.keys():
                del tmpmol.__dict__['optimization_trajectory']
                mol.optimization_trajectory.steps[-1].molecule = tmpmol
                
    if molecule is None:
        return mol

if __name__ == '__main__': 
    pass