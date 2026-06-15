#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gaussian_interface: interface to the Gaussian program                     ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, re, sys, subprocess
import numpy as np

from .. import constants, data
from ..utils import wait_for_file_stable
from ..model_cls import method_model
from ..decorators import doc_inherit


def _has_density_keyword(text):
    return bool(text) and re.search(r'(^|[\s,(])density\s*=', text, flags=re.IGNORECASE) is not None


def _has_td_keyword(text):
    return bool(text) and re.search(r'(^|[\s,(])td(\(|$|[\s,])', text, flags=re.IGNORECASE) is not None


def _normalize_td_method(method, gaussian_keywords='', nstates=1, current_state=0):
    if method is None:
        return method, gaussian_keywords

    stripped_method = method.strip()
    if re.match(r'td-', stripped_method, flags=re.IGNORECASE) is None:
        return method, gaussian_keywords

    try:
        nstates = int(nstates)
    except (TypeError, ValueError):
        return method, gaussian_keywords
    if nstates <= 1:
        return method, gaussian_keywords

    try:
        current_state = int(current_state)
    except (TypeError, ValueError):
        current_state = 0

    normalized_method = re.sub(r'^td-', '', stripped_method, flags=re.IGNORECASE)
    if not _has_td_keyword(normalized_method) and not _has_td_keyword(gaussian_keywords):
        root = 1 if current_state == 0 else current_state
        normalized_method += f' TD(NStates={nstates-1},Root={root})'

    if not _has_density_keyword(normalized_method) and not _has_density_keyword(gaussian_keywords):
        gaussian_keywords = f'{gaussian_keywords}\ndensity=current'.strip()

    return normalized_method, gaussian_keywords


def _rotate_dipole_to_input_orientation(dipole_moment, standard_orientation, input_orientation):
    if dipole_moment is None:
        return dipole_moment
    if standard_orientation is None or input_orientation is None:
        return dipole_moment

    standard_orientation = np.asarray(standard_orientation, dtype=float)
    input_orientation = np.asarray(input_orientation, dtype=float)
    if standard_orientation.shape != input_orientation.shape:
        return dipole_moment
    if standard_orientation.ndim != 2 or standard_orientation.shape[1] != 3:
        return dipole_moment

    from .. import xyz as xyz_utils

    centered_standard = standard_orientation - xyz_utils.get_center_of_mass(standard_orientation)
    centered_input = input_orientation - xyz_utils.get_center_of_mass(input_orientation)
    if np.allclose(centered_standard, 0.0) or np.allclose(centered_input, 0.0):
        return dipole_moment

    rotation_matrix = xyz_utils.rotation_matrix(centered_standard, centered_input)
    rotated_components = np.asarray(dipole_moment[:3], dtype=float).dot(rotation_matrix)

    if len(dipole_moment) > 3:
        return data.array([rotated_components[0], rotated_components[1], rotated_components[2], dipole_moment[3]])
    return data.array(rotated_components)

def _rotate_nm_to_input_orientation(normal_modes, standard_orientation, input_orientation):
    if normal_modes is None:
        return normal_modes
    if standard_orientation is None or input_orientation is None:
        return normal_modes
    standard_orientation = np.asarray(standard_orientation, dtype=float)
    input_orientation = np.asarray(input_orientation, dtype=float)
    if standard_orientation.shape != input_orientation.shape:
        return normal_modes
    if standard_orientation.ndim != 2 or standard_orientation.shape[1] != 3:
        return normal_modes
    
    from .. import xyz as xyz_utils
    centered_standard = standard_orientation - xyz_utils.get_center_of_mass(standard_orientation)
    centered_input = input_orientation - xyz_utils.get_center_of_mass(input_orientation)
    if np.allclose(centered_standard, 0.0) or np.allclose(centered_input, 0.0):
        return normal_modes
    
    rotation_matrix = xyz_utils.rotation_matrix(centered_standard, centered_input)
    
    nm = np.asarray(normal_modes, dtype=float)
    return np.matmul(nm, rotation_matrix)

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
        method, gaussian_keywords = _normalize_td_method(method,
                                                         gaussian_keywords,
                                                         nstates=nstates,
                                                         current_state=current_state)
        self.gaussian_keywords = gaussian_keywords
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
                stable = wait_for_file_stable(os.path.join(tmpdirname,f'{filename_wo_extension}.log'))
                if not stable:
                    print(f'! Warning, the file {filename_wo_extension}.log is still changing its size, its parsing might be incomplete')
                parse_gaussian_output(filename=os.path.join(tmpdirname,f'{filename_wo_extension}.log'),molecule=imolecule)
def run_gaussian_job(filename=None,
                     molecule=None, reactants=None, products=None,
                     model=None,
                     gaussian_keywords='',
                     nthreads=None, memory=None,
                     working_directory='.',
                     # The block below is for using Gaussian for SP calculations
                     method=None,
                     chkfilename=None,
                     # The block below is for using Gaussian as engine for jobs with external potential
                     external_task=None,
                     model_predict_kwargs=None,
                     extra_keywords='', # string with extra keywords
                     additional_input='',
                     opt_keywords=None,
                     freq_keywords=[], # list with additional arguments such as ['NoRaman',] etc.
                     irc_keywords=['CalcFC']
                     ):
    model_is_gaussian = isinstance(model, gaussian_methods)
    if method is None and model_is_gaussian:
        method = model.method

    if model_is_gaussian:
        if gaussian_keywords in [None, ''] and model.gaussian_keywords is not None:
            gaussian_keywords = model.gaussian_keywords
        if additional_input == '' and model.additional_input is not None:
            additional_input = model.additional_input
        if chkfilename is None:
            if model.chkfilename is True and filename is not None:
                chkfilename = os.path.splitext(os.path.basename(filename))[0] + '.chk'
            elif isinstance(model.chkfilename, str):
                chkfilename = model.chkfilename
        if memory is None:
            memory = model.memory
        if nthreads is None and model.nthreads not in [None, 0]:
            nthreads = model.nthreads
        td_kwargs = {} if model_predict_kwargs is None else model_predict_kwargs
        method, gaussian_keywords = _normalize_td_method(method,
                                                         gaussian_keywords,
                                                         nstates=td_kwargs.get('nstates', 1),
                                                         current_state=td_kwargs.get('current_state', 0))
    if nthreads is None:
        if external_task is None:
            from multiprocessing import cpu_count
            nthreads = cpu_count()
        elif external_task is not None and not model_is_gaussian:
            nthreads = 1
        elif external_task is not None and model_is_gaussian:
            from multiprocessing import cpu_count
            nthreads = cpu_count()

    task = external_task
    external_command = None
    wrapper_task = external_task is not None and model_predict_kwargs is not None and not model_is_gaussian
    if wrapper_task:
        pythonbin = sys.executable
        path_to_this_file=os.path.abspath(__file__)
        path_to_gaussian_external = os.path.join(os.path.dirname(path_to_this_file), 'gaussian_external.py')
        
        model_predict_kwargs_str = str(model_predict_kwargs)
        model_predict_kwargs_str_file = 'model_predict_kwargs'
        with open(os.path.join(working_directory, model_predict_kwargs_str_file), 'w') as f:
            f.write(model_predict_kwargs_str)
        external_command = f"external='{pythonbin} {path_to_gaussian_external} {model_predict_kwargs_str_file}'"
    keyword_parts = []
    if gaussian_keywords:
        keyword_parts.append(gaussian_keywords.strip())
    if method:
        keyword_parts.append(method.strip())
    if task is not None:
        task_lower = task.casefold()
        opt_keywords = [] if opt_keywords is None else list(opt_keywords)
        freq_keywords = [] if freq_keywords is None else list(freq_keywords)
        irc_keywords = ['CalcFC'] if irc_keywords is None else list(irc_keywords)
        if task_lower == 'opt':
            if wrapper_task and 'nomicro' not in [kwd.casefold() for kwd in opt_keywords]:
                opt_keywords.append('nomicro')
            if len(opt_keywords) > 0:
                keyword_parts.append(f"opt({','.join(opt_keywords)})")
            else:
                keyword_parts.append('opt')
        elif 'freq' in task_lower:
            if len(freq_keywords) != 0:
                freq_str = ','.join(freq_keywords)
                if task_lower == 'freq(anharmonic)':
                    keyword_parts.append(f'freq(anharmonic,{freq_str})')
                elif task_lower == 'freq':
                    keyword_parts.append(f'freq({freq_str})')
                else:
                    keyword_parts.append(task)
            else:
                if task_lower == 'freq(anharmonic)':
                    keyword_parts.append('Freq(anharmonic)')
                elif task_lower == 'freq':
                    keyword_parts.append('Freq')
                else:
                    keyword_parts.append(task)
        elif task_lower in ['ts', 'qst2', 'qst3']:
            if len(opt_keywords) == 0:
                if task_lower == 'ts':
                    opt_keywords = ['CalcFC', 'noeigen']
                else:
                    opt_keywords = ['CalcFC']
            if wrapper_task and 'nomicro' not in [kwd.casefold() for kwd in opt_keywords]:
                opt_keywords.append('nomicro')
            if task_lower not in [kwd.casefold() for kwd in opt_keywords]:
                opt_keywords = [task.upper()] + opt_keywords
            keyword_parts.append(f"Opt({','.join(opt_keywords)})")
        elif task_lower == 'irc':
            if len(irc_keywords) > 0:
                keyword_parts.append(f"irc({','.join(irc_keywords)})")
            else:
                keyword_parts.append('irc')
        else:
            keyword_parts.append(task)
    if extra_keywords:
        keyword_parts.append(extra_keywords)
    gaussian_keywords = ' '.join(part for part in keyword_parts if part)
    if external_command:
        gaussian_keywords = f'{gaussian_keywords}\n{external_command}' if gaussian_keywords else external_command
     
    if not gaussian_keywords.strip()[:1] == '#':
        if task is not None or 'freq' in gaussian_keywords.casefold():
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

    if not model_is_gaussian or filename is None:
        return

    output_path = os.path.join(working_directory, os.path.splitext(filename)[0] + '.log')
    if not os.path.exists(output_path):
        output_path = os.path.join(working_directory, os.path.splitext(filename)[0] + '.out')
    if not os.path.exists(output_path):
        return

    task_lower = task.casefold() if isinstance(task, str) else ''
    if task_lower in ['opt', 'ts', 'qst2', 'qst3']:
        parsed_molecule = parse_gaussian_output(filename=output_path)
        optimization_trajectory = parsed_molecule.optimization_trajectory if 'optimization_trajectory' in parsed_molecule.__dict__ else data.molecular_trajectory()
        if len(optimization_trajectory.steps) == 0:
            optimization_trajectory.steps.append(data.molecular_trajectory_step(step=0, molecule=parsed_molecule))
        if model_predict_kwargs is not None and model_predict_kwargs.get('filename') is not None:
            trajectory_filename = os.path.join(working_directory, model_predict_kwargs['filename'])
            trajectory_format = model_predict_kwargs.get('format', 'json')
            optimization_trajectory.dump(filename=trajectory_filename, format=trajectory_format)
            if model_predict_kwargs.get('dump_trajectory_interval') is not None:
                molecular_database = data.molecular_database()
                molecular_database.molecules = [step.molecule for step in optimization_trajectory.steps]
                xyzfilename = os.path.splitext(trajectory_filename)[0] + '.xyz'
                molecular_database.write_file_with_xyz_coordinates(xyzfilename)
    elif 'freq' in task_lower and molecule is not None:
        parse_gaussian_output(filename=output_path, molecule=molecule)
        freq_mol_filename = os.path.join(working_directory, 'gaussian_freq_mol.json')
        if os.path.exists(freq_mol_filename):
            molecule.dump(filename=freq_mol_filename, format='json')
    
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

def parse_chunk(mol, chunk):
    forces_iatom = -4
    i_read_hessian = -1
    mulliken_charge_iatom = -3
    mulliken_charge_spin_dens_iatom = -3
    read_dipole_moment = False
    readNMs = False
    ireadS = -1
    es_mults = []
    es_contribs = []
    readES = False
    current_state = None
    current_state_energy = None
    i_read_input_orientation = -1
    i_read_standard_orientation = -1
    xyz_input_orientation = []
    xyz_standard_orientation = []
    energy_gradients = []
    recovered_energy = None
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
    input_orientation_read = False
    input_orientation = None
    standard_orientation = None
    
    # Remove duplicate normal modes 
    for atom in mol.atoms:
        if 'normal_modes' in atom.__dict__:
            del atom.__dict__['normal_modes']

    for line in chunk:
        if 'Input orientation:' in line:
            i_read_input_orientation = 0
            xyz_input_orientation.append([])
            continue
        if i_read_input_orientation > -1:
            i_read_input_orientation += 1
            if i_read_input_orientation >= 5:
                if '-------------------------' in line:
                    i_read_input_orientation = -1
                    input_orientation_read = True
                    if len(xyz_input_orientation) > 0 and len(xyz_input_orientation[-1]) > 0:
                        input_orientation = np.array(xyz_input_orientation[-1], dtype=float)
                    continue
                xyz_coords = [float(xx) for xx in line.split()[-3:]]
                xyz_input_orientation[-1].append(xyz_coords)
                if len(mol.atoms) < i_read_input_orientation-5 + 1:
                    mol.atoms.append(data.atom(atomic_number=int(line.split()[1]), xyz_coordinates=xyz_coords))
                else:
                    mol.atoms[i_read_input_orientation-5].xyz_coordinates = xyz_coords
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
                    if len(xyz_standard_orientation) > 0 and len(xyz_standard_orientation[-1]) > 0:
                        standard_orientation = np.array(xyz_standard_orientation[-1], dtype=float)
                    continue
                xyz_coords = [float(xx) for xx in line.split()[-3:]]
                xyz_standard_orientation[-1].append(xyz_coords)
                if not input_orientation_read:
                    if len(mol.atoms) < i_read_standard_orientation + 1:
                        mol.atoms.append(data.atom(atomic_number=int(line.split()[1]), xyz_coordinates=xyz_coords))
                    else:
                        mol.atoms[i_read_standard_orientation-5].xyz_coordinates = xyz_coords
                continue
        if 'SCF Done:' in line:
            mol.scf_energy = float(line.split()[4])
        if 'Recovered energy= ' in line: # get the energy under recovered energy
            recovered_energy = float(line.split()[2])
        if 'Forces (Hartrees/Bohr)' in line:
            forces_iatom = -3
            continue
        if forces_iatom > -4:
            forces_iatom += 1
            if forces_iatom < 0: continue
            if '------------------' in line:
                forces_iatom = -4
                mol.energy_gradients = data.array(energy_gradients)
                continue
            energy_gradients.append(-data.array([float(each) for each in line.split()[2:]]) / constants.Bohr2Angstrom)
            continue
        if 'The second derivative matrix:' in line:
            i_read_hessian = 0
            natoms = len(mol.atoms)
            mol.hessian = np.zeros((3*natoms,3*natoms))
            continue
        if i_read_hessian > -1:
            i_read_hessian += 1
            if i_read_hessian == 1:
                if not line.split()[0][0] in 'XYZ':
                    i_read_hessian = -1
                    del mol.__dict__['hessian']
                    continue
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
        if 'Mulliken charges and spin densities:' in line:
            mulliken_charge_spin_dens_iatom = -2
            continue
        if mulliken_charge_spin_dens_iatom > -3:
            if len(line.split()) > 4:
                mulliken_charge_spin_dens_iatom = -3
                continue
            mulliken_charge_spin_dens_iatom += 1
            if mulliken_charge_spin_dens_iatom < 0:
                continue
            if len(mol.atoms) < mulliken_charge_spin_dens_iatom+1:
                mol.atoms.append(data.atom(element_symbol=line.split()[1]))
            mol.atoms[mulliken_charge_spin_dens_iatom].mulliken_charge = float(line.split()[-2])
            mol.atoms[mulliken_charge_spin_dens_iatom].spin_density = float(line.split()[-1])
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
            elif 'This state for optimization and/or second-order correction.' in line:
                current_state = len(mol.excitation_energies)
            elif 'Total Energy,' in line:
                current_state_energy = float(line.split()[-1])
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
        # IRC #
        if 'NET REACTION COORDINATE UP TO THIS POINT ' in line:
            mol.reaction_coordinates = float(line.split()[-1])
        if 'Point Number:' and 'Path Number:' in line:
            mol.n_point = int(line.split()[2])
            mol.n_path = int(line.split()[-1])

    if nOMOs['all'] > 0:
        mol.n_occ_mos = nOMOs['all']
    if 'dipole_moment' in mol.__dict__.keys():
        mol.dipole_moment = _rotate_dipole_to_input_orientation(mol.dipole_moment,
                                                                standard_orientation,
                                                                input_orientation)
    if len(mol.excitation_energies) > 0:
        mol.excitation_energies = data.array(mol.excitation_energies)
        mol.electronic_states = [mol.copy(atomic_labels=[], molecular_labels=[]) for ii in range(mol.nstates)]
        for istate, mol_state in enumerate(mol.electronic_states):
            # for TD-DFT, others might be different
            if istate == 0:
                mol_state.energy = mol.scf_energy
                continue
            mol_state.energy = mol.scf_energy + mol.excitation_energies[istate-1]
            if 'Singlet' in es_mults[istate-1]:
                mol_state.multiplicity = 1
            elif 'Doublet' in es_mults[istate-1]:
                mol_state.multiplicity = 2
            elif 'Triplet' in es_mults[istate-1]:
                mol_state.multiplicity = 3
            elif 'Quartet' in es_mults[istate-1]:
                mol_state.multiplicity = 4
            mol_state.mo_contributions = es_contribs[istate-1]
        if current_state is not None:
            mol.current_state = current_state
            if current_state_energy is not None:
                mol.energy = current_state_energy
                # sanity check
                if abs(current_state_energy - mol.electronic_states[current_state].energy) > 1e-5:
                    print(f''' * Warning* Parsed current state energy is different from the calculated one,
 please check the outputs carefully, MLatom might not support parsing of this kind of output.
 current_state_energy = {current_state_energy}
 calculatd state energy = {mol.electronic_states[current_state].energy}
''')
            else:
                mol.energy = mol.electronic_states[current_state].energy
            if 'energy_gradients' in mol.atoms[0].__dict__:
                mol.electronic_states[current_state].energy_gradients = mol.energy_gradients
    
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
    # IMPORTANT: the normal modes read from Gaussian output are in the standard orientation, we need to rotate them back to the input orientation
    if 'normal_modes' in mol.atoms[0].__dict__.keys():
        nm = mol.get_xyz_vectorial_properties('normal_modes')
        nm = _rotate_nm_to_input_orientation(nm, standard_orientation, input_orientation)
        for iatom in range(len(mol.atoms)):
            mol.atoms[iatom].normal_modes = nm[iatom]
    if 'infrared_intensities' in mol.__dict__.keys():
        if np.all(mol.infrared_intensities == 0):
            del mol.__dict__['infrared_intensities']
            
    if not 'energy' in mol.__dict__.keys():
        if 'scf_energy' in mol.__dict__.keys():
            mol.energy = mol.scf_energy
        elif recovered_energy is not None:
            mol.energy = recovered_energy

def parse_archive(mol, archive, chkfile_path=None):
    archiveText = r''
    for line in archive:
        archiveText += line.strip()
        if 'NImag=' in archiveText:
            if line[-len('NImag='):] != 'NImag=':
                break
    method_energies = []
    if archiveText != r'':
        method_names = ['HF', 'MP2', 'MP3', 'MP4D', 'MP4DQ', 'MP4SDQ', 'MP4SDTQ', 'MP5', 'CISD', 'QCISD', 'QCISD(T)', 'CCSD', 'CCSD(T)']
        archiveTextSplit = archiveText.split('\\\\') # here it splits with '\\' as a delimiter
        if 'opt'.casefold() in archiveTextSplit[0].casefold():
            mol.job_type = 'opt'
        if 'freq'.casefold() in archiveTextSplit[1].casefold():
            mol.job_type = 'freq'
            if chkfile_path is not None:
                read_Hessian_matrix_from_Gaussian_chkfile(chkfile_path, mol)
        coords = archiveTextSplit[3]
        coords = coords.split('\\') # here it splits with '\' as a delimiter
        charge, mult = (int(xx) for xx in coords[0].split(','))
        mol.charge = charge
        mol.multiplicity = mult
        coords = coords[1:]
        if len(mol.atoms) == 0:
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
            
    if len(mol.electronic_states) == 0:
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

def parse_gaussian_output(filename=None, molecule=None):
    """
    Parse Gaussian output file and return a molecule obj
    :param filename: Name of Gaussian output file (e.g. benzene.log or benzene.out)
    :return: molecule if molecule is not provided
    """
    
    if molecule is not None:
        mol = molecule
    else:
        mol = data.molecule()
    
    lines = []
    with open(filename, 'r') as ff:
        for line in ff:
            lines.append(line.rstrip())
           
    chkfilename = None
    jobs = []
    i_archive = [] # list [index_start, index_end]
    i_input_orient = [] # start indices
    i_stand_orient = [] # start indices
    i_normal_termination = [] # indices
    error_message = None
    
    i_optjob = []
    i_ircjob = []
    readInp = False
    readArchive = False
    
    for iline, line in enumerate(lines):
        if ' #' in line[:2] and not readArchive:
            readInp = True
        if readInp:
            if 'opt' in line.casefold():  
                i_optjob.append(iline)
                readInp = False
            if 'irc' in line.casefold():
                i_ircjob.append(iline)
                readInp = False
            if '--------' in line:
                readInp = False
            if len(line) == 0:
                readInp = False
        if '%chk=' in line:
            chkfilename = line.split('=')[-1].strip()
            if not '.chk' in chkfilename:
                chkfilename += '.chk'
            continue
        if ' 1\\1\\' in line or ' 1|1|' in line:
            i_archive.append([iline, None])
            readArchive = True
            continue
        if readArchive:
            if '\\\\@' in line or '||@' in line or len(line.strip()) == 0:
                readArchive = False
                i_archive[-1][1] = iline
            continue
        if 'Input orientation:' in line:
            i_input_orient.append(iline)
            continue
        if 'Standard orientation:' in line:
            i_stand_orient.append(iline)
            continue
        if 'Normal termination of Gaussian' in line:
            i_normal_termination.append(iline)
            continue
        if 'Error termination via ' in line:
            error_message = '\n'.join(lines[iline-5:iline+1])
            continue
    class job():
        def __init__(self, start=None, end=None):
            self.start = start
            self.end = end
            self.molecules = []
            self.job_type = None
    
    # make list of line indices corresponding to separate jobs in the Gaussian output
    if len(i_normal_termination) == 0:
        n_jobs = 1
        jobs= [job(start=0, end=len(lines)-1)]
    else:
        n_jobs = len(i_normal_termination)
        jobs= [job(start=0, end=i_normal_termination[0])]
        for ii in range(1,len(i_normal_termination)):
            jobs.append(job(start=i_normal_termination[ii-1]+1, end=i_normal_termination[ii]))
        if i_normal_termination[-1] + 10 < len(lines):
            n_jobs += 1
            jobs.append(job(start=i_normal_termination[-1]+1, end=len(lines)-1))
    
    for i_job in range(n_jobs):
        for i_opt in i_optjob:
            if i_opt > jobs[i_job].start and i_opt < jobs[i_job].end:
                jobs[i_job].job_type = 'opt'
        for i_irc in i_ircjob:
            if i_irc > jobs[i_job].start and i_irc < jobs[i_job].end:
                jobs[i_job].job_type = 'irc'
    
    # get indices with chunks based on the 'Input orientation:' or 'Standard orientation:'
    if len(i_input_orient) >= len(i_stand_orient):
        ichunks = i_input_orient
    else:
        ichunks = i_stand_orient
    
    for ii in range(len(ichunks)):
        mol_temp = mol.copy()
        istart = ichunks[ii]
        if ii == len(ichunks)-1:
            iend = len(lines)
        else:
            iend = ichunks[ii+1]
        chunk = lines[ichunks[ii]:iend]
        parse_chunk(mol_temp, chunk)
        for i_job in range(n_jobs):
            if istart > jobs[i_job].start and istart < jobs[i_job].end:
                jobs[i_job].molecules.append(mol_temp)
                break

    for i_job, idx_tuple in enumerate(i_archive):
        istart, iend = idx_tuple
        chunk = lines[istart:iend+1]
        if chkfilename is not None:
            dir_path = os.path.dirname(os.path.realpath(filename))
            chkfile_path = f'{dir_path}/{chkfilename}'
            if not os.path.exists(chkfile_path):
                chkfile_path = None
        else:
            chkfile_path = None
        mol_temp = jobs[i_job].molecules[-1]
        parse_archive(mol_temp, chunk, chkfile_path)
        if 'job_type' in mol_temp.__dict__.keys():
            jobs[i_job].job_type = mol_temp.job_type
    
    for job in jobs:
        for moltmp in job.molecules:
            moltmp.job_type = job.job_type
    
    if len(i_archive) > 0:
        mol = jobs[len(i_archive)-1].molecules[-1].copy()
    elif len(jobs[-1].molecules) > 0:
        mol = jobs[-1].molecules[-1].copy()

    if n_jobs > 1 or len(jobs[0].molecules) > 1:
        mol.molecular_database = data.molecular_database()
        for job in jobs:
            for tmpmol in job.molecules:
                mol.molecular_database += tmpmol
    for job in jobs:
        if job.job_type == 'opt':
            mol.optimization_trajectory = data.molecular_trajectory()
            for istep, mol_step in enumerate(job.molecules):
                mol.optimization_trajectory.steps.append(data.molecular_trajectory_step(step=istep, molecule=mol_step))
            break

    if error_message is not None:
        mol.error_message = error_message
    
    if molecule is None:
        return mol
    else:
        molecule.update_from(mol)

if __name__ == '__main__': 
    pass