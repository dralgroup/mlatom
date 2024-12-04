#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gaussian: interface to the Gaussian program                               ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, sys, subprocess
import numpy as np
from mlatom import constants
from mlatom import data
from mlatom import models
from mlatom.decorators import doc_inherit

class gaussian_methods(models.model, metaclass=models.meta_method):
    '''
    Gaussian interface

    Arguments:
        method (str): Method to use
        nthreads (int): equivalent to %proc in Gaussian input file
        save_files_in_current_directory (bool): whether to keep input and output files, default ``'False'``
        working_directory (str): path to the directory where the program output files and other tempory files are saved, default ``'None'``

    .. note::

        The format of method should be the same as that in Gaussian, e.g., ``'B3LYP/6-31G*'``
        
    '''
    
    def __init__(self,method='B3LYP/6-31G*',additional_input='',nthreads=None,save_files_in_current_directory=False, working_directory=None, **kwargs):
        if not "GAUSS_EXEDIR" in os.environ:
            raise ValueError('enviromental variable GAUSS_EXEDIR is not set')
        self.method = method
        self.additional_input = additional_input
        if nthreads is None:
            from multiprocessing import cpu_count
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        if 'writechk' in kwargs:
            self.writechk = kwargs['writechk']
        else:
            self.writechk = False

    @doc_inherit
    def predict(self,molecular_database=None,molecule=None,
                nstates=1, 
                current_state=0, 
                calculate_energy=True,
                calculate_energy_gradients=False,
                calculate_hessian=False,
                calculate_dipole_derivatives=False,
                gaussian_keywords=None,):
        '''
            nstates (int):                 number of electronic structure states (default: 1, ground state)
            current_state (int):           default is the ground state (for nstates=1) or the first excited state (nstates > 1)
            gaussian_keywords (some type): ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)           
        method = self.method
        #self.energy_to_read = energy_to_read
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_energy = calculate_energy
        self.calculate_hessian = calculate_hessian
        self.calculate_dipole_derivatives = calculate_dipole_derivatives
        # self.writechk = False
        if gaussian_keywords == None:
            self.gaussian_keywords = ''
        else:
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
            self.writechk = True
            if 'freq'.casefold() in method.casefold():
                pass 
            else:
                method += ' freq'     
            if calculate_dipole_derivatives:
                method += ' IOp(7/33=1)'

        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdirname:
            if self.save_files_in_current_directory: tmpdirname = '.'
            if self.working_directory is not None:
                tmpdirname = self.working_directory
                if not os.path.exists(tmpdirname):
                    os.makedirs(tmpdirname)
                tmpdirname = os.path.abspath(tmpdirname)
            for imol in range(len(molDB.molecules)):
                imolecule = molDB.molecules[imol]
                # Run Gaussian job
                if self.gaussian_keywords != '':
                    run_gaussian_job(filename='molecule'+str(imol)+'.com',molecule=imolecule,gaussian_keywords=self.gaussian_keywords,nthreads=self.nthreads,method=method,cwd=tmpdirname,writechk=self.writechk,additional_input=self.additional_input)
                else:
                    run_gaussian_job(filename='molecule'+str(imol)+'.com',molecule=imolecule,nthreads=self.nthreads,method=method,cwd=tmpdirname,writechk=self.writechk,additional_input=self.additional_input)

                # Read Gaussian output file
                parse_gaussian_output(filename=os.path.join(tmpdirname,'molecule'+str(imol)+'.log'),molecule=imolecule)
            
def run_gaussian_job(**kwargs):
    if 'filename' in kwargs:
        filename = kwargs['filename']
    if 'molecule' in kwargs: molecule = kwargs['molecule']
    gaussian_keywords = ''
    if 'gaussian_keywords' in kwargs:
        gaussian_keywords = kwargs['gaussian_keywords']
    if 'nthreads' in kwargs:
        nthreads = kwargs['nthreads']
    else:
        nthreads = 1
    memory = ''
    if 'memory' in kwargs:
        memory = f"%mem={kwargs['memory']}\n"
    gaussian_keywords = f'{memory}%nproc={nthreads}\n' + gaussian_keywords 
    
    if 'cwd' in kwargs:
        cwd = kwargs['cwd']
    else:
        cwd='.'
        
    if 'model_predict_kwargs' in kwargs:
        model_predict_kwargs_str = str(kwargs['model_predict_kwargs'])
    else:
        model_predict_kwargs_str = "{}"
    
    model_predict_kwargs_str_file = 'model_predict_kwargs'
    with open(os.path.join(cwd, model_predict_kwargs_str_file), 'w') as f:
        f.write(model_predict_kwargs_str)
    
    if 'external_task' in kwargs:
        pythonbin = sys.executable
        path_to_this_file=os.path.abspath(__file__)
        external_task = kwargs['external_task']
        if 'gaussian_keywords' in kwargs:
            gaussian_keywords += "\nexternal='%s %s'\n" % (pythonbin, path_to_this_file)
        elif external_task.lower() == 'opt':
            gaussian_keywords += "#p opt(nomicro) external='%s %s %s'\n" % (pythonbin, path_to_this_file, model_predict_kwargs_str_file)
        elif external_task.lower() == 'freq':
            gaussian_keywords += "#p freq external='%s %s %s'\n" % (pythonbin, path_to_this_file, model_predict_kwargs_str_file)
        elif external_task.lower() == 'freq(anharmonic)':
            if 'frequency_keywords' in kwargs:
                if len(kwargs['frequency_keywords']) !=0:
                    extra_keywords = ','.join(kwargs['frequency_keywords'])
                    extra_keywords = ','+extra_keywords
                else:
                    extra_keywords = ''
            else:
                extra_keywords = ''
            gaussian_keywords += f"#p freq(anharmonic{extra_keywords}) external='%s %s %s'\n" % (pythonbin, path_to_this_file, model_predict_kwargs_str_file)
        elif external_task.lower() == 'ts':
            gaussian_keywords += "#p opt(ts,calcfc,noeigen,nomicro) external='%s %s %s'\n" % (pythonbin, path_to_this_file, model_predict_kwargs_str_file)
        elif external_task.lower() == 'irc':
            gaussian_keywords += "#p irc(calcfc) external='%s %s %s'\n" % (pythonbin, path_to_this_file, model_predict_kwargs_str_file)
    else:
        if 'gaussian_keywords' in kwargs:
            gaussian_keywords += '\n'
        if 'method' in kwargs:
            if 'freq' in kwargs['method']:
                kwargs['method'] = 'p ' + kwargs['method']
            gaussian_keywords += '# '+kwargs['method']+'\n'

    if 'writechk' in kwargs:
        writechk = kwargs['writechk']
    else:
        writechk = False

    if writechk:
        gaussian_keywords = f'%chk={filename[:-4]}.chk\n'+gaussian_keywords

    if 'additional_input' in kwargs:
        additional_input = kwargs['additional_input']
    else:
        additional_input = ''
    
    write_gaussian_input_file(filename=os.path.join(cwd,filename), molecule=molecule, gaussian_keywords=gaussian_keywords,additional_input=additional_input)
    Gaussianbin, _ = check_gaussian()
    proc = subprocess.Popen([Gaussianbin, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, universal_newlines=True)
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

def write_gaussian_input_file(**kwargs):
    if 'filename' in kwargs: filename = kwargs['filename']
    if 'molecule' in kwargs: molecule = kwargs['molecule']
    if 'gaussian_keywords' in kwargs: gaussian_keywords = kwargs['gaussian_keywords']
    if 'additional_input' in kwargs: 
        additional_input = kwargs['additional_input']
    else:
        additional_input = ''
        
    if 'comment' in molecule.__dict__:
        if molecule.comment != '':
            title = molecule.comment
    elif molecule.id != '':
        title = molecule.id
    else:
        title = 'Gaussian calculations from MLatom interface'
    
    with open(filename, 'w') as f:
        f.writelines(gaussian_keywords)
        f.writelines(f'\n{title}\n')
        f.writelines(f'\n{molecule.charge} {molecule.multiplicity}\n')
        for atom in molecule.atoms:
            f.writelines('%-3s %25.13f %25.13f %25.13f\n' % (atom.element_symbol,
                              atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2]))
        f.writelines('\n') 
        f.writelines(additional_input)
        f.writelines('\n\n')

def gaussian_external(EIn_file, EOu_file, model_predict_kwargs):
    # write new coordinate into 'xyz_temp.dat'
    derivs, molecule = read_gaussian_EIn(EIn_file)
    # calculate energy, gradients, hessian for new coordinates
    # import json
    # with open('model.json', 'r') as fjson:
    #     model_dict = json.load(fjson)
    # if 'method' in model_dict:
    #     kwargs = {}
    #     if 'kwargs' in model_dict:
    #         kwargs = model_dict['kwargs']
    #         del model_dict['kwargs']
    #     model = models.methods(**model_dict, **kwargs)
    model = models.load('model.json')
    calc_hessian = False
    if derivs == 2: calc_hessian = True
    if 'filename' in model_predict_kwargs:
        if model_predict_kwargs['print_properties'] is not None:
            printstrs = model._predict_geomopt(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=calc_hessian, **model_predict_kwargs)
            filename=model_predict_kwargs['filename']
            with open(f'{filename}_tmp_out.out', 'a') as ff:
                ff.writelines(printstrs)
        else:
            model._predict_geomopt(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=calc_hessian, **model_predict_kwargs)
    else:
        model.predict(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=calc_hessian, **model_predict_kwargs)
    if not 'energy' in molecule.__dict__:
        raise ValueError('model did not return any energy')
    write_gaussian_EOu(EOu_file, derivs, molecule)
    
    if os.path.exists('gaussian_freq_mol.json'):
        molecule.dump(filename='gaussian_freq_mol.json', format='json')

def read_gaussian_EIn(EIn_file):
    molecule = data.molecule()
    with open(EIn_file, 'r') as fEIn:
        lines = fEIn.readlines()
        line0 = lines[0].strip().split()
        natoms = int(line0[0]); derivs = int(line0[1])
        molecule.charge = int(line0[2]); molecule.multiplicity = int(line0[3])
        
        for i in range(1, natoms+1):
            xx = lines[i].strip().split()
            atom = data.atom(atomic_number=int(xx[0]),
                             xyz_coordinates=np.array(xx[1:-1]).astype('float')*constants.Bohr2Angstrom)
            molecule.atoms.append(atom)
    
    return derivs, molecule

def write_gaussian_EOu(EOu_file, derivs, molecule):
    import fortranformat
    with open(EOu_file, 'w') as fEOu:
        # energy, dipole-moment (xyz)   E, Dip(I), I=1,3
        if 'dipole_moment' in molecule.__dict__.keys():
            dp = molecule.dipole_moment
        else:
            dp = [0.0,0.0,0.0]
        writer = fortranformat.FortranRecordWriter('(4D20.12)')
        output = writer.write([molecule.energy, dp[0], dp[1], dp[2]])
        fEOu.write(output)
        fEOu.write('\n')
        
        writer = fortranformat.FortranRecordWriter('(3D20.12)')
        # gradient on atom (xyz)        FX(J,I), J=1,3; I=1,NAtoms
        output = writer.write(molecule.get_energy_gradients().flatten()*constants.Bohr2Angstrom)
        fEOu.write(output)
        fEOu.write('\n')
        if derivs == 2:
            natoms = len(molecule.atoms)
            # polarizability                Polar(I), I=1,6
            polor = np.zeros(6)
            output = writer.write(polor)
            fEOu.write(output)
            fEOu.write('\n')
            # dipole derivatives            DDip(I), I=1,9*NAtoms
            ddip = np.zeros(9*natoms)
            if 'dipole_derivatives' in molecule.__dict__.keys():
                ddip = molecule.dipole_derivatives * constants.Bohr2Angstrom * constants.Debye2au 
            output = writer.write(ddip)
            fEOu.write(output)
            fEOu.write('\n')
            # force constants               FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2
            output = writer.write(molecule.hessian[np.tril_indices(natoms*3)]*constants.Bohr2Angstrom**2)
            fEOu.write(output)

def read_Hessian_matrix_from_Gaussian_chkfile(filename,molecule):
    natoms = len(molecule.atoms)
    hessian = np.zeros((3*natoms,3*natoms))
    cmdargs = ['chkchk','-p',filename]
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
            if 'Forces (Hartrees/Bohr)' in line:
                forces_iatom = -3
                continue
            if forces_iatom > -4:
                forces_iatom += 1
                if forces_iatom < 0: continue
                if '------------------' in line:
                    forces_iatom = -4
                    continue
                if len(mol.atoms) < forces_iatom+1:
                    mol.atoms.append(data.atom(atomic_number=int(line.split()[1])))
                mol.atoms[forces_iatom].energy_gradients = -data.array([float(each) for each in line.split()[2:]]) / constants.Bohr2Angstrom
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
    if archiveText != r'':
        method_names = ['HF', 'MP2', 'MP3', 'MP4D', 'MP4DQ', 'MP4SDQ', 'MP4SDTQ', 'MP5', 'CISD', 'QCISD', 'QCISD(T)', 'CCSD', 'CCSD(T)']
        method_energies = []
        archiveTextSplit = archiveText.split('\\\\') # here it splits with '\\' as a delimiter
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
        
    # to-do: check whether gradients for electronic states are read correctly
    # to-do: check whether the infrared_intensities have the correct length after anharmonic frequencies calculation
    
    if molecule is None:
        return mol

if __name__ == '__main__': 
    model_predict_kwargs_str_file, _, EIn_file, EOu_file, _, _, _ = sys.argv[1:]
    with open(model_predict_kwargs_str_file) as f:
        model_predict_kwargs =  eval(f.read())
    gaussian_external(EIn_file, EOu_file, model_predict_kwargs)