#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gaussian: interface to the Gaussian program                               ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, sys, subprocess, math
import numpy as np
from mlatom import constants
from mlatom import data
from mlatom import models
from mlatom import stopper
from mlatom import simulations
from mlatom.decorators import doc_inherit


class gaussian_methods(models.model):
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
        self.find_energy_to_read_in_Gaussian()
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
                calculate_energy=True,
                calculate_energy_gradients=False,
                calculate_hessian=False,
                gaussian_keywords=None,):
        '''
            gaussian_keywords (some type): ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)           
        method = self.method
        #self.energy_to_read = energy_to_read
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_energy = calculate_energy
        self.calculate_hessian = calculate_hessian
        # self.writechk = False
        if gaussian_keywords == None:
            self.gaussian_keywords = ''
        else:
            self.gaussian_keywords = gaussian_keywords
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
                self.parse_gaussian_output(os.path.join(tmpdirname,'molecule'+str(imol)+'.log'),imolecule)
            

    def parse_gaussian_output(self, filename, molecule):
        assistant = Gaussian_output_reading_assistant(self.calculate_energy,self.calculate_energy_gradients,self.calculate_hessian,self.energy_to_read)
        assistant.read_output(filename,molecule)

    def find_energy_to_read_in_Gaussian(self):
        HF = data.properties_tree_node(name='HF')
        MP2 = data.properties_tree_node(name='MP2')
        MP3 = data.properties_tree_node(name='MP3')
        MP4D = data.properties_tree_node(name='MP4D')
        MP4DQ = data.properties_tree_node(name='MP4DQ')
        MP4SDQ = data.properties_tree_node(name='MP4SDQ')
        MP4SDTQ = data.properties_tree_node(name='MP4SDTQ')
        MP5 = data.properties_tree_node(name='MP5')
        CISD = data.properties_tree_node(name='CISD')
        QCISD = data.properties_tree_node(name='QCISD')
        QCISD_T = data.properties_tree_node(name='QCISD(T)')
        CCSD = data.properties_tree_node(name='CCSD')
        CCSD_T = data.properties_tree_node(name='CCSD(T)')

        children_properties = []
        method = self.method 
        parse_method = method.split()
        for each in method.split():
            if '/' in each:
                parse_method = each.split('/')
        functional = parse_method[0]
        if 'HF'.casefold() in functional.casefold():
            children_properties = [HF]
        elif 'B3LYP'.casefold() in functional.casefold():
            children_properties = [HF]
        # MP2
        elif 'MP2'.casefold() in functional.casefold():
            children_properties = [HF,MP2]
        # MP3
        elif 'MP3'.casefold() in functional.casefold():
            children_properties = [HF,MP2,MP3]
        # MP4(SDQ) & MP4(SDTQ) (by default latter)
        elif 'MP4'.casefold() in functional.casefold():
            if 'SDQ'.casefold() in functional.casefold(): # MP4(SDQ)
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ]
            elif 'SDTQ'.casefold() in functional.casefold(): # MP4(SDTQ)
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,MP4SDTQ]
            else: # MP4
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,MP4SDTQ]
        # CISD
        elif 'CISD'.casefold() in functional.casefold():
            children_properties = [HF,MP2,MP3,CISD]
        # QCISD & QCISD(T)
        elif 'QCISD'.casefold() in functional.casefold():
            if 'QCISD(T'.casefold() in functional.casefold(): # QCISD(T)
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,QCISD,QCISD_T]
            else: # QCISD
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,QCISD]
        # CCSD & CCSD(T)
        elif 'CCSD'.casefold() in functional.casefold():
            if 'CCSD(T'.casefold() in functional.casefold(): # CCSD(T)
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,CCSD,CCSD_T]
            else: # CCSD
                children_properties = [HF,MP2,MP3,MP4D,MP4DQ,MP4SDQ,CCSD]
        else:
            children_properties = [HF]


        self.energy_to_read = data.properties_tree_node(name='energy',children=children_properties)

            
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
                # Rescale the dipole derivatives by a factor of 0.1 so that anharmonic 
                # infrared intensities can be printed properly when using AIQM1
                # Later the intensities will be rescaled by a factor of 100
                ddip = molecule.dipole_derivatives * constants.Bohr2Angstrom * constants.Debye2au / 10.0
            output = writer.write(ddip)
            fEOu.write(output)
            fEOu.write('\n')
            # force constants               FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2
            output = writer.write(molecule.hessian[np.tril_indices(natoms*3)]*constants.Bohr2Angstrom**2)
            fEOu.write(output)
            
def read_freq_thermochemistry_from_Gaussian_output(outputfile, molecule):
    successful = False
    if os.path.exists('gaussian_freq_mol.json'):
        molecule.load(filename='gaussian_freq_mol.json', format='json')
    lines = open(outputfile, 'r').readlines()
    natoms = len(molecule.atoms)
    molecule.frequencies = []
    molecule.force_constants = []
    molecule.reduced_masses = []
    molecule.infrared_intensities = []
    for atom in molecule.atoms:
        atom.normal_modes = []
    
    # Ugly fix: read from last job step
    termination_list = [0]
    for iline in range(len(lines)):
        if 'termination' in lines[iline]:
            termination_list.append(iline+1)
    istart = termination_list[-2]

    for iline in range(istart,len(lines)):
        if 'Frequencies --' in lines[iline]: molecule.frequencies += [float(xx) for xx in lines[iline].split()[2:]]
        if 'Red. masses --' in lines[iline]: molecule.reduced_masses += [float(xx) for xx in lines[iline].split()[3:]]
        if 'Frc consts  --' in lines[iline]: molecule.force_constants += [float(xx) for xx in lines[iline].split()[3:]]
        if 'IR Inten    --' in lines[iline]: 
            if 'dipole_derivatives' in molecule.__dict__.keys(): molecule.infrared_intensities += [float(xx)*100 for xx in lines[iline].split()[3:]]
        if 'Atom  AN      X      Y      Z' in lines[iline]:
            for iatom in range(natoms):
                xyzs = np.array([float(xx) for xx in lines[iline+iatom+1].split()[2:]])
                nxyzs = len(xyzs) / 3
                for xyz in np.array_split(xyzs, nxyzs):
                    molecule.atoms[iatom].normal_modes.append(list(xyz))
        if 'Zero-point correction=' in lines[iline]: molecule.ZPE = float(lines[iline].split()[2])
        if 'Thermal correction to Energy=' in lines[iline]: molecule.DeltaE2U = float(lines[iline].split()[-1])
        if 'Thermal correction to Enthalpy=' in lines[iline]: molecule.DeltaE2H = float(lines[iline].split()[-1])
        if 'Thermal correction to Gibbs Free Energy=' in lines[iline]: molecule.DeltaE2G = float(lines[iline].split()[-1])
        if 'Sum of electronic and zero-point Energies=' in lines[iline]:
            molecule.U0 = float(lines[iline].split()[-1])
            molecule.H0 = molecule.U0
        if 'Sum of electronic and thermal Energies=' in lines[iline]: molecule.U = float(lines[iline].split()[-1])
        if 'Sum of electronic and thermal Enthalpies=' in lines[iline]: molecule.H = float(lines[iline].split()[-1])
        if 'Sum of electronic and thermal Free Energies=' in lines[iline]: molecule.G = float(lines[iline].split()[-1])
        if 'E (Thermal)             CV                S' in lines[iline]:
            molecule.S = float(lines[iline+2].split()[-1]) * constants.kcalpermol2Hartree / 1000.0
        if 'Normal termination of Gaussian' in lines[iline]:
            successful = True
    for atom in molecule.atoms:
        atom.normal_modes = np.array(atom.normal_modes)
    return successful

def read_anharmonic_frequencies(outputfile,molecule):
    freq_len = len(molecule.frequencies)
    anharmonic_frequencies = []
    harmonic_frequencies = []
    anharmonic_overtones = []
    harmonic_overtones = [] 
    anharmonic_combination_bands = [] 
    harmonic_combination_bands = []
    fundamental_bands_flag = True 
    overtones_flag = True 
    combination_bands_flag = True 
    anharmonic_infrared_intensities = []
    anharmonic_overtones_infrared_intensities = [] 
    anharmonic_combination_bands_infrared_intensities = [] 
    IRflags = []
    with open(outputfile,'r') as f:
        lines = f.readlines()
    for iline in range(len(lines)):
        if 'Fundamental Bands' in lines[iline] and fundamental_bands_flag:
            flag = iline+3
            for ii in range(len(molecule.frequencies)):
                templine = lines[flag+ii]
                anharmonic_frequencies.append(eval(templine[38:48]))
                harmonic_frequencies.append(eval(templine[24:38]))
                # temp = lines[flag+ii].strip().split()
                # anharmonic_frequencies.append(eval(temp[-4]))
                # harmonic_frequencies.append(eval(temp[-5]))
            fundamental_bands_flag = False
                
        if 'Overtones' in lines[iline] and overtones_flag:
            flag = iline+3 
            read_overtone = True  
            icount = 0
            while read_overtone:
                templine = lines[flag+icount]
                temp = lines[flag+icount].strip().split()
                if temp == []:
                    read_overtone = False 
                else:
                    anharmonic_overtones.append(eval(templine[38:48]))
                    harmonic_overtones.append(eval(templine[24:38]))
                    icount += 1 
            overtones_flag = False
        if 'Combination Bands' in lines[iline] and combination_bands_flag:
            flag = iline+3 
            read_combinaion_band = True 
            icount = 0 
            while read_combinaion_band:
                templine = lines[flag+icount]
                temp = lines[flag+icount].strip().split() 
                if temp == []:
                    read_combinaion_band = False 
                else:
                    anharmonic_combination_bands.append(eval(templine[38:48]))
                    harmonic_combination_bands.append(eval(templine[24:38]))
                    icount += 1 
            combination_bands_flag = False
        if 'ZPE(anh)' in lines[iline]:
            # molecule.anharmonic_ZPE = eval(lines[iline].strip().split()[-2].replace('D','E')) * constants.kJpermol2Hartree
            molecule.anharmonic_ZPE = eval(lines[iline][48:61].replace('D','E')) * constants.kJpermol2Hartree
        if 'Input values of T(K) and P(atm):' in lines[iline]:
            temperature = eval(lines[iline].strip().split()[-2])
            E = molecule.U - molecule.DeltaE2U
            molecule.anharmonic_U0 = E + molecule.anharmonic_ZPE
            molecule.anharmonic_H0 = E + molecule.anharmonic_ZPE
            molecule.anharmonic_DeltaE2U = eval(lines[iline+4].strip().split()[-2].replace('D','E')) * constants.kJpermol2Hartree
            molecule.anharmonic_U = E + molecule.anharmonic_DeltaE2U
            molecule.anharmonic_DeltaE2H = eval(lines[iline+5].strip().split()[-2].replace('D','E')) * constants.kJpermol2Hartree
            molecule.anharmonic_H = E + molecule.anharmonic_DeltaE2H
            molecule.anharmonic_S = eval(lines[iline+6].strip().split()[-3].replace('D','E')) * constants.kJpermol2Hartree / 1000.0
            molecule.anharmonic_G = molecule.anharmonic_H - temperature * molecule.anharmonic_S
            molecule.anharmonic_DeltaE2G = molecule.anharmonic_G - E 
        # Read infrared intensities if found
        if 'I(anharm)' in lines[iline]:
            if 'dipole_derivatives' in molecule.__dict__.keys():
                IRflags.append(iline+1) 
    
    # Read infrared intensities if found
    if len(IRflags) == 3:
        # Read fundamental bands intensities
        flag = IRflags[0]
        for ii in range(len(molecule.frequencies)):
            templine = lines[flag+ii]
            try:
                intensity = eval(templine.strip().split()[-1]) * 100 # the intensities are rescaled by a factor of 100
                anharmonic_infrared_intensities.append(intensity)
            except:
                anharmonic_infrared_intensities.append(math.nan)
        # Read overtones intensities
        flag = IRflags[1]
        read_overtone = True  
        icount = 0
        while read_overtone:
            templine = lines[flag+icount]
            temp = lines[flag+icount].strip().split()
            if temp == []:
                read_overtone = False 
            else:
                try:
                    intensity = eval(temp[-1]) * 100 # the intensities are rescaled by a factor of 100
                    anharmonic_overtones_infrared_intensities.append(intensity)
                except:
                    anharmonic_overtones_infrared_intensities.append(math.nan)
                icount += 1 

        # Read combination bands intensities
        flag = IRflags[2]
        read_combinaion_band = True 
        icount = 0 
        while read_combinaion_band:
            templine = lines[flag+icount]
            temp = lines[flag+icount].strip().split() 
            if temp == []:
                read_combinaion_band = False 
            else:
                try:
                    intensity = eval(temp[-1]) * 100 # the intensities are rescaled by a factor of 100
                    anharmonic_combination_bands_infrared_intensities.append(intensity)
                except:
                    anharmonic_combination_bands_infrared_intensities.append(math.nan)
                icount += 1 
        

    molecule.anharmonic_frequencies = []
    order = np.argsort(harmonic_frequencies)
    for each in order:
        molecule.anharmonic_frequencies.append(anharmonic_frequencies[each])
    if anharmonic_infrared_intensities != []:
        molecule.anharmonic_infrared_intensities = []
        for each in order:
            molecule.anharmonic_infrared_intensities.append(anharmonic_infrared_intensities[each])
        molecule.anharmonic_overtones_infrared_intensities = anharmonic_overtones_infrared_intensities
        molecule.anharmonic_combination_bands_infrared_intensities = anharmonic_combination_bands_infrared_intensities
    molecule.anharmonic_overtones = anharmonic_overtones
    molecule.harmonic_overtones = harmonic_overtones 
    molecule.anharmonic_combination_bands = anharmonic_combination_bands 
    molecule.harmonic_conmination_bands = harmonic_combination_bands

def check_Gaussian_job_status(lines,flag,mol):
    if flag == -1:
        pass
    mol.Gaussian_job_status = False
    # for line in lines:
    #     if 'Normal termination of Gaussian' in line:
    #         successful = True 
    if 'Normal termination of Gaussian' in lines[flag]:
        mol.Gaussian_job_status = True
    #return successful

def read_energy_gradients_from_Gaussian_output(lines,flag,mol):
    if flag == -1:
       return
    Natoms = len(mol.atoms)
    for iatom in range(Natoms):
        force = np.array([eval(each) for each in lines[flag+iatom].strip().split()[2:]])
        mol.atoms[iatom].energy_gradients = - force / constants.Bohr2Angstrom

def read_mulliken_charges_from_Gaussian_output(lines,flag,mol):
    if flag == -1:
        return
    Natoms = len(mol.atoms)
    for iatom in range(Natoms):
        charge = eval(lines[flag+iatom].strip().split()[2])
        mol.atoms[iatom].mulliken_charge = charge 

def read_dipole_moment_from_Gaussian_output(lines,flag,mol):
    if flag == -1:
        return
    try:
        raw = lines[flag].strip().split() 
        dipole = np.array([eval(raw[1]),eval(raw[3]),eval(raw[5]),eval(raw[7])])
        mol.dipole_moment = dipole
    except: 
        return 

def read_electronic_energy_from_Gaussian_output(lines,flag,mol,energy_to_read):
    if flag == -1:
        return
    tmpline = lines[flag]
    archive = ''
    while(tmpline!='\n'):
        archive += tmpline[1:-1]
        flag += 1
        tmpline = lines[flag]
    archive_split = archive.split('\\') # split with \
    for ii in range(len(archive_split)):
        for each_children in energy_to_read.children:
            if each_children.name.casefold()+'=' in archive_split[ii].casefold():
                energy = eval(archive_split[ii].split('=')[-1])
                mol.__dict__[each_children.name] = energy
    mol.energy = mol.__dict__[energy_to_read.children[-1].name]

def read_Hessian_matrix_from_Gaussian_output(filename,molecule):
    natoms = len(molecule.atoms)
    hessian = np.zeros((3*natoms,3*natoms))
    lines = open(filename,'r').readlines()
    for iline in range(len(lines)):
        if 'The second derivative matrix:' in lines[iline]:
            flag = iline + 1
    nblocks = int((3*natoms-0.5)//5 + 1)
    icount = 0
    for iblock in range(nblocks):
        for ii in range(3*natoms-5*iblock):
            temp = [eval(each) for each in lines[flag+icount+ii+1].strip().split()[1:]]
            for jj in range(min(5,3*natoms-5*iblock)):
                if ii >= jj:
                    hessian[ii+5*iblock,jj+5*iblock] = temp[jj]
        icount += 3*natoms-5*iblock+1
    for ii in range(3*natoms):
        for jj in range(ii+1,3*natoms):
            hessian[ii,jj] = hessian[jj,ii]
    hessian /= constants.Bohr2Angstrom**2
    molecule.hessian = hessian

def read_xyz_coordinates_from_Gaussian_output(filename:str):
    lines = open(filename,'r').readlines()
    # Choose the last standard orientation 
    for iline in range(len(lines)):
        if 'Standard orientation:' in lines[iline]:
            flag = iline+5
    xyz_string = ''
    icount = 0
    while True:
        xyz = lines[flag+icount].strip().split()
        if len(xyz) > 1:
            xyz_string += f'{xyz[1]}  {xyz[3]}  {xyz[4]}  {xyz[5]}\n'
        else:
            break 
        icount += 1
    natoms = icount + 1
    xyz_string = f'{natoms}\n\n' + xyz_string[:-1]
    return data.molecule.from_xyz_string(string=xyz_string)

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



class Gaussian_output_reading_assistant():
    def __init__(self,calculate_energy=True,calculate_energy_gradients=False,calculate_hessian=False,
                 energy_to_read=None):
        # Some arguments of Gaussian job
        self.calculate_energy = calculate_energy 
        self.calculate_energy_gradients = calculate_energy_gradients 
        self.calculate_hessian = calculate_hessian
        self.energy_to_read = energy_to_read
        # What to read in Gaussian output file
        self.read_energy = False 
        self.read_energy_gradients = False 
        self.read_hessian = False 
        self.read_Mulliken_charges = True 
        self.read_dipole_moment = True 
        self.read_job_status = True 
        if self.calculate_energy:
            self.read_energy = True 
        if self.calculate_energy_gradients:
            self.read_energy_gradients = True 
        if self.calculate_hessian:
            self.read_hessian = True 

    def read_output(self,filename,molecule):
        job_status_flag = -1
        mulliken_charge_flag = -1
        dipole_moment_flag = -1
        forces_flag = -1
        energy_flag = -1
        with open(filename,'r') as f:
            lines = f.readlines() 
            for iline in range(len(lines)):
                if self.read_job_status:
                    if 'termination' in lines[iline]:
                        job_status_flag = iline
                        check_Gaussian_job_status(lines,job_status_flag,molecule)
                        self.read_job_status = False
                if self.read_energy:
                    if '1\\1\\' in lines[iline]:
                        energy_flag = iline
                        read_electronic_energy_from_Gaussian_output(lines,energy_flag,molecule,energy_to_read=self.energy_to_read)
                        self.read_energy = False
                if self.read_Mulliken_charges:
                    if 'Mulliken charges' in lines[iline]:
                        mulliken_charge_flag = iline+2
                        read_mulliken_charges_from_Gaussian_output(lines,mulliken_charge_flag,molecule)
                        self.read_Mulliken_charges = False 
                if self.read_dipole_moment:
                    if 'Dipole moment' in lines[iline]:
                        dipole_moment_flag = iline+1
                        read_dipole_moment_from_Gaussian_output(lines,dipole_moment_flag,molecule)
                        self.read_dipole_moment = False
                if self.calculate_energy_gradients:
                    if self.read_energy_gradients:
                        if 'Forces (Hartrees/Bohr)' in lines[iline]:
                            forces_flag = iline+3
                            read_energy_gradients_from_Gaussian_output(lines,forces_flag,molecule)
        if self.calculate_hessian:
            newmolecule = molecule.copy()
            freq_flag = read_freq_thermochemistry_from_Gaussian_output(filename,newmolecule)
            #read_Hessian_matrix_from_Gaussian_output(filename,molecule)
            if os.path.exists(filename[:-4]+'.chk'):
                read_Hessian_matrix_from_Gaussian_chkfile(filename[:-4]+'.chk',molecule)
        
        # !!!
        # need to be fixed later in method **read_freq_thermochemistry_from_Gaussian_output** above
        # now just move properties of new mol to old mol
        # !!!
            if freq_flag:
                molecule.frequencies = newmolecule.frequencies
                molecule.force_constants = newmolecule.force_constants
                molecule.reduced_masses = newmolecule.reduced_masses
                for ii in range(len(newmolecule.atoms)):
                    molecule.atoms[ii].normal_modes = newmolecule.atoms[ii].normal_modes
                molecule.ZPE = newmolecule.ZPE
                molecule.DeltaE2U = newmolecule.DeltaE2U
                molecule.DeltaE2H = newmolecule.DeltaE2H
                molecule.DeltaE2G = newmolecule.DeltaE2G
                molecule.U0 = newmolecule.U0
                molecule.H0 = newmolecule.H0
                molecule.U = newmolecule.U
                molecule.H = newmolecule.H
                molecule.G = newmolecule.G
                molecule.S = newmolecule.S


if __name__ == '__main__': 
    model_predict_kwargs_str_file, _, EIn_file, EOu_file, _, _, _ = sys.argv[1:]
    with open(model_predict_kwargs_str_file) as f:
        model_predict_kwargs =  eval(f.read())
    gaussian_external(EIn_file, EOu_file, model_predict_kwargs)