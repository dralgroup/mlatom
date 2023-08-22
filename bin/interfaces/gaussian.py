#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! gaussian: interface to the Gaussian program                               ! 
  ! Implementations by: Pavlo O. Dral, Peikun Zheng, Yi-Fan Hou               !
  !---------------------------------------------------------------------------! 
'''

import os, sys, subprocess
import numpy as np
pythonpackage = True
try:
    from .. import constants
    from .. import data
    from .. import models
    from .. import stopper
except:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import constants
    import data
    import models
    import stopper
    pythonpackage = False


class gaussian_methods():
    def __init__(self,method='B3LYP/6-31G*',nthreads=1):
        if not "GAUSS_EXEDIR" in os.environ:
            if pythonpackage: raise ValueError('enviromental variable GAUSS_EXEDIR is not set')
            else: stopper.stopMLatom('enviromental variable GAUSS_EXEDIR is not set')
        self.method = method 
        self.find_energy_to_read_in_Gaussian()
        self.nthreads = nthreads

    def predict(self,molecular_database=None,molecule=None,gaussian_keywords=None,
                calculate_energy=True,
                calculate_energy_gradients=False,
                calculate_hessian=False):            
        method = self.method
        #self.energy_to_read = energy_to_read
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_energy = calculate_energy
        self.calculate_hessian = calculate_hessian
        if calculate_energy_gradients:
            if not calculate_hessian:
                if 'force'.casefold() in method.casefold():
                    pass 
                else:
                    method += ' force(nostep)'
        if calculate_hessian:
            if 'freq'.casefold() in method.casefold():
                pass 
            else:
                method += ' freq'     
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)

        #print(molDB.molecules[0].__dict__)
        for imol in range(len(molDB.molecules)):
            imolecule = molDB.molecules[imol]
            # Run Gaussian job
            if gaussian_keywords != None:
                run_gaussian_job(filename='molecule'+str(imol)+'.com',molecule=imolecule,gaussian_keywords=gaussian_keywords,nthreads=self.nthreads,method=method)
            else:
                run_gaussian_job(filename='molecule'+str(imol)+'.com',molecule=imolecule,nthreads=self.nthreads,method=method)

            # Read Gaussian output file
            self.parse_gaussian_output('molecule'+str(imol)+'.log',imolecule)
            

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
    if not 'gaussian_keywords' in kwargs: gaussian_keywords = '%nproc=' + '%d\n' % nthreads
    if 'external_task' in kwargs:
        pythonbin = sys.executable
        path_to_this_file=os.path.abspath(__file__)
        external_task = kwargs['external_task']
        if 'gaussian_keywords' in kwargs:
            gaussian_keywords += "\nexternal='%s %s'\n" % (pythonbin, path_to_this_file)
        elif external_task.lower() == 'opt':
            gaussian_keywords += "#p opt(nomicro) external='%s %s'\n" % (pythonbin, path_to_this_file)
        elif external_task.lower() == 'freq':
            gaussian_keywords += "#p freq external='%s %s'\n" % (pythonbin, path_to_this_file)
        elif external_task.lower() == 'ts':
            gaussian_keywords += "#p opt(ts,calcfc,noeigen,nomicro) external='%s %s'\n" % (pythonbin, path_to_this_file)
        elif external_task.lower() == 'irc':
            gaussian_keywords += "#p irc(calcfc) external='%s %s'\n" % (pythonbin, path_to_this_file)
    else:
        if 'gaussian_keywords' in kwargs:
            gaussian_keywords += '\n'
        if 'method' in kwargs:
            if 'freq' in kwargs['method']:
                kwargs['method'] = 'p ' + kwargs['method']
            gaussian_keywords += '# '+kwargs['method']+'\n'
    
    write_gaussian_input_file(filename=filename, molecule=molecule, gaussian_keywords=gaussian_keywords, nthreads=nthreads)
    
    Gaussianbin, _ = check_gaussian()
    proc = subprocess.Popen([Gaussianbin, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    proc.wait()
    
def check_gaussian():
    status = os.popen('echo $GAUSS_EXEDIR').read().strip()
    if len(status) != 0:
        Gaussianroot = status.split('bsd')[0]
        if 'g16' in Gaussianroot:
            version = 'g16'
        elif 'g09' in Gaussianroot:
            version = 'g09'
        Gaussianbin = Gaussianroot + version
    else :
        stopper.stopMLatom('Can not find Gaussian software in the environment variable, $GAUSS_EXEDIR variable not exists')
    version = version.replace('g', '')
    return Gaussianbin, version

def write_gaussian_input_file(**kwargs):
    if 'filename' in kwargs: filename = kwargs['filename']
    if 'molecule' in kwargs: molecule = kwargs['molecule']
    if 'gaussian_keywords' in kwargs: gaussian_keywords = kwargs['gaussian_keywords']
        
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

def gaussian_external(EIn_file, EOu_file):
    # write new coordinate into 'xyz_temp.dat'
    derivs, molecule = read_gaussian_EIn(EIn_file)
    # calculate energy, gradients, hessian for new coordinates
    import json
    with open('model.json', 'r') as fjson:
        model_dict = json.load(fjson)
    if 'method' in model_dict:
        kwargs = {}
        if 'kwargs' in model_dict:
            kwargs = model_dict['kwargs']
            del model_dict['kwargs']
        model = models.methods(**model_dict, **kwargs)
    calc_hessian = False
    if derivs == 2: calc_hessian = True
    model.predict(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=calc_hessian)
    if not 'energy' in molecule.__dict__:
        if pythonpackage: raise ValueError('model did not return any energy')
        else: stopper.stopMLatom('model did not return any energy')
    write_gaussian_EOu(EOu_file, derivs, molecule)
    
    if os.path.exists('gaussian_opttraj.json'):
        opttraj = data.molecular_trajectory()
        opttraj.load(filename='gaussian_opttraj.json', format='json')
        nsteps = len(opttraj.steps)
        opttraj.steps.append(data.molecular_trajectory_step(step=nsteps, molecule=molecule))
        opttraj.dump(filename='gaussian_opttraj.json', format='json')
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
    try:
        import fortranformat
    except:
        if pythonpackage: raise ValueError('Python module fortranformat was not found')
        else: stopper.stopMLatom('Python module fortranformat was not found')
    with open(EOu_file, 'w') as fEOu:
        # energy, dipole-moment (xyz)   E, Dip(I), I=1,3
        writer = fortranformat.FortranRecordWriter('(4D20.12)')
        output = writer.write([molecule.energy, 0.0, 0.0, 0.0])
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
    for atom in molecule.atoms:
        atom.normal_modes = []
    for iline in range(len(lines)):
        if 'Frequencies --' in lines[iline]: molecule.frequencies += [float(xx) for xx in lines[iline].split()[2:]]
        if 'Red. masses --' in lines[iline]: molecule.force_constants += [float(xx) for xx in lines[iline].split()[3:]]
        if 'Frc consts  --' in lines[iline]: molecule.reduced_masses += [float(xx) for xx in lines[iline].split()[3:]]
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
            break
    for atom in molecule.atoms:
        atom.normal_modes = np.array(atom.normal_modes)
    return successful

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
    raw = lines[flag].strip().split() 
    dipole = np.array([eval(raw[1]),eval(raw[3]),eval(raw[5]),eval(raw[7])])
    mol.dipole_moment = dipole

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
            read_freq_thermochemistry_from_Gaussian_output(filename,molecule)




if __name__ == '__main__': 
    _, EIn_file, EOu_file, _, _, _ = sys.argv[1:]    
    gaussian_external(EIn_file, EOu_file)