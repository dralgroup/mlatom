#!/usr/bin/env python3
'''
.. code-block::

  !--------------------------------------------------------------------! 
  ! dftbplus_interface: interface to the DFTB+ program                 ! 
  ! Implementations by: Xinyu Tong and Pavlo O. Dral                   !
  !--------------------------------------------------------------------! 
'''


import os
import numpy as np
from .. import data
from ..constants import eV2Hartree, Bohr2Angstrom
import subprocess
from ..model_cls import OMP_model, method_model
from ..decorators import doc_inherit


# todo: Update the angular momentum with the Slater Koster files.
angluar_momentums = {
        "H": "s", "He": "s",
        "Li": "s", "Be": "s", "B": "p", "C": "p", "N": "p", "O": "p", "F": "p", "Ne": "p",
        "Na": "s", "Mg": "s", "Al": "p", "Si": "p", "P": "p", "S": "p", "Cl": "p", "Ar": "p",
        "K": "s", "Ca": "s", "Sc": "d", "Ti": "d", "V": "d", "Cr": "d", "Mn": "d", "Fe": "d",
        "Co": "d", "Ni": "d", "Cu": "d", "Zn": "d", "Br": "p"
    }

Hubbar_Derivs = {"Br": -0.0573, "C": -0.1492, "Ca": -0.0340, "Cl": -0.0697, "F": -0.1623, "H": -0.1857, "I": -0.0433, "K": -0.0339,
                 "Mg": -0.02,   "N": -0.1535, "Na": -0.0454,  "O": -0.1575, "P": -0.14,   "S": -0.11,  "Zn": -0.03}

class dftbplus_methods(OMP_model, method_model):
    """
    DFTB+ interface

    ----- Check "https://dftbplus-recipes.readthedocs.io/en/latest/introduction.html"
            and "https://www.dftbplus.org/documentation.html" for tutorial.         -----

    Relevant Publications
    =====================
    [JCTC2013] J. Chem. Theory Comput., 2013, 9, 338-354.
    [JCTC2014] J. Chem. Theory Comput., 2014, 10, 1518–1537.
    [JCTC2015-1] J. Phys. Chem. B, 2015, 119, 1062–1082.
    [JCTC2015-2] J. Chem. Theory Comput., 2015, 11, 332–342.

    Required references
    ===================
    O - N - C - H:		[JCTC2013]
    P,S-*:	    		[JCTC2014]
    Mg,Zn-*:		    [JCTC2015-1]
    Na,F,K,Ca,Cl,Br,I-*:[JCTC2015-2]

    Arguments:
    method (str): method used in DFTB+.
    nthreads(int): number of threads to use in calculations (default: all).
    """
    supported_methods = ['DFTB']

    def __init__(self, method="DFTB", nthreads=None, save_files_in_current_directory=True, working_directory=None, scf_tolerance=None):
        self.method = method
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.scf_tolerance = scf_tolerance
        # print("Please use '3ob' parameterizations if your system contains only the following atoms:\nBr-C-Ca-Cl-F-H-I-K-Mg-N-Na-O-P-S-Zn")
        if self.is_program_found:
            self.skdir = self.get_skdir()
            if self.skdir is None:
                raise ValueError("Cannot find Slater-Koster files, please set the env var: export skfiles=... after downloading them.")
        else:
            errmsg = 'Cannot find the DFTB+ program, please install it so that it can be called via dftb+ command.'
            raise ValueError(errmsg)
             
        if nthreads is None:
            from multiprocessing import cpu_count
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads	

    @classmethod
    def is_program_found(cls):
        import shutil
        found = False
        if shutil.which('dftb+'):
            skdir = cls.get_skdir()
            if skdir is not None:
                found = True
        return found

    @classmethod
    def get_skdir(cls):
        if 'skfiles' in os.environ:
            return os.environ['skfiles']
        else:
            return None
 
    def generate_dftbplus_input(self, dirname='.', molecule:data.molecule=None):
        molecule.write_file_with_xyz_coordinates(filename=f"{dirname}/ini_guess.xyz")  # Generate .xyz file as initial guess
        input_lines = ['Geometry = xyzFormat {\n<<< "ini_guess.xyz"\n}\n']  # List to store input info

        if self.calculate_energy_gradients:
            if (self.nstates > 1 and self.current_state > 0) or self.nstates == 1:
                input_lines.append("Analysis{\n\tPrintForces = Yes\n}\n\n")
            elif self.nstates > 1 and self.current_state == 0:
                print("Excited-state forces only available when cuerrent_state > 0, forces will not be calculated.")
        if self.calculate_hessian == True:
            opt = 'Driver = {SecondDerivatives{}}\n\n'
            input_lines.append(opt)
        # else:
        #     opt = 'Driver = {}\n\n'
        #     input_lines.append(opt)

        if self.calculate_hessian:
            if self.scf_tolerance is None:
                scc_convergence = "    SCCTolerance = 1E-7\n" # The manual suggests SCCTolerance=1E-7 or better to get accurate results for the second derivatives.
            else:
                scc_convergence = f"    SCCTolerance = {self.scf_tolerance}" + '\n'
        else:
            if self.scf_tolerance is None:
                scc_convergence = "" # The default SCCTolerance is 1E-5
            else:
                scc_convergence = f"    SCCTolerance = {self.scf_tolerance}" + '\n'

        # Figure out the element types and import the element interaction files (Slater-Koster files)
        element_type = set(molecule.element_symbols)
        file_name = []
        for ii in element_type:
            for jj in element_type:
                file_name.append(f'{ii}-{jj} = "{self.skdir}/{ii}-{jj}.skf"')
        file_name = tuple(set(file_name))

        # Specify the Hamiltonian
        hamiltonian_1 = "Hamiltonian = DFTB {\n\tScc = Yes\n\tSlaterKosterFiles {\n"
        input_lines.append(hamiltonian_1)
        for i in range(len(file_name)):
            input_lines.append("\t\t"+file_name[i]+"\n")
        input_lines.append("\t}\nMaxAngularMomentum{")
        for ii in element_type:
            input_lines.append(f'{ii} = "{angluar_momentums[ii]}"\n')
        input_lines.append("}\n")
        input_lines.append(scc_convergence)
        try:
            if self.nstates == 1:
                tmpline = []
                tmpline.append("HubbardDerivs{")
                for ii in element_type:
                    tmpline.append(f"{ii} = {Hubbar_Derivs[ii]}\n")
                tmpline.append("}\n")
                tmpline.append("ThirdOrderFull = Yes\n")
                tmpline.append("HCorrection = Damping{Exponent=4.0}\n}\n\n")
                input_lines += tmpline
            else:
                raise ValueError()
        except:
            input_lines.append("}\n")

        # Linear response time-dependent DFTB calculation
        if self.nstates > 1:
            input_lines.append("ExcitedState{\n\tCasida{\n")
            input_lines.append(f"\t\tNrOfExcitations = {self.nstates-1}\n")

            # Check multiplicity
            if molecule.multiplicity == 1:
                input_lines.append("\t\tSymmetry = singlet\n")
            elif molecule.multiplicity == 3:
                input_lines.append("\t\tSymmetry = triplet\n")

            input_lines.append("\t\tDiagonalizer = Arpack{}\n")
            input_lines.append("\t\tWriteTransitions = Yes\n")
            input_lines.append("\t\tWriteSPTransitions = Yes\n")

            # Check excited state of interest
            if self.current_state > 0:
                input_lines.append(f"\t\tStateOfInterest = {self.current_state}\n")
                input_lines.append("\t\tExcitedStateForces = Yes\n\n")
            input_lines.append("}\n}\n")

        # todo: Add electron dynamic input lines

        # Lines of Options, Analysis and ParserOptions
        input_lines.append("Options {\n\tWriteResultsTag = Yes}\n\n")

        input_lines.append("ParserOptions {\n\tParserVersion = 12\n}\n")

        input_lines.append("Parallel = {UseOmpThreads = Yes}")

        with open(f"{dirname}/dftb_in.hsd", "w") as input_file:
            for item in input_lines:
                input_file.write(item)

    
    def parse_output(self, dirname='.', molecule:data.molecule=None):
        """
        Parsing DFTB+ standard output files.
        """
        if molecule is None:
            molecule = data.molecule.from_xyz_file(f"{dirname}/ini_guess.xyz")
                            
        forces = []
        energy_gradients = []
        with open(f"{dirname}/results.tag", "r") as file:
            lines = file.readlines()

            # Read total_energy
            for idx in range(len(lines)):
                if "total_energy" in lines[idx]:
                    molecule.energy = float(lines[idx+1])
                    continue
                
            # Read forces and energy gradients
            for idx in range(len(lines)):
                if "forces" in lines[idx]:
                    for line in lines[idx+1:]:
                        temp_list = line.split()
                        if len(temp_list) != 3:
                            break
                        forces.append([float(val) for val in temp_list])
            if len(forces) > 0:
                forces = np.array(forces)
                energy_gradients = -forces / Bohr2Angstrom
        
        # Read hessian
        if self.calculate_hessian == True:
            hessian = []
            shape = len(molecule.atoms) * 3

            with open(f"{dirname}/results.tag", "r") as file:
                lines = file.readlines()

                for idx in range(len(lines)):
                    if "hessian" in lines[idx]:
                        for line in lines[idx+1:]:
                            tmplst = line.split()
                            if len(tmplst) != 3:
                                break
                            hessian.append([float(val) for val in tmplst])
            hessian = np.array(hessian).reshape(shape, shape) / (Bohr2Angstrom ** 2)
            if self.current_state == 0:
                molecule.hessian = hessian
        

        # Read excitation info
        if self.nstates > 1:
            xyzstring = molecule.get_xyz_string()
            gsmol = data.molecule.from_xyz_string(xyzstring)
            gsmol.energy = molecule.energy
            excitation_energies, oscillator_strengths = [], []
            with open(f"{dirname}/EXC.DAT", "r") as file:
                lines = file.readlines()
                for line in lines[5:]:
                    if line == "\n":
                        break
                    temp_list = line.split()
                    excitation_energies.append(float(temp_list[0]) * eV2Hartree)
                    oscillator_strengths.append(float(temp_list[1]))
            molecule.excitation_energies = np.array(excitation_energies)
            molecule.oscillator_strengths = np.array(oscillator_strengths)
            molecule.electronic_states = [gsmol]
            
            if self.current_state == 0:
                for ii in range(len(excitation_energies)):
                    state_mol = data.molecule.from_xyz_string(xyzstring)
                    state_mol.energy = gsmol.energy + excitation_energies[ii]
                    molecule.electronic_states.append(state_mol)
                # if self.calculate_energy_gradients:
                #     molecule.add_xyz_vectorial_property(vector=energy_gradients, xyz_vectorial_property="energy_gradients")
            
            elif self.current_state > 0:
                molecule.electronic_states += [data.molecule.from_xyz_string(xyzstring) for _ in range(self.nstates - 1)]
                molecule.electronic_states[self.current_state].energy = molecule.energy
                molecule.electronic_states[0].energy = molecule.electronic_states[self.current_state].energy - molecule.excitation_energies[self.current_state-1]
                for istate in range(1,self.nstates):
                    molecule.electronic_states[istate].energy = molecule.electronic_states[0].energy + molecule.excitation_energies[istate-1]
                if self.calculate_energy_gradients:
                    molecule.electronic_states[self.current_state].add_xyz_derivative_property(derivative=energy_gradients, property_name="energy", xyz_derivative_property="energy_gradients")
                    molecule.add_xyz_derivative_property(derivative=energy_gradients, property_name="energy", xyz_derivative_property="energy_gradients")
                if self.calculate_hessian:
                    molecule.electronic_states[self.current_state].hessian = hessian
                    molecule.hessian = hessian

    @doc_inherit
    def predict(self, 
                molecule:data.molecule=None, 
                molecular_database:data.molecular_database=None, 
                calculate_energy:bool=True,
                calculate_energy_gradients:bool=False,
                calculate_hessian:bool=False,
                nstates:int=1, 
                current_state:int=0):
        """
         Making prediction using DFTB+ program.
        :param molecule: ml.molecule obj.
        :param molecular_database: ml.molecular_database obj.
        :param calculate_energy: bool, default True.
        :param calculate_energy_gradients: bool, default False.
        :param nstates: int, number of electronic states to be calculated.
        :param current_state: int, the electronic state of interest.
        :return: None
        """
        self.calculate_energy = calculate_energy
        self.calculate_energy_gradients = calculate_energy_gradients
        self.calculate_hessian = calculate_hessian
        self.nstates = nstates
        self.current_state = current_state
        
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        command = ['dftb+', 'dftb_in.hsd']
        
        import tempfile
        for idx in range(len(molDB)):
            with tempfile.TemporaryDirectory() as tmpdirname:
                if self.save_files_in_current_directory: tmpdirname = '.'
                if self.working_directory is not None:
                    tmpdirname = self.working_directory
                tmpdirname += f'/molecule{idx+1}'
                if not os.path.exists(tmpdirname):
                    os.makedirs(tmpdirname)
                tmpdirname = os.path.abspath(tmpdirname)

                std, err = [], []
                
                self.generate_dftbplus_input(dirname=tmpdirname, molecule=molDB[idx])
                execute_calc = subprocess.Popen(args=' '.join(command),
                                                stdout=subprocess.PIPE, 
                                                stderr=subprocess.PIPE,
                                                cwd=tmpdirname,
                                                shell=True, 
                                                universal_newlines=True)
                
                stdout, stderr = execute_calc.communicate()
                self.parse_output(dirname=tmpdirname, molecule=molDB[idx])
                if len(stdout) > 0:
                    std.append(stdout)
                if len(stderr) > 0:
                    err.append(stderr)

                if len(std) > 0:
                    with open(f"{tmpdirname}/dftbplus.log", "w") as file:
                        for item in std:
                            file.write(item)
                if len(err) > 0:
                    with open(f"{tmpdirname}/dftbplus.err", "w") as file:
                        for item in err:
                            file.write(item)
