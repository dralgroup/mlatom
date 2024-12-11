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
from .. import models
from ..decorators import doc_inherit


# todo: Update the angular momentum with the Slater Koster files.
angluar_momentums = {
        "H": "s", "He": "s",
        "Li": "s", "Be": "s", "B": "p", "C": "p", "N": "p", "O": "p", "F": "p", "Ne": "p",
        "Na": "s", "Mg": "s", "Al": "p", "Si": "p", "P": "p", "S": "p", "Cl": "p", "Ar": "p",
        "K": "s", "Ca": "s", "Sc": "d", "Ti": "d", "V": "d", "Cr": "d", "Mn": "d", "Fe": "d",
        "Co": "d", "Ni": "d", "Cu": "d", "Zn": "d", "Br": "p"
    }

class dftbplus_methods(models.OMP_model, metaclass=models.meta_method):
    """
    DFTB+ interface

    ----- Check "https://dftbplus-recipes.readthedocs.io/en/latest/introduction.html"
            and "https://www.dftbplus.org/documentation.html" for tutorial.         -----

    Arguments:
    method (str): method used in DFTB+.
    nthreads(int): number of threads to use in calculations (default: all).
    """

    def __init__(self, method="DFTB", nthreads=None, save_files_in_current_directory=True, working_directory=None):
        self.method = method
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        try:
            self.skdir = os.environ['skfiles']
        except:
            raise ValueError("Cannot find Slater-Koster files, please set the env var: export skfiles=... after downloading them.")
        if nthreads is None:
            from multiprocessing import cpu_count
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads	
 
    def generate_dftbplus_input(self, dirname='.', molecule:data.molecule=None):
        molecule.write_file_with_xyz_coordinates(filename=f"{dirname}/ini_guess.xyz")  # Generate .xyz file as initial guess
        input_lines = ['Geometry = xyzFormat {\n<<< "ini_guess.xyz"\n}\n']  # List to store input info

        if self.calculate_energy_gradients:
            input_lines.append("Analysis{\n\tPrintForces = Yes\n}\n\n")
        # if self.calculate_hessian == True:
        #     opt = 'Driver = {SecondDerivatives{}}\n\n'
        #     input_lines.append(opt)
        # else:
        #     opt = 'Driver = {}\n\n'
        #     input_lines.append(opt)

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
        input_lines.append("}\n}\n")

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
                molecule.energy_gradients = -forces / Bohr2Angstrom
        
        # Read excitation info
        if self.nstates > 1:
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
            molecule.electronic_states = [molecule.copy(atomic_labels=[], molecular_labels=[]) for ii in range(molecule.nstates)]
            molecule.electronic_states[self.current_state].energy = molecule.energy
            if self.calculate_energy_gradients:
                molecule.electronic_states[self.current_state].energy_gradients = molecule.energy_gradients
            if self.current_state > 0:
                molecule.electronic_states[0].energy = molecule.electronic_states[self.current_state].energy - molecule.excitation_energies[self.current_state-1]
                for istate in range(1,molecule.nstates):
                    molecule.electronic_states[istate].energy = molecule.electronic_states[0].energy + molecule.excitation_energies[istate-1]

    @doc_inherit
    def predict(self, 
                molecule:data.molecule=None, 
                molecular_database:data.molecular_database=None, 
                calculate_energy:bool=True,
                calculate_energy_gradients:bool=False,
                #calculate_hessian:bool=False,
                nstates:int=1, 
                current_state:int=0):
        """
         Making prediction using DFTB+ program.
        :param molecule: ml.molecule obj.
        :param molecular_database: ml.molecular_database obj.
        :param calculate_energy: bool, default True,
        :param calculate_energy_gradients: bool, default False,
        :param nstates: int, number of electronic states to be calculated.
        :param current_state: int, the electronic state of interest.
        :return: None
        """
        self.calculate_energy = calculate_energy
        self.calculate_energy_gradients = calculate_energy_gradients
        #self.calculate_hessian = calculate_hessian
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
