#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------!
  ! orca: interface to the ORCA program                                       !
  ! Implementations by: Yuxinxin Chen                                         !
  ! NEVPT2 and TD-DFT implemented by Mikołaj Martyka, Li Wang and Xinyu Tong  !
  !---------------------------------------------------------------------------!
'''

import os
import re
import numpy as np
import math
import tempfile
import subprocess
from .. import constants, data
from ..model_cls import model, method_model
from ..decorators import doc_inherit


class orca_methods(method_model):
    '''
    ORCA interface

    Arguments:
        method (str): Method specification. Use ``'B3LYP/6-31G*'`` for ground state DFT,
                      ``'TD-wb97x-d3bj/def2-SVP'`` for TD-DFT, ``'TDA-wb97x-d3bj/def2-SVP'`` for TDA,
                      ``'qd-nevpt2/cc-pVDZ'`` for QD-NEVPT2, or ``'CCSD(T)*/CBS'`` for composite method.
        save_files_in_current_directory (bool): Keep input/output files, default ``True``
        working_directory (str): Directory for output files, default ``None``
        nthreads (int): Number of parallel processes (``%pal nprocs``)
        maxcore (int): Memory per core in MB (``%maxcore``)
        solvent (str): Solvent name for implicit solvation
        solvation_model (str): ``'SMD'`` or ``'CPCM'``, default ``'SMD'``
        additional_keywords (list): Extra keywords for ORCA input line
        Example setup:
        .. code-block:: python

            td_dft = ml.models.methods(method='TD-B3LYP/def2-SVP', program='orca',
                                    working_directory='./orca_tddft',
                                    nthreads=1,
                                    maxcore=1000,
                                    )

            qd_nevpt2 = ml.models.methods(method='QD-NEVPT2/def2-SVP', program='orca',
                            nthreads=1,
                            maxcore=1000,
                            working_directory='./orca_NEVPT2',
                            )
    '''

    bin_env_name = 'orcabin'

    def __init__(self,
                 method='wb97x/6-31G*',
                 save_files_in_current_directory: bool = True,
                 working_directory: str = None,
                 nthreads: int = 1,
                 maxcore: int = 1000,
                 solvent: str = None,
                 solvation_model: str = "SMD",
                 additional_keywords: list = None,
                 input_file: str = '',
                 output_keywords: list = None,
                 **kwargs):

        self.orcabin = self.get_bin_env_var()
        if self.orcabin is None:
            raise ValueError('Cannot find the Orca program, please set the environment variable: export orcabin=...')

        self.method = method
        self.nthreads = nthreads
        self.maxcore = maxcore
        self.solvent = solvent
        self.solvation_model = solvation_model if solvent else None
        self.save_files_in_current_directory = save_files_in_current_directory
        self.working_directory = working_directory
        self.additional_keywords = additional_keywords
        self.input_file = input_file
        self.output_keywords = output_keywords

        # For CCSD(T)*/CBS composite method
        if self.method == 'CCSD(T)*/CBS':
            self.cc_method = ccsdtstarcbs(**kwargs)

    @doc_inherit
    def predict(self,
                molecular_database: data.molecular_database = None,
                molecule: data.molecule = None,
                calculate_energy: bool = True,
                calculate_energy_gradients = False,
                calculate_hessian: bool = False,
                nstates: int = 1,
                current_state: int = 0,
                active_space: tuple = None,
                casscf_kwargs: dict = None):
        '''
        Run ORCA calculation.

        Arguments:
            molecular_database: collection of molecules stored in ml.data.molecular_database object to make predictions for.
            molecule: molecule to predict for.
            calculate_energy: bool, whether to update the energy of the molecule.
            calculate_energy_gradients: bool or list of bool.
                If list, e.g. [False, True, False] requests gradient for S1 only.
                Only one state gradient can be requested at a time.
            current_state: 0-indexed state (0 = S0, 1 = S1, etc.), for which the energy and gradient will be used.
            active_space: tuple (nel, norb) for CASSCF/QD-NEVPT2, e.g. (6, 6)
            casscf_kwargs: dict of additional CASSCF keywords (maxiter, actorbs, etc.)
            nstates: number of electronic states to compute
            calculate_hessian: whether to calculate hessian matrix.
        Example usage:
        .. code-block:: python

            td_dft.predict(
                molecule=mol,
                nstates=2,
                current_state=0,
                calculate_energy=True,
                calculate_energy_gradients=True,
            )

            qd_nevpt2.predict(
            molecule=mol,
            nstates=2,
            current_state=1,
            active_space = (2,2)
            )

        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        # Parse TD-/TDA- prefix from method string
        method_upper = self.method.upper()
        if method_upper.startswith('TDA-'):
            self._tda = True
            self._method_cleaned = self.method[4:]  # Strip 'TDA-'
            is_tddft = True
        elif (method_upper.startswith('TD-') or (nstates>1 and not ("PT2" in method_upper)) ): #This is disgustingly ugly, handwritten code that works
            self._tda = False
            self._method_cleaned = self.method[3:]  # Strip 'TD-'
            is_tddft = True
        else:
            self._tda = False
            self._method_cleaned = self.method
            is_tddft = False

        # Validate nstates for TD-DFT
        if is_tddft and nstates == 1:
            raise ValueError("TD-DFT requires nstates > 1. Use nstates=N to calculate N-1 excited states.")

        # Handle calculate_energy_gradients as bool or list
        if isinstance(calculate_energy_gradients, list):
            # Validate: only one True allowed
            true_count = sum(calculate_energy_gradients)
            if true_count > 1:
                raise ValueError("Only one state gradient can be calculated at a time. "
                                f"Got {true_count} states requested: {calculate_energy_gradients}")
            if true_count == 1:
                # Find which state has gradient requested
                self.gradient_state = calculate_energy_gradients.index(True)
                self.calculate_energy_gradients = True
            else:
                self.gradient_state = None
                self.calculate_energy_gradients = False
        else:
            self.calculate_energy_gradients = calculate_energy_gradients
            self.gradient_state = current_state if calculate_energy_gradients else None
        if self.gradient_state ==0 and is_tddft:
            raise ValueError("ORCA TD-DFT does not support ground-state gradient calculations. Please create a composite TD-DFT/DFT model in MLatom.")
        # Store calculation parameters
        self.calculate_energy = calculate_energy
        self.calculate_hessian = calculate_hessian
        self.nstates = nstates
        self.current_state = current_state
        self.active_space = active_space
        self.casscf_kwargs = casscf_kwargs if casscf_kwargs else {}

        if self.method == 'CCSD(T)*/CBS':
            self.cc_method.predict(molecular_database=molDB)
            return

        # Setup working directory
        tmpdirname = self._setup_working_directory()

        for imol, imolecule in enumerate(molDB.molecules):
            self.inpfile = os.path.join(tmpdirname, f'molecule{imol}_{self.input_file}')

            # Generate input and run ORCA
            self._generate_input(imolecule)
            self._run_orca(tmpdirname)
            out_file = f'{self.inpfile}.out'
            if not os.path.exists(out_file):
                raise FileNotFoundError(f"Output file not found: {out_file}")
            with open(out_file, 'r') as f:
                lines = f.readlines()
                # Check last 20 lines for normal termination
                tail = lines[-20:] if len(lines) >= 20 else lines
                if not any('ORCA TERMINATED NORMALLY' in line for line in tail):
                    print(f"ORCA did not terminate normally. Check output: {out_file}. The program will try to continue.")
            import re
            version_match = re.search(r'Program Version (\d+)\.(\d+)\.(\d+)', ''.join(lines))
            if version_match:
                major_version = int(version_match.group(1))
                self._orca_version = major_version
            # Parse output based on method type
            if self._method_cleaned.lower().startswith('qd_nevpt2') or self._method_cleaned.lower().startswith('qd-nevpt2'):
                self._parse_qd_nevpt2_output(imolecule)
            elif not self.calculate_energy and self.output_keywords:
                # Custom keyword parsing (used by CCSD(T)*/CBS components)
                self._parse_output_keywords(imolecule)
            else:
                self._parse_tddft_output(imolecule)
            imolecule.orca_version = self._orca_version
    def _parse_output_keywords(self, molecule):
        '''Parse custom keywords from property.txt file.

        Used by CCSD(T)*/CBS and similar methods to extract specific values.
        '''
        prop_file = f'{self.inpfile}_property.txt'
        if not os.path.exists(prop_file):
            raise FileNotFoundError(f"Property file not found: {prop_file}")

        with open(prop_file, 'r') as f:
            lines = f.readlines()

        for keyword in self.output_keywords:
            keyword_name = self.input_file + '_' + keyword.lower().replace(' ', '_')
            for line in lines:
                if keyword in line:
                    molecule.__dict__[keyword_name] = float(line.split()[-1])
                    break

    def _setup_working_directory(self) -> str:
        '''Setup and return the working directory path.'''
        if self.working_directory is not None:
            tmpdirname = self.working_directory
            if not os.path.exists(tmpdirname):
                os.makedirs(tmpdirname)
        elif self.save_files_in_current_directory:
            tmpdirname = '.'
        else:
            # Create temporary directory (caller should handle cleanup)
            tmpdirname = tempfile.mkdtemp()
        return os.path.abspath(tmpdirname)

    def _generate_input(self, molecule):
        '''Generate ORCA input file.'''
        is_qd_nevpt2 = self._method_cleaned.lower().startswith('qd_nevpt2') or self._method_cleaned.lower().startswith('qd-nevpt2')

        with open(f'{self.inpfile}.inp', 'w') as f:
            # Method/keywords line
            if is_qd_nevpt2:
                # QD-NEVPT2: parse basis from method string (e.g., qd_nevpt2/cc-pVDZ)
                if '/' in self._method_cleaned:
                    basis = self._method_cleaned.split('/')[1]
                else:
                    basis = 'def2-SVP'  # default
                method_line = f'{basis} RIJK AUTOAUX LargePrint'
            elif '/' in self._method_cleaned and len(self._method_cleaned.split('/')) == 2:
                method_line = ' '.join(self._method_cleaned.split('/'))
            else:
                method_line = self._method_cleaned

            if self.additional_keywords:
                method_line += ' ' + ' '.join(self.additional_keywords)
            f.write(f'! {method_line}\n')

            # Calculation type keywords (separate line) - NOT for QD-NEVPT2 or custom keyword extraction (e.g., CCSD(T)*/CBS)
            if not is_qd_nevpt2 and not (not self.calculate_energy and self.output_keywords):
                calc_keywords = []
                if self.calculate_energy:
                    calc_keywords.append('SP')
                if self.calculate_energy_gradients:
                    calc_keywords.append('ENGRAD')
                if self.calculate_hessian:
                    calc_keywords.append('FREQ')
                f.write(f'! {" ".join(calc_keywords)}\n')

            # Parallel and memory settings
            f.write(f'%pal nprocs {self.nthreads} end\n')
            f.write(f'%maxcore {self.maxcore}\n')

            # TD-DFT block (only for non-CASSCF excited states)
            if self.nstates > 1 and not is_qd_nevpt2:
                f.write('%tddft\n')
                f.write(f'  nroots {self.nstates - 1}\n')
                # iroot specifies which state to compute gradient for
                # Use gradient_state if gradients requested, otherwise current_state
                iroot_state = self.gradient_state if self.calculate_energy_gradients else self.current_state
                if iroot_state is not None and iroot_state > 0:
                    f.write(f'  iroot {iroot_state}\n')
                f.write(f'  TDA {str(self._tda).lower()}\n')
                f.write('end\n')

            # CASSCF block for QD-NEVPT2
            if is_qd_nevpt2 and self.active_space:
                # SCF settings - need Conventional mode for RIJK
                f.write('%scf\n')
                f.write('  SCFMode Conventional\n')
                f.write('end\n')

                nel, norb = self.active_space
                f.write('%casscf\n')
                f.write(f'  nel {nel}\n')
                f.write(f'  norb {norb}\n')
                f.write(f'  mult {molecule.multiplicity}\n')
                f.write(f'  nroots {self.nstates}\n')
                # Write any additional casscf_kwargs
                for key, value in self.casscf_kwargs.items():
                    f.write(f'  {key} {value}\n')
                f.write('  trafostep ri\n')
                f.write('  PTMethod SC_NEVPT2\n')
                f.write('  PTSettings\n')
                f.write('    QDType 1\n')
                f.write('  end\n')
                f.write('end\n')

            # Solvation
            if self.solvent:
                if self.solvation_model and self.solvation_model.upper() == 'SMD':
                    f.write('%cpcm\n')
                    f.write('  smd true\n')
                    f.write(f'  smdsolvent "{self.solvent}"\n')
                    f.write('end\n')
                elif self.solvation_model and self.solvation_model.upper() == 'CPCM':
                    f.write(f'! CPCM({self.solvent})\n')

            # Geometry
            f.write(f'* xyz {molecule.charge} {molecule.multiplicity}\n')
            for atom in molecule.atoms:
                f.write(f'{atom.element_symbol:2s}  {atom.xyz_coordinates[0]:14.8f}  '
                        f'{atom.xyz_coordinates[1]:14.8f}  {atom.xyz_coordinates[2]:14.8f}\n')
            f.write('*\n')

    def _run_orca(self, workdir: str):
        '''Execute ORCA calculation.'''
        cmd = f'{self.orcabin} {self.inpfile}.inp > {self.inpfile}.out 2>&1'
        proc = subprocess.Popen(cmd, shell=True, cwd=workdir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.communicate()

    def _parse_qd_nevpt2_output(self, molecule):
        '''Parse QD-NEVPT2 output from .out file.'''
        out_file = f'{self.inpfile}.out'

        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Output file not found: {out_file}")

        with open(out_file, 'r') as f:
            lines = f.readlines()

        # Extract QD-NEVPT2 and CASSCF energies from "QD-NEVPT2 Results" block
        pt2_energies, casscf_energies = self._extract_qd_energies_from_out(lines)

        if not pt2_energies:
            raise ValueError(f"No QD-NEVPT2 Results block found in {out_file}")

        # Build electronic states
        molecule.electronic_states = []
        for root_idx in sorted(pt2_energies.keys()):
            state_mol = data.molecule()
            state_mol.atoms = [atom.copy() for atom in molecule.atoms]
            state_mol.energy = pt2_energies[root_idx]
            if root_idx in casscf_energies:
                state_mol.casscf_energy = casscf_energies[root_idx]
            # Initialize gradients as NaN (not available for QD-NEVPT2)
            for atom in state_mol.atoms:
                atom.energy_gradients = np.full(3, np.nan).tolist()
            molecule.electronic_states.append(state_mol)

        # Set molecule energy to current state
        if self.current_state < len(molecule.electronic_states):
            molecule.energy = molecule.electronic_states[self.current_state].energy

        # QD-NEVPT2 gradients not implemented
        if self.calculate_energy_gradients:
            raise NotImplementedError("Analytical gradients are not available for QD-NEVPT2")

        # Parse Hessian if requested
        if self.calculate_hessian:
            self._parse_hessian(molecule)

        # Parse oscillator strengths from .out file
        self._parse_qd_absorption_spectrum(molecule, out_file)

    def _extract_qd_energies_from_out(self, lines: list) -> tuple:
        '''Extract QD-NEVPT2 and CASSCF energies from .out file.

        Parses "QD-NEVPT2 Results" block:
        ROOT = 0
        ...
        Zero Order Energy       : E0 = -230.38678408697643   (CASSCF)
        Total Energy (E0+dE)    : E  = -231.11493305070951   (QD-NEVPT2)

        Returns:
            tuple: (pt2_energies dict, casscf_energies dict)
        '''
        pt2_energies = {}
        casscf_energies = {}
        in_block = False
        current_root = None

        for line in lines:
            if 'QD-NEVPT2 Results' in line:
                in_block = True
                continue

            if in_block:
                line_stripped = line.strip()

                # Match "ROOT = 0"
                if line_stripped.startswith('ROOT ='):
                    parts = line_stripped.split('=')
                    if len(parts) >= 2:
                        try:
                            current_root = int(parts[1].strip())
                        except ValueError:
                            current_root = None

                # Match "Zero Order Energy       : E0 = -230.386..."
                elif 'Zero Order Energy' in line and current_root is not None:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            casscf_energies[current_root] = float(parts[-1].strip())
                        except ValueError:
                            pass

                # Match "Total Energy (E0+dE)    : E  = -231.114..."
                elif 'Total Energy (E0+dE)' in line and current_root is not None:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            pt2_energies[current_root] = float(parts[-1].strip())
                        except ValueError:
                            pass

                # End of block
                elif '======' in line_stripped and pt2_energies:
                    break

        return pt2_energies, casscf_energies

    def _parse_qd_absorption_spectrum(self, molecule, out_file: str):
        '''Parse oscillator strengths from QD-NEVPT2 absorption spectrum.'''
        if not os.path.exists(out_file):
            return

        with open(out_file, 'r') as f:
            lines = f.readlines()

        # Find last occurrence of absorption spectrum
        last_idx = None
        for i, line in enumerate(lines):
            if 'ABSORPTION SPECTRUM' in line:
                last_idx = i

        if last_idx is None:
            return

        excitation_energies = []
        oscillator_strengths = []

        # Parse spectrum table (skip header lines)
        for line in lines[last_idx + 5:]:
            if '-' * 20 in line or line.strip() == '':
                break
            parts = line.split()
            if len(parts) >= 8:
                # Column 5 is energy in cm^-1, column 7 is oscillator strength
                exc_energy_cm = float(parts[5])
                exc_energy_hartree = exc_energy_cm * 4.556335e-6
                excitation_energies.append(exc_energy_hartree)
                oscillator_strengths.append(float(parts[7]))

        if oscillator_strengths:
            molecule.oscillator_strengths = oscillator_strengths

    def _parse_tddft_output(self, molecule):
        '''Parse TD-DFT/ground-state DFT output.'''
        if self._orca_version != 6:
            prop_file = f'{self.inpfile}_property.txt'
        else:
            prop_file = f'{self.inpfile}.property.txt'

        out_file = f'{self.inpfile}.out'

        # Get ground state energy
        gs_energy = None
        energy_from_prop_file = False
        if os.path.exists(prop_file) and self._orca_version != 6:
            with open(prop_file, 'r') as f:
                for line in f:
                    if 'SCF Energy:' in line or 'TOTAL ENERGY' in line:
                        gs_energy = float(line.split()[-1])
                        energy_from_prop_file = True
                        break
        elif os.path.exists(prop_file):
            with open(prop_file, 'r') as f:
                in_scf_block = False
                for line in f:
                    if '$SCF_Energy' in line:
                        in_scf_block = True
                        continue
                    if in_scf_block:
                        if '&SCF_ENERGY' in line:
                            # Extract the float value at the end
                            gs_energy = float(line.split()[-1])
                            energy_from_prop_file = True
                            break
                        elif line.startswith('$'):
                            # New block started, SCF_Energy block ended
                            break
        if gs_energy is None and os.path.exists(out_file):
            with open(out_file, 'r') as f:
                for line in f:
                    if 'FINAL SINGLE POINT ENERGY' in line:
                        gs_energy = float(line.split()[-1])
                        break

        if gs_energy is None:
            raise ValueError("Could not parse ground state energy")

        # Parse dispersion correction from .out file if energy came from property.txt
        # (FINAL SINGLE POINT ENERGY already includes dispersion, but SCF Energy does not)
        dispersion_correction = 0.0
        if energy_from_prop_file and os.path.exists(out_file):
            with open(out_file, 'r') as f:
                for line in f:
                    if 'Dispersion correction' in line:
                        try:
                            dispersion_correction = float(line.split()[-1])
                        except ValueError:
                            pass
                        break
            gs_energy += dispersion_correction

        # Build electronic states
        molecule.electronic_states = []
        natoms = len(molecule.atoms)

        # Ground state (S0)
        gs_mol = data.molecule()
        gs_mol.atoms = [atom.copy() for atom in molecule.atoms]
        gs_mol.energy = gs_energy
        # Initialize gradients as NaN
        for atom in gs_mol.atoms:
            atom.energy_gradients = np.full(3, np.nan).tolist()
        molecule.electronic_states.append(gs_mol)

        # Parse excited states if TD-DFT was run
        if self.nstates > 1 and os.path.exists(out_file):
            excitation_energies, oscillator_strengths = self._parse_tddft_spectrum(out_file)

            for i, exc_e in enumerate(excitation_energies):
                state_mol = data.molecule()
                state_mol.atoms = [atom.copy() for atom in molecule.atoms]
                state_mol.energy = gs_energy + exc_e
                # Initialize gradients as NaN
                for atom in state_mol.atoms:
                    atom.energy_gradients = np.full(3, np.nan).tolist()
                molecule.electronic_states.append(state_mol)

            if oscillator_strengths:
                molecule.oscillator_strengths = oscillator_strengths

        # Set molecule energy to current state
        # For TD-DFT: current_state=0 means S0, current_state=1 means S1, etc.
        state_idx = self.current_state if self.current_state < len(molecule.electronic_states) else 0
        molecule.energy = molecule.electronic_states[state_idx].energy

        # Parse gradients for the requested gradient state
        if self.calculate_energy_gradients and self.gradient_state is not None:
            self._parse_gradients(molecule)
            # Store gradients in the appropriate electronic state using add_xyz_derivative_property
            grad_idx = self.gradient_state
            if grad_idx < len(molecule.electronic_states):
                # Get gradient array from molecule.atoms
                grad_array = np.array([atom.energy_gradients for atom in molecule.atoms])
                molecule.electronic_states[grad_idx].add_xyz_derivative_property(
                    grad_array, 'energy', 'energy_gradients'
                )

        # Parse Hessian
        if self.calculate_hessian:
            self._parse_hessian(molecule)

    def _parse_tddft_spectrum(self, out_file: str) -> tuple:
        '''Parse TD-DFT absorption spectrum from output file.'''
        with open(out_file, 'r') as f:
            lines = f.readlines()

        # Find absorption spectrum section
        last_idx = None
        for i, line in enumerate(lines):
            if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
                last_idx = i

        excitation_energies = []
        oscillator_strengths = []

        if last_idx is not None:
            for line in lines[last_idx + 5:]:
                if line.strip() == '':
                    break
                parts = line.split()
                if len(parts) >= 4:
                    # Column 1 is energy in cm^-1, column 3 is oscillator strength
                    if self._orca_version == 6:
                        exc_energy_cm = float(parts[4])
                    else:
                        exc_energy_cm = float(parts[1])
                    exc_energy_hartree = exc_energy_cm * 4.556335e-6
                    excitation_energies.append(exc_energy_hartree)
                    if self._orca_version == 6:
                        oscillator_strengths.append(float(parts[6]))
                    else:
                        oscillator_strengths.append(float(parts[3]))


        return excitation_energies, oscillator_strengths

    def _parse_gradients(self, molecule):
        '''Parse energy gradients from .engrad file.'''
        engrad_file = f'{self.inpfile}.engrad'
        if not os.path.exists(engrad_file):
            raise FileNotFoundError(f"Gradient file not found: {engrad_file}. ORCA may have failed to compute gradients.")

        with open(engrad_file, 'r') as f:
            lines = f.readlines()

        gradients = []
        in_gradient_section = False

        for i, line in enumerate(lines):
            if 'The current gradient in Eh/bohr' in line:
                in_gradient_section = True
                continue
            if in_gradient_section:
                line_stripped = line.strip()
                # Skip comment lines starting with #
                if line_stripped.startswith('#'):
                    # Check if this is the end marker (# followed by new section)
                    # Look ahead to see if next non-empty line is a new section header
                    if i + 1 < len(lines) and 'atomic' in lines[i + 1].lower():
                        break
                    continue
                if line_stripped:
                    try:
                        # Convert from Eh/bohr to Eh/Angstrom
                        gradients.append(float(line_stripped) * constants.Angstrom2Bohr)
                    except ValueError:
                        # Not a number, likely end of section
                        break

        if gradients:
            grad_array = np.array(gradients).reshape(-1, 3)
            for i, atom in enumerate(molecule.atoms):
                atom.energy_gradients = grad_array[i].tolist()
        else:
            raise ValueError(f"No gradients found in {engrad_file}")

    def _parse_hessian(self, molecule):
        '''Parse Hessian matrix from .hess file.'''
        hess_file = f'{self.inpfile}.hess'
        if not os.path.exists(hess_file):
            return

        with open(hess_file, 'r') as f:
            lines = f.readlines()

        # Find $hessian section
        start_line = None
        for i, line in enumerate(lines):
            if '$hessian' in line:
                start_line = i
                break

        if start_line is None:
            return

        matrix_size = int(lines[start_line + 1].strip())
        hessian = np.zeros((matrix_size, matrix_size))

        current_line = start_line + 2
        current_col = 0

        while current_col < matrix_size and current_line < len(lines):
            # Read column header
            header = lines[current_line].strip()
            if not header:
                current_line += 1
                continue

            cols = [int(x) for x in header.split()]
            num_cols = len(cols)
            current_line += 1

            # Read data rows
            for row in range(matrix_size):
                if current_line >= len(lines):
                    break
                data_line = lines[current_line].strip()
                if not data_line:
                    current_line += 1
                    continue

                parts = data_line.split()
                # First element is row index, rest are values
                for col_idx, col_num in enumerate(cols):
                    if col_idx + 1 < len(parts):
                        hessian[row, col_num] = float(parts[col_idx + 1])
                current_line += 1

            current_col += num_cols

        # Convert from Eh/bohr^2 to Eh/Angstrom^2
        molecule.hessian = hessian / (constants.Bohr2Angstrom ** 2)

        # Also parse frequencies if available
        self._parse_frequencies(molecule, lines)

    def _parse_frequencies(self, molecule, hess_lines: list):
        '''Parse vibrational frequencies from .hess file content.'''
        frequencies = []
        ir_intensities = []

        for i, line in enumerate(hess_lines):
            if '$vibrational_frequencies' in line:
                for freq_line in hess_lines[i + 2:]:
                    if freq_line.strip() == '' or '$' in freq_line:
                        break
                    parts = freq_line.split()
                    if len(parts) >= 2:
                        frequencies.append(float(parts[-1]))

            if '$ir_spectrum' in line:
                for ir_line in hess_lines[i + 2:]:
                    if ir_line.strip() == '' or '$' in ir_line:
                        break
                    parts = ir_line.split()
                    if len(parts) >= 3:
                        ir_intensities.append(float(parts[2]))

        # Remove zero frequencies (translations/rotations)
        if frequencies:
            zero_count = sum(1 for f in frequencies if f == 0)
            molecule.frequencies = frequencies[zero_count:]
            if ir_intensities:
                molecule.infrared_intensities = ir_intensities[zero_count:]



class ccsdtstarcbs(model):

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


def parse_orca_output(filename, molecule=None):
    """
    Parse ORCA output file and return/populate a molecule object.

    Automatically detects calculation type (QD-NEVPT2, TD-DFT, or ground state DFT).
    Also looks for .engrad and .hess files in the same directory.

    Args:
        filename: Path to ORCA output file (.out)
        molecule: Optional molecule object to update. If None, returns new molecule.


    Returns:
        molecule object if molecule is None, otherwise updates provided molecule

    Example:
        >>> mol = parse_orca_output('calculation.out')
        >>> print(mol.energy)
        >>> print(mol.electronic_states[1].energy)  # S1 energy
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ORCA output file not found: {filename}")

    with open(filename, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    # Build new molecule from scratch
    mol = data.molecule()

    # Parse geometry from output
    _parse_geometry_from_content(mol, lines)

    # Auto-detect calculation type
    is_qd_nevpt2 = 'QD-NEVPT2 Results' in content
    is_tddft = 'TD-DFT/TDA EXCITED STATES' in content or 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in content

    # Get base path for auxiliary files (.engrad, .hess)
    base_path = filename.rsplit('.', 1)[0]

    current_state = 0
    try:
        for line in lines:
            if 'State of interest' in line:
                current_state = int(line.split()[-1])
                break
    except:
        current_state = 0
    if is_qd_nevpt2:
        _parse_qd_nevpt2_from_content(mol, lines, current_state)
    elif is_tddft:
        _parse_tddft_from_content(mol, lines, current_state)
    else:
        _parse_ground_state_from_content(mol, lines)

    # Try to parse gradients if .engrad file exists
    engrad_file = f'{base_path}.engrad'
    if os.path.exists(engrad_file):
        # Default gradient_state to current_state if not specified
        _parse_gradients_from_file(mol, engrad_file, current_state)

    # Try to parse Hessian if .hess file exists
    hess_file = f'{base_path}.hess'
    if os.path.exists(hess_file):
        _parse_hessian_from_file(mol, hess_file)

    if molecule is None:
        return mol
    else:
        molecule.update_from(mol)


def _parse_geometry_from_content(mol, lines):
    """Parse molecular geometry from ORCA output."""
    # Look for "CARTESIAN COORDINATES (ANGSTROEM)" section
    in_coords = False
    atoms_data = []

    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            in_coords = True
            continue
        if in_coords:
            if line.strip() == '' or '---' in line:
                if atoms_data:
                    break
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    atoms_data.append((element, x, y, z))
                except ValueError:
                    continue

    # Build atoms
    for element, x, y, z in atoms_data:
        atom = data.atom(element_symbol=element)
        atom.xyz_coordinates = np.array([x, y, z])
        mol.atoms.append(atom)

    # Parse charge and multiplicity from input section
    for line in lines:
        if '* xyz' in line.lower() or '*xyz' in line.lower():
            parts = line.replace('*', '').split()
            if 'xyz' in parts[0].lower():
                parts = parts[1:]
            if len(parts) >= 2:
                try:
                    mol.charge = int(parts[0])
                    mol.multiplicity = int(parts[1])
                except ValueError:
                    pass
            break


def _parse_qd_nevpt2_from_content(mol, lines, current_state):
    """Parse QD-NEVPT2 energies from output content."""
    pt2_energies = {}
    casscf_energies = {}
    in_block = False
    current_root = None

    for line in lines:
        if 'QD-NEVPT2 Results' in line:
            in_block = True
            continue

        if in_block:
            line_stripped = line.strip()

            if line_stripped.startswith('ROOT ='):
                parts = line_stripped.split('=')
                if len(parts) >= 2:
                    try:
                        current_root = int(parts[1].strip())
                    except ValueError:
                        current_root = None

            elif 'Zero Order Energy' in line and current_root is not None:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        casscf_energies[current_root] = float(parts[-1].strip())
                    except ValueError:
                        pass

            elif 'Total Energy (E0+dE)' in line and current_root is not None:
                parts = line.split('=')
                if len(parts) >= 2:
                    try:
                        pt2_energies[current_root] = float(parts[-1].strip())
                    except ValueError:
                        pass

            elif '======' in line_stripped and pt2_energies:
                break

    if not pt2_energies:
        raise ValueError("No QD-NEVPT2 Results block found in output")

    # Build electronic states
    mol.electronic_states = []
    for root_idx in sorted(pt2_energies.keys()):
        state_mol = data.molecule()
        state_mol.atoms = [atom.copy() for atom in mol.atoms]
        state_mol.energy = pt2_energies[root_idx]
        if root_idx in casscf_energies:
            state_mol.casscf_energy = casscf_energies[root_idx]
        # Initialize gradients as NaN
        for atom in state_mol.atoms:
            atom.energy_gradients = np.full(3, np.nan).tolist()
        mol.electronic_states.append(state_mol)

    # Set molecule energy to current state
    if current_state < len(mol.electronic_states):
        mol.energy = mol.electronic_states[current_state].energy

    # Parse oscillator strengths
    _parse_absorption_spectrum(mol, lines, is_qd_nevpt2=True)


def _parse_tddft_from_content(mol, lines,  current_state):
    """Parse TD-DFT energies from output content."""
    # Get ground state energy
    gs_energy = None
    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            gs_energy = float(line.split()[-1])
            break

    if gs_energy is None:
        raise ValueError("Could not parse ground state energy")

    # Build electronic states
    mol.electronic_states = []

    # Ground state (S0)
    gs_mol = data.molecule()
    gs_mol.atoms = [atom.copy() for atom in mol.atoms]
    gs_mol.energy = gs_energy
    for atom in gs_mol.atoms:
        atom.energy_gradients = np.full(3, np.nan).tolist()
    mol.electronic_states.append(gs_mol)

    # Parse excited states
    excitation_energies, oscillator_strengths = _parse_tddft_spectrum_from_lines(lines)

    for exc_e in excitation_energies:
        state_mol = data.molecule()
        state_mol.atoms = [atom.copy() for atom in mol.atoms]
        state_mol.energy = gs_energy + exc_e
        for atom in state_mol.atoms:
            atom.energy_gradients = np.full(3, np.nan).tolist()
        mol.electronic_states.append(state_mol)

    if oscillator_strengths:
        mol.oscillator_strengths = oscillator_strengths

    # Set molecule energy to current state
    state_idx = current_state if current_state < len(mol.electronic_states) else 0
    mol.energy = mol.electronic_states[state_idx].energy


def _parse_ground_state_from_content(mol, lines):
    """Parse ground state DFT/HF energy from output content."""
    gs_energy = None
    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            gs_energy = float(line.split()[-1])
            break

    if gs_energy is None:
        raise ValueError("Could not parse ground state energy")

    mol.energy = gs_energy

    # Create single electronic state
    mol.electronic_states = []
    gs_mol = data.molecule()
    gs_mol.atoms = [atom.copy() for atom in mol.atoms]
    gs_mol.energy = gs_energy
    for atom in gs_mol.atoms:
        atom.energy_gradients = np.full(3, np.nan).tolist()
    mol.electronic_states.append(gs_mol)


def _parse_tddft_spectrum_from_lines(lines):
    """Parse TD-DFT absorption spectrum from output lines."""
    last_idx = None
    for i, line in enumerate(lines):
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
            last_idx = i

    excitation_energies = []
    oscillator_strengths = []

    if last_idx is not None:
        for line in lines[last_idx + 5:]:
            if line.strip() == '':
                break
            parts = line.split()
            if len(parts) >= 4:
                exc_energy_cm = float(parts[1])
                exc_energy_hartree = exc_energy_cm * 4.556335e-6
                excitation_energies.append(exc_energy_hartree)
                oscillator_strengths.append(float(parts[3]))

    return excitation_energies, oscillator_strengths


def _parse_absorption_spectrum(mol, lines, is_qd_nevpt2=False):
    """Parse oscillator strengths from absorption spectrum."""
    last_idx = None
    for i, line in enumerate(lines):
        if 'ABSORPTION SPECTRUM' in line:
            last_idx = i

    if last_idx is None:
        return

    oscillator_strengths = []

    for line in lines[last_idx + 5:]:
        if '-' * 20 in line or line.strip() == '':
            break
        parts = line.split()
        if is_qd_nevpt2 and len(parts) >= 8:
            oscillator_strengths.append(float(parts[7]))
        elif not is_qd_nevpt2 and len(parts) >= 4:
            oscillator_strengths.append(float(parts[3]))

    if oscillator_strengths:
        mol.oscillator_strengths = oscillator_strengths


def _parse_gradients_from_file(mol, engrad_file, current_state=0):
    """Parse gradients from .engrad file and add to electronic state."""
    with open(engrad_file, 'r') as f:
        lines = f.readlines()

    gradients = []
    in_gradient_section = False

    for i, line in enumerate(lines):
        if 'The current gradient in Eh/bohr' in line:
            in_gradient_section = True
            continue
        if in_gradient_section:
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                if i + 1 < len(lines) and 'atomic' in lines[i + 1].lower():
                    break
                continue
            if line_stripped:
                try:
                    gradients.append(float(line_stripped) * constants.Angstrom2Bohr)
                except ValueError:
                    break

    if gradients and mol.atoms:
        grad_array = np.array(gradients).reshape(-1, 3)
        # Add to top-level molecule atoms
        for i, atom in enumerate(mol.atoms):
            if i < len(grad_array):
                atom.energy_gradients = grad_array[i].tolist()
        # Add to the appropriate electronic state using add_xyz_derivative_property
        if hasattr(mol, 'electronic_states') and current_state < len(mol.electronic_states):
            mol.electronic_states[current_state].add_xyz_derivative_property(
                grad_array, 'energy', 'energy_gradients'
            )


def _parse_hessian_from_file(mol, hess_file):
    """Parse Hessian matrix from .hess file."""
    with open(hess_file, 'r') as f:
        lines = f.readlines()

    # Find $hessian section
    start_line = None
    for i, line in enumerate(lines):
        if '$hessian' in line:
            start_line = i
            break

    if start_line is None:
        return

    matrix_size = int(lines[start_line + 1].strip())
    hessian = np.zeros((matrix_size, matrix_size))

    current_line = start_line + 2
    current_col = 0

    while current_col < matrix_size and current_line < len(lines):
        header = lines[current_line].strip()
        if not header:
            current_line += 1
            continue

        cols = [int(x) for x in header.split()]
        num_cols = len(cols)
        current_line += 1

        for row in range(matrix_size):
            if current_line >= len(lines):
                break
            data_line = lines[current_line].strip()
            if not data_line:
                current_line += 1
                continue

            parts = data_line.split()
            for col_idx, col_num in enumerate(cols):
                if col_idx + 1 < len(parts):
                    hessian[row, col_num] = float(parts[col_idx + 1])
            current_line += 1

        current_col += num_cols

    mol.hessian = hessian / (constants.Bohr2Angstrom ** 2)

    # Parse frequencies
    _parse_frequencies_from_lines(mol, lines)


def _parse_frequencies_from_lines(mol, hess_lines):
    """Parse vibrational frequencies from .hess file content."""
    frequencies = []
    ir_intensities = []

    for i, line in enumerate(hess_lines):
        if '$vibrational_frequencies' in line:
            for freq_line in hess_lines[i + 2:]:
                if freq_line.strip() == '' or '$' in freq_line:
                    break
                parts = freq_line.split()
                if len(parts) >= 2:
                    frequencies.append(float(parts[-1]))

        if '$ir_spectrum' in line:
            for ir_line in hess_lines[i + 2:]:
                if ir_line.strip() == '' or '$' in ir_line:
                    break
                parts = ir_line.split()
                if len(parts) >= 3:
                    ir_intensities.append(float(parts[2]))

    if frequencies:
        zero_count = sum(1 for f in frequencies if f == 0)
        mol.frequencies = frequencies[zero_count:]
        if ir_intensities:
            mol.infrared_intensities = ir_intensities[zero_count:]
