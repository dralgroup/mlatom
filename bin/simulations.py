#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! simulations: Module for simulations                                       ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  ! To-do:                                                                    ! 
  !   * cite TorchANI when using freq_modified_from_TorchANI                  !
  !   * implement Gaussian keywords                                            !
  !   * successful, max_number_of_steps, convergence criteria in geomopt              !
  !   * implement temperature in freq                                         !
  !   * nthreads                                                              !
  !---------------------------------------------------------------------------! 
'''

pythonpackage = True
try:
    from . import constants, data, stopper
    from .md import md as md
    from .initial_conditions import generate_initial_conditions
except:
    import constants, data, stopper
    from md import md as md
    from initial_conditions import generate_initial_conditions
    pythonpackage = False
import os, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class optimize_geometry():
    def __init__(self, model=None, method=None, initial_molecule=None, ts=False, program=None, optimization_algorithm=None, maximum_number_of_steps=None, convergence_criterion_for_forces=None):
        if model != None:
            self.model = model
        elif method != None:
            self.model = method
        self.initial_molecule = initial_molecule
        self.ts = ts
        if program != None:
            self.program = program
        else:
            if "GAUSS_EXEDIR" in os.environ: self.program = 'Gaussian'
            else:
                try:
                    import ase
                    self.program = 'ASE'
                except:
                    try: import scipy.optimize
                    except:
                        if pythonpackage: raise ValueError('please set $GAUSS_EXEDIR or install ase or install scipy')
                        else: stopper.stopMLatom('please set $GAUSS_EXEDIR or install ase or install scipy')
        
        self.optimization_algorithm = optimization_algorithm
        
        # START of block with parameters which are not used in scipy & Gaussian but only in ASE optimization
        if maximum_number_of_steps != None: self.maximum_number_of_steps = maximum_number_of_steps
        else: self.maximum_number_of_steps = 200
        
        if convergence_criterion_for_forces != None: self.convergence_criterion_for_forces = convergence_criterion_for_forces
        else: self.convergence_criterion_for_forces = 0.02 # Forces convergence criterion in ASE: 0.02 eV/A
        # END   of block with parameters which are not used in scipy & Gaussian but only in ASE optimization
        
        if self.ts and self.program.casefold() != 'Gaussian'.casefold():
            msg = 'Transition state geometry optmization can currently only be done with optimization_program=Gaussian'
            if pythonpackage: raise ValueError(msg)
            else: stopper.stopMLatom(msg)
        
        if self.program.casefold() == 'Gaussian'.casefold(): self.opt_geom_gaussian()
        elif self.program.casefold() == 'ASE'.casefold(): self.opt_geom_ase()
        else: self.opt_geom()
        
    def opt_geom_gaussian(self):
        self.successful = False
        try: from .interfaces import gaussian
        except: import interfaces.gaussian as gaussian
        if 'number' in self.initial_molecule.__dict__.keys(): suffix = f'_{self.initial_molecule.number}'
        else: suffix = ''
        filename = f'gaussian{suffix}'
        self.model.dump(filename='model.json', format='json')
        self.optimization_trajectory = data.molecular_trajectory()
        self.optimization_trajectory.dump(filename='gaussian_opttraj.json', format='json')
        
        # Run Gaussian
        external_task='opt'
        if self.ts: external_task = 'ts'
        gaussian.run_gaussian_job(filename=f'{filename}.com', molecule=self.initial_molecule, external_task=external_task)
        
        # Get results
        outputfile = f'{filename}.log'
        if not os.path.exists(outputfile): outputfile = f'{filename}.out'
        with open(outputfile, 'r') as fout:
            for line in fout:
                if '!   Optimized Parameters   !' in line:
                    self.successful = True
                    break
        self.optimization_trajectory.load(filename='gaussian_opttraj.json', format='json')
        if self.successful: self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        if os.path.exists('gaussian_opttraj.json'): os.remove('gaussian_opttraj.json')
        
    def opt_geom_ase(self):
        try:
            from .interfaces import ase_interface
        except:
            import interfaces.ase_interface as ase_interface
        self.optimization_trajectory = ase_interface.optimize_geometry(initial_molecule=self.initial_molecule,
                                        model=self.model,
                                        convergence_criterion_for_forces=self.convergence_criterion_for_forces,
                                        maximum_number_of_steps=self.maximum_number_of_steps,
                                        optimization_algorithm=self.optimization_algorithm)
        self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        
    def opt_geom(self):
        try: import scipy.optimize
        except:
            if pythonpackage: raise ValueError('scipy is not installed')
            else: stopper.stopMLatom('scipy is not installed')
        istep = -1
        self.optimization_trajectory = data.molecular_trajectory()
        #self.optimization_trajectory.steps.append(data.molecular_trajectory_step(step=istep, molecule=self.initial_molecule))
        
        def molecular_energy(coordinates):
            nonlocal istep
            istep += 1
            current_molecule = self.initial_molecule.copy()
            current_molecule.update_xyz_coordinates(xyz_coordinates = coordinates.reshape(len(current_molecule.atoms),3))
            self.model.predict(molecule=current_molecule, calculate_energy=True, calculate_energy_gradients=True)
            if not 'energy' in current_molecule.__dict__:
                if pythonpackage: raise ValueError('model did not return any energy')
                else: stopper.stopMLatom('model did not return any energy')
            molecular_energy = current_molecule.energy
            gradient = current_molecule.get_energy_gradients()
            gradient = gradient.flatten()
            self.optimization_trajectory.steps.append(data.molecular_trajectory_step(step=istep, molecule=current_molecule))
            return molecular_energy, gradient
    
        initial_coordinates = self.initial_molecule.get_xyz_coordinates().flatten()
        res = scipy.optimize.minimize(molecular_energy, initial_coordinates, method=self.optimization_algorithm, jac=True)
        optimized_coordinates = res.x
        molecular_energy(optimized_coordinates)
        self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule

class irc():
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model = kwargs['model']
        elif 'method' in kwargs:
            self.model = kwargs['method']
        if 'ts_molecule' in kwargs:
            self.ts_molecule = kwargs['ts_molecule']

        try: from .interfaces import gaussian
        except: import interfaces.gaussian as gaussian
        if 'number' in self.ts_molecule.__dict__.keys(): suffix = f'_{self.ts_molecule.number}'
        else: suffix = ''
        filename = f'gaussian{suffix}'
        self.model.dump(filename='model.json', format='json')
        
        # Run Gaussian
        gaussian.run_gaussian_job(filename=f'{filename}.com', molecule=self.ts_molecule, external_task='irc')
        
        if os.path.exists('model.json'): os.remove('model.json')

class freq():
    def __init__(self, model=None, method=None, molecule=None, program=None, normal_mode_normalization='mass deweighted unnormalized'):
        if model != None:
            self.model = model
        elif method != None:
            self.model = method
        self.molecule = molecule
        if program != None:
            self.program = program
        else:
            if "GAUSS_EXEDIR" in os.environ: self.program = 'Gaussian'
        self.normal_mode_normalization = normal_mode_normalization
    
        if self.program.casefold() == 'Gaussian'.casefold(): self.freq_gaussian()
        else:
            if not 'shape' in self.molecule.__dict__:
                self.molecule.shape = 'nonlinear'
            self.freq_modified_from_TorchANI()
        
    def freq_gaussian(self):
        self.successful = False
        try: from .interfaces import gaussian
        except: import interfaces.gaussian as gaussian
        if 'number' in self.molecule.__dict__.keys(): suffix = f'_{self.molecule.number}'
        else: suffix = ''
        filename = f'gaussian{suffix}'
        self.model.dump(filename='model.json', format='json')
        self.optimization_trajectory = data.molecular_trajectory()
        self.molecule.dump(filename='gaussian_freq_mol.json', format='json')
        
        # Run Gaussian
        gaussian.run_gaussian_job(filename=f'{filename}.com', molecule=self.molecule, external_task='freq')
        
        # Get results
        outputfile = f'{filename}.log'
        if not os.path.exists(outputfile): outputfile = f'{filename}.out'
        self.successful = gaussian.read_freq_thermochemistry_from_Gaussian_output(outputfile, self.molecule)
        if os.path.exists('model.json'): os.remove('model.json')
        if os.path.exists('gaussian_freq_mol.json'): os.remove('gaussian_freq_mol.json')
    
    def freq_modified_from_TorchANI(self):
        # this function is modified from TorchANI by Peikun Zheng
        # cite TorchANI when using it
        """Computing the vibrational wavenumbers from hessian.

        Note that normal modes in many popular software packages such as
        Gaussian and ORCA are output as mass deweighted normalized (MDN).
        Normal modes in ASE are output as mass deweighted unnormalized (MDU).
        Some packages such as Psi4 let ychoose different normalizations.
        Force constants and reduced masses are calculated as in Gaussian.

        mode_type should be one of:
        - MWN (mass weighted normalized)
        - MDU (mass deweighted unnormalized)
        - MDN (mass deweighted normalized)

        MDU modes are not orthogonal, and not normalized,
        MDN modes are not orthogonal, and normalized.
        MWN modes are orthonormal, but they correspond
        to mass weighted cartesian coordinates (x' = sqrt(m)x).
        """
        # Calculate hessian
        self.model.predict(molecule=self.molecule, calculate_hessian=True)
        
        mhessian2fconst = 4.359744650780506
        unit_converter = 17091.7006789297
        # Solving the eigenvalue problem: Hq = w^2 * T q
        # where H is the hessian matrix, q is the normal coordinates,
        # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
        # We solve this eigenvalue problem through Lowdin diagnolization:
        # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
        # Letting q' = T^(1/2) q, we then have
        # T^(-1/2) H T^(-1/2) q' = w^2 * q'
        masses = np.expand_dims(self.molecule.get_nuclear_masses(), axis=0)
        inv_sqrt_mass = np.repeat(np.sqrt(1 / masses), 3, axis=1) # shape (3 * atoms)
        mass_scaled_hessian = self.molecule.hessian * np.expand_dims(inv_sqrt_mass, axis=1) * np.expand_dims(inv_sqrt_mass, axis=2)
        mass_scaled_hessian = np.squeeze(mass_scaled_hessian, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(mass_scaled_hessian)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        angular_frequencies = np.sqrt(eigenvalues)
        frequencies = angular_frequencies / (2 * math.pi)
        # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 
        wavenumbers = unit_converter * frequencies

        # Note that the normal modes are the COLUMNS of the eigenvectors matrix
        mw_normalized = eigenvectors.T
        md_unnormalized = mw_normalized * inv_sqrt_mass
        norm_factors = 1 / np.linalg.norm(md_unnormalized, axis=1)  # units are sqrt(AMU)
        md_normalized = md_unnormalized * np.expand_dims(norm_factors, axis=1)

        rmasses = norm_factors**2  # units are AMU
        # converting from Ha/(AMU*A^2) to mDyne/(A*AMU) 
        fconstants = mhessian2fconst * eigenvalues * rmasses  # units are mDyne/A

        if self.normal_mode_normalization == 'mass deweighted normalized':
            modes = (md_normalized).reshape(frequencies.size, -1, 3)
        elif self.normal_mode_normalization == 'mass deweighted unnormalized':
            modes = (md_unnormalized).reshape(frequencies.size, -1, 3)
        elif self.normal_mode_normalization == 'mass weighted normalized':
            modes = (mw_normalized).reshape(frequencies.size, -1, 3)

        # the first 6 (5 for linear) entries are for rotation and translation
        # we skip them because we are only interested in vibrational modes
        nskip = 6
        if self.molecule.shape.lower() == 'linear':
            nskip = 5
        self.molecule.frequencies = wavenumbers[nskip:]    # in cm^-1
        self.molecule.force_constants = fconstants[nskip:] # in mDyne/A
        self.molecule.reduced_masses = rmasses[nskip:]     # in AMU
        for iatom in range(len(self.molecule.atoms)):
            self.molecule.atoms[iatom].normal_modes = []
            for imode in range(len(modes)):
                if imode < nskip: continue
                self.molecule.atoms[iatom].normal_modes.append(list(modes[imode][iatom]))
            self.molecule.atoms[iatom].normal_modes = np.array(self.molecule.atoms[iatom].normal_modes)
            
class thermochemistry():
    def __init__(self, model=None, method=None, molecule=None, program=None, normal_mode_normalization='mass deweighted unnormalized'):
        if model != None:
            self.model = model
        elif method != None:
            self.model = method
        self.molecule = molecule
        if program != None:
            self.program = program
        else:
            if "GAUSS_EXEDIR" in os.environ: self.program = 'Gaussian'
            else:
                try:
                    import ase
                    self.program = 'ASE'
                except:
                    if pythonpackage: raise ValueError('please set $GAUSS_EXEDIR or install ase')
                    else: stopper.stopMLatom('please set $GAUSS_EXEDIR or install ase')
        
        freq(model=model, method=method, molecule=molecule, program=program, normal_mode_normalization=normal_mode_normalization)
        if self.program.casefold() == 'ASE'.casefold(): self.thermochem_ase()
        # Calculate heats of formation
        self.calculate_heats_of_formation()
    
    def thermochem_ase(self):
        try:
            from .interfaces import ase_interface
        except:
            import interfaces.ase_interface as ase_interface
        ase_interface.thermochemistry(molecule=self.molecule)
        
    def calculate_heats_of_formation(self):
        if 'H0' in self.molecule.__dict__:
            atoms_have_H0 = True
            for atom in self.molecule.atoms:
                if not 'H0' in atom.__dict__:
                    atoms_have_H0 = False
                    break
            if not atoms_have_H0:
                return
        
        DeltaH_atom = 1.4811 * constants.kcalpermol2Hartree
        sum_E_atom = 0.0
        sum_H0_atom = 0.0
        sum_DeltaH_atom = 0.0
        try:
            for atom in self.molecule.atoms:
                atomic_molecule = data.molecule(multiplicity=atom.multiplicity, atoms=[atom])
                self.model.predict(molecule=atomic_molecule)
                sum_E_atom += atomic_molecule.energy
                sum_H0_atom += atom.H0
                sum_DeltaH_atom += DeltaH_atom
        except:
            return
        
        atomization_energy = sum_E_atom - self.molecule.U0
        DeltaHf298 = sum_H0_atom - atomization_energy + (self.molecule.H - self.molecule.H0) - sum_DeltaH_atom
        self.molecule.atomization_energy_0K = atomization_energy
        self.molecule.ZPE_exclusive_atomization_energy_0K = atomization_energy + self.molecule.ZPE
        self.molecule.DeltaHf298 = DeltaHf298
        
def numerical_grad(molecule, model_with_function_to_predict_energy, eps, kwargs_funtion_predict_energy = {}):
    from scipy.optimize import approx_fprime
    def get_energy(coordinates):
        current_molecule = molecule.copy()
        current_molecule.update_xyz_coordinates(xyz_coordinates = coordinates.reshape(len(current_molecule.atoms),3))
        model_with_function_to_predict_energy.predict(molecule=current_molecule, **kwargs_funtion_predict_energy)
        return current_molecule.energy
    coordinates = molecule.get_xyz_coordinates().reshape(-1)
    gradients = approx_fprime(coordinates, get_energy, eps)
    gradients = gradients.reshape(-1, 3)
    return gradients

def numerical_hessian(molecule, model_with_function_to_predict_energy, eps, epsgrad, kwargs_funtion_predict_energy = {}):
    g1 = numerical_grad(molecule, model_with_function_to_predict_energy, epsgrad, kwargs_funtion_predict_energy)
    coordinates1 = molecule.get_xyz_coordinates().reshape(-1)
    ndim = len(coordinates1)
    hess = np.zeros((ndim, ndim))
    coordinates2 = coordinates1
    for i in range(ndim):
        x0 = coordinates2[i]
        coordinates2[i] = coordinates1[i] + eps
        molecule2 = molecule.copy()
        molecule2.update_xyz_coordinates(xyz_coordinates = coordinates2.reshape(len(molecule2.atoms),3))
        g2 = numerical_grad(molecule2, model_with_function_to_predict_energy, epsgrad, kwargs_funtion_predict_energy)
        hess[:, i] = (g2.reshape(-1) - g1.reshape(-1)) / eps
        coordinates2[i] = x0

    return hess
        