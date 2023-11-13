#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! simulations: Module for simulations                                       ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''
pythonpackage = True
from . import constants, data, models, stopper
from .md import md as md
from .initial_conditions import generate_initial_conditions
from .environment_variables import env
from .md2vibr import vibrational_spectrum
import os, math, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def run_in_parallel(molecular_database=None, task=None, task_kwargs={}, nthreads=None, create_and_keep_temp_directories=False):
    import joblib
    from joblib import Parallel, delayed
    def task_loc(imol):
        mol = molecular_database[imol]
        if create_and_keep_temp_directories:
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            directory = time.strftime(f'job_{task}_{imol}_mol{mol.id}_{current_time}')
            os.chdir(directory)
        result = task(molecule=mol, **task_kwargs)
        return result
    if nthreads == None: nthreads = joblib.cpu_count()
    results = Parallel(n_jobs=nthreads)(delayed(task_loc)(i) for i in range(len(molecular_database)))
    return results

class optimize_geometry():
    """
    Geometry optimization.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces.
        initial_molecule (:class:`mlatom.data.molecule`): The molecule object to relax.
        ts (bool, optional): Whether to do transition state search. Currently only be done with program=Gaussian or ASE.
        program (str, optional): The engine used in geomtry optimization. Currently support Gaussian, ASE, and scipy.
        optimization_algorithm (str, optional): The optimization algorithm used in ASE. Default value: LBFGS (ts=False), dimer (ts=False).
        maximum_number_of_steps (int, optional): The maximum steps. Default value: 200.
        convergence_criterion_for_forces (float, optional): Forces convergence criterion in ASE. Default value: 0.02 eV/Angstroms.
        working_directory (str, optional): Working directory. Default value: '.', i.e., current directory.

    Examples:

    .. code-block:: python


        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Initialize methods
        aiqm1 = ml.models.methods(method='AIQM1', qm_program='MNDO')
        # Run geometry optimization
        geomopt = ml.simulations.optimize_geometry(model = aiqm1, initial_molecule=mol, program = 'ASE')
        # Get the optimized geometry, energy, and gradient
        optmol = geomopt.optimized_molecule
        geo = optmol.get_xyz_coordinates()
        energy = optmol.energy
        gradient = optmol.get_energy_gradients()


    """

    def __init__(self, model=None, initial_molecule=None, molecule=None, ts=False, program=None, optimization_algorithm=None, maximum_number_of_steps=None, convergence_criterion_for_forces=None,working_directory=None):
        if model != None:
            self.model = model

        if not initial_molecule is None and not molecule is None:
            stopper.stopMLatom('molecule and initial_molecule cannot be used at the same time')
        overwrite = False
        if not initial_molecule is None:
            self.initial_molecule = initial_molecule.copy(atomic_labels=['xyz_coordinates','number'],molecular_labels=[])
        if not molecule is None:
            overwrite = True
            self.initial_molecule = molecule.copy(atomic_labels=['xyz_coordinates','number'],molecular_labels=[])
        
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

        if working_directory != None:
            self.working_directory = working_directory
        else:
            self.working_directory = '.'
        
        if self.ts and self.program.casefold() not in ['Gaussian'.casefold(), 'ASE'.casefold()]:
            msg = 'Transition state geometry optmization can currently only be done with optimization_program=Gaussian or ASE'
            if pythonpackage: raise ValueError(msg)
            else: stopper.stopMLatom(msg)
        
        if self.program.casefold() == 'Gaussian'.casefold(): self.opt_geom_gaussian()
        elif self.program.casefold() == 'ASE'.casefold(): self.opt_geom_ase()
        else: self.opt_geom()

        if overwrite:
            molecule.optimization_trajectory = self.optimization_trajectory
            for each in self.optimized_molecule.__dict__:
                molecule.__dict__[each] = self.optimized_molecule.__dict__[each]
            del self.optimization_trajectory
            del self.optimized_molecule
        
    def opt_geom_gaussian(self):
        self.successful = False
        from .interfaces import gaussian_interface
        if 'number' in self.initial_molecule.__dict__.keys(): suffix = f'_{self.initial_molecule.number}'
        else: suffix = ''
        filename = os.path.join(self.working_directory,f'gaussian{suffix}')
        self.model.dump(filename=os.path.join(self.working_directory,'model.json'), format='json')
        self.optimization_trajectory = data.molecular_trajectory()
        self.optimization_trajectory.dump(filename=os.path.join(self.working_directory,'gaussian_opttraj.json'), format='json')
        
        # Run Gaussian
        external_task='opt'
        if self.ts: external_task = 'ts'
        gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.initial_molecule, external_task=external_task, cwd=self.working_directory)
        
        # Get results
        outputfile = f'{filename}.log'
        if not os.path.exists(outputfile): outputfile = f'{filename}.out'
        with open(outputfile, 'r') as fout:
            for line in fout:
                if '!   Optimized Parameters   !' in line:
                    self.successful = True
                    break
        self.optimization_trajectory.load(filename=os.path.join(self.working_directory,'gaussian_opttraj.json'), format='json')
        if self.successful: 
            self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        else:
            self.optimized_molecule = self.initial_molecule.copy() 
            for atom in self.optimized_molecule.atoms:
                atom.xyz_coordinates = np.array([None,None,None])
        if os.path.exists(os.path.join(self.working_directory,'gaussian_opttraj.json')): os.remove(os.path.join(self.working_directory,'gaussian_opttraj.json'))
        
    def opt_geom_ase(self):
        from .interfaces import ase_interface
        if self.ts:
            self.optimization_trajectory = ase_interface.transition_state(initial_molecule=self.initial_molecule,
                                            model=self.model,
                                            convergence_criterion_for_forces=self.convergence_criterion_for_forces,
                                            maximum_number_of_steps=self.maximum_number_of_steps,
                                            optimization_algorithm='dimer')
        else:
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
            current_molecule.xyz_coordinates = coordinates.reshape(len(current_molecule.atoms),3)
            self.model.predict(molecule=current_molecule, calculate_energy=True, calculate_energy_gradients=True)
            if not 'energy' in current_molecule.__dict__:
                if pythonpackage: raise ValueError('model did not return any energy')
                else: stopper.stopMLatom('model did not return any energy')
            molecular_energy = current_molecule.energy
            gradient = current_molecule.get_energy_gradients()
            gradient = gradient.flatten()
            self.optimization_trajectory.steps.append(data.molecular_trajectory_step(step=istep, molecule=current_molecule))
            return molecular_energy, gradient
    
        initial_coordinates = self.initial_molecule.xyz_coordinates.flatten()
        res = scipy.optimize.minimize(molecular_energy, initial_coordinates, method=self.optimization_algorithm, jac=True)
        optimized_coordinates = res.x
        molecular_energy(optimized_coordinates)
        self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule

class irc():
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model = kwargs['model']
        if 'ts_molecule' in kwargs:
            self.ts_molecule = kwargs['ts_molecule'].copy(atomic_labels=['xyz_coordinates','number'],molecular_labels=[])

        from .interfaces import gaussian_interface
        if 'number' in self.ts_molecule.__dict__.keys(): suffix = f'_{self.ts_molecule.number}'
        else: suffix = ''
        filename = f'gaussian{suffix}'
        self.model.dump(filename='model.json', format='json')
        
        # Run Gaussian
        gaussian_interface.run_gaussian_job(filename=f'{filename}.com', molecule=self.ts_molecule, external_task='irc')
        
        #if os.path.exists('model.json'): os.remove('model.json')

class freq():
    """
    Frequence analysis.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces and Hessian.
        molecule (:class:`mlatom.data.molecule`): The molecule object with necessary information.
        program (str, optional): The engine used in frequence analysis through modified TorchANI (if Gaussian not found or any other string is given), pyscf or Gaussian interfaces.
        normal_mode_normalization (str, optional): Normal modes output scheme. It should be one of: mass weighted normalized, mass deweighted unnormalized, and mass deweighted unnormalized (default). 
        anharmonic (bool): Whether to do anharmonic frequence calculation.
        working_directory (str, optional): Working directory. Default value: '.', i.e., current directory.

    Examples:

    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Initialize methods
        aiqm1 = ml.models.methods(method='AIQM1', qm_program='MNDO')
        # Run frequence analysis
        ml.simulations.freq(model=aiqm1, molecule=mol, program='ASE')
        # Get frequencies
        frequencies = mol.frequencies


    """
    def __init__(self, model=None, molecule=None, program=None, normal_mode_normalization='mass deweighted unnormalized', anharmonic=False, anharmonic_kwargs={}, working_directory=None):
        if model != None:
            self.model = model
        self.molecule = molecule
        if program != None:
            self.program = program
        else:
            if "GAUSS_EXEDIR" in os.environ: 
                self.program = 'Gaussian'
            else:
                self.program = ''
        self.normal_mode_normalization = normal_mode_normalization
        if working_directory != None:
            self.working_directory = working_directory
        else:
            self.working_directory = '.'
        if self.model.program != None:
            if self.model.program.casefold() == 'pyscf'.casefold(): self.freq_pyscf()
            else: self.freq_gaussian(anharmonic)
        elif self.program.casefold() == 'Gaussian'.casefold(): self.freq_gaussian(anharmonic)
        else:
            if not 'shape' in self.molecule.__dict__:
                self.molecule.shape = 'nonlinear'
            self.freq_modified_from_TorchANI()
        
    def freq_gaussian(self, anharmonic):
        self.successful = False
        from .interfaces import gaussian_interface
        if 'number' in self.molecule.__dict__.keys(): suffix = f'_{self.molecule.number}'
        else: suffix = ''
        filename = os.path.join(self.working_directory,f'gaussian{suffix}')
        self.model.dump(filename=os.path.join(self.working_directory,'model.json'), format='json')
        self.optimization_trajectory = data.molecular_trajectory()
        self.molecule.dump(filename=os.path.join(self.working_directory,'gaussian_freq_mol.json'), format='json')
        
        # Run Gaussian
        if anharmonic:
            gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.molecule, external_task='freq(anharmonic)',cwd=self.working_directory)
        else:
            gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.molecule, external_task='freq',cwd=self.working_directory)
        
        # Get results
        outputfile = f'{filename}.log'
        if not os.path.exists(outputfile): outputfile = f'{filename}.out'
        self.successful = gaussian_interface.read_freq_thermochemistry_from_Gaussian_output(outputfile, self.molecule)
        if anharmonic:
            freq_len = len(self.molecule.frequencies)//2
            self.molecule.frequencies = self.molecule.frequencies[:freq_len]
            self.molecule.force_constants = self.molecule.force_constants[:freq_len]
            self.molecule.reduced_masses = self.molecule.reduced_masses[:freq_len]
            for iatom in range(len(self.molecule.atoms)):
                self.molecule.atoms[iatom].normal_modes = self.molecule.atoms[iatom].normal_modes[:freq_len]
            self.molecule.harmonic_frequencies = np.copy(self.molecule.frequencies)
            gaussian_interface.read_anharmonic_frequencies(outputfile,self.molecule)
            self.molecule.frequencies = self.molecule.anharmonic_frequencies             
            thermochemistry_properties = ['ZPE','DeltaE2U','DeltaE2H','DeltaE2G','U0','H0','U','H','G','S']
            for each_property in thermochemistry_properties:
                self.molecule.__dict__['harmonic_'+each_property] = self.molecule.__dict__[each_property]
                self.molecule.__dict__[each_property] = self.molecule.__dict__['anharmonic_'+each_property]
        #if os.path.exists('model.json'): os.remove('model.json')
        if os.path.exists(os.path.join(self.working_directory,'gaussian_freq_mol.json')): os.remove(os.path.join(self.working_directory,'gaussian_freq_mol.json'))

    def freq_pyscf(self):
        self.successful = False
        self.successful = self.model.interface.thermo_calculation(molecule=self.molecule)
    
    def freq_modified_from_TorchANI(self):
        # Copyright 2018- Xiang Gao and other ANI developers
        # 
        # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
        # 
        # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
        # 
        # the function freq_modified_from_TorchANI is modified from TorchANI by Peikun Zheng
        # cite TorchANI when using it (X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, A. E. Roitberg,\nJ. Chem. Inf. Model. 2020, 60, 3408)
        #
        # """Computing the vibrational wavenumbers from hessian.

        # Note that normal modes in many popular software packages such as
        # Gaussian and ORCA are output as mass deweighted normalized (MDN).
        # Normal modes in ASE are output as mass deweighted unnormalized (MDU).
        # Some packages such as Psi4 let ychoose different normalizations.
        # Force constants and reduced masses are calculated as in gaussian_interface.

        # mode_type should be one of:
        # - MWN (mass weighted normalized)
        # - MDU (mass deweighted unnormalized)
        # - MDN (mass deweighted normalized)

        # MDU modes are not orthogonal, and not normalized,
        # MDN modes are not orthogonal, and normalized.
        # MWN modes are orthonormal, but they correspond
        # to mass weighted cartesian coordinates (x' = sqrt(m)x).
        # """
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
    """
    Thermochemical properties calculation.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces and Hessian.
        molecule (:class:`mlatom.data.molecule`): The molecule object with necessary information.
        program (str): The engine used in thermochemical properties calculation. Currently support Gaussian and ASE.
        normal_mode_normalization (str, optional): Normal modes output scheme. It should be one of: mass weighted normalized, mass deweighted unnormalized, and mass deweighted unnormalized (default). 
    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Initialize methods
        aiqm1 = ml.models.methods(method='AIQM1', qm_program='MNDO')
        # Run thermochemical properties calculation
        ml.simulations.thermochemistry(model=aiqm1, molecule=mol, program='ASE')
        # Get ZPE and heat of formation
        ZPE = mol.ZPE
        Hof = mol.DeltaHf298


    """
    def __init__(self, model=None, molecule=None, program=None, normal_mode_normalization='mass deweighted unnormalized'):
        if model != None:
            self.model = model
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
        freq(model=model, molecule=self.molecule, program=program, normal_mode_normalization=normal_mode_normalization)
        if self.program.casefold() == 'ASE'.casefold(): self.thermochem_ase()
        # Calculate heats of formation
        self.calculate_heats_of_formation()
    
    def thermochem_ase(self):
        from .interfaces import ase_interface
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
        
def numerical_gradients(molecule, model_with_function_to_predict_energy, eps=1e-5, kwargs_funtion_predict_energy={}, return_molecular_database=False):
    if return_molecular_database:
        molDB = data.molecular_database()
    coordinates = molecule.xyz_coordinates.reshape(-1)
    coordinates_list = []
    natoms = len(coordinates) // 3
    for ii in range(len(coordinates)):
        new_coordinates = np.copy(coordinates)
        new_coordinates[ii] += eps
        coordinates_list.append(new_coordinates)
    coordinates_list.append(coordinates)
    def get_energy(coordinates):
        current_molecule = molecule.copy()
        current_molecule.xyz_coordinates = coordinates.reshape(len(current_molecule.atoms),3)
        model_with_function_to_predict_energy.predict(molecule=current_molecule, **kwargs_funtion_predict_energy)
        if return_molecular_database: molDB.molecules.append(current_molecule)
        return current_molecule.energy
    nthreads = env.get_nthreads()
    if nthreads == 1:
        energies = np.array([get_energy(each) for each in coordinates_list])
    else:
        from multiprocessing.pool import ThreadPool as Pool
        env.set_nthreads(1)
        pool = Pool(processes=nthreads)
        energies = np.array(pool.map(get_energy, coordinates_list))
        env.set_nthreads(nthreads)
    relenergy = energies[-1]
    gradients = (energies[:-1]-relenergy)/eps
    if return_molecular_database: return gradients.reshape(natoms,3), molDB
    else:                         return gradients.reshape(natoms,3)

def numerical_hessian(molecule, model_with_function_to_predict_energy, eps=5.29167e-4, epsgrad=1e-5, kwargs_funtion_predict_energy={}):
    g1 = numerical_gradients(molecule, model_with_function_to_predict_energy, epsgrad, kwargs_funtion_predict_energy)
    coordinates1 = molecule.xyz_coordinates.reshape(-1)
    ndim = len(coordinates1)
    hess = np.zeros((ndim, ndim))
    coordinates2 = coordinates1
    for i in range(ndim):
        x0 = coordinates2[i]
        coordinates2[i] = coordinates1[i] + eps
        molecule2 = molecule.copy()
        molecule2.xyz_coordinates = coordinates2.reshape(len(molecule2.atoms),3)
        g2 = numerical_gradients(molecule2, model_with_function_to_predict_energy, epsgrad, kwargs_funtion_predict_energy)
        hess[:, i] = (g2.reshape(-1) - g1.reshape(-1)) / eps
        coordinates2[i] = x0

    return hess
