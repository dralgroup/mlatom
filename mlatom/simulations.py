#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! simulations: Module for simulations                                       ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
  
Geomopt, freq, DMC
+++++++++++++++++++++++
'''
from . import constants, data, models
from .md import md as md
from .md_parallel import md_parallel as md_parallel
from .initial_conditions import generate_initial_conditions
from .md2vibr import vibrational_spectrum
import os, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def run_in_parallel(molecular_database=None, task=None, task_kwargs={}, nthreads=None, create_and_keep_temp_directories=False):
    import joblib
    from joblib import Parallel, delayed
    nmols = len(molecular_database)
    if nthreads == None: nthreads = joblib.cpu_count()
    if nmols < nthreads:
        nthreads_per_model = [nthreads//nmols for ii in range(nmols)]
        extra_threads = nthreads - sum(nthreads_per_model)
        nthreads = nmols
        for ii in range(extra_threads):
            nthreads_per_model[ii] = +1
    else:
        nthreads_per_model = [1 for ii in range(nthreads)]
    def task_loc(imol):
        mol = molecular_database[imol]
        if create_and_keep_temp_directories:
            cwd = os.getcwd()
            directory = f'job_{task.__name__}_{imol+1}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            os.chdir(directory)
        savednthreads = 'savednthreads'
        if 'model' in task_kwargs:
            mm = task_kwargs['model']
            if 'nthreads' in mm.__dict__:
                if mm.nthreads is None or mm.nthreads == 0:
                    savednthreads = mm.nthreads
                    mm.nthreads = nthreads_per_model[imol]
        result = task(molecule=mol, **task_kwargs)
        if savednthreads != 'savednthreads': mm.nthreads = savednthreads
        if create_and_keep_temp_directories: os.chdir(cwd)
        return result
    results = Parallel(n_jobs=nthreads)(delayed(task_loc)(i) for i in range(len(molecular_database)))
    return results

class optimize_geometry():
    """
    Geometry optimization.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): any model or method which provides energies and forces.
        initial_molecule (:class:`mlatom.data.molecule`): the molecule object to optimize.
        ts (bool, optional): whether to do transition state search. Currently only be done with program=Gaussian, ASE and geometric.
        program (str, optional): the engine used in geometry optimization. Currently supports Gaussian, ASE, scipy and PySCF.
        optimization_algorithm (str, optional): the optimization algorithm used in ASE. Default value: LBFGS (ts=False), dimer (ts=False).
        maximum_number_of_steps (int, optional): the maximum number of steps for ASE, SciPy and geometric. Default value: 200.
        convergence_criterion_for_forces (float, optional): forces convergence criterion in ASE. Default value: 0.02 eV/Angstroms.
        working_directory (str, optional): working directory. Default value: '.', i.e., current directory.
        constraints (dict, optional): constraints for geometry optimization. Currently only available with program=ASE and program=geometric. For program=ASE, constraints follows the same conventions as in ASE: ``constraints={'bonds':[[target,[index0,index1]], ...],'angles':[[target,[index0,index1,index2]], ...],'dihedrals':[[target,[index0,index1,index2,index3]], ...]}`` (check `FixInternals class in ASE <https://wiki.fysik.dtu.dk/ase/ase/constraints.html>`__ for more information). For program=geometric, the name of constraint file should be provided and please refer to `constrained optimization <https://geometric.readthedocs.io/en/latest/constraints.html#constraint-types>`__ for the format of the constraint file. 
        print_properties (None or str, optional): properties to print. Default: None. Possible 'all'.
        dump_trajectory_interval (int, optional): dump trajectory at every time step (1). Set to ``None`` to disable dumping (default).
        filename (str, optional): the file that saves the dumped trajectory.
        format (str, optional): format in which the dumped trajectory is saved.

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

    def __init__(self, model=None,  model_predict_kwargs={}, initial_molecule=None, molecule=None, ts=False, program=None, optimization_algorithm=None, maximum_number_of_steps=None, convergence_criterion_for_forces=None,working_directory=None, 
    print_properties=None,
    dump_trajectory_interval=None, # Only None and 1 are supported at the moment
    filename=None, format='json',   
    **kwargs): # Delete the kwargs!
        self.kwargs = kwargs
        if model != None:
            self.model = model
        self.print_properties = print_properties
        self.model_predict_kwargs = model_predict_kwargs

        if not initial_molecule is None and not molecule is None:
            raise ValueError('molecule and initial_molecule cannot be used at the same time')
        overwrite = False
        if not initial_molecule is None:
            self.initial_molecule = initial_molecule.copy(atomic_labels=['xyz_coordinates'],molecular_labels=['number'])
        if not molecule is None:
            overwrite = True
            self.initial_molecule = molecule.copy(atomic_labels=['xyz_coordinates'],molecular_labels=['number'])
        
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
                    except: raise ValueError('please set $GAUSS_EXEDIR or install ase or install scipy')
        
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
        
        self.dump_trajectory_interval = dump_trajectory_interval
        if self.program.casefold() == 'Gaussian'.casefold(): self.dump_trajectory_interval = 1 # Gaussian optimizer needs traj file to get the optimization trajectory
        if self.program.casefold() == 'geometric'.casefold(): self.dump_trajectory_interval = 1
        self.filename = filename
        self.format = format
        if self.print_properties != None and self.dump_trajectory_interval == None:
            self.dump_trajectory_interval = 1
        if self.dump_trajectory_interval != None:
            self.format = format
            if format == 'h5md': ext = '.h5'
            elif format == 'json': ext = '.json'
            if self.filename == None:
                import uuid
                self.filename = str(uuid.uuid4()) + ext 
            # Dump trajectory every step
            self.optimization_trajectory = data.molecular_trajectory()
            self.optimization_trajectory.dump(filename=os.path.join(self.working_directory,self.filename), format=self.format) 
        
        if self.ts and self.program.casefold() not in ['Gaussian'.casefold(), 'ASE'.casefold(), 'geometric'.casefold()]:
            msg = 'Transition state geometry optmization can currently only be done with optimization_program=Gaussian, ASE or geometric'
            raise ValueError(msg)
        
        # Pack the required geomopt-related kwargs into the model kwargs
        self.model_predict_kwargs['return_string'] = False
        self.model_predict_kwargs['dump_trajectory_interval'] = self.dump_trajectory_interval
        self.model_predict_kwargs['filename'] = self.filename
        self.model_predict_kwargs['format'] = self.format
        self.model_predict_kwargs['print_properties'] = self.print_properties
        
        if self.program.casefold() == 'Gaussian'.casefold(): self.opt_geom_gaussian()
        elif self.program.casefold() == 'ASE'.casefold(): self.opt_geom_ase()
        elif self.program.casefold() == 'geometric'.casefold(): self.opt_geom_geometric()
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
        if 'number' in self.initial_molecule.__dict__.keys(): suffix = f'{self.initial_molecule.number}'
        else: suffix = ''
        #print('debug', suffix, self.initial_molecule.number)
        filename = os.path.join(self.working_directory,f'gaussian{suffix}')
        self.model.dump(filename=os.path.join(self.working_directory,'model.json'), format='json')
        
        # Run Gaussian
        external_task='opt'
        if self.ts: external_task = 'ts'
        if self.print_properties is not None:
            print(f' Optimization with Gaussian started.\n Check Gaussian output file "gaussian{suffix}.log" for the progress of optimization.\n')
            filename_json = self.model_predict_kwargs['filename']
            if os.path.exists(f'{filename_json}_tmp_out.out'): os.remove(f'{filename_json}_tmp_out.out')
            self.model_predict_kwargs['return_string'] = True
        gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.initial_molecule, external_task=external_task, cwd=self.working_directory, model_predict_kwargs=self.model_predict_kwargs)
        # Get results
        outputfile = f'{filename}.log'
        if not os.path.exists(outputfile): outputfile = f'{filename}.out'
        with open(outputfile, 'r') as fout:
            for line in fout:
                if 'Stationary point found' in line:
                    self.successful = True
                    break
        self.optimization_trajectory.load(filename=os.path.join(self.working_directory,self.filename), format='json')
        if self.successful: 
            self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        else:
            self.optimized_molecule = self.initial_molecule.copy() 
            for atom in self.optimized_molecule.atoms:
                atom.xyz_coordinates = np.array([None,None,None])                
        if self.print_properties is not None:
            if os.path.exists(f'{filename_json}_tmp_out.out'):
                printstrs = open(f'{filename_json}_tmp_out.out', 'r').readlines()
                for line in printstrs:
                    print(line.rstrip())
                os.remove(f'{filename_json}_tmp_out.out')
        
    def opt_geom_ase(self):
        from .interfaces import ase_interface   

        if self.ts:
            self.optimization_trajectory = ase_interface.transition_state(initial_molecule=self.initial_molecule,
                                            model=self.model,
                                            model_predict_kwargs=self.model_predict_kwargs,
                                            convergence_criterion_for_forces=self.convergence_criterion_for_forces,
                                            maximum_number_of_steps=self.maximum_number_of_steps,
                                            optimization_algorithm=self.optimization_algorithm,
                                            **self.kwargs
                                            )
        else:
            self.optimization_trajectory = ase_interface.optimize_geometry(initial_molecule=self.initial_molecule,
                                            model=self.model,
                                            model_predict_kwargs=self.model_predict_kwargs,
                                            convergence_criterion_for_forces=self.convergence_criterion_for_forces,
                                            maximum_number_of_steps=self.maximum_number_of_steps,
                                            optimization_algorithm=self.optimization_algorithm,
                                            **self.kwargs)
        #self.optimization_trajectory.dump(filename=os.path.join(self.working_directory,self.filename), format=self.format)
        moldb = data.molecular_database()
        moldb.molecules = [each.molecule for each in self.optimization_trajectory.steps]
       # moldb.write_file_with_xyz_coordinates(self.filename.split('.')[0] + '.xyz')
        self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        
    def opt_geom(self):
        try: import scipy.optimize
        except: raise ValueError('scipy is not installed')
            
        istep = -1
        self.optimization_trajectory = data.molecular_trajectory()
        
        def molecular_energy(coordinates):
            nonlocal istep
            istep += 1
            current_molecule = self.initial_molecule.copy()
            current_molecule.xyz_coordinates = coordinates.reshape(len(current_molecule.atoms),3)
            self.model._predict_geomopt(molecule=current_molecule, calculate_energy=True, calculate_energy_gradients=True, **self.model_predict_kwargs)
            if not 'energy' in current_molecule.__dict__:
                raise ValueError('model did not return any energy')
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

    def opt_geom_geometric(self):

        # default optimization algorithm is BFGS

        import geometric
        import geometric.molecule

        if 'constraints' in self.kwargs:
            constraints = self.kwargs['constraints']
        else:
            constraints = None 

        convergence_criterion = {}

        if 'convergence_energy' in self.kwargs:
            convergence_criterion['convergence_energy'] = self.kwargs['convergence_energy']        # default 1e-6 Eh
        if 'convergence_gradient_rms' in self.kwargs:
            convergence_criterion['convergence_grms'] = self.kwargs['convergence_gradient_rms']    # default 3e-4 Eh/Bohr
        if 'convergence_gradient_max' in self.kwargs:
            convergence_criterion['convergence_gmax'] = self.kwargs['convergence_gradient_max']    # default 4.5e-4 Eh/Bohr
        if 'convergence_step_rms' in self.kwargs:
            convergence_criterion['convergence_drms'] = self.kwargs['convergence_step_rms']        # default 1.2e-3 Angstrom
        if 'convergence_step_max' in self.kwargs:
            convergence_criterion['convergence_dmax'] = self.kwargs['convergence_step_max']        # default 1.8e-3 Angstrom

        maximum_number_of_steps = self.maximum_number_of_steps
        model_predict_kwargs = self.model_predict_kwargs
        class MLatomEngine(geometric.engine.Engine):
            def __init__(self, MLatomMol, model):
                
                molecule = geometric.molecule.Molecule()
                self.mol = MLatomMol 
                self.model = model
                molecule.elem = MLatomMol.element_symbols.tolist()
                molecule.xyzs = [MLatomMol.xyz_coordinates]
                super(MLatomEngine, self).__init__(molecule)
                self.cycle = 0
                self.e_last = 0
                self.maxsteps = maximum_number_of_steps

            def calc_new(self, coords, dirname):
                mol = self.mol
                mol.xyz_coordinates = coords.reshape(-1,3)*constants.Bohr2Angstrom
                self.model._predict_geomopt(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, **model_predict_kwargs)
                energy = mol.energy
                gradients = mol.get_energy_gradients()/constants.Angstrom2Bohr
                self.cycle += 1
                return {"energy": energy, "gradient": gradients.ravel()}
        
        import tempfile, contextlib
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = os.path.abspath(tmpdirname)

        import logging
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if logger.name in ['geometric.nifty', 'geometric']:
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
        mlatom_engine = MLatomEngine(self.initial_molecule, self.model)
        try:
            geometric.optimize.run_optimizer(customengine=mlatom_engine, input=tmpdirname, constraints=constraints, transition=self.ts, maxiter=self.maximum_number_of_steps, **convergence_criterion)
            self.successful = True
            self.converge = True
        except Exception as ex:
            if type(ex) == geometric.errors.GeomOptNotConvergedError:
                print('Warning: Geometry optimization with geometric failed to converge. The last geometry will be used as the optimized geometry.')
                self.converge = False
                self.successful = True
            else:
                print('Warning: Geometry optimization with geometric failed. The initial geometry will be used as the optimized geometry.')
                self.converge = False
                self.successful = False

        if self.successful:
            self.optimization_trajectory.load(filename=os.path.join(self.working_directory,self.filename), format='json')
            self.optimized_molecule = self.optimization_trajectory.steps[-1].molecule
        else:
            self.optimized_molecule = self.initial_molecule
            self.model.predict(molecule=self.optimized_molecule, calculate_energy=True)

class irc():
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model = kwargs['model']
        if 'ts_molecule' in kwargs:
            self.ts_molecule = kwargs['ts_molecule'].copy(atomic_labels=['xyz_coordinates','number'],molecular_labels=[])

        if 'model_predict_kwargs' in kwargs:
            self.model_predict_kwargs = kwargs['model_predict_kwargs']
        else:
            self.model_predict_kwargs = {}

        from .interfaces import gaussian_interface
        if 'number' in self.ts_molecule.__dict__.keys(): suffix = f'_{self.ts_molecule.number}'
        else: suffix = ''
        filename = f'gaussian{suffix}'
        self.model.dump(filename='model.json', format='json')
        
        # Run Gaussian
        gaussian_interface.run_gaussian_job(filename=f'{filename}.com', molecule=self.ts_molecule, external_task='irc', model_predict_kwargs=self.model_predict_kwargs)
        
        #if os.path.exists('model.json'): os.remove('model.json')

class freq():
    """
    Frequence analysis.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): any model or method which provides energies and forces and Hessian.
        molecule (:class:`mlatom.data.molecule`): the molecule object with necessary information.
        program (str, optional): the engine used in frequence analysis through modified TorchANI (if Gaussian not found or any other string is given), pyscf or Gaussian interfaces.
        normal_mode_normalization (str, optional): normal modes output scheme. It should be one of: mass weighted normalized, mass deweighted unnormalized, and mass deweighted normalized (default). 
        anharmonic (bool): whether to do anharmonic frequence calculation.
        working_directory (str, optional): working directory. Default value: '.', i.e., current directory.

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
    def __init__(self, model=None, model_predict_kwargs={}, molecule=None, program=None, normal_mode_normalization='mass deweighted normalized', anharmonic=False, anharmonic_kwargs={}, working_directory=None):
        if model != None:
            self.model = model
        self.model_predict_kwargs = model_predict_kwargs
        self.molecule = molecule
        if program != None:
            self.program = program
        else:
            if "GAUSS_EXEDIR" in os.environ: 
                self.program = 'Gaussian'
            else:
                try: 
                    import pyscf
                    self.program = 'PySCF'
                except:
                    self.program = ''
        self.normal_mode_normalization = normal_mode_normalization
        self.anharmonic_kwargs = anharmonic_kwargs
        if working_directory != None:
            self.working_directory = working_directory
        else:
            self.working_directory = '.'
        if self.program.casefold() == 'Gaussian'.casefold(): self.freq_gaussian(anharmonic)
        elif self.program.casefold() == 'pyscf'.casefold(): self.freq_pyscf()
        else:
            if not 'shape' in self.molecule.__dict__:
                self.molecule.shape = 'nonlinear'
            self.freq_modified_from_TorchANI(molecule=self.molecule,normal_mode_normalization=self.normal_mode_normalization,model=self.model, model_predict_kwargs=self.model_predict_kwargs)
        
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
            gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.molecule, external_task='freq(anharmonic)',cwd=self.working_directory,**self.anharmonic_kwargs, model_predict_kwargs=self.model_predict_kwargs)
        else:
            gaussian_interface.run_gaussian_job(filename=f'gaussian{suffix}.com', molecule=self.molecule, external_task='freq',cwd=self.working_directory, model_predict_kwargs=self.model_predict_kwargs)
        
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
        if self.molecule.infrared_intensities == []:
            del(self.molecule.infrared_intensities)
        if os.path.exists(os.path.join(self.working_directory,'gaussian_freq_mol.json')): os.remove(os.path.join(self.working_directory,'gaussian_freq_mol.json'))

    def freq_pyscf(self):
        self.successful = False
        self.model.predict(molecule=self.molecule, calculate_energy=True, calculate_hessian=True, **self.model_predict_kwargs)
        from .interfaces import pyscf_interface
        self.successful = pyscf_interface.thermo_calculation(molecule=self.molecule)
    
    @classmethod
    def freq_modified_from_TorchANI(cls,molecule,normal_mode_normalization,model=None, **kwargs):
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
        if 'model_predict_kwargs' in kwargs:
            model_predict_kwargs = kwargs['model_predict_kwargs']
        else:
            model_predict_kwargs = {}

        if not model is None:
            model.predict(molecule=molecule, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True)
        
        mhessian2fconst = 4.359744650780506
        unit_converter = 17091.7006789297
        # Solving the eigenvalue problem: Hq = w^2 * T q
        # where H is the hessian matrix, q is the normal coordinates,
        # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
        # We solve this eigenvalue problem through Lowdin diagnolization:
        # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
        # Letting q' = T^(1/2) q, we then have
        # T^(-1/2) H T^(-1/2) q' = w^2 * q'
        masses = np.expand_dims(molecule.get_nuclear_masses(), axis=0)
        inv_sqrt_mass = np.repeat(np.sqrt(1 / masses), 3, axis=1) # shape (3 * atoms)
        mass_scaled_hessian = molecule.hessian * np.expand_dims(inv_sqrt_mass, axis=1) * np.expand_dims(inv_sqrt_mass, axis=2)
        mass_scaled_hessian = np.squeeze(mass_scaled_hessian, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(mass_scaled_hessian)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        angular_frequencies = [] 
        for each in eigenvalues:
            if each < 0:
                angular_frequencies.append(-np.sqrt(-each))
            else:
                angular_frequencies.append(np.sqrt(each))
        angular_frequencies = np.array(angular_frequencies)

        frequencies = angular_frequencies / (2 * math.pi)
        # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 
        wavenumbers = unit_converter * frequencies

        # In case of complex numbers, get real part of them
        wavenumbers = wavenumbers.real

        # Note that the normal modes are the COLUMNS of the eigenvectors matrix
        mw_normalized = eigenvectors.T
        md_unnormalized = mw_normalized * inv_sqrt_mass
        norm_factors = 1 / np.linalg.norm(md_unnormalized, axis=1)  # units are sqrt(AMU)
        md_normalized = md_unnormalized * np.expand_dims(norm_factors, axis=1)
        md_normalized = md_normalized.real

        rmasses = norm_factors**2  # units are AMU
        # converting from Ha/(AMU*A^2) to mDyne/(A*AMU) 
        fconstants = mhessian2fconst * eigenvalues * rmasses  # units are mDyne/A
        fconstants = fconstants.real

        if normal_mode_normalization == 'mass deweighted normalized':
            modes = (md_normalized).reshape(frequencies.size, -1, 3)
        elif normal_mode_normalization == 'mass deweighted unnormalized':
            modes = (md_unnormalized).reshape(frequencies.size, -1, 3)
        elif normal_mode_normalization == 'mass weighted normalized':
            modes = (mw_normalized).reshape(frequencies.size, -1, 3)

        # the first 6 (5 for linear) entries are for rotation and translation
        # we skip them because we are only interested in vibrational modes
        #nskip = 6
        #if molecule.is_it_linear():
        #    nskip = 5
        # Ugly fix of negative frequency problem in local minimum:
        # If there are two large negative frequencies ( <-100 cm^-1 ), skip the first 5 or 6 frequencies 
        # Otherwise, sort by absolute value of frequecies and skip the first 5 or 6 frequencies 
        #if wavenumbers[1] > -100:
        #    idx = np.sort(abs(wavenumbers).argsort()[nskip:])
        #else:
        #    idx = np.array([ii for ii in range(nskip,len(wavenumbers))])
        nskip = 0
        idx = np.array([ii for ii in range(nskip,len(wavenumbers))])
        molecule.frequencies = wavenumbers[idx]    # in cm^-1
        molecule.force_constants = fconstants[idx] # in mDyne/A
        molecule.reduced_masses = rmasses[idx]     # in AMU
        for iatom in range(len(molecule.atoms)):
            molecule.atoms[iatom].normal_modes = []
            for imode in idx:
                molecule.atoms[iatom].normal_modes.append(list(modes[imode][iatom]))
            molecule.atoms[iatom].normal_modes = np.array(molecule.atoms[iatom].normal_modes)
 
            
class thermochemistry():
    """
    Thermochemical properties calculation.

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): any model or method which provides energies and forces and Hessian.
        molecule (:class:`mlatom.data.molecule`): the molecule object with necessary information.
        program (str): the engine used in thermochemical properties calculation. Currently support Gaussian and ASE.
        normal_mode_normalization (str, optional): normal modes output scheme. It should be one of: mass weighted normalized, mass deweighted unnormalized, and mass deweighted unnormalized (default). 
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

    The thermochemical properties available in ``molecule`` object after the calculation:
    
    * ``ZPE``: Zero-point energy
    
    * ``DeltaE2U``: Thermal correction to Energy (only available in Gaussian)
    
    * ``DeltaE2H``: Thermal correction to Enthalpy (only available in Gaussian)
    
    * ``DeltaE2G``: Thermal correction to Gibbs free energy (only available in Gaussian)
    
    * ``U0``: Internal energy at 0K
    
    * ``H0``: Enthalpy at 0K       
    
    * ``U``: Internal energy (only available in Gaussian)
    
    * ``H``: Enthalpy
    
    * ``G``: Gibbs free energy
    
    * ``S``: Entropy (only available in Gaussian)
    
    * ``atomization_energy_0K``
    
    * ``ZPE_exclusive_atomization_energy_0K``
    
    * ``DeltaHf298``: Heat of formation at 298 K
    """
    def __init__(self, model=None, molecule=None, program=None, normal_mode_normalization='mass deweighted normalized'):
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
                    raise ValueError('please set $GAUSS_EXEDIR or install ase')
        freq(model=model, molecule=self.molecule, program=program, normal_mode_normalization=normal_mode_normalization)
        if self.program.casefold() == 'ASE'.casefold(): self.thermochem_ase()
        # Calculate heats of formation
        self.calculate_heats_of_formation()
    
    def thermochem_ase(self):
        from .interfaces import ase_interface
        ase_interface.thermochemistry(molecule=self.molecule)
        
    def calculate_heats_of_formation(self):
        if 'scf_enthalpy_of_formation_at_298_K' in self.molecule.__dict__:
            self.molecule.DeltaHf298 = self.molecule.energy
            return
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
                self.model.predict(molecule=atomic_molecule, calculate_energy=True)
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
        
class dmc():
    '''
    Run diffusion Monte Carlo simulation for molecule(s) using `PyVibDMC <https://github.com/rjdirisio/pyvibdmc>`_.
    
    Arguments:
        model (:class:`mlatom.models.model`): The potential energy surfaces model. The unit should be Hartree, otherwise a correct ``energy_scaling_factor`` need to be set. 
        initial_molecule (:class:`mlatom.data.molecule`): The initial geometry for the walkers. Usually a energy minimum geometry should be provided. By default every coordinate will be scaled by 1.01 to make it slightly distorted.
        energy_scaling_factor (float, optional): A factor that will be multiplied to the model's energy pridiction.    
    '''

    def __init__(self, model: models.model, initial_molecule:data.molecule = None, initial_molecular_database: data.molecular_database = None, energy_scaling_factor:float = 1., ):
        from .constants import Bohr2Angstrom

        if not initial_molecular_database:
            initial_molecular_database = data.molecular_database([initial_molecule])
        
        self.model = model
        self.atoms = list(initial_molecular_database[0].element_symbols)
        self.start_structures = initial_molecular_database.xyz_coordinates / Bohr2Angstrom * 1.01
        self.energy_scaling_factor = energy_scaling_factor


    
    def potential_function(self, coordinates):
        from .constants import Bohr2Angstrom
        molDB = data.molecular_database.from_numpy(coordinates=coordinates * Bohr2Angstrom, species=np.repeat([self.atoms], coordinates.shape[0], axis=0))
        self.model.predict(molecular_database=molDB, calculate_energy=True)
        return molDB.get_properties('energy') * self.energy_scaling_factor
    
    def initialize(self, number_of_walkers, generation_method='harmonic_sampling',**kwargs):
        import pyvibdmc as pv
        initializer = pv.InitialConditioner(coord=self.start_structures,
                                    atoms=self.atoms,
                                    num_walkers=number_of_walkers,
                                    technique=generation_method,
                                    **kwargs)
        self.start_structures = initializer.run()

    def run(self, run_dir: str = 'DMC', weighting: str = 'discrete', number_of_walkers: int = 5000, number_of_timesteps: int = 10000, equilibration_steps: int = 500, dump_trajectory_interval: int = 500, dump_wavefunction_interval: int = 1000, descendant_weighting_steps: int = 300, time_step: float = 1 * constants.au2fs, initialize: bool = False):
        '''
        Run the DMC simulation.
        
        Arguments:
            run_dir (str): The folder for the output files.
            weighting (str): ``'discrete'`` or ``'continuous'``. ``'continuous'`` keeps the ensemble size constant.
            number_of_walkers (int): The number of geometries exploring the potential surface.
            number_of_timesteps (int): The number of steps the simulation will go.
            equilibration_steps (int): The number of steps for equilibration.
            dump_trajectory_interval (int): The interval for dumping walkers' trajectories.
            dump_wavefunction_interval (int): The interval for collecting wave function.
            descendant_weighting_steps (int): The number of time steps for descendant weighting per wave function.
            time_step (float): The length of each time step in fs.            
        '''
        from pyvibdmc import potential_manager as pm
        import pyvibdmc as pv
        
        if initialize:
            self.initialize(number_of_walkers=number_of_walkers,)
        
        DMC_job = pv.DMC_Sim(sim_name='DMC',
                            output_folder=run_dir,
                            weighting=weighting, #or 'continuous'. 'continuous' keeps the ensemble size constant.
                            num_walkers=number_of_walkers, #number of geometries exploring the potential surface
                            num_timesteps=number_of_timesteps, #how long the simulation will go. (num_timesteps *      atomic units of time)
                            equil_steps=equilibration_steps, #how long before we start collecting wave functions
                            chkpt_every=dump_trajectory_interval, #checkpoint the simulation every "chkpt_every" time steps
                            wfn_every=dump_wavefunction_interval, #collect a wave function every "wfn_every" time steps
                            desc_wt_steps=descendant_weighting_steps, #number of time steps you allow for descendant weighting per wave function
                            atoms=self.atoms,
                            delta_t=time_step * constants.fs2au, #the size of the time step in fs
                            potential=pm.Potential_Direct(potential_function=self.potential_function),
                            start_structures=self.start_structures, #can provide a single geometry, or an ensemble of geometries
                            masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
        )
        DMC_job.run()
        self.load(f"{run_dir}/DMC_sim_info.hdf5")
    
    def load(self, filename):
        '''
        Load previous simulation results from a HDF5 file.
        '''
        import pyvibdmc as pv
        self.result = pv.SimInfo(filename)

    def get_zpe(self, start_step=1000) -> float:
        '''
        Return calculated zero-point energy in Hartree.
        
        Arguments:
            start_step (int): The starting step for averaging the energies.
        '''
        return self.result.get_zpe(onwards=start_step, ret_cm=False)

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
    from multiprocessing import cpu_count
    nthreads = cpu_count()
    if nthreads == 1:
        energies = np.array([get_energy(each) for each in coordinates_list])
    else:
        from multiprocessing.pool import ThreadPool as Pool
        model_with_function_to_predict_energy.set_num_threads(1)
        pool = Pool(processes=nthreads)
        energies = np.array(pool.map(get_energy, coordinates_list))
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
