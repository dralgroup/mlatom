import sys
from . import data, stats, models, simulations, optimize_geometry, md, md_parallel, constants, generate_initial_conditions
import numpy as np 
import os
import random 
from . import gap_md, namd

class Sampler():
    def __init__(self,sampler_function=None):
        if type(sampler_function) == str:
            if sampler_function.casefold() == 'wigner'.casefold():
                self.sampler_function = self.wigner 
            elif sampler_function.casefold() == 'geomopt'.casefold():
                self.sampler_function = self.geometry_optimization 
            elif sampler_function.casefold() == 'md'.casefold():
                self.sampler_function = self.molecular_dynamics
            elif sampler_function.casefold() == 'batch_md'.casefold():
                self.sampler_function = self.batch_molecular_dynamics
            elif sampler_function.casefold() == 'md_excess_energy'.casefold():
                self.sampler_function = self.molecular_dynamics_excess_energy
            elif sampler_function.casefold() == 'random'.casefold():
                self.sampler_function = self.random
            elif sampler_function.casefold() == 'fixed'.casefold():
                self.sampler_function = self.fixed
            elif sampler_function.casefold() == 'harmonic-quantum-boltzmann'.casefold():
                self.sampler_function = self.harmonic_quantum_boltzmann
            elif sampler_function.casefold() == 'lzsh'.casefold():
                self.sampler_function = self.surface_hopping_molecular_dynamics
            else:
                stopper(f"Unsupported sampler function type: {sampler_function}")
        else:
            self.sampler_function = sampler_function

    def sample(self,**kwargs):
        return self.sampler_function(**kwargs)
    
    def fixed(
            self,
            al_info={},
            molecule=None,
            molecular_database=None,
            **kwargs
        ):

        if molecule is None and molecular_database is None:
            stopper("Sampler(fixed): Neither molecule nor molecular_database is provided")
        elif not molecule is None and not molecular_database is None:
            stopper("Sampler(fixed): Both molecule and molecular_database are provided")
        elif not molecule is None:
            molecular_database = data.molecular_database()
            molecular_database.append(molecule)
        elif not molecular_database is None:
            pass 

        # add the source of molecule
        for mol in molecular_database:
            mol.sampling = 'fixed'

        return molecular_database

    def random(
            self,
            al_info={},
            molecule=None,
            number_of_initial_conditions=1,
            scale=0.1,
            **kwargs
        ):

        moldb = data.molecular_database()
        rand = (np.random.rand(number_of_initial_conditions,len(molecule),3) - 0.5) * scale
        mol_xyz = molecule.xyz_coordinates 
        rand_xyz = rand + mol_xyz 
        for each in rand_xyz:
            mol = molecule.copy() 
            mol.xyz_coordinates = each 
            moldb.append(mol)
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'random'
        return moldb

    def wigner(
            self,
            al_info={},
            molecule=None,
            number_of_initial_conditions=1,
            initial_temperature=300,
            use_hessian=False,
            random_seed=None,
            **kwargs
        ):
    
        if isinstance(molecule,data.molecule):
            moldb = generate_initial_conditions(molecule=molecule,
                                                generation_method='wigner',
                                                number_of_initial_conditions=number_of_initial_conditions,
                                                initial_temperature=initial_temperature,
                                                use_hessian=use_hessian,
                                                random_seed=random_seed)
        elif isinstance(molecule,data.molecular_database):
            nmols = len(molecule)
            nsample_each = number_of_initial_conditions // nmols 
            nremaining = number_of_initial_conditions % nmols 
            nsample_list = [nsample_each for imol in range(nmols)]
            for ii in range(nremaining):
                nsample_list[ii] += 1 
            moldb = data.molecular_database() 
            # print(nsample_list)
            for imol in range(nmols):
                mol = molecule[imol]
                moldb_each = generate_initial_conditions(molecule=mol,
                                                        generation_method='wigner',
                                                        number_of_initial_conditions=nsample_list[imol],
                                                        initial_temperature=initial_temperature,
                                                        use_hessian=use_hessian,
                                                        random_seed=random_seed)
                moldb += moldb_each
                # print(len(moldb))
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'wigner'
        return moldb 
    
    def geometry_optimization(
            self,
            al_info={},
            initcond_sampler=None,
            initcond_sampler_kwargs=None,
            ml_model=None,
            program='ase',
            uq_threshold=None,
            **kwargs
        ):

        if not uq_threshold is None:
            print(f"UQ threshold is provided in sampler, overwriting saved threshold")
            print(f"Current threshold: {uq_threshold}")
            ml_model.uq_threshold = uq_threshold

        initial_molecular_database = initcond_sampler.sample(al_info=al_info,**initcond_sampler_kwargs)
        
        moldb = data.molecular_database() 
        opt_moldb = data.molecular_database()
        itraj = 0
        for mol in initial_molecular_database:
            itraj += 1
            opt = optimize_geometry(model=ml_model,program=program,initial_molecule=mol)
            opt_moldb.append(opt.optimized_molecule)
            # if not stop_function is None:
            for istep in range(len(opt.optimization_trajectory.steps)):
                step = opt.optimization_trajectory.steps[istep]
                stop = internal_stop_function(step.molecule)
                if stop:
                    if 'need_to_be_labeled' in step.molecule.__dict__:
                        print(f'Adding molecule from trajectory {itraj} at step {istep}')
                        moldb.append(step.molecule)
                        break
            # else:
            #     moldb.append(opt.optimized_molecule)
        opt_moldb.dump(filename='geomopt_db.json',format='json')
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'geometry-optimization'
        return moldb 
    
    def molecular_dynamics(
            self,
            al_info={},
            ml_model=None,
            initcond_sampler=None,
            initcond_sampler_kwargs={},
            maximum_propagation_time=1000, 
            time_step=0.1, 
            ensemble='NVE', 
            thermostat=None, 
            dump_trajs=False, 
            dump_trajectory_interval=None, 
            stop_function=None, 
            nthreads=None,
            uq_threshold=None,
            **kwargs
        ):

        import inspect 
        args = inspect.getfullargspec(initcond_sampler.sample)[0]
        if 'al_info' in args:
            initial_molecular_database = initcond_sampler.sample(al_info=al_info,**initcond_sampler_kwargs)
        else:
            initial_molecular_database = initcond_sampler.sample(**initcond_sampler_kwargs)
        if nthreads is None:
            import joblib
            nthreads = joblib.cpu_count()
        if stop_function is None:
            stop_function = internal_stop_function

        if not uq_threshold is None:
            print(f"UQ threshold is provided in sampler, overwriting saved threshold")
            print(f"Current threshold: {uq_threshold}")
            ml_model.uq_threshold = uq_threshold

        moldb = data.molecular_database()
        md_kwargs = {
                'model': ml_model,
                'time_step': time_step,
                'maximum_propagation_time': maximum_propagation_time,
                'ensemble': ensemble,
                'thermostat': thermostat,
                'dump_trajectory_interval': dump_trajectory_interval,
                'stop_function': stop_function
                }
        dyns = simulations.run_in_parallel(molecular_database=initial_molecular_database,
                                            task=md,
                                            task_kwargs=md_kwargs,
                                            create_and_keep_temp_directories=False)
        
        trajs = [d.molecular_trajectory for d in dyns]
        sys.stdout.flush() 

        itraj=0 
        for traj in trajs:
            itraj+=1 
            print(f"Trajectory {itraj} number of steps: {len(traj.steps)}")
            # if 'need_to_be_labeled' in traj.steps[-1].molecule.__dict__:# and len(traj.steps) > 1:
            if traj.steps[-1].molecule.uncertain:
                print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, traj.steps[-1].time))
                moldb.molecules.append(traj.steps[-1].molecule)

            # Dump traj
            if dump_trajs:
                import os
                if 'working_directory' in al_info.keys():
                    os.path.join(al_info['working_directory'],"trajs")
                    # dirname = f"{al_info['working_directory']}/trajs"
                else:
                    dirname = 'trajs'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                traj.dump(f"{dirname}/traj{itraj}.h5",format='h5md')
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'md'
        return moldb

    
    def batch_molecular_dynamics(
            self,
            al_info={},
            ml_model=None,
            initcond_sampler=None,
            initcond_sampler_kwargs={},
            maximum_propagation_time=1000, 
            time_step=0.1, 
            ensemble='NVE', 
            thermostat=None, 
            dump_trajs=False, 
            dump_trajectory_interval=None, 
            stop_function=None, 
            nthreads=None,
            uq_threshold=None,
            **kwargs
        ):

        import inspect 
        args = inspect.getfullargspec(initcond_sampler.sample)[0]
        if 'al_info' in args:
            initial_molecular_database = initcond_sampler.sample(al_info=al_info,**initcond_sampler_kwargs)
        else:
            initial_molecular_database = initcond_sampler.sample(**initcond_sampler_kwargs)
        if nthreads is None:
            import joblib
            nthreads = joblib.cpu_count()
        if stop_function is None:
            stop_function = internal_stop_function

        if not uq_threshold is None:
            print(f"UQ threshold is provided in sampler, overwriting saved threshold")
            print(f"Current threshold: {uq_threshold}")
            ml_model.uq_threshold = uq_threshold

        moldb = data.molecular_database()
        dyn = md_parallel(model=ml_model,
                            molecular_database=initial_molecular_database,
                            ensemble=ensemble,
                            thermostat=thermostat,
                            time_step=time_step,
                            maximum_propagation_time=maximum_propagation_time,
                            dump_trajectory_interval=dump_trajectory_interval,
                            stop_function=stop_function)
        trajs = dyn.molecular_trajectory 
        for itraj in range(len(trajs.steps[0])):
            print(f"Trajectory {itraj} number of steps: {trajs.traj_len[itraj]}")
            # if 'need_to_be_labeled' in trajs.steps[trajs.traj_len[itraj]][itraj].__dict__:
            if trajs.steps[trajs.traj_len[itraj]][itraj].uncertain:
                print(f'Adding molecule from trajectory {itraj} at time {trajs.traj_len[itraj]*time_step} fs')
                moldb.molecules.append(trajs.steps[trajs.traj_len[itraj]][itraj])

            # Dump traj
            if dump_trajs:
                import os
                traj = data.molecular_trajectory()
                for istep in range(trajs.traj_len[itraj]+1):

                    step = data.molecular_trajectory_step()
                    step.step = istep
                    step.time = istep * time_step
                    step.molecule = trajs.steps[istep][itraj]
                    traj.steps.append(step)
                if 'working_directory' in al_info.keys():
                    dirname = os.path.join(al_info['working_directory'],"trajs")
                    # dirname = f"{al_info['working_directory']}/trajs"
                else:
                    dirname = 'trajs'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                traj.dump(f"{dirname}/traj{itraj}.h5",format='h5md')
                
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'batch_md'
        return moldb


    def molecular_dynamics_excess_energy(
            self,
            al_info={},
            ml_model=None,
            initcond_sampler=None,
            initcond_sampler_kwargs={},
            maximum_propagation_time=1000.0,
            time_step=0.1,
            ensemble='NVE',
            thermostat=None,
            dump_trajs=False,
            dump_trajectory_interval=None,
            stop_function=None,
            batch_parallelization=True,
            nthreads=None,
            uq_threshold=None,
            **kwargs
        ):

        import inspect 
        args = inspect.getfullargspec(initcond_sampler.sample)[0]
        if 'al_info' in args:
            initial_molecular_database = initcond_sampler.sample(al_info=al_info,**initcond_sampler_kwargs)
        else:
            initial_molecular_database = initcond_sampler.sample(**initcond_sampler_kwargs)
        if nthreads is None:
            import joblib
            nthreads = joblib.cpu_count()
        if stop_function is None:
            stop_function = internal_stop_function
        if not uq_threshold is None:
            print(f"UQ threshold is provided in sampler, overwriting saved threshold")
            print(f"Current threshold: {uq_threshold}")
            ml_model.uq_threshold = uq_threshold

        # if 'excess_energy' in kwargs:
        #     excess_energy = kwargs['excess_energy']
        # else:
        #     excess_energy = 0

        # initial_molecular_database = initcond_sampler.sample(al_object=al_object,**initcond_sampler_kwargs)

        excess_energy = [mol.excess_energy for mol in initial_molecular_database]
    
        moldb = data.molecular_database()

        if batch_parallelization:
            dyn = md_parallel(model=ml_model,
                              molecular_database=initial_molecular_database,
                                 ensemble=ensemble,
                                 thermostat=thermostat,
                                 time_step=time_step,
                                 maximum_propagation_time=maximum_propagation_time,
                                 excess_energy=excess_energy,
                                 dump_trajectory_interval=dump_trajectory_interval,
                                 stop_function=stop_function,)
            trajs = dyn.molecular_trajectory 
            for itraj in range(len(trajs.steps[0])):
                print(f"Trajectory {itraj} number of steps: {trajs.traj_len[itraj]}")
                if 'need_to_be_labeled' in trajs.steps[trajs.traj_len[itraj]][itraj].__dict__:
                    print(f'Adding molecule from trajectory {itraj} at time {trajs.traj_len[itraj]*time_step} fs')
                    moldb.molecules.append(trajs.steps[trajs.traj_len[itraj]][itraj])

                # Dump traj
                if dump_trajs:
                    import os
                    traj = data.molecular_trajectory()
                    for istep in range(trajs.traj_len[itraj]+1):

                        step = data.molecular_trajectory_step()
                        step.step = istep
                        step.time = istep * time_step
                        step.molecule = trajs.steps[istep][itraj]
                        traj.steps.append(step)
                    if 'working_directory' in al_info.keys():
                        dirname = os.path.join(al_info['working_directory'],"trajs")
                        # dirname = f"{al_info['working_directory']}/trajs"
                    else:
                        dirname = 'trajs'
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    traj.dump(f"{dirname}/traj{itraj}.h5",format='h5md')
        else:
            def run_traj(imol):
                initmol = initial_molecular_database.molecules[imol]

                dyn = md(model=ml_model,
                            molecule_with_initial_conditions=initmol,
                            ensemble='NVE',
                            time_step=time_step,
                            maximum_propagation_time=maximum_propagation_time,
                            excess_energy=excess_energy[imol],
                            dump_trajectory_interval=None,
                            stop_function=internal_stop_function,)
                traj = dyn.molecular_trajectory 
                return traj 
            
            from joblib import Parallel, delayed 
            trajs = Parallel(n_jobs=nthreads)(delayed(run_traj)(i) for i in range(len(initial_molecular_database)))
            sys.stdout.flush() 

            itraj=0 
            for traj in trajs:
                itraj+=1 
                print(f"Trajectory {itraj} number of steps: {len(traj.steps)}")
                # if 'need_to_be_labeled' in traj.steps[-1].molecule.__dict__:# and len(traj.steps) > 1:
                if traj.steps[-1].molecule.uncertain:
                    print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, traj.steps[-1].time))
                    moldb.molecules.append(traj.steps[-1].molecule)

                # Dump traj
                if dump_trajs:
                    import os
                    if 'working_directory' in al_info.keys():
                        os.path.join(al_info['working_directory'],"trajs")
                        # dirname = f"{al_info['working_directory']}/trajs"
                    else:
                        dirname = 'trajs'
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    traj.dump(f"{dirname}/traj{itraj}.h5",format='h5md')
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'md'
        return moldb


    def harmonic_quantum_boltzmann(
            self,
            molecule,
            number_of_initial_conditions=1,
            initial_temperature=300,
            use_hessian=False,
            random_seed=None,
            **kwargs
        ):
        
        if isinstance(molecule,data.molecule):
            moldb = generate_initial_conditions(molecule=molecule,
                                                   generation_method='harmonic-quantum-boltzmann',
                                                   number_of_initial_conditions=number_of_initial_conditions,
                                                   initial_temperature=initial_temperature,
                                                   use_hessian=use_hessian,
                                                   random_seed=random_seed)
        elif isinstance(molecule,data.molecular_database):
            nmols = len(molecule)
            nsample_each = number_of_initial_conditions // nmols 
            nremaining = number_of_initial_conditions % nmols 
            nsample_list = [nsample_each for imol in range(nmols)]
            for ii in range(nremaining):
                nsample_list[ii] += 1 
            moldb = data.molecular_database() 
            # print(nsample_list)
            for imol in range(nmols):
                mol = molecule[imol]
                moldb_each = generate_initial_conditions(molecule=mol,
                                                            generation_method='harmonic-quantum-boltzmann',
                                                            number_of_initial_conditions=nsample_list[imol],
                                                            initial_temperature=initial_temperature,
                                                            use_hessian=use_hessian,
                                                            random_seed=random_seed)
                moldb += moldb_each
        # add the source of molecule
        for mol in moldb:
            mol.sampling = 'harmonic-quantum-bolzmann'
        return moldb 
    
    def get_excess_energy(self,**kwargs):

        if 'molecular_database' in kwargs:
            molecular_database = kwargs['molecular_database']
        if 'molecule' in kwargs:
            molecular_database = data.molecular_database()
            molecular_database += kwargs['molecule']
        if 'averate_IEE_per_atom' in kwargs:
            averate_IEE_per_atom = kwargs['averate_IEE_per_atom']
        else:
            averate_IEE_per_atom = 0.6 # Unit: eV 
        if 'impulsion_energy' in kwargs:
            impulsion_energy = kwargs['impulsion_energy']
        else:
            impulsion_energy = 70

        if not 'excess_energy_generator' in self.__dict__:
            self.excess_energy_generator = excess_energy_generator(
                molecule=molecular_database[0],
                average_IEE_per_atom=averate_IEE_per_atom,
                impulsion_energy=impulsion_energy,
            )
        
        excess_energies = self.excess_energy_generator.get_excess_energies(nsample=len(molecular_database))

        for imol in range(len(molecular_database)):
            molecular_database[imol].excess_energy = excess_energies[imol]

        return molecular_database
    
    def surface_hopping_molecular_dynamics(self, al_info={}, ml_model=None, **kwargs):
        if 'initcond_sampler_kwargs' in kwargs:
            initcond_sampler_kwargs = kwargs['initcond_sampler_kwargs']
        else:
            stopper('Sampler(namd): Please provide method for initial conditions sampling')
        if 'eqmol' in initcond_sampler_kwargs:
            eqmol = initcond_sampler_kwargs['eqmol']
        else:
            stopper('Sampler(wigner): Please specify equiilibrium geometry for wigner sampling')
        if 'initial_temperature' in initcond_sampler_kwargs:
            initial_temperature = initcond_sampler_kwargs['initial_temperature']
        else:
            initial_temperature = None  
        if 'number_of_initial_conditions' in initcond_sampler_kwargs:
            ntraj = initcond_sampler_kwargs['number_of_initial_conditions']
        else:
            ntraj=50
        if 'ngapmd_trajs' in kwargs:
            N_gapMD = int(0.5*kwargs['ngapmd_trajs'])
        else:
            N_gapMD = int(ntraj*0.5)
        if N_gapMD > ntraj:
            N_gapMD = ntraj
        if 'min_surfuq_points' in kwargs:
            min_surfuq_points = kwargs['min_surfuq_points']
        else:
            min_surfuq_points = 5
        if 'number_of_points_to_sample' in kwargs:
            max_points = kwargs['number_of_points_to_sample']
        else:
            max_points = None
        if 'model_predict_kwargs' in kwargs:
            model_predict_kwargs = kwargs['model_predict_kwargs']
        else:
            model_predict_kwargs = {}
        if 'stop_uncertain_md' in kwargs:
            if kwargs['stop_uncertain_md'] == True:
                stop_function = internal_stop_function_namd
            else:
                stop_function = None
        else:
            stop_function = None
        if 'maximum_propagation_time' in kwargs:
            maximum_propagation_time = kwargs['maximum_propagation_time']
        else:
            stopper("Sampler(md): Please provide maximum propagation time for MD")
        if 'time_step' in kwargs:
            time_step = kwargs['time_step']
        else:
            stopper("Sampler(md): Please provide time step for MD")
        if 'nstates' in kwargs:
            nstates = kwargs['nstates']
        else:
            stopper("Sampler(md): Please provide nstates for MD")
        if 'initial_state' in kwargs:
            initstate = kwargs['initial_state']
        else:
            stopper("Sampler(md): Please provide initial state for MD")
        # if 'seed' in kwargs:
        #     seed = kwargs['seed']
        # else:
        #     seed = 1
        if 'prevent_back_hop' in kwargs:
            pre_back_hop = kwargs['prevent_back_hop']
        else:
            pre_back_hop = False
        # if 'md_parallel' in kwargs:
        #     md_parallel = kwargs['md_parallel']
        # else:
        #     md_parallel = True
        if 'nthreads' in kwargs:
            nthreads = kwargs['nthreads']
        else:
            import joblib
            nthreads = joblib.cpu_count()
        if 'uq_tresholds' in kwargs:
            uq_tresholds = kwargs['uq_tresholds']
        if 'max_hopprobuq_points' in kwargs:
            N_probUQ = kwargs['max_hopprobuq_points']
        else:
            N_probUQ = 15
        if 'dump_trajectory_interval' in kwargs:
            dump_trajectory_interval = kwargs['dump_trajectory_interval']
        else:
            dump_trajectory_interval = None
        if 'reduce_kinetic_energy' in kwargs:
            reduce_kinetic_energy = kwargs['reduce_kinetic_energy']
        else:
            reduce_kinetic_energy = True
        if 'reduce_memory_usage' in kwargs:
            reduce_memory_usage = kwargs['reduce_memory_usage']
        else:
            reduce_memory_usage = False
        if 'time_step_gapMD' in kwargs:
            time_step_gapMD = kwargs['time_step_gapMD']
        else:
            time_step_gapMD = 0.5
        if 'time_max_gapMD' in kwargs:
            time_max_gapMD = kwargs['time_max_gapMD']
        else:
            time_max_gapMD = maximum_propagation_time
        if 'plot_populations' in kwargs:
            plot_populations = kwargs['plot_populations']
        else:
            plot_populations = False
        
        if 'prob_uq_threshold' in kwargs:
            prob_threshold = kwargs['prob_uq_threshold']
        else:
            prob_threshold = 0.1

        def run_gapMD(moldb4gap,imol4gap, lower_surface=0, current_surface=1):
            mol4gap = moldb4gap[imol4gap]
            init_Ekin = mol4gap.calculate_kinetic_energy()*random.random()
            dyn = gap_md.gap_md(
                model = gap_md.gap_model(model=ml_model, surface0=lower_surface, surface1=lower_surface+1, current_surface=current_surface, init_Ekin=init_Ekin),
                molecule_with_initial_conditions = mol4gap,
                ensemble='NVE',
                time_step=time_step_gapMD,
                maximum_propagation_time = time_max_gapMD,
                reduce_memory_usage = reduce_memory_usage,
                dump_trajectory_interval = dump_trajectory_interval,
                format = "h5md",
                filename = "gapMD_traj{}.h5".format(imol4gap),
                stop_function=stop_function, current_state=current_surface
            )
            traj = dyn.molecular_trajectory
            return traj
        def predict_probabilites(traj, time_step=0.5, prevent_back_hop=False):
            idx_list = []
            for idx, step in enumerate(traj.steps):
                if np.isnan(step.current_state):
                    idx_list.append(idx)
            traj_cut = data.molecular_trajectory()
            for idx, ele in enumerate(traj.steps): 
                if idx not in idx_list:
                    traj_cut.steps.append(ele)
            traj = traj_cut
            prob_list = []
            aux_prob_list =[]
            reduce_kinetic_energy_factor = 3* len(traj.steps[0].molecule.atoms) - 6
            def lz_prob(gap_per_stat):
                gap = gap_per_stat[1]
                gap_sotd = ((gap_per_stat[2] + gap_per_stat[0] - 2 * gap) / (time_step * constants.fs2au)**2)
                return np.exp((-np.pi/2.0) * np.sqrt(abs(gap)**3 / abs(gap_sotd)))  
            for index, step in enumerate(traj.steps):
                if index != 0 and index != len(traj.steps)-1:
                    hopping_probabilities = []
                    current_state = int(step.current_state)
                    for stat in range(len(step.molecule.electronic_states)):
                        gap_per_stat = []
                        if stat == current_state:
                            prob = -1.0      
                        else:
                            for iistep in [index-1, index, index+1]:
                                gap_per_stat.append(abs(traj.steps[iistep].molecule.electronic_states[current_state].energy
                                                    -traj.steps[iistep].molecule.electronic_states[stat].energy))
                            prob = lz_prob(gap_per_stat)
                        hopping_probabilities.append(prob)
                    prob_list.append(hopping_probabilities)
            for index, step in enumerate(traj.steps):
                if index != 0 and index != len(traj.steps)-1:
                    hopping_probabilities = []
                    current_state = int(step.current_state)
                    for stat in range(len(step.molecule.electronic_states)):
                        gap_per_stat = []
                        if stat == current_state:
                            prob = -1.0      
                        else:
                            for iistep in [index-1, index, index+1]:
                                gap_per_stat.append(abs(traj.steps[iistep].molecule.electronic_states[current_state].aux_energy
                                                    -traj.steps[iistep].molecule.electronic_states[stat].aux_energy))
                            prob = lz_prob(gap_per_stat)
                        hopping_probabilities.append(prob)
                    aux_prob_list.append(hopping_probabilities)
            return prob_list, aux_prob_list
            
        def predict_probabilites_gapMD(traj, current_state=1, time_step=0.1, prevent_back_hop=False):
            prob_list = []
            aux_prob_list =[]
            reduce_kinetic_energy_factor = 3* len(traj.steps[0].molecule.atoms) - 6
            def lz_prob(gap_per_stat):
                gap = gap_per_stat[1]
                gap_sotd = ((gap_per_stat[2] + gap_per_stat[0] - 2 * gap) / (time_step * constants.fs2au)**2)
                return np.exp((-np.pi/2.0) * np.sqrt(abs(gap)**3 / abs(gap_sotd)))  
            for index, step in enumerate(traj.steps):
                if index != 0 and index != len(traj.steps)-1:
                    hopping_probabilities = []
                    for stat in range(len(step.molecule.electronic_states)):
                        gap_per_stat = []
                        if stat == current_state:
                            prob = -1.0      
                        else:
                            for iistep in [index-1, index, index+1]:
                                gap_per_stat.append(abs(traj.steps[iistep].molecule.electronic_states[current_state].energy
                                                    -traj.steps[iistep].molecule.electronic_states[stat].energy))
                            prob = lz_prob(gap_per_stat)
                        hopping_probabilities.append(prob)
                    prob_list.append(hopping_probabilities)
            for index, step in enumerate(traj.steps):
                if index != 0 and index != len(traj.steps)-1:
                    hopping_probabilities = []
                    for stat in range(len(step.molecule.electronic_states)):
                        gap_per_stat = []
                        if stat == current_state:
                            prob = -1.0      
                        else:
                            for iistep in [index-1, index, index+1]:
                                gap_per_stat.append(abs(traj.steps[iistep].molecule.electronic_states[current_state].aux_energy
                                                    -traj.steps[iistep].molecule.electronic_states[stat].aux_energy))
                            prob = lz_prob(gap_per_stat)
                        hopping_probabilities.append(prob)
                    aux_prob_list.append(hopping_probabilities)
            return prob_list, aux_prob_list
            
                


        namd_kwargs = {
            'model': ml_model,
            #'model_predict_kwargs': model_predict_kwargs,
            'time_step': time_step, # fs
            'maximum_propagation_time': maximum_propagation_time,
            'dump_trajectory_interval': dump_trajectory_interval,
            'filename':"traj.h5",
            'hopping_algorithm': 'LZBL',
            'nstates': nstates,
            'reduce_kinetic_energy': reduce_kinetic_energy,
            'reduce_memory_usage': reduce_memory_usage,
            'initial_state': initstate,
            'prevent_back_hop':pre_back_hop,
            'rescale_velocity_direction':'along velocities',
            'stop_function':stop_function,
            }
        def sample_from_DB(db, N):
            if len(db) > N:
                frac = N/len(db)
                db1, db2 = db.split(fraction_of_points_in_splits=[frac,1-frac])
                return db1
            else:
                return db  
        #generate initial conditions from wigner
 

        def loop_alnamd():
            moldb = data.molecular_database()
            initial_molecular_database = generate_initial_conditions(molecule=eqmol,
                                                generation_method='wigner',
                                                number_of_initial_conditions=ntraj,
                                                initial_temperature=initial_temperature,
                                                use_hessian=False)
            for i in range(1,len(initial_molecular_database)+1):
                if os.path.exists("job_surface_hopping_md_{}".format(i)):
                    os.system("rm -r job_surface_hopping_md_{}".format(i))
            dyns = simulations.run_in_parallel(molecular_database=initial_molecular_database, task=namd.surface_hopping_md, task_kwargs=namd_kwargs, create_and_keep_temp_directories=True)
            sys.stdout.flush()
            if plot_populations:
                iteration = al_info['iteration']
                namd.plot_population_from_disk(ntraj=len(initial_molecular_database), time_step=time_step, 
                                    max_propagation_time=maximum_propagation_time, nstates=nstates, filename=f'Iteration{iteration}_pop.png', ref_pop_filename='ref.txt', pop_filename=f'Iteration{iteration}_pop.txt',dirname="job_surface_hopping_md_",traj_filename="traj.h5" )     
      
            
            itraj=0
            init_cond_for_gapMD = data.molecular_database()
            prob_db = data.molecular_database()
            stop_indices = []
            for i in range(1,len(initial_molecular_database)+1):
                traj = data.molecular_trajectory()
                traj.load("job_surface_hopping_md_{}/traj.h5".format(i), format="h5md")
                traj_stopped= False
                itraj+=1
                stop_index = -1
                print(f"Trajectory {itraj} number of steps: {len(traj.steps)}")
                for index, step in enumerate(traj.steps):
                    if step.molecule.uncertain==True:
                        traj_stopped = True
                        moldb.molecules.append(step.molecule)
                        print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, step.time))
                        stop_index = index
                        stop_indices.append(stop_index)
                        sample_index = random.randint(0, index)
                        init_cond_for_gapMD.append(traj.steps[sample_index].molecule)
                        print('Starting gapMD from trajectory %d at time %.2f fs' % (itraj, (sample_index)*time_step))
                        break
                        
                prob_list, prob_list_aux =  predict_probabilites(traj, time_step=time_step) 
                UQ_db = data.molecular_database()
                for i in range(len(prob_list)):
                    if i > stop_index -1 and stop_index !=-1:
                        break
                    diff = abs(max(prob_list[i])-max(prob_list_aux[i]))
                    if diff > prob_threshold:
                        UQ_db.molecules.append(traj.steps[i].molecule)
                        #print("probUQ diverged")
                
                if len(UQ_db) >0:
                    prob_db.append(random.choice(UQ_db))
                
                if not traj_stopped:
                    sample_index = random.randint(0, len(traj.steps)-1)
                    init_cond_for_gapMD.append(traj.steps[sample_index].molecule)
                    print('Starting gapMD from trajectory %d at time %.2f fs' % (itraj, sample_index*time_step))

            print("The surface stopping times are:")
            for i in stop_indices:
                print((i)*time_step)
            convergance_degree = (1 - len(stop_indices)/len(initial_molecular_database))*100
            print("The model is {} % converged".format(convergance_degree))
            if len(stop_indices) <= min_surfuq_points:
                print("The ML-NAMD trajectories are converged, AL-NAMD converged.")
                return data.molecular_database()

            #run gapMD trajs  
            print("Running gapMD trajs on lower surfaces")
            lower_surface_list = []
            for i in range(N_gapMD):
                lower_surface_list.append(random.randrange(0,nstates-1))
            init_cond_for_gapMD_down = sample_from_DB(init_cond_for_gapMD, N_gapMD)
            
            for i in range(N_gapMD):
                if os.path.exists("gapMD_traj{}.h5".format(i)):
                    os.system("rm gapMD_traj{}.h5".format(i))
            
            from joblib import Parallel, delayed 
            gapMD_trajs_down = Parallel(n_jobs=nthreads)(delayed(run_gapMD)(init_cond_for_gapMD_down,i, lower_surface=lower_surface_list[i], current_surface=lower_surface_list[i]) for i in range(len(init_cond_for_gapMD_down)))
            
            sys.stdout.flush()
            itraj=0 
            for i in range(int(N_gapMD)):
                gap_traj = data.molecular_trajectory()
                gap_traj.load("gapMD_traj{}.h5".format(i), format="h5md")
                itraj+=1
                stop_index = -1
                print(f"Trajectory {itraj} number of steps: {len(gap_traj.steps)}")
                for index, step in enumerate(gap_traj.steps):
                    if step.molecule.uncertain==True:
                        print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, step.time))
                        moldb.molecules.append(step.molecule)
                        stop_index = index
                        break
                prob_list, prob_list_aux =  predict_probabilites_gapMD(gap_traj, current_state=lower_surface_list[i]+1, time_step=time_step_gapMD) 
                UQ_db = data.molecular_database()
                for k in range(len(prob_list)):
                    if k > stop_index -1 and stop_index != -1:
                        break
                    diff = abs(max(prob_list[k])-max(prob_list_aux[k]))
                    if diff > prob_threshold:
                        UQ_db.molecules.append(gap_traj.steps[k].molecule)
                        #print('prob UQ diverged at gapMD trajectory %d at time %.2f fs' % (itraj, (i)*time_step))
                
                if len(UQ_db) >0:
                    prob_db.append(random.choice(UQ_db))
                    
                        
            print("Running gapMD trajs on upper surfaces")
            for i in range(int(N_gapMD)):
                if os.path.exists("gapMD_traj{}.h5".format(i)):
                    os.system("rm gapMD_traj{}.h5".format(i))
            
            init_cond_for_gapMD_up = sample_from_DB(init_cond_for_gapMD, N_gapMD)
            upper_surface_list = []
            for i in range(N_gapMD):
                upper_surface_list.append(random.randrange(1,nstates))
            from joblib import Parallel, delayed 
            gapMD_trajs_up = Parallel(n_jobs=nthreads)(delayed(run_gapMD)(init_cond_for_gapMD_up,i, lower_surface=upper_surface_list[i]-1, current_surface=upper_surface_list[i]) for i in range(len(init_cond_for_gapMD_up)))
            sys.stdout.flush()
            itraj=0 
            for i in range(N_gapMD):
                itraj+=1 
                gap_traj = data.molecular_trajectory()
                gap_traj.load("gapMD_traj{}.h5".format(i), format="h5md")
                print(f"Trajectory {itraj} number of steps: {len(gap_traj.steps)}")
                for index, step in enumerate(gap_traj.steps):
                    if step.molecule.uncertain==True:
                        print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, step.time))
                        moldb.molecules.append(step.molecule)
                        stop_index = index
                        break
                prob_list, prob_list_aux =  predict_probabilites_gapMD(gap_traj, current_state=0, time_step=time_step_gapMD) 
                UQ_db = data.molecular_database()
                for k in range(len(prob_list)):
                    if k > stop_index -1 and stop_index != -1:
                        break
                    diff = abs(max(prob_list[k])-max(prob_list_aux[k]))
                    if diff > prob_threshold:
                        UQ_db.molecules.append(gap_traj.steps[k].molecule)
                        #print('prob UQ diverged at gapMD trajectory %d at time %.2f fs' % (itraj, (i)*time_step))
                
                if len(UQ_db) >0:
                    prob_db.append(random.choice(UQ_db))

            if len(prob_db)>N_probUQ:
                print("Too many probUQ geometries, adding only " +str(N_probUQ))
            
            moldb.append(sample_from_DB(prob_db,N_probUQ))
            return moldb
        if max_points:
            moldb = data.molecular_database()
            while len(moldb) < max_points:
                sampled_db = loop_alnamd()
                if len(sampled_db) != 0:
                    moldb.append(sampled_db)
                else:
                    return data.molecular_database()
            return sample_from_DB(moldb, max_points)
        else:
            moldb = loop_alnamd()
            return moldb
class excess_energy_generator():

    def __init__(self,molecule,average_IEE_per_atom,impulsion_energy):
        self.molecule = molecule 
        self.average_IEE_per_atom = average_IEE_per_atom
        self.impulsion_energy = impulsion_energy
        self.number_of_valence_electrons = self.get_number_of_valence_electrons(self.molecule)

    def get_excess_energies(self,nsample):
        self.optimize_IEE_parameters()
        internal_excess_energies = []
        for isample in range(nsample):
            while True:
                iee = self.impulsion_energy * np.random.random() 
                rand = np.random.random() 
                if rand <= self.Poisson_type_IEE_distribution_function(iee,self.aa,self.bb,self.number_of_valence_electrons)/self.maximum_probability:
                    break 
            # Sampled internal excess energies are saved in Hartree
            internal_excess_energies.append(iee*constants.eV2hartree)
        
        return internal_excess_energies

    def optimize_IEE_parameters(self):
        if not 'aa' in self.__dict__ and not 'bb' in self.__dict__:
            target = len(self.molecule)*self.average_IEE_per_atom
            def func(aa):
                bb = 7*aa 
                aa = min(aa,0.3)
                return self.get_average_IEE(aa,bb,self.number_of_valence_electrons) - target 
            from scipy.optimize import fsolve
            self.aa = fsolve(func,[0.2])[0]
            self.bb = 7*self.aa
            self.aa = min(self.aa,0.3)

            self.maximum_probability = self.get_maximum_probability(self.aa,self.bb,self.number_of_valence_electrons)

            print(f"  Optimized IEE distribution function parameters: a={self.aa}, b={self.bb}")
            print(f"  Maximum probability: {self.maximum_probability}")
        else:
            print(f"  Current IEE distribution function parameters: a={self.aa}, b={self.bb}")
            print(f"  Maximum probability: {self.maximum_probability}")

    def get_average_IEE(self,aa,bb,Nel):
        energy_range = np.linspace(0.05,self.impulsion_energy,self.impulsion_energy*20)
        value = self.Poisson_type_IEE_distribution_function(energy_range,aa,bb,Nel)
        average_IEE = np.sum(energy_range*value) / np.sum(value)
        return average_IEE 
    
    def get_maximum_probability(self,aa,bb,Nel):
        energy_range = np.linspace(0.05,self.impulsion_energy,self.impulsion_energy*20)
        value = self.Poisson_type_IEE_distribution_function(energy_range,aa,bb,Nel)
        return np.max(value)
    
    def Poisson_type_IEE_distribution_function(self,E,aa,bb,Nel):
        '''
        Poisson_type probability to have an IEE of E in the ion

        Arguments:
            E (float): Internal excess energy (IEE), unit: eV
            aa (float): Parameter of IEE function, unit: eV
            bb (float): Parameter of IEE function
            Nel (int): Number of valence electrons

        '''

        cc = 1.0 / aa / Nel 
        
        return np.exp(cc*E*(1+np.log(bb/cc/E))-bb) / np.sqrt(aa*E+1)
    
    def get_number_of_valence_electrons(self,molecule):
        return np.sum(molecule.get_atomic_numbers())

class ml_model_trainer():
    def __init__(self,ml_model_type=None,hyperparameters={}):
        self.ml_model_type = ml_model_type 
        self.hyperparameters = hyperparameters
    
    def hyperparameters_setter(self):
        # if self.ml_model_type.casefold() == 'kreg':
        #     self.hyperparameters = {
        #         'sigma'
        #     }
        pass 

    # # Use holdout validation
    # def train(self,molecular_database,property_to_learn,xyz_derivative_property_to_learn):
    #     pass 

    def main_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False,**kwargs):
        if self.ml_model_type.casefold() == 'kreg':
            return self.main_model_trainer_kreg(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'ani':
            return self.main_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'msani':
            return self.main_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'mace':
            return self.main_model_trainer_mace(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)
         

    def aux_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        if self.ml_model_type.casefold() == 'kreg':
            return self.aux_model_trainer_kreg(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'ani':
            return self.aux_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'msani':
            return self.aux_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'mace':
            return self.aux_model_trainer_mace(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)
        pass

    def main_model_trainer_kreg(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False):
        print("Training the main KREG model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = models.kreg(model_file=filename,ml_program='KREG_API')
        else:
            main_model = models.kreg(model_file=filename,ml_program='KREG_API')
            main_model.hyperparameters['sigma'].minval = 2**-5
            # if not 'lmbd' in self.hyperparameters and not 'sigma' in self.hyperparameters:
            main_model.optimize_hyperparameters(subtraining_molecular_database=subtraining_molDB,
                                                validation_molecular_database=validation_molDB,
                                                optimization_algorithm='grid',
                                                hyperparameters=['lambda','sigma'],
                                                training_kwargs={'property_to_learn': property_to_learn, 'xyz_derivative_property_to_learn': xyz_derivative_property_to_learn, 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_'+property_to_learn, 'xyz_derivative_property_to_predict': 'estimated_'+xyz_derivative_property_to_learn},
                                                validation_loss_function=None)
            lmbd_ = main_model.hyperparameters['lambda'].value ; sigma_ = main_model.hyperparameters['sigma'].value
            self.hyperparameters['lambda'] = lmbd_; self.hyperparameters['sigma'] = sigma_
            print(f"Optimized hyperparameters for {property_to_learn} main model: lambda={lmbd_}, sigma={sigma_}")
            main_model.kreg_api.save_model(filename)
        return main_model 
    
    def aux_model_trainer_kreg(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False):
        print("Training the main auxiliary model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = models.kreg(model_file=filename,ml_program='KREG_API')
        else:
            aux_model = models.kreg(model_file=filename,ml_program='KREG_API')
            aux_model.hyperparameters['sigma'].minval = 2**-5
            aux_model.optimize_hyperparameters(subtraining_molecular_database=subtraining_molDB,
                                                validation_molecular_database=validation_molDB,
                                            optimization_algorithm='grid',
                                            hyperparameters=['lambda', 'sigma'],
                                            training_kwargs={'property_to_learn': property_to_learn, 'prior': 'mean'},
                                            prediction_kwargs={'property_to_predict': 'estimated'+property_to_learn})
            lmbd_ = aux_model.hyperparameters['lambda'].value ; sigma_ = aux_model.hyperparameters['sigma'].value
            self.hyperparameters['aux_lambda'] = lmbd_ ; self.hyperparameters['sigma'] = sigma_
            print(f"Optimized hyperparameters for {property_to_learn} aux model: lambda={lmbd_}, sigma={sigma_}")
            aux_model.kreg_api.save_model(filename)
        return aux_model

    def main_model_trainer_ani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the main ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = models.ani(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                main_model = models.ani(model_file=filename+'.tmp',device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                main_model = models.ani(model_file=filename,device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        return main_model 
    
    def aux_model_trainer_ani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the auxiliary ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = models.ani(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                aux_model = models.ani(model_file=filename+'.tmp',device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                aux_model = models.ani(model_file=filename,device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
        return aux_model 
    
    def main_model_trainer_mace(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the main MACE model")
        os.system("rm -rf MACE_*")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = models.mace(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                main_model = models.mace(model_file=filename+'.tmp',device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                main_model = models.mace(model_file=filename,device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        return main_model 
    
    def aux_model_trainer_mace(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the auxiliary MACE model")
        os.system("rm -rf MACE_*")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = models.mace(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                aux_model = models.mace(model_file=filename+'.tmp',device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                aux_model = models.mace(model_file=filename,device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
        return aux_model
    def main_model_trainer_msani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the main MS-ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = models.ani(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                main_model = models.msani(model_file=filename+'.tmp',device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                main_model = models.ani(model_file=filename,device=device,verbose=False)
                main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        return main_model 
    
    def aux_model_trainer_msani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        print("Training the auxiliary MS-ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = models.ani(model_file=filename,device=device,verbose=False)
        else:
            if reuse_model:
                if not os.path.exists(filename+'.tmp'):
                    stopper(f"Temp model file {filename+'.tmp'} not found")
                aux_model = models.ani(model_file=filename+'.tmp',device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
                os.system(f"mv {filename+'.tmp'} {filename}")
            else:
                aux_model = models.ani(model_file=filename,device=device,verbose=False)
                aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
        return aux_model 
class delta_ml_model_trainer(ml_model_trainer):
    def __init__(self,ml_model_type=None,hyperparameters={},baseline=None,target=None):
        super().__init__(ml_model_type=ml_model_type,hyperparameters=hyperparameters)
        self.baseline = baseline 
        self.target = target

    def main_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model=False,**kwargs):
        # for mol in subtraining_molDB:
        #     mol.__dict__[property_to_learn] = mol.__dict__[self.target].__dict__[property_to_learn] - mol.__dict__[self.baseline].__dict__[property_to_learn]
        #     mol.update_xyz_vectorial_properties(xyz_derivative_property_to_learn,mol.__dict__[self.target].__dict__[xyz_derivative_property_to_learn]-mol.__dict__[self.baseline].__dict__[xyz_derivative_property_to_learn])
        # for mol in validation_molDB:
        #     mol.__dict__[property_to_learn] = mol.__dict__[self.target].__dict__[property_to_learn] - mol.__dict__[self.baseline].__dict__[property_to_learn]
        #     mol.update_xyz_vectorial_properties(xyz_derivative_property_to_learn,mol.__dict__[self.target].__dict__[xyz_derivative_property_to_learn]-mol.__dict__[self.baseline].__dict__[xyz_derivative_property_to_learn])
        return super().main_model_trainer(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)

    def aux_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        # for mol in subtraining_molDB:
        #     mol.__dict__[property_to_learn] = mol.__dict__[self.target].__dict__[property_to_learn] - mol.__dict__[self.baseline].__dict__[property_to_learn]
            
        # for mol in validation_molDB:
        #     mol.__dict__[property_to_learn] = mol.__dict__[self.target].__dict__[property_to_learn] - mol.__dict__[self.baseline].__dict__[property_to_learn]
            
        return super().aux_model_trainer(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)

class ml_model(models.ml_model):
    def __init__(self,al_info={},model_file=None,device=None,verbose=False,ml_model_type='ANI',**kwargs):
        if device is None:
            # to-do: that should be done just for the models using torch, otherwise loading takes time
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ml_model_type = ml_model_type
        if model_file is None:
            if 'mlmodel_file' in al_info.keys():
                self.model_file = al_info['mlmodel_file']
            else:
                self.model_file = 'mlmodel'
                al_info['mlmodel_file'] = self.model_file
        else:
            self.model_file = model_file
            al_info['mlmodel_file'] = self.model_file
        if 'main_mlmodel_file' in al_info.keys():
            main_mlmodel_file = al_info['main_mlmodel_file']
        else:
            if self.ml_model_type.casefold() == 'kreg':
                main_mlmodel_file = f'{self.model_file}.npz'
            else:
                main_mlmodel_file = f'{self.model_file}.pt'
            al_info['main_mlmodel_file'] = main_mlmodel_file
        if 'aux_mlmodel_file' in al_info.keys():
            aux_mlmodel_file = al_info['aux_mlmodel_file']
        else:
            if self.ml_model_type.casefold() == 'kreg':
                aux_mlmodel_file = f'aux_{self.model_file}.npz'
            else:
                aux_mlmodel_file = f'aux_{self.model_file}.pt'
            al_info['aux_mlmodel_file'] = aux_mlmodel_file
        self.device = device
        self.verbose = verbose
        
        self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=main_mlmodel_file,
                                                device=self.device,
                                                verbose=self.verbose)
        self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=aux_mlmodel_file,
                                               device=self.device,
                                               verbose=self.verbose)
        # if ml_model_type.casefold() == 'ani':
        #     self.main_model = models.ani(model_file=main_mlmodel_file,device=device,verbose=verbose)
        #     self.aux_model = models.ani(model_file=aux_mlmodel_file,device=device,verbose=verbose)
        # elif ml_model_type.casefold() == 'kreg':
        #     self.main_model = models.kreg(model_file=main_mlmodel_file,ml_program='KREG_API')
        #     self.aux_model = models.kreg(model_file=aux_mlmodel_file,ml_program='KREG_API')
        # elif ml_model_type.casefold() == 'mace':
        #     self.main_model = models.mace(model_file=main_mlmodel_file,device=device,verbose=verbose)
        #     self.aux_model = models.mace(model_file=aux_mlmodel_file,device=device,verbose=verbose)
        self.uq_threshold = None

    def train(self,molecular_database=None, al_info={}):
        if 'working_directory' in al_info.keys():
            workdir = al_info['working_directory']
            if self.ml_model_type.casefold() == 'kreg':
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.npz")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.npz")
            else:
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.pt")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.pt")
            
            
        else:
            workdir='.'
        validation_set_fraction = 0.1
        if not os.path.exists(os.path.join(workdir,'training_db.json')):
            [subtraindb, valdb] = molecular_database.split(number_of_splits=2, fraction_of_points_in_splits=[1-validation_set_fraction, validation_set_fraction], sampling='random')
            trainingdb = subtraindb+valdb
            trainingdb.dump(os.path.join(workdir,'training_db.json'),format='json')
            valdb.dump("valdb.json", format="json")
        else:
            trainingdb = data.molecular_database.load(filename=os.path.join(workdir,'training_db.json'),format='json')
            Nsubtrain = int(len(trainingdb)*0.9)
            subtraindb = trainingdb[:Nsubtrain]
            valdb = trainingdb[Nsubtrain:]


        # train the model on energies and gradients
        if not os.path.exists(self.main_model.model_file):
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=self.main_model.model_file,
                                                device=self.device,
                                                verbose=self.verbose)
            # self.main_model = models.ani(
            #     model_file=self.main_model.model_file,
            #     device=self.device,
            #     verbose=self.verbose
            # )
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.main_model,
                subtraindb=subtraindb,
                valdb=valdb,
            )
            # self.main_model.train(
            #     molecular_database=subtraindb,
            #     validation_molecular_database=valdb,
            #     property_to_learn='energy',
            #     xyz_derivative_property_to_learn='energy_gradients'
            # )
        else:
            print(f"Model file {self.main_model.model_file} found, skip training")
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=self.main_model.model_file,
                                                device=self.device,
                                                verbose=self.verbose)

        # train the auxiliary model only on energies
        if not os.path.exists(self.aux_model.model_file):
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)
            # self.aux_model = models.ani(
            #     model_file=self.aux_model.model_file,
            #     device=self.device,
            #     verbose=self.verbose
            # )
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.aux_model,
                subtraindb=subtraindb,
                valdb=valdb,
                en_only=True,
            )
            # self.aux_model.train(
            #     molecular_database=subtraindb,
            #     validation_molecular_database=valdb,
            #     property_to_learn='energy'
            # )
        else:
            print(f"Model file {self.aux_model.model_file} found, skip training")
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)

        if not 'uq_threshold' in al_info.keys():
            self.predict(molecular_database=valdb)
            uqs = valdb.get_properties('uq')
            al_info['uq_threshold'] = self.threshold_metric(uqs,metric='m+3mad')
            print(f"New threshold: {al_info['uq_threshold']}")
        else:
            print(f"Current threshold: {al_info['uq_threshold']}")
        self.uq_threshold = al_info['uq_threshold']

        # if the models were trained successfully, let's update al info where we can find them
        al_info['main_mlmodel_file'] = self.main_model.model_file
        al_info['aux_mlmodel_file'] = self.aux_model.model_file

        self.summary(subtraindb=subtraindb,valdb=valdb,workdir=workdir)

    def predict(self,molecule=None,molecular_database=None,**kwargs):
        if not molecule is None:
            molecular_database = data.molecular_database(molecule)
        else:
            if molecular_database is None:
                stopper("Please provide molecule or molecular database")
        self.main_model.predict(molecular_database=molecular_database,property_to_predict='energy',xyz_derivative_property_to_predict='energy_gradients')
        self.aux_model.predict(molecular_database=molecular_database,property_to_predict='aux_energy',xyz_derivative_property_to_predict='aux_energy_gradients') 

        # Calculate uncertainties
        for mol in molecular_database:
            mol.uq = abs(mol.energy-mol.aux_energy)
            if not self.uq_threshold is None:
                if mol.uq > self.uq_threshold:
                    mol.uncertain = True 
                else:
                    mol.uncertain = False

    @property 
    def nthreads(self):
        return self.main_model.nthreads 

    @nthreads.setter 
    def nthreads(self,value):
        self.main_model.nthreads = value 
        self.aux_model.nthreads = value   

    def threshold_metric(self,absdevs,metric):
        '''
        Function that calculate thresholds

        Arguments:
            absdevs (List[float]): List of absolute deviations 
            metric (str): Threshold metric
        '''
        if metric.casefold() == 'max'.casefold():
            return np.max(absdevs)
        elif metric.casefold() =='M+3MAD'.casefold():
            if len(absdevs) >= 2:
                return np.median(absdevs) + 3*stats.calc_median_absolute_deviation(absdevs) 
            else:
                return 0.0
    
    
    def summary(self,subtraindb,valdb,workdir):
        Nsubtrain = len(subtraindb)
        Nvalidate = len(valdb)
        Ntrain = Nsubtrain + Nvalidate
        trainingdb_ref = subtraindb + valdb 
        trainingdb = trainingdb_ref.copy()
        self.predict(molecular_database=trainingdb)
        print(f'    Number of training points: {Nsubtrain+Nvalidate}')
        print(f'        Number of subtraining points: {Nsubtrain}')
        print(f'        Number of validation points: {Nvalidate}')

        values = trainingdb_ref.get_properties('energy')
        estimated_values = trainingdb.get_properties('energy')
        aux_estimated_values = trainingdb.get_properties('aux_energy')

        gradients = trainingdb_ref.get_xyz_vectorial_properties('energy_gradients')
        estimated_gradients = trainingdb.get_xyz_vectorial_properties('energy_gradients')

        # Evaluate main model performance 
        # .RMSE of values
        main_model_subtrain_vRMSE = stats.rmse(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vRMSE = stats.rmse(estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        main_model_subtrain_vPCC = stats.correlation_coefficient(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vPCC = stats.correlation_coefficient(estimated_values[Nsubtrain:],values[Nsubtrain:])
        try:
            main_model_subtrain_gRMSE = stats.rmse(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gRMSE = stats.rmse(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gRMSE = 'not calculated'
            main_model_validate_gRMSE = 'not calculated'
        # .Pearson correlation coeffcient of gradients 
        try:
            main_model_subtrain_gPCC = stats.correlation_coefficient(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gPCC = stats.correlation_coefficient(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gPCC = 'not calculated'
            main_model_validate_gPCC = 'not calculated'
        # Evaluate auxiliary model performance
        # .RMSE of values 
        aux_model_subtrain_vRMSE = stats.rmse(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vRMSE = stats.rmse(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        aux_model_subtrain_vPCC = stats.correlation_coefficient(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vPCC = stats.correlation_coefficient(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])

        print("        Main model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {main_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_vPCC}")
        print(f"                RMSE of gradients = {main_model_subtrain_gRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_gPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {main_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_vPCC}")
        print(f"                RMSE of gradients = {main_model_validate_gRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_gPCC}")
        print("        Auxiliary model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {aux_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_subtrain_vPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {aux_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_validate_vPCC}")

        # Value scatter plot of the main model
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(estimated_values)]),max([max(values),max(estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        plt.suptitle(f'Main model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'mlmodel_energies.png'),dpi=300)
        fig.clear()
        # Gradient scatter plot of the main model 
        try:
            fig,ax = plt.subplots()
            fig.set_size_inches(15,12)
            diagonal_line = [min([np.min(gradients),np.min(estimated_gradients)]),max([np.max(gradients),np.max(estimated_gradients)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(gradients[0:Nsubtrain].flatten(),estimated_gradients[0:Nsubtrain].flatten(),color='C0',label='subtraining points')
            ax.scatter(gradients[Nsubtrain:Ntrain].flatten(),estimated_gradients[Nsubtrain:Ntrain].flatten(),color='C1',label='validation points')
            ax.set_xlabel(f'Energy gradients (Hartree/Angstrom)')
            ax.set_ylabel(f'Estimated energy gradients (Hartree/Angstrom)')
            ax.set_title(f'Main model (energy gradients)')
            plt.legend()
            plt.savefig(os.path.join(workdir,'mlmodel_energy_gradients.png'),dpi=300)
            fig.clear()
        except:
            print('Cannot plot gradients plot of main model')

        # Value scatter plot of the auxiliary model
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(aux_estimated_values)]),max([max(values),max(aux_estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],aux_estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],aux_estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        ax.set_title(f'Auxiliary model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'aux_mlmodel_energies.png'),dpi=300)

    def initialize_model(self,ml_model_type,model_file,device,verbose):
        if ml_model_type.casefold() == 'ani':
            model = models.ani(model_file=model_file,
                               device=device,
                               verbose=verbose)
        elif ml_model_type.casefold() == 'kreg':
            model = models.kreg(model_file=model_file,
                                ml_program='KREG_API')
        elif ml_model_type.casefold() == 'mace':
            model = models.mace(model_file=model_file,
                                device=device,
                                verbose=verbose)
        return model
    
    def model_trainer(self,ml_model_type,model,subtraindb,valdb,en_only=False):
        # Train aux model
        if en_only:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb,
                                                    validation_molecular_database=valdb,
                                                optimization_algorithm='grid',
                                                hyperparameters=['lambda', 'sigma'],
                                                training_kwargs={'property_to_learn': 'energy', 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_energy'})
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                # self.hyperparameters['aux_lambda'] = lmbd_ ; self.hyperparameters['sigma'] = sigma_
                print(f"Optimized hyperparameters for aux model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy'
                )
        # Train main model
        else:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn='energy_gradients'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb,
                                                    validation_molecular_database=valdb,
                                                    optimization_algorithm='grid',
                                                    hyperparameters=['lambda','sigma'],
                                                    training_kwargs={'property_to_learn': 'energy', 'xyz_derivative_property_to_learn': 'energy_gradients', 'prior': 'mean'},
                                                    prediction_kwargs={'property_to_predict': 'estimated_energy', 'xyz_derivative_property_to_predict': 'estimated_energy_gradients'},
                                                    validation_loss_function=None)
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                # self.hyperparameters['lambda'] = lmbd_; self.hyperparameters['sigma'] = sigma_
                print(f"Optimized hyperparameters for main model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn='energy_gradients'
                )
class ml_model_msani(models.ml_model):
    def __init__(self,ml_model_type = 'msani',al_info={},model_file=None,device=None,verbose=False, **kwargs):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ml_model_type = "msani"
        if model_file is None:
            if 'mlmodel_file' in al_info.keys():
                self.model_file = al_info['mlmodel_file']
            else:
                self.model_file = 'mlmodel'
                al_info['mlmodel_file'] = self.model_file
        else:
            self.model_file = model_file
            al_info['mlmodel_file'] = self.model_file
        if 'main_mlmodel_file' in al_info.keys():
            main_mlmodel_file = al_info['main_mlmodel_file']
        else:
            main_mlmodel_file = f'{self.model_file}.pt'
            al_info['main_mlmodel_file'] = main_mlmodel_file
        if 'aux_mlmodel_file' in al_info.keys():
            aux_mlmodel_file = al_info['aux_mlmodel_file']
        else:
            aux_mlmodel_file = f'aux_{self.model_file}.pt'
            al_info['aux_mlmodel_file'] = aux_mlmodel_file
        if 'nstates' in kwargs:
            self.nstates = kwargs['nstates']
        else:
            self.nstates = 1
        if 'gap_weight' in kwargs:
            self.hyperparameters = {}
            self.hyperparameters["gap coefficient"] = kwargs['gap_weight']
        else:
            self.hyperparameters = {}
            self.hyperparameters["gap coefficient"] = 1.0

        self.device = device
        self.verbose = verbose

        self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=main_mlmodel_file,
                                                device=self.device,
                                                verbose=self.verbose)
        self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=aux_mlmodel_file,
                                               device=self.device,
                                               verbose=self.verbose)
        self.uq_thresholds = None

    def train(self,molecular_database=None, al_info={}):
        if 'working_directory' in al_info.keys():
            workdir = al_info['working_directory']
            if self.ml_model_type.casefold() == 'kreg':
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.npz")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.npz")
            else:
                self.main_model.model_file = os.path.join(workdir,f"{self.model_file}.pt")
                self.aux_model.model_file = os.path.join(workdir,f"aux_{self.model_file}.pt")            
            
        else:
            workdir='.'
        validation_set_fraction = 0.1
        if not os.path.exists(os.path.join(workdir,'training_db.json')):
            [subtraindb, valdb] = molecular_database.split(number_of_splits=2, fraction_of_points_in_splits=[1-validation_set_fraction, validation_set_fraction], sampling='random')
            trainingdb = subtraindb+valdb
            trainingdb.dump(os.path.join(workdir,'training_db.json'),format='json')
        else:
            trainingdb = data.molecular_database.load(filename=os.path.join(workdir,'training_db.json'),format='json')
            Nsubtrain = int(len(trainingdb)*0.9)
            subtraindb = trainingdb[:Nsubtrain]
            valdb = trainingdb[Nsubtrain:]
        # train the model on energies and gradients
        if not os.path.exists(self.main_model.model_file):
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=self.main_model.model_file,
                                                device=self.device,
                                                verbose=self.verbose)
            # self.main_model = models.ani(
            #     model_file=self.main_model.model_file,
            #     device=self.device,
            #     verbose=self.verbose
            # )
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.main_model,
                subtraindb=subtraindb,
                valdb=valdb,
            )
            # self.main_model.train(
            #     molecular_database=subtraindb,
            #     validation_molecular_database=valdb,
            #     property_to_learn='energy',
            #     xyz_derivative_property_to_learn='energy_gradients'
            # )
        else:
            print(f"Model file {self.main_model.model_file} found, skip training")
            self.main_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                                model_file=self.main_model.model_file,
                                                device=self.device,
                                                verbose=self.verbose)

        # train the auxiliary model only on energies
        if not os.path.exists(self.aux_model.model_file):
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)
            # self.aux_model = models.ani(
            #     model_file=self.aux_model.model_file,
            #     device=self.device,
            #     verbose=self.verbose
            # )
            self.model_trainer(
                ml_model_type=self.ml_model_type,
                model=self.aux_model,
                subtraindb=subtraindb,
                valdb=valdb,
                en_only=True,
            )
            # self.aux_model.train(
            #     molecular_database=subtraindb,
            #     validation_molecular_database=valdb,
            #     property_to_learn='energy'
            # )
        else:
            print(f"Model file {self.aux_model.model_file} found, skip training")
            self.aux_model = self.initialize_model(ml_model_type=self.ml_model_type,
                                               model_file=self.aux_model.model_file,
                                               device=self.device,
                                               verbose=self.verbose)

        if not 'uq_thresholds' in al_info.keys():
            self.predict(molecular_database=valdb,nstates=self.nstates)
            uq_thresholds =[]
            for istate in range(self.nstates):
                uqs = valdb.get_properties('uq_state{}'.format(istate))
                uq_thresholds.append(self.threshold_metric(uqs,metric='m+3mad'))
                print(f"New threshold: {uq_thresholds[istate]}")
            al_info['uq_thresholds'] = uq_thresholds 
        else:
            for i in range(self.nstates):
                print(f"Current threshold for state {i}: {al_info['uq_thresholds'][i]}")
        self.uq_thresholds = al_info['uq_thresholds']

        # if the models were trained successfully, let's update al info where we can find them
        al_info['main_mlmodel_file'] = self.main_model.model_file
        al_info['aux_mlmodel_file'] = self.aux_model.model_file

        self.summary(subtraindb=subtraindb,valdb=valdb,workdir=workdir)

    def predict(self,molecule=None,molecular_database=None,calculate_energy=True,calculate_energy_gradients=True, nstates=1, current_state=0, **kwargs):
        if not molecule is None:
            molecular_database = data.molecular_database(molecule)
        else:
            if molecular_database is None:
                stopper("Please provide molecule or molecular database")
        self.aux_model.predict(molecular_database=molecular_database,property_to_predict='aux_energy',xyz_derivative_property_to_predict='aux_energy_gradients',nstates=nstates, current_state=current_state, calculate_energy=False)
        self.main_model.predict(molecular_database=molecular_database,property_to_predict='energy',xyz_derivative_property_to_predict='energy_gradients',nstates=nstates, current_state=current_state, calculate_energy=False)


        for mol in molecular_database:
            mol_copy = mol.copy()
            mol_copy.electronic_states = []
            for _ in range(nstates - len(mol.electronic_states)):
                mol.electronic_states.append(mol_copy.copy())
            for istat in range(0, nstates):
                aux_energy_str = 'mol.aux_energy_state{}'.format(istat)
                exec('mol.electronic_states[istat].aux_energy = '+aux_energy_str)
                main_energy_str = 'mol.energy_state{}'.format(istat)
                exec('mol.electronic_states[istat].energy = '+main_energy_str)
                mol_state_energy_gradients = "energy_gradients_state{}".format(istat)
                exec("import numpy as np; mol.electronic_states[istat].add_xyz_derivative_property(np.array([mol.atoms[i]." 
                         + mol_state_energy_gradients + " for i in range(len(mol.atoms))]).astype(float), 'energy', 'energy_gradients')", locals())
            mol.energy = mol.electronic_states[current_state].energy
            mol.add_xyz_derivative_property(np.array(mol.electronic_states[current_state].get_energy_gradients()).astype(float), 'energy', 'energy_gradients')                 
        # Calculate uncertainties
            mol.uq = []
            for i in range(nstates):
                mol.uq.append(abs(mol.electronic_states[i].energy - mol.electronic_states[i].aux_energy))
                exec('mol.uq_state{} = mol.uq[i]'.format(i))
            
            if not self.uq_thresholds is None:
                check_flag = False
                istate = int(current_state)
                if mol.uq[istate] > self.uq_thresholds[istate]:
                    check_flag = True
                    #debug 
                    #print("mol is uncertain due to current surface")
                if nstates != 1:
                    if istate == nstates - 1:
                        surface_for_check = [self.nstates - 1 - 1]
                    elif istate == 0:
                        surface_for_check = [1]
                    else:
                        surface_for_check = [istate+1,istate-1]
                    for jstate in surface_for_check:
                        if mol.uq[jstate] > self.uq_thresholds[jstate]:
                            check_flag = True
                            #debug 
                            #print("mol is uncertain due to adjecent surface"+str(jstate))
                if check_flag:
                    mol.uncertain = True
                else:
                    mol.uncertain = False
    @property 
    def nthreads(self):
        return self.main_model.nthreads 

    @nthreads.setter 
    def nthreads(self,value):
        self.main_model.nthreads = value 
        self.aux_model.nthreads = value   

    def threshold_metric(self,absdevs,metric):
        '''
        Function that calculate thresholds

        Arguments:
            absdevs (List[float]): List of absolute deviations 
            metric (str): Threshold metric
        '''
        if metric.casefold() == 'max'.casefold():
            return np.max(absdevs)
        elif metric.casefold() =='M+3MAD'.casefold():
            if len(absdevs) >= 2:
                return np.median(absdevs) + 3*stats.calc_median_absolute_deviation(absdevs) 
            else:
                return 0.0
    
    
    def summary(self,subtraindb,valdb,workdir):
        Nsubtrain = len(subtraindb)
        Nvalidate = len(valdb)
        Ntrain = Nsubtrain + Nvalidate
        trainingdb_ref = subtraindb + valdb 
        trainingdb = trainingdb_ref.copy()
        self.predict(molecular_database=trainingdb,nstates=self.nstates)
        print(f'    Number of training points: {Nsubtrain+Nvalidate}')
        print(f'        Number of subtraining points: {Nsubtrain}')
        print(f'        Number of validation points: {Nvalidate}')

        values = trainingdb_ref.get_properties('energy')
        estimated_values = trainingdb.get_properties('energy')
        aux_estimated_values = trainingdb.get_properties('aux_energy_state0')

        gradients = trainingdb_ref.get_xyz_vectorial_properties('energy_gradients')
        estimated_gradients = trainingdb.get_xyz_vectorial_properties('energy_gradients')

        # Evaluate main model performance 
        # .RMSE of values
        main_model_subtrain_vRMSE = stats.rmse(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vRMSE = stats.rmse(estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        main_model_subtrain_vPCC = stats.correlation_coefficient(estimated_values[:Nsubtrain],values[:Nsubtrain])
        main_model_validate_vPCC = stats.correlation_coefficient(estimated_values[Nsubtrain:],values[Nsubtrain:])
        try:
            main_model_subtrain_gRMSE = stats.rmse(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gRMSE = stats.rmse(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gRMSE = 'not calculated'
            main_model_validate_gRMSE = 'not calculated'
        # .Pearson correlation coeffcient of gradients 
        try:
            main_model_subtrain_gPCC = stats.correlation_coefficient(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
            main_model_validate_gPCC = stats.correlation_coefficient(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
        except:
            main_model_subtrain_gPCC = 'not calculated'
            main_model_validate_gPCC = 'not calculated'
        # Evaluate auxiliary model performance
        # .RMSE of values 
        aux_model_subtrain_vRMSE = stats.rmse(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vRMSE = stats.rmse(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])
        # .Pearson correlation coefficient of values
        aux_model_subtrain_vPCC = stats.correlation_coefficient(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
        aux_model_validate_vPCC = stats.correlation_coefficient(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])

        print("        Main model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {main_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_vPCC}")
        print(f"                RMSE of gradients = {main_model_subtrain_gRMSE}")
        print(f"                Correlation coefficient = {main_model_subtrain_gPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {main_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_vPCC}")
        print(f"                RMSE of gradients = {main_model_validate_gRMSE}")
        print(f"                Correlation coefficient = {main_model_validate_gPCC}")
        print("        Auxiliary model")
        print("            Subtraining set:")
        print(f"                RMSE of values = {aux_model_subtrain_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_subtrain_vPCC}")
        print("            Validation set:")
        print(f"                RMSE of values = {aux_model_validate_vRMSE}")
        print(f"                Correlation coefficient = {aux_model_validate_vPCC}")

        # Value scatter plot of the main model
        import matplotlib.pyplot as plt 
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(estimated_values)]),max([max(values),max(estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        plt.suptitle(f'Main model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'mlmodel_energies.png'),dpi=300)
        fig.clear()
        # Gradient scatter plot of the main model 
        try:
            fig,ax = plt.subplots()
            fig.set_size_inches(15,12)
            diagonal_line = [min([np.min(gradients),np.min(estimated_gradients)]),max([np.max(gradients),np.max(estimated_gradients)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(gradients[0:Nsubtrain].flatten(),estimated_gradients[0:Nsubtrain].flatten(),color='C0',label='subtraining points')
            ax.scatter(gradients[Nsubtrain:Ntrain].flatten(),estimated_gradients[Nsubtrain:Ntrain].flatten(),color='C1',label='validation points')
            ax.set_xlabel(f'Energy gradients (Hartree/Angstrom)')
            ax.set_ylabel(f'Estimated energy gradients (Hartree/Angstrom)')
            ax.set_title(f'Main model (energy gradients)')
            plt.legend()
            plt.savefig(os.path.join(workdir,'mlmodel_energy_gradients.png'),dpi=300)
            fig.clear()
        except:
            print('Cannot plot gradients plot of main model')

        # Value scatter plot of the auxiliary model
        fig,ax = plt.subplots() 
        fig.set_size_inches(15,12)
        diagonal_line = [min([min(values),min(aux_estimated_values)]),max([max(values),max(aux_estimated_values)])]
        ax.plot(diagonal_line,diagonal_line,color='C3')
        ax.scatter(values[0:Nsubtrain],aux_estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
        ax.scatter(values[Nsubtrain:Ntrain],aux_estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
        ax.set_xlabel(f'Energy (Hartree)')
        ax.set_ylabel(f'Estimated energy (Hartree)')
        ax.set_title(f'Auxiliary model (energies)')
        plt.legend()
        plt.savefig(os.path.join(workdir,'aux_mlmodel_energies.png'),dpi=300)

    def initialize_model(self,ml_model_type,model_file,device,verbose):
        if ml_model_type.casefold() == 'ani':
            model = models.ani(model_file=model_file,
                               device=device,
                               verbose=verbose)
        elif ml_model_type.casefold() == 'kreg':
            model = models.kreg(model_file=model_file,
                                ml_program='KREG_API')
        elif ml_model_type.casefold() == 'mace':
            model = models.mace(model_file=model_file,
                                device=device,
                                verbose=verbose)
        elif ml_model_type.casefold() == 'msani':
            model = models.msani(model_file=model_file,
                                device=device,
                                verbose=verbose, 
                                nstates = self.nstates)
        return model
    
    def model_trainer(self,ml_model_type,model,subtraindb,valdb,en_only=False):
        # Train aux model
        if en_only:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb,
                                                    validation_molecular_database=valdb,
                                                optimization_algorithm='grid',
                                                hyperparameters=['lambda', 'sigma'],
                                                training_kwargs={'property_to_learn': 'energy', 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_energy'})
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                # self.hyperparameters['aux_lambda'] = lmbd_ ; self.hyperparameters['sigma'] = sigma_
                print(f"Optimized hyperparameters for aux model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy'
                )
            elif ml_model_type.casefold() == 'msani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    hyperparameters = self.hyperparameters
                )
        # Train main model
        else:
            if ml_model_type.casefold() == 'ani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn='energy_gradients'
                )
            elif ml_model_type.casefold() == 'kreg':
                model_file_saved = model.model_file
                model.model_file = 'mlmodel.npz'
                model.hyperparameters['sigma'].minval = 2**-5
                model.optimize_hyperparameters(subtraining_molecular_database=subtraindb,
                                                    validation_molecular_database=valdb,
                                                    optimization_algorithm='grid',
                                                    hyperparameters=['lambda','sigma'],
                                                    training_kwargs={'property_to_learn': 'energy', 'xyz_derivative_property_to_learn': 'energy_gradients', 'prior': 'mean'},
                                                    prediction_kwargs={'property_to_predict': 'estimated_energy', 'xyz_derivative_property_to_predict': 'estimated_energy_gradients'},
                                                    validation_loss_function=None)
                lmbd_ = model.hyperparameters['lambda'].value ; sigma_ = model.hyperparameters['sigma'].value
                # self.hyperparameters['lambda'] = lmbd_; self.hyperparameters['sigma'] = sigma_
                print(f"Optimized hyperparameters for main model: lambda={lmbd_}, sigma={sigma_}")
                model.model_file = model_file_saved
                model.kreg_api.save_model(model.model_file)
            elif ml_model_type.casefold() == 'mace':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn='energy_gradients'
                )
            elif ml_model_type.casefold() == 'msani':
                model.train(
                    molecular_database=subtraindb,
                    validation_molecular_database=valdb,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn='energy_gradients',
                    hyperparameters = self.hyperparameters
                )
def stop_function_deprecated(mol,properties,thresholds,bonds=[]):
    stop = False 

    # # Check bond lengths
    # dist_matrix = mol.get_internuclear_distance_matrix()
    # for bond in bonds:
    #     ii = bond[0] ; jj = bond[1]
    #     ian = mol.atoms[ii].atomic_number ; jan = mol.atoms[jj].atomic_number
    #     dist = dist_matrix[ii][jj]
    #     if (ian == 1 and (jan > 1 and jan < 10)) or (jan == 1 and (ian > 1 and ian < 10)):
    #         if dist > 1.5: stop = True
    #     if (ian > 1 and ian < 10) and (jan > 1 and jan < 10):
    #         if dist > 1.8: stop = True
    # # prevent too short bond lengths too
    # for ii in range(len(mol.atoms)):
    #     for jj in range(ii+1, len(mol.atoms)):
    #         ian = mol.atoms[ii].atomic_number ; jan = mol.atoms[jj].atomic_number
    #         dist = dist_matrix[ii][jj]
    #         if ian == 1 and jan == 1 and dist < 0.6: stop = True
    #         elif ((ian == 1 and (jan > 1 and jan < 10)) or (jan == 1 and (ian > 1 and ian < 10))) and dist < 0.85: stop = True
    #         if (ian > 1 and ian < 10) and (jan > 1 and jan < 10) and dist < 1.1: stop = True        
    # if stop: 
    #     return stop
    
    # Check UQs
    # User-defined parts: which properties do you want to check?
    for property in properties:
        try:
            abs_dev = np.linalg.norm(mol.__dict__[property] - mol.__dict__['aux_'+property])
        except:
            abs_dev = np.linalg.norm(mol.get_xyz_vectorial_properties(property) - mol.get_xyz_vectorial_properties('aux_'+property))
        if abs_dev > thresholds.__dict__[property]:
            stop = True 
            mol.need_to_be_labeled = True 
            break 
    return stop 

def internal_stop_function(mol):
    stop = False 
    if mol.uncertain:
        stop = True 
    return stop
def internal_stop_function_namd(stop_check, current_state, mol):
    stop = False 
    if mol.uncertain:
        stop = True 
    return stop, stop_check
def stopper(errMsg):
    '''
    function printing error message
    '''
    print(f"<!> {errMsg} <!>")
    exit()