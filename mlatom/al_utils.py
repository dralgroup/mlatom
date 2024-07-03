import sys
from . import data, stats, models, simulations, optimize_geometry, md, md_parallel, constants, generate_initial_conditions
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import scipy 
import torch
from scipy.optimize import fsolve
import random 
import joblib 
from joblib import Parallel, delayed 
import timeit 
import json




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
            internal_excess_energies.append(iee*constants.eV2Hartree)
        
        return internal_excess_energies

    def optimize_IEE_parameters(self):
        if not 'aa' in self.__dict__ and not 'bb' in self.__dict__:
            target = len(self.molecule)*self.average_IEE_per_atom
            def func(aa):
                bb = 7*aa 
                aa = min(aa,0.3)
                return self.get_average_IEE(aa,bb,self.number_of_valence_electrons) - target 
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
        elif self.ml_model_type.casefold() == 'mace':
            return self.main_model_trainer_mace(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,reuse_model,**kwargs)
         

    def aux_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model=False,**kwargs):
        if self.ml_model_type.casefold() == 'kreg':
            return self.aux_model_trainer_kreg(filename,subtraining_molDB,validation_molDB,property_to_learn,device,reuse_model,**kwargs)
        elif self.ml_model_type.casefold() == 'ani':
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

def stopper(errMsg):
    '''
    function printing error message
    '''
    print(f"<!> {errMsg} <!>")
    exit()