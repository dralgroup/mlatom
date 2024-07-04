import sys
from . import data, stats, models, optimize_geometry, freq, md, md_parallel, generate_initial_conditions
from .al_utils import *
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import scipy 
import torch
import random 
import joblib 
from joblib import Parallel, delayed 
from multiprocessing.pool import ThreadPool as Pool
import timeit 
import json
import copy


class al():
    '''
    Active learning procedure 

    Arguments:
        job_name (str): Job name. Output files will be named after it.
        initdata_sampler (:class:`Sampler`): Initial points sampler.
        initdata_sampler_kwargs (dict): Initial points sampler kwargs.
        initial_points_refinement (str, optional): Initial points refinement method. Options: 'cross-validation', 'validation', 'one-shot'(by default).
        init_ncvsplits (int, optional): Number of CV splits for cross validation initial points sampling. 5 by default.
        init_validation_set_fraction (float): Fraction of validation set in validation initial points sampling. 0.2 by default.
        init_RMSE_threshold (float): RMSE threshold for intial points sampling. Stop sampling if RMSE is smaller than this threshold.
        init_train_energies_only (bool): Train on energies only when sampling initial points. False by default.
        minimum_number_of_fitting_points (int):
        init_ntrain_next (int, optional): 
        label_nthreads (int, optional): Number of threads for labelling new points. CPU count by default. 
        ml_model_type (str, optional): ML model type. Options: 'KREG'(by default), 'ANI'.
        ml_model_trainer (:class:`ml_model_trainer`): ML model trainer.
        collective_ml_models (:class:`collective_ml_models`): Collective ML models.
        device (str, optional): Device to train ANI model. Options: 'cuda'(by default), 'cpu'.
        property_to_learn (List[str]): Properties to learn in active learning.
        property_to_check (List[str]): Properties to check in uncertainty quantification.
        validation_set_fraction (float): Validation set fraction. 0.1 by default.
        initial_conditions_sampler (:class:`Sampler`)
        initial_conditions_sampler_kwargs (dict)
        sampler (:class:`Sampler`): Active learning sampler.
        sampler_kwargs (Dict): Kwargs of active learning sampler.
        uncertainty_quantification (:class:`uq`): UQ class that calculate UQ thresholds and UQs.
        new_points (int): Maximum number of sampled points in each iteration. Same as number_of_trajectories by default.
        min_new_points (int): Minimum number of sampled points in each iteration. If number of sampled points is less than this value, active learning is considered as converged. 1 by default.
        reference_method (:class:`ml.models.methods`): Reference method.
        max_iterations
        init_sampling_only (bool): Exit after sampling initial points
    '''
    def __init__(self,**kwargs):
        # Some default settings 
        self.default_number_of_trajectories = 100


        # Options of actice learning
        if 'job_name' in kwargs:
            self.job_name = kwargs['job_name'] 
        else:
            self.job_name = "AL"

        # .Reference method
        if 'reference_method' in kwargs:
            self.reference_method = kwargs['reference_method']
        else:
            stopper("Reference method is needed")

        if 'molecule' in kwargs:
            self.molecule = kwargs['molecule']
            self.eqmol = self.optfreq(self.molecule)
        else:
            self.molecule = None 
            self.eqmol = None

        # .Initial points sampling 
        # ..initdata_sampler: Initial points sampler, should be a "sampler" class or a class that inherits "sampler". 
        if 'initdata_sampler' in kwargs:
            self.initdata_sampler = kwargs['initdata_sampler']
            if not isinstance(self.initdata_sampler,Sampler):
                self.initdata_sampler = Sampler(sampler_function=self.initdata_sampler)
        else:
            self.initdata_sampler = Sampler('wigner')
        # ..initdata_sampler_kwargs: Initial points sampler kwargs, a dict with all the arguments.
        if 'initdata_sampler_kwargs' in kwargs:
            self.initdata_sampler_kwargs = kwargs['initdata_sampler_kwargs']
        else:
            if self.eqmol is None:
                stopper("Please provide molecule")
            self.initdata_sampler_kwargs = {
                'molecule':self.eqmol,
                'number_of_initial_conditions':50,
                'initial_temperature':300,
            }
            # stopper('Initial points sampler kwargs not provided')
        # ..initial_points_refinement: Initial points refinement method, a string which specifies the method.
        #   ...Options:
        #       "cross-validation": Check the cross validation error and fit the learning curve. Keep adding points until doubling Ntr improves accuracy by less than 10%.
        #       "validation": Check the validation error and fit the learning curve. Keep adding points until doubling Ntr improves accuracy by less than 10%.
        #       "one-shot": Sample points only once, without checking anything. [default]
        if 'initial_points_refinement' in kwargs:
            self.initial_points_refinement = kwargs['initial_points_refinement']
        else:
            self.initial_points_refinement = 'cross-validation'
        # ..init_ncvsplits: Number of cross validation splits, only works for initial_points_refinement="cross_validation". 5 by default.
        if 'init_ncvsplits' in kwargs:
            self.init_ncvsplits = kwargs['init_ncvsplits']
        else:
            self.init_ncvsplits = 5 
        # ..init_validation_set_fraction: Fraction of validation set in validation initial points sampling. 0.2 by default.
        if 'init_validation_set_fraction' in kwargs:
            self.init_validation_set_fraction = kwargs['init_validation_set_fraction']
        else:
            self.init_validation_set_fraction = 0.2
        # ..init_RMSE_threshold: RMSE threshold for intial points sampling. Stop sampling if RMSE is smaller than this threshold.
        if 'init_RMSE_threshold' in kwargs:
            self.init_RMSE_threshold = kwargs['init_RMSE_threshold']
        else:
            self.init_RMSE_threshold = None
        # ..init_train_energies_only: Train on energies only when sampling initial points. False by default.
        if 'init_train_energies_only' in kwargs:
            self.init_train_energies_only = kwargs['init_train_energies_only']
        else:
            self.init_train_energies_only = True
        # ..minimum_number_of_fitting_points
        if 'minimum_number_of_fitting_points' in kwargs:
            self.minimum_number_of_fitting_points = kwargs['minimum_number_of_fitting_points']
        else:
            self.minimum_number_of_fitting_points = 5
        # ..init_ntrain_next: Number of additional training points to check the convergence of training set, only works for initial_points_refinement="cross_validation". 
        if 'init_ntrain_next' in kwargs:
            self.init_ntrain_next = kwargs['init_ntrain_next']
        else:
            self.init_ntrain_next = 50

        # .Nthreads 
        # ..label_nthreads: Number of processes used for labeling
        if 'label_nthreads' in kwargs:
            self.label_nthreads = kwargs['label_nthreads']
        else:
            self.label_nthreads = joblib.cpu_count()
        
        if 'model_predict_kwargs' in kwargs:
            self.model_predict_kwargs = kwargs['model_predict_kwargs']
        else:
            self.model_predict_kwargs = {}

        # .ML models
        # ..ml_model: ML model class
        if 'ml_model' in kwargs:
            self.ml_model = kwargs['ml_model']
            if isinstance(self.ml_model,str):
                self.ml_model_type = self.ml_model 
                self.ml_model = ml_model
            else:
                self.ml_model_type = None
        else:
            self.ml_model_type = 'ani'
            self.ml_model = ml_model

        # ..ml_model_trainer: ML model trainer 
        if 'ml_model_trainer' in kwargs:
            self.ml_model_trainer = kwargs['ml_model_trainer']
            if isinstance(self.ml_model_trainer,str):
                self.ml_model_trainer = ml_model_trainer(ml_model_type=self.ml_model_trainer)
        else:
            if not self.ml_model_type is None:
                self.ml_model_trainer = ml_model_trainer(ml_model_type=self.ml_model_type)

        # ..device: Use what device to train ML model 
        #   ...Options:
        #       "cpu": Use CPU
        #       "cuda": Use GPU (cuda)
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ..property_to_learn: A list of properties to learn with ML
        #   ...Note:
        #       If you want to learn value property and its corresponding gradient property, name the gradient property by adding "_gradient" suffix to the value property name, e.g., "energy" and "energy_gradients". This will create one single model trained on both values and gradients.
        if 'property_to_learn' in kwargs:
            self.property_to_learn = kwargs['property_to_learn']
        else:
            self.property_to_learn = ['energy','energy_gradients']
        # ..property_to_check: A list of properties to check by uncentainty quantification ("uq" class)
        if 'property_to_check' in kwargs:
            self.property_to_check = kwargs['property_to_check']
        else:
            self.property_to_check = ['energy'] # Property to check in UQ
        # ..validation_set_fraction: Fraction of the whole training set to be used as validation set.
        if 'validation_set_fraction' in kwargs:
            self.validation_set_fraction = kwargs['validation_set_fraction']
        else:
            self.validation_set_fraction = 0.1
        # ..reuse_previous_model:
        if 'reuse_previous_model' in kwargs:
            self.reuse_previous_model = kwargs['reuse_previous_model']
        else:
            self.reuse_previous_model = False 
        # ..train_model_from_scratch_interval:
        if 'train_model_from_scratch_interval' in kwargs:
            self.train_model_from_scratch_interval = kwargs['train_model_from_scratch_interval']
        else:
            self.train_model_from_scratch_interval = 5


        # .Sampler 
        # ..sampler: Sampler used in active learning iterations
        if 'sampler' in kwargs:
            self.sampler = kwargs['sampler']
            if not isinstance(self.sampler, Sampler):
                self.sampler = Sampler(sampler_function=self.sampler)
        else:
            self.sampler = Sampler(sampler_function='batch_md')
        # ..sampler_kwargs: Kwargs for sampler used in active learning iterations
        if 'sampler_kwargs' in kwargs:
            self.sampler_kwargs = kwargs['sampler_kwargs']
            if 'initcond_sampler' in self.sampler_kwargs.keys():
                self.sampler_kwargs['initcond_sampler'] = Sampler(sampler_function=self.sampler_kwargs['initcond_sampler'])
        else:
            if self.eqmol is None:
                stopper('Please provide molecule')
            self.sampler_kwargs = {
                'initcond_sampler':Sampler(sampler_function='wigner'),
                'initcond_sampler_kwargs':{
                    'molecule': self.eqmol,
                    'number_of_initial_conditions':self.default_number_of_trajectories,
                    'initial_temperature':300,
                },
                'maximum_propagation_time':1000.0,
                'time_step':0.1,
                'nthreads':self.label_nthreads,
            }

        # .Molecular dynamics
        # ..new_points: Maximum number of sampled points in each iteration
        if 'new_points' in kwargs:
            self.new_points = kwargs['new_points']
        else:
            self.new_points = None #self.default_number_of_trajectories
        # ..min_new_points: Minimum number of sampled points in each iteration
        if 'min_new_points' in kwargs:
            self.min_new_points = kwargs['min_new_points']
        else:
            self.min_new_points = 5  # Active learning is considered converged if number/partition of sampled points is less than this value

        # ..max_iterations:
        if 'max_iterations' in kwargs:
            self.max_iterations = kwargs['max_iterations']
        else:
            self.max_iterations = np.inf
        # .Options for debugging and testing
        if 'init_sampling_only' in kwargs: 
            self.init_sampling_only = kwargs['init_sampling_only']
        else:
            self.init_sampling_only = False
        if 'debug' in kwargs:
            self.debug = kwargs['debug']
        else:
            self.debug = False

        # Others
        self.hyperparameters = {}
        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue 
            self.hyperparameters[property] = {}
            self.hyperparameters['aux_'+property] = {}

        self.mlmodel = {}
        self.aux_mlmodel = {}

        self.validation_error_list = []
        self.Ntrain_list = []

        self.al_info = self.load()
        if 'iteration' in self.al_info.keys():
            self.iteration = self.al_info['iteration']
        else:
            self.iteration = 0
            self.al_info['iteration'] = self.iteration 
        # Dump AL info
        self.dump() 


        # self.iteration = -1 
        # while True:
        #     if os.path.exists(f'db_to_label_iteration{self.iteration+1}.json'):
        #         self.iteration += 1
        #     else:
        #         break
        self.converged = False
        if not self.debug:
            self.main()

    # Dump AL info
    def dump(self):
        jsonfile = open('al_info.json','w') 
        json.dump(self.dict_to_json_dict(self.al_info),jsonfile,indent=4)
        jsonfile.close()

    # Load AL info
    def load(self):
        if os.path.exists('al_info.json'):
            jsonfile = open('al_info.json','r') 
            al_info = json.load(jsonfile)
        else:
            al_info = {}
        return al_info

    def dict_to_json_dict(self,dict_to_convert):
        dd = copy.deepcopy(dict_to_convert)
        for key in dd.keys():
            if type(dd[key]) == np.ndarray: dd[key] = dd[key].tolist()
            elif type(dd[key]) == np.float32: dd[key] = dd[key].item()
            elif type(dd[key]) == np.float64: dd[key] = dd[key].item()
            elif type(dd[key]) == np.int16: dd[key] = dd[key].item()
            elif type(dd[key]) == np.int32: dd[key] = dd[key].item()
            elif type(dd[key]) == np.int64: dd[key] = dd[key].item()
            elif type(dd[key]) == np.cfloat: dd[key] = dd[key].item()
            elif type(dd[key]) == list:
                for ii in range(len(dd[key])):
                    if type(dd[key][ii]) == list:
                        for jj in range(len(dd[key][ii])):
                            if type(dd[key][ii][jj]) == np.ndarray: dd[key][ii][jj] = dd[key][ii][jj].tolist()
                    elif type(dd[key][ii]) == np.ndarray: dd[key][ii] = dd[key][ii].tolist()
            elif type(dd[key]) == dict:
                dd[key] = self.dict_to_json_dict(dd[key])
        return dd
        
    def main(self):
        self.converged = False
        if self.iteration == 0:
            time0 = timeit.default_timer()
            self.get_initial_data_pool()
            time1 = timeit.default_timer() 
            print(f"Initial points sampling time: {time1-time0} s")
        else:
            print(f"Restarting active learning from iteration {self.iteration}")
            self.labeled_database = data.molecular_database() 
            for ii in range(self.iteration):
                self.labeled_database += data.molecular_database.load(f'iterations/{ii}/labeled_db.json',format='json')

        if self.init_sampling_only:
            return
        
        if not os.path.exists('iterations'):
            os.mkdir('iterations')
        
        while not self.converged and self.iteration < self.max_iterations:
            iteration_start_time = timeit.default_timer()
            print(f'Active learning iteration {self.iteration}')
            if not os.path.exists(f'iterations/{self.iteration}'):
                os.mkdir(f'iterations/{self.iteration}')
            self.al_info['working_directory'] = f'iterations/{self.iteration}'
            self.dump()
            self.label_points()
            labeling_finish_time = timeit.default_timer()
            self.dump()
            self.create_ml_model()
            training_finish_time = timeit.default_timer()
            self.dump()
            self.use_ml_model()
            sampling_finish_time = timeit.default_timer()
            self.dump()
            print(f"Iteration {self.iteration} takes {sampling_finish_time - iteration_start_time} s")
            print(f'  Labeling time: {labeling_finish_time - iteration_start_time} s')
            print(f'  Training time: {training_finish_time - labeling_finish_time} s')
            print(f'  Sampling time: {sampling_finish_time - training_finish_time} s')
            # exit()
            print('\n\n\n')
            # Update and dump AL info
            self.iteration += 1
            self.al_info['iteration'] = self.iteration 
            self.dump()

        if self.iteration >= self.max_iterations:
            print(f"Number of iterations is larger than {self.max_iterations}, stop active learning")

    def optfreq(self,mol):
        mol_copy = mol.copy()
        if not 'frequencies' in mol_copy.__dict__:
            geomopt = optimize_geometry(model=self.reference_method,initial_molecule=mol_copy)
            optmol = geomopt.optimized_molecule
            frequency = freq(model=self.reference_method,molecule=optmol)
        else:
            optmol = mol_copy
        return optmol
        
    def get_initial_data_pool(self):
        # Sample initial points if previous sampling is not found
        if not os.path.exists('init_cond_db.json'):
            # Get initial data pool (eqmol is not included)
            if self.initial_points_refinement.casefold() == 'cross-validation'.casefold():
                print(" Initial points sampling: Use cross validation")
                print(f" Initial points sampling: Number of CV splits = {self.init_ncvsplits}")
                self.get_initial_data_pool_cross_validation(self.initdata_sampler,self.initdata_sampler_kwargs,option=self.initial_points_refinement)
            elif self.initial_points_refinement.casefold() == 'validation'.casefold():
                print(" Initial points sampling: Use validation")
                print(f" Initial points sampling: Fraction of validation set = {self.init_validation_set_fraction}")
                self.get_initial_data_pool_cross_validation(self.initdata_sampler,self.initdata_sampler_kwargs,option=self.initial_points_refinement)
            elif self.initial_points_refinement.casefold() == 'one-shot'.casefold():
                self.get_initial_data_pool_one_shot(self.initdata_sampler,self.initdata_sampler_kwargs)
            else:
                stopper(f'Unrecognized intial points refinement method: {self.initial_points_refinement}')

            self.init_cond_db.dump(filename='init_cond_db.json', format='json')
            self.molecular_pool_to_label = self.init_cond_db
        

    # Increase the number of initial points until cross validation error does not improve much
    def get_initial_data_pool_cross_validation(self,sampler,sampler_kwargs,option):
        sample_initial_conditions = True 
        self.init_cond_db = data.molecular_database()
        Ntrain_list = []
        eRMSE_list = []
        def linear_fit_error(slope,intercept,x):
            return np.exp(intercept)*x**slope
        print("Start initial points sampling...")

        if self.init_RMSE_threshold is None:
            print(" Initial points samplimg: initial points RMSE threshold not found, fit learning curve instead")
        else:
            print(" Initial points samplimg: initial points RMSE threshold found, stop sampling if RMSE is smaller than threshold")
            print(f" Initial points sampling: RMSE threshold = {self.init_RMSE_threshold}")

        if self.init_train_energies_only:
            print(" Initial points sampling: Train on energies only")
        else:
            print(" Initial points sampling: Train on both energies and gradients")

        while sample_initial_conditions:
            init_cond_db = sampler.sample(al_object=self,**sampler_kwargs)
            self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=init_cond_db,nthreads=self.label_nthreads)
            fail_count = 0
            for init_mol in init_cond_db:
                if 'energy' in init_mol.__dict__ and 'energy_gradients' in init_mol.atoms[0].__dict__:
                    self.init_cond_db += init_mol
                else:
                    fail_count += 1 
                if fail_count != 0:
                    print(f"{fail_count} molecules are abandoned due to failed calculation")

            Ntrain_list.append(len(self.init_cond_db))
            if option.casefold() == 'cross-validation'.casefold():
                eRMSE_list.append(self.init_cross_validation())
            elif option.casefold() == 'validation'.casefold():
                eRMSE_list.append(self.init_validation())
            print(f"    Number of points: {Ntrain_list[-1]}")
            print(f"    eRMSE = {eRMSE_list[-1]} Hartree")
            if self.init_RMSE_threshold is None:
                if len(Ntrain_list) > 1 and len(Ntrain_list) >= self.minimum_number_of_fitting_points:
                    x = np.log(Ntrain_list)
                    y = np.log(eRMSE_list)
                    linreg = scipy.stats.linregress(x,y)
                    slope = linreg.slope 
                    intercept = linreg.intercept 
                    rvalue = linreg.rvalue
                    print(f'        Linear regression: log(e) = {intercept} + {slope} log(Ntr)')
                    print(f'        Linear regression: Pearson correlation coefficient = {rvalue}')
                    if slope > 0:
                        # If slope is larger than 0, skip this iteration
                        print("        Linear regression: slope is larger than 0")
                        continue 
                    else:
                        eNtr_next = linear_fit_error(slope,intercept,Ntrain_list[-1]+self.init_ntrain_next)
                        eNtr = linear_fit_error(slope,intercept,Ntrain_list[-1])
                        value = (eNtr-eNtr_next) / eNtr 
                        print(f'        [e(Ntr)-e(Ntr+{self.init_ntrain_next})]/e(Ntr) = {value}')
                        if value >= 0.1:
                            print('        Improvement is large, continue sampling')
                        else:
                            print('        Improvement is small, initial points sampling done')
                            print(f'    Number of initial points: {Ntrain_list[-1]}')
                            sample_initial_conditions = False 
            else:
                rmse = eRMSE_list[-1]
                if rmse >= self.init_RMSE_threshold:
                    print('        RMSE is larger than threshold, continue sampling')
                else:
                    print('        RMSE is smaller than threshold, initial points sampling done')
                    print(f'    Number of initial points: {Ntrain_list[-1]}')
                    sample_initial_conditions = False
            sys.stdout.flush()


    def init_cross_validation(self):
        ncvsplits = self.init_ncvsplits 
        cvsplits = data.sample(molecular_database_to_split=self.init_cond_db,number_of_splits=ncvsplits,split_equally=True)
        moldb = data.molecular_database()
        for isplit in range(ncvsplits):
            validation_molDB = cvsplits[isplit]
            subtraining_molDB = data.molecular_database()
            for ii in range(ncvsplits):
                if ii != isplit:
                    subtraining_molDB += cvsplits[ii]
            if os.path.exists(f'{self.job_name}_initial_points.npz'):
                os.remove(f'{self.job_name}_initial_points.npz')
            if self.init_train_energies_only:
                mlmodel = self.ml_model_trainer.aux_model_trainer(filename=f'{self.job_name}_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                device=self.device)
            else:
                mlmodel = self.ml_model_trainer.main_model_trainer(filename=f'{self.job_name}_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                xyz_derivative_property_to_learn='energy_gradients',
                                                                device=self.device)
            mlmodel.predict(molecular_database=validation_molDB, property_to_predict='estimated_energy',xyz_derivative_property_to_predict='estimated_energy_gradients')
            moldb += validation_molDB 
        energies = moldb.get_properties('energy')
        estimated_energies = moldb.get_properties('estimated_energy')
        eRMSE = stats.rmse(energies,estimated_energies)
        return eRMSE
    
    def init_validation(self):
        fraction = self.init_validation_set_fraction
        subtraining_molDB, validation_molDB = self.init_cond_db.split(number_of_splits=2,fraction_of_points_in_splits=[1-fraction,fraction],sampling='random')
        if os.path.exists(f'{self.job_name}_initial_points.npz'):
            os.remove(f'{self.job_name}_initial_points.npz')
        if self.init_train_energies_only:
            mlmodel = self.ml_model_trainer.aux_model_trainer(filename=f'{self.job_name}_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                device=self.device)
        else:
            mlmodel = self.ml_model_trainer.main_model_trainer(filename=f'{self.job_name}_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                xyz_derivative_property_to_learn='energy_gradients',
                                                                device=self.device)
        mlmodel.predict(molecular_database=validation_molDB, property_to_predict='estimated_energy',xyz_derivative_property_to_predict='estimated_energy_gradients')
        energies = validation_molDB.get_properties('energy')
        estimated_energies = validation_molDB.get_properties('estimated_energy')
        eRMSE = stats.rmse(energies,estimated_energies)
        return eRMSE

    # Sample intial points only once 
    def get_initial_data_pool_one_shot(self,sampler,sampler_kwargs):
        self.init_cond_db = data.molecular_database()
        init_cond_db = sampler.sample(al_object=self,**sampler_kwargs)
        self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=init_cond_db,nthreads=self.label_nthreads)
        fail_count = 0
        init_cond_db.dump('debug.json',format='json')
        for init_mol in init_cond_db:
            if 'energy' in init_mol.__dict__ and 'energy_gradients' in init_mol.atoms[0].__dict__:
                self.init_cond_db += init_mol
            else:
                fail_count += 1 
            if fail_count != 0:
                print(f"{fail_count} molecules are abandoned due to failed calculation")

    # Label points
    def label_points(self):
        if not 'labeled_database' in self.__dict__:
            self.labeled_database = data.molecular_database()
        if not 'molecular_pool_to_label' in self.__dict__: 
            self.molecular_pool_to_label = data.molecular_database()
            if self.iteration == 0:
                print("Loading existing initial condition database")
                self.molecular_pool_to_label = data.molecular_database.load(filename=f'init_cond_db.json', format='json')
            elif os.path.exists(f'iterations/{self.iteration-1}/db_to_label.json'): self.molecular_pool_to_label = data.molecular_database.load(filename=f'iterations/{self.iteration-1}/db_to_label.json', format='json')
    
        nmols = len(self.molecular_pool_to_label)
        # print(nmols)
        labeled_database_iteration = data.molecular_database() 
        if nmols > 0:
            self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=self.molecular_pool_to_label,nthreads=self.label_nthreads)
            for mol in self.molecular_pool_to_label:
                if 'energy' in mol.__dict__ and 'energy_gradients' in mol.atoms[0].__dict__:
                    self.labeled_database.molecules.append(mol)
                    labeled_database_iteration.molecules.append(mol)
            self.labeled_database.dump(filename='labeled_db.json', format='json')
            labeled_database_iteration.dump(filename=os.path.join(self.al_info['working_directory'],f'labeled_db.json'),format='json')
            print('Points to label:', len(self.molecular_pool_to_label.molecules))
            print('New labeled points:', len(labeled_database_iteration.molecules))
            print(f'{len(self.molecular_pool_to_label.molecules) - len(labeled_database_iteration.molecules)} points are abandoned due to failed calculation')
            print('Number of points in the labeled data set:', len(self.labeled_database.molecules))
    
    
    def create_ml_model(self):
        if not 'labeled_database' in self.__dict__:
            self.labeled_database = data.molecular_database()
            if os.path.exists('labeled_db.json'): self.labeled_database.load(filename='labeled_db.json', format='json')
        if len(self.labeled_database.molecules) == 0: return

        if self.ml_model_type is None:
            self.model = self.ml_model(
                al_info=self.al_info,
                device=self.device,
            )
        else:
            self.model = self.ml_model(
                al_info=self.al_info,
                device=self.device,
                ml_model_type=self.ml_model_type,
            )
        # Make a copy of the labeled database in case that it is polluted
        labeled_db_copy = self.labeled_database.copy()
        self.model.train(
            molecular_database=labeled_db_copy,
            al_info=self.al_info,
        )

    def use_ml_model(self):
        # For parallel execution 
        self.model.nthreads = 1

        # Grab points to label from trajectories 
        self.molecular_pool_to_label = data.molecular_database() 

        self.molecular_pool_to_label = self.sampler.sample(al_info=self.al_info,ml_model=self.model,
                                                           iteration=self.iteration,
                                                           **self.sampler_kwargs)
        # Remove energies and energy gradients
        # for imol in range(len(self.molecular_pool_to_label.molecules)):
        #     self.molecular_pool_to_label.molecules[imol] = self.molecular_pool_to_label.molecules[imol].copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=[])

        self.original_number_of_molecules_to_label = len(self.molecular_pool_to_label.molecules)
        if self.min_new_points >= 1:
            if self.original_number_of_molecules_to_label < self.min_new_points:
                print(f'Number of points to be labeled is less than {self.min_new_points}, active learning converged')
                self.converged = True
        if not self.converged and not self.new_points is None:
            
            if self.original_number_of_molecules_to_label > self.new_points:
                self.molecular_pool_to_label.molecules = random.sample(self.molecular_pool_to_label.molecules, self.new_points)
                print(f'Number of points to be labeled is larger than {self.new_points}, sample {self.new_points} points from them')
            
            while self.original_number_of_molecules_to_label < self.new_points:
                self.molecular_pool_to_label += self.sampler.sample(al_info=self.al_info,ml_model=self.model,
                                                            iteration=self.iteration,
                                                            **self.sampler_kwargs)
                if len(self.molecular_pool_to_label) > self.new_points:
                    self.molecular_pool_to_label.molecules = random.sample(self.molecular_pool_to_label.molecules, self.new_points)
                self.original_number_of_molecules_to_label = len(self.molecular_pool_to_label.molecules)

        # Dump the moldb2label with all the information of sampling
        self.molecular_pool_to_label.dump(filename=os.path.join(self.al_info['working_directory'],'db_to_label.json'), format='json')
        
        # Clean up the moldb2label
        self.molecular_pool_to_label = self.molecular_pool_to_label.copy(atomic_labels=['xyz_coordinates'],molecular_labels=['sampling'])
        sys.stdout.flush()

    # Label points
    def label_points_moldb(self,method,model_predict_kwargs={},moldb=None,calculate_energy=True,calculate_energy_gradients=True,calculate_hessian=False,nthreads=1):
        '''
        function labeling points in molecular database

        Arguments:
            method (:class:`ml.models.model`): method that provides energies, energy gradients, etc.
            moldb (:class:`ml.data.molecular_database`): molecular database to label
            calculate_energy (bool): calculate energy
            calculate_energy_gradients (bool): calculate_energy_gradients 
            calculate_hessian (bool): calculate Hessian
            nthreads (int): number of threads
        '''
        def label(imol):
            mol2label = moldb[imol]
            if not ('energy' in mol2label.__dict__ and 'energy_gradients' in mol2label[0].__dict__):
                method.predict(molecule=mol2label,calculate_energy=calculate_energy,calculate_energy_gradients=calculate_energy_gradients,calculate_hessian=calculate_hessian,**model_predict_kwargs)
            return mol2label
        
        nmols = len(moldb)
        if nthreads > 1:
            pool = Pool(processes=nthreads)
            mols = pool.map(label,list(range(nmols)))
            # mols = Parallel(n_jobs=nthreads)(delayed(label)(i) for i in range(nmols))
        else:
            moldb2label = data.molecular_database()
            for mol2label in moldb:
                if not ('energy' in mol2label.__dict__ and 'energy_gradients' in mol2label[0].__dict__):
                    moldb2label += mol2label
            method.predict(molecular_database=moldb2label,calculate_energy=calculate_energy,calculate_energy_gradients=calculate_energy_gradients,calculate_hessian=calculate_hessian,**model_predict_kwargs)








