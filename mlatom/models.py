#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! models: Module with models                                                ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''
from __future__ import annotations
from typing import Any, Union, Dict, List, Iterable, Callable
import os, tempfile, uuid, sys
import numpy as np
from collections import UserDict

from . import data, stats, stopper, interfaces
from .decorators import doc_inherit

class model():
    nthreads = 0
    def set_num_threads(self, nthreads=0):
        # implement for each subclass
        if nthreads:
            self.nthreads = nthreads

    def config_multiprocessing(self):
        # for scripts that need to be executed before running model in parallel
        pass

    def parse_args(self, args):
        # for command-line arguments parsing
        pass

    def _predict_geomopt(self,
        return_string=False,
        dump_trajectory_interval=None,
        filename=None,
        format='json',
        print_properties=None,
        molecule: data.molecule = None,
        calculate_energy: bool = True, 
        calculate_energy_gradients: bool = True,
        **kwargs):
        self.predict(molecule=molecule, 
                     calculate_energy=calculate_energy,
                     calculate_energy_gradients=calculate_energy_gradients, 
                     **kwargs)
        if dump_trajectory_interval != None:
            opttraj = data.molecular_trajectory()
            opttraj.load(filename=filename, format=format)
            nsteps = len(opttraj.steps)
            if print_properties == 'all' or type(print_properties) == list:
                printstrs = []
                printstrs += [' %s ' % ('-'*78)]
                printstrs += [f' Iteration {nsteps+1}']
                printstrs += [' %s \n' % ('-'*78)]
                printstrs += [molecule.info(properties=print_properties, return_string=True)]
                printstrs = '\n'.join(printstrs) + '\n'
                if not return_string:
                    print(printstrs)
            opttraj.steps.append(data.molecular_trajectory_step(step=nsteps, molecule=molecule))
            opttraj.dump(filename=filename, format=format)
            moldb = data.molecular_database()
            moldb.molecules = [each.molecule for each in opttraj.steps]
            xyzfilename = os.path.splitext(os.path.basename(filename))[0]
            moldb.write_file_with_xyz_coordinates(f'{xyzfilename}.xyz')
        if return_string and (dump_trajectory_interval != None) and (print_properties == 'all' or type(print_properties) == list): return printstrs

    def predict(
        self, 
        molecular_database: data.molecular_database = None, 
        molecule: data.molecule = None,
        calculate_energy: bool = False, 
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False,
        **kwargs,
    ):
        '''
        Make predictions for molecular geometries with the model.

        Arguments:
            molecular_database (:class:`mlatom.data.molecular_database`, optional): A database contains the molecules whose properties need to be predicted by the model.
            molecule (:class:`mlatom.models.molecule`, optional): A molecule object whose property needs to be predicted by the model.
            calculate_energy (bool, optional): Use the model to calculate energy.
            calculate_energy_gradients (bool, optional): Use the model to calculate energy gradients.
            calculate_hessian (bool, optional): Use the model to calculate energy hessian.
        '''
        # for universal control of predicting behavior
        self.set_num_threads()

        if molecular_database != None:
            molecular_database = molecular_database
        elif molecule != None:
            molecular_database = data.molecular_database([molecule])
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            raise ValueError(errmsg)
        return molecular_database
    
    def _call_impl(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    __call__ : Callable[..., Any] = _call_impl

class torch_model(model):
    # models that utilize PyTorch should inherit this class
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            import torch
            torch.set_num_threads(self.nthreads) 

    def config_multiprocessing(self):
        super().config_multiprocessing()
        import torch
        torch.set_num_threads(1)

class torchani_model(torch_model):
    # models that utilize TorchANI should inherit this class
    def config_multiprocessing(self):
        return super().config_multiprocessing()

class tensorflow_model(model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = str(self.nthreads)

class MKL_model(model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            os.environ["MKL_NUM_THREADS"] = str(self.nthreads)
        
class OMP_model(model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            os.environ["OMP_NUM_THREADS"] = str(self.nthreads)

class methods(model):
    '''
    Create a model object with a specified method.

    Arguments:
        method (str): Specify the method. Available methods are listed in the section below.
        program (str, optional): Specify the program to use.
        **kwargs: Other method-specific options

    **Available Methods:**

        ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'AM1'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'CCSD(T)*/CBS'``, ``'CNDO/2'``, ``'D4'``, ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'GFN2-xTB'``, ``'MINDO/3'``, ``'MNDO'``, ``'MNDO/H'``, ``'MNDO/d'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM2*'``, ``'ODM3'``, ``'ODM3*'``, ``'OM1'``, ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'PM6'``, ``'RM1'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``.

        Methods listed above can be accepted without specifying a program.
        The required programs still have to be installed though as described in the installation manual.
    
    **Available Programs and Their Corresponding Methods:** 

        .. table::
            :align: center

            ===============  ==========================================================================================================================================================================
            Program          Methods                                                                                                                                                                   
            ===============  ==========================================================================================================================================================================
            TorchANI         ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'ANI-1xnr'``                                              
            dftd4            ``'AIQM1'``, ``'AIQM1@DFT'``, ``'ANI-1x-D4'``, ``'ANI-2x-D4'``, ``'D4'``                                                                                                  
            MNDO or Sparrow  ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'MNDO'``, ``'MNDO/d'``, ``'ODM2*'``, ``'ODM3*'``,  ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``
            MNDO             ``'CNDO/2'``, ``'MINDO/3'``, ``'MNDO/H'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM3'``, ``'OM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                  
            Sparrow          ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'PM6'``, ``'RM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                                                              
            xTB              ``'GFN2-xTB'``, semiempirical GFNx-TB methods                                                                                                                                                           
            Orca             ``'CCSD(T)*/CBS'``, DFT                                                                                                                                                      
            Gaussian         ab initio methods, DFT
            PySCF            ab initio methods, DFT
            ===============  ==========================================================================================================================================================================
    
    '''

    methods_map = {
    'aiqm1': ['AIQM1', 'AIQM1@DFT', 'AIQM1@DFT*'],
    'ani': ["ANI-1x", "ANI-1ccx", "ANI-2x", 'ANI-1x-D4', 'ANI-2x-D4', 'ANI-1xnr'], 
    'aimnet2': ["AIMNet2@b973c", "AIMNet2@wb97m-d3"],
    'mndo': ['ODM2*', 'ODM2', 'ODM3', 'OM3', 'OM2', 'OM1', 'PM3', 'AM1', 'MNDO/d', 'MNDOC', 'MNDO', 'MINDO/3', 'CNDO/2', 'SCC-DFTB', 'SCC-DFTB-heats', 'MNDO/H', 'MNDO/dH'],
    'sparrow': ['DFTB0', 'DFTB2', 'DFTB3', 'MNDO', 'MNDO/d', 'AM1', 'RM1', 'PM3', 'PM6', 'OM2', 'OM3', 'ODM2*', 'ODM3*', 'AIQM1'],
    'xtb': ['GFN2-xTB'],
    'dftd4': ['D4'],
    'ccsdtstarcbs': ['CCSD(T)*/CBS'],
    'gaussian': [],
    'columbus': [],
    'pyscf': [],
    'turbomole': [],
    'orca': [],
    }
    
    def __init__(self, method: str = None, program: str = None, **kwargs):
        # !!! IMPORTANT !!! 
        # It is neccesary to save all the arguments in the model, otherwise it would not be dumped correctly!
        self.method  = method
        self.program = program
        if kwargs != {}: self.kwargs = kwargs
        
        if program != None:
            if program.casefold() in self.methods_map:
                self.interface = interfaces.__dict__[program.casefold()](method=method, **kwargs)
        elif self.method.casefold() in [mm.casefold() for mm in self.methods_map['mndo']] or self.method in [mm.casefold() for mm in self.methods_map['sparrow']]:
            if self.method.casefold() in [mm.casefold() for mm in self.methods_map['mndo']] and 'mndobin' in os.environ:
                from .interfaces.mndo_interface import mndo_methods
                self.interface = mndo_methods(method=method, **kwargs)
            elif self.method.casefold() in [mm.casefold() for mm in self.methods_map['sparrow']] and 'sparrowbin' in os.environ:
                from .interfaces.sparrow_interface import sparrow_methods
                self.interface = sparrow_methods(method=method, **kwargs)
            else:
                errmsg = "Can't find appropriate program for the requested method, please set the environment variable: export mndobin=... or export sparrowbin=..."
                raise ValueError(errmsg)
        else:
            for interface, interfaced_methods in self.methods_map.items():
                if self.method.casefold() in [mm.casefold() for mm in interfaced_methods]:
                    self.interface = interfaces.__dict__[interface](method=method, **kwargs)
                    break
    
    @property
    def nthreads(self):
        return self.interface.nthreads
    
    @nthreads.setter
    def nthreads(self, nthreads):
        self.interface.nthreads = nthreads

    def predict(self, *args, **kwargs):
        self.interface.predict(*args, **kwargs)
    
    def config_multiprocessing(self):
        super().config_multiprocessing()
        self.interface.config_multiprocessing()

    @classmethod
    def known_methods(cls):
        methods = set(method for interfaced_methods in cls.methods_map.values() for method in interfaced_methods)
        return methods

    @classmethod
    def is_known_method(cls, method=None):
        methodcasefold = [mm.casefold() for mm in cls.known_methods()]
        if method.casefold() in methodcasefold: return True
        else: return False
        
    def dump(self, filename=None, format='json'):
        model_dict = {'type': 'method'}
        for key in self.__dict__:
            tt = type(self.__dict__[key])
            if tt in [str, dict]:
                model_dict[key] = self.__dict__[key]
            model_dict['nthreads'] = self.nthreads

        if format == 'json':
            import json
            with open(filename, 'w') as fjson:
                json.dump(model_dict, fjson, indent=4)
        if format == 'dict':
            return model_dict

# Parent model class
class ml_model(model):
    def train(
        self,
        molecular_database: data.molecular_database,
        property_to_learn: Union[str, None] = 'y',
        xyz_derivative_property_to_learn: str = None,
    ) -> None:
        '''
        Train the model with molecular database provided.

        Arguments:
            molecular_database (:class:`mlatom.data.molecular_database`): The database of molecules for training.
            property_to_learn (str, optional): The label of property to be learned in model training.
            xyz_derivative_property_to_learn (str, optional): The label of XYZ derivative property to be learned.
        '''
        
        self.set_num_threads()

    @doc_inherit
    def predict(
        self, 
        molecular_database: data.molecular_database = None, molecule: data.molecule = None,
        calculate_energy: bool = False, property_to_predict: Union[str, None] = 'estimated_y',
        calculate_energy_gradients: bool = False, xyz_derivative_property_to_predict: Union[str, None] = 'estimated_xyz_derivatives_y', 
        calculate_hessian: bool = False, hessian_to_predict: Union[str, None] = 'estimated_hessian_y',
    ) -> None:
        '''
            property_to_predict (str, optional): The label name where the predicted properties to be saved.
            xyz_derivative_property_to_predict (str, optional): The label name where the predicted XYZ derivatives to be saved.
            hessian_to_predict (str, optional): The label name where the predicted Hessians to be saved.
        '''
        molecular_database = super().predict(molecular_database=molecular_database, molecule=molecule)

        if calculate_energy:
            property_to_predict = 'energy'
                       
        if calculate_energy_gradients:
            xyz_derivative_property_to_predict = 'energy_gradients'
        
        if calculate_hessian:
            hessian_to_predict = 'hessian'
        
        return molecular_database, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict
    
    def generate_model_dict(self):
        model_dict = {
            'type': 'ml_model',
            'ml_model_type': str(type(self)).split("'")[1],
            'kwargs': {
                'model_file': os.path.abspath(self.model_file)
            },
            # 'hyperparameters': self.hyperparameters,
            'nthreads': self.nthreads,
        }
        return model_dict

    def reset(self):
        if os.path.exists(self.model_file): os.remove(self.model_file)

    def dump(self, filename=None, format='json'):
        if not self.model_file:
            self.save()

        model_dict = self.generate_model_dict()

        if format == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(model_dict, f, indent=4)
        if format == 'dict':
            return model_dict
    
    def parse_args(self, args):
        super().parse_args(args)

    def parse_hyperparameter_optimization(self, args, arg_key):
        space_map = {
            'loguniform': 'log',
            'uniform': 'linear',
        }
        if args.hyperparameter_optimization['optimization_algorithm'] == 'tpe':
            value = args._hyperopt_str_dict[arg_key]
            space = space_map[value.split('(')[0].split('.')[-1]]
            lb = float(value.split('(')[1][:-1].split(',')[0])
            hb = float(value.split('(')[1][:-1].split(',')[1])
            self.hyperparameters[arg_key].optimization_space = space
            if space == 'log':
                self.hyperparameters[arg_key].minval = 2**lb
                self.hyperparameters[arg_key].maxval = 2**hb
            else:
                self.hyperparameters[arg_key].minval = lb
                self.hyperparameters[arg_key].maxval = hb
 
    def calculate_validation_loss(self,
                                 training_kwargs=None,
                                 prediction_kwargs=None,
                                 cv_splits_molecular_databases=None, calculate_CV_split_errors=False,
                                 subtraining_molecular_database=None, validation_molecular_database=None,
                                 validation_loss_function=None, validation_loss_function_kwargs={},
                                 debug=False):
        
        property_to_learn = self.get_property_to_learn(training_kwargs)
        xyz_derivative_property_to_learn = self.get_xyz_derivative_property_to_learn(training_kwargs)
        if property_to_learn == None and xyz_derivative_property_to_learn == None:
            property_to_learn = 'y'
            if training_kwargs is None:
                training_kwargs = {'property_to_learn': 'y'}
            else:
                training_kwargs['property_to_learn'] = 'y'
        
        property_to_predict = self.get_property_to_predict(prediction_kwargs)
        xyz_derivative_property_to_predict = self.get_xyz_derivative_property_to_predict(prediction_kwargs)
        if property_to_predict == None and xyz_derivative_property_to_predict == None:
            if prediction_kwargs == None: prediction_kwargs = {}
            if property_to_learn != None:
                property_to_predict = f'estimated_{property_to_learn}'
                prediction_kwargs['property_to_predict'] = property_to_predict
            if xyz_derivative_property_to_learn != None:
                xyz_derivative_property_to_predict = f'estimated_{xyz_derivative_property_to_learn}'
                prediction_kwargs['xyz_derivative_property_to_predict'] = xyz_derivative_property_to_predict
            
        estimated_y=None; y=None; estimated_xyz_derivatives=None; xyz_derivatives=None

        if type(cv_splits_molecular_databases) == type(None):
            self.holdout_validation(subtraining_molecular_database=subtraining_molecular_database,
                                    validation_molecular_database=validation_molecular_database,
                                    training_kwargs=training_kwargs,
                                    prediction_kwargs=prediction_kwargs)
            if property_to_learn != None:
                y = validation_molecular_database.get_properties(property_name=property_to_learn)
                estimated_y = validation_molecular_database.get_properties(property_name=property_to_predict)
            if xyz_derivative_property_to_learn != None:
                xyz_derivatives = validation_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_learn)
                estimated_xyz_derivatives = validation_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_predict)
        else:
            self.cross_validation(cv_splits_molecular_databases=cv_splits_molecular_databases,
                                    training_kwargs=training_kwargs,
                                    prediction_kwargs=prediction_kwargs)
            training_molecular_database = data.molecular_database()
            if calculate_CV_split_errors:
                nsplits = len(cv_splits_molecular_databases)
                CV_y=[None for ii in range(nsplits)]; CV_yest=[None for ii in range(nsplits)]; CV_xyz_derivatives=[None for ii in range(nsplits)]; CV_estimated_xyz_derivatives=[None for ii in range(nsplits)]
            for CVsplit in cv_splits_molecular_databases:
                training_molecular_database.molecules += CVsplit.molecules
            if property_to_learn != None:
                y = training_molecular_database.get_properties(property_name=property_to_learn)
                estimated_y = training_molecular_database.get_properties(property_name=property_to_predict)
                if calculate_CV_split_errors:
                    CV_y = [] ; CV_yest = []
                    for ii in range(nsplits):
                        CV_y.append(cv_splits_molecular_databases[ii].get_properties(property_name=property_to_learn))
                        CV_yest.append(cv_splits_molecular_databases[ii].get_properties(property_name=property_to_predict))
            if xyz_derivative_property_to_learn != None:
                xyz_derivatives = training_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_learn)
                estimated_xyz_derivatives = training_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_predict)
                if calculate_CV_split_errors:
                    CV_xyz_derivatives = [] ; CV_estimated_xyz_derivatives = []
                    for ii in range(nsplits):
                        CV_xyz_derivatives.append(cv_splits_molecular_databases[ii].get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_learn))
                        CV_estimated_xyz_derivatives.append(cv_splits_molecular_databases[ii].get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_predict))
        
        def geomRMSEloc(estimated_y,y,estimated_xyz_derivatives,xyz_derivatives):
            total_rmse = 1
            if property_to_learn != None:
                total_rmse *= stats.rmse(estimated_y,y)
            if xyz_derivative_property_to_learn != None:
                total_rmse *= stats.rmse(estimated_xyz_derivatives.reshape(estimated_xyz_derivatives.size),xyz_derivatives.reshape(xyz_derivatives.size))
            if property_to_learn != None and xyz_derivative_property_to_learn != None:
                total_rmse = np.sqrt(total_rmse)
            return total_rmse

        if validation_loss_function == None: error = geomRMSEloc(estimated_y,y,estimated_xyz_derivatives,xyz_derivatives)
        else: error = validation_loss_function(**validation_loss_function_kwargs)
        
        self.reset()
        
        if type(cv_splits_molecular_databases) != type(None) and calculate_CV_split_errors:
            CV_errors = []
            for ii in range(nsplits):
                if validation_loss_function == None: CVerror = geomRMSEloc(CV_yest[ii],CV_y[ii],CV_estimated_xyz_derivatives[ii],CV_xyz_derivatives[ii])
                else: CVerror = validation_loss_function(**validation_loss_function_kwargs)
                CV_errors.append(CVerror)
                
        if debug:
            for each in self.hyperparameters.keys():
                print(f"  Hyperparameter {each} = {self.hyperparameters[each].value}")
            print(f"    Validation loss: {error}")
        
        if type(cv_splits_molecular_databases) != type(None) and calculate_CV_split_errors:
            return error, CV_errors
        else:
            return error
    
    def optimize_hyperparameters(self,
                                 hyperparameters=None,
                                 training_kwargs=None,
                                 prediction_kwargs=None,
                                 cv_splits_molecular_databases=None,
                                 subtraining_molecular_database=None, validation_molecular_database=None,
                                 optimization_algorithm=None, optimization_algorithm_kwargs={},
                                 maximum_evaluations=10000,
                                 validation_loss_function=None, validation_loss_function_kwargs={},
                                 debug=False):
        
        def validation_loss(current_hyperparameters):
            for ii in range(len(current_hyperparameters)):
                self.hyperparameters[hyperparameters[ii]].value = current_hyperparameters[ii]
            return self.calculate_validation_loss(  training_kwargs=training_kwargs,
                                                    prediction_kwargs=prediction_kwargs,
                                                    cv_splits_molecular_databases=cv_splits_molecular_databases,
                                                    subtraining_molecular_database=subtraining_molecular_database,
                                                    validation_molecular_database=validation_molecular_database,
                                                    validation_loss_function=validation_loss_function, validation_loss_function_kwargs=validation_loss_function_kwargs,
                                                    debug=debug)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_name = self.model_file
            self.model_file = f'{tmpdirname}/{saved_name}'
            if optimization_algorithm.casefold() in [mm.casefold() for mm in ['Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-krylov', 'trust-exact']]:
                import scipy.optimize
                import numpy as np
                initial_hyperparameters = np.array([self.hyperparameters[key].value for key in hyperparameters])
                bounds = np.array([[self.hyperparameters[key].minval, self.hyperparameters[key].maxval] for key in hyperparameters])
                
                res = scipy.optimize.minimize(validation_loss, initial_hyperparameters, method=optimization_algorithm, bounds=bounds,
                            options={'xatol': 1e-8, 'disp': True, 'maxiter': maximum_evaluations})
                for ii in range(len(res.x)):
                    self.hyperparameters[hyperparameters[ii]].value = res.x[ii]
                    
            elif optimization_algorithm.casefold() in [mm.casefold() for mm in ['grid', 'brute']]:
                import scipy.optimize
                import numpy as np
                grid_slices = []
                for key in hyperparameters:
                    if 'grid_size' in optimization_algorithm_kwargs.keys(): 
                        grid_size = optimization_algorithm_kwargs['grid_size']
                    else: 
                        grid_size=9
                    if self.hyperparameters[key].optimization_space == 'linear': grid_slices.append(list(np.linspace(self.hyperparameters[key].minval, self.hyperparameters[key].maxval, num=grid_size)))
                    if self.hyperparameters[key].optimization_space == 'log':    grid_slices.append(list(np.logspace(np.log(self.hyperparameters[key].minval), np.log(self.hyperparameters[key].maxval), num=grid_size, base=np.exp(1))))
                params, _ = optimize_grid(validation_loss, grid_slices)
                for ii in range(len(params)):
                    self.hyperparameters[hyperparameters[ii]].value = params[ii]

            elif optimization_algorithm.lower() == 'tpe':
                import hyperopt
                import numpy as np
                from hyperopt.std_out_err_redirect_tqdm import DummyTqdmFile
                def fileno(self):
                    if self.file.name == '<stdin>':
                        return 0
                    elif self.file.name == '<stdout>':
                        return 1
                    elif self.file.name == '<stderr>':
                        return 2
                    else:
                        return 3

                DummyTqdmFile.fileno = fileno

                validation_loss_wraper_for_hyperopt = lambda d: validation_loss([d[k] for k in hyperparameters])
                space_mapping = {'linear': hyperopt.hp.uniform, 'log': hyperopt.hp.loguniform, 'normal': hyperopt.hp.normal, 'lognormal': hyperopt.hp.lognormal, 'discrete': hyperopt.hp.quniform, 'discretelog': hyperopt.hp.qloguniform, 'discretelognormal': hyperopt.hp.qlognormal, 'choices': hyperopt.hp.choice}
                def get_space(key):
                    space_type = self.hyperparameters[key].optimization_space
                    if space_type in ['log']:
                        args = [np.log(self.hyperparameters[key].minval), np.log(self.hyperparameters[key].maxval)]
                    elif space_type in ['linear']:
                        args = [self.hyperparameters[key].minval, self.hyperparameters[key].maxval]
                    else:
                        raise NotImplementedError
                    return space_mapping[space_type](key, *args)
                
                space = {key: get_space(key) for key in hyperparameters}
                res = hyperopt.fmin(fn=validation_loss_wraper_for_hyperopt, space=space, algo=hyperopt.tpe.suggest, max_evals=maximum_evaluations, show_progressbar=True)#, points_to_evaluate=initial_hyperparameters
                for k, v in res.items():
                    self.hyperparameters[k].value = v
                
            self.model_file = saved_name
            
        # Use the final hyperparameters to train the model and get the validation errors
        self.validation_loss = validation_loss(np.array([self.hyperparameters[key].value for key in hyperparameters]))

    def holdout_validation(self, subtraining_molecular_database=None, validation_molecular_database=None,
                     training_kwargs=None, prediction_kwargs=None):
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        self.train(molecular_database=subtraining_molecular_database, **training_kwargs)
        self.predict(molecular_database = validation_molecular_database, **prediction_kwargs)

    def cross_validation(self, cv_splits_molecular_databases=None,
                     training_kwargs=None, prediction_kwargs=None):
        
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        
        nsplits = len(cv_splits_molecular_databases)
        for ii in range(nsplits):
            subtraining_molecular_database = data.molecular_database()
            for jj in range(nsplits):
                if ii != jj: subtraining_molecular_database.molecules += cv_splits_molecular_databases[jj].molecules
            validation_molecular_database = cv_splits_molecular_databases[ii]
            self.reset()
            self.train(molecular_database=subtraining_molecular_database, **training_kwargs)
            self.predict(molecular_database=validation_molecular_database, **prediction_kwargs)
        
    
    def get_property_to_learn(self, training_kwargs=None):
        if type(training_kwargs) == type(None):
            property_to_learn = None
        else:
            if 'property_to_learn' in training_kwargs:
                property_to_learn = training_kwargs['property_to_learn']
            else:
                property_to_learn = None
        return property_to_learn

    def get_xyz_derivative_property_to_learn(self, training_kwargs=None):
        if type(training_kwargs) == type(None):
            xyz_derivative_property_to_learn = None
        else:
            if 'xyz_derivative_property_to_learn' in training_kwargs:
                xyz_derivative_property_to_learn = training_kwargs['xyz_derivative_property_to_learn']
            else:
                xyz_derivative_property_to_learn = None
        return xyz_derivative_property_to_learn
    
    def get_property_to_predict(self, prediction_kwargs=None):
        if type(prediction_kwargs) != type(None):
            if 'property_to_predict' in prediction_kwargs:
                property_to_predict = prediction_kwargs['property_to_predict']
            else:
                if 'calculate_energy' in prediction_kwargs:
                    property_to_predict = 'estimated_energy'
                else:
                    property_to_predict = 'estimated_y'
        else:
            property_to_predict = None
        return property_to_predict

    def get_xyz_derivative_property_to_predict(self,prediction_kwargs=None):
        if type(prediction_kwargs) != type(None):
            if 'xyz_derivative_property_to_predict' in prediction_kwargs:
                xyz_derivative_property_to_predict = prediction_kwargs['xyz_derivative_property_to_predict']
            else:
                if 'calculate_energy_gradients' in prediction_kwargs:
                    xyz_derivative_property_to_predict = 'estimated_energy_gradients'
                else:
                    xyz_derivative_property_to_predict = 'estimated_xyz_derivatives_y'
        else:
            xyz_derivative_property_to_predict = None
        return xyz_derivative_property_to_predict

def optimize_grid(func, grid):
    last = True
    for ii in grid[:-1]:
        if len(ii) != 1:
            last = False
            break
    if last:
        other_params = [jj[0] for jj in grid[:-1]]
        opt_param = grid[-1][0]
        min_val = func(other_params + [opt_param])
        for param in grid[-1][1:]:
            val = func(other_params + [param])
            if val < min_val:
                opt_param = param
                min_val = val
        return other_params + [opt_param], min_val
    else:
        min_val = None
        for kk in range(len(grid))[:-1]:
            if len(grid[kk]) != 1:
                if kk == 0: other_params_left = []
                else: other_params_left = [[grid[ii][0]] for ii in range(kk)]
                other_params_right = grid[kk+1:]
                for param in grid[kk]:
                    params, val = optimize_grid(func,other_params_left + [[param]] + other_params_right)
                    if min_val == None:
                        min_val = val
                        opt_params = params
                    elif val < min_val:
                        opt_params = params
                        min_val = val
                break
        return opt_params, min_val
 
class krr(ml_model):
    def train(self, molecular_database=None,
              property_to_learn='y',
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None,
              kernel_function_kwargs=None,prior=None):
        
        xyz = np.array([mol.xyz_coordinates for mol in molecular_database.molecules]).astype(float)
        yy = molecular_database.get_properties(property_name=property_to_learn)
        if prior == None:
            self.prior = 0.0
        elif type(prior) == float or type(prior) == int:
            self.prior = prior 
        elif prior.casefold() == 'mean'.casefold():
            self.prior = np.mean(yy)
        else:
            stopper.stopMLatom(f'Unsupported prior: {prior}')
        yy = yy - self.prior
        #self.train_x = xx
        self.Ntrain = len(xyz) 
        self.train_xyz = xyz 
        self.kernel_function = self.gaussian_kernel_function
        self.kernel_function_kwargs = kernel_function_kwargs
        self.kernel_matrix_size = 0 
        Natoms = len(xyz[0])
        self.Natoms = Natoms 

        self.train_property = False 
        self.train_xyz_derivative_property = False 
        yyref = np.array([])
        if property_to_learn != None:
            self.train_property = True
            self.kernel_matrix_size += len(yy)
            yyref = np.concatenate((yyref,yy))
        if xyz_derivative_property_to_learn != None:
            self.train_xyz_derivative_property = True
            self.kernel_matrix_size += 3*Natoms*len(yy)
            yygrad = molecular_database.get_xyz_derivative_properties().reshape(3*Natoms*len(yy))
            yyref = np.concatenate((yyref,yygrad))

        kernel_matrix = np.zeros((self.kernel_matrix_size,self.kernel_matrix_size))

        kernel_matrix = kernel_matrix + np.identity(self.kernel_matrix_size)*self.hyperparameters['lambda'].value
        for ii in range(len(yy)):
            value_and_derivatives = self.kernel_function(xyz[ii],xyz[ii],calculate_value=self.train_property,calculate_gradients=self.train_xyz_derivative_property,calculate_Hessian=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
            kernel_matrix[ii][ii] += value_and_derivatives['value']
            if self.train_xyz_derivative_property:
                kernel_matrix[ii,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['gradients'].reshape(3*Natoms)
                kernel_matrix[len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['Hessian']
            for jj in range(ii+1,len(yy)):
                value_and_derivatives = self.kernel_function(xyz[ii],xyz[jj],calculate_value=self.train_property,calculate_gradients=self.train_xyz_derivative_property,calculate_Hessian=self.train_xyz_derivative_property,calculate_gradients_j=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                kernel_matrix[ii][jj] += value_and_derivatives['value']
                kernel_matrix[jj][ii] += value_and_derivatives['value']
                if self.train_xyz_derivative_property:
                    kernel_matrix[jj,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] = value_and_derivatives['gradients'].reshape(3*Natoms)
                    kernel_matrix[ii,len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms] = value_and_derivatives['gradients_j'].reshape(3*Natoms)
                    kernel_matrix[len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms,len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms] += value_and_derivatives['Hessian']
                    kernel_matrix[len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['Hessian']
        if self.train_xyz_derivative_property:
            kernel_matrix[len(yy):,:len(yy)] = kernel_matrix[:len(yy),len(yy):].T


        if invert_matrix:
            self.alphas = np.dot(np.linalg.inv(kernel_matrix), yyref)
        else:
            from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
            if matrix_decomposition==None:
                try:
                    c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = cho_solve((c, low), yyref, check_finite=False)
                except:
                    c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = lu_solve((c, low), yyref, check_finite=False)
            elif matrix_decomposition.casefold()=='Cholesky'.casefold():
                c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = cho_solve((c, low), yyref, check_finite=False)
            elif matrix_decomposition.casefold()=='LU'.casefold():
                c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = lu_solve((c, low), yyref, check_finite=False)
            
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False,  calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
               property_to_predict = None, xyz_derivative_property_to_predict = None,
                hessian_to_predict=None,):
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)

        Natoms = len(molDB.molecules[0].atoms)
        kk_size = 0 
        if self.train_property:
            kk_size += self.Ntrain 
        if self.train_xyz_derivative_property:
            kk_size += self.Ntrain * Natoms*3

        for mol in molDB.molecules:
            kk = np.zeros(kk_size)
            kk_der = np.zeros((kk_size,3*Natoms))
            for ii in range(self.Ntrain):
                value_and_derivatives = self.kernel_function(self.train_xyz[ii],mol.xyz_coordinates,calculate_gradients=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                kk[ii] = value_and_derivatives['value']
                if self.train_xyz_derivative_property:
                    kk[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms] = value_and_derivatives['gradients'].reshape(3*Natoms)

                if xyz_derivative_property_to_predict:
                    value_and_derivatives = self.kernel_function(mol.xyz_coordinates,self.train_xyz[ii],calculate_gradients=bool(xyz_derivative_property_to_predict),calculate_Hessian=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                    kk_der[ii] = value_and_derivatives['gradients'].reshape(3*Natoms)
                if self.train_xyz_derivative_property:
                    kk_der[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms,:] = value_and_derivatives['Hessian'].T
            gradients = np.matmul(self.alphas.reshape(1,len(self.alphas)),kk_der)[0]
            mol.__dict__[property_to_predict] = np.sum(np.multiply(self.alphas, kk))+self.prior
            for iatom in range(len(mol.atoms)):
                mol.atoms[iatom].__dict__[xyz_derivative_property_to_predict] = gradients[3*iatom:3*iatom+3]
    
    def gaussian_kernel_function(self,coordi,coordj,calculate_value=True,calculate_gradients=False,calculate_gradients_j=False,calculate_Hessian=False,**kwargs):
        import torch
        if 'Req' in kwargs:
            Req = kwargs['Req']
        if calculate_Hessian:
            calculate_gradients = True
            calculate_value = True 
        elif calculate_gradients:
            calculate_value = True
        coordi_tensor = torch.tensor(coordi,requires_grad=True)
        coordj_tensor = torch.tensor(coordj,requires_grad=True)
        value = None 
        gradients = None 
        gradients_j = None
        Hessian = None
        xi_tensor = self.RE_descriptor_tensor(coordi_tensor,Req)
        xj_tensor = self.RE_descriptor_tensor(coordj_tensor,Req)
        if calculate_value:
            value_tensor = torch.exp(torch.sum(torch.square(xi_tensor - xj_tensor))/(-2*self.hyperparameters['sigma'].value**2))
            value = value_tensor.detach().numpy()
        if calculate_gradients:
            gradients_tensor = torch.autograd.grad(value_tensor,coordi_tensor,create_graph=True,retain_graph=True)[0]
            gradients = gradients_tensor.detach().numpy()
        if calculate_gradients_j:
            gradients_j_tensor = torch.autograd.grad(value_tensor,coordj_tensor,create_graph=True,retain_graph=True)[0]
            gradients_j = gradients_j_tensor.detach().numpy()
        if calculate_Hessian:
            Hessian = []
            gradients_tensor = gradients_tensor.reshape(len(gradients_tensor)*3)
            for ii in range(len(gradients_tensor)):
                Hessian.append(torch.autograd.grad(gradients_tensor[ii],coordj_tensor,create_graph=True,retain_graph=True)[0].detach().numpy().reshape((len(gradients_tensor))))
            Hessian = np.array(Hessian)
        output = {'value':value,'gradients':gradients,'gradients_j':gradients_j,'Hessian':Hessian}
        return output
    
    def RE_descriptor_tensor(self,coord_tensor,Req):
        import torch
        Natoms = len(coord_tensor)
        icount = 0 
        for iatom in range(Natoms):
            for jatom in range(iatom+1,Natoms):
                output = Req[icount] / self.distance_tensor(coord_tensor[iatom],coord_tensor[jatom])
                if icount == 0:
                    descriptor = output.reshape(1)
                else:
                    descriptor = torch.cat((descriptor,output.reshape(1)))
                icount += 1
        return descriptor
    
    def distance_tensor(self, atomi,atomj):
        import torch
        return torch.sqrt(self.distance_squared_tensor(atomi,atomj))
    
    def distance_squared_tensor(self, atomi,atomj):
        import torch
        return torch.sum(torch.square(atomi-atomj))

class hyperparameter():
    '''
    Class of hyperparameter object, containing data could be used in hyperparameter optimizations.

    Arguments:
        value (Any, optional): The value of the hyperparameter.
        optimization_space (str, optional): Defines the space for hyperparameter. Currently supports ``'linear'``, and ``'log'``.
        dtype (Callable, optional): A callable object that forces the data type of value. Automatically choose one if set to ``None``.
       
    '''
    def __init__(self, value: Any = None, optimization_space: str = 'linear', dtype: Union[Callable, None] = None, name: str = "", minval: Any = None, maxval: Any = None, step: Any = None, choices: Iterable[Any] = [], **kwargs):
        self.name = name
        self.dtype = dtype if dtype else None if value is None else type(value)
        self.value = value# @Yifan
        self.optimization_space = optimization_space  # 'linear' or 'log'
        self.minval = minval
        self.maxval = maxval
        self.step = step
        self.choices = choices
    def __setattr__(self, key, value):
        if key == 'value':
            value = (value if isinstance(value, self.dtype) else self._cast_dtype(value)) if self.dtype else value
        if key == 'dtype':
            self._set_dtype_cast_method(value)
        super().__setattr__(key, value)
    def __repr__(self):
        return f'hyperparameter {str(self.__dict__)}'
    def _set_dtype_cast_method(self, dtype):
        if type(dtype) == tuple:
            dtype = dtype[0] 
        if dtype == np.ndarray:
            self._cast_dtype = np.array
        else:
            self._cast_dtype = dtype
    def update(self, new_hyperparameter:hyperparameter) -> None:
        '''
        Update hyperparameter with data in another instance.
        
        Arguments:
            new_hyperparameter (:class:`mlatom.models.hyperparamters`): Whose data are to be applied to the current instance.
        '''
        self.__dict__.update(new_hyperparameter.__dict__)
    def copy(self):
        '''
        Returns a copy of current instance.

        Returns:
            :class:`mlatom.models.hyperparamter`: a new instance copied from current one.
        '''
        return hyperparameter(**self.__dict__)

class hyperparameters(UserDict):
    '''
    Class for storing hyperparameters, values are auto-converted to :class:`mlatom.models.hyperparameter` objects.
    Inherit from collections.UserDict.

    Initiaion:
        Initiate with a dictinoary or kwargs or both.
        
        e.g.:
        
        .. code-block::

           hyperparamters({'a': 1.0}, b=hyperparameter(value=2, minval=0, maxval=4))
       
    '''
    def __setitem__(self, key, value):
        if isinstance(value, hyperparameter):
            if key in self:
                super().__getitem__(key).update(value)
            else:
                super().__setitem__(key, value)
        elif key in self:
            super().__getitem__(key).value = value
        else:
            super().__setitem__(key, hyperparameter(value=value, name=key))
    def __getattr__(self, key):
        if key in self:
            return self[key].value
        else:
            return self.__dict__[key]
    def __setattr__(self, key, value):
        if key.startswith('__') or (key in self.__dict__) or key == 'data':
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)
    def __getstate__(self):
        return vars(self)
    def __setstate__(self, state):
        vars(self).update(state)
    def copy(self, keys: Union[Iterable[str], None] = None) -> hyperparameters:
        '''
        Returns a copy of current instance.
        
        Arguments:
            keys (Iterable[str], optional): If keys provided, only the hyperparameters selected by keys will be copied, instead of all hyperparameters.

        Returns:
            :class:`mlatom.models.hyperparamters`: a new instance copied from current one.
        '''
        if keys is None:
            keys = self.keys()
        return hyperparameters({key: self[key].copy() for key in keys})
   
class kreg(krr, OMP_model, MKL_model):
    '''
    Create a KREG model object

    Arguments:
        model_file (str, optional): The name of the file where the model should be dumped to or loaded from.
        ml_program (str, optional): Specify which ML program to use. Avaliable options: ``'KREG_API'``, ``'MLatomF``.
        equilibrium_molecule (:class:`mlatom.data.molecule` | None): Specify the equilibrium geometry to be used to generate RE descriptor. The geometry with lowest energy/value will be selected if set to ``None``.
        prior (default - None): the prior can be 'mean', None (0.0), or any floating point number.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
    '''
    hyperparameters = hyperparameters({'lambda': hyperparameter(value=2**-35, 
                                                         minval=2**-35, 
                                                         maxval=1.0, 
                                                         optimization_space='log',
                                                         name='lambda'),
                                'sigma':  hyperparameter(value=1.0,
                                                         minval=2**-5,
                                                         maxval=2**9,
                                                         optimization_space='log',
                                                         name='sigma')}) 
    def __init__(self, model_file: Union[str, None] = None, ml_program: str = 'KREG_API', equilibrium_molecule: Union[data.molecule, None] = None, prior: float = 0, nthreads: Union[int, None] = None, hyperparameters: Union[Dict[str,Any], hyperparameters]={}):
        self.model_file = model_file
        self.equilibrium_molecule = equilibrium_molecule
        self.ml_program = ml_program
        if self.ml_program.casefold() == 'KREG_API'.casefold():
            from .kreg_api import KREG_API
            if self.model_file != None:
                if self.model_file[-4:] != '.npz':
                    self.model_file += '.npz'
                if os.path.exists(self.model_file):
                    self.kreg_api = KREG_API()
                    self.kreg_api.load_model(self.model_file)
        if self.ml_program.casefold() == 'MLatomF'.casefold():
            from . import interface_MLatomF
            self.interface_mlatomf = interface_MLatomF
        
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
            
        self.nthreads = nthreads

    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                if args.data[hyperparam].lower() == 'opt':
                    self.hyperparameters[hyperparam].dtype = str
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
        if args.eqXYZfileIn:
            self.equilibrium_molecule = data.molecule.from_xyz_file(args.eqXYZfileIn)
    
    def generate_model_dict(self):
        model_dict = super().generate_model_dict()
        model_dict['kwargs']['ml_program'] =  self.ml_program
        return model_dict
    
    def get_descriptor(self, molecule=None, molecular_database=None, descriptor_name='re', equilibrium_molecule=None):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            raise ValueError(errmsg)
        
        if 'reference_distance_matrix' in self.__dict__:
            eq_distmat = self.reference_distance_matrix
        else:
            if equilibrium_molecule == None:
                if 'energy' in molDB.molecules[0].__dict__:
                    energies = molDB.get_properties(property_name='energy')
                    equilibrium_molecule = molDB.molecules[np.argmin(energies)]
                elif 'y' in molecular_database.molecules[0].__dict__:
                    y = molecular_database.get_properties(property_name='y')
                    equilibrium_molecule = molecular_database.molecules[np.argmin(y)]
                else:
                    errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
                    raise ValueError(errmsg)
            eq_distmat = equilibrium_molecule.get_internuclear_distance_matrix()
            self.reference_distance_matrix = eq_distmat
        natoms = len(molDB.molecules[0].atoms)
        for mol in molDB.molecules:
            descriptor = np.zeros(int(natoms*(natoms-1)/2))
            distmat = mol.get_internuclear_distance_matrix()
            ii = -1
            for iatomind in range(natoms):
                for jatomind in range(iatomind+1,natoms):
                    ii += 1
                    descriptor[ii] = eq_distmat[iatomind][jatomind]/distmat[iatomind][jatomind]
            mol.__dict__[descriptor_name] = descriptor
            mol.descriptor = descriptor

    def get_equilibrium_distances(self,molecular_database=None):
        if self.equilibrium_molecule == None: self.equilibrium_molecule = self.get_equilibrium_molecule(molecular_database=molecular_database)
        eq_distmat = self.equilibrium_molecule.get_internuclear_distance_matrix()
        Req = np.array([])
        for ii in range(len(eq_distmat)):
            Req = np.concatenate((Req,eq_distmat[ii][ii+1:]))
        return Req
    
    def get_equilibrium_molecule(self,molecular_database=None):
        if 'energy' in molecular_database.molecules[0].__dict__:
            energies = molecular_database.get_properties(property_name='energy')
            equilibrium_molecule = molecular_database.molecules[np.argmin(energies)]
        elif 'y' in molecular_database.molecules[0].__dict__:
            y = molecular_database.get_properties(property_name='y')
            equilibrium_molecule = molecular_database.molecules[np.argmin(y)]
        else:
            errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
            raise ValueError(errmsg)
        return equilibrium_molecule
    
    def train(self, molecular_database=None,
              property_to_learn=None,
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None,
              prior=None,
              hyperparameters: Union[Dict[str,Any], hyperparameters] = {},):
        self.hyperparameters.update(hyperparameters)
        if self.ml_program.casefold() == 'MLatomF'.casefold():
            mlatomfargs = ['createMLmodel'] + ['%s=%s' % (param, self.hyperparameters[param].value) for param in self.hyperparameters.keys()]
            if save_model:
                if self.model_file == None:
                    self.model_file =f'kreg_{str(uuid.uuid4())}.unf'
            with tempfile.TemporaryDirectory() as tmpdirname:
                molecular_database.write_file_with_xyz_coordinates(filename = f'{tmpdirname}/train.xyz')
                mlatomfargs.append(f'XYZfile={tmpdirname}/train.xyz')
                if property_to_learn != None:
                    molecular_database.write_file_with_properties(filename = f'{tmpdirname}/y.dat', property_to_write = property_to_learn)
                    mlatomfargs.append(f'Yfile={tmpdirname}/y.dat')
                if xyz_derivative_property_to_learn != None:
                    molecular_database.write_file_with_xyz_derivative_properties(filename = f'{tmpdirname}/ygrad.xyz', xyz_derivative_property_to_write = xyz_derivative_property_to_learn)
                    mlatomfargs.append(f'YgradXYZfile={tmpdirname}/ygrad.xyz')
                if prior != None:
                    mlatomfargs.append(f'prior={prior}')
                mlatomfargs.append(f'MLmodelOut={self.model_file}')
                if 'additional_mlatomf_args' in self.__dict__:
                    mlatomfargs += self.additional_mlatomf_args
                self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
        elif self.ml_program.casefold() == 'KREG_API'.casefold():
            from .kreg_api import KREG_API
            if save_model:
                if self.model_file == None:
                    self.model_file = f'kreg_{str(uuid.uuid4())}.npz'
                else:
                    if self.model_file[-4:] != '.npz':
                        self.model_file += '.npz'
            self.kreg_api = KREG_API()
            self.kreg_api.nthreads = self.nthreads
            if self.equilibrium_molecule == None: self.equilibrium_molecule = self.get_equilibrium_molecule(molecular_database=molecular_database)
            if not 'lambdaGradXYZ' in self.hyperparameters.keys(): lambdaGradXYZ = self.hyperparameters['lambda'].value
            else: lambdaGradXYZ = self.hyperparameters['lambdaGradXYZ'].value
            if 'prior' in self.hyperparameters.keys(): prior = self.hyperparameters['prior'].value
            self.kreg_api.train(self.hyperparameters['sigma'].value,self.hyperparameters['lambda'].value,lambdaGradXYZ,molecular_database,self.equilibrium_molecule,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,prior=prior)
            if save_model:
                self.kreg_api.save_model(self.model_file)

        else:
            if 'reference_distance_matrix' in self.__dict__: del self.__dict__['reference_distance_matrix']
            Req = self.get_equilibrium_distances(molecular_database=molecular_database)
            super().train(molecular_database=molecular_database, property_to_learn=property_to_learn,
              invert_matrix=invert_matrix,
              matrix_decomposition=matrix_decomposition,
              kernel_function_kwargs={'Req':Req},prior=prior)
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False, calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
                property_to_predict = None, xyz_derivative_property_to_predict = None, hessian_to_predict = None,):
        if self.ml_program.casefold() != 'MLatomF'.casefold() and self.ml_program.casefold() != 'KREG_API'.casefold():
            super().predict(molecular_database=molecular_database, molecule=molecule,
                            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, property_to_predict=property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict)
        else:
            molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super(krr, self).predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)

            if self.ml_program.casefold() == 'KREG_API'.casefold():
                if 'kreg_api' in dir(self):
                    self.kreg_api.nthreads = self.nthreads
                    self.kreg_api.predict(molecular_database=molecular_database,molecule=molecule,property_to_predict=property_to_predict,xyz_derivative_property_to_predict=xyz_derivative_property_to_predict)
                else:
                    stopper.stopMLatom('KREG_API model not found')
            elif self.ml_program.casefold() == 'MLatomF'.casefold():
                with tempfile.TemporaryDirectory() as tmpdirname:
                    molDB.write_file_with_xyz_coordinates(filename = f'{tmpdirname}/predict.xyz')
                    mlatomfargs = ['useMLmodel', 'MLmodelIn=%s' % self.model_file]
                    mlatomfargs.append(f'XYZfile={tmpdirname}/predict.xyz')
                    if property_to_predict: mlatomfargs.append(f'YestFile={tmpdirname}/yest.dat')
                    if xyz_derivative_property_to_predict: mlatomfargs.append(f'YgradXYZestFile={tmpdirname}/ygradest.xyz')
                    self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
                    if not os.path.exists(f'{tmpdirname}/yest.dat'):
                        import time
                        time.sleep(0.0000001) # Sometimes program needs more time to write file yest.dat / P.O.D., 2023-04-18
                        os.system(f'cp -r {tmpdirname} .')
                    if property_to_predict != None: molDB.add_scalar_properties_from_file(filename = f'{tmpdirname}/yest.dat', property_name = property_to_predict)
                    if xyz_derivative_property_to_predict != None: molDB.add_xyz_derivative_properties_from_file(filename = f'{tmpdirname}/ygradest.xyz', xyz_derivative_property = xyz_derivative_property_to_predict)

def ani(**kwargs):
    '''
    Returns an ANI model object (see :class:`mlatom.interfaces.torchani_interface.ani`).
    '''
    from .interfaces.torchani_interface import ani
    return ani(**kwargs)

def dpmd(**kwargs):
    '''
    Returns a DPMD model object (see :class:`mlatom.interfaces.dpmd_interface.dpmd`).
    '''
    from .interfaces.dpmd_interface import dpmd
    return dpmd(**kwargs)

def gap(**kwargs):
    '''
    Returns a GAP model object (see :class:`mlatom.interfaces.gap_interface.gap`).
    '''
    from .interfaces.gap_interface import gap
    return gap(**kwargs)

def physnet(**kwargs):
    '''
    Returns a PhysNet model object (see :class:`mlatom.interfaces.physnet_interface.physnet`).
    '''
    from .interfaces.physnet_interface import physnet
    return physnet(**kwargs)

def sgdml(**kwargs):
    '''
    Returns an sGDML model object (see :class:`mlatom.interfaces.sgdml_interface.sgdml`).
    '''
    from .interfaces.sgdml_interface import sgdml
    return sgdml(**kwargs)

def mace(**kwargs):
    '''
    Returns an MACE model object (see :class:`mlatom.interfaces.mace_interface.mace`).
    '''
    from .interfaces.mace_interface import mace
    return mace(**kwargs)


class model_tree_node(model):
    '''
    Create a model tree node.

    Arguments:
        name (str): The name assign to the object.
        parent: The parent of the model node.
        children: The children of this model tree node.
        operator: Specify the operation to be made when making predictions.
    '''

    def __init__(self, name=None, parent=None, children=None, operator=None, model=None):
        self.name = name
        self.parent = parent
        self.children = children
        if self.parent != None:
            if self.parent.children == None: self.parent.children = []   
            if not self in self.parent.children:
                self.parent.children.append(self)
        if self.children != None:
            for child in self.children:
                child.parent=self
        self.operator = operator
        self.model = model

    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            if self.children != None:
                for child in self.children:
                    child.set_num_threads(self.nthreads)
            else:
                self.model.set_num_threads(self.nthreads)
    
    def predict(self, **kwargs):
        molDB = super().predict(**kwargs)
        
        if len(molDB) == 0: return
            
        if 'calculate_energy' in kwargs: calculate_energy = kwargs['calculate_energy']
        else: calculate_energy = True
        if 'calculate_energy_gradients' in kwargs: calculate_energy_gradients = kwargs['calculate_energy_gradients']
        else: calculate_energy_gradients = False
        if 'calculate_hessian' in kwargs: calculate_hessian = kwargs['calculate_hessian']
        else: calculate_hessian = False
        if 'nstates' in kwargs: nstates = kwargs['nstates']
        else: nstates = 1
        if 'current_state' in kwargs: current_state = kwargs['current_state']
        else: current_state = 0

        properties = [] ; atomic_properties = []
        if calculate_energy: properties.append('energy')
        if calculate_energy_gradients: atomic_properties.append('energy_gradients')
        if calculate_hessian: properties.append('hessian')

        for mol in molDB.molecules:
            if nstates:
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                if nstates >1:
                    for _ in range(nstates - len(mol.electronic_states)):
                        mol.electronic_states.append(mol_copy.copy())

                for mol_el_st in mol.electronic_states:
                    if not self.name in mol_el_st.__dict__:
                        parent = None
                        if self.parent != None:
                            if self.parent.name in mol_el_st.__dict__:
                                parent = mol_el_st.__dict__[self.parent.name]
                        children = None
                        if self.children != None:
                            for child in self.children:
                                if child.name in mol_el_st.__dict__:
                                    if children == None: children = []
                                    children.append(mol_el_st.__dict__[child.name])
                        mol_el_st.__dict__[self.name] = data.properties_tree_node(name=self.name, parent=parent, children=children)
                
            if not self.name in mol.__dict__:
                parent = None
                if self.parent != None:
                    if self.parent.name in mol.__dict__:
                        parent = mol.__dict__[self.parent.name]
                children = None
                if self.children != None:
                    for child in self.children:
                        if child.name in mol.__dict__:
                            if children == None: children = []
                            children.append(mol.__dict__[child.name])
                mol.__dict__[self.name] = data.properties_tree_node(name=self.name, parent=parent, children=children)
        
        if self.children == None and self.operator == 'predict':
            self.model.predict(**kwargs)
            for mol in molDB.molecules:
                if not mol.electronic_states:
                    self.get_properties_from_molecule(mol, properties, atomic_properties)
                for mol_el_st in mol.electronic_states:
                    # mol_el_st.__dict__[self.name] = data.properties_tree_node(name=self.name, parent=parent, children=children)
                    self.get_properties_from_molecule(mol_el_st, properties, atomic_properties)
        else:
            for child in self.children:
                child.predict(**kwargs)

            if self.operator == 'sum':
                for mol in molDB.molecules:
                    if not mol.electronic_states:
                        mol.__dict__[self.name].sum(properties+atomic_properties)
                    for mol_el_st in mol.electronic_states:
                        mol_el_st.__dict__[self.name].sum(properties+atomic_properties)
            if self.operator == 'average':
                for mol in molDB.molecules:
                    if not mol.electronic_states:
                        mol.__dict__[self.name].average(properties+atomic_properties)
                    for mol_el_st in mol.electronic_states:
                        mol_el_st.__dict__[self.name].average(properties+atomic_properties)
                    
        if self.parent == None:
            self.update_molecular_properties(molecular_database=molDB, properties=properties, atomic_properties=atomic_properties, current_state=current_state)
        
    def get_properties_from_molecule(self, molecule, properties=[], atomic_properties=[]):
        property_values = molecule.__dict__[self.name].__dict__
        for property_name in properties:
            if property_name in molecule.__dict__: property_values[property_name] = molecule.__dict__.pop(property_name)
        for property_name in atomic_properties:
            property_values[property_name] = []
            for atom in molecule.atoms:
                property_values[property_name].append(atom.__dict__.pop(property_name))
            property_values[property_name] = np.array(property_values[property_name]).astype(float)
    
    def update_molecular_properties(self, molecular_database=None, molecule=None, properties=[], atomic_properties=[], current_state=0):
        molDB = molecular_database
        if molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)

        for mol in molDB.molecules:
            for property_name in properties:
                for mol_el_st in mol.electronic_states:
                    mol_el_st.__dict__[property_name] = mol_el_st.__dict__[self.name].__dict__[property_name]
                if not mol.electronic_states:
                    mol.__dict__[property_name] = mol.__dict__[self.name].__dict__[property_name]
                else:
                    mol.__dict__[property_name] = mol.electronic_states[current_state].__dict__[property_name]
            for property_name in atomic_properties:
                for mol_el_st in mol.electronic_states:
                    for iatom in range(len(mol_el_st.atoms)):
                        mol_el_st.atoms[iatom].__dict__[property_name] = mol_el_st.__dict__[self.name].__dict__[property_name][iatom]
                if not mol.electronic_states:
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].__dict__[property_name] = mol.__dict__[self.name].__dict__[property_name][iatom]
                else:
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].__dict__[property_name] = mol.electronic_states[current_state].atoms[iatom].__dict__[property_name]

    def dump(self, filename=None, format='json'):
        '''
        Dump the object to a file.
        '''
        model_dict = {
            'type': 'model_tree_node',
            'name': self.name,
            'children': [child.dump(format='dict') for child in self.children] if self.children else None,
            'operator': self.operator,
            'model': self.model.dump(format='dict') if self.model else None,
            'nthreads': self.nthreads,
        }

        if format == 'json':
            import json           
            with open(filename, 'w') as f:
                json.dump(model_dict, f, indent=4)
        
        if format == 'dict':
            return model_dict

def load(filename, format=None):
    '''
    Load a saved model object.
    '''
    if filename[-5:] == '.json' or format == 'json':
        try:
            return load_json(filename)
        except:
            pass
    if filename[-5:] == '.npz' or format == 'npz':
        try:
            return load_npz(filename)
        except:
            pass
    
    else:
        return load_pickle(filename)

def load_json(filename):
    import json
    with open(filename) as f:
        model_dict = json.load(f)
    return load_dict(model_dict)

def load_npz(filename):
    pass

def load_pickle(filename):
    import pickle
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_dict(model_dict):
    type = model_dict.pop('type')
    nthreads = model_dict.pop('nthreads') if 'nthreads' in model_dict else 0
    if type == 'method':
        kwargs = {}
        if 'kwargs' in model_dict:
            kwargs = model_dict.pop('kwargs')
        model = methods(**model_dict, **kwargs)

    if type == 'ml_model':
        model = globals()[model_dict['ml_model_type'].split('.')[-1]](**model_dict['kwargs'])

    if type == 'model_tree_node':
        children = [load_dict(child_dict) for child_dict in model_dict['children']] if model_dict['children'] else None
        name = model_dict['name']
        operator = model_dict['operator']
        model = load_dict(model_dict['model']) if model_dict['model'] else None
        model = model_tree_node(name=name, children=children, operator=operator, model=model)

    model.set_num_threads(nthreads)
    return model

if __name__ == '__main__':
    pass