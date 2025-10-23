#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! model_cls: Module with generic classes for models                         ! 
  ! Implementations by: Pavlo O. Dral, Fuchun Ge, Yi-Fan Hou, Yuxinxin Chen,  !
  !                     Peikun Zheng                                          ! 
  !---------------------------------------------------------------------------! 
'''
from __future__ import annotations
from typing import Any, Union, Iterable, Callable
import os, sys, shutil
import numpy as np
from collections import UserDict

from . import data, stats
from .decorators import doc_inherit

class model():
    '''
    Parent (super) class for models to enable useful features such as logging during geometry optimizations.
    '''
    nthreads = 0
    def set_num_threads(self, nthreads=0):
        # implement for each subclass
        if nthreads:
            self.nthreads = nthreads

    def config_multiprocessing(self):
        '''
        for scripts that need to be executed before running model in parallel
        '''
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
        **kwargs):
        self.predict(molecule=molecule,
                     **kwargs)
        if np.any(np.isnan(molecule.energy_gradients)):
            print(' * Warning* No gradients were calculated, check the logs for any reasons of this critical failure.')
            if 'error_message' in molecule.__dict__:
                print(' Error message retrieved from the calculations:')
                print('-'*10)
                print(molecule.error_message)
                print('-'*10)
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

    def dump(self,filename=None,format='json'):
        modelname = self.__class__.__name__
        modulename = self.__module__
        modulepath = sys.modules[modulename].__spec__.origin
        model_dict = {
            'type': modelname,
            'module':{
                'path':modulepath,
                'name':modulename
                }}

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

class method_model(model):
    @classmethod
    def is_method_supported(cls, method):
        if 'supported_methods' in cls.__dict__:
            if method.casefold() in [m.casefold() for m in cls.supported_methods]:
                return True
            else:
                return False
        else:
            return None
        
    @classmethod
    def is_program_found(cls):
        if 'bin_env_name' in cls.__dict__:
            bin_env_name = cls.get_bin_env_var()
            if bin_env_name is None:
                return False
            else:
                return True
        else:
            return None

    @classmethod
    def get_bin_env_var(cls):
        if cls.bin_env_name in os.environ:
            return os.environ[cls.bin_env_name]
        else:
            return None
    
    @classmethod
    def raise_unsupported_method_error(cls, method):
        raise ValueError(f'The method "{method}" is not supported by this class. You might have misspelled method, please check the class documentation.')

class downloadable_model(model):

    def check_model_path(self, model_dir=None, model_files=None):
        '''
        Check if model exists in recognizable paths. Available paths are: 
            - $MODELSPATH
            - [mlatom dir]/models 
            - ~/.mlatom/models
        Model will only be downloaded to ~/.mlatom/models.

        Parameters:
            model_dir (str): The name of the folder that contains all the model files. 
            model_files (list, str): The list of the model files inside the model directory.

        Returns:
            mlatom_model_dir (str): the model dir with the mlatom recognizable prefix
            to_download (bool): whether download is needed
        ''' 

        assert model_dir is not None, 'please provide the model directory'
        assert model_files is not None, 'Please provide model files inside model directory'
        
        to_download = False

        def check_model_files(prefix, model_dir, model_files):
            model_path = os.path.join(prefix, model_dir)
            if not isinstance(model_files, list):
                model_files = [model_files]
            for mf in model_files:
                if not os.path.exists(os.path.join(model_path, mf)): 
                    return model_path, True
            return model_path, False

        # check if model exists in $MODELSPATH/[model_dir]/
        if 'MODELSPATH' in os.environ:
            modelspath = os.environ['MODELSPATH'].split(os.pathsep)
            for mm in modelspath:
                model_path, to_download = check_model_files(mm, model_dir, model_files)
                if not to_download: return model_path, to_download

        # check location of mlatom folder
        if 'mlatom' in sys.modules:
            prefix2 = os.path.join(os.path.dirname(sys.modules['mlatom'].__file__), 'mlatom_models')
        elif 'aitomic' in sys.modules:
            prefix2 = os.path.join(os.path.dirname(sys.modules['aitomic'].__file__), 'mlatom_models')
        else:
            raise ValueError('Neither aitomic nor mlatom is imported.')
        
        # check if model exists in mlatom/[model_dir]
        if os.path.exists(prefix2):
            model_path, to_download = check_model_files(prefix2, model_dir, model_files)
            if not to_download: return model_path, to_download

        # check location of mlatom default models folder
        home_dir = os.path.expanduser("~")
        prefix3 = os.path.join(home_dir,'.mlatom/models')
        
        # check if model exists in ~/.mlatom/models/[model_dir]/
        if os.path.exists(prefix3):
            model_path, to_download = check_model_files(prefix3, model_dir, model_files)
            if not to_download: return model_path, to_download
        
        # create ~/.mlatom/[model_dir]
        model_path = os.path.join(prefix3, model_dir)
        os.makedirs(model_path, exist_ok=True)

        return model_path, to_download
    
    def _download(self, link=None, headers=None, target=None):

        # download to [target].temp

        import requests
        from tqdm import tqdm

        assert link is not None, "Please provide downloadable link to the file"
        assert target is not None, "Please provide target directory to be downloaded"
        print(f'Start downloading model from {link} to {target}'); sys.stdout.flush()
        try:
            response = requests.get(link, headers=headers, stream=True, allow_redirects=True)
            total_size = int(response.headers.get("content-length", 0))
            target += '.temp'

            with open(target, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=target) as pbar:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk: 
                            f.write(chunk)
                            pbar.update(len(chunk))
            return target
        except: 
            print(f'Failed to download model from {link} to {target}'); sys.stdout.flush()
            return None
    
    def extract_zip(self, src=None, target=None):
        import zipfile
        with zipfile.ZipFile(src, 'r') as zipf:
            zipf.extractall(target)
        os.remove(src)

    def flatten(self, target):
        for root, dirs, files in os.walk(target):
            for ff in files:
                shutil.move(os.path.join(root, ff), target)

    def download(self, download_links, target, extract=True, flatten=True):
        '''
        Download models from links to defined folder. 
        
        Parameters:
            download_links (list, str): a list of links to get files from
            target (str): the target path to be downloaded
            extract (bool): whether to extract the downloaded files
            flatten (bool): whether to flatten the folders in the downloaded files
        '''

        if not isinstance(download_links, list):
            download_links = [download_links]
        for dlink in download_links:
            downloaded_file = self._download(link=dlink, target=target)
            if downloaded_file is not None:
                if extract:
                    # zipped folder
                    self.extract_zip(src=downloaded_file, target=target)
                    if flatten:
                        self.flatten(target)
                else:
                    # file
                    shutil.move(downloaded_file, target)
                return 
        
        link_string = '\n'.join([f'link{ii}: {download_links[ii]}' for ii in range(len(download_links))])
        raise ValueError(f'Failed to download required model files. Possible solutions:\n 1. Check your internet connection.\n 2. Download from links below:\n{link_string}\nThe model .pt files should be placed under {target}')
 
# Parent model class
class ml_model(model):
    '''
    Useful as a superclass for the ML models that need to be trained.
    '''
    def train(
        self,
        ml_database: data.ml_database,
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
        ml_database: data.ml_database = None, entry: data.entry = None,
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
        '''
        Generates model dictionary for dumping in json format.
        '''
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
        '''
        Resets model (deletes the ML model file from the hard disk).
        '''
        if os.path.exists(self.model_file): os.remove(self.model_file)

    def dump(self, filename=None, format='json'):
        '''
        Dumps model class object information in a json file (do not confused with saving the model itself, i.e., its parameters!).
        '''
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
                                 cv_splits_ml_databases=None,
                                 cv_splits_molecular_databases=None, calculate_CV_split_errors=False,
                                 subtraining_ml_database=None, validation_ml_database=None,
                                 subtraining_molecular_database=None, validation_molecular_database=None,
                                 validation_loss_function=None, validation_loss_function_kwargs={},
                                 debug=False):
        '''
        Returns the validation loss for the given hyperparameters.
        
        By default, the validation loss is RMSE evaluated as a geometric mean of scalar and vectorial properties, e.g., energies and gradients.
        
        Arguments:
        
            training_kwargs (dict, optional): the kwargs to be passed to ``yourmodel.train()`` function.
            prediction_kwargs (dict, optional): the kwargs to be passed to ``yourmodel.predict()`` function.
            cv_splits_molecular_databases (list, optional): the list with cross-validation splits, each element is :class:`molecular_database <mlatom.data.molecular_database>`.
            calculate_CV_split_errors (bool, optional): requests to return the errors for each cross-validation split as a list in addtion to the aggregate cross-validation error.
            subtraining_molecular_database (:class:`molecular_database <mlatom.data.molecular_database>`, optional): molecular database for sub-training to be passed to ``yourmodel.train()`` function.
            validation_molecular_database (:class:`molecular_database <mlatom.data.molecular_database>`, optional): molecular database for validation to be passed to ``yourmodel.predict()`` function.
            validation_loss_function (function, optional): user-defined validation function.
            validation_loss_function_kwargs (dict, optional): kwargs for above ``validation_loss_function``.
        '''
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
            self.holdout_validation(subtraining_ml_database=subtraining_ml_database,
                                    validation_ml_database=validation_ml_database,
                                    subtraining_molecular_database=subtraining_molecular_database,
                                    validation_molecular_database=validation_molecular_database,
                                    training_kwargs=training_kwargs,
                                    prediction_kwargs=prediction_kwargs)
            if not subtraining_ml_database is None and not validation_ml_database is None:
                if all(isinstance(val, (list, np.ndarray)) for val in validation_ml_database.y) and xyz_derivative_property_to_learn == None:
                    xyz_derivative_property_to_learn = property_to_learn
                    property_to_learn = None
                    y = None
                    estimated_y = None
                    xyz_derivatives = validation_ml_database.y 
                    estimated_xyz_derivatives = validation_ml_database.get_property(property_name=property_to_predict)
                else:
                    y = validation_ml_database.y 
                    estimated_y = validation_ml_database.get_property(property_name=property_to_predict)
                    xyz_derivatives = None 
                    estimated_xyz_derivatives = None
            if not subtraining_molecular_database is None and not validation_molecular_database is None:
                if property_to_learn != None:
                    y = validation_molecular_database.get_properties(property_name=property_to_learn)
                    estimated_y = validation_molecular_database.get_properties(property_name=property_to_predict)
                if xyz_derivative_property_to_learn != None:
                    xyz_derivatives = validation_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_learn)
                    estimated_xyz_derivatives = validation_molecular_database.get_xyz_vectorial_properties(property_name=xyz_derivative_property_to_predict)
        else:
            self.cross_validation(cv_splits_ml_databases=cv_splits_ml_databases,
                                    cv_splits_molecular_databases=cv_splits_molecular_databases,
                                    training_kwargs=training_kwargs,
                                    prediction_kwargs=prediction_kwargs)
            if not cv_splits_ml_databases is None:
                training_ml_database = data.ml_database()
                if calculate_CV_split_errors:
                    nsplits = len(cv_splits_ml_databases)
                    CV_y=[None for ii in range(nsplits)]; CV_yest=[None for ii in range(nsplits)]; CV_xyz_derivatives=[None for ii in range(nsplits)]; CV_estimated_xyz_derivatives=[None for ii in range(nsplits)]
                for CVsplit in cv_splits_ml_databases:
                    training_molecular_database += CVsplit 
                if all(isinstance(val, (list, np.ndarray)) for val in training_ml_database.y) and xyz_derivative_property_to_learn == None:
                    xyz_derivative_property_to_learn = property_to_learn
                    property_to_learn = None
                    y = None
                    estimated_y = None
                    xyz_derivatives = training_ml_database.y 
                    estimated_xyz_derivatives = training_ml_database.get_property(property_name=property_to_predict)
                else:
                    y = training_ml_database.y 
                    estimated_y = training_ml_database.get_property(property_name=property_to_predict)
                    xyz_derivatives = None 
                    estimated_xyz_derivatives = None
                if calculate_CV_split_errors:
                    CV_y = []; CV_yest = [] 
                    for ii in range(nsplits):
                        CV_y.append(cv_splits_ml_databases[ii].y)
                        CV_yest.append(cv_splits_ml_databases[ii].get_property(property_name=property_to_predict))
            if not cv_splits_molecular_databases is None:
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
                                 cv_splits_ml_databases=None,
                                 cv_splits_molecular_databases=None,
                                 subtraining_ml_database=None, validation_ml_database=None,
                                 subtraining_molecular_database=None, validation_molecular_database=None,
                                 optimization_algorithm=None, optimization_algorithm_kwargs={},
                                 maximum_evaluations=10000,
                                 validation_loss_function=None, validation_loss_function_kwargs={},
                                 debug=False):
        '''
        Optimizes hyperparameters by minimizing the validation loss.
        
        By default, the validation loss is RMSE evaluated as a geometric mean of scalar and vectorial properties, e.g., energies and gradients.
        
        Arguments:
        
            hyperparameters (list, required): the list with strings - names of hyperparameters. Hyperparameters themselves must be in ``youmodel.hyperparameters`` defined with class instance :class:`hyperparameters <mlatom.models.hyperparameters>` consisting of :class:`hyperparameter <mlatom.models.hyperparameter>` defining the optimization space.
            training_kwargs (dict, optional): the kwargs to be passed to ``yourmodel.train()`` function.
            prediction_kwargs (dict, optional): the kwargs to be passed to ``yourmodel.predict()`` function.
            cv_splits_molecular_databases (list, optional): the list with cross-validation splits, each element is :class:`molecular_database <mlatom.data.molecular_database>`.
            calculate_CV_split_errors (bool, optional): requests to return the errors for each cross-validation split as a list in addtion to the aggregate cross-validation error.
            subtraining_molecular_database (:class:`molecular_database <mlatom.data.molecular_database>`, optional): molecular database for sub-training to be passed to ``yourmodel.train()`` function.
            validation_molecular_database (:class:`molecular_database <mlatom.data.molecular_database>`, optional): molecular database for validation to be passed to ``yourmodel.predict()`` function.
            validation_loss_function (function, optional): user-defined validation function.
            validation_loss_function_kwargs (dict, optional): kwargs for above ``validation_loss_function``.
            optimization_algorithm (str, required): optimization algorithm. No default, must be specified among: 'grid' ('brute'), 'TPE', 'Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-krylov', 'trust-exact'.
            optimization_algorithm_kwargs (dict, optional): kwargs to be passed to optimization algorithm, e.g., ``{'grid_size': 5}`` (default 9 for the grid search).
            maximum_evaluations (int, optional): maximum number of optimization evaluations (default: 10000) supported by all optimizers except for grid search.
            
        Saves the final hyperparameters in ``yourmodel.hyperparameters`` adn validation loss in ``yourmodel.validation_loss``.
        '''
    
        def validation_loss(current_hyperparameters):
            for ii in range(len(current_hyperparameters)):
                self.hyperparameters[hyperparameters[ii]].value = current_hyperparameters[ii]
            return self.calculate_validation_loss(  training_kwargs=training_kwargs,
                                                    prediction_kwargs=prediction_kwargs,
                                                    cv_splits_ml_databases=cv_splits_ml_databases,
                                                    cv_splits_molecular_databases=cv_splits_molecular_databases,
                                                    subtraining_ml_database=subtraining_ml_database,
                                                    validation_ml_database=validation_ml_database,
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
                    # if 'grid_range' in self.hyperparameter[key].__dict__: 
                    #     grid_slices.append(self.hyperparameter[key].grid_range) #@Yifan
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
                # initial_hyperparameters = [{key: self.hyperparameters[key] for key in hyperparameters}]
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

    def holdout_validation(self, subtraining_ml_database=None, validation_ml_database=None,
                           subtraining_molecular_database=None, validation_molecular_database=None,
                     training_kwargs=None, prediction_kwargs=None):
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        if not subtraining_ml_database is None and not validation_ml_database is None:
            self.train(ml_database=subtraining_ml_database, **training_kwargs)
            self.predict(ml_database=validation_ml_database, **prediction_kwargs)
        elif not subtraining_molecular_database is None and not validation_molecular_database is None:
            self.train(molecular_database=subtraining_molecular_database, **training_kwargs)
            self.predict(molecular_database=validation_molecular_database, **prediction_kwargs)
        

    def cross_validation(self, cv_splits_ml_databases=None,
                         cv_splits_molecular_databases=None,
                     training_kwargs=None, prediction_kwargs=None):
        
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        
        if not cv_splits_ml_databases is None:
            nsplits = len(cv_splits_ml_databases)
            for ii in range(nsplits):
                subtraining_ml_database = data.molecular_database()
                for jj in range(nsplits):
                    if ii != jj: subtraining_ml_database += cv_splits_ml_databases[jj]
                validation_ml_database = cv_splits_ml_databases[ii]
                self.reset()
                self.train(**training_kwargs)
                # self.train(ml_database=subtraining_ml_database, **training_kwargs)
                self.predict(**prediction_kwargs)
                # self.predict(ml_database=validation_ml_database, **prediction_kwargs)
        if not cv_splits_molecular_databases is None:
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
    '''
    Optimizes on the given grid by finding parameters (provided by grid) leading to the minimum value of the given function.
    '''
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
                if 'weight' in child.__dict__.keys():
                    mol.__dict__[child.name].__dict__['weight'] = child.weight

            if self.operator == 'sum':
                for mol in molDB.molecules:
                    if not mol.electronic_states:
                        mol.__dict__[self.name].sum(properties+atomic_properties)
                    for mol_el_st in mol.electronic_states:
                        mol_el_st.__dict__[self.name].sum(properties+atomic_properties)
            if self.operator == 'weighted_sum':
                for mol in molDB.molecules:
                    if not mol.electronic_states:
                        mol.__dict__[self.name].weighted_sum(properties+atomic_properties)
                    for mol_el_st in mol.electronic_states:
                        mol_el_st.__dict__[self.name].weighted_sum(properties+atomic_properties)
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
            'weight': self.weight if 'weight' in self.__dict__ else None
        }

        if format == 'json':
            import json           
            with open(filename, 'w') as f:
                json.dump(model_dict, f, indent=4)
        
        if format == 'dict':
            return model_dict

if __name__ == '__main__':
    pass