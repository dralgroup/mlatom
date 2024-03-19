'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_DeePMDkit: Interface between DeePMD-kit and MLatom              ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''
from __future__ import annotations
from typing import Any, Union, Dict, Callable
import os, sys, uuid, time, tempfile, subprocess, json
import numpy as np
from .. import constants
from .. import data
from .. import models
from .. import stopper
from ..decorators import doc_inherit

if 'DeePMDkit' in os.environ:
    DeePMDdir = os.environ['DeePMDkit']
    DeePMDbin = DeePMDdir+'/dp'
else:
    print('please set $DeePMDkit')
    DeePMDbin = ''


def molDB2DPdata(dir_name, molecular_database, validation_molecular_database, type_map=None,
                      property_to_learn=None,
                      xyz_derivative_property_to_learn=None):
    if os.path.exists(dir_name):
        if not os.path.isdir(dir_name):
            print(f'file {dir_name} already exists')
            return
    else:
        os.mkdir(dir_name)
    
    if os.path.exists(f'{dir_name}/set.000'):
        print(f'{dir_name}/set.000 already exists')
        return
    else:
        os.mkdir(f'{dir_name}/set.000')

    if os.path.exists(f'{dir_name}/set.001'):
        print(f'{dir_name}/set.001 already exists')
        return
    else:
        os.mkdir(f'{dir_name}/set.001')

    if type_map is None:
        # type_map = list(np.sort(np.unique(molDB.get_element_symbols())))
        type_map = list(sorted(set(molecular_database[0].get_element_symbols()), key=lambda x: list(molDB[0].get_element_symbols()).index(x)))

    with open(f'{dir_name}/type.raw', 'w') as f:
        for element in molecular_database[0].get_element_symbols():
            f.write("%d " % type_map.index(element))

    np.save(f'{dir_name}/set.000/box.npy', np.repeat(100 * np.eye(3).reshape([1, -1]), len(molecular_database), axis=0))
    np.save(f'{dir_name}/set.001/box.npy', np.repeat(100 * np.eye(3).reshape([1, -1]), len(validation_molecular_database), axis=0))
    np.save(f'{dir_name}/set.000/coord.npy', molecular_database.xyz_coordinates.reshape(len(molecular_database), -1))
    np.save(f'{dir_name}/set.001/coord.npy', validation_molecular_database.xyz_coordinates.reshape(len(validation_molecular_database), -1))
    if property_to_learn:
        np.save(f'{dir_name}/set.000/energy.npy', molecular_database.get_properties(property_to_learn))
        np.save(f'{dir_name}/set.001/energy.npy', validation_molecular_database.get_properties(property_to_learn))
    if xyz_derivative_property_to_learn:
        np.save(f'{dir_name}/set.000/force.npy', -1 * molecular_database.get_xyz_vectorial_properties(xyz_derivative_property_to_learn).reshape(len(molecular_database), -1))
        np.save(f'{dir_name}/set.001/force.npy', -1 * validation_molecular_database.get_xyz_vectorial_properties(xyz_derivative_property_to_learn).reshape(len(validation_molecular_database), -1))

class dpmd(models.ml_model, models.tensorflow_model):
    '''
    Create an `DeepPot-SE <https://proceedings.neurips.cc/paper_files/paper/2018/hash/e2ad76f2326fbc6b56a45a56c59fafdb-Abstract.html>`_ model object. 
    
    Interfaces to `DeepMD-kit <https://doi.org/10.1016/j.cpc.2018.03.016>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''

    hyperparameters = models.hyperparameters({
        'early_stopping':           models.hyperparameter(value=True, choices=[True, False], optimization_space='choices', dtype=bool),
        'early_stopping_paticence': models.hyperparameter(value=60, minval=16, maxval=256, optimization_space='linear', dtype=int),
        'early_stopping_threshold': models.hyperparameter(value=0.0000, minval=-0.0001, maxval=0.0001, optimization_space='linear', dtype=float),
        'batch_size':               models.hyperparameter(value=4, minval=1, maxval=1024, optimization_space='linear', dtype=int),
        'disp_freq':                models.hyperparameter(value=100, minval=10, maxval=10000, optimization_space='linear', dtype=int),
        'stop_batch':               models.hyperparameter(value=400000, minval=100, maxval=1000000, optimization_space='linear', dtype=int),
        "start_lr":                 models.hyperparameter(value=0.005, minval=0.0001, maxval=0.01, optimization_space='log', dtype=float),
        "decay_steps":              models.hyperparameter(value=2000, minval=100, maxval=10000, optimization_space='log', dtype=int),
        "decay_rate":               models.hyperparameter(value=0.95, minval=0.5, maxval=1, optimization_space='log', dtype=float),
        "start_pref_e":             models.hyperparameter(value=0.02, minval=0, maxval=100, optimization_space='log', dtype=float),
        "limit_pref_e":             models.hyperparameter(value=2, minval=0, maxval=1000, optimization_space='log', dtype=float),
        "start_pref_f":             models.hyperparameter(value=1000, minval=0, maxval=1000, optimization_space='log', dtype=float),
        "limit_pref_f":             models.hyperparameter(value=1, minval=0, maxval=1000, optimization_space='log', dtype=float),
        "start_pref_v":             models.hyperparameter(value=0, minval=0, maxval=1000, optimization_space='log', dtype=float),
        "limit_pref_v":             models.hyperparameter(value=0, minval=0, maxval=1000, optimization_space='log', dtype=float),
        'rcut':                     models.hyperparameter(value=6.0, minval=1.0, maxval=10.0, optimization_space='linear', dtype=float),
        'neuron':                   models.hyperparameter('30,60'),
        'n_neuron':                 models.hyperparameter('80,80,80'),
        'sel':                      models.hyperparameter(value=16, minval=1, maxval=32, optimization_space='linear', dtype=str),
    })

    json = {
        "model": {
            "descriptor": {
                "type": "se_a",
                "sel": [
                    16,
                    16,
                    16 
                ],
                "rcut_smth": 6.3,
                "rcut": 6.5,
                "neuron": [
                    30,
                    60
                ],
                "resnet_dt": False,
                "axis_neuron": 6,
                "seed": 42
            },
            "fitting_net": {
                "n_neuron": [
                    80,
                    80,
                    80
                ],
                "resnet_dt": True,
                "seed": 42
            },
            "type_map": [
                "C",
                "H",
                "O"
            ]
        },
        "learning_rate": {
            "type": "exp",
            "start_lr": 0.005,
            "decay_steps": 2000,
            "decay_rate": 0.95
        },
        "loss": {
            "start_pref_e": 0.02,
            "limit_pref_e": 2,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 0,
            "limit_pref_v": 0
        },
        "training": {
            "systems": [
                "./"
            ],
            "set_prefix": "set",
            "stop_batch": 400000,
            "batch_size": 4,
            "seed": 42,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "numb_test": 10,
            "save_freq": 100,
            "save_ckpt": "model.ckpt",
            "load_ckpt": "model.ckpt",
            "disp_training": True,
            "time_training": True,
            "profiling": False,
            "profiling_file": "timeline.json"
        }
    }

    type_map = []
    json_input = None
    property_name = 'y'
    program = 'DeepMD-kit'
    meta_data = {
        "genre": "neural network"
    }
    model_file = None
    model = None
    verbose = False

    def __init__(self, model_file=None, hyperparameters={}, verbose=1,):
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        if model_file: 
            self.model_file = model_file
            if os.path.exists(self.model_file):
                if os.path.exists(f"{self.model_file}.json"):
                    with open(f"{self.model_file}.json") as f:
                        self.json = json.load(f)
                        self.type_map = self.json["model"]["type_map"]
        else:
            self.model_file = f'DPMD_{str(uuid.uuid4())}.pb'

    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'deepmd' in args.data and hyperparam in args.deepmd.data:
                self.hyperparameters[hyperparam].value = args.deepmd.data[hyperparam]

        if 'deepmd' in args.data:
            if 'input' in args.deepmd.data:
                self.update_json(json_input=args.deepmd.input)

    def update_json(self, **kwargs):
        if 'json_input' in kwargs and kwargs['json_input']:
            with open(kwargs['json_input'], 'r') as f:
                self.json.update(json.load(f))
            self.json['input'] =  kwargs['json_input']
        else:
            self.json["model"]["type_map"] = self.type_map
            if len(self.json['model']['descriptor']['sel']) != len(self.type_map):
                self.json['model']['descriptor']['sel'] = [self.json['model']['descriptor']['sel'][0]] * len(self.type_map)
            self.json['training']['batch_size'] = kwargs['batch_size'].value
            self.json['training']['stop_batch'] = kwargs['stop_batch'].value
            self.json['training']['disp_freq'] = kwargs['disp_freq'].value
            self.json['training']['save_freq'] = kwargs['disp_freq'].value
            self.json['learning_rate']["start_lr"]    = kwargs["start_lr"].value
            self.json['learning_rate']["decay_steps"] = kwargs["decay_steps"].value
            self.json['learning_rate']["decay_rate"]  = kwargs["decay_rate"].value
            self.json['loss']["start_pref_e"] = kwargs["start_pref_e"].value
            self.json['loss']["limit_pref_e"] = kwargs["limit_pref_e"].value
            self.json['loss']["start_pref_f"] = kwargs["start_pref_f"].value
            self.json['loss']["limit_pref_f"] = kwargs["limit_pref_f"].value
            self.json['loss']["start_pref_v"] = kwargs["start_pref_v"].value
            self.json['loss']["limit_pref_v"] = kwargs["limit_pref_v"].value
            if 'property_to_learn' in kwargs and not kwargs['property_to_learn']:
                self.json['loss']['limit_pref_e'] = 0
                self.json['loss']['start_pref_e'] = 0
            if 'xyz_derivative_property_to_learn' in kwargs and not kwargs['xyz_derivative_property_to_learn']:
                self.json['loss']['limit_pref_f'] = 0
                self.json['loss']['start_pref_f'] = 0
            self.json['model']["descriptor"]['sel'] = [int(i) for i in kwargs["sel"].value.strip('[]').split(',')]
            if len(self.json['model']["descriptor"]['sel']) == 1:
                self.json['model']["descriptor"]['sel'] = self.json['model']["descriptor"]['sel'] * len(self.json["model"]["type_map"])
            self.json['model']["descriptor"]['rcut'] = kwargs["rcut"].value
            self.json['model']["descriptor"]['neuron'] = [int(i) for i in kwargs["neuron"].value.strip('[]').split(',')]
            self.json['model']["fitting_net"]['n_neuron'] = [int(i) for i in kwargs["n_neuron"].value.strip('[]').split(',')]

        self.type_map = self.json["model"]["type_map"]
        self.json['training']['set_prefix'] = 'set'
        if 'dirname' in kwargs:
            self.json['training']['systems'] = kwargs['dirname']
        with open(f"{self.model_file}.json", "w") as f:
            json.dump(self.json, f, indent=4)
        
    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        hyperparameters: Union[Dict[str,Any], models.hyperparameters] = {},
        spliting_ratio=0.8,
        stdout=None,
        stderr=None,
        json_input=None,
        dirname=None,
    ):
        self.hyperparameters.update(hyperparameters)

        FNULL = open(os.devnull, 'w')
        if not self.verbose:
            if not stdout:
                stdout = FNULL
            if not stderr:
                stderr = FNULL
        else:
            stdout = sys.stdout
            stderr = sys.stderr

        if os.path.exists(self.model_file):
            print(f'{self.model_file} exists')
            self.model_file = f'DPMD_{str(uuid.uuid4())}.pb'

        if self.verbose:
            print(f'Trained DPMD model will be saved in {self.model_file}.')

        if dirname:
            rundir = dirname
        else:
            tmpdir = tempfile.TemporaryDirectory()
            rundir = tmpdir.name

        if not self.type_map:
            self.type_map = list(sorted(set(molecular_database[0].get_element_symbols()), key=lambda x: list(molecular_database[0].get_element_symbols()).index(x)))
        
        self.update_json(**self.hyperparameters, 
                         json_input=json_input if json_input else self.json_input, 
                         dirname=rundir,
                         property_to_learn=property_to_learn,
                         xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        
        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]
        elif not validation_molecular_database:
            raise NotImplementedError("please specify validation_molecular_database or set it to 'sample_from_molecular_database'")

        molDB2DPdata(rundir, molecular_database, validation_molecular_database, self.json["model"]["type_map"], property_to_learn, xyz_derivative_property_to_learn)

        if self.hyperparameters['early_stopping'].value:
            earlyStop = earlyStopCls(
                patience=self.hyperparameters['early_stopping_paticence'].value, threshold=self.hyperparameters['early_stopping_threshold'].value
            )

        if self.verbose:
            print(f' > {DeePMDbin} train {self.model_file}.json')
            sys.stdout.flush()

        if os.path.exists(self.json['training']['disp_file']):
            os.remove(self.json['training']['disp_file'])

        proc = subprocess.Popen([DeePMDbin, "train", f"{self.model_file}.json"], stdout=subprocess.PIPE ,stderr=stderr, universal_newlines=True)
        for line in proc.stdout:
            if self.verbose:
                print(line.replace('\n',''))
            if self.hyperparameters['early_stopping'].value:
                try:
                    lastline=subprocess.check_output(['tail', '-1', self.json['training']['disp_file']], stderr=FNULL).split()
                    loss = float(lastline[1])
                    n_batch = float(lastline[0])
                    if earlyStop.current(loss, n_batch, shutup=self.shutup):
                        print('met early-stopping conditions')
                        sys.stdout.flush()
                        proc.terminate()
                except:
                    pass
            sys.stdout.flush()
        
        if not dirname:
            tmpdir.cleanup()

        if self.verbose:
            print(' > %s freeze -o %s' % (DeePMDbin, self.model_file))
            sys.stdout.flush()
        subprocess.call([DeePMDbin, "freeze", "-o", self.model_file], stdout=stdout,stderr=stderr)

        if os.path.exists(self.model_file) and self.verbose:
            print('model saved in %s' % self.model_file)

        FNULL.close()
    
    @doc_inherit
    def predict(
        self, 
        molecular_database: data.molecular_database = None, 
        molecule: data.molecule = None,
        calculate_energy: bool = False,
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False,
        property_to_predict: Union[str, None] = 'estimated_y', 
        xyz_derivative_property_to_predict: Union[str, None] = None, 
        hessian_to_predict: Union[str, None] = None,
    ) -> None:
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            np.save(f'{tmpdirname}/coord.npy', molDB.xyz_coordinates.reshape(len(molDB), -1))
            if not self.type_map:
                self.type_map = list(sorted(set(molDB[0].get_element_symbols()), key=lambda x: list(molDB[0].get_element_symbols()).index(x)))
            atom_type = [self.type_map.index(element) for element in molDB[0].get_element_symbols()]
            code = f'''import deepmd.DeepPot as DP
import numpy as np
import sys

coord = np.load('{tmpdirname}/coord.npy')
cell = np.repeat(100 * np.eye(3).reshape([1, -1]), len(coord), axis=0)
atom_type = {atom_type}
dp = DP('{self.model_file}')
e, f, v = dp.eval(coord, cell, atom_type)
np.save('{tmpdirname}/energy.npy', e)
np.save('{tmpdirname}/force.npy', f)
'''

            FNULL = open(os.devnull, 'w')
            subprocess.call([DeePMDdir+"/python", "-c", code],stdout=FNULL , stderr=FNULL)
            FNULL.close()


            if property_to_predict:
                enest = np.load(f'{tmpdirname}/energy.npy').flatten()
                molDB.add_scalar_properties(enest, property_to_predict)

            if xyz_derivative_property_to_predict:
                gradest = np.load(f'{tmpdirname}/force.npy') * -1
                
                molDB.add_xyz_vectorial_properties(gradest, xyz_derivative_property_to_predict)

class earlyStopCls():
    def __init__(self, patience = 60, threshold = 0.0001 ):
        self.bestloss = np.inf
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.last_batch =0
    
    def current(self, loss, n_batch, verbose=1):
        if n_batch == self.last_batch:
            return False
        else:
            self.last_batch = n_batch

        if loss < (1 - self.threshold)*self.bestloss:
            self.bestloss = loss
            self.counter = 0
        else:
            self.counter += 1

        if verbose:
            print('bestloss: %s   early-stopping counter: %s/%s'% (self.bestloss, self.counter, self.patience))
            sys.stdout.flush()
        return self.counter > self.patience
    
def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
  To use Interface_DeePMDkit, please define environmental variable $DeePMDkit
  to where dp binary is located (e.g "/home/xxx/deepmd-kit-1.2/bin").

  Arguments with their default values:
    MLprog=DeePMD-kit          enables this interface
    MLmodelType=S              requests model S
      DeepPot-SE               [defaut]
      DPMD
      
      deepmd.stop_batch=4000000        
                      number of batches to be trained before stopping       
      deepmd.batch_size=32 
                      size of each batch
      deepmd.start_lr=0.001
                               initial learning rate
      deepmd.decay_steps=4000
                               number of batches for one decay of learning rate
      deepmd.decay_rate=0.95
                               decay rate of each decay of learning rate
      deepmd.rcut=6.0        
                               cutoff radius for local environment
      deepmd.start_pref_e/limit_pref_e/start_pref_f/limit_pref_f
                               parameters for loss prefactors
      deepmd.sel=16 or deepmd.sel=a,b,c
                               number of selected atoms for each specie
      deepmd.neuron=30,60
                               NN structure of descriptor
      deepmd.n_neuron=80,80,80
                               NN structure of fitting network
        
      deepmd.input=S           file S with DeePMD input parameters
                               in json format

  Cite DeePMD-kit:
    H. Wang, L. Zhang, J. Han, W. E, Comput. Phys. Commun. 2018, 228, 178
    
  Cite DeepPot-SE method, if you use it:
    L.F. Zhang, J.Q. Han, H. Wang, W.A. Saidi, R. Car, W.N. E,
    Adv. Neural. Inf. Process. Syst. 2018, 31, 4436
    
  Cite DPMD method, if you use it:
    L. Hang, J. Han, H. Wang, R. Car, W. E, Phys. Rev. Lett. 2018, 120, 143001
'''
    print(helpText)