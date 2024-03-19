'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_PhysNet: Interface between PhysNet and MLatom                   ! 
  ! Implementations by: Fuchun Ge and Max Pinheiro Jr                         ! 
  !---------------------------------------------------------------------------! 
'''

from __future__ import annotations
from typing import Any, Union, Dict
import os, sys, uuid, time
import numpy as np

if 'PhysNet' in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    PhysNetdir = os.environ['PhysNet']
    sys.path.append(os.path.dirname(PhysNetdir))
    PhysNet = __import__(os.path.basename(PhysNetdir))
    from PhysNet.neural_network.NeuralNetwork import *
    from PhysNet.neural_network.activation_fn import *
    from PhysNet.training.Trainer        import *
    from PhysNet.training.DataContainer import *
    from PhysNet.training.DataProvider  import *
    from PhysNet.training.DataQueue     import *
else:
    DataContainer = object
    print('Please specify PhysNet installation dir in $PhysNet')
    pass

import logging
import string
import random
# tf.get_logger().setLevel(logging.ERROR)
# tf.autograph.set_verbosity(1)

from .. import constants
from .. import data
from .. import models
from .. import stopper
from ..decorators import doc_inherit

def molDB2PhysNetData(molDB, 
                      property_to_learn=None,
                      xyz_derivative_property_to_learn = None):
    dataset =  {
        'N': molDB.get_number_of_atoms(),
        'R': molDB.xyz_coordinates,
        'Z': molDB.get_atomic_numbers(),
    }
    if xyz_derivative_property_to_learn: 
        dataset['F'] = -1 * molDB.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
    if property_to_learn: 
        dataset['E'] = molDB.get_properties(property_to_learn)
    return _DataContainer(dataset)

class physnet(models.ml_model, models.tensorflow_model):
    '''
    Create an `PhysNet <https://doi.org/10.1021/acs.jctc.9b00181>`_ model object. 
    
    Interfaces to `PhysNet <https://github.com/MMunibas/PhysNet>`_ program.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''
    hyperparameters = models.hyperparameters({
        'earlystopping':                models.hyperparameter(value=1, choices=[0, 1], optimization_space='choices', dtype=int),
        'threshold':                    models.hyperparameter(value=0.0000, minval=-0.0001, maxval=0.0001, optimization_space='linear', dtype=float),
        'patience':                     models.hyperparameter(value=60, minval=0, maxval=1, optimization_space='linear', dtype=int),
        'restart':                      models.hyperparameter(value=0, choices=[0, 1], optimization_space='choices', dtype=int),
        'num_features':                 models.hyperparameter(value=128, minval=1, maxval=256, optimization_space='linear', dtype=int),
        'num_basis':                    models.hyperparameter(value=64, minval=1, maxval=256, optimization_space='linear', dtype=int),
        'num_blocks':                   models.hyperparameter(value=5, minval=1, maxval=16, optimization_space='linear', dtype=int),
        'num_residual_atomic':          models.hyperparameter(value=2, minval=1, maxval=16, optimization_space='linear', dtype=int),
        'num_residual_interaction':     models.hyperparameter(value=3, minval=1, maxval=16, optimization_space='linear', dtype=int),
        'num_residual_output':          models.hyperparameter(value=1, minval=1, maxval=4, optimization_space='linear', dtype=int),
        'cutoff':                       models.hyperparameter(value=10.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'use_electrostatic':            models.hyperparameter(value=0, choices=[0, 1], optimization_space='choices', dtype=int),
        'use_dispersion':               models.hyperparameter(value=0, choices=[0, 1], optimization_space='choices', dtype=int),
        'grimme_s6':                    models.hyperparameter(value=0.5, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'grimme_s8':                    models.hyperparameter(value=0.2130, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'grimme_a1':                    models.hyperparameter(value=0.0, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'grimme_a2':                    models.hyperparameter(value=6.0519, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'seed':                         models.hyperparameter(value=42, minval=0, maxval=1, optimization_space='linear', dtype=int),
        'max_steps':                    models.hyperparameter(value=1024, minval=0, maxval=1, optimization_space='linear', dtype=int),
        'learning_rate':                models.hyperparameter(value=0.0008, minval=0.0001, maxval=0.01, optimization_space='linear', dtype=float),
        'max_norm':                     models.hyperparameter(value=1000.0, minval=1, maxval=10000, optimization_space='linear', dtype=float),
        'ema_decay':                    models.hyperparameter(value=0.999, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'keep_prob':                    models.hyperparameter(value=1.0, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'l2lambda':                     models.hyperparameter(value=0.0, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'nhlambda':                     models.hyperparameter(value=0.01, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'decay_steps':                  models.hyperparameter(value=10000000, minval=10, maxval=10000000, optimization_space='linear', dtype=int),
        'decay_rate':                   models.hyperparameter(value=0.1, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'batch_size':                   models.hyperparameter(value=12, minval=0, maxval=1, optimization_space='linear', dtype=int),
        'valid_batch_size':             models.hyperparameter(value=2, minval=0, maxval=1, optimization_space='linear', dtype=int),
        'force_weight':                 models.hyperparameter(value=52.91772105638412, minval=0, maxval=100, optimization_space='linear', dtype=float),
        'charge_weight':                models.hyperparameter(value=0, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'dipole_weight':                models.hyperparameter(value=0, minval=0, maxval=1, optimization_space='linear', dtype=float),
        'summary_interval':             models.hyperparameter(value=0, minval=1, maxval=1000, optimization_space='linear', dtype=int),
        'validation_interval':          models.hyperparameter(value=0, minval=1, maxval=1000, optimization_space='linear', dtype=int),
        'save_interval':                models.hyperparameter(value=0, minval=1, maxval=1000, optimization_space='linear', dtype=int),
    })

    property_name = 'y'
    meta_data = {
        "genre": "neural network"
    }
    model_file = None
    model = None
    verbose = 1
    session = None

    def __init__(self, model_file=None, hyperparameters={}, verbose=1,):
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        self.new_session()
        self.set_best_dict()
        if model_file: 
            if os.path.exists(model_file):
                self.load(model_file)
            else:
                if self.verbose: print(f'the trained PhysNet model will be saved in {model_file}')
                self.model_file = model_file

    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'physnet' in args.data and hyperparam in args.physnet.data:
                self.hyperparameters[hyperparam].value = args.physnet.data[hyperparam]
    
    def reset(self):
        super().reset()
        self.model = None
        self.new_session()

    def set_best_dict(self, **kwargs):
        if kwargs:
            self.best_dict.update(kwargs)
        else:
            self.best_dict = {
                'loss':    np.Inf ,
                'emae':    np.Inf,
                'ermse':   np.Inf,
                'fmae':    np.Inf,
                'frmse':   np.Inf,
                'qmae':    np.Inf,
                'qrmse':   np.Inf,
                'dmae':    np.Inf,
                'drmse':   np.Inf,
                'step':    0.,
            }

    def new_session(self):
        if self.session:
            self.session.close()
        self.session = tf.Session()

    def load(self, model_file):
        self.new_session()
        with self.session.graph.as_default():
            directory = model_file
            best_dir = os.path.join(directory, 'best')
            best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
            best_checkpoint = os.path.join(best_dir, 'best_model.ckpt')
            self.set_best_dict(**dict(np.load(best_loss_file)))
            self.NASdict = np.load(os.path.join(directory, 'NAS.npz')) if os.path.exists(os.path.join(directory, 'NAS.npz')) else None
            self.model = NeuralNetwork(F=self.hyperparameters.num_features,           
                                    K=self.hyperparameters.num_basis,                
                                    sr_cut=self.hyperparameters.cutoff,              
                                    num_blocks=self.hyperparameters.num_blocks, 
                                    num_residual_atomic=self.hyperparameters.num_residual_atomic,
                                    num_residual_interaction=self.hyperparameters.num_residual_interaction,
                                    num_residual_output=self.hyperparameters.num_residual_output,
                                    use_electrostatic=(self.hyperparameters.use_electrostatic == 1),
                                    use_dispersion=(self.hyperparameters.use_dispersion == 1),
                                    s6=self.hyperparameters.grimme_s6,
                                    s8=self.hyperparameters.grimme_s8,
                                    a1=self.hyperparameters.grimme_a1,
                                    a2=self.hyperparameters.grimme_a2,
                                    Eshift=self.NASdict['Eshift'] if self.NASdict else 0.0,  
                                    Escale=self.NASdict['Escale'] if self.NASdict else 1.0, 
                                    activation_fn=shifted_softplus, 
                                    seed=None,
                                    scope="neural_network"+str(time.time()))
            checkpoint = tf.train.latest_checkpoint(best_dir)
            assert checkpoint is not None
            self.model.restore(self.session, checkpoint)
            if self.verbose: print(f'model loaded from {model_file}')

    def save(self, model_file=None):
        if not model_file:
            model_file =f'physnet_{str(uuid.uuid4())}'

        directory = model_file
        if not os.path.exists(directory):
            os.makedirs(directory)
        best_dir = os.path.join(directory, 'best')
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
        best_checkpoint = os.path.join(best_dir, 'best_model.ckpt')

        np.savez(best_loss_file, **self.best_dict)
        np.savez(os.path.join(directory, 'NAS.npz'), **self.NASdict)
        self.model.save(self.session, best_checkpoint, global_step=self.step)

        if self.verbose: print(f'model saved in {model_file}')

    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        hyperparameters: Union[Dict[str,Any], models.hyperparameters] = {},
        spliting_ratio=0.8,
        save_model=True,
        log=True,
        check_point=False,
        use_last_model=False,
        summary_interval=1,
        validation_interval=1,
        save_interval=1,
        record_run_metadata=0,
    ) -> None:
        
        check_point = save_model and check_point
        log = save_model and log
        self.hyperparameters.update(hyperparameters)
        if save_model:
            if not self.model_file: self.model_file = f'physnet_{str(uuid.uuid4())}'
            directory = self.model_file
            if not os.path.exists(directory):
                os.makedirs(directory)
            if log: logging.basicConfig(filename=os.path.join(directory, 'train.log'),level=logging.DEBUG)
            best_dir = os.path.join(directory, 'best')
            if not os.path.exists(best_dir):
                os.makedirs(best_dir)
            log_dir = os.path.join(directory, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            best_loss_file  = os.path.join(best_dir, 'best_loss.npz')
            best_checkpoint = os.path.join(best_dir, 'best_model.ckpt')
            step_checkpoint = os.path.join(log_dir,  'model.ckpt')

        tf.compat.v1.reset_default_graph()
        
        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]
        elif not validation_molecular_database:
            raise NotImplementedError("please specify validation_molecular_database or set it to 'sample_from_molecular_database'")

        with self.session.graph.as_default():
            n_subtrain = len(molecular_database)
            n_valid =  len(validation_molecular_database)
            batch_size= min(self.hyperparameters.batch_size, n_subtrain)
            valid_batch_size = min(self.hyperparameters.valid_batch_size, n_valid)
            dataset = molDB2PhysNetData(molecular_database, property_to_learn, xyz_derivative_property_to_learn)
            valid_dataset = molDB2PhysNetData(validation_molecular_database, property_to_learn, xyz_derivative_property_to_learn)
            subtrain_provider = DataProvider(dataset, n_subtrain, 0, batch_size, 0, seed=self.hyperparameters.seed)
            validate_provider = DataProvider(valid_dataset, 0, n_valid, 0, self.hyperparameters.valid_batch_size, seed=self.hyperparameters.seed)

            self.NASdict = {'Eshift': subtrain_provider.EperA_mean,'Escale': subtrain_provider.EperA_stdev}

            steps_per_epoch = min((n_subtrain - 1) // batch_size + 1, self.hyperparameters.max_steps)
            
            summary_interval = self.hyperparameters.summary_interval if self.hyperparameters.summary_interval else steps_per_epoch
            validation_interval = self.hyperparameters.validation_interval if self.hyperparameters.validation_interval else steps_per_epoch
            save_interval = self.hyperparameters.save_interval if self.hyperparameters.save_interval else steps_per_epoch

            train_queue = DataQueue(subtrain_provider.next_batch, capacity=n_subtrain//batch_size, scope="train_data_queue")
            valid_queue = DataQueue(validate_provider.next_valid_batch, capacity=n_valid//valid_batch_size, scope="valid_data_queue")
            Eref_t, Earef_t, Fref_t, Z_t, Dref_t, Qref_t, Qaref_t, R_t, idx_i_t, idx_j_t, batch_seg_t = train_queue.dequeue_op
            Eref_v, Earef_v, Fref_v, Z_v, Dref_v, Qref_v, Qaref_v, R_v, idx_i_v, idx_j_v, batch_seg_v = valid_queue.dequeue_op

            if not self.model:
                self.model = NeuralNetwork(F=self.hyperparameters.num_features,           
                                        K=self.hyperparameters.num_basis,                
                                        sr_cut=self.hyperparameters.cutoff,              
                                        num_blocks=self.hyperparameters.num_blocks, 
                                        num_residual_atomic=self.hyperparameters.num_residual_atomic,
                                        num_residual_interaction=self.hyperparameters.num_residual_interaction,
                                        num_residual_output=self.hyperparameters.num_residual_output,
                                        use_electrostatic=(self.hyperparameters.use_electrostatic == 1),
                                        use_dispersion=(self.hyperparameters.use_dispersion == 1),
                                        s6=self.hyperparameters.grimme_s6,
                                        s8=self.hyperparameters.grimme_s8,
                                        a1=self.hyperparameters.grimme_a1,
                                        a2=self.hyperparameters.grimme_a2,
                                        Eshift=subtrain_provider.EperA_mean,  
                                        Escale=subtrain_provider.EperA_stdev,   
                                        activation_fn=shifted_softplus, 
                                        seed=None,
                                        scope="neural_network")
                
            #calculate all necessary quantities (unscaled partial charges, energies, forces)
            Ea_t, Qa_t, Dij_t, nhloss_t = self.model.atomic_properties(Z_t, R_t, idx_i_t, idx_j_t)
            Ea_v, Qa_v, Dij_v, nhloss_v = self.model.atomic_properties(Z_v, R_v, idx_i_v, idx_j_v)
            energy_t, forces_t = self.model.energy_and_forces_from_atomic_properties(Ea_t, Qa_t, Dij_t, Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg_t)
            energy_v, forces_v = self.model.energy_and_forces_from_atomic_properties(Ea_v, Qa_v, Dij_v, Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg_v)
            #total charge
            Qtot_t = tf.segment_sum(Qa_t, batch_seg_t)
            Qtot_v = tf.segment_sum(Qa_v, batch_seg_v)
            #dipole moment vector
            QR_t = tf.stack([Qa_t*R_t[:,0], Qa_t*R_t[:,1], Qa_t*R_t[:,2]],1)
            QR_v = tf.stack([Qa_v*R_v[:,0], Qa_v*R_v[:,1], Qa_v*R_v[:,2]],1)
            D_t = tf.segment_sum(QR_t, batch_seg_t)
            D_v = tf.segment_sum(QR_v, batch_seg_v)

            def calculate_errors(val1, val2, weights=1):
                with tf.name_scope("calculate_errors"):
                    delta  = tf.abs(val1-val2)
                    delta2 = delta**2
                    mse    = tf.reduce_mean(delta2)
                    mae    = tf.reduce_mean(delta)
                    loss   = mae #mean absolute error loss
                return loss, mse, mae

            with tf.name_scope("loss"):
                #calculate energy, force, charge and dipole errors/loss
                #energy
                if dataset.E is not None:
                    eloss_t, emse_t, emae_t = calculate_errors(Eref_t, energy_t)
                    eloss_v, emse_v, emae_v = calculate_errors(Eref_v, energy_v)
                else:
                    eloss_t, emse_t, emae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    eloss_v, emse_v, emae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                #atomic energies
                if dataset.Ea is not None:
                    ealoss_t, eamse_t, eamae_t = calculate_errors(Earef_t, Ea_t)
                    ealoss_v, eamse_v, eamae_v = calculate_errors(Earef_v, Ea_v)
                else:
                    ealoss_t, eamse_t, eamae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    ealoss_v, eamse_v, eamae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                #forces
                if dataset.F is not None:
                    floss_t, fmse_t, fmae_t = calculate_errors(Fref_t, forces_t)
                    floss_v, fmse_v, fmae_v = calculate_errors(Fref_v, forces_v)
                else:
                    floss_t, fmse_t, fmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    floss_v, fmse_v, fmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                #charge
                if dataset.Q is not None:     
                    qloss_t, qmse_t, qmae_t = calculate_errors(Qref_t, Qtot_t)
                    qloss_v, qmse_v, qmae_v = calculate_errors(Qref_v, Qtot_v)
                else:
                    qloss_t, qmse_t, qmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    qloss_v, qmse_v, qmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                #atomic charges
                if dataset.Qa is not None:
                    qaloss_t, qamse_t, qamae_t = calculate_errors(Qaref_t, Qa_t)
                    qaloss_v, qamse_v, qamae_v = calculate_errors(Qaref_v, Qa_v)
                else:
                    qaloss_t, qamse_t, qamae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    qaloss_v, qamse_v, qamae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                #dipole
                if dataset.D is not None:
                    dloss_t, dmse_t, dmae_t = calculate_errors(Dref_t, D_t)
                    dloss_v, dmse_v, dmae_v = calculate_errors(Dref_v, D_v)
                else:
                    dloss_t, dmse_t, dmae_t = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
                    dloss_v, dmse_v, dmae_v = tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)

                #define additional variables (such that certain losses can be overwritten)
                eloss_train = eloss_t
                floss_train = floss_t
                qloss_train = qloss_t
                dloss_train = dloss_t
                eloss_valid = eloss_v
                floss_valid = floss_v
                qloss_valid = qloss_v
                dloss_valid = dloss_v

                #atomic energies are present, so they replace the normal energy loss
                if dataset.Ea is not None:
                    eloss_train = ealoss_t
                    eloss_valid = ealoss_v

                #atomic charges are present, so they replace the normal charge loss and nullify dipole loss
                if dataset.Qa is not None:
                    qloss_train = qaloss_t
                    qloss_valid = qaloss_v
                    dloss_train = tf.constant(0.0)
                    dloss_valid = tf.constant(0.0)

                #define loss function (used to train the model)
                l2loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 
                loss_t = eloss_train + self.hyperparameters.force_weight * floss_train + self.hyperparameters.charge_weight * qloss_train + self.hyperparameters.dipole_weight * dloss_train + self.hyperparameters.nhlambda * nhloss_t + self.hyperparameters.l2lambda * l2loss
                loss_v = eloss_valid + self.hyperparameters.force_weight * floss_valid + self.hyperparameters.charge_weight * qloss_valid + self.hyperparameters.dipole_weight * dloss_valid + self.hyperparameters.nhlambda * nhloss_v + self.hyperparameters.l2lambda * l2loss

            #create trainer
            trainer  = Trainer(self.hyperparameters.learning_rate, self.hyperparameters.decay_steps, self.hyperparameters.decay_rate, scope="trainer")
            with tf.name_scope("trainer_ops"):
                train_op = trainer.build_train_op(loss_t, self.hyperparameters.ema_decay, self.hyperparameters.max_norm)
                save_variable_backups_op = trainer.save_variable_backups()
                load_averaged_variables_op = trainer.load_averaged_variables()
                restore_variable_backups_op = trainer.restore_variable_backups()

            #creates a summary from key-value pairs given a dictionary
            def create_summary(dictionary):
                summary = tf.Summary()
                for key, value in dictionary.items():
                    summary.value.add(tag=key, simple_value=value)
                return summary

            #create summary writer
            if log:
                nn_summary_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())

            #create saver
            if check_point:
                with tf.name_scope("saver"):
                    saver = tf.train.Saver(max_to_keep=50)

            #save/load best recorded loss (only the best model is saved)
            best_loss   = self.best_dict["loss"]
            best_emae   = self.best_dict["emae"]
            best_ermse  = self.best_dict["ermse"]
            best_fmae   = self.best_dict["fmae"]
            best_frmse  = self.best_dict["frmse"]
            best_qmae   = self.best_dict["qmae"]
            best_qrmse  = self.best_dict["qrmse"]
            best_dmae   = self.best_dict["dmae"]
            best_drmse  = self.best_dict["drmse"]
            best_step   = self.best_dict["step"]

            #for calculating average performance on the training set
            def reset_averages():
                return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            def update_averages(num, tmploss_avg, tmploss, emse_avg, emse, emae_avg, emae, fmse_avg, fmse, fmae_avg, fmae, 
                                qmse_avg, qmse, qmae_avg, qmae, dmse_avg, dmse, dmae_avg, dmae):
                num += 1
                tmploss_avg += (tmploss-tmploss_avg)/num
                emse_avg += (emse-emse_avg)/num
                emae_avg += (emae-emae_avg)/num
                fmse_avg += (fmse-fmse_avg)/num
                fmae_avg += (fmae-fmae_avg)/num
                qmse_avg += (qmse-qmse_avg)/num
                qmae_avg += (qmae-qmae_avg)/num
                dmse_avg += (dmse-dmse_avg)/num
                dmae_avg += (dmae-dmae_avg)/num
                return num, tmploss_avg, emse_avg, emae_avg, fmse_avg, fmae_avg, qmse_avg, qmae_avg, dmse_avg, dmae_avg

            def early_stop(val_loss, best_loss, counter, patience=60, threshold_ratio=0.0001):

                stop_trainer = False
                if val_loss > (1 - threshold_ratio) * best_loss:
                    counter += 1
                else:
                    best_loss = val_loss
                    counter = 0
                if self.verbose: print('early-stopping counter: %s'% counter)
                if counter >patience:
                    stop_trainer = True
                    if self.verbose: print("met early-stopping conditions")

                return stop_trainer, counter

            #initialize training set error averages
            num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()

            # original session start
            
            if (record_run_metadata > 0):
                run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options  = None
                run_metadata = None

            #start data queues
            coord = tf.train.Coordinator()
            train_queue.create_thread(self.session, coord)
            valid_queue.create_thread(self.session, coord)

            #initialize variables
            tf.global_variables_initializer().run(session=self.session)

            #restore latest checkpoint
            self.step = 0
            if check_point:
                checkpoint = tf.train.latest_checkpoint(log_dir)
                if checkpoint is not None:
                    self.step = int(checkpoint.split('-')[-1]) #reads step from checkpoint filename
                    saver.restore(self.session, checkpoint)
                    self.session.run(trainer.global_step.assign(self.step))

            counter = 0 
            stop_train = False

            #training loop
            if log: logging.info("starting training...")
            while not coord.should_stop():
                #finish training when maximum number of iterations is reached
                if self.step > self.hyperparameters.max_steps or (stop_train == True):
                    coord.request_stop()
                    break

                #perform training step 
                self.step += 1
                _, tmploss, emse, emae, fmse, fmae, qmse, qmae, dmse, dmae = self.session.run([train_op, loss_t, emse_t, emae_t, fmse_t, fmae_t, qmse_t, qmae_t, dmse_t, dmae_t], options=run_options, feed_dict={self.model.keep_prob: self.hyperparameters.keep_prob}, run_metadata=run_metadata)
                
                #update averages
                num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = update_averages(num_t, tmploss_avg_t, tmploss, emse_avg_t, emse, emae_avg_t, emae, fmse_avg_t, fmse, fmae_avg_t, fmae, qmse_avg_t, qmse, qmae_avg_t, qmae, dmse_avg_t, dmse, dmae_avg_t, dmae)

                #save progress
                if check_point and (self.step % save_interval == 0):
                    saver.save(self.session, step_checkpoint, global_step=self.step)   

                #check performance on the validation set
                if (self.step % validation_interval == 0):
                    #save backup variables and load averaged variables
                    self.session.run(save_variable_backups_op)
                    self.session.run(load_averaged_variables_op)

                    #initialize averages to 0
                    num_v, tmploss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = reset_averages()
                    #compute averages
                    for i in range(n_valid // self.hyperparameters.valid_batch_size):
                        tmploss, emse, emae, fmse, fmae, qmse, qmae, dmse, dmae = self.session.run([loss_v, emse_v, emae_v, fmse_v, fmae_v, qmse_v, qmae_v, dmse_v, dmae_v])
                        num_v, tmploss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = update_averages(num_v, tmploss_avg_v, tmploss, emse_avg_v, emse, emae_avg_v, emae, fmse_avg_v, fmse, fmae_avg_v, fmae, qmse_avg_v, qmse, qmae_avg_v, qmae, dmse_avg_v, dmse, dmae_avg_v, dmae)

                    #store results in dictionary
                    results = {}
                    results["loss_valid"] = tmploss_avg_v

                    if self.hyperparameters.earlystopping:
                        stop_train, counter = early_stop(tmploss_avg_v, best_loss, counter, patience=self.hyperparameters.patience, threshold_ratio=self.hyperparameters.threshold)

                    if dataset.E is not None:
                        results["energy_mae_valid"]  = emae_avg_v
                        results["energy_rmse_valid"] = np.sqrt(emse_avg_v)
                    if dataset.F is not None:
                        results["forces_mae_valid"]  = fmae_avg_v
                        results["forces_rmse_valid"] = np.sqrt(fmse_avg_v)
                    if dataset.Q is not None:
                        results["charge_mae_valid"]  = qmae_avg_v
                        results["charge_rmse_valid"] = np.sqrt(qmse_avg_v)
                    if dataset.D is not None:
                        results["dipole_mae_valid"]  = dmae_avg_v
                        results["dipole_rmse_valid"] = np.sqrt(dmse_avg_v)

                    if results["loss_valid"] < best_loss:
                        best_loss   = results["loss_valid"]
                        if dataset.E is not None:
                            best_emae   = results["energy_mae_valid"]
                            best_ermse  = results["energy_rmse_valid"]
                        else:
                            best_emae  = np.Inf
                            best_ermse = np.Inf
                        if dataset.F is not None:
                            best_fmae   = results["forces_mae_valid"]
                            best_frmse  = results["forces_rmse_valid"]
                        else:
                            best_fmae  = np.Inf
                            best_frmse = np.Inf
                        if dataset.Q is not None:
                            best_qmae   = results["charge_mae_valid"]
                            best_qrmse  = results["charge_rmse_valid"]
                        else:
                            best_qmae  = np.Inf
                            best_qrmse = np.Inf
                        if dataset.D is not None:
                            best_dmae   = results["dipole_mae_valid"]
                            best_drmse  = results["dipole_rmse_valid"]
                        else:
                            best_dmae  = np.Inf
                            best_drmse = np.Inf
                        best_step = self.step
                        self.set_best_dict(loss=best_loss, 
                                        emae=best_emae, ermse=best_ermse, 
                                        fmae=best_fmae, frmse=best_frmse, 
                                        qmae=best_qmae, qrmse=best_qrmse, 
                                        dmae=best_dmae, drmse=best_drmse, 
                                        step=best_step)
                        if save_model: 
                            self.save(self.model_file)
                    results["loss_best"] = best_loss
                    if dataset.E is not None:
                        results["energy_mae_best"]  = best_emae
                        results["energy_rmse_best"] = best_ermse
                    if dataset.F is not None:
                        results["forces_mae_best"]  = best_fmae
                        results["forces_rmse_best"] = best_frmse
                    if dataset.Q is not None:
                        results["charge_mae_best"]  = best_qmae
                        results["charge_rmse_best"] = best_qrmse
                    if dataset.D is not None:
                        results["dipole_mae_best"]  = best_dmae
                        results["dipole_rmse_best"] = best_drmse
                    if log:
                        summary = create_summary(results)
                        summary_writer.add_summary(summary, global_step=self.step)

                    #restore backup variables
                    self.session.run(restore_variable_backups_op)
        
                #generate summaries
                if (self.step % summary_interval == 0) and (self.step > 0): 
                    results = {}            
                    results["loss_train"] = tmploss_avg_t
                    if dataset.E is not None:
                        results["energy_mae_train"]  = emae_avg_t
                        results["energy_rmse_train"] = np.sqrt(emse_avg_t)
                    if dataset.F is not None:
                        results["forces_mae_train"]  = fmae_avg_t
                        results["forces_rmse_train"] = np.sqrt(fmse_avg_t)
                    if dataset.Q is not None:
                        results["charge_mae_train"]  = qmae_avg_t
                        results["charge_rmse_train"] = np.sqrt(qmse_avg_t)
                    if dataset.D is not None:
                        results["dipole_mae_train"]  = dmae_avg_t
                        results["dipole_rmse_train"] = np.sqrt(dmse_avg_t)
                    num_t, tmploss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t = reset_averages()

                    if log:
                        summary = create_summary(results)
                        summary_writer.add_summary(summary, global_step=self.step)
                        nn_summary = nn_summary_op.eval(session=self.session)
                        summary_writer.add_summary(nn_summary, global_step=self.step)
                        if (record_run_metadata > 0):
                            summary_writer.add_run_metadata(run_metadata, 'step %d' % self.step, global_step=self.step)

                    if dataset.E is not None:
                        if self.verbose: print(str(self.step)+'/'+str(self.hyperparameters.max_steps), "loss:", results["loss_train"], "best:", best_loss, "emae:", results["energy_mae_train"], "best:", best_emae)
                        sys.stdout.flush()
        # original session stop

        if save_model and not use_last_model:
            self.load(self.model_file)
    
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

        with self.session.graph.as_default():
            Eref      = tf.placeholder(tf.float32, shape=[None, ], name="Eref")
            Fref      = tf.placeholder(tf.float32, shape=[None,3], name="Fref") 
            Z         = tf.placeholder(tf.int32,   shape=[None, ], name="Z")     
            Dref      = tf.placeholder(tf.float32, shape=[None,3], name="Dref") 
            Qref      = tf.placeholder(tf.float32, shape=[None, ], name="Qref")   
            R         = tf.placeholder(tf.float32, shape=[None,3], name="R")      
            idx_i     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_i") 
            idx_j     = tf.placeholder(tf.int32,   shape=[None, ], name="idx_j") 
            batch_seg = tf.placeholder(tf.int32,   shape=[None, ], name="batch_seg") 

            #model energy/forces
            Ea, Qa, Dij, nhloss = self.model.atomic_properties(Z, R, idx_i, idx_j)
            if property_to_predict and not xyz_derivative_property_to_predict:
                energy = self.model.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Qref, batch_seg)
            if xyz_derivative_property_to_predict:
                energy, forces = self.model.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Qref, batch_seg)
            #total charge
            Qtot = tf.segment_sum(Qa, batch_seg)
            #dipole moment vector
            QR = tf.stack([Qa*R[:,0], Qa*R[:,1], Qa*R[:,2]],1)
            D  = tf.segment_sum(QR, batch_seg)

            def fill_feed_dict(data):
                feed_dict = { 
                    Eref:      data["E"], 
                    Fref:      data["F"],
                    Z:         data["Z"], 
                    Dref:      data["D"],
                    Qref:      data["Q"],
                    R:         data["R"],
                    idx_i:     data["idx_i"],
                    idx_j:     data["idx_j"],
                    batch_seg: data["batch_seg"] 
                }
                return feed_dict
            
            all_data = molDB2PhysNetData(molDB)
            for i, mol in enumerate(molDB.molecules):
                data = all_data[i]
                feed_dict = fill_feed_dict(data)
                if property_to_predict and not xyz_derivative_property_to_predict: 
                    PhysNet_energy = self.session.run([energy], feed_dict=feed_dict)
                    mol.__dict__[property_to_predict] = float(PhysNet_energy)
                if xyz_derivative_property_to_predict:
                    PhysNet_energy, PhysNet_forces = self.session.run([energy, forces], feed_dict=feed_dict)
                    grads = PhysNet_forces * -1
                    mol.__dict__[property_to_predict] = float(PhysNet_energy)
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].__dict__[xyz_derivative_property_to_predict] = grads[iatom]

class _DataContainer(DataContainer):
    def __init__(self, dictionary):
        #number of atoms
        if 'N' in dictionary: 
            self._N = dictionary['N'] 
        else:
            self._N = None
        #atomic numbers/nuclear charges
        if 'Z' in dictionary: 
            self._Z = dictionary['Z'] 
        else:
            self._Z = None
        #reference dipole moment vector
        if 'D' in dictionary: 
            self._D = dictionary['D'] 
        else:
            self._D = None
        #reference total charge
        if 'Q' in dictionary: 
            self._Q = dictionary['Q'] 
        else:
            self._Q = None
        #reference atomic charges
        if 'Qa' in dictionary: 
            self._Qa = dictionary['Qa'] 
        else:
            self._Qa = None
        #positions (cartesian coordinates)
        if 'R' in dictionary:     
            self._R = dictionary['R'] 
        else:
            self._R = None
        #reference energy
        if 'E' in dictionary:
            self._E = dictionary['E'] 
        else:
            self._E = None
        #reference atomic energies
        if 'Ea' in dictionary:
            self._Ea = dictionary['Ea']
        else:
            self._Ea = None
        #reference forces
        if 'F' in dictionary:
            self._F = dictionary['F'] 
        else:
            self._F = None

        self._N_max    = self.Z.shape[1] 
        
        self._idx_i = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_i.shape[0]):
            for j in range(self.idx_i.shape[1]):
                self._idx_i[i,j] = i

        self._idx_j = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_j.shape[0]):
            c = 0
            for j in range(self.idx_j.shape[0]):
                if j != i:
                    self._idx_j[i,c] = j
                    c += 1

def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
  To use Interface_PhysNet, please define $PhysNet to where PhysNet is located

  Arguments with their default values:
    MLprog=PhysNet             enables this interface
    MLmodelType=PhysNet        requests PhysNet model
    
    physnet.earlystopping=1    enable early-stopping
    physnet.threshold=0        threshold for early-stopping
    physnet.patience=0         patience for early-stopping
    physnet.num_features=128   number of input features
    physnet.num_basis=64       number of radial basis functions
    physnet.num_blocks=5       number of stacked modular building blocks
                                                
    physnet.num_residual_atomic=2 
                               number of residual blocks for 
                               atom-wise refinements
    physnet.num_residual_interaction=3
                               number of residual blocks for 
                               refinements of proto-message
    physnet.num_residual_output=1 
                               number of residual blocks in 
                               output blocks
    physnet.cutoff=10.0        cutoff radius for interactions 
                               in the neural network
    physnet.seed=42            random seed
    physnet.max_steps=10000000
                               max steps to perform in training
    physnet.learning_rate=0.0008 
                               starting learning rate
    physnet.decay_steps=10000000 
                               decay steps
    physnet.decay_rate=0.1     decay rate for learning rate
    physnet.batch_size=12      training batch size
    physnet.valid_batch_size=2 validation batch size
    physnet.force_weight=52.91772105638412 
                               weight for force
    physnet.charge_weight=0    weight for charge
    physnet.dipole_weight=0    weight for dipole
    physnet.summary_interval=0 
                               interval for summary 
                               (0 for an auto-decision, the same below)
    physnet.validation_interval=0 
                               interval for validation
    physnet.save_interval=0    interval for model saving

  Cite PhysNet:
    O. T. Unke, M. Meuwly, J. Chem. Theory Comput. 2019, 15, 3678
'''
    print(helpText)