'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_sGDML: Interface between sGDML and MLatom                       ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''

from __future__ import annotations
from typing import Any, Union, Dict
import os, sys, uuid, time
import numpy as np
import shutil
import logging
from sgdml import __version__ as sGDMLver
from sgdml import DONE, NOT_DONE, ColoredLogger
from sgdml.train import GDMLTrain
from sgdml.predict import GDMLPredict
from sgdml.utils import io, ui
from sgdml.cli import log, AssistantError, _print_model_properties, _batch, _online_err
from functools import partial

from .. import constants
from .. import data
from .. import models
from .. import stopper
from ..decorators import doc_inherit

logging.setLoggerClass(logging.Logger)

def molDB2sGDMLdata(molDB,
                    property_to_learn='energy',
                    xyz_derivative_property_to_learn = 'energy_gradients',
                    name='nameless dataset',
                    theory='unknown',
                    r_unit='Ang',
                    e_unit='Hatree'):

    e_factor = 1.0
    r_factor = 1.0
    dataset =  {
        'type': 'd',
        'code_version': sGDMLver,
        'name': name,
        'theory': theory,
        'R': molDB.xyz_coordinates,
        'z': molDB.molecules[0].get_atomic_numbers(),
        'F': -1 * molDB.get_xyz_vectorial_properties(xyz_derivative_property_to_learn) * e_factor / r_factor,
        'E': molDB.get_properties(property_to_learn) * e_factor,
        'r_unit': r_unit,
        'e_unit': e_unit,
    }
    dataset['F_min'], dataset['F_max'] = np.min(dataset['F'].ravel()), np.max(dataset['F'].ravel())
    dataset['F_mean'], dataset['F_var'] = np.mean(dataset['F'].ravel()), np.var(dataset['F'].ravel())
    dataset['E_min'], dataset['E_max'] = np.min(dataset['E']), np.max(dataset['E'])
    dataset['E_mean'], dataset['E_var'] = np.mean(dataset['E']), np.var(dataset['E'])
    dataset['md5'] = io.dataset_md5(dataset)
    # dataset['lattice'] = lattice
    return _dictTransform(dataset)

def _dictTransform(d):
    return {k: np.array(v) for k, v in d.items()}

class sgdml(models.ml_model):
    '''
    Create an `sGDML <https://doi.org/10.1038/s41467-018-06169-2>`__ model object. 
    
    Interfaces to `sGDML <https://doi.org/10.1016/j.cpc.2019.02.007>`__.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.

    Hyperparameters:

        Please refer to the sgdml manual

        no_E:       mlatom.models.hyperparameter(value=False)
        gdml:       mlatom.models.hyperparameter(value=False)
        perms:      mlatom.models.hyperparameter(value=None)
        sigma:      mlatom.models.hyperparameter(value=None)
        E_cstr:     mlatom.models.hyperparameter(value=False)
        cprsn:      mlatom.models.hyperparameter(value=False)

    '''

    hyperparameters = models.hyperparameters({
        'no_E':     models.hyperparameter(value=False),
        'gdml':     models.hyperparameter(value=False),
        'perms':    models.hyperparameter(value=None),
        'sigma':    models.hyperparameter(value=None),
        'E_cstr':   models.hyperparameter(value=False),
        'cprsn':    models.hyperparameter(value=False),
    })

    property_name = 'y'
    program = 'sGDML'
    meta_data = {
        "genre": "kernel method"
    }
    model_file = None
    model = None
    gdml_train = None
    verbose = 1
    

    def __init__(self, model_file=None, hyperparameters={}, verbose=1,
              max_memory=None, max_processes=None, use_torch=False, lazy_training=False):
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        self.max_memory = max_memory
        self.max_processes = max_processes
        self.use_torch = use_torch
        self.lazy_training = lazy_training

        if model_file: 
            if os.path.isfile(model_file):
                self.load(model_file)
            else:
                if self.verbose: print(f'the trained sGDML model will be saved in {model_file}')
            self.model_file = model_file
    
    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'sgdml' in args.data and hyperparam in args.sgdml.data:
                self.hyperparameters[hyperparam].value = args.sgdml.data[hyperparam]

    def reset(self):
        super().reset()
        self.model = None
    
    def load(self, model_file):
        with np.load(model_file, allow_pickle=True) as model:
            self.model = dict(model)
        if self.verbose: print(f'model loaded from {model_file}')

    def save(self, model_file=None):
        if not model_file:
            model_file =f'sgdml_{str(uuid.uuid4())}.npz'
        np.savez_compressed(model_file, **self.model)
        if self.verbose: print(f'model saved in {model_file}')

    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = None,
        hyperparameters: Union[Dict[str,Any], models.hyperparameters] = {},
        spliting_ratio=0.8,
        save_model=True,
        task_dir=None,
        overwrite=False,
        max_memory=None,
        max_processes=None,
        use_torch=None,
        lazy_training=None
    ):
        if max_memory is None: max_memory = self.max_memory
        if max_processes is None: max_processes = self.max_processes
        if use_torch is None: use_torch = self.use_torch
        if lazy_training is None: lazy_training = self.lazy_training

        self.hyperparameters.update(hyperparameters)
        
        logging.setLoggerClass(ColoredLogger)

        dataset = molDB2sGDMLdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn, name='training set')
        n_train = dataset['R'].shape[0]

        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]

        if validation_molecular_database:
            valid_dataset = molDB2sGDMLdata(validation_molecular_database, property_to_learn, xyz_derivative_property_to_learn, name='validation set')
            n_valid = len(validation_molecular_database)
        else:
            valid_dataset = dataset
            n_valid = int(n_train * (1-spliting_ratio))
            n_train -= n_valid
        
        tasks, task_dir = self.create(dataset=dataset,
                                      valid_dataset=valid_dataset,
                                      n_train=n_train,
                                      n_valid=n_valid,
                                      task_dir=task_dir,
                                      sigs=self.hyperparameters.sigma,
                                      overwrite=overwrite,
                                      max_memory=max_memory,
                                      max_processes=max_processes,
                                      use_torch=use_torch,
                                      gdml=self.hyperparameters.gdml,
                                      use_E=not self.hyperparameters.no_E,
                                      use_E_cstr=self.hyperparameters.E_cstr,
                                      perms=self.hyperparameters.perms)
        models = self.train_models(tasks,
                         task_dir,
                         valid_dataset,
                         overwrite,
                         max_memory,
                         max_processes,
                         use_torch,
                         lazy_training)

        self.model = self.select(models)

        if save_model: self.save(self.model_file)
        
        logging.setLoggerClass(logging.Logger)

    def create(self, dataset,
               valid_dataset,
               n_train,
               n_valid,
               task_dir,
               sigs,
               overwrite,
               max_memory,
               max_processes,
               use_torch,
               gdml,
               use_E,
               use_E_cstr,
               perms):
        
        if sigs is None:
            if self.verbose:
                log.info(
                    'Kernel hyper-parameter sigma (length scale) was automatically set to range \'10:10:100\'.'
                )
            sigs = list(range(10, 100, 10)) 

        if task_dir == True:
            task_dir = io.train_dir_name(
                dataset,
                n_train,
                use_sym=not gdml,
                use_E=use_E,
                use_E_cstr=use_E_cstr,
            )

        task_file_names = []
        tasks = []
        if task_dir:
            if os.path.exists(task_dir):
                if overwrite:
                    if self.verbose: log.info('Overwriting existing training directory')
                    shutil.rmtree(task_dir, ignore_errors=True)
                    os.makedirs(task_dir)
                else:
                    if io.is_task_dir_resumeable(
                        task_dir, dataset, valid_dataset, n_train, n_valid, sigs, gdml
                    ):
                        if self.verbose: 
                            log.info('Resuming existing hyper-parameter search in \'{}\'.'.format(task_dir))

                        # Get all task file names.
                        try:
                            _, task_file_names = io.is_dir_with_file_type(task_dir, 'task')
                        except Exception:
                            pass
                    else:
                        raise AssistantError(
                            'Unfinished hyper-parameter search found in \'{}\'.\n'.format(
                                task_dir
                            )
                            + ' Set overwrite=True to overwrite.'
                        )
            else:
                os.makedirs(task_dir)
            
        if task_file_names:
            for task_file in task_file_names:
                with np.load(
                    os.path.join(task_dir, task_file), allow_pickle=True
                ) as task:
                    tasks.append(dict(task))
            task = tasks[0]
        else:
            if self.hyperparameters.no_E:
                if self.verbose: log.info(
                    'Energy labels will be ignored for training.\n'
                    + 'Note: If available in the dataset file, the energy labels will however still be used to generate stratified training, test and validation datasets. Otherwise a random sampling is used.'
                )

            if 'E' not in dataset:
                if self.verbose: log.warning(
                    'Training dataset will be sampled with no guidance from energy labels (i.e. randomly)!'
                )

            if 'E' not in valid_dataset:
                if self.verbose: log.warning(
                    'Validation dataset will be sampled with no guidance from energy labels (i.e. randomly)!\n'
                    + 'Note: Larger validation datasets are recommended due to slower convergence of the error.'
                )

            if ('lattice' in dataset) ^ ('lattice' in valid_dataset):
                if self.verbose: log.error('One of the datasets specifies lattice vectors and one does not!')
                # TODO: stop program?

            if 'lattice' in dataset or 'lattice' in valid_dataset:
                if self.verbose: log.info(
                    'Lattice vectors found in dataset: applying periodic boundary conditions.'
                )

            if not self.gdml_train:
                self.gdml_train = GDMLTrain(
                    max_memory=max_memory, max_processes=max_processes, use_torch=use_torch
                    )
                if not self.verbose:
                    self.gdml_train.log.disabled = True

            task = self.gdml_train.create_task(
                dataset,
                n_train,
                valid_dataset,
                n_valid,
                sig=1,
                perms=perms,
                use_sym=not gdml,
                use_E=use_E,
                use_E_cstr=self.hyperparameters.E_cstr,
                callback=ui.callback if self.verbose else None,
            )

        n_written = 0
        for sig in sigs:
            task['sig'] = sig
            tasks.append(_dictTransform(task.copy()))
            if task_dir:
                task_file_name = io.task_file_name(task)
                task_path = os.path.join(task_dir, task_file_name)
                if os.path.isfile(task_path):
                    if self.verbose: log.info('Skipping existing task \'{}\'.'.format(task_file_name))
                else:
                    np.savez_compressed(task_path, **task)
                    task_file_names.append(task_file_name)
                    n_written += 1
        if n_written > 0:
            if self.verbose: log.done(
                'Writing {:d}/{:d} task(s) with m={} training points each'.format(
                    n_written, len(self.hyperparameters.sigma), task['R_train'].value.shape[0].value
                )
            )
        
        return tasks, task_dir

    def train_models(self,
                    tasks,
                    task_dir,
                    valid_dataset,
                    overwrite,
                    max_memory,
                    max_processes,
                    use_torch,
                    lazy_training):
        models = []

        if task_dir: _, task_file_names = io.is_dir_with_file_type(task_dir, 'task')
        n_tasks = len(tasks)


        def save_progr_callback(
            unconv_model, unconv_model_path=None
        ):  # Saves current (unconverged) model during iterative training

            if unconv_model_path is None:
                log.critical(
                    'Path for unconverged model not set in \'save_progr_callback\'.'
                )
                print()

            np.savez_compressed(unconv_model_path, **unconv_model)

        if not self.gdml_train:
            self.gdml_train = GDMLTrain(
                max_memory=max_memory, max_processes=max_processes, use_torch=use_torch
                )
            if not self.verbose: self.gdml_train.log.disabled = True
        
        prev_valid_err = -1
        has_converged_once = False

        for i, task in enumerate(tasks):
            if n_tasks > 1:
                if self.verbose:
                    if i > 0:
                        print()

                    n_train = len(task['idxs_train'])
                    n_valid = len(task['idxs_valid'])
                    ui.print_two_column_str(
                        ui.color_str('Task {:d} of {:d}'.format(i + 1, n_tasks), bold=True),
                        '{:,} + {:,} points (training + validation), sigma (length scale): {}'.format(
                            n_train, n_valid, task['sig']
                        ),
                    )
                    
            if task_dir:
                task_file_path = os.path.join(task_dir, task_file_names[i])
                model_file_name = io.model_file_name(task, is_extended=False)
                model_file_path = os.path.join(task_dir, model_file_name)

            if task_dir and os.path.isfile(model_file_path) and not overwrite: 
                if os.path.isfile(
                    model_file_path
                ):  # Train model found, validate if necessary
                    if self.verbose: log.info(
                        'Model \'{}\' already exists.'.format(model_file_name)
                        + ' Set overwrite=True to overwrite.'
                    )

                    _, model = io.is_file_type(model_file_path, 'model')
                    models.append(model)

                    e_err = {'mae': 0.0, 'rmse': 0.0}
                    if model['use_E']:
                        e_err = model['e_err'].item()
                    f_err = model['f_err'].item()

                    is_conv = True
                    if 'solver_resid' in model:
                        is_conv = (
                            model['solver_resid']
                            <= model['solver_tol'] * model['norm_y_train']
                        )

                    is_model_validated = not (
                        np.isnan(f_err['mae']) or np.isnan(f_err['rmse'])
                    )
                    if is_model_validated:

                        disp_str = (
                            'energy %.3f/%.3f, ' % (e_err['mae'], e_err['rmse'])
                            if model['use_E']
                            else ''
                        )
                        disp_str += 'forces %.3f/%.3f' % (f_err['mae'], f_err['rmse'])
                        disp_str = 'Validation errors (MAE/RMSE): ' + disp_str
                        if self.verbose: ui.callback(1, 1, disp_str=disp_str)

                        valid_errs = [f_err['rmse']]

            else:  # Train and validate model

                # Check if training this task has been attempted before.
                if lazy_training and n_tasks > 1:
                    if 'tried_training' in task and task['tried_training']:
                        if self.verbose: log.warning(
                            'Skipping task, because it has been tried before (without success).'
                        )
                        continue

                # Record in task file that there was a training attempt.
                task['tried_training'] = True
                n_train, n_atoms = task['R_train'].shape[:2]

                if task_dir: 
                    np.savez_compressed(task_file_path, **task)
                    unconv_model_file = '_unconv_{}'.format(model_file_name)
                    unconv_model_path = os.path.join(task_dir, unconv_model_file)

                model = self.gdml_train.train(
                    task,
                    partial(
                        save_progr_callback, unconv_model_path=unconv_model_path
                    ) if task_dir else None,
                    ui.callback if self.verbose else None,
                )

                model = _dictTransform(model)

                if task_dir: 
                    if self.verbose: log.done('Writing model to file \'{}\''.format(model_file_path))
                    np.savez_compressed(model_file_path, **model)

                # Delete temporary model, if one exists.
                    unconv_model_exists = os.path.isfile(unconv_model_path)
                    if unconv_model_exists:
                        os.remove(unconv_model_path)

                is_model_validated = False

            if not is_model_validated:

                if (
                    n_tasks == 1
                ):  # Only validate if there is more than one training task.
                    if self.verbose: log.info(
                        'Skipping validation step as there is only one model to validate.'
                    )
                    break

                # Validate model.
                model_dir = (task_dir, [model_file_name] if task_dir else [None])
                valid_errs = self.test(
                    [model],
                    model_dir,
                    valid_dataset,
                    -1,  # n_test = -1 -> validation mode
                    overwrite,
                    max_memory,
                    max_processes,
                    use_torch,
                )

                is_conv = True
                if 'solver_resid' in model:
                    is_conv = (
                        model['solver_resid']
                        <= model['solver_tol'] * model['norm_y_train']
                    )

            has_converged_once = has_converged_once or is_conv
            if (
                has_converged_once
                and prev_valid_err != -1
                and prev_valid_err < valid_errs[0]
            ):
                if self.verbose: 
                    print()
                    log.info(
                    'Skipping remaining training tasks, as validation error is rising again.'
                    )
                break

            prev_valid_err = valid_errs[0]

            model = _dictTransform(model)
            models.append(model)

        model_dir_or_file_path = model_file_path if n_tasks == 1 else task_dir
        return models

    def select(self, models):  # noqa: C901


        any_model_not_validated = False
        any_model_is_tested = False

        if len(models) > 1:

            use_E = True

            rows = []
            data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
            for i, model in enumerate(models):

                use_E = model['use_E']

                if i == 0:
                    idxs_train = set(model['idxs_train'])
                    md5_train = model['md5_train']
                    idxs_valid = set(model['idxs_valid'])
                    md5_valid = model['md5_valid']
                else:
                    if (
                        md5_train != model['md5_train']
                        or md5_valid != model['md5_valid']
                        or idxs_train != set(model['idxs_train'])
                        or idxs_valid != set(model['idxs_valid'])
                    ):
                        raise AssistantError(
                            'models trained or validated on different datasets.'
                        )

                e_err = {'mae': 0.0, 'rmse': 0.0}
                if model['use_E']:
                    e_err = model['e_err'].item()
                f_err = model['f_err'].item()

                is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))
                if not is_model_validated:
                    any_model_not_validated = True

                is_model_tested = model['n_test'] > 0
                if is_model_tested:
                    any_model_is_tested = True

                rows.append(
                    [model['sig'], e_err['mae'], e_err['rmse'], f_err['mae'], f_err['rmse']]
                )

            if any_model_not_validated:
                if self.verbose: 
                    log.warning(
                    'One or more models in the given directory have not been validated.'
                    )
                    print()

            if any_model_is_tested:
                log.error(
                    'One or more models in the given directory have already been tested. This means that their recorded expected errors are test errors, not validation errors. However, one should never perform model selection based on the test error!\n'
                    + 'Please run the validation command (again) with the overwrite option \'-o\', then this selection command.'
                )
                return

            f_rmse_col = [row[4] for row in rows]
            best_idx = f_rmse_col.index(min(f_rmse_col))  # idx of row with lowest f_rmse
            best_sig = rows[best_idx][0]

            rows = sorted(rows, key=lambda col: col[0])  # sort according to sigma
            print(ui.color_str('Cross-validation errors', bold=True))
            print(' ' * 7 + 'Energy' + ' ' * 6 + 'Forces')
            print((' {:>3} ' + '{:>5} ' * 4).format(*data_names))
            print(' ' + '-' * 27)
            format_str = ' {:>3} ' + '{:5.2f} ' * 4
            format_str_no_E = ' {:>3}     -     - ' + '{:5.2f} ' * 2
            for row in rows:
                if use_E:
                    row_str = format_str.format(*row)
                else:
                    row_str = format_str_no_E.format(*[row[0], row[3], row[4]])

                if row[0] != best_sig:
                    row_str = ui.color_str(row_str, fore_color=ui.GRAY)
                print(row_str)
            print()

            sig_col = [row[0] for row in rows]
            if best_sig == min(sig_col) or best_sig == max(sig_col):
                if self.verbose:
                    log.warning(
                    'The optimal sigma (length scale) lies on the boundary of the search grid.\n'
                    + 'Model performance might improve if the search grid is extended in direction sigma {} {:d}.'.format(
                        '<' if best_idx == 0 else '>', best_sig
                    )
                    )

        else:  # only one model available
            if self.verbose: log.info('Skipping model selection step as there is only one model to select.')

            best_idx = 0

        return models[best_idx]

    def test(self,
             models,
             model_dir,
             test_dataset,
             n_test,
             overwrite,
             max_memory,
             max_processes,
             use_torch,):

        model_dir, model_file_names = model_dir
        n_models = len(models)

        n_test = 0 if n_test is None else n_test
        is_validation = n_test < 0
        is_test = n_test >= 0

        dataset = test_dataset

 
        F_rmse = []

        # NEW

        DEBUG_WRITE = False

        if DEBUG_WRITE:
            if os.path.exists('test_pred.xyz'):
                os.remove('test_pred.xyz')
            if os.path.exists('test_ref.xyz'):
                os.remove('test_ref.xyz')
            if os.path.exists('test_diff.xyz'):
                os.remove('test_diff.xyz')

        # NEW

        num_workers, batch_size = -1, -1
        for i, model in enumerate(models):

            if i == 0 :
                if self.verbose:
                    print()
                    _print_model_properties(model)
                    print()

            if not np.array_equal(model['z'], dataset['z']):
                raise AssistantError(
                    'Atom composition or order in dataset does not match the one in model.'
                )

            if ('lattice' in model) is not ('lattice' in dataset):
                if 'lattice' in model:
                    raise AssistantError(
                        'Model contains lattice vectors, but dataset does not.'
                    )
                elif 'lattice' in dataset:
                    raise AssistantError(
                        'Dataset contains lattice vectors, but model does not.'
                    )

            if model['use_E']:
                e_err = model['e_err'].item()
            f_err = model['f_err'].item()

            is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))

            if n_models > 1:
                if i > 0:
                    print()
                print(
                    ui.color_str(
                        '%s model %d of %d'
                        % ('Testing' if is_test else 'Validating', i + 1, n_models),
                        bold=True,
                    )
                )

            if is_validation:
                if is_model_validated and not overwrite:
                    if self.verbose: log.info(
                        'Skipping already validated model. Set overwrite=True to overwrite.'
                    )
                    continue

                if dataset['md5'] != model['md5_valid']:
                    raise AssistantError(
                        'Fingerprint of provided validation dataset does not match the one specified in model file.'
                    )

            test_idxs = model['idxs_valid']
            if is_test:

                # exclude training and/or test sets from validation set if necessary
                excl_idxs = np.empty((0,), dtype=np.uint)
                if dataset['md5'] == model['md5_train']:
                    excl_idxs = np.concatenate([excl_idxs, model['idxs_train']]).astype(
                        np.uint
                    )
                if dataset['md5'] == model['md5_valid']:
                    excl_idxs = np.concatenate([excl_idxs, model['idxs_valid']]).astype(
                        np.uint
                    )

                n_data = dataset['F'].shape[0]
                n_data_eff = n_data - len(excl_idxs)

                if (
                    n_test == 0 and n_data_eff != 0
                ):  # test on all data points that have not been used for training or testing
                    n_test = n_data_eff
                    if self.verbose: log.info(
                        'Test set size was automatically set to {:,} points.'.format(n_test)
                    )

                if n_test == 0 or n_data_eff == 0:
                    if self.verbose: log.warning('Skipping! No unused points for test in provided dataset.')
                    return
                elif n_data_eff < n_test:
                    n_test = n_data_eff
                    if self.verbose: log.warning(
                        'Test size reduced to {:d}. Not enough unused points in provided dataset.'.format(
                            n_test
                        )
                    )

                if 'E' in dataset:
                    if not self.gdml_train:
                        self.gdml_train = GDMLTrain(
                            max_memory=max_memory, max_processes=max_processes
                        )
                        if not self.verbose: self.gdml_train.log.disabled = True
                    test_idxs = self.gdml_train.draw_strat_sample(
                        dataset['E'], n_test, excl_idxs=excl_idxs
                    )
                else:
                    test_idxs = np.delete(np.arange(n_data), excl_idxs)

                    if self.verbose: log.warning(
                        'Test dataset will be sampled with no guidance from energy labels (randomly)!\n'
                        + 'Note: Larger test datasets are recommended due to slower convergence of the error.'
                    )
            # shuffle to improve convergence of online error
            np.random.shuffle(test_idxs)

            # NEW
            if DEBUG_WRITE:
                test_idxs = np.sort(test_idxs)

            z = dataset['z']
            R = dataset['R'][test_idxs, :, :]
            F = dataset['F'][test_idxs, :, :]

            if model['use_E']:
                E = dataset['E'][test_idxs]

            gdml_predict = GDMLPredict(
                model,
                max_memory=max_memory,
                max_processes=max_processes,
                use_torch=use_torch,
            )

            b_size = min(1000, len(test_idxs))

            if not use_torch:
                if num_workers == -1 or batch_size == -1:
                    if self.verbose: ui.callback(NOT_DONE, disp_str='Optimizing parallelism')

                    gps, is_from_cache = gdml_predict.prepare_parallel(
                        n_bulk=b_size, return_is_from_cache=True
                    )
                    num_workers, chunk_size, bulk_mp = (
                        gdml_predict.num_workers,
                        gdml_predict.chunk_size,
                        gdml_predict.bulk_mp,
                    )

                    sec_disp_str = 'no chunking'.format(chunk_size)
                    if chunk_size != gdml_predict.n_train:
                        sec_disp_str = 'chunks of {:d}'.format(chunk_size)

                    if num_workers == 0:
                        sec_disp_str = 'no workers / ' + sec_disp_str
                    else:
                        sec_disp_str = (
                            '{:d} workers {}/ '.format(
                                num_workers, '[MP] ' if bulk_mp else ''
                            )
                            + sec_disp_str
                        )

                    if self.verbose: ui.callback(
                        DONE,
                        disp_str='Optimizing parallelism'
                        + (' (from cache)' if is_from_cache else ''),
                        sec_disp_str=sec_disp_str,
                    )
                else:
                    gdml_predict._set_num_workers(num_workers)
                    gdml_predict._set_chunk_size(chunk_size)
                    gdml_predict._set_bulk_mp(bulk_mp)

            n_atoms = z.shape[0]

            if model['use_E']:
                e_mae_sum, e_rmse_sum = 0, 0
            f_mae_sum, f_rmse_sum = 0, 0
            cos_mae_sum, cos_rmse_sum = 0, 0
            mag_mae_sum, mag_rmse_sum = 0, 0

            n_done = 0
            t = time.time()
            for b_range in _batch(list(range(len(test_idxs))), b_size):

                n_done_step = len(b_range)
                n_done += n_done_step

                r = R[b_range].reshape(n_done_step, -1)
                e_pred, f_pred = gdml_predict.predict(r)

                # energy error
                if model['use_E']:
                    e = E[b_range]
                    e_mae, e_mae_sum, e_rmse, e_rmse_sum = _online_err(
                        np.squeeze(e) - e_pred, 1, n_done, e_mae_sum, e_rmse_sum
                    )

                    # import matplotlib.pyplot as plt
                    # plt.hist(np.squeeze(e) - e_pred)
                    # plt.show()

                # force component error
                f = F[b_range].reshape(n_done_step, -1)
                f_mae, f_mae_sum, f_rmse, f_rmse_sum = _online_err(
                    f - f_pred, 3 * n_atoms, n_done, f_mae_sum, f_rmse_sum
                )

                # magnitude error
                f_pred_mags = np.linalg.norm(f_pred.reshape(-1, 3), axis=1)
                f_mags = np.linalg.norm(f.reshape(-1, 3), axis=1)
                mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = _online_err(
                    f_pred_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum
                )

                # normalized cosine error
                f_pred_norm = f_pred.reshape(-1, 3) / f_pred_mags[:, None]
                f_norm = f.reshape(-1, 3) / f_mags[:, None]
                cos_err = (
                    np.arccos(np.clip(np.einsum('ij,ij->i', f_pred_norm, f_norm), -1, 1))
                    / np.pi
                )
                cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = _online_err(
                    cos_err, n_atoms, n_done, cos_mae_sum, cos_rmse_sum
                )

                # NEW

                if is_test and DEBUG_WRITE:

                    try:
                        with open('test_pred.xyz', 'a') as file:

                            n = r.shape[0]
                            for i, ri in enumerate(r):

                                r_out = ri.reshape(-1, 3)
                                e_out = e_pred[i]
                                f_out = f_pred[i].reshape(-1, 3)

                                ext_xyz_str = (
                                    io.generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                    + '\n'
                                )

                                file.write(ext_xyz_str)

                    except IOError:
                        sys.exit("ERROR: Writing xyz file failed.")

                    try:
                        with open('test_ref.xyz', 'a') as file:

                            n = r.shape[0]
                            for i, ri in enumerate(r):

                                r_out = ri.reshape(-1, 3)
                                e_out = (
                                    None
                                    if not model['use_E']
                                    else np.squeeze(E[b_range][i])
                                )
                                f_out = f[i].reshape(-1, 3)

                                ext_xyz_str = (
                                    io.generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                    + '\n'
                                )
                                file.write(ext_xyz_str)

                    except IOError:
                        sys.exit("ERROR: Writing xyz file failed.")

                    try:
                        with open('test_diff.xyz', 'a') as file:

                            n = r.shape[0]
                            for i, ri in enumerate(r):

                                r_out = ri.reshape(-1, 3)
                                e_out = (
                                    None
                                    if not model['use_E']
                                    else (np.squeeze(E[b_range][i]) - e_pred[i])
                                )
                                f_out = (f[i] - f_pred[i]).reshape(-1, 3)

                                ext_xyz_str = (
                                    io.generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                    + '\n'
                                )
                                file.write(ext_xyz_str)

                    except IOError:
                        sys.exit("ERROR: Writing xyz file failed.")

                # NEW

                sps = n_done / (time.time() - t)  # examples per second
                disp_str = 'energy %.3f/%.3f, ' % (e_mae, e_rmse) if model['use_E'] else ''
                disp_str += 'forces %.3f/%.3f' % (f_mae, f_rmse)
                disp_str = (
                    '{} errors (MAE/RMSE): '.format('Test' if is_test else 'Validation')
                    + disp_str
                )
                sec_disp_str = '@ %.1f geo/s' % sps if b_range is not None else ''

                if self.verbose: ui.callback(
                    n_done,
                    len(test_idxs),
                    disp_str=disp_str,
                    sec_disp_str=sec_disp_str,
                    newline_when_done=False,
                )

            if is_test:
                if self.verbose: ui.callback(
                    DONE,
                    disp_str='Testing on {:,} points'.format(n_test),
                    sec_disp_str=sec_disp_str,
                )
            else:
                if self.verbose: ui.callback(DONE, disp_str=disp_str, sec_disp_str=sec_disp_str)

            if model['use_E']:
                e_rmse_pct = (e_rmse / e_err['rmse'] - 1.0) * 100
            f_rmse_pct = (f_rmse / f_err['rmse'] - 1.0) * 100

            if is_test and n_models == 1:
                n_train = len(model['idxs_train'])
                n_valid = len(model['idxs_valid'])
                print()
                ui.print_two_column_str(
                    ui.color_str('Test errors (MAE/RMSE)', bold=True),
                    '{:,} + {:,} points (training + validation), sigma (length scale): {}'.format(
                        n_train, n_valid, model['sig']
                    ),
                )

                r_unit = 'unknown unit'
                e_unit = 'unknown unit'
                f_unit = 'unknown unit'
                if 'r_unit' in dataset and 'e_unit' in dataset:
                    r_unit = dataset['r_unit']
                    e_unit = dataset['e_unit']
                    f_unit = str(dataset['e_unit']) + '/' + str(dataset['r_unit'])

                format_str = '  {:<18} {:>.4f}/{:>.4f} [{}]'
                if model['use_E']:
                    ui.print_two_column_str(
                        format_str.format('Energy:', e_mae, e_rmse, e_unit),
                        'relative to expected: {:+.1f}%'.format(e_rmse_pct),
                    )

                ui.print_two_column_str(
                    format_str.format('Forces:', f_mae, f_rmse, f_unit),
                    'relative to expected: {:+.1f}%'.format(f_rmse_pct),
                )

                print(format_str.format('  Magnitude:', mag_mae, mag_rmse, r_unit))
                ui.print_two_column_str(
                    format_str.format('  Angle:', cos_mae, cos_rmse, '0-1'),
                    'lower is better',
                )
                print()


            model_needs_update = (
                overwrite
                or (is_test and model['n_test'] < len(test_idxs))
                or (is_validation and not is_model_validated)
            )
            if model_needs_update:

                if is_validation and overwrite:
                    model['n_test'] = 0  # flag the model as not tested

                if is_test:
                    model['n_test'] = len(test_idxs)
                    model['md5_test'] = dataset['md5']

                if model['use_E']:
                    model['e_err'] = {
                        'mae': e_mae.item(),
                        'rmse': e_rmse.item(),
                    }

                model['f_err'] = {'mae': f_mae.item(), 'rmse': f_rmse.item()}
                
                if model_dir: 
                    model_path = os.path.join(model_dir, model_file_names[i])
                    np.savez_compressed(model_path, **model)

                if is_test and model['n_test'] > 0:
                    if self.verbose: log.info('Expected errors were updated in model file.')

            else:
                add_info_str = (
                    'the same number of'
                    if model['n_test'] == len(test_idxs)
                    else 'only {:,}'.format(len(test_idxs))
                )
                if self.verbose: log.warning(
                    'This model has previously been tested on {:,} points, which is why the errors for the current test run with {} points have NOT been used to update the model file.\n'.format(
                        model['n_test'], add_info_str
                    )
                    + ' Set overwrite=True to overwrite.'
                )

            F_rmse.append(f_rmse)

        return F_rmse
    
    def show(self):
        _print_model_properties(self.model)
    
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
        batch_size: int = 2**16,
    ) -> None:
        '''
            batch_size (int, optional): The batch size for batch-predictions.
        '''
        logging.setLoggerClass(ColoredLogger)
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)
        
        gdml_pred = GDMLPredict(self.model)

        z = molDB.get_atomic_numbers()
        mask = np.all(z == self.model['z'], axis=1)
        if np.sum(mask) < len(molDB):
            molDB = molDB[mask]
            print('not all molecules are compatible with the model')

        for batch in molDB.batches(batch_size):
            R = batch.xyz_coordinates.reshape(len(batch), -1)
            E, F = gdml_pred.predict(R)
            if property_to_predict: 
                batch.add_scalar_properties(E, property_to_predict)
            if xyz_derivative_property_to_predict:
                grads = F.reshape(len(batch), -1, 3) * -1
                batch.add_xyz_vectorial_properties(grads, xyz_derivative_property_to_predict)
                
        logging.setLoggerClass(logging.Logger)

def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
  To use Interface_sGDML, please set enviromental variable $sGDML
  to the path of the sGDML executable

  Arguments with their default values:
    MLprog=sGDML               enables this interface
    MLmodelType=sGDML          requests model S

    sgdml.gdml=False           use GDML instead of sGDML
    sgdml.cprsn=False          compress kernel matrix along symmetric 
                               degrees of freedom
    sgdml.no_E=False           do not predict energies
    sgdml.E_cstr=False         include the energy constraints in the
                               kernel
    sgdml.sigma=<s1>[,<s2>[,...]]  
                               set hyperparameter sigma
                                    =<start>:[<step>:]<stop>  
    sgdml.perms                set permutations            
 
  Cite sGDML program:
    S. Chmiela, H. E. Sauceda, I. Poltavsky, K.-R. Müller, A. Tkatchenko,
    Comput. Phys. Commun. 2019, 240, 38

  Cite GDML method, if you use it:
    S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K. T. Schütt,
    K.-R. Müller, Sci. Adv. 2017, 3, e1603015
    
  Cite sGDML method, if you use it:
    S. Chmiela, H. E. Sauceda, K.-R. Müller, A. Tkatchenko,
    Nat. Commun. 2018, 9, 3887
    
'''
    print(helpText)