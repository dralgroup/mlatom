'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_TorchANI: Interface between TorchANI and MLatom                 ! 
  ! Implementations by: Fuchun Ge and Max Pinheiro Jr                         !
  ! Defines the MS-ANI model                                                  !
  ! Implemented by: Mikolaj Martyka                                           ! 
  !---------------------------------------------------------------------------! 
'''

from typing import Any, Union, Dict, List, Callable
from types import FunctionType
import os, sys, uuid
import numpy as np
import tqdm
from collections import OrderedDict

from .. import data, model_cls
from ..model_cls import ml_model, torchani_model, method_model, hyperparameters, hyperparameter, model_tree_node, downloadable_model
from ..decorators import doc_inherit

def median(yp,yt):
    import torch
    return torch.median(torch.abs(yp-yt))
def molDB2ANIdata_state(molDB, 
                  property_to_learn=None,
                  xyz_derivative_property_to_learn=None,
                 use_state=True):
    def molDBiter():
        for mol in molDB.molecules:
            ret = {'species': mol.get_element_symbols(), 'coordinates': mol.xyz_coordinates}
            if property_to_learn is not None:
                ret['energies'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn is not None:
                ret['forces'] = -1 * mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            if use_state:
                ret['state']=mol.__dict__["current_state"]
            if False: #debug
                ret['id']=mol.__dict__["mol_id"]
            yield ret
    from torchani.data import TransformableIterable, IterableAdapter
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))

def unpackData2State(mol_db):
    train_data = data.molecular_database()
    for mol_id, i in enumerate(mol_db):
        for idx, state in enumerate(i.electronic_states):
            new_mol = data.molecule()
            new_mol.read_from_xyz_string(i.get_xyz_string())
            new_mol.current_state = idx
            new_mol.energy = state.energy
            new_mol.add_xyz_derivative_property(state.get_xyz_vectorial_properties("energy_gradients"), 'energy', 'energy_gradients' )
            new_mol.mol_id = mol_id
            train_data.append(new_mol)
    return train_data
#The two MOLDB2ANIdata should be merged, i just didnt have the time to do it.
def molDB2ANIdata(molDB, 
                  property_to_learn=None,
                  xyz_derivative_property_to_learn=None):
    def molDBiter():
        for mol in molDB.molecules:
            ret = {'species': mol.get_element_symbols(), 'coordinates': mol.xyz_coordinates}
            if property_to_learn is not None:
                ret['energies'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn is not None:
                ret['forces'] = -1 * mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            if mol.pbc is not None:
                ret['pbc'] = mol.pbc
            if mol.cell is not None:
                ret['cell'] = mol.cell
            yield ret
    from torchani.data import TransformableIterable, IterableAdapter
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0,
    'pbc': False,
    'cell': 0.0,
}    


class ani(ml_model, torchani_model):
    '''
    Create an `ANI <http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A>`_ (`ANAKIN <https://www.google.com/search?q=Anakin+Skywalker>`_-ME: Accurate NeurAl networK engINe for Molecular Energie) model object. 
    
    Interfaces to `TorchANI <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        device (str, optional): Indicate which device the calculation will be run on. i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        hyperparameters (Dict[str, Any] | :class:`model_cls.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''

    hyperparameters = model_cls.hyperparameters({
        #### Training ####
        'batch_size':           hyperparameter(value=8, minval=1, maxval=1024, optimization_space='linear', dtype=int),
        'max_epochs':           hyperparameter(value=1000000, minval=100, maxval=1000000, optimization_space='log', dtype=int),
        'learning_rate':                    hyperparameter(value=0.001, minval=0.0001, maxval=0.01, optimization_space='log'),
        'early_stopping_learning_rate':     hyperparameter(value=1.0E-5, minval=1.0E-6, maxval=1.0E-4, optimization_space='log'),
        'lr_reduce_patience':   hyperparameter(value=64, minval=16, maxval=256, optimization_space='linear'),
        'lr_reduce_factor':     hyperparameter(value=0.5, minval=0.1, maxval=0.9, optimization_space='linear'),
        'lr_reduce_threshold':  hyperparameter(value=0.0, minval=-0.01, maxval=0.01, optimization_space='linear'),
        #### Loss ####
        'force_coefficient':    hyperparameter(value=0.1, minval=0.05, maxval=5, optimization_space='linear'),
        'median_loss':          hyperparameter(value=False),
        'validation_loss_type': hyperparameter(value='MSE', choices=['MSE', 'mean_RMSE']),
        'loss_type':            hyperparameter(value='weighted', choices=['weighted', 'geometric']),
        #### Network ####
        "neurons":              hyperparameter(value=[[160, 128, 96]]),
        "activation_function":  hyperparameter(value='CELU(0.1)',#lambda: torch.nn.CELU(0.1), 
                                                      optimization_space='choice', choices=["CELU", "ReLU", "GELU"], dtype=(str, type, FunctionType)),
        "fixed_layers":         hyperparameter(value=False),
        #### AEV ####
        'Rcr':                  hyperparameter(value=5.2000e+00, minval=1.0, maxval=10.0, optimization_space='linear'),
        'Rca':                  hyperparameter(value=3.5000e+00, minval=1.0, maxval=10.0, optimization_space='linear'),
        'EtaR':                 hyperparameter(value=[1.6000000e+01]),
        'ShfR':                 hyperparameter(value=[9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00]),
        'Zeta':                 hyperparameter(value=[3.2000000e+01]),
        'ShfZ':                 hyperparameter(value=[1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00]),
        'EtaA':                 hyperparameter(value=[8.0000000e+00]),
        'ShfA':                 hyperparameter(value=[9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00]),
    })
    
    argsdict = {}
    model_file = None
    model = None
    property_name = 'y'
    species_order = []
    program = 'TorchANI'
    meta_data = {
        "genre": "neural network"
    }
    verbose = 1 # 2 can give more training information

    def __init__(self, model_file: str = None, device: str = None, hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters]={}, verbose=1, **kwargs):
        import torch, torchani
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        self.energy_shifter = torchani.utils.EnergyShifter(None)
        if 'key' in kwargs:
            self.key = kwargs['key']
        else:
            self.key = None
        if model_file: 
            if os.path.isfile(model_file):
                self.load(model_file)
            else:
                if self.verbose: print(f'the trained ANI model will be saved in {model_file}')
            self.model_file = model_file

    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            # elif hyperparam in args.data:
            #     self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'ani' in args.data and hyperparam in args.ani.data:
                self.hyperparameters[hyperparam].value = args.ani.data[hyperparam]

    def reset(self):
        super().reset()
        self.model = None


    def save(self, model_file: str = '') -> None:
        '''
        Save the model to file (.pt format).
        
        Arguments:
            model_file (str, optional): The filename that the model to be saved into. If not provided, a randomly generated string will be used.
        '''
        import torch
        if not model_file:
            model_file =f'ani_{str(uuid.uuid4())}.pt'
            self.model_file = model_file
        torch.save(
            {   
                'network': self.networkdict,
                'args': self.argsdict,
                'nn': self.nn.state_dict(),
                'AEV_computer': self.aev_computer,
                'energy_shifter': self.energy_shifter,
            }
            , model_file
        )
        if self.verbose: print(f'model saved in {model_file}')

    def load(self, model_file: str = '', species_order: Union[List[str], None] = None, AEV_parameters: Union[Dict, None] = None, self_energies: Union[List[float], None] = None, reset_parameters: bool = False, method: str = '') -> None:
        '''
        Load a saved ANI model from file (.pt format).

        Arguments:
            model_file (str): The filename of the model to be loaded.
            species_order(List[str], optional): Manually provide species order if it's not present in the saved model.
            AEV_parameters(Dict, optional): Manually provide AEV parameters if it's not present in the saved model.
            self_energies(List[float], optional): Manually provide self energies if it's not present in the saved model.
            reset_parameters(bool): Reset network paramters in the loaded model.
            method(str): Load an ANI method, see also :meth:`ani.load_ani_model`.
        '''
        import torch, torchani
        
        if method:
            self.load_ani_model(method)
            return
            
        # encryption
        model_dict = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)

        if 'property' in model_dict['args']:
            self.property_name = model_dict['args']['property']

        if 'species_order' in model_dict['args']:
            self.species_order = model_dict['args']['species_order']
            # if type(self.species_order[0]) == [int, np.int_]:
            #     self.species_order = [data.atomic_number2element_symbol[z] for z in self.species_order]
            # if self.species_order[0].lower() == self.species_order[0]:
            #     self.species_order = [data.atomic_number2element_symbol[str(z)] for z in self.species_order]
        else:
            print('species order not found, please provide explictly')
            self.species_order = species_order
        self.argsdict.update({'species_order': self.species_order})

        if 'AEV_computer' in model_dict:
            self.aev_computer = model_dict['AEV_computer']
            if 'use_cuda_extension' not in self.aev_computer.__dict__:
                self.aev_computer.use_cuda_extension = False
            self.argsdict.update({'Rcr': self.aev_computer.Rcr, 'Rca': self.aev_computer.Rca, 'EtaR': self.aev_computer.EtaR, 'ShfR': self.aev_computer.ShfR, 'Zeta': self.aev_computer.Zeta, 'ShfZ': self.aev_computer.ShfZ, 'EtaA': self.aev_computer.EtaA, 'ShfA': self.aev_computer.ShfA})
        elif 'Rcr' in model_dict['args']:
            self.AEV_setup(**model_dict['args'])
        else:
            print('AEV parameters not found, please provide explictly')
            self.AEV_setup(**AEV_parameters)
        
        if 'energy_shifter' in model_dict:
            self.energy_shifter = model_dict['energy_shifter']
        elif 'energy_shifter_train' in model_dict['args']:
            self.energy_shifter = model_dict['args']['energy_shifter_train']
        elif 'self_energies_train' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies_train'])
        elif 'self_energies' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies'])
        else:
            print('self energy information not found, please provide explictly')
            self.energy_shifter = torchani.utils.EnergyShifter(self_energies)
        self.energy_shifter.to(self.device)

        if 'network' in model_dict and 'nn' in model_dict:
            self.networkdict = model_dict['network']
            if isinstance(self.networkdict, OrderedDict) or type(self.networkdict) == dict:
                self.neurons = [[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in self.networkdict.values()]
                self.nn = torchani.ANIModel(self.networkdict if isinstance(self.networkdict, OrderedDict) else self.networkdict.values())
            elif type(self.networkdict) == list:
                self.neurons = [[[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in subdict.values()] for subdict in self.networkdict]
                self.nn = torchani.nn.Ensemble([torchani.ANIModel(subdict if isinstance(subdict, OrderedDict) else subdict.values()) for subdict in self.networkdict])
            if reset_parameters:
                self.NN_initialize()
            else:
                self.nn.load_state_dict(model_dict['nn'])
            self.optimizer_setup(**self.hyperparameters)
        else:
            print('network parameters not found')
        
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device)
        self.model.eval()
        if self.verbose: print(f'model loaded from {model_file}')

    def load_ani_model(self, method: str, **hyperparameters) -> None:
        '''
        Load an ANI model.
        
            Arguments:
                method(str): Can be ``'ANI-1x'``, ``'ANI-1ccx'``, or ``'ANI-2x'``.
        '''
        import torch, torchani
        self.hyperparameters.update(hyperparameters)
        if 'ANI-1x'.casefold() in method.casefold():
            model = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        elif 'ANI-1ccx'.casefold() in method.casefold():
            model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device)
        elif 'ANI-2x'.casefold() in method.casefold():
            model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        else:
            print("method not found, please check ANI_methods().supported_methods")
            return

        self.species_order = model.species
        self.argsdict.update({'species_order': self.species_order})
        self.aev_computer = model.aev_computer
        self.networkdict = [OrderedDict(**{k: v for k, v in nn.items()}) for nn in model.neural_networks]
        self.neurons = [[[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in subdict.values()] for subdict in self.networkdict]
        self.nn = model.neural_networks
        self.optimizer_setup(**self.hyperparameters)
        self.energy_shifter = model.energy_shifter
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).float()
        if self.verbose: print(f'loaded {method} model')
    
    @doc_inherit
    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters] = {},
        spliting_ratio: float = 0.8, 
        save_model: bool = True,
        file_to_save_model: str = None,
        check_point: str = None,
        reset_optim_state: bool = False,
        use_last_model: bool = False,
        reset_parameters: bool = False,
        reset_network: bool = False,
        reset_optimizer: bool = False,
        reset_energy_shifter = False,
        # reset_energy_shifter: Union[bool, torchani.utils.EnergyShifter, list] = False, # too heavy to load torchani for definitions
        save_every_epoch: bool = False,
        save_epoch_interval: int = None,
        energy_weighting_function: Callable = None,
        energy_weighting_function_kwargs: dict = {},
        verbose: int = None
    ) -> None:
        r'''
            validation_molecular_database (:class:`mlatom.data.molecular_database` | str, optional): Explicitly defines the database for validation, or use ``'sample_from_molecular_database'`` to make it sampled from the training set.
            hyperparameters (Dict[str, Any] | :class:`mlatom.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
            spliting_ratio (float, optional): The ratio sub-training dataset in the whole training dataset.
            save_model (bool, optional): Whether save the model to disk during training process. Note that the model might be saved many times during training.
            reset_optim_state (bool, optional): Whether to reset the state of optimizer.
            use_last_model (bool, optional): Whether to keep the ``self.model`` as it is in the last training epoch. If ``False``, the best model will be loaded to memory at the end of training.
            reset_parameters (bool, optional): Whether to reset the model's parameters before training.
            reset_network (bool, optional): Whether to re-construct the network before training.
            reset_optimizer (bool, optional): Whether to re-define the optimizer before training .
            save_every_epoch (bool, optional): Whether to save model in every epoch, valid when ``save_model`` is ``True``.
            save_epoch_interval (int, optional): The interval to save each epoch.
            energy_weighting_function (Callable, optional): A weighting function :math:`\mathit{W}(\mathbf{E_ref})` that assign weights to training points based on their reference energies.
            energy_weighting_function_kwargs (dict, optional): Extra weighting function arguments in a dictionary.
        '''
        import torch, torchani
        
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        if verbose:
            self.verbose = verbose
        if file_to_save_model:
            self.model_file = file_to_save_model

        energy_weighting_function_kwargs = {k: (v.value if isinstance(v, hyperparameter) else v) for k, v in energy_weighting_function_kwargs.items()}
        
        if reset_energy_shifter:
            if self.energy_shifter:
                self.energy_shifter_ = self.energy_shifter
            if type(reset_energy_shifter) == list:
                self.energy_shifter = torchani.utils.EnergyShifter(reset_energy_shifter)
            elif isinstance(reset_energy_shifter, torchani.utils.EnergyShifter):
                self.energy_shifter = reset_energy_shifter
            else:
                self.energy_shifter = torchani.utils.EnergyShifter(None)
            
        if not molecular_database._is_uniform_cell():
            print('non-uniform PBC cells detected, using batch_size=1')
            self.hyperparameters.batch_size = 1
        self.data_setup(molecular_database, validation_molecular_database, spliting_ratio, property_to_learn, xyz_derivative_property_to_learn)

        # print energy shifter information
        if self.verbose and self.verbose == 2:
            print('\nDelta self atomic energies: ', )
            for sp, esp in zip(self.species_order, self.energy_shifter.self_energies):
                print(f'{sp} : {esp}')
            print('\n')
            sys.stdout.flush()

        if not self.model:
            self.model_setup(**self.hyperparameters)
        else:
            if 'fixed_layers' in hyperparameters:
                if hyperparameters['fixed_layers'] and type(hyperparameters['fixed_layers'])==list:
                    self.fix_layers(layers_to_fix=hyperparameters['fixed_layers'])

        if reset_network:
            self.NN_setup(**self.hyperparameters)

        if reset_parameters:
            self.NN_initialize()
        
        if reset_optimizer:
            self.optimizer_setup(**self.hyperparameters)

        self.model.train()

        if self.verbose: print(self.model)

        if check_point and os.path.isfile(check_point):
            if self.verbose and self.verbose == 2:
                print(f'\nCheckpoint file {check_point} found. Model training will start from checkpoint.\n')
                sys.stdout.flush()
            checkpoint = torch.load(check_point)
            self.nn.load_state_dict(checkpoint['nn'])
            if not reset_optim_state:
                self.AdamW.load_state_dict(checkpoint['AdamW'])
                self.SGD.load_state_dict(checkpoint['SGD'])
                self.AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
                self.SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    
        def validate():
            total_error  = 0.0
            mae_sum = torch.nn.L1Loss(reduction='sum'); energy_mae = 0.0
            # mse_sum = torch.nn.MSELoss(reduction='sum'); energy_mse = 0.0
            energy_loss = 0.0
            count = 0
            for properties in self.validation_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                pbc = properties['pbc'][0].to(self.device) if 'pbc' in properties else None
                cell = properties['cell'][0].float().to(self.device) if 'cell' in properties else None
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                if callable(energy_weighting_function):
                    weightings_e = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)
                else:
                    weightings_e = 1
                coordinates = properties['coordinates'].to(self.device).float()
                _, predicted_energies = self.model((species, coordinates), pbc=pbc, cell=cell)

                # add MAE, RMSE, loss report
                # energy_mse += mse_sum(predicted_energies*weightings_e, true_energies*weightings_e).item()
                energy_mae += mae_sum(predicted_energies*weightings_e, true_energies*weightings_e).item()
                energy_loss += (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).sum().item()

                if self.hyperparameters.validation_loss_type == 'mean_RMSE':                    
                    total_error += loss_function(predicted_energies, true_energies, weightings_e, reduction='sum').nanmean().sqrt().item()
                else:
                    total_error += loss_function(predicted_energies, true_energies, weightings_e, reduction='sum').nanmean().item()
                count += predicted_energies.shape[0]

            return total_error/count, (total_error/count)**0.5, energy_mae/count, energy_loss/count

        def loss_function(prediction, reference, weightings=1, reduction='none'):
            return torch.nn.functional.mse_loss(prediction*weightings, reference*weightings, reduction=reduction)

        if self.verbose: print("training starting from epoch", self.AdamW_scheduler.last_epoch + 1)
        for _ in range(self.AdamW_scheduler.last_epoch + 1, self.hyperparameters.max_epochs + 1):
            mse, rmse, mae, validation_loss = validate()
            if self.verbose: 
                print('validation loss:', mse, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
                if self.verbose == 2:
                    print('validation MAE:', mae, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
                    print('validation RMSE:', rmse, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
                    print('validation energy loss:', validation_loss, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
                    print('best validation MSE:', self.AdamW_scheduler.best)
            sys.stdout.flush()
            learning_rate = self.AdamW.param_groups[0]['lr']
            if self.verbose: print('learning_rate:',learning_rate)

            if learning_rate < self.hyperparameters.early_stopping_learning_rate:
                break

            if self.AdamW_scheduler.is_better(mse, self.AdamW_scheduler.best) or save_every_epoch:
                if save_model:
                    self.save(self.model_file)
            if save_epoch_interval:
                if (self.AdamW_scheduler.last_epoch + 1)%save_epoch_interval==0 and save_model:
                    self.save(f'{self.model_file}.epoch{self.AdamW_scheduler.last_epoch + 1}')

            self.AdamW_scheduler.step(mse)
            self.SGD_scheduler.step(mse)
            for properties in tqdm.tqdm(
                self.subtraining_set,
                total=len(self.subtraining_set),
                desc="epoch {}".format(self.AdamW_scheduler.last_epoch),
                disable=not self.verbose,
            ):
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                pbc = properties['pbc'][0].to(self.device) if 'pbc' in properties else None
                cell = properties['cell'][0].float().to(self.device) if 'cell' in properties else None

                if callable(energy_weighting_function):
                    weightings_e = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)
                    weightings_f = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)[:, None, None]
                else:
                    weightings_e = 1
                    weightings_f = 1 


                if xyz_derivative_property_to_learn:
                    coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                    true_forces = properties['forces'].to(self.device).float()
                    _, predicted_energies = self.model((species, coordinates), pbc=pbc, cell=cell)
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    # true_energies[true_energies.isnan()]=predicted_energies[true_energies.isnan()]
                    if self.hyperparameters.median_loss:
                        energy_loss= median(predicted_energies,true_energies)
                    else:
                        energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    # true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    if self.hyperparameters.loss_type == 'weighted':
                        loss = energy_loss + self.hyperparameters.force_coefficient * force_loss
                    if self.hyperparameters.loss_type == 'geometric':
                        loss = (energy_loss * force_loss)**0.5
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    _, predicted_energies = self.model((species, coordinates), pbc=pbc, cell=cell)
                    loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()

                self.AdamW.zero_grad()
                self.SGD.zero_grad()
                loss.backward()
                self.AdamW.step()
                self.SGD.step()

            if self.verbose and self.verbose == 2:
                print('Training loss:', loss.item(), 'at epoch', self.AdamW_scheduler.last_epoch,'\n')
                sys.stdout.flush()

            if check_point:
                torch.save({
                    'nn':               self.nn.state_dict(),
                    'AdamW':            self.AdamW.state_dict(),
                    'SGD':              self.SGD.state_dict(),
                    'AdamW_scheduler':  self.AdamW_scheduler.state_dict(),
                    'SGD_scheduler':    self.SGD_scheduler.state_dict(),
                }, check_point)

        # print the performance of the best model
        if self.verbose and self.verbose == 2:
            print('\nPerformance of the best model on validation set')
            _, best_rmse, best_mae, best_val_loss = validate()
            print('best validation MAE:', best_mae,)
            print('best validation RMSE:', best_rmse,)
            print('best validation energy loss:', best_val_loss,)

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
            batch_size: int = 2**16,
        ) -> None:
        '''
            batch_size (int, optional): The batch size for batch-predictions.
        '''
        import torch, torchani
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)
        
        if not molDB._is_uniform_cell():
            batch_size = 1
        
        for batch in molDB.batches(batch_size):
            for properties in molDB2ANIdata(batch).species_to_indices(self.species_order).collate(batch_size, PADDING):
                species = properties['species'].to(self.device)
                xyz_coordinates = properties['coordinates'].float().to(self.device)
                break
            pbc = properties['pbc'][0].to(self.device) if 'pbc' in properties else None
            cell = properties['cell'][0].float().to(self.device) if 'cell' in properties else None
            if pbc is not None and cell is not None:
                xyz_coordinates = torchani.utils.map2central(cell, xyz_coordinates, pbc)
            xyz_coordinates = xyz_coordinates.requires_grad_(bool(xyz_derivative_property_to_predict or hessian_to_predict))
            ANI_NN_energies = self.energy_shifter(self.model((species, xyz_coordinates), pbc=pbc, cell=cell)).energies
            if property_to_predict: 
                batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), property_to_predict)
            if xyz_derivative_property_to_predict or hessian_to_predict:
                ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energies.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                if xyz_derivative_property_to_predict:
                    grads = ANI_NN_energy_gradients.detach().cpu().numpy()
                    batch.add_xyz_vectorial_properties(grads, xyz_derivative_property_to_predict)
                if hessian_to_predict:
                    ANI_NN_hessians = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energies)
                    batch.add_hessian_properties(ANI_NN_hessians.detach().cpu().numpy(), hessian_to_predict)


    def AEV_setup(self, **kwargs):
        import torch, torchani
        kwargs = hyperparameters(kwargs)
        Rcr = kwargs.Rcr
        Rca = kwargs.Rca
        EtaR = torch.tensor(kwargs.EtaR).to(self.device)
        ShfR = torch.tensor(kwargs.ShfR).to(self.device)
        Zeta = torch.tensor(kwargs.Zeta).to(self.device)
        ShfZ = torch.tensor(kwargs.ShfZ).to(self.device)
        EtaA = torch.tensor(kwargs.EtaA).to(self.device)
        ShfA = torch.tensor(kwargs.ShfA).to(self.device)
        self.aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, len(self.species_order))
        self.argsdict.update({'Rcr': Rcr, 'Rca': Rca, 'EtaR': EtaR, 'ShfR': ShfR, 'Zeta': Zeta, 'ShfZ': ShfZ, 'EtaA': EtaA, 'ShfA': ShfA, 'species_order': self.species_order})

    def NN_setup(self, **kwargs):
        import torch, torchani
        kwargs = hyperparameters(kwargs)
        if len(kwargs.neurons) == 1:
            self.neurons = [kwargs.neurons[0].copy() for _ in range(len(self.species_order))]
        else:
            self.neurons = kwargs.neurons

        self.networkdict = OrderedDict()
        for i, specie in enumerate(self.species_order):
            self.neurons[i] += [1]
            layers = [torch.nn.Linear(self.aev_computer.aev_length, self.neurons[i][0])]
            for j in range(len(self.neurons[i]) - 1):
                if type(kwargs.activation_function) == str:
                    act_fun = kwargs.activation_function
                    if '(' in act_fun:
                        xx = act_fun.split('(')
                        act_fun = xx[0]
                        alpha = float(xx[1].strip(')'))
                        layers += [torch.nn.__dict__[act_fun](alpha)]
                    else:  
                        layers += [torch.nn.__dict__[act_fun]()]
                elif callable(kwargs.activation_function):
                    layers += [kwargs.activation_function()]
                layers += [torch.nn.Linear(self.neurons[i][j], self.neurons[i][j + 1])]
            self.networkdict[specie] = torch.nn.Sequential(*layers)

        self.nn = torchani.ANIModel(self.networkdict)

        self.NN_initialize()
        self.optimizer_setup(**kwargs)  

    def NN_initialize(self, a: float = 1.0) -> None:
        '''
        Reset the network parameters using :meth:`torch.nn.init.kaiming_normal_`.

        Arguments:
            a(float): Check `torch.nn.init.kaiming_normal_() <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_>`_.
        '''
        import torch
        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_params)

    def optimizer_setup(self, **kwargs):
        import torch
        kwargs = hyperparameters(kwargs)
        if isinstance(self.networkdict, OrderedDict) or type(self.networkdict) == dict:
            wlist2d = [
                [
                    {'params': [self.networkdict[specie][j * 2].weight]} if j == 0 or j == len(self.neurons[i]) - 1 else {'params': [self.networkdict[specie][j*2].weight], 'weight_decay': 0.0001 / 10**j} for j in range(len(self.neurons[i]))
                ] for i, specie in enumerate(self.species_order)
            ]

            blist2d = [
                [
                    {'params': [self.networkdict[specie][j * 2].bias]} for j in range(len(self.neurons[i]))
                ] for i, specie in enumerate(self.species_order)
            ]
            self.AdamW = torch.optim.AdamW([i for j in wlist2d for i in j],lr=kwargs.learning_rate)
            self.SGD = torch.optim.SGD([i for j in blist2d for i in j], lr=kwargs.learning_rate)
        elif type(self.networkdict) == list:
            wlist3d =[[
                [
                    {'params': [subdict[specie][j * 2].weight]} if j == 0 or j == len(self.neurons[k][i]) - 1 else {'params': [subdict[specie][j*2].weight], 'weight_decay': 0.0001 / 10**j} for j in range(len(self.neurons[k][i]))
                ] for i, specie in enumerate(self.species_order)
            ] for k, subdict in enumerate(self.networkdict)]

            blist3d = [[
                [
                    {'params': [subdict[specie][j * 2].bias]} for j in range(len(self.neurons[k][i]))
                ] for i, specie in enumerate(self.species_order)
            ] for k, subdict in enumerate(self.networkdict)]

            self.AdamW = torch.optim.AdamW([i for k in wlist3d for j in k for i in j],lr=kwargs.learning_rate)
            self.SGD = torch.optim.SGD([i for k in blist3d for j in k  for i in j], lr=kwargs.learning_rate)
            
        self.AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.AdamW, factor=kwargs.lr_reduce_factor, patience=kwargs.lr_reduce_patience, threshold=kwargs.lr_reduce_threshold)
        self.SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.SGD, factor=kwargs.lr_reduce_factor, patience=kwargs.lr_reduce_patience, threshold=kwargs.lr_reduce_threshold)

    def model_setup(self, **kwargs):
        import torchani
        self.AEV_setup(**kwargs)
        self.NN_setup(**kwargs)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).float().to(self.device)

    def fix_layers(self, layers_to_fix: Union[List[List[int]],List[int]]):
        '''
        Fix specific layers to be non-trainable for each element.

        Arguments:
            layers_to_fix (List): Should be: 
            
                - A list of integers. Layers indicated by the integers will be fixed.
                - A list of lists of integers. Each sub-list defines the layers to be fixed for each species, in the order of `self.species_order`. 
        Examples:

            .. code-block:: python
            
                import mlatom as ml
                >>> ani = ml.ani(model_file='ANI.pt')
                model loade from ANI.pt
                >>> ani.model # show model summary
                Sequential(
                (0): AEVComputer()
                (1): ANIModel(
                    (C): Sequential(
                    (0): Linear(in_features=240, out_features=160, bias=True)
                    (1): CELU(alpha=0.1)
                    (2): Linear(in_features=160, out_features=128, bias=True)
                    (3): CELU(alpha=0.1)
                    (4): Linear(in_features=128, out_features=96, bias=True)
                    (5): CELU(alpha=0.1)
                    (6): Linear(in_features=96, out_features=1, bias=True)
                    )
                    (H): Sequential(
                    (0): Linear(in_features=240, out_features=160, bias=True)
                    (1): CELU(alpha=0.1)
                    (2): Linear(in_features=160, out_features=128, bias=True)
                    (3): CELU(alpha=0.1)
                    (4): Linear(in_features=128, out_features=96, bias=True)
                    (5): CELU(alpha=0.1)
                    (6): Linear(in_features=96, out_features=1, bias=True)
                    )
                    (O): Sequential(
                    (0): Linear(in_features=240, out_features=160, bias=True)
                    (1): CELU(alpha=0.1)
                    (2): Linear(in_features=160, out_features=128, bias=True)
                    (3): CELU(alpha=0.1)
                    (4): Linear(in_features=128, out_features=96, bias=True)
                    (5): CELU(alpha=0.1)
                    (6): Linear(in_features=96, out_features=1, bias=True)
                    )
                  )
                )
                >>> parameters = dict(a.model.named_parameters())
                >>> {key: parameters[key].requires_grad for key in parameters} # show the trainability of parameters
                {'1.C.0.weight': True, '1.C.0.bias': True, '1.C.2.weight': True, '1.C.2.bias': True, '1.C.4.weight': True, '1.C.4.bias': True, '1.C.6.weight': True, '1.C.6.bias': True, '1.H.0.weight': True, '1.H.0.bias': True, '1.H.2.weight': True, '1.H.2.bias': True, '1.H.4.weight': True, '1.H.4.bias': True, '1.H.6.weight': True, '1.H.6.bias': True, '1.O.0.weight': True, '1.O.0.bias': True, '1.O.2.weight': True, '1.O.2.bias': True, '1.O.4.weight': True, '1.O.4.bias': True, '1.O.6.weight': True, '1.O.6.bias': True}
                >>> ani.fix_layers([[0, 2], [0], [4]]) # fix layer 0 and 2 for C, layer 0 for H, and layer 4 for O
                >>> {key: parameters[key].requires_grad for key in parameters}
                {'1.C.0.weight': False, '1.C.0.bias': False, '1.C.2.weight': False, '1.C.2.bias': False, '1.C.4.weight': True, '1.C.4.bias': True, '1.C.6.weight': True, '1.C.6.bias': True, '1.H.0.weight': False, '1.H.0.bias': False, '1.H.2.weight': True, '1.H.2.bias': True, '1.H.4.weight': True, '1.H.4.bias': True, '1.H.6.weight': True, '1.H.6.bias': True, '1.O.0.weight': True, '1.O.0.bias': True, '1.O.2.weight': True, '1.O.2.bias': True, '1.O.4.weight': False, '1.O.4.bias': False, '1.O.6.weight': True, '1.O.6.bias': True}
                >>> ani.fix_layers([[0, 2, 4]]) # fix layer 0, 2, and 4 for all networks
                {'1.C.0.weight': False, '1.C.0.bias': False, '1.C.2.weight': False, '1.C.2.bias': False, '1.C.4.weight': False, '1.C.4.bias': False, '1.C.6.weight': True, '1.C.6.bias': True, '1.H.0.weight': False, '1.H.0.bias': False, '1.H.2.weight': False, '1.H.2.bias': False, '1.H.4.weight': False, '1.H.4.bias': False, '1.H.6.weight': True, '1.H.6.bias': True, '1.O.0.weight': False, '1.O.0.bias': False, '1.O.2.weight': False, '1.O.2.bias': False, '1.O.4.weight': False, '1.O.4.bias': False, '1.O.6.weight': True, '1.O.6.bias': True}
                
        '''
        if layers_to_fix:
            if len(layers_to_fix) == 1:
                layers_to_fix = layers_to_fix * len(self.species_order)
            for name, parameter in self.model.named_parameters():
                indices = name.split('.')
                if int(indices[-2]) in layers_to_fix[self.species_order.index(indices[-3] if indices[-3] in data.element_symbol2atomic_number else data.atomic_number2element_symbol[int(indices[-3])])]:
                    parameter.requires_grad=False

    def data_setup(self, molecular_database, validation_molecular_database, spliting_ratio,
                   property_to_learn, xyz_derivative_property_to_learn, ):
        import torch
        assert molecular_database, 'provide molecular database'

        self.property_name = property_to_learn
        
        data_element_symbols = list(np.sort(np.unique(np.concatenate(molecular_database.element_symbols))))

        if not self.species_order: 
            self.species_order = data_element_symbols
        else:
            for element in data_element_symbols:
                if element not in self.species_order:
                    print('element(s) outside supported species detected, please check the database')
                    return
                
        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]
        elif not validation_molecular_database:
            raise NotImplementedError("please specify validation_molecular_database or set it to 'sample_from_molecular_database'")

        if self.energy_shifter.self_energies is None:
            if np.isnan(molecular_database.get_properties(property_to_learn)).sum():
                molDB2ANIdata(molecular_database.filter_by_property(property_to_learn), property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order)
                self.subtraining_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
            else:   
                self.subtraining_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order).species_to_indices(self.species_order).shuffle()
        else:
            self.subtraining_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()

        if len(self.species_order) != len(self.energy_shifter.self_energies):
            true_species_order = sorted(data_element_symbols, key=lambda x: self.species_order.index(x))
            expanded_self_energies = np.zeros((len(self.species_order)))
            for ii, sp in enumerate(self.species_order):
                if sp in true_species_order:
                    expanded_self_energies[ii] = self.energy_shifter.self_energies[true_species_order.index(sp)]
                elif 'energy_shifter_' in self.__dict__:
                    expanded_self_energies[ii] = self.energy_shifter_.self_energies[ii]

            self.energy_shifter.self_energies = torch.tensor(expanded_self_energies)
        
        self.validation_set = molDB2ANIdata(validation_molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
        
        self.energy_shifter = self.energy_shifter.to(self.device)
        
        self.subtraining_set = self.subtraining_set.collate(self.hyperparameters.batch_size, PADDING)
        self.validation_set = self.validation_set.collate(self.hyperparameters.batch_size, PADDING)

        self.argsdict.update({'self_energies': self.energy_shifter.self_energies, 'property': self.property_name})
class msani(ml_model, torchani_model):
    '''
    Create an `MS-ANI', an extension of the ANI NN model for multi-state learning. First described in <10.26434/chemrxiv-2024-dtc1w>. 
    
    Interfaces to `TorchANI <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        device (str, optional): Indicate which device the calculation will be run on. i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        hyperparameters (Dict[str, Any] | :class:`mlatom.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''

    hyperparameters = model_cls.hyperparameters({
        #### Training ####
        'batch_size':           hyperparameter(value=16, minval=1, maxval=1024, optimization_space='linear', dtype=int),
        'max_epochs':           hyperparameter(value=1000000, minval=100, maxval=1000000, optimization_space='log', dtype=int),
        'learning_rate':                    hyperparameter(value=0.001, minval=0.0001, maxval=0.01, optimization_space='log'),
        'early_stopping_learning_rate':     hyperparameter(value=1.0E-5, minval=1.0E-6, maxval=1.0E-4, optimization_space='log'),
        'lr_reduce_patience':   hyperparameter(value=64, minval=16, maxval=256, optimization_space='linear'),
        'lr_reduce_factor':     hyperparameter(value=0.5, minval=0.1, maxval=0.9, optimization_space='linear'),
        'lr_reduce_threshold':  hyperparameter(value=0.0, minval=-0.01, maxval=0.01, optimization_space='linear'),
        #### Loss ####
        'force_coefficient':    hyperparameter(value=0.1, minval=0.05, maxval=5, optimization_space='linear'),
        'median_loss':          hyperparameter(value=False),
        'gap_coefficient':             hyperparameter(value=1.0, minval=0.05, maxval=5, optimization_space='linear'),
        #### Network ####
        "neurons":              hyperparameter(value=[[160, 128, 96]]),
        "activation_function":  hyperparameter(value='CELU(0.1)',#lambda: torch.nn.CELU(0.1), 
                                                      optimization_space='choice', choices=["CELU", "ReLU", "GELU"], dtype=(str, type, FunctionType)),
        "fixed_layers":         hyperparameter(value=False),
        #### AEV ####
        'Rcr':                  hyperparameter(value=5.2000e+00, minval=1.0, maxval=10.0, optimization_space='linear'),
        'Rca':                  hyperparameter(value=3.5000e+00, minval=1.0, maxval=10.0, optimization_space='linear'),
        'EtaR':                 hyperparameter(value=[1.6000000e+01]),
        'ShfR':                 hyperparameter(value=[9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00]),
        'Zeta':                 hyperparameter(value=[3.2000000e+01]),
        'ShfZ':                 hyperparameter(value=[1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00]),
        'EtaA':                 hyperparameter(value=[8.0000000e+00]),
        'ShfA':                 hyperparameter(value=[9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00]),
    })
    
    argsdict = {}
    model_file = None
    model = None
    property_name = 'y'
    species_order = []
    program = 'TorchANI'
    meta_data = {
        "genre": "neural network"
    }
    verbose = 1

    def __init__(self, model_file: str = None, device: str = None, hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters]={}, verbose=1, nstates=1,validate_train=True):
        import torch, torchani
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        self.nstates = nstates
        self.validate_train = validate_train
        self.energy_shifter = torchani.utils.EnergyShifter(None)
        if model_file: 
            if os.path.isfile(model_file):
                self.load(model_file)
            else:
                if self.verbose: print(f'the trained MS-ANI model will be saved in {model_file}')
            self.model_file = model_file
        self.hyperparameters.batch_size = self.hyperparameters.batch_size*self.nstates
    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            # elif hyperparam in args.data:
            #     self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'ani' in args.data and hyperparam in args.ani.data:
                self.hyperparameters[hyperparam].value = args.ani.data[hyperparam]
    
    def reset(self):
        super().reset()
        self.model = None


    def save(self, model_file: str = '') -> None:
        '''
        Save the model to file (.pt format).
        
        Arguments:
            model_file (str, optional): The filename that the model to be saved into. If not provided, a randomly generated string will be used.
        '''
        import torch
        if not model_file:
            model_file =f'ani_{str(uuid.uuid4())}.pt'
            self.model_file = model_file
        torch.save(
            {   
                'network': self.networkdict,
                'args': self.argsdict,
                'nn': self.nn.state_dict(),
                'AEV_computer': self.aev_computer,
                'energy_shifter': self.energy_shifter,
            }
            , model_file
        )
        if self.verbose: print(f'model saved in {model_file}')

    def load(self, model_file: str = '', species_order: Union[List[str], None] = None, AEV_parameters: Union[Dict, None] = None, self_energies: Union[List[float], None] = None, reset_parameters: bool = False, method: str = '') -> None:
        '''
        Load a saved ANI model from file (.pt format).

        Arguments:
            model_file (str): The filename of the model to be loaded.
            species_order(List[str], optional): Manually provide species order if it's not present in the saved model.
            AEV_parameters(Dict, optional): Manually provide AEV parameters if it's not present in the saved model.
            self_energies(List[float], optional): Manually provide self energies if it's not present in the saved model.
            reset_parameters(bool): Reset network paramters in the loaded model.
            method(str): Load an ANI method, see also :meth:`ani.load_ani_model`.
        '''
        import torch, torchani
        if method:
            self.load_ani_model(method)
            return
        
        model_dict = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)

        if 'property' in model_dict['args']:
            self.property_name = model_dict['args']['property']

        if 'species_order' in model_dict['args']:
            self.species_order = model_dict['args']['species_order']
            # if type(self.species_order[0]) == [int, np.int_]:
            #     self.species_order = [data.atomic_number2element_symbol[z] for z in self.species_order]
            # if self.species_order[0].lower() == self.species_order[0]:
            #     self.species_order = [data.atomic_number2element_symbol[str(z)] for z in self.species_order]
        else:
            print('species order not found, please provide explictly')
            self.species_order = species_order
        self.argsdict.update({'species_order': self.species_order})

        if 'AEV_computer' in model_dict:
            self.aev_computer = model_dict['AEV_computer']
            if 'use_cuda_extension' not in self.aev_computer.__dict__:
                self.aev_computer.use_cuda_extension = False
            self.argsdict.update({'Rcr': self.aev_computer.Rcr, 'Rca': self.aev_computer.Rca, 'EtaR': self.aev_computer.EtaR, 'ShfR': self.aev_computer.ShfR, 'Zeta': self.aev_computer.Zeta, 'ShfZ': self.aev_computer.ShfZ, 'EtaA': self.aev_computer.EtaA, 'ShfA': self.aev_computer.ShfA})
        elif 'Rcr' in model_dict['args']:
            self.AEV_setup(**model_dict['args'])
        else:
            print('AEV parameters not found, please provide explictly')
            self.AEV_setup(**AEV_parameters)
        
        if 'energy_shifter' in model_dict:
            self.energy_shifter = model_dict['energy_shifter']
        elif 'energy_shifter_train' in model_dict['args']:
            self.energy_shifter = model_dict['args']['energy_shifter_train']
        elif 'self_energies_train' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies_train'])
        elif 'self_energies' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies'])
        else:
            print('self energy information not found, please provide explictly')
            self.energy_shifter = torchani.utils.EnergyShifter(self_energies)
        self.energy_shifter.to(self.device)

        if 'network' in model_dict and 'nn' in model_dict:
            self.networkdict = model_dict['network']
            if isinstance(self.networkdict, OrderedDict) or type(self.networkdict) == dict:
                self.neurons = [[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in self.networkdict.values()]
                self.nn = torchani.ANIModel(self.networkdict if isinstance(self.networkdict, OrderedDict) else self.networkdict.values())
            elif type(self.networkdict) == list:
                self.neurons = [[[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in subdict.values()] for subdict in self.networkdict]
                self.nn = torchani.nn.Ensemble([torchani.ANIModel(subdict if isinstance(subdict, OrderedDict) else subdict.values()) for subdict in self.networkdict])
            if reset_parameters:
                self.NN_initialize()
            else:
                self.nn.load_state_dict(model_dict['nn'])
            self.optimizer_setup(**self.hyperparameters)
        else:
            print('network parameters not found')
        
        from .torchani_heavy_parts import StateInputNet
        self.model = torch.nn.DataParallel(StateInputNet(self.aev_computer, self.nn)).to(self.device)
        self.model.eval()
        if self.verbose: print(f'model loaded from {model_file}')

    def load_ani_model(self, method: str, **hyperparameters) -> None:
        '''
        Load an ANI model.
        
            Arguments:
                method(str): Can be ``'ANI-1x'``, ``'ANI-1ccx'``, or ``'ANI-2x'``.
        '''
        import torch, torchani
        self.hyperparameters.update(hyperparameters)
        if 'ANI-1x'.casefold() in method.casefold():
            model = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        elif 'ANI-1ccx'.casefold() in method.casefold():
            model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device)
        elif 'ANI-2x'.casefold() in method.casefold():
            model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        else:
            print("method not found, please check ANI_methods().supported_methods")
            return

        self.species_order = model.species
        self.argsdict.update({'species_order': self.species_order})
        self.aev_computer = model.aev_computer
        self.networkdict = [OrderedDict(**{k: v for k, v in nn.items()}) for nn in model.neural_networks]
        self.neurons = [[[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in subdict.values()] for subdict in self.networkdict]
        self.nn = model.neural_networks
        self.optimizer_setup(**self.hyperparameters)
        self.energy_shifter = model.energy_shifter
        from .torchani_heavy_parts import StateInputNet
        self.model = torch.nn.DataParallel(StateInputNet(self.aev_computer, self.nn)).to(self.device).float()
        if self.verbose: print(f'loaded {method} model')
    
    @doc_inherit
    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters] = {},
        spliting_ratio: float = 0.8, 
        save_model: bool = True,
        check_point: str = None,
        reset_optim_state: bool = False,
        use_last_model: bool = False,
        reset_parameters: bool = False,
        reset_network: bool = False,
        reset_optimizer: bool = False,
        save_every_epoch: bool = False,
        energy_weighting_function: Callable = None,
        energy_weighting_function_kwargs: dict = {},
    ) -> None:
        r'''
            validation_molecular_database (:class:`mlatom.data.molecular_database` | str, optional): Explicitly defines the database for validation, or use ``'sample_from_molecular_database'`` to make it sampled from the training set.
            hyperparameters (Dict[str, Any] | :class:`mlatom.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
            spliting_ratio (float, optional): The ratio sub-training dataset in the whole training dataset.
            save_model (bool, optional): Whether save the model to disk during training process. Note that the model might be saved many times during training.
            reset_optim_state (bool, optional): Whether to reset the state of optimizer.
            use_last_model (bool, optional): Whether to keep the ``self.model`` as it is in the last training epoch. If ``False``, the best model will be loaded to memory at the end of training.
            reset_parameters (bool, optional): Whether to reset the model's parameters before training.
            reset_network (bool, optional): Whether to re-construct the network before training.
            reset_optimizer (bool, optional): Whether to re-define the optimizer before training .
            save_every_epoch (bool, optional): Whether to save model in every epoch, valid when ``save_model`` is ``True``.
            energy_weighting_function (Callable[Array-like], optional): A weighting function :math:`\mathit{W}(\mathbf{E_ref})` that assign weights to training points based on their reference energies.
        '''
        import torch
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        energy_weighting_function_kwargs = {k: (v.value if isinstance(v, hyperparameter) else v) for k, v in energy_weighting_function_kwargs.items()}
        
        self.data_setup(molecular_database, validation_molecular_database, spliting_ratio, property_to_learn, xyz_derivative_property_to_learn)

        if not self.model:
            self.model_setup(**self.hyperparameters)

        if reset_network:
            self.NN_setup(**self.hyperparameters)

        if reset_parameters:
            self.NN_initialize()
        
        if reset_optimizer:
            self.optimizer_setup(**self.hyperparameters)

        self.model.train()

        if self.verbose: print(self.model)

        if check_point and os.path.isfile(check_point):
            checkpoint = torch.load(check_point)
            self.nn.load_state_dict(checkpoint['nn'])
            if not reset_optim_state:
                self.AdamW.load_state_dict(checkpoint['AdamW'])
                self.SGD.load_state_dict(checkpoint['SGD'])
                self.AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
                self.SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    
        def validate():
            total_mse = 0.0
            count = 0
            for properties in self.validation_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                if callable(energy_weighting_function):
                    weightings_e = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)
                else:
                    weightings_e = 1
                coordinates = properties['coordinates'].to(self.device).float()
                state = properties['state'].to(self.device).float()
                _, predicted_energies = self.model(species, coordinates,state)
                total_mse += loss_function(predicted_energies, true_energies, weightings_e, reduction='sum').nanmean().item()
                count += predicted_energies.shape[0]
            return total_mse/count
        
        def validate_training():
            total_mse = 0.0
            count = 0
            for properties in self.validation_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                state = properties['state'].to(self.device).float()
                
                true_gap_list = []

                for i in range(int(len(true_energies)/self.nstates)):
                    for j in range(1,self.nstates):
                        true_gap_list.append(abs(true_energies[i+j]-true_energies[i+j-1]))
                true_gaps = torch.FloatTensor(true_gap_list).to(self.device)
                

                if callable(energy_weighting_function):
                    weightings_e = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)
                    weightings_f = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)[:, None, None]
                else:
                    weightings_e = 1
                    weightings_f = 1
                if xyz_derivative_property_to_learn:
                    coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                    true_forces = properties['forces'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates,state)

                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i+j]-predicted_energies[i+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    # true_energies[true_energies.isnan()]=predicted_energies[true_energies.isnan()]
                    if self.hyperparameters.median_loss:
                        energy_loss= median(predicted_energies,true_energies)
                    else:
                        energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1) / num_atoms[0].sqrt()).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_
                    # true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    loss = energy_loss + self.hyperparameters.force_coefficient * force_loss + self.hyperparameters.gap_coefficient * gap_loss
                    total_mse += loss.item()
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates, state)
                    
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i+j]-predicted_energies[i+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    
                    energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1) / num_atoms[0].sqrt()).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_
                    loss = energy_loss + self.hyperparameters.gap_coefficient * gap_loss
                    total_mse += loss.item()
            return total_mse
        
        def loss_function(prediction, reference, weightings=1, reduction='none'):
            return torch.nn.functional.mse_loss(prediction*weightings, reference*weightings, reduction=reduction)

        if self.verbose: print("training starting from epoch", self.AdamW_scheduler.last_epoch + 1)
        for _ in range(self.AdamW_scheduler.last_epoch + 1, self.hyperparameters.max_epochs + 1):
            if self.validate_train == True:
                rmse = validate_training()
            else:
                rmse = validate()
            if self.verbose: print('validation MSE:', rmse, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
            sys.stdout.flush()
            learning_rate = self.AdamW.param_groups[0]['lr']
            if self.verbose: print('learning_rate:',learning_rate)

            if learning_rate < self.hyperparameters.early_stopping_learning_rate:
                break

            if self.AdamW_scheduler.is_better(rmse, self.AdamW_scheduler.best) or save_every_epoch:
                if save_model:
                    self.save(self.model_file)

            self.AdamW_scheduler.step(rmse)
            self.SGD_scheduler.step(rmse)
            for properties in tqdm.tqdm(
                self.subtraining_set,
                total=len(self.subtraining_set),
                desc="epoch {}".format(self.AdamW_scheduler.last_epoch),
                disable=not self.verbose,
            ):
                
                true_energies = properties['energies'].to(self.device).float()
                
                true_gap_list = []

                for i in range(int(len(true_energies)/self.nstates)):
                    for j in range(1,self.nstates):
                        true_gap_list.append(abs(true_energies[i+j]-true_energies[i+j-1]))
                true_gaps = torch.FloatTensor(true_gap_list).to(self.device)

                species = properties['species'].to(self.device)
                state = properties['state'].to(self.device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                if callable(energy_weighting_function):
                    weightings_e = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)
                    weightings_f = energy_weighting_function(self.energy_shifter((species, true_energies)).energies, **energy_weighting_function_kwargs)[:, None, None]
                else:
                    weightings_e = 1
                    weightings_f = 1 


                if xyz_derivative_property_to_learn:
                    coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                    true_forces = properties['forces'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates,state)
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i+j]-predicted_energies[i+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    #print(predicted_gaps)
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    # true_energies[true_energies.isnan()]=predicted_energies[true_energies.isnan()]
                    if self.hyperparameters.median_loss:
                        energy_loss= median(predicted_energies,true_energies)
                    else:
                        energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                   # print(predicted_gaps.shape, true_gaps.shape)
                    #print(num_atoms.sqrt())
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1) / num_atoms[0].sqrt()).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_
                    # true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    loss = energy_loss + self.hyperparameters.force_coefficient * force_loss + self.hyperparameters.gap_coefficient * gap_loss
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates, state)
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i+j]-predicted_energies[i+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    #print(predicted_gaps)

                    energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1) / num_atoms[0].sqrt()).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_
                    loss = energy_loss + self.hyperparameters.gap_coefficient * gap_loss

                self.AdamW.zero_grad()
                self.SGD.zero_grad()
                loss.backward()
                self.AdamW.step()
                self.SGD.step()

            if check_point:
                torch.save({
                    'nn':               self.nn.state_dict(),
                    'AdamW':            self.AdamW.state_dict(),
                    'SGD':              self.SGD.state_dict(),
                    'AdamW_scheduler':  self.AdamW_scheduler.state_dict(),
                    'SGD_scheduler':    self.SGD_scheduler.state_dict(),
                }, check_point)

        if save_model and not use_last_model:
            self.load(self.model_file)

    @doc_inherit
    def predict(
            self, 
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            nstates = 1,
            current_state=0,
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
        import torch, torchani
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)
        
        for batch in molDB.batches(batch_size):
            state_energies =[]
            state_gradients =[]
            state_hessians = []
            for i in range(nstates):
                for properties in molDB2ANIdata_state(batch, use_state=False).species_to_indices(self.species_order).collate(batch_size, PADDING):
                    species = properties['species'].to(self.device)
                    state = torch.full((len(batch),),i).to(self.device)
                    xyz_coordinates = properties['coordinates'].float().to(self.device).requires_grad_(bool(xyz_derivative_property_to_predict or hessian_to_predict))
                    break
                ANI_NN_energies = self.energy_shifter(self.model(species, xyz_coordinates, state)).energies
                #print("raw energy", self.model(species,xyz_coordinates,state).energies)
                #print("raw energy - shifted energy = SAE", torch.sub(self.model(species,xyz_coordinates,state).energies, ANI_NN_energies))
                #print("total energy", ANI_NN_energies)
                if property_to_predict:
                    if nstates>1:
                        batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), property_to_predict+"_state"+str(i))
                        state_energies.append(ANI_NN_energies.detach().cpu().numpy())
                    else:
                        batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), property_to_predict)
                        state_energies.append(ANI_NN_energies.detach().cpu().numpy())                        
                if xyz_derivative_property_to_predict or hessian_to_predict:
                    ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energies.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                    if xyz_derivative_property_to_predict:
                        if nstates>1:
                            grads = ANI_NN_energy_gradients.detach().cpu().numpy()
                            state_gradients.append(grads)
                            batch.add_xyz_vectorial_properties(grads, xyz_derivative_property_to_predict+"_state"+str(i))
                        else:
                            grads = ANI_NN_energy_gradients.detach().cpu().numpy()
                            state_gradients.append(grads)
                            batch.add_xyz_vectorial_properties(grads, xyz_derivative_property_to_predict)   
                    if hessian_to_predict:
                        ANI_NN_hessians = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energies)
                        state_hessians.append(ANI_NN_hessians.detach().cpu().numpy())
                        batch.add_scalar_properties(ANI_NN_hessians.detach().cpu().numpy(), hessian_to_predict)
            for idx, mol in enumerate(batch):
                mol_copy = mol.copy()
                mol_copy.electronic_states = []
                for _ in range(nstates - len(mol.electronic_states)):
                    mol.electronic_states.append(mol_copy.copy())
                if calculate_energy:
                    for i in range(nstates):
                        mol.electronic_states[i].energy = state_energies[i][idx]
                    mol.__dict__["energy"] = mol.electronic_states[current_state].energy
                if xyz_derivative_property_to_predict:
                    for i in range(nstates):
                         mol.electronic_states[i].add_xyz_derivative_property(np.array(state_gradients[i][idx]).astype(float), 'energy', 'energy_gradients')
                    mol.add_xyz_derivative_property(np.array(state_gradients[current_state][idx]).astype(float), 'energy', 'energy_gradients')
                        
                    


    def AEV_setup(self, **kwargs):
        import torch, torchani
        kwargs = hyperparameters(kwargs)
        Rcr = kwargs.Rcr
        Rca = kwargs.Rca
        EtaR = torch.tensor(kwargs.EtaR).to(self.device)
        ShfR = torch.tensor(kwargs.ShfR).to(self.device)
        Zeta = torch.tensor(kwargs.Zeta).to(self.device)
        ShfZ = torch.tensor(kwargs.ShfZ).to(self.device)
        EtaA = torch.tensor(kwargs.EtaA).to(self.device)
        ShfA = torch.tensor(kwargs.ShfA).to(self.device)
        self.aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, len(self.species_order))
        self.argsdict.update({'Rcr': Rcr, 'Rca': Rca, 'EtaR': EtaR, 'ShfR': ShfR, 'Zeta': Zeta, 'ShfZ': ShfZ, 'EtaA': EtaA, 'ShfA': ShfA, 'species_order': self.species_order})

    def NN_setup(self, **kwargs):
        import torch, torchani
        kwargs = hyperparameters(kwargs)
        if len(kwargs.neurons) == 1:
            self.neurons = [kwargs.neurons[0].copy() for _ in range(len(self.species_order))]

        self.networkdict = OrderedDict()
        for i, specie in enumerate(self.species_order):
            self.neurons[i] += [1]
            layers = [torch.nn.Linear(self.aev_computer.aev_length+1, self.neurons[i][0])]
            for j in range(len(self.neurons[i]) - 1):
                if type(kwargs.activation_function) == str:
                    act_fun = kwargs.activation_function
                    if '(' in act_fun:
                        xx = act_fun.split('(')
                        act_fun = xx[0]
                        alpha = float(xx[1].strip(')'))
                        layers += [torch.nn.__dict__[act_fun](alpha)]
                    else:  
                        layers += [torch.nn.__dict__[act_fun]()]
                elif callable(kwargs.activation_function):
                    layers += [kwargs.activation_function()]
                layers += [torch.nn.Linear(self.neurons[i][j], self.neurons[i][j + 1])]
            self.networkdict[specie] = torch.nn.Sequential(*layers)

        self.nn = torchani.ANIModel(self.networkdict)

        self.NN_initialize()
        self.optimizer_setup(**kwargs)  

    def NN_initialize(self, a: float = 1.0) -> None:
        '''
        Reset the network parameters using :meth:`torch.nn.init.kaiming_normal_`.

        Arguments:
            a(float): Check `torch.nn.init.kaiming_normal_() <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_>`_.
        '''
        import torch
        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_params)

    def optimizer_setup(self, **kwargs):
        import torch
        kwargs = hyperparameters(kwargs)
        if isinstance(self.networkdict, OrderedDict) or type(self.networkdict) == dict:
            wlist2d = [
                [
                    {'params': [self.networkdict[specie][j * 2].weight]} if j == 0 or j == len(self.neurons[i]) - 1 else {'params': [self.networkdict[specie][j*2].weight], 'weight_decay': 0.0001 / 10**j} for j in range(len(self.neurons[i]))
                ] for i, specie in enumerate(self.species_order)
            ]

            blist2d = [
                [
                    {'params': [self.networkdict[specie][j * 2].bias]} for j in range(len(self.neurons[i]))
                ] for i, specie in enumerate(self.species_order)
            ]
            self.AdamW = torch.optim.AdamW([i for j in wlist2d for i in j],lr=kwargs.learning_rate)
            self.SGD = torch.optim.SGD([i for j in blist2d for i in j], lr=kwargs.learning_rate)
        elif type(self.networkdict) == list:
            wlist3d =[[
                [
                    {'params': [subdict[specie][j * 2].weight]} if j == 0 or j == len(self.neurons[k][i]) - 1 else {'params': [subdict[specie][j*2].weight], 'weight_decay': 0.0001 / 10**j} for j in range(len(self.neurons[k][i]))
                ] for i, specie in enumerate(self.species_order)
            ] for k, subdict in enumerate(self.networkdict)]

            blist3d = [[
                [
                    {'params': [subdict[specie][j * 2].bias]} for j in range(len(self.neurons[k][i]))
                ] for i, specie in enumerate(self.species_order)
            ] for k, subdict in enumerate(self.networkdict)]

            self.AdamW = torch.optim.AdamW([i for k in wlist3d for j in k for i in j],lr=kwargs.learning_rate)
            self.SGD = torch.optim.SGD([i for k in blist3d for j in k  for i in j], lr=kwargs.learning_rate)
            
        self.AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.AdamW, factor=kwargs.lr_reduce_factor, patience=kwargs.lr_reduce_patience, threshold=kwargs.lr_reduce_threshold)
        self.SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.SGD, factor=kwargs.lr_reduce_factor, patience=kwargs.lr_reduce_patience, threshold=kwargs.lr_reduce_threshold)

    def model_setup(self, **kwargs):
        self.AEV_setup(**kwargs)
        self.NN_setup(**kwargs)
        from .torchani_heavy_parts import StateInputNet
        self.model = StateInputNet(self.aev_computer, self.nn).float().to(self.device)

    def fix_layers(self, layers_to_fix: Union[List[List[int]],List[int]]):
        '''
        Fix specific layers to be non-trainable for each element.

        Arguments:
            layers_to_fix (List): Should be: 
            
                - A list of integers. Layers indicate by the integers will be fixed
                - A list of lists of integers. Each sub-list defines the layers to be fixed for each species, in the order of `self.species_order`. 
        '''
        if layers_to_fix:
            if len(layers_to_fix) == 1:
                layers_to_fix = layers_to_fix * len(self.species_order)
            for name, parameter in self.model.named_parameters():
                indices = name.split('.')
                if int(indices[-2]) in layers_to_fix[self.species_order.index(indices[-3] if indices[-3] in data.element_symbol2atomic_number else data.atomic_number2element_symbol[int(indices[-3])])]:
                    parameter.requires_grad=False

    def data_setup(self, molecular_database, validation_molecular_database, spliting_ratio,
                   property_to_learn, xyz_derivative_property_to_learn, ):
        assert molecular_database, 'provide molecular database'

        self.property_name = property_to_learn
        
        data_element_symbols = list(np.sort(np.unique(np.concatenate(molecular_database.element_symbols))))

        if not self.species_order: 
            self.species_order = data_element_symbols
        else:
            for element in data_element_symbols:
                if element not in self.species_order:
                    print('element(s) outside supported species detected, please check the database')
                    return
                
        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]
        elif not validation_molecular_database:
            raise NotImplementedError("please specify validation_molecular_database or set it to 'sample_from_molecular_database'")
        molecular_database = unpackData2State(molecular_database)
        validation_molecular_database = unpackData2State(validation_molecular_database)
        if self.energy_shifter.self_energies is None:
            if np.isnan(molecular_database.get_properties(property_to_learn)).sum():
                print(property_to_learn)
                print(xyz_derivative_property_to_learn)
                molDB2ANIdata_state(molecular_database.filter_by_property(property_to_learn), property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order)
                self.subtraining_set = molDB2ANIdata_state(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()
            else:   
                self.subtraining_set = molDB2ANIdata_state(molecular_database, property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order).species_to_indices(self.species_order).cache()
        else:
            self.subtraining_set = molDB2ANIdata_state(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()
        self.validation_set = molDB2ANIdata_state(validation_molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()
        
        self.energy_shifter = self.energy_shifter.to(self.device)
        
        self.subtraining_set = self.subtraining_set.collate(self.hyperparameters.batch_size, PADDING)
        self.validation_set = self.validation_set.collate(self.hyperparameters.batch_size, PADDING)

        self.argsdict.update({'self_energies': self.energy_shifter.self_energies, 'property': self.property_name})
class ani_child(torchani_model):
    def __init__(self, parent, index, name='ani_child', device=None):
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.name = name
        self.model = parent.__getitem__(index)

    def predict(
            self, 
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            calculate_energy: bool = False,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
            batch_size: int = 2**16,
        ) -> None:
        import torch, torchani
        
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        if not molDB._is_uniform_cell():
            batch_size = 1

        for batch in molDB.batches(batch_size):
            for properties in molDB2ANIdata(batch).species_to_indices('periodic_table').collate(batch_size, PADDING):
                species = properties['species'].to(self.device)
                if torchani.utils.PERIODIC_TABLE[0] == 'H':
                    species += 1
                xyz_coordinates = properties['coordinates'].float().to(self.device)
                break
            pbc = properties['pbc'][0].to(self.device) if 'pbc' in properties else None
            cell = properties['cell'][0].float().to(self.device) if 'cell' in properties else None
            if pbc is not None and cell is not None:
                xyz_coordinates = torchani.utils.map2central(cell, xyz_coordinates, pbc)
            xyz_coordinates = xyz_coordinates.requires_grad_(calculate_energy_gradients or calculate_hessian)
            
            ANI_NN_energies = self.model((species, xyz_coordinates), pbc=pbc, cell=cell).energies
            if calculate_energy: 
                batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), 'energy')
            if calculate_energy_gradients or calculate_hessian:
                ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energies.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                if calculate_energy_gradients:
                    grads = ANI_NN_energy_gradients.detach().cpu().numpy()
                    batch.add_xyz_vectorial_properties(grads, 'energy_gradients')
                if calculate_hessian:
                    ANI_NN_hessians = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energies)
                    batch.add_hessian_properties(ANI_NN_hessians.detach().cpu().numpy(), 'hessian')
    
    def node(self,):
        return model_tree_node(name=self.name, operator='predict', model=self)
    
class ani_methods(torchani_model, method_model, downloadable_model):
    '''
    Create a model object with one of the ANI methods

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'ANI-1x'``, ``'ANI-1ccx'``, ``'ANI-2x'``, ``'ANI-1ccx-gelu'`` or ``'ANI-1x-gelu'``.
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.

    '''
    
    supported_methods = ["ANI-1x", "ANI-1ccx", "ANI-2x", 'ANI-1x-D4', 'ANI-2x-D4', 'ANI-1xnr', 'ANI-1ccx-gelu', 'ANI-1ccx-gelu-D4', 'ANI-1x-gelu', 'ANI-1x-gelu-d4']
    # no atomic energies for ANI-1ccx-gelu thus calculating energy on single atom will get error
    atomic_energies = {'ANI-1ccx': {1:-0.50088088, 6:-37.79199048, 7:-54.53379230, 8:-75.00968205}}

    def __init__(self, method: str = 'ANI-1ccx', model_index=None, device = None):
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.method = method
        self.model_index = model_index
        self.model_setup(method)

    def model_setup(self, method):
        import torchani
        from ..models import methods
        self.method = method
        if 'ANI-1xnr'.casefold() in method.casefold():
            self.model = load_ani1xnr_model().to(self.device)
        elif 'ANI-1ccx-gelu'.casefold() in method.casefold():
            self.model = self.load_ani_gelu_model('ani_1ccx_gelu',model_index=self.model_index)
        elif 'ANI-1x-gelu'.casefold() in method.casefold():
            self.model = self.load_ani_gelu_model('ani_1x_gelu',model_index=self.model_index)
        elif 'ANI-1x'.casefold() in method.casefold():
            self.model = torchani.models.ANI1x(periodic_table_index=True).to(self.device).double()
        elif 'ANI-1ccx'.casefold() in method.casefold():
            self.model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device).double()
        elif 'ANI-2x'.casefold() in method.casefold():
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device).double()
        else:
            print("method not found, please check ANI_methods().supported_methods")

        modelname = method.lower().replace('-','')
        if 'gelu'.casefold() in method.casefold():
            self.element_symbols_available = self.model[0].species_order
            self.children = [model_tree_node(name=f'{modelname}_nn{index}', model=self.model[index], operator='predict') for index in range(len(self.model))]
        else:
            self.element_symbols_available = self.model.species
            if self.model_index == None:
                self.children = [ani_child(self.model, index, name=f'{modelname}_nn{index}', device=self.device).node() for index in range(len(self.model))]
            else:
                self.children = [ani_child(self.model, self.model_index, name=f'{modelname}_nn{self.model_index}', device=self.device).node()]
        if 'D4'.casefold() in self.method.casefold():
            d4 = model_tree_node(name='d4_wb97x', operator='predict', model=methods(method='D4', functional='wb97x'))
            ani_nns = model_tree_node(name=f'{modelname}_nn', children=self.children, operator='average')
            self.model = model_tree_node(name=modelname, children=[ani_nns, d4], operator='sum')
        else:
            self.model = model_tree_node(name=modelname, children=self.children, operator='average')

    def load_ani_gelu_model(self, method, model_index=None):
        
        model_name, model_path, download = self.check_model_path(method)
        if download: self.download(model_name, model_path)
        
        if not model_index:
            model_ensemble = []
            for ii in range(8):
                # if not os.path.exists(f'{dirname}/{method}_cv{ii}.pt'):
                #     raise ValueError(f'Please put {method}_cv{ii}.pt file under $MODELSPATH/{method}_model/')
                model_ensemble.append(ani(model_file=f'{model_path}/cv{ii}.pt', verbose=0))
            return model_ensemble
        else:
            # if not os.path.exists(f'{dirname}/{method}_cv{model_index}.pt'):
            #     raise ValueError(f'Please put {method}_cv{model_index}.pt under $MODELSPATH/{method}_model/')
            return [ani(model_file=f'{model_path}/cv{model_index}.pt', verbose=0)]
            
    @doc_inherit
    def predict(
            self, 
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            calculate_energy: bool = True,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
            batch_size: int = 2**16,
        ) -> None:
        '''
            batch_size (int, optional): The batch size for batch-predictions.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for element_symbol in np.unique(np.concatenate(molDB.element_symbols)):
            if element_symbol not in self.element_symbols_available:
                print(f' * Warning * Molecule contains elements \'{element_symbol}\', which is not supported by method \'{self.method}\' that only supports {self.element_symbols_available}, no calculations performed')
                return
            
        monoatomic_idx = molDB.number_of_atoms == 1

        for mol in molDB[monoatomic_idx]:
            self.predict_monoatomic(molecule=mol,
                                    calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)
            
        self.model.predict(molecular_database=molDB[~monoatomic_idx],
                           calculate_energy=calculate_energy,
                           calculate_energy_gradients=calculate_energy_gradients, 
                           calculate_hessian=calculate_hessian,
                           batch_size=batch_size)
        
        for mol in molDB[~monoatomic_idx]:
            self.get_SD(molecule=mol,
                        calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)

    def predict_monoatomic(self, molecule, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        if calculate_energy:
            molecule.energy = self.atomic_energies[self.method][molecule.atoms[0].atomic_number]
        if calculate_energy_gradients:
            molecule.add_xyz_vectorial_property(np.zeros_like(molecule.xyz_coordinates), 'energy_gradients')
        if calculate_hessian:
            molecule.hessian = np.zeros((len(molecule),)*2)
        
    def get_SD(self, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        properties = [] ; atomic_properties = []
        if calculate_energy: properties.append('energy')
        if calculate_energy_gradients: atomic_properties.append('energy_gradients')
        if calculate_hessian: properties.append('hessian')
        if 'D4'.casefold() in self.method.casefold():
            modelname = self.method.lower().replace('-','')
            modelname = f'{modelname}_nn'
        else:
            modelname = self.method.lower().replace('-','')
        molecule.__dict__[f'{modelname}'].standard_deviation(properties=properties+atomic_properties)
    
    def train(self, **kwargs):
        import torchani
        
        # default settings
        kwargs['save_model'] = True # force saving model
        if 'file_to_save_model' in kwargs:
            file_to_save_model = kwargs['file_to_save_model']
        else:
            file_to_save_model = None
        if 'reset_energy_shifter' not in kwargs:
            kwargs['reset_energy_shifter'] = True
        if 'verbose' not in kwargs:
            verbose = 0
        else:
            verbose = kwargs['verbose']

        if 'hyperparameters' in kwargs:
            _hyperparameters = kwargs['hyperparameters']
        else:
            _hyperparameters = {}
        # check hyperparameters
        if 'fix_layers' not in _hyperparameters:
            _hyperparameters['fixed_layers'] = [[0,4]]
        if 'loss_type' not in _hyperparameters:
            _hyperparameters['loss_type'] = 'geometric'
        if 'max_epochs' not in _hyperparameters:
            _hyperparameters['max_epochs'] = 100

        kwargs['hyperparameters'] = _hyperparameters
        
        pretrained_models = []
        if 'ANI-1xnr'.casefold() in self.method.casefold():
            raise ValueError('Currently ANI-1xnr can not be retrained on')
        elif 'gelu'.casefold() in self.method.casefold():
            if 'ANI-1ccx-gelu'.casefold() in self.method.casefold():
                pretrained_models = self.load_ani_gelu_model(method='ani_1ccx_gelu', model_index=self.model_index)
            elif 'ANI-1x-gelu'.casefold() in self.method.casefold():
                pretrained_models = self.load_ani_gelu_model(method='ani_1x_gelu', model_index=self.model_index)
        else:
            animodel = ani()
            if 'ANI-1ccx'.casefold() in self.method.casefold():
                animodel.load_ani_model('ANI-1ccx')
            elif 'ANI-1x'.casefold() in self.method.casefold():
                animodel.load_ani_model('ANI-1x')
            elif 'ANI-2x'.casefold() in self.method.casefold():
                animodel.load_ani_model('ANI-2x')
            else:
                raise ValueError('Not supported pretrained model type')

            if self.model_index == None:
                for ii in range(8):
                    pmodel = ani(verbose=verbose)
                    pmodel.species_order = animodel.species_order
                    pmodel.aev_computer = animodel.aev_computer
                    pmodel.networkdict = animodel.networkdict[ii]
                    pmodel.neurons = animodel.neurons[ii]
                    pmodel.nn = animodel.nn[ii]
                    pmodel.energy_shifter = animodel.energy_shifter
                    pmodel.optimizer_setup(**pmodel.hyperparameters)
                    pmodel.model = torchani.nn.Sequential(
                        animodel.aev_computer, animodel.nn[ii]).to(pmodel.device).float()
                    pretrained_models.append(pmodel)
            else:
                pmodel = ani(verbose=verbose)
                pmodel.species_order = animodel.species_order
                pmodel.aev_computer = animodel.aev_computer
                pmodel.networkdict = animodel.networkdict[self.model_index]
                pmodel.neurons = animodel.neurons[self.model_index]
                pmodel.nn = animodel.nn[self.model_index]
                pmodel.energy_shifter = animodel.energy_shifter
                pmodel.optimizer_setup(**pmodel.hyperparameters)

                pmodel.model = torchani.nn.Sequential(
                    animodel.aev_computer, animodel.nn).to(pmodel.device).float()
                pretrained_models.append(pmodel)

        retrained_models = []
        modelname = self.method.lower().replace('-','')

        if isinstance(self.model_index, int):
            print(f'\nStart retraining on model {self.model_index}...')
            if file_to_save_model:
                kwargs['file_to_save_model'] = file_to_save_model + f'.cv{self.model_index}'
            else:
                kwargs['file_to_save_model'] = f'{modelname}_retrained.pt.cv{self.model_index}'
            pretrained_models[0].train(**kwargs)
            retrained_models.append(pretrained_models[0])
        else:
            for ii, pmodel in enumerate(pretrained_models):
                print(f'\nStart retraining on model {ii}...')
                sys.stdout.flush()
                if file_to_save_model:
                    kwargs['file_to_save_model'] = file_to_save_model + f'.cv{ii}'
                else:
                    kwargs['file_to_save_model'] = f'{modelname}_retrained.pt.cv{ii}'
                pmodel.train(**kwargs)
                retrained_models.append(pmodel)

        children = [
            model_tree_node(
                name=f'{modelname}_nn{ii}', 
                model=rmodel, 
                operator='predict') for ii, rmodel in enumerate(retrained_models)]
        self.model = model_tree_node(
            name=modelname, 
            children=children, 
            operator='average')

def load_ani1xnr_model():
    # ANI-1xnr https://github.com/atomistic-ml/ani-1xnr/
    # Universal reactive ML potential ANI-1xnr: https://doi.org/10.1038/s41557-023-01427-3
    import torchani
    species = ['H', 'C', 'N', 'O']
    def parse_ani1xnr_resources():
        import requests
        import zipfile
        import io
        if os.path.exists('/export/home/xacscloud/ANI/ANI1xnr/'):
            local_dir = '/export/home/xacscloud/ANI/ANI1xnr/'
        else:
            local_dir = os.path.expanduser('~/.local/ANI1xnr/')
        url = "https://github.com/atomistic-ml/ani-1xnr/archive/refs/heads/main.zip"
        if not os.path.exists(local_dir+'ani-1xnr-main'):
            os.makedirs(local_dir, exist_ok=True)
            print(f'Downloading ANI-1xnr model parameters ...')
            resource_res = requests.get(url)
            resource_zip = zipfile.ZipFile(io.BytesIO(resource_res.content))
            resource_zip.extractall(local_dir)
        return local_dir
    model_prefix = parse_ani1xnr_resources() + 'ani-1xnr-main/model/ani-1xnr/'
    # const_file, sae_file, ensemble_prefix, ensemble_size = parse_ani1xnr_resources()
    const_file = model_prefix + 'rHCNO-5.2R_32-3.5A_a8-4.params'
    sae_file = model_prefix + 'sae_linfit.dat'
    consts = torchani.neurochem.Constants(const_file)
    aev_computer = torchani.AEVComputer(**consts)
    energy_shifter = torchani.neurochem.load_sae(sae_file)
    species_converter = torchani.nn.SpeciesConverter(species)

    neural_networks = []
    for ii in range(8):
        neural_network = torchani.neurochem.load_model(consts.species, model_prefix + f'train{ii}/networks')
        neural_networks.append(torchani.nn.Sequential(species_converter, aev_computer, neural_network, energy_shifter))

    model_ensemble = torchani.nn.Ensemble(modules=neural_networks)
    model_ensemble.species = species

    return model_ensemble


def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
  To use Interface_ANI, please install TorchANI and its dependencies

  Arguments with their default values:
    MLprog=TorchANI            enables this interface
    MLmodelType=ANI            requests ANI model

    ani.batch_size=8           batch size
    ani.max_epochs=10000000    max epochs
    ani.learning_rate=0.001    learning rate used in the Adam and SGD optimizers
    
    ani.early_stopping_learning_rate=0.00001
                               learning rate that triggers early-stopping
    
    ani.force_coefficient=0.1  weight for force
    ani.Rcr=5.2                radial cutoff radius
    ani.Rca=3.5                angular cutoff radius
    ani.EtaR=1.6               radial smoothness in radial part
    
    ani.ShfR=0.9,1.16875,      radial shifts in radial part
      1.4375,1.70625,1.975,
      2.24375,2.5125,2.78125,
      3.05,3.31875,3.5875,
      3.85625,4.125,4.9375,
      4.6625,4.93125
    
    ani.Zeta=32                angular smoothness
    
    ani.ShfZ=0.19634954,       angular shifts
      0.58904862,0.9817477,
      1.3744468,1.7671459,
      2.1598449,2.552544,
      2.9452431
    
    ani.EtaA=8                 radial smoothness in angular part
    ani.ShfA=0.9,1.55,2.2,2.85 radial shifts in angular part

  Cite TorchANI:
    X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, A. E. Roitberg,
    J. Chem. Inf. Model. 2020, 60, 3408
    
  Cite ANI model:
    J. S. Smith, O. Isayev, A. E. Roitberg, Chem. Sci. 2017, 8, 3192
'''
    print(helpText)
