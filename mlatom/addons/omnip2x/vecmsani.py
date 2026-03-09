'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! A model behind OMNI-P2x                                                   !
  ! Implemented by: Mikolaj Martyka                                           !                                             
  !---------------------------------------------------------------------------! 
'''

from typing import Any, Union, Dict, List, Callable
from types import FunctionType
import os, sys, uuid
import numpy as np
import tqdm
from collections import OrderedDict

from ... import data, model_cls
from ...model_cls import ml_model, torchani_model, method_model, hyperparameters, hyperparameter, model_tree_node
from ...decorators import doc_inherit

def calculate_re_descriptor(eqmol, mol):
    natoms = len(eqmol.atoms)  
    eq_distmat = eqmol.get_internuclear_distance_matrix()
    descriptor = np.zeros(int(natoms*(natoms-1)/2))
    distmat = mol.get_internuclear_distance_matrix()
    ii = -1
  
    for iatomind in range(natoms):
        for jatomind in range(iatomind+1,natoms):
            ii += 1
            descriptor[ii] = eq_distmat[iatomind][jatomind]/distmat[iatomind][jatomind]
    mol.__dict__["RE"] = descriptor
    mol.descriptor = descriptor
    
def median(yp,yt):
    import torch
    return torch.median(torch.abs(yp-yt))

def molDB2ANIdata_state(molDB, 
                  property_to_learn=None,
                  xyz_derivative_property_to_learn=None,
                 use_state=True):
    from torchani.data import TransformableIterable, IterableAdapter
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
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))

def molDB2ANIdata_state_Re(molDB, 
                  property_to_learn=None,
                  xyz_derivative_property_to_learn=None,
                 use_state=True):
    from torchani.data import TransformableIterable, IterableAdapter
    def molDBiter():
        for mol in molDB.molecules:
            ret = {'species': mol.get_element_symbols(), 'coordinates': mol.xyz_coordinates}
            ret["descriptor"] = mol.descriptor
            if property_to_learn is not None:
                ret['energies'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn is not None:
                ret['forces'] = -1 * mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            if use_state:
                ret['state']=mol.__dict__["current_state"]
            if False: #debug
                ret['id']=mol.__dict__["mol_id"]
            yield ret
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))
    
def unpackData2State_RE(mol_db, property_to_learn=None,
                  xyz_derivative_property_to_learn=None, eqmol=None, descriptor_dim=2):
    train_data = data.molecular_database()
    for mol_id, i in enumerate(mol_db):
        try:
            dsc = i.descriptor
        except:
            dsc = [0] * descriptor_dim
            dsc[-1] = 1
        i.descriptor = np.array(dsc)
        for idx, state in enumerate(i.electronic_states):
            new_mol = data.molecule()
            new_mol.read_from_xyz_string(i.get_xyz_string())
            new_mol.current_state = idx
            new_mol.descriptor = i.descriptor
            cmd = 'new_mol.{} = state.{}'.format(property_to_learn,property_to_learn)
            exec(cmd)
            cmd2 = "new_mol.add_xyz_derivative_property(state.get_xyz_vectorial_properties(\"{}\"), \'{}\', \'{}\' )".format(xyz_derivative_property_to_learn, property_to_learn, xyz_derivative_property_to_learn)
            #new_mol.add_xyz_derivative_property(state.get_xyz_vectorial_properties("energy_gradients"), 'energy', 'energy_gradients' )
            if xyz_derivative_property_to_learn:
                exec(cmd2)
            new_mol.mol_id = mol_id
            train_data.append(new_mol)
    return train_data

#The two MOLDB2ANIdata should be merged, i just didnt have the time to do it.
def molDB2ANIdata(molDB, 
                  property_to_learn=None,
                  xyz_derivative_property_to_learn=None):
    from torchani.data import TransformableIterable, IterableAdapter
    def molDBiter():
        for mol in molDB.molecules:
            ret = {'species': mol.get_element_symbols(), 'coordinates': mol.xyz_coordinates}
            if property_to_learn is not None:
                ret['energies'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn is not None:
                ret['forces'] = -1 * mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            yield ret
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))
    
class vecmsani(ml_model, torchani_model):
    '''
    Create an `MS-ANI', an extension of the ANI NN model for multi-state learning. First described in <10.26434/chemrxiv-2024-dtc1w>. 
    
    Interfaces to `TorchANI <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        device (str, optional): Indicate which device the calculation will be run on. i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
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

    def __init__(self, model_file: str = None, device: str = None, hyperparameters: Union[Dict[str,Any], model_cls.hyperparameters]={}, verbose=1, nstates=1,validate_train=True, descriptor_dim=2):
        import torch, torchani
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        self.nstates = nstates
        self.descriptor_dim = descriptor_dim

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
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
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
        
        from .re_state_input_net import ReStateInputNet
        self.model = ReStateInputNet(self.aev_computer, self.nn).to(self.device)
        self.model.eval()
        if self.verbose: print(f'model loaded from {model_file}')
    
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
        reset_energy_shifter = False,
        reset_optimizer: bool = False,
        save_every_epoch: bool = False,
        energy_weighting_function: Callable = None,
        energy_weighting_function_kwargs: dict = {},
    ) -> None:
        r'''
            validation_molecular_database (:class:`mlatom.data.molecular_database` | str, optional): Explicitly defines the database for validation, or use ``'sample_from_molecular_database'`` to make it sampled from the training set.
            hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
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
        if reset_energy_shifter:
            import torch, torchani
            if self.energy_shifter:
                self.energy_shifter_ = self.energy_shifter
            if type(reset_energy_shifter) == list:
                self.energy_shifter = torchani.utils.EnergyShifter(reset_energy_shifter)
            elif isinstance(reset_energy_shifter, torchani.utils.EnergyShifter):
                self.energy_shifter = reset_energy_shifter
            else:
                self.energy_shifter = torchani.utils.EnergyShifter(None)
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
    
        
        def validate_training():
            total_mse = 0.0
            count = 0
            mae_sum = torch.nn.L1Loss(reduction='sum'); energy_mae = 0.0
            for properties in self.validation_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                state = properties['state'].to(self.device).float()
                descriptor = properties['descriptor'].to(self.device).float()

                true_gap_list = []
                for i in range(int(len(true_energies)/self.nstates)):
                    for j in range(1,self.nstates):
                        true_gap_list.append(abs(true_energies[i*self.nstates+j]-true_energies[i*self.nstates+j-1]))
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
                    _, predicted_energies = self.model(species, coordinates,state, descriptor)

                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i*self.nstates+j]-predicted_energies[i*self.nstates+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    # true_energies[true_energies.isnan()]=predicted_energies[true_energies.isnan()]
                    if self.hyperparameters.median_loss:
                        energy_loss= median(predicted_energies,true_energies)
                    else:
                        energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    #auxnumber = int(len(predicted_energies)/self.nstates)
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1)).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    loss = energy_loss + self.hyperparameters.force_coefficient * force_loss + self.hyperparameters.gap_coefficient * gap_loss
                    total_mse += loss.item()
                    energy_mae += mae_sum(predicted_energies*weightings_e, true_energies*weightings_e).item()
                    count += predicted_energies.shape[0]
                else:
                    coordinates = properties['coordinates'].to(self.device).float()

                    _, predicted_energies = self.model(species, coordinates, state,descriptor)
                    
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i*self.nstates+j]-predicted_energies[i*self.nstates+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    
                    energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    #auxnumber = int(len(predicted_energies)/self.nstates)
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1)).nanmean()
                    else:
                        gap_loss = 0
                    loss = energy_loss + self.hyperparameters.gap_coefficient * gap_loss
                    total_mse += loss.item()
                    energy_mae += mae_sum(predicted_energies*weightings_e, true_energies*weightings_e).item()
                    count += predicted_energies.shape[0]
                
            return total_mse/count, energy_mae/count
        
        def loss_function(prediction, reference, weightings=1, reduction='none'):
            return torch.nn.functional.mse_loss(prediction*weightings, reference*weightings, reduction=reduction)

        if self.verbose: print("training starting from epoch", self.AdamW_scheduler.last_epoch + 1)
        for _ in range(self.AdamW_scheduler.last_epoch + 1, self.hyperparameters.max_epochs + 1):
            rmse, mae = validate_training()
            if self.verbose: print('validation MSE:', rmse, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
            if self.verbose: print('Energy MAE:', mae, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
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
                        true_gap_list.append(abs(true_energies[i*self.nstates+j]-true_energies[i*self.nstates+j-1]))
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
                    descriptor = properties['descriptor'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates,state,descriptor)
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i*self.nstates+j]-predicted_energies[i*self.nstates+j-1]))
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
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1)).nanmean()
                    else:
                        gap_loss = 0
                    # true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    loss = energy_loss + self.hyperparameters.force_coefficient * force_loss + self.hyperparameters.gap_coefficient * gap_loss
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    descriptor = properties['descriptor'].to(self.device).float()
                    _, predicted_energies = self.model(species, coordinates, state,descriptor)
                    predicted_gap_list = []

                    for i in range(int(len(predicted_energies)/self.nstates)):
                        for j in range(1,self.nstates):
                            predicted_gap_list.append(abs(predicted_energies[i*self.nstates+j]-predicted_energies[i*self.nstates+j-1]))
                    predicted_gaps = torch.FloatTensor(predicted_gap_list).to(self.device)
                    #print(predicted_gaps)

                    energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    if self.hyperparameters.gap_coefficient != 0.0:
                        gap_loss = (loss_function(predicted_gaps,true_gaps, 1)).nanmean()
                    else:
                        gap_loss = 0
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
            def_descriptor = None,
            calculate_energy: bool = False,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
            property_to_predict: Union[str, None] = 'estimated_y', 
            xyz_derivative_property_to_predict: Union[str, None] = None, 
            hessian_to_predict: Union[str, None] = None, 
            batch_size: int = 2**16,
        ) -> None:
        if def_descriptor is None:
            def_descriptor = [0] * self.descriptor_dim
            def_descriptor[-1] = 1
        '''
            batch_size (int, optional): The batch size for batch-predictions.
        '''
        import torch, torchani

        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)


        for mol in molDB:
            try:
                mol.descriptor = mol.descriptor
            except:
                mol.descriptor = def_descriptor
        for batch in molDB.batches(batch_size):
            state_energies =[]
            state_gradients =[]
            state_hessians = []
            PADDING = {
            'species': -1,
            'coordinates': 0.0,
            'forces': 0.0,
            'energies': 0.0,
            'descriptor':0.0,
            }

            for i in range(nstates):
                for properties in molDB2ANIdata_state_Re(batch, use_state=False).species_to_indices(self.species_order).collate(batch_size,padding=PADDING):
                    species = properties['species'].to(self.device)
                    descriptor = (properties['descriptor']).to(self.device).float()
                    state = torch.full((len(batch),),i).to(self.device)
                    xyz_coordinates = properties['coordinates'].float().to(self.device).requires_grad_(bool(xyz_derivative_property_to_predict or hessian_to_predict))
                    break
                ANI_NN_energies = self.energy_shifter(self.model(species, xyz_coordinates, state, descriptor)).energies
                if property_to_predict:
                    if nstates>1:
                        batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), property_to_predict+"_state"+str(i))
                        state_energies.append(ANI_NN_energies.detach().cpu().numpy())
                    else:
                        batch.add_scalar_properties(ANI_NN_energies.detach().cpu().numpy(), property_to_predict)
                        state_energies.append(ANI_NN_energies.detach().cpu().numpy())                        
                if xyz_derivative_property_to_predict or hessian_to_predict:
                    ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energies.sum(), xyz_coordinates, create_graph=False, retain_graph=False)[0]
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
        descriptor_len = self.descriptor_dim
        kwargs = hyperparameters(kwargs)
        if len(kwargs.neurons) == 1:
            self.neurons = [kwargs.neurons[0].copy() for _ in range(len(self.species_order))]

        self.networkdict = OrderedDict()
        for i, specie in enumerate(self.species_order):
            self.neurons[i] += [1]
            layers = [torch.nn.Linear(self.aev_computer.aev_length+1+descriptor_len, self.neurons[i][0])]
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
        from .re_state_input_net import ReStateInputNet
        self.model = ReStateInputNet(self.aev_computer, self.nn).float().to(self.device)

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
                   property_to_learn, xyz_derivative_property_to_learn):
        assert molecular_database, 'provide molecular database'
        import torch
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
        molecular_database = unpackData2State_RE(molecular_database, property_to_learn=property_to_learn,
                  xyz_derivative_property_to_learn=xyz_derivative_property_to_learn, descriptor_dim=self.descriptor_dim)
        validation_molecular_database = unpackData2State_RE(validation_molecular_database, property_to_learn=property_to_learn,
                  xyz_derivative_property_to_learn=xyz_derivative_property_to_learn, descriptor_dim=self.descriptor_dim)
        if self.energy_shifter.self_energies is None:
            if np.isnan(molecular_database.get_properties(property_to_learn)).sum():
                molDB2ANIdata_state_Re(molecular_database.filter_by_property(property_to_learn), property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order)
                self.subtraining_set = molDB2ANIdata_state_Re(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()
            else:   
                self.subtraining_set = molDB2ANIdata_state_Re(molecular_database, property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order).species_to_indices(self.species_order).cache()
        else:
            self.subtraining_set = molDB2ANIdata_state_Re(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()

        if len(self.species_order) != len(self.energy_shifter.self_energies):
            true_species_order = sorted(data_element_symbols, key=lambda x: self.species_order.index(x))
            expanded_self_energies = np.zeros((len(self.species_order)))
            for ii, sp in enumerate(self.species_order):
                if sp in true_species_order:
                    expanded_self_energies[ii] = self.energy_shifter.self_energies[true_species_order.index(sp)]
                elif 'energy_shifter_' in self.__dict__:
                    expanded_self_energies[ii] = self.energy_shifter_.self_energies[ii]
            self.energy_shifter.self_energies = torch.tensor(expanded_self_energies)
        self.energy_shifter = self.energy_shifter.to(self.device)    
        self.validation_set = molDB2ANIdata_state_Re(validation_molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).cache()
        
        self.energy_shifter = self.energy_shifter.to(self.device)
        PADDING = {
        'species': -1,
        'coordinates': 0.0,
        'forces': 0.0,
        'energies': 0.0,
        'descriptor':0.0,
        }

        self.subtraining_set = self.subtraining_set.collate(self.hyperparameters.batch_size, padding=PADDING)
        self.validation_set = self.validation_set.collate(self.hyperparameters.batch_size, padding=PADDING)

        self.argsdict.update({'self_energies': self.energy_shifter.self_energies, 'property': self.property_name})