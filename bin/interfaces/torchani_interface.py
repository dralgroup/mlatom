import os, sys, uuid, math
import numpy as np
import tqdm
from collections import OrderedDict
import torch
import torchani
from torchani.utils import ChemicalSymbolsToInts
from torchani.data import TransformableIterable, IterableAdapter

pythonpackage = True
try:
    from .. import constants
    from .. import data
    from .. import models
    from .. import stopper
except:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import constants
    import data
    import models
    import stopper
    pythonpackage = False

def median(yp,yt):
        return torch.median(torch.abs(yp-yt))

def molDB2ANIdata(molDB, 
                  property_to_learn='energy',
                  xyz_derivative_property_to_learn = None):
    def molDBiter():
        for mol in molDB.molecules:
            ret = {'species': mol.get_element_symbols(), 'coordinates': mol.get_xyz_coordinates()}
            ret['energies'] = mol.__dict__[property_to_learn]
            if xyz_derivative_property_to_learn is not None:
                ret['forces'] = -1 * mol.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            yield ret
    return TransformableIterable(IterableAdapter(lambda: molDBiter()))

class ani(models.ml_model):
    hyperparameters = {
        #### Training ####
        'batchSize':            8,
        'maxEpochs':            1000000,
        'initLR':               0.001,
        'earlyStopLR':          1.0E-5,
        'lrReducePatience':     100,
        'lrReduceFactor':       0.5,
        'lrReduceThreshold':    0,
        #### Loss ####
        'fCoef':                0.1,
        'medianLoss':           False,
        #### Network ####
        "neurons":              [[160, 128, 96]],
        # "actFun":               [["CELU", "CELU", "CELU"]],
        "fixedLayers":           None,
        #### AEV ####
        'Rcr':                  5.2000e+00,
        'Rca':                  3.5000e+00,
        'EtaR':                 torch.tensor([1.6000000e+01]),
        'ShfR':                 torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00]),
        'Zeta':                 torch.tensor([3.2000000e+01]),
        'ShfZ':                 torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00]),
        'EtaA':                 torch.tensor([8.0000000e+00]),
        'ShfA':                 torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00])
    }

    hyperparameters_min = {
        'batchSize':            1,
        'maxEpochs':            100,
        'initLR':               0.0001,
        'earlyStopLR':          1.0E-6,
        'lrReducePatience':     16,
        'lrReduceFactor':       0.2,
        'lrReduceThreshold':    0,
        }
    hyperparameters_max = {
        'batchSize':            1024,
        'maxEpochs':            1000000,
        'initLR':               0.01,
        'earlyStopLR':          1.0E-4,
        'lrReducePatience':     128,
        'lrReduceFactor':       0.9,
        'lrReduceThreshold':    0.001,
        }
    
    argsdict = {}
    model_file = None
    model = None
    property_name = 'y'
    species_order = []
    meta_data = {}
    shutup = False

    def __init__(self, model_file=None, device=None, hyperparameters={}, shutup=False):
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.hyperparameters.update(hyperparameters)
        self.shutup = shutup
        self.energy_shifter = torchani.utils.EnergyShifter(None)
        if model_file: 
            if os.path.isfile(model_file):
                self.load(model_file)
            else:
                if not self.shutup: print(f'the trained ANI model will be saved in {model_file}')
                self.model_file = model_file

    def load(self, model_file=None, species_order=None, AEV_parameters=None, self_energies=None, reset_parameters=False, method=None):
        if method:
            self.load_ani_model(method)
            return
        
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))

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

        if 'AEV_computer' in model_dict:
            self.aev_computer = model_dict['AEV_computer']
        elif 'Rcr' in model_dict['args']:
            self.AEV_setup(model_dict['args'])
        else:
            print('AEV parameters not found, please provide explictly')
            self.AEV_setup(AEV_parameters)
        
        if 'energy_shifter' in model_dict:
            self.energy_shifter = model_dict['energy_shifter']
        elif 'self_energies' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies'])
        elif 'energy_shifter_train' in model_dict['args']:
            self.energy_shifter = model_dict['args']['energy_shifter_train']
        elif 'self_energies_train' in model_dict['args']:
            self.energy_shifter = torchani.utils.EnergyShifter(model_dict['args']['self_energies'])
        else:
            print('self energy information not found, please provide explictly')
            self.energy_shifter = torchani.utils.EnergyShifter(self_energies)
        self.energy_shifter.to(self.device)

        if 'network' in model_dict and 'nn' in model_dict:
            self.networkdict = model_dict['network']
            if isinstance(self.networkdict, OrderedDict):
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
        self.model_file = model_file
        if not self.shutup: print(f'model loaded from {model_file}')

    def load_ani_model(self, method, **hyperparameters):
        self.hyperparameters.update(hyperparameters)
        if 'ANI-1x'.casefold() in method.casefold():
            model = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        elif 'ANI-1ccx'.casefold() in method.casefold():
            model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device)
        elif 'ANI-2x'.casefold() in method.casefold():
            model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        else:
            print("method not found, please check ANI_methods().available_methods")

        self.species_order = model.species
        self.argsdict.update({'species_order': self.species_order})
        self.aev_computer = model.aev_computer
        self.networkdict = [OrderedDict(**{k: v for k, v in nn.items()}) for nn in model.neural_networks]
        self.neurons = [[[layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)] for network in subdict.values()] for subdict in self.networkdict]
        self.nn = model.neural_networks
        self.optimizer_setup(**self.hyperparameters)
        self.energy_shifter = model.energy_shifter
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).float()

    def AEV_setup(self, **kwargs):
        Rcr = kwargs['Rcr']
        Rca = kwargs['Rca']
        EtaR = kwargs['EtaR'].to(self.device)
        ShfR = kwargs['ShfR'].to(self.device)
        Zeta = kwargs['Zeta'].to(self.device)
        ShfZ = kwargs['ShfZ'].to(self.device)
        EtaA = kwargs['EtaA'].to(self.device)
        ShfA = kwargs['ShfA'].to(self.device)
        self.aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, len(self.species_order))
        self.argsdict.update({'Rcr': Rcr, 'Rca': Rca, 'EtaR': EtaR, 'ShfR': ShfR, 'Zeta': Zeta, 'ShfZ': ShfZ, 'EtaA': EtaA, 'ShfA': ShfA,'species_order': self.species_order})

    def NN_setup(self, **kwargs):
        if len(kwargs['neurons']) == 1:
            self.neurons = [kwargs['neurons'][0].copy() for _ in range(len(self.species_order))]

        # if len(kwargs['actFun']) == 1:
        #     kwargs['actFun'] *= len(self.species_order)

        self.networkdict = OrderedDict()
        for i, specie in enumerate(self.species_order):
            self.neurons[i] += [1]
            layers = [torch.nn.Linear(self.aev_computer.aev_length, self.neurons[i][0])]
            for j in range(len(self.neurons[i]) - 1):
                layers += [torch.nn.CELU(0.1)]
                layers += [torch.nn.Linear(self.neurons[i][j], self.neurons[i][j + 1])]
            self.networkdict[specie] = torch.nn.Sequential(*layers)

        self.nn = torchani.ANIModel(self.networkdict)

        self.NN_initialize()
        self.optimizer_setup(**kwargs)  

    def NN_initialize(self, a=1.0):
        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a)
                torch.nn.init.zeros_(m.bias)

        self.nn.apply(init_params)

    def optimizer_setup(self, **kwargs):
        if isinstance(self.networkdict, OrderedDict):
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
            self.AdamW = torch.optim.AdamW([i for j in wlist2d for i in j],lr=kwargs['initLR'])
            self.SGD = torch.optim.SGD([i for j in blist2d for i in j], lr=kwargs['initLR'])
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

            self.AdamW = torch.optim.AdamW([i for k in wlist3d for j in k for i in j],lr=kwargs['initLR'])
            self.SGD = torch.optim.SGD([i for k in blist3d for j in k  for i in j], lr=kwargs['initLR'])
            
        self.AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.AdamW, factor=kwargs['lrReduceFactor'], patience=kwargs['lrReducePatience'], threshold=kwargs['lrReduceThreshold'])
        self.SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.SGD, factor=kwargs['lrReduceFactor'], patience=kwargs['lrReducePatience'], threshold=kwargs['lrReduceThreshold'])

    def model_setup(self, **kwargs):
        self.AEV_setup(**kwargs)
        self.NN_setup(**kwargs)
        self.model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device)

    def fix_layers(self, layer_to_fix):
        if layer_to_fix:
            if len(layer_to_fix) == 1:
                layer_to_fix = layer_to_fix * len(self.species_order)
            for name, parameter in self.model.named_parameters():
                indices = name.split('.')
                if int(indices[-2]) in layer_to_fix[self.species_order.index(indices[-3] if indices[-3] in data.element_symbol2atomic_number else data.atomic_number2element_symbol[int(indices[-3])])]:
                    parameter.requires_grad=False

    def data_setup(self, molecular_database, spliting_ratio,
                   property_to_learn, xyz_derivative_property_to_learn, ):
        assert molecular_database, 'provide molecular database'

        self.property_name = property_to_learn

        data_atomic_numbers = np.sort(np.unique([mol.get_atomic_numbers() for mol in molecular_database.molecules]))
        data_element_symbols = [data.atomic_number2element_symbol[z] for z in data_atomic_numbers]

        if not self.species_order: 
            self.species_order = data_element_symbols
        else:
            for element in data_element_symbols:
                if element not in self.species_order:
                    print('element(s) outside supported species detected, please check the database')
                    return
                
        if self.energy_shifter.self_energies is None:
            if isinstance(molecular_database[0], data.molecular_database):
                assert len(molecular_database) == 2, 'provide only two databases in an iterable, or just pass one database'
                if np.isnan(molecular_database[0].get_properties(property_to_learn)).sum():
                    molDB2ANIdata(molecular_database[0].filter_by_property(property_to_learn), property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order)
                    self.subtraining_set = molDB2ANIdata(molecular_database[0], property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
                else:   
                    self.subtraining_set = molDB2ANIdata(molecular_database[0], property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order).species_to_indices(self.species_order).shuffle()
                self.validation_set = molDB2ANIdata(molecular_database[1], property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
            else:
                if np.isnan(molecular_database.get_properties(property_to_learn)).sum():
                    molDB2ANIdata(molecular_database.filter_by_property(property_to_learn), property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order)
                    self.subtraining_set, self.validation_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle().split(spliting_ratio, None)
                else:
                    self.subtraining_set, self.validation_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).subtract_self_energies(self.energy_shifter, self.species_order).species_to_indices(self.species_order).shuffle().split(spliting_ratio, None)
        else:
            if isinstance(molecular_database[0], data.molecular_database):
                assert len(molecular_database) == 2, 'provide only two databases in an iterable, or just pass one database'
                self.subtraining_set = molDB2ANIdata(molecular_database[0], property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
                self.validation_set = molDB2ANIdata(molecular_database[1], property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle()
            else:
                self.subtraining_set, self.validation_set = molDB2ANIdata(molecular_database, property_to_learn, xyz_derivative_property_to_learn).species_to_indices(self.species_order).subtract_self_energies(self.energy_shifter.self_energies).shuffle().split(spliting_ratio, None)
        
        self.energy_shifter = self.energy_shifter.to(self.device)
        
        self.subtraining_set = self.subtraining_set.collate(self.hyperparameters['batchSize']).cache()
        self.validation_set = self.validation_set.collate(self.hyperparameters['batchSize']).cache()

        self.argsdict.update({'self_energies': self.energy_shifter.self_energies, 'property': self.property_name})


    def save(self, model_file=None):
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
        if not self.shutup: print(f'model saved in {model_file}')

    def train(self, molecular_database=None,
              spliting_ratio=0.8, 
              property_to_learn='energy',
              xyz_derivative_property_to_learn=None,
              save_model=True,
              check_point=None,
              reset_optim_state=False,
              use_last_model=False,
              reset_parameters=False,
              reset_optimizer=False,
              hyperparameters={},
              weight_by_energy=False):
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        
        self.data_setup(molecular_database, spliting_ratio, property_to_learn, xyz_derivative_property_to_learn)

        if not self.model:
            self.model_setup(**self.hyperparameters)

        elif reset_parameters:
            self.NN_initialize()
        
        if reset_optimizer:
            self.optimizer_setup(**self.hyperparameters)

        self.model.train()

        if check_point and os.path.isfile(check_point):
            checkpoint = torch.load(check_point)
            self.nn.load_state_dict(checkpoint['nn'])
            if not reset_optim_state:
                self.AdamW.load_state_dict(checkpoint['AdamW'])
                self.SGD.load_state_dict(checkpoint['SGD'])
                self.AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
                self.SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])
    
        def validate():
            mse= torch.nn.MSELoss(reduction='none')
            total_mse = 0.0
            count = 0
            for properties in self.validation_set:
                species = properties['species'].to(self.device)
                coordinates = properties['coordinates'].to(self.device).float()
                true_energies = properties['energies'].to(self.device).float()
                _, predicted_energies = self.model((species, coordinates))
                total_mse += mse(predicted_energies, true_energies).nansum().item()
                count += predicted_energies.shape[0]
            return math.sqrt(total_mse / count)

        def loss_function(prediction, reference, weightings=1):
            return torch.nn.functional.mse_loss(prediction, reference, reduction='none')*weightings**2

        if not self.shutup: print("training starting from epoch", self.AdamW_scheduler.last_epoch + 1)
        for _ in range(self.AdamW_scheduler.last_epoch + 1, self.hyperparameters['maxEpochs']):
            rmse = validate()
            if not self.shutup: print('validation RMSE:', rmse, 'at epoch', self.AdamW_scheduler.last_epoch + 1)
            sys.stdout.flush()
            learning_rate = self.AdamW.param_groups[0]['lr']
            if not self.shutup: print('learning_rate:',learning_rate)

            if learning_rate < self.hyperparameters['earlyStopLR']:
                break

            if self.AdamW_scheduler.is_better(rmse, self.AdamW_scheduler.best):
                if save_model:
                    self.save(self.model_file)

            self.AdamW_scheduler.step(rmse)
            self.SGD_scheduler.step(rmse)
            for properties in tqdm.tqdm(
                self.subtraining_set,
                total=len(self.subtraining_set),
                desc="epoch {}".format(self.AdamW_scheduler.last_epoch),
                disable=self.shutup,
            ):
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                if weight_by_energy:
                    weightings_e = weight_by_energy(true_energies)
                    weightings_f = weight_by_energy(true_energies)[:, None, None]
                else:
                    weightings_e = 1
                    weightings_f = 1 


                if xyz_derivative_property_to_learn:
                    coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                    true_forces = properties['forces'].to(self.device).float()
                    _, predicted_energies = self.model((species, coordinates))
                    forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    true_energies[true_energies.isnan()]=predicted_energies[true_energies.isnan()]
                    if self.hyperparameters['medianLoss']:
                        energy_loss= median(predicted_energies,true_energies)
                    else:
                        energy_loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()
                    true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                    force_loss = (loss_function(true_forces, forces, weightings_f).sum(dim=(1, 2)) / num_atoms).nanmean()
                    loss = energy_loss + self.hyperparameters['fCoef'] * force_loss
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    _, predicted_energies = self.model((species, coordinates))
                    loss = (loss_function(predicted_energies, true_energies, weightings_e) / num_atoms.sqrt()).nanmean()

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

    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False,  calculate_hessian=False,
                predict_property = True, property_to_predict = None,
                predict_xyz_derivative_property = False, xyz_derivative_property_to_predict = None,
                predict_hessian = False, hessian_to_predict = None):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)

        if  calculate_energy:
            predict_property = True
            property_to_predict = 'energy'
        elif predict_property and property_to_predict == None:
            property_to_predict = f'estimated_{self.property_name}'
        elif property_to_predict != None:
            predict_property = True
                       
        if calculate_energy_gradients:
            predict_xyz_derivative_property = True
            xyz_derivative_property_to_predict = 'energy_gradients'
        elif predict_xyz_derivative_property and xyz_derivative_property_to_predict == None:
            xyz_derivative_property_to_predict = f'estimated_xyz_derivatives_{self.property_name}'
        elif xyz_derivative_property_to_predict != None:
            predict_xyz_derivative_property = True
        
        if calculate_hessian:
            predict_hessian = True
            hessian_to_predict = 'hessian'
        elif predict_hessian and hessian_to_predict == None:
            hessian_to_predict = f'estimated_hessian_{self.property_name}'
        elif hessian_to_predict != None:
            predict_hessian = True
        
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)

        for mol in molDB.molecules:
            species = species_to_tensor(mol.get_element_symbols()).to(self.device).unsqueeze(0)
            xyz_coordinates = torch.tensor(mol.get_xyz_coordinates()).float().to(self.device).requires_grad_(calculate_energy_gradients or calculate_hessian).unsqueeze(0)
            ANI_NN_energy = self.energy_shifter(self.model((species, xyz_coordinates))).energies
            if predict_property: mol.__dict__[property_to_predict] = float(ANI_NN_energy)
            if predict_xyz_derivative_property or predict_hessian:
                ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energy.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                if calculate_energy_gradients:
                    grads = ANI_NN_energy_gradients[0].detach().cpu().numpy()
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].__dict__[xyz_derivative_property_to_predict] = grads[iatom]
            if predict_hessian:
                ANI_NN_hessian = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energy)
                mol.__dict__[hessian_to_predict] = np.array(ANI_NN_hessian[0])

class ani_child():
    def __init__(self, parent, index, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.model = parent.__getitem__(index)

    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, batch_size=1024):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)

        for mol in molDB.molecules:
            species = torch.tensor([atom.atomic_number for atom in mol.atoms]).to(self.device).unsqueeze(0)
            xyz_coordinates = torch.tensor(mol.get_xyz_coordinates()).double().to(self.device).requires_grad_(calculate_energy_gradients or calculate_hessian).unsqueeze(0)
            ANI_NN_energy = self.model((species, xyz_coordinates)).energies
            if calculate_energy: mol.energy = float(ANI_NN_energy)
            if calculate_energy_gradients or calculate_hessian:
                ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energy.sum(), xyz_coordinates, create_graph=True, retain_graph=True)[0]
                if calculate_energy_gradients:
                    grads = ANI_NN_energy_gradients[0].detach().cpu().numpy()
                    for iatom in range(len(mol.atoms)):
                        mol.atoms[iatom].energy_gradients = grads[iatom]
            if calculate_hessian:
                ANI_NN_hessian = torchani.utils.hessian(xyz_coordinates, energies=ANI_NN_energy)
                mol.hessian = np.array(ANI_NN_hessian[0])
    
    def node(self, name):
        return models.model_tree_node(name=name, operator='predict', model=self)
    
class ani_methods():
    available_methods = ["ANI-1x", "ANI-1ccx", "ANI-2x", 'ANI-1x-D4', 'ANI-2x-D4']
    atomic_energies = {'ANI-1ccx': {1:-0.50088088, 6:-37.79199048, 7:-54.53379230, 8:-75.00968205}}

    def __init__(self, method='ANI-1ccx', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.model_setup(method)
        modelname = method.lower().replace('-','')
        self.children = [ani_child(self.model, index, device=device).node(f'{modelname}_nn{index}') for index in range(len(self.model))]
        if 'D4'.casefold() in self.method.casefold():
            d4 = models.model_tree_node(name='d4_wb97x', operator='predict', model=models.methods(method='D4', functional='wb97x'))
            ani_nns = models.model_tree_node(name=f'{modelname}_nn', children=self.children, operator='average')
            self.model = models.model_tree_node(name=modelname, children=[ani_nns, d4], operator='sum')
        else:
            self.model = models.model_tree_node(name=modelname, children=self.children, operator='average')

    def model_setup(self, method):
        self.method = method
        if 'ANI-1x'.casefold() in method.casefold():
            self.model = torchani.models.ANI1x(periodic_table_index=True).to(self.device).double()
            self.atomic_number_available = [1, 6, 7, 8]     
        elif 'ANI-1ccx'.casefold() in method.casefold():
            self.model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device).double()
            self.atomic_number_available = [1, 6, 7, 8]          
        elif 'ANI-2x'.casefold() in method.casefold():
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device).double()
            self.atomic_number_available = [1, 6, 7, 8, 9, 16, 17]
        else:
            print("method not found, please check ANI_methods().available_methods")

    def predict(self, molecular_database=None, molecule=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)

        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol,
                                    calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)
        
    def predict_for_molecule(self, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        
        for atom in molecule.atoms:
            if not atom.atomic_number in self.atomic_number_available:
                print(f' * Warning * Molecule contains elements other than {"C/H/N/O/F/S/Cl" if self.method.casefold() == "ANI-2x".casefold() else "CHNO"}, no calculations performed')
                return
        
        if len(molecule.atoms) == 1:
            molecule.energy = self.atomic_energies[self.method][molecule.atoms[0].atomic_number]
            
        else:
            self.model.predict(molecule=molecule,
                               calculate_energy=calculate_energy,
                               calculate_energy_gradients=calculate_energy_gradients, 
                               calculate_hessian=calculate_hessian)
            
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
