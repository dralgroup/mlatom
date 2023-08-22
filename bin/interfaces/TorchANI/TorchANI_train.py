import torch
import torchani
import os
import sys
import math
# import torch.utils.tensorboard
import tqdm
import numpy as np

def train(args):    
    # if args.nthreads:
    #     torch.set_num_threads(args.nthreads)
    #     torch.set_num_interop_threads(args.nthreads)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training with %s'% device)
    max_epochs = args.ani.max_epochs
    best_model_checkpoint = args.mlmodelout 
    early_stopping_learning_rate = args.ani.early_stopping_learning_rate
    if args.ygradxyzfile:
        force_coefficient = args.ani.force_coefficient

    if args.ani.transfer_learning_fixed_layer:
        best_model = torch.load(args.mlmodelin)
        Rcr = best_model['args']['Rcr']
        Rca = best_model['args']['Rca']
        EtaR = best_model['args']['EtaR'].to(device)
        ShfR = best_model['args']['ShfR'].to(device)
        Zeta = best_model['args']['Zeta'].to(device)
        ShfZ = best_model['args']['ShfZ'].to(device)
        EtaA = best_model['args']['EtaA'].to(device)
        ShfA = best_model['args']['ShfA'].to(device)

        try: species_order=best_model['args']['species_order']
        except: species_order = args.atype
        
        num_species = len(species_order)
        self_energies_train=best_model['args']['self_energies_train']
        energy_shifter_train = torchani.utils.EnergyShifter(self_energies_train)
        
    else:
        Rcr = args.ani.Rcr
        Rca = args.ani.Rca
        EtaR = args.ani.EtaR.to(device)
        ShfR = args.ani.ShfR.to(device)
        Zeta = args.ani.Zeta.to(device)
        ShfZ = args.ani.ShfZ.to(device)
        EtaA = args.ani.EtaA.to(device)
        ShfA = args.ani.ShfA.to(device)

        species_order=args.atype
        num_species = len(species_order)
        energy_shifter_train = torchani.utils.EnergyShifter(None)

    energy_shifter_validate = torchani.utils.EnergyShifter(None)

    trainfile = 'ANI_subtrain.h5'
    validatefile = 'ANI_validate.h5'

    batch_size = args.ani.batch_size
    
    if args.ygradxyzfile:
        training = torchani.data.load(trainfile, additional_properties=('forces',)).subtract_self_energies(energy_shifter_train, species_order).species_to_indices(species_order).shuffle()
        validation= torchani.data.load(validatefile, additional_properties=('forces',)).subtract_self_energies(energy_shifter_validate, species_order).species_to_indices(species_order).shuffle()
    else:
        training = torchani.data.load(trainfile).subtract_self_energies(energy_shifter_train, species_order).species_to_indices(species_order).shuffle()
        validation = torchani.data.load(validatefile).subtract_self_energies(energy_shifter_validate, species_order).species_to_indices(species_order).shuffle()

    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()

    self_energies_train=np.array(energy_shifter_train.self_energies)
    self_energies_validate=np.array(energy_shifter_validate.self_energies)

    if args.ani.transfer_learning_fixed_layer:
        argdic=best_model['args']
    else:
        argdic = {'Rcr': Rcr, 'Rca': Rca, 'EtaR': EtaR, 'ShfR': ShfR, 'Zeta': Zeta, 'ShfZ': ShfZ, 'EtaA': EtaA, 'ShfA': ShfA, 
        'energy_shifter_validate':energy_shifter_validate, 'energy_shifter_train':energy_shifter_train,     
        'self_energies_validate':self_energies_validate, 'self_energies_train':self_energies_train, 
        'species_order':species_order}



    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    aev_dim = aev_computer.aev_length

    if args.ani.transfer_learning_fixed_layer:
        networkdic = best_model['network']
        nn = torchani.ANIModel([networkdic[specie] for specie in species_order])
        nn.load_state_dict(best_model['nn'])
    else:
        networkdic ={}
        for i, specie in enumerate(species_order):
            args.ani.Neurons[i]+=[1]
            layers=[torch.nn.Linear(aev_dim,args.ani.Neurons[i][0])]
            for j in range(len(args.ani.Neurons[i])-1):
                layers+=[torch.nn.CELU(0.1)]
                layers+=[torch.nn.Linear(args.ani.Neurons[i][j],args.ani.Neurons[i][j+1])]
            networkdic[specie] = torch.nn.Sequential(*layers)
        nn = torchani.ANIModel([networkdic[specie] for specie in species_order])

        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=1.0)
                torch.nn.init.zeros_(m.bias)

        nn.apply(init_params)
    
    print('batch_size: %s' % batch_size)

    print('Neural Network Structure:\n', nn)


    model = torchani.nn.Sequential(aev_computer, nn).to(device)
    # print(args.ani.transfer_learning_fixed_layer)
    if args.ani.transfer_learning_fixed_layer:
        print(' parameters below are fixed:')
        for name, parameter in model.named_parameters():
            if name.split('.')[-2] in args.ani.transfer_learning_fixed_layer:
                parameter.requires_grad=False
                print(name)

    # wlist2d = [
    #     [
    #         {'params': [networkdic[specie][0].weight]},
    #         {'params': [networkdic[specie][2].weight], 'weight_decay': 0.00001},
    #         {'params': [networkdic[specie][4].weight], 'weight_decay': 0.000001},
    #         {'params': [networkdic[specie][6].weight]},
    #     ]
    #     for specie in species_order
    # ]
    wlist2d = [
        [
            {'params': [networkdic[specie][j*2].weight]} if j==0 or j==len(args.ani.Neurons[i])-1 else {'params': [networkdic[specie][j*2].weight], 'weight_decay': 0.0001/10**j}
            for j in range(len(args.ani.Neurons[i]))
        ]
        for i, specie in enumerate(species_order)
    ]
    # print(wlist2d)

    AdamW = torch.optim.AdamW([i for j in wlist2d for i in j],lr=args.ani.learning_rate)

    # blist2d = [
    #     [
    #         {'params': [networkdic[specie][0].bias]},
    #         {'params': [networkdic[specie][2].bias]},
    #         {'params': [networkdic[specie][4].bias]},
    #         {'params': [networkdic[specie][6].bias]},
    #     ]
    #     for specie in species_order
    # ]

    blist2d = [
        [
            {'params': [networkdic[specie][j*2].bias]}
            for j in range(len(args.ani.Neurons[i]))
        ]
        for i, specie in enumerate(species_order)
    ]

    SGD = torch.optim.SGD([i for j in blist2d for i in j], lr=args.ani.learning_rate)
  
    AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=args.ani.lrfactor, patience=args.ani.patience, threshold=0)
    SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=args.ani.lrfactor, patience=args.ani.patience, threshold=0)    

    latest_checkpoint = 'latest.pt'

    # if os.path.isfile(latest_checkpoint) and args.ani.restart:
    if os.path.isfile(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        nn.load_state_dict(checkpoint['nn'])
        if not args.ani.reset_optim_state:
            AdamW.load_state_dict(checkpoint['AdamW'])
            SGD.load_state_dict(checkpoint['SGD'])
            AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
            SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

    def getOffsets(self_energies, species):
        offsets=[]
        for sp in species:
            offset=0
            for idx in sp:
                if idx in range(len(species_order)):
                    try: offset+=self_energies[idx]
                    except: pass
            offsets.append(offset)
        return torch.Tensor(offsets).to(device).float()

    def validate():
        # run validation
        mse_sum = torch.nn.MSELoss(reduction='sum')
        total_mse = 0.0
        count = 0
        for properties in validation:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            _, predicted_energies = model((species, coordinates))
            # print(true_energies,predicted_energies)
            # print(species,self_energies_train,true_energies,predicted_energies,true_energies+predicted_energies)
            # print(getOffsets(self_energies_validate,species) , getOffsets(self_energies_train,species))
            # print(getOffsets(self_energies_validate,species) - getOffsets(self_energies_train,species))
            # print(true_energies-deltaOffset,'\n\n',predicted_energies)
            total_mse += mse_sum(predicted_energies, true_energies+ getOffsets(self_energies_validate,species) - getOffsets(self_energies_train,species)).item()
            # total_mse += mse_sum(predicted_energies, true_energies + energy_shifter_validate.sae(species) - energy_shifter_train.sae(species)).item()
            count += predicted_energies.shape[0]
        return math.sqrt(total_mse / count)

    # tensorboard = torch.utils.tensorboard.SummaryWriter()

    mse = torch.nn.MSELoss(reduction='none')
    def median(yp,yt):
        return torch.median(torch.abs(yp-yt))
    

    print("training starting from epoch", AdamW_scheduler.last_epoch + 1)

    for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
        rmse = validate()
        print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)
        sys.stdout.flush()

        learning_rate = AdamW.param_groups[0]['lr']
        print('learning_rate:',learning_rate)

        if learning_rate < early_stopping_learning_rate:
            break

        if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
            torch.save(
                {   
                    'network': networkdic,
                    'args': argdic,
                    'nn': nn.state_dict()
                }
                , best_model_checkpoint
            )
            print('best model saved')
        AdamW_scheduler.step(rmse)
        SGD_scheduler.step(rmse)

        # tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
        # tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
        # tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
        for i, properties in tqdm.tqdm(
            enumerate(training),
            total=len(training),
            desc="epoch {}".format(AdamW_scheduler.last_epoch)
        ):
            species = properties['species'].to(device)
            if args.ygradxyzfile: 
                coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
                true_energies = properties['energies'].to(device).float()
                true_forces = properties['forces'].to(device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                _, predicted_energies = model((species, coordinates))
                forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                if args.minimizeError.lower()=='median':
                    energy_loss=median(predicted_energies,true_energies)
                else:
                    energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
                # print(true_forces)
                true_forces[true_forces.isnan()]=forces[true_forces.isnan()]
                force_loss = (mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms).nanmean()
                loss = energy_loss + force_coefficient * force_loss
                # print(true_forces)
                # print(forces)
                # print(mse(true_forces, forces).sum(dim=(1, 2)) / num_atoms)
                # print(force_loss)
                # print(loss)
                # print('\n\n',true_energies+energyOffset,'\n\n',predicted_energies+energyOffset)
            else:
                coordinates = properties['coordinates'].to(device).float()
                true_energies = properties['energies'].to(device).float()
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                _, predicted_energies = model((species, coordinates))
                loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

            AdamW.zero_grad()
            SGD.zero_grad()
            loss.backward()
            AdamW.step()
            SGD.step()

            # tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

        torch.save({
            'nn': nn.state_dict(),
            'AdamW': AdamW.state_dict(),
            'SGD': SGD.state_dict(),
            'AdamW_scheduler': AdamW_scheduler.state_dict(),
            'SGD_scheduler': SGD_scheduler.state_dict(),
        }, latest_checkpoint)
        
