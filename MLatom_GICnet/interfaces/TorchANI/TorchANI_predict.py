import torch
import torchani
import os
import math
# import torch.utils.tensorboard
import tqdm
import numpy as np
import h5py
from torchani.units import hartree2kcalmol



def predict(args):    
    # if args.nthreads:
    #     torch.set_num_threads(args.nthreads)
    #     torch.set_num_interop_threads(args.nthreads)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    energyOffset = 0
    best_model = torch.load(args.mlmodelin)

    Rcr = best_model['args']['Rcr']
    Rca = best_model['args']['Rca']
    EtaR = best_model['args']['EtaR'].to(device)
    ShfR = best_model['args']['ShfR'].to(device)
    Zeta = best_model['args']['Zeta'].to(device)
    ShfZ = best_model['args']['ShfZ'].to(device)
    EtaA = best_model['args']['EtaA'].to(device)
    ShfA = best_model['args']['ShfA'].to(device)

    sp_z = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118}
    z_sp={v:k for k,v in sp_z.items()}
    def switchSPandZ(d):
        if d[0] in sp_z.keys():
            out=[sp_z[i] for i in d]
        else:
            out=[z_sp[int(i)] for i in d]
        return out

    try: species_order=best_model['args']['species_order']
    except: species_order = args.atype
    num_species = len(species_order)

    # self_energies=best_model['args']['self_energies']
    # energyOffset=best_model['args']['energyOffset']
    self_energies_train=best_model['args']['self_energies_train']        
    # energy_shifter_train=best_model['args']['energy_shifter_train']
    # self_energies_validate=best_model['args']['self_energies_validate']

    aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    aev_dim = aev_computer.aev_length
    networkdic = best_model['network']

    nn = torchani.ANIModel([networkdic[specie] for specie in species_order])
    nn.load_state_dict(best_model['nn'])
    
    model = torchani.nn.Sequential(aev_computer, nn).to(device)
    model.eval()

    batch_size = args.ani.batch_size
                
    if args.setname: args.setname='_'+args.setname
    testfile=h5py.File('ANI'+args.setname+'.h5','r')
    test=testfile.get('dataset')
    # print(test.keys())
    # all_coordinates=torch.tensor(test.get('coordinates')[()]).to(device).float()

    try: test = torchani.data.load('ANI'+args.setname+'.h5').species_to_indices(species_order).collate(batch_size).cache()
    except: test = torchani.data.load('ANI'+args.setname+'.h5').species_to_indices(switchSPandZ(species_order)).collate(batch_size).cache()
    
    

    def getOffsets(self_energies, species):
        offsets=[]
        for sp in species:
            offset=0
            for idx in sp:
                if idx in range(len(species_order)):
                    try: offset+=self_energies[idx]
                    except: pass
            offsets.append(offset)
        return torch.Tensor(offsets)

    with open(args.yestfile,'wb') as fy,  open(args.ygradxyzestfile,'wb') as fgrad:
        # for i in range(math.ceil(len(all_coordinates)/batch_size)):
        #     coordinates=all_coordinates[batch_size*i:min(batch_size*(i+1),len(all_coordinates))].requires_grad_(True)

        #     species=test.get('species')[()]
        for properties in test:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
            
            # else:
            #     species=torch.tensor([[species_order.index(i.decode('ascii')) for i in species] for j in range(len(coordinates))]).to(device)

            _, predicted_energies = model((species, coordinates))
            # print(predicted_energies)
            predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

            predicted_energies = np.array(predicted_energies.cpu().detach().numpy()).astype(np.float64)+getOffsets(self_energies_train,species).numpy().astype(np.float64)
            # predicted_energies = np.array(predicted_energies.cpu().detach().numpy())+ energy_shifter_train.sae(species).numpy()
            # predicted_energies = np.array(predicted_energies.cpu().detach().numpy())
            
            predicted_forces = np.array(predicted_forces.cpu().detach().numpy()).astype(np.float64)
            # print(predicted_forces,coordinates)
                        
            np.savetxt(fy, predicted_energies.reshape(-1,1), fmt='%20.12f', delimiter=" ")
            for i in range(len(species)):
                natom=len([a for a in species[i] if a!=-1])
                line="%d\n\n" % natom
                fgrad.write(line.encode('utf-8'))
                np.savetxt(fgrad, -1*predicted_forces[i,:natom], fmt='%20.12f', delimiter=" ")
                
    if args.hessianestfile:
        with open(args.hessianestfile+args.setname,'wb') as fhess:
            # for i in range(math.ceil(len(all_coordinates)/batch_size)):
            #     coordinates=all_coordinates[batch_size*i:min(batch_size*(i+1),len(all_coordinates))].requires_grad_(True)
            #     species=test.get('species')[()]
            #     species=torch.tensor([[species_order.index(i.decode('ascii')) for i in species] for j in range(len(coordinates))]).to(device)
            for properties in test:
                species = properties['species'].to(device)
                coordinates = properties['coordinates'].to(device).float()
            
                energies = model((species, coordinates)).energies
                hessians = np.array(torchani.utils.hessian(coordinates, energies=energies).cpu().detach().numpy())
                np.savetxt(fhess, hessians, fmt='%20.12f', delimiter=" ")


    import time
    time.sleep(0.0000001) # Sometimes program hangs without no reason, adding short sleep time helps to exit this module without a problem / P.O.D., 2021-02-17

def ani_predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    element2number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'S':16, 'CL':17}
    coordinates = []; numbers = []
    with open(args.xyzfile,'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].strip().isdigit():
                element = []; coordinate = []
                nat = int(lines[i])
                for j in range(i+2, i+2+nat):
                    element.append(lines[j].strip().split()[0])
                    coordinate.append(lines[j].strip().split()[1:])
                if element[0].isdigit():
                    number = element
                else:    
                    number = np.array([element2number[i.upper()] for i in element])
                number = np.array(number).astype('int')
                coordinate = np.array(coordinate).astype('float')
                numbers.append(number)
                coordinates.append(coordinate)
    if args.mlmodeltype.lower() == 'ani1x':
        model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
    elif args.mlmodeltype.lower() == 'ani1ccx':
        model = torchani.models.ANI1ccx(periodic_table_index=True).to(device).double()
    elif args.mlmodeltype.lower() == 'ani2x':
        model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    
    fenergy = open(args.yestfile, 'w')
    fgrad = open(args.ygradxyzestfile, 'w')
    if args.hessianestfile:
        fhess = open(args.hessianestfile, 'w')
    
    fmt = ' %-40s: %15.8f Hartree'
    for i in range(len(numbers)):
        species = torch.tensor(numbers[i]).to(device).unsqueeze(0)
        coordinate = torch.tensor(coordinates[i]).to(device).requires_grad_(True).unsqueeze(0)
        energies = []; grads = []; hesses = []
        for mod in model:
            energy = mod((species, coordinate)).energies
            grad = torch.autograd.grad(energy.sum(), coordinate, create_graph=True, retain_graph=True)[0]
            energies.append(energy)
            grads.append(grad)
            if args.hessianestfile:
                hess = torchani.utils.hessian(coordinate, energies=energy)
                hesses.append(hess)
        
        energies = torch.stack([e for e in energies], dim=0)
        e_std = energies.std(dim=0, unbiased=False)
        print(fmt % ('Standard deviation of NN contribution', e_std), end='')
        energy = energies.mean(0)
        grad = torch.stack([g for g in grads], dim=0).mean(0)
        # members_energies is in the development version of TorchANI
        #energy = model((species, coordinate)).energies
        #grad = torch.autograd.grad(energy.sum(), coordinate, create_graph=True, retain_graph=True)[0]
        #e_std = model.members_energies((species, coordinate)).energies.std(dim=0, unbiased=False)
        e_std = hartree2kcalmol(e_std).detach().numpy()
        np.savetxt('std', e_std)
        print('%15.5f kcal/mol' % e_std)
        print(fmt % ('Total energy', energy))

        fenergy.write('%20.12f\n' % energy)
        nat = len(numbers[i])
        fgrad.write('%d\n\n' % nat)
        for j in range(nat):
            fgrad.write('%20.12f %20.12f %20.12f\n' % (grad[0][j][0], grad[0][j][1], grad[0][j][2]))
        if args.hessianestfile:
            hessian = torch.stack([h for h in hesses], dim=0).mean(0).flatten()
            fhess.write('%d\n\n' % nat)
            for j in range(len(hessian)):
                fhess.write('%20.12f\n' % hessian[j])
    
    fenergy.close()
    fgrad.close()
    if args.hessianestfile:
        fhess.close()
