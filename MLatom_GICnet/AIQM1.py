#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! AIQM1: Artificial intelligence quantum-mechanical method 1                ! 
  ! Implementations by: Peikung Zheng                                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os
import sys
from collections import Counter
import stopper
import json
import struct
import re
#import time
import warnings
warnings.filterwarnings("ignore")

mlatomdir=os.path.dirname(__file__)
bohr2a = 0.52917721067

class AIQM1Cls(object):
    def __init__(self, argsAIQM1 = sys.argv[1:]):
        
        args.parse(argsAIQM1)
        self.mndobin = args.mndobin
        self.dftd4bin = args.dftd4bin

        self.xyzfile = args.xyzfile
        self.device = args.device
        self.level = args.level
        self.d4 = args.d4
        self.model_index = args.model_index
        self.mndokw = args.mndokw
        self.yestfile = args.yestfile
        self.ygradestfile = args.ygradxyzestfile
        self.hessianestfile = args.hessianestfile
        self.hesscalc = args.hesscalc
        self.species_order =  [1, 6, 7, 8]
        self.dftsae = [-4.27888067e-02, -3.34869833e+01, -4.69896148e+01, -6.30294433e+01]
        self.ccsae = [-4.29365862e-02, -3.34329586e+01, -4.69301173e+01, -6.29634763e+01]
    
    def define_aev(self):
        Rcr = 5.2000e+00
        Rca = 4.0000e+00
        EtaR = torch.tensor([1.6000000e+01], device=self.device)
        ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=self.device)
        Zeta = torch.tensor([3.2000000e+01], device=self.device)
        ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=self.device)
        EtaA = torch.tensor([8.0000000e+00], device=self.device)
        ShfA = torch.tensor([9.0000000e-01, 1.6750000e+00,  2.4499998e+00, 3.2250000e+00], device=self.device)
        num_species = len(self.species_order)
        aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        self.aev_computer = aev_computer

    def define_nn(self):
        aev_dim = self.aev_computer.aev_length
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 144),
            torch.nn.GELU(),
            torch.nn.Linear(144, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 112),
            torch.nn.GELU(),
            torch.nn.Linear(112, 96),
            torch.nn.GELU(),
            torch.nn.Linear(96, 1)
        )
        
        nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
        #print(nn)
        self.nn = nn

    def load_models(self):
        dirname = os.path.join(mlatomdir, 'aiqm1_model')
        method = 'aiqm1_' + self.level
        models = list()
        if self.model_index is None:
            for i in range(8):
                self.define_nn()
                checkpoint = torch.load(os.path.join(dirname, f'{method}_cv{i}.pt'), map_location=self.device)
                self.nn.load_state_dict(checkpoint['nn'])
                model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).double()
                models.append(model)
        else:
            self.define_nn()
            checkpoint = torch.load(os.path.join(dirname, f'{method}_cv{self.model_index}.pt'), map_location=self.device)
            self.nn.load_state_dict(checkpoint['nn'])
            model = torchani.nn.Sequential(self.aev_computer, self.nn).to(self.device).double()
            models.append(model)
        self.models = models

    def predict(self, device, d4, sae):
        
        numbers, coordinates = get_xyz(self.xyzfile)
        charges, mults, odm2_energies, odm2_grads, odm2_hesses = run_odm2_calculation(self.mndobin, numbers, coordinates, self.hesscalc, self.mndokw)
        shifts = cal_shifts(numbers, sae)
        if d4:
            d4_energies, d4_grads, d4_hesses = run_d4_calculation(self.dftd4bin, self.xyzfile, numbers, coordinates, charges, self.hesscalc)
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)
        
        fenergy = open(args.yestfile, 'w')
        fgrad = open(args.ygradxyzestfile, 'w')
        if self.hesscalc:
            fhess = open(args.hessianestfile, 'w')

        for i in range(len(numbers)):
            nat = len(numbers[i])
            coordinate =  torch.tensor(coordinates[i]).to(device).requires_grad_(True)
            shift = torch.tensor(shifts[i]).to(device)
            odm2_energy = torch.tensor(odm2_energies[i]).to(device)
            odm2_grad = torch.tensor(odm2_grads[i]).to(device)
            coordinate = coordinate.unsqueeze(0)
            species = species_to_tensor(numbers[i]).to(device).unsqueeze(0)
            delta_energies = []; delta_grads = []; delta_hesses = []
            for mod in self.models:
                delta_energy = mod((species, coordinate)).energies
                delta_grad = torch.autograd.grad(delta_energy.sum(), coordinate, create_graph=True, retain_graph=True)[0]
                delta_energies.append(delta_energy)
                delta_grads.append(delta_grad)
                if self.hesscalc:
                    delta_hess = torchani.utils.hessian(coordinate, energies=delta_energy)
                    delta_hesses.append(delta_hess)
                    
            delta_energy = torch.stack([e for e in delta_energies], dim=0)

            fmt = ' %-40s: %15.8f Hartree'
            if len(self.models) == 8:
                e_std = delta_energy.std(dim=0, unbiased=False)
                print(fmt % ('Standard deviation of NN contribution', e_std), end='')
                e_std = hartree2kcalmol(e_std)
                print('%15.5f kcal/mol' % e_std)
                np.savetxt('std', e_std.detach().numpy())
            delta_energy = delta_energy.mean(0)
            delta_grad = torch.stack([g for g in delta_grads], dim=0).mean(0)
            energy = delta_energy[0] + odm2_energy + shift
            grad = delta_grad[0] + odm2_grad
            print(fmt % ('NN contribution', delta_energy))
            print(fmt % ('Sum of atomic self energies', shift))
            print(fmt % ('ODM2* contribution', odm2_energy), file=sys.stdout)
            if self.hesscalc:
                odm2_hess = torch.tensor(odm2_hesses[i]).to(device)
                delta_hess = torch.stack([h for h in delta_hesses], dim=0).mean(0)
                hess = delta_hess + odm2_hess
            if d4:
                d4_energy = torch.tensor(d4_energies[i]).to(device)
                d4_grad = torch.tensor(d4_grads[i]).to(device)
                energy = energy + d4_energy
                grad = grad + d4_grad
                print(fmt % ('D4 contribution', d4_energy))
                if self.hesscalc:
                    d4_hess = torch.tensor(d4_hesses[i]).to(device)
                    hess = hess + d4_hess
            print(fmt % ('Total energy', energy))
            print('')
            fenergy.write('%20.12f\n' % energy)
            fgrad.write('%d\n\n' % nat)
            for j in range(nat):
                fgrad.write('%20.12f %20.12f %20.12f\n' % (grad[j][0], grad[j][1], grad[j][2]))
            if self.hesscalc:
                hess = hess.flatten()
                fhess.write('%d\n\n' % nat)
                for j in range(len(hess)):
                    fhess.write('%20.12f\n' % hess[j])
        fenergy.close()
        fgrad.close()
        if self.hesscalc:
            fhess.close()
        return charges, mults

    def forward(self):
        if self.level == 'dft':
            sae = self.dftsae
        elif self.level == 'cc':
            sae = self.ccsae
        self.define_aev()
        self.load_models()
        charges, mults = self.predict(self.device, self.d4, sae)
        return charges, mults

class args(object):

    @classmethod
    def check_prog(cls):
        # MNDO
        status = os.popen('echo $mndobin').read().strip()
        if len(status) != 0:
            mndobin = status
        else:
            stopper.stopMLatom('Can not find MNDO software, please set the environment variable: export mndobin=...')
        # dftd4
        dftd4bin = None
        if cls.d4:
            status = os.popen('echo $dftd4bin').read().strip()
            if len(status) != 0:
                dftd4bin = status
            else:
                stopper.stopMLatom('Can not find dftd4 software, please set the environment variable: export dftd4bin=...')
        # TorchANI
        try: 
            import torch
            import torchani
            from torchani.utils import ChemicalSymbolsToInts
            from torchani.units import hartree2kcalmol
        except: 
            stopper.stopMLatom('Please install all Python module required for TorchANI')
        globals()['torch'] = torch
        globals()['torchani'] = torchani
        globals()['ChemicalSymbolsToInts'] = ChemicalSymbolsToInts
        globals()['hartree2kcalmol'] = hartree2kcalmol
        
        cls.mndobin  = mndobin
        cls.dftd4bin = dftd4bin

    @classmethod
    def parse(cls, argsraw):
        # Default values:
        cls.xyzfile = None
        cls.level = 'cc'
        cls.d4 = False
        cls.model_index = None
        cls.mndokw = None
        cls.yestfile = 'enest.dat'
        cls.ygradxyzestfile = 'gradest.dat'
        cls.hessianestfile = None
        cls.hesscalc = False
        
        argslower = [arg.lower() for arg in argsraw]
        if 'xyzfile' not in ''.join(argslower):
            printHelp()
            stopper.stopMLatom('At least xyzfile should be provided')
        for arg in argsraw:
            if  (arg.lower() == 'help'
              or arg.lower() == '-help'
              or arg.lower() == '-h'
              or arg.lower() == '--help'):
                printHelp()
                stopper.stopMLatom('')
            elif arg.lower()[0:len('aiqm1dftstar')]      == 'aiqm1dftstar':  
                cls.level = 'dft'
            elif arg.lower()[0:len('aiqm1dft')]          == 'aiqm1dft':
                cls.level = 'dft'
                cls.d4 = True
            elif arg.lower()[0:len('aiqm1')]             == 'aiqm1':
                cls.level = 'cc'
                cls.d4 = True
            elif arg.lower()[0:len('xyzfile=')]          == 'xyzfile=':  
                cls.xyzfile = arg[len('xyzfile='):]
            #elif arg.lower()[0:len('gpu=')]              == 'gpu=':
            #    os.environ["CUDA_VISIBLE_DEVICES"] = arg[len('gpu='):]
            #    cls.device = torch.device('cuda')
            elif arg.lower()[0:len('model_index=')]      == 'model_index=':
                cls.model_index = int(arg[len('model_index='):])
            elif arg.lower()[0:len('mndokeywords=')]     == 'mndokeywords=':
                cls.mndokw = arg[len('mndokeywords='):]
            elif arg.lower()[0:len('yestfile=')]         == 'yestfile=':
                cls.yestfile = arg[len('yestfile='):]
            elif arg.lower()[0:len('ygradxyzestfile=')]  == 'ygradxyzestfile=':
                cls.ygradxyzestfile = arg[len('ygradxyzestfile='):]
            elif arg.lower()[0:len('hessianestfile=')]   == 'hessianestfile=':
                cls.hessianestfile = arg[len('hessianestfile='):]
                cls.hesscalc = True
            #else:
            #    printHelp()
            #    stopper.stopMLatom('Option "%s" is not recognized' % arg)
        cls.check_prog()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_xyz(xyzfile):
    element2number = {'H':1, 'C':6, 'N':7, 'O':8}
    coordinates = []; numbers = []
    with open(xyzfile, 'r') as f:
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
    return numbers, coordinates

def run_odm2_calculation(mndobin, numbers, coordinates, hesscalc, mndokw):
    charges = np.full(len(numbers), 0)
    mults = np.full(len(numbers), 1)
    energies = []; grads = []; hesses = []
    if mndokw: 
        mndokeywords, charges, mults = read_mndokw(mndokw)
        if len(mndokeywords) != len(numbers):
            stopper.stopMLatom('Inconsistency of the nubmer of keywords and molecules')
    for i in range(len(numbers)):
        with open('mndo.inp', 'w') as f:
            if mndokw:
                f.write(mndokeywords[i] + '\n\n\n')
            else:
                if hesscalc:
                    f.write('jop=2 +\n')
                else:
                    f.write('jop=-2 +\n')
                f.write('iop=-22 igeom=1 iform=1 immdp=-1 nsav15=3 +\n')
                f.write('icuts=-1 icutg=-1 kitscf=9999 iscf=9 iplscf=9 +\n')
                f.write('iprint=-1 kprint=-5 lprint=-2 mprint=0 jprint=-1 +\n')
                f.write('kharge=0 imult=0 nprint=-1\n\n\n')
            number = numbers[i]
            coordinate = coordinates[i]
            for j in range(len(number)):
                f.write('%2s' % (number[j]) + ' '*5)
                f.write('%12.8f %3d' % (coordinate[j][0], 1))
                f.write(' '*5)
                f.write('%12.8f %3d' % (coordinate[j][1], 1))
                f.write(' '*5)
                f.write('%12.8f %3d' % (coordinate[j][2], 1))
                f.write('\n')
            f.write('\n')
        os.system(mndobin + " < mndo.inp > mndo.out")
        energy, grad, hess = get_odm2_result(hesscalc)
        energies.append(energy)
        grads.append(grad)
        hesses.append(hess)

    #energies = np.array(energies)
    return charges, mults, energies, grads, hesses

def read_mndokw(mndokw, split=False):
    with open(mndokw, 'r') as f:
        mndokeywords = f.read().strip('\n')
    mndokeywords = mndokeywords.split('\n\n')
    charges = np.full(len(mndokeywords), 0)
    mults = np.full(len(mndokeywords), 1)
    for i in range(len(mndokeywords)):
        status = re.search('kharge=-?\+?\d+', mndokeywords[i])
        if status:
            charges[i] = int(status.group().split('=')[1])
        status = re.search('imult=\d+', mndokeywords[i])
        if status:
            imult = int(status.group().split('=')[1])
            if imult == 0:
                mults[i] = 1
            else:
                mults[i] = imult
    if split:
        for i in range(len(mndokeywords)):
            with open(f'mndokw_split{i+1}', 'w') as f:
                f.write(mndokeywords[i])

    return mndokeywords, charges, mults

def get_odm2_result(hesscalc):
    hess = None
    with open('fort.15', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if 'NUMAT' in lines[i]:
                nat = int(lines[i].strip().split()[-1])
            if 'ENERGY' in lines[i]:
                energy = lines[i+1].strip().split()[0]
                energy = float(energy) / (27.21 * 23.061)
            if 'CARTESIAN GRADIENT' in lines[i]:
                grad = []
                for j in range(nat):
                    grad.append(lines[i+j+1].strip().split()[-3:])
                grad = np.array(grad).astype('float') / (27.21 * 23.061)
            if hesscalc:
                lhess = nat * nat * 9
                fhess = open('fort.4', 'rb')
                data = fhess.read()
                dt = f'id{lhess}d'
                dat_size = struct.calcsize(dt)
                temp = struct.unpack(dt, data[:dat_size])
                hess = np.array(temp[2:]).reshape(-1, nat*3) / (0.529167 * 0.529167)

    return energy, grad, hess

def cal_shifts(numbers, sae):
    shifts = []
    for i in range(len(numbers)):
        count = Counter(numbers[i])
        ncount = [count[1], count[6], count[7], count[8]]
        shift = np.dot(ncount, sae)
        shifts.append(shift)
    return shifts

def run_d4_calculation(dftd4bin, xyzfile, numbers, coordinates, charges, hesscalc):
    number2element = {1:'H', 6:'C', 7:'N', 8:'O'}
    shifts = []
    d4_energies = []; d4_grads = []; d4_hesses = []
    for i in range(len(numbers)):
        number = numbers[i]
        coordinate = coordinates[i]
        charge = charges[i]
        nat = len(numbers[i])
        with open('coordinate', 'w') as f:
            f.write(str(nat) + '\n\n')
            for j in range(nat):
                f.write('%s %15.12f %15.12f %15.12f\n' 
                    % (number2element[number[j]], coordinate[j][0], coordinate[j][1], coordinate[j][2]))
        if hesscalc:
            os.system(dftd4bin + " coordinate -s --grad --hess --orca --json -f wb97x -c " + str(charge) + " > /dev/null 2>&1")
        else:
            os.system(dftd4bin + " coordinate -s --grad --orca --json -f wb97x -c " + str(charge) + " > /dev/null 2>&1")
        with open('dftd4.json', 'r') as f:
            d4_results = json.load(f)
        #try:
        #    with open('dftd4.json', 'r') as f:
        #        d4_results = json.load(f)
        #except:
        #    time.sleep(0.5)
        #    with open('dftd4.json', 'r') as f:
        #        d4_results = json.load(f)
        d4_energy = np.array(d4_results['energy'])
        d4_grad = np.array(d4_results['gradient']) / bohr2a
        d4_grad = d4_grad.reshape(-1, 3)
        d4_energies.append(d4_energy)
        d4_grads.append(d4_grad)
        if hesscalc:
            d4_hess = np.array(d4_results['hessian']).reshape(-1, nat*3) / (bohr2a * bohr2a)
            d4_hesses.append(d4_hess)
        #os.system('rm -f coordinate dftd4.json')
    os.system('rm -f coordinate dftd4.json')
    return d4_energies, d4_grads, d4_hesses

def printHelp():
    helpText = '''
  !---------------------------------------------------------------------------! 
  !                          AIQM1 usage and options                          ! 
  !---------------------------------------------------------------------------!
  
  Usage:
    MLatom.py AIQM1@DFT* xyzfile=... [options]
    MLatom.py AIQM1@DFT xyzfile=... [options]
    MLatom.py AIQM1 xyzfile=... [options]
    
  Options:
      xyzfile=S                file S with xyz coordinates
      model_index=N            0-7 (by default the ensemble of 8 models are used)
      YestFile=S               file S with estimated Y values
      YgradXYZestFile=S        file S with estimated XYZ gradients
      HessianestFile=S         file S with estimated Hessians
      mndokeywords=S           read MNDO keyword from file S, separating by a blank line for each molecules
      (it is required to set iop=-22, jop=-2 or 2, igeom=1, nsav15=3, immdp=-1)
'''
    print(helpText)

    
