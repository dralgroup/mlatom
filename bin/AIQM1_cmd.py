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
import time
import warnings
from scipy.optimize import approx_fprime
import MLtasks
from utils import *
#import numdifftools as nd

warnings.filterwarnings("ignore")

mlatomdir=os.path.dirname(__file__)
bohr2a = 0.52917721067

atomEnergyODM2 = [-0.4589994395, 0.0, 0.0, 0.0, 0.0, -4.3614347247, -7.6095532253, -12.0462778809]
atomEnergyAIQM1 = [-0.50088038, 0.0, 0.0, 0.0, 0.0, -37.79221710, -54.53360298, -75.00986203]

class AIQM1Cls(object):
    def __init__(self, argsAIQM1 = sys.argv[1:]):
        
        args.parse(argsAIQM1)
        self.qmbin = args.qmbin
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
        self.gradcalc = args.gradcalc
        self.hesscalc = args.hesscalc
        self.addDelta = args.addDelta
        self.deltaType = args.deltaType
        self.deltaModel = args.deltaModel

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
        
        numbers, elements, coordinates = get_xyz(self.xyzfile)
        check_element(elements)
        
        if 'sparrow' in self.qmbin:
            charges, mults, odm2_energies, odm2_grads, odm2_hesses, scf_status = run_odm2_sparrow(self.qmbin, elements, coordinates, self.gradcalc, self.hesscalc)
        else:
            charges, mults, odm2_energies, odm2_grads, odm2_hesses, scf_status = run_odm2_mndo(self.qmbin, numbers, coordinates, self.gradcalc, self.hesscalc, self.mndokw)
        shifts = cal_shifts(numbers, sae)
        
        if d4:
            d4_energies, d4_grads, d4_hesses = run_d4_calculation(self.dftd4bin, self.xyzfile, elements, coordinates, 
                                                                  charges, self.gradcalc, self.hesscalc)
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)
        
        fenergy = open(args.yestfile, 'w')
        if self.gradcalc:
            fgrad = open(args.ygradxyzestfile, 'w')
        if self.hesscalc:
            fhess = open(args.hessianestfile, 'w')
         
        for i in range(len(numbers)):
            number = numbers[i]
            nat = len(number)
            coordinate =  torch.tensor(coordinates[i]).to(device).requires_grad_(self.gradcalc)
            shift = torch.tensor(shifts[i]).to(device)

            fmt = ' %-40s: %15.8f Hartree'
            if nat != 1:
                odm2_energy = torch.tensor(odm2_energies[i]).to(device)
                coordinate = coordinate.unsqueeze(0)
                species = species_to_tensor(number).to(device).unsqueeze(0)
                delta_energies = []; delta_grads = []; delta_hesses = []
                for mod in self.models:
                    delta_energy = mod((species, coordinate)).energies
                    delta_energies.append(delta_energy)
                    if self.gradcalc:
                        delta_grad = torch.autograd.grad(delta_energy.sum(), coordinate, create_graph=True, retain_graph=True)[0]
                        delta_grads.append(delta_grad)
                    if self.hesscalc:
                        delta_hess = torchani.utils.hessian(coordinate, energies=delta_energy)
                        delta_hesses.append(delta_hess)
                        
                delta_energy = torch.stack([e for e in delta_energies], dim=0)

                if len(self.models) == 8:
                    e_std = delta_energy.std(dim=0, unbiased=False)
                    print(fmt % ('Standard deviation of NN contribution', e_std), end='')
                    e_std = hartree2kcalmol(e_std)
                    print('%15.5f kcal/mol' % e_std)
                    np.savetxt('std', e_std.cpu().detach().numpy())
                    e_std = e_std.item()
                delta_energy = delta_energy.mean(0)
                energy = delta_energy[0] + odm2_energy + shift
                if self.gradcalc:
                    odm2_grad = torch.tensor(odm2_grads[i]).to(device)
                    delta_grad = torch.stack([g for g in delta_grads], dim=0).mean(0)
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
                    energy = energy + d4_energy
                    print(fmt % ('D4 contribution', d4_energy))
                    if self.gradcalc:
                        d4_grad = torch.tensor(d4_grads[i]).to(device)
                        grad = grad + d4_grad
                    if self.hesscalc:
                        d4_hess = torch.tensor(d4_hesses[i]).to(device)
                        hess = hess + d4_hess
            else:
                odm2_energy = odm2_energies[i]
                energy = atomEnergyAIQM1[number[0]-1] \
                       + odm2_energy \
                       - atomEnergyODM2[number[0]-1]
                grad = torch.zeros([nat, 3]).to(device)
                hess = torch.zeros([nat*3, nat*3]).to(device)

                if charges[i] == 0:
                    print(' atomic energies calculated by fitting to the heats of formation in the CHNO set')

            print(fmt % ('Total energy', energy))
            if not scf_status[i]:
                print(' * Warning * ODM2* calculation not converge!')
            if self.addDelta:
                saveXYZ('delta_temp.xyz', elements[i], coordinates[i])
                deltaEnergy, deltaGrad = delta_correction(self.deltaType, self.deltaModel)
                energy = energy + torch.tensor(deltaEnergy).to(device)
                grad = grad + torch.tensor(deltaGrad).to(device)
                print(fmt % ('Delta energy correction', deltaEnergy))
                print(fmt % ('Corrected energy', energy))
            print('')
            fenergy.write('%20.12f\n' % energy)
            if self.gradcalc:
                fgrad.write('%d\n\n' % nat)
                for j in range(nat):
                    fgrad.write('%20.12f %20.12f %20.12f\n' % (grad[j][0], grad[j][1], grad[j][2]))
            if self.hesscalc:
                hess = hess.flatten()
                fhess.write('%d\n\n' % nat)
                for j in range(len(hess)):
                    fhess.write('%20.12f\n' % hess[j])
        fenergy.close()
        if self.gradcalc:
            fgrad.close()
        if self.hesscalc:
            fhess.close()
        return charges, mults, e_std

    def forward(self):
        if self.level == 'dft':
            sae = self.dftsae
        elif self.level == 'cc':
            sae = self.ccsae
        self.define_aev()
        self.load_models()
        charges, mults, e_std = self.predict(self.device, self.d4, sae)
        return charges, mults, e_std

class args(object):

    @classmethod
    def check_prog(cls):
        # MNDO or Sparrow
        if cls.qmprog is None or cls.qmprog=='None':
            if 'mndobin' in os.environ.keys():
                cls.qmprog = 'mndo'
            elif 'sparrowbin' in os.environ.keys():
                cls.qmprog = 'sparrow'
        qmbin = None
        if cls.qmprog.lower() == 'mndo':
            print(' MNDO program used in ODM2* calculation')
            status = os.popen('echo $mndobin').read().strip()
            if len(status) != 0:
                qmbin = status
            else:
                stopper.stopMLatom('Can not find MNDO software, please set the environment variable: export mndobin=...')
        elif cls.qmprog.lower() == 'sparrow':
            print(' Sparrow program used in ODM2* calculation')
            status = os.popen('echo $sparrowbin').read().strip()
            if len(status) != 0:
                qmbin = status
            else:
                stopper.stopMLatom('Can not find Sparrow software, please set the environment variable: export sparrowbin=...')
        else:
            stopper.stopMLatom('AIQM1 calculations require Sparrow or MNDO program')
        
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
        
        cls.qmbin = qmbin
        cls.dftd4bin = dftd4bin

    @classmethod
    def parse(cls, argsraw):
        # Default values:
        cls.qmprog = None
        cls.xyzfile = None
        cls.level = 'cc'
        cls.d4 = False
        cls.model_index = None
        cls.mndokw = None
        cls.yestfile = 'enest.dat'
        cls.ygradxyzestfile = None
        cls.hessianestfile = None
        cls.gradcalc = False
        cls.hesscalc = False

        cls.addDelta = False
        cls.deltaType = None
        cls.deltaModel = None
        
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
                cls.gradcalc = True
            elif arg.lower()[0:len('hessianestfile=')]   == 'hessianestfile=':
                cls.hessianestfile = arg[len('hessianestfile='):]
                cls.hesscalc = True
            elif arg.lower()[0:len('qmprog=')]           == 'qmprog=':
                cls.qmprog = arg[len('qmprog='):]

            elif arg.lower()                             == 'adddelta':
                cls.addDelta = True
            elif arg.lower()[0:len('addDeltaType')]      == 'adddeltatype':
                cls.deltaType = arg[len('addDeltaType='):]
            elif arg.lower()[0:len('addDeltaModelIn')]   == 'adddeltamodelin':
                cls.deltaModel = arg[len('addDeltaModelIn='):]
            #else:
            #    printHelp()
            #    stopper.stopMLatom('Option "%s" is not recognized' % arg)
        if cls.addDelta:
            if cls.deltaType is None or cls.deltaModel is None:
                stopper.stopMLatom("Please specify 'addDeltaType' and 'addDeltaModelIn'")
        cls.check_prog()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_xyz(xyzfile):
    coordinates = []; numbers = []; elements = []
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
                    element = np.array([number2element[int(i)] for i in number])
                else:    
                    element = np.array(element)
                    number = np.array([element2number[i.upper()] for i in element])
                number = np.array(number).astype('int')
                coordinate = np.array(coordinate).astype('float')
                elements.append(element)
                numbers.append(number)
                coordinates.append(coordinate)
    return numbers, elements, coordinates

def save_xyz(fname, element, coordinate):
    nat = len(element)
    with open(fname, 'w') as f:
        f.write(str(nat) + '\n\n')
        for j in range(nat):
            f.write('%s %20.12f %20.12f %20.12f\n' % (element[j], coordinate[j][0], coordinate[j][1], coordinate[j][2]))

def run_odm2_sparrow(qmbin, elements, coordinates, gradcalc, hesscalc):
    charges = np.full(len(elements), 0)
    mults = np.full(len(elements), 1)
    energies = []; grads = []; hesses = []
    scf_status = []
    for i in range(len(elements)):
        element = elements[i]
        coord = coordinates[i]
        energy = get_sparrow_energy(coord, element, qmbin)
        converged = open('sparrow.out', 'r').read().find('SCF not converged')
        if converged == -1:
            scf_status.append(True)
            if gradcalc:
                grad = numerical_grad(coord, 1e-5, element, qmbin)
                #gfun = nd.Gradient(get_sparrow_energy)
                #grad = gfun(coord.reshape(-1), element, qmbin)
                #grad = grad.reshape(-1, 3)
            if hesscalc:
                hess = numerical_hess(coord, 5.29167e-4, element, qmbin)
                #hfun = nd.Hessian(get_sparrow_energy)
                #hess = hfun(coord.reshape(-1), element, qmbin)
        else:   
            scf_status.append(False)
            energy = 0.0
            nat =  len(element)
            grad = np.zeros([nat, 3])
            hess = np.zeros([nat*3, nat*3])
        energies.append(energy)
        if gradcalc:
            grads.append(grad)
        if hesscalc:
            hesses.append(hess)
    os.system('rm -f energy.dat sparrow_temp.xyz warning')
    return charges, mults, energies, grads, hesses, scf_status

def get_sparrow_energy(fcoord, *args):
    element = args[0]
    qmbin = args[1]
    coord = fcoord.reshape(-1, 3)
    save_xyz('sparrow_temp.xyz', element, coord)
    os.system(qmbin + " -x sparrow_temp.xyz -c 0 -s 1 -M ODM2 -o > sparrow.out")
    energy = np.loadtxt('energy.dat', comments='#').tolist()
    return energy

def numerical_grad(coord, eps, element, qmbin):
    fcoord = coord.reshape(-1)
    fgrad = approx_fprime(fcoord, get_sparrow_energy, eps, element, qmbin)
    grad = fgrad.reshape(-1, 3)
    return grad

def numerical_hess(coord, eps, element, qmbin):
    fcoord1 = coord.reshape(-1)
    g1 = approx_fprime(fcoord1, get_sparrow_energy, eps, element, qmbin)
    ndim = len(fcoord1)
    hess = np.zeros((ndim, ndim))
    fcoord2 = fcoord1
    for i in range(ndim):
        x0 = fcoord2[i]
        fcoord2[i] = fcoord1[i] + eps
        g2 = approx_fprime(fcoord2, get_sparrow_energy, eps, element, qmbin) 
        hess[:, i] = (g2 - g1) / eps
        fcoord2[i] = x0

    return hess

def run_odm2_mndo(qmbin, numbers, coordinates, gradcalc, hesscalc, mndokw):
    charges = np.full(len(numbers), 0)
    mults = np.full(len(numbers), 1)
    energies = []; grads = []; hesses = []
    scf_status = []
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
                elif gradcalc:
                    f.write('jop=-2 +\n')
                else:
                    f.write('jop=-1 +\n')
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
        os.system(qmbin + " < mndo.inp > mndo.out")
        converged = open('mndo.out', 'r').read().find('UNABLE TO ACHIEVE SCF CONVERGENCE')
        if converged == -1:
            energy, grad, hess = get_mndo_result(gradcalc, hesscalc)
            scf_status.append(True)
        else:   
            energy = 0.0
            nat =  len(number)
            grad = np.zeros([nat, 3])
            hess = np.zeros([nat*3, nat*3])
            scf_status.append(False)
        energies.append(energy)
        grads.append(grad)
        hesses.append(hess)

    energies = np.array(energies)
    return charges, mults, energies, grads, hesses, scf_status

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

def get_mndo_result(gradcalc, hesscalc):
    grad = None
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

def run_d4_calculation(dftd4bin, xyzfile, elements, coordinates, charges, gradcalc, hesscalc):
    shifts = []
    d4_energies = []; d4_grads = []; d4_hesses = []
    for i in range(len(elements)):
        element = elements[i]
        coordinate = coordinates[i]
        charge = charges[i]
        nat = len(element)
        save_xyz('coordinate', element, coordinate)
        if hesscalc:
            os.system(dftd4bin + " coordinate -s --grad --hess --orca --json -f wb97x -c " + str(charge) + " > /dev/null 2>&1")
        elif gradcalc:
            os.system(dftd4bin + " coordinate -s --grad --orca --json -f wb97x -c " + str(charge) + " > /dev/null 2>&1")
            
        else:
            os.system(dftd4bin + " coordinate -s --orca --json -f wb97x -c " + str(charge) + " > /dev/null 2>&1")
        
        with open('dftd4.json', 'r') as f:
            d4_results = json.load(f)
        d4_energy = np.array(d4_results['energy'])
        d4_energies.append(d4_energy)
        if gradcalc:
            d4_grad = np.array(d4_results['gradient']) / bohr2a
            d4_grad = d4_grad.reshape(-1, 3)
            d4_grads.append(d4_grad)
        if hesscalc:
            d4_hess = np.array(d4_results['hessian']).reshape(-1, nat*3) / (bohr2a * bohr2a)
            d4_hesses.append(d4_hess)
        #os.system('rm -f coordinate dftd4.json')
    os.system('rm -f coordinate dftd4.json')
    return d4_energies, d4_grads, d4_hesses

def delta_correction(deltaType, deltaModel):
    args2pass = ['useMLmodel', 'XYZfile=delta_temp.xyz', 'YestFile=delta_energy_temp.dat']
    args2pass.append('YgradXYZestFile=delta_grad_temp.dat')
    args2pass.append(f'MLmodelType={deltaType}')
    args2pass.append(f'MLmodelIn={deltaModel}')
    MLtasks.MLtasksCls(args2pass)
    deltaEnergy = np.loadtxt('delta_energy_temp.dat')
    deltaGrad = np.loadtxt('delta_grad_temp.dat', skiprows=2)
    os.system('rm -f delta_temp.xyz delta_energy_temp.dat delta_grad_temp.dat')
    return deltaEnergy, deltaGrad

def check_element(elements):
    element_AIQM1 = ['H', 'C', 'N', 'O']
    for element in elements:
        for s in element.flatten():
            if s not in element_AIQM1:
                stopper.stopMLatom('AIQM1 is only available for H, C, N, O elements now')


def printHelp():
    helpText = '''
  !---------------------------------------------------------------------------! 
  !                          AIQM1 usage and options                          ! 
  !---------------------------------------------------------------------------!
  
  Usage:
    MLatom.py AIQM1@DFT* xyzfile=... [options]
    MLatom.py AIQM1@DFT  xyzfile=... [options]
    MLatom.py AIQM1      xyzfile=... [options]
    
  Options:
      xyzfile=S                file S with xyz coordinates
      qmprog=S                 program used in ODM2* calculation, MNDO or Sparrow [default]
      model_index=N            0-7 (by default the ensemble of 8 models are used)
      YestFile=S               file S with estimated Y values
      YgradXYZestFile=S        file S with estimated XYZ gradients
      HessianestFile=S         file S with estimated Hessians
      mndokeywords=S           read MNDO keyword from file S, separating by a blank line for each molecules
      (it is required to set iop=-22, jop=-2 or 2, igeom=1, nsav15=3, immdp=-1)
'''
    print(helpText)

    
