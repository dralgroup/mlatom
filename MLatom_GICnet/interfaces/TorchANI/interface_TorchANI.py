#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_TorchANI: Interface between TorchANI and MLatom                 ! 
  ! Implementations by: Fuchun Ge and Max Pinheiro Jr                         ! 
  !---------------------------------------------------------------------------! 
'''
from lib2to3.pgen2.tokenize import TokenError
import numpy as np
import os, sys, subprocess, time, shutil, re, math, random
import stopper
from args_class import ArgsBase

filedir = os.path.dirname(__file__)

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'learningCurve', 'geomopt', 'freq', 'ts', 'irc','useMLmodel','CVopt','CVtest'
            ],
            bool
        )
        self.add_default_dict_args([
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin','setname'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodeltype': 'ANI',
            'mlmodelout': "ANIbestmodel.pt",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat",
            'hessianestfile': "",
            'lcNtrains': [],
            'atype': [],
            'nthreads': None
        })
        self.parse_input_content([
            'ani.batch_size=8',
            'ani.max_epochs=10000000',
            'ani.early_stopping_learning_rate=0.00001',
            'ani.force_coefficient=0.1',
            'ani.batch_sizes=0',
            'ani.patience=100',
            'ani.lrfactor=0.5',
            'ani.Rcr=5.2',
            'ani.Rca=3.5',
            'ani.EtaR=16',
            'ani.ShfR=0.9,1.16875,1.4375,1.70625,1.975,2.24375,2.5125,2.78125,3.05,3.31875,3.5875,3.85625,4.125,4.39375,4.6625,4.93125',
            'ani.Zeta=32',
            'ani.ShfZ=0.19634954,0.58904862,0.9817477,1.3744468,1.7671459,2.1598449,2.552544,2.9452431',
            'ani.EtaA=8',
            'ani.ShfA=0.90,1.55,2.20,2.85',
            'ani.Neuron_l1=160',
            'ani.Neuron_l2=128',
            'ani.Neuron_l3=96',
            'ani.AF1=CELU',
            'ani.AF2=CELU',
            'ani.AF3=CELU'
            ])

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        for k, v in self.ani.data.items():
            if k.lower() in ['shfr','etar','zeta','shfz','shfz','etaa','shfa']:
                self.ani.data[k] = torch.tensor([float(i) for i in str(v).split(',')])
            elif k.lower() in ['neuron_l1','neuron_l2','neuron_l3']:
                self.ani.data[k] =[int(i) for i in str(v).split(',')]
            elif k.lower() in ['af1','af2','af3']:
                self.ani.data[k] = v.split(',')


        with open(self.xyzfile,'r') as f:
            for line in f:
                natom = int(line)
                f.readline()
                for i in range(natom):
                    sp=f.readline().split()[0]
                    if sp not in self.atype:
                        self.atype.append(sp)
        
        if not self.mlmodelin:
            self.mlmodelin = self.mlmodelout
        
        species_order = sorted(set(self.atype), key=lambda x: self.atype.index(x))
        if len(self.ani.Neuron_l1) == 1: self.ani.Neuron_l1 = self.ani.Neuron_l1*len(species_order)
        if len(self.ani.Neuron_l2) == 1: self.ani.Neuron_l2 = self.ani.Neuron_l2*len(species_order)
        if len(self.ani.Neuron_l3) == 1: self.ani.Neuron_l3 = self.ani.Neuron_l3*len(species_order)
        # if self.lcyonly: self.ygradxyzfile=''
        
        if not self.ani.batch_size:
            if self.ntrain: self.ani.batch_size = int(math.sqrt(self.ntrain))
            else: self.ani.batch_size=8

        if self.learningcurve:
            self.lcNtrains = [int(i) for i in str(self.lcNtrains).split(',')]
            if self.ani.batch_sizes:
                self.ani.batch_sizes = [int(i) for i in self.lcNtrains.split(',')]
                self.ani.batch_size = self.ani.batch_sizes[self.lcNtrains.index(self.ntrain)]
            

class ANICls(object):
    dataConverted = False
    loaded=False
    coreset=False
    
    @classmethod
    def setCore(cls, n):
        if not cls.coreset:
            if n:
                torch.set_num_threads(n)
                torch.set_num_interop_threads(n)
            cls.coreset=True
    @classmethod
    def load(cls):
        if not cls.loaded:
            
            available = True
            try: 
                import h5py
                import torch
                from . import TorchANI_train
                from . import TorchANI_predict               
                
            except: 
                available = False
            

            globals()['available'] = available
            globals()['h5py'] = h5py
            globals()['torch'] = torch
            globals()['TorchANI_train'] = TorchANI_train
            globals()['TorchANI_predict'] = TorchANI_predict

            cls.batch_size = 8
            cls.max_epochs = 10000000
            cls.early_stopping_learning_rate = 1.0E-5
            cls.force_coefficient = 0.1
            cls.batch_sizes = []
            cls.patience = 100
            cls.lrfactor = 0.5
            cls.Rcr = 5.2000e+00
            cls.Rca = 3.5000e+00
            cls.EtaR = torch.tensor([1.6000000e+01])
            cls.ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00])
            cls.Zeta = torch.tensor([3.2000000e+01])
            cls.ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00])
            cls.EtaA = torch.tensor([8.0000000e+00])
            cls.ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.2000000e+00, 2.8500000e+00])
            cls.Neuron_l1 = [160]
            cls.Neuron_l2 = [128]
            cls.Neuron_l3 = [96]
            cls.AF1 = ['CELU']
            cls.AF2 = ['CELU']
            cls.AF3 = ['CELU']

            cls.loaded=True

    def __init__(self, argsANI = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)
    
    @classmethod
    def convertdata(cls, argsANI, subdatasets):
        cls.load()
        def convert(fileout, setname, yorgrad=False):
            # print(f'converting {fileout}...')
            # if os.path.isfile(fileout):
            #     if input(f'filtout exists, use it?    y/n\n').lower() in ['y','yes','yeah','ya','da']:
            #         return
            # convert data to .h5 format
            prefix = ''
            if args.learningcurve: prefix = '../'
            if setname: 
                coordfile = prefix+'xyz.dat_'+setname
                yfile = prefix+'y.dat_'+setname
                gradfile = prefix+'grad.dat_'+setname
            else: 
                coordfile = args.xyzfile
                if args.yfile and yorgrad: yfile = args.yfile
                if args.ygradxyzfile and yorgrad: gradfile = args.ygradxyzfile
            # print(args.atype)

            # with open(coordfile ,'r') as fxyz:
            #     ngeom=0
            #     for line in fxyz:
            #         natom=int(line)
            #         for i in range(natom+1):
            #             fxyz.readline()
            #         ngeom+=1

            with open(coordfile ,'r') as fxyz:
                hf = h5py.File(fileout, 'w')
                grp = hf.create_group('dataset')
                # natom_max=0
                # for line in fxyz:
                #     natom = int(line)
                #     natom_max = max(natom_max, natom)
                #     for i in range(natom+1):
                #         fxyz.readline()
                fxyz.seek(0)
                idx=0
                if args.yfile and yorgrad: 
                    fy =  open(yfile,'r')
                if args.ygradxyzfile and yorgrad: 
                    fgrad = open(gradfile,'r')

                for line in fxyz:
                    data={}
                    # data['smiles'] = np.array([args.smiles.encode('ascii')])
                    data['species'] = []
                    data['coordinates'] = np.array([])
                    if args.yfile and yorgrad: 
                        data['energies'] = np.array([float(fy.readline())])
                    if args.ygradxyzfile and yorgrad: 
                        data['forces'] = np.array([])
                        
                    natom = int(line)
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    # if args.yfile and yorgrad: data['energies'] =np.append(data['energies'], float(fy.readline()))
                    fxyz.readline()
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    for i in range(natom):
                        ll=fxyz.readline().split()
                        data['species'].append(ll[0])
                        data['coordinates'] = np.append(data['coordinates'], [float(i) for i in ll[-3:]])
                        if args.ygradxyzfile and yorgrad: 
                            data['forces'] = np.append(data['forces'], [-1*float(i) for i in fgrad.readline().split()[-3:]]) 
                    # for i in range(natom_max-natom):
                    #     data['coordinates'] = np.append(data['coordinates'], np.zeros(3))
                    #     if args.ygradxyzfile and yorgrad: 
                    #         data['forces'] = np.append(data['forces'], np.zeros(3)) 
                    data['species'] = np.array([i.encode('ascii') for i in data['species']])
                    data['coordinates'] = data['coordinates'].reshape(-1,natom,3)
                    if args.ygradxyzfile and yorgrad: data['forces'] = data['forces'].reshape(-1,natom,3)

                    subgrp = grp.create_group('molecule%08d'%idx)
                    idx+=1
                    for k, v in data.items():
                        subgrp[k] = v

                if args.yfile and yorgrad: fy.close()
                if args.ygradxyzfile and yorgrad: fgrad.close()

        if not available:
            stopper.stopMLatom('Please install all Python module required for TorchANI')
            
        args = Args()
        args.parse(argsANI)
        cls.setCore(args.nthreads)
        print('\n Converting data...\n\n')
        if subdatasets:
            if 'Subtrain'  in subdatasets:
                for i in subdatasets:
                    if i in ['Subtrain', 'Validate']: convert('ANI_'+i.lower()+'.h5', i.lower(),True)
                    else: convert('ANI_'+i.lower()+'.h5', i.lower())
            else:
                for i in subdatasets:
                    if i == 'Train': convert('ANI_'+i.lower()+'.h5', i.lower(),True)
                    else: convert('ANI_'+i.lower()+'.h5', i.lower())
        else:   convert('ANI.h5', '')

    @classmethod
    def createMLmodel(cls, argsANI, subdatasets):
        cls.load()
        args = Args()
        args.parse(argsANI)
        cls.setCore(args.nthreads)

        if not cls.dataConverted or args.learningcurve or args.cvtest or args.cvopt:
            cls.convertdata(argsANI, subdatasets)
            cls.dataConverted = True
        
        starttime = time.time()

        # if args.mlmodeltype.lower() in ['ani1x','ani1ccx','ani2x']:
        #     print('ML model already built in TorchANI')
        if False:
            pass
        else:
            TorchANI_train.train(args)

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

    @classmethod
    def useMLmodel(cls, argsANI, subdatasets):
        cls.load()
        args = Args()
        args.parse(argsANI)
        cls.setCore(args.nthreads)
        
        if args.mlmodeltype.lower() in ['ani1x', 'ani1ccx', 'ani2x']:
            cls.dataConverted = True
        if not cls.dataConverted and not args.learningcurve: 
            cls.convertdata(argsANI, subdatasets)
            cls.dataConverted = True
        # This is used for ASE
        if args.useMLmodel:
            cls.dataConverted = False
        if args.geomopt or args.freq or args.ts or args.irc :
            cls.dataConverted = False

        starttime = time.time()

        if args.mlmodeltype.lower() in ['ani1x', 'ani1ccx', 'ani2x']:
            TorchANI_predict.ani_predict(args)
        elif args.mlmodeltype.lower() == 'ani-tl':
            TorchANI_predict.tl_predict(args)
        else:
            TorchANI_predict.predict(args)
        
        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

def printHelp():
    helpText = __doc__ + '''
  To use Interface_ANI, please install TorchANI and its dependencies

  Arguments with their default values:
    MLprog=TorchANI            enables this interface
    MLmodelType=ANI            requests ANI model

    ani.batch_size=8           batch size
    ani.max_epochs=10000000    max epochs
    
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
    ani.Neuron_l1=160          number of neurons in layer 1
    ani.Neuron_l2=128          number of neurons in layer 2
    ani.Neuron_l3=96           number of neurons in layer 3
    ani.AF1='CELU'             acitivation function for layer 1
    ani.AF2='CELU'             acitivation function for layer 2
    ani.AF3='CELU'             acitivation function for layer 3

  Cite TorchANI:
    X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, A. E. Roitberg,
    J. Chem. Inf. Model. 2020, 60, 3408
    
  Cite ANI model:
    J. S. Smith, O. Isayev, A. E. Roitberg, Chem. Sci. 2017, 8, 3192
'''
    print(helpText)

if __name__ == '__main__':
    ANICls()
