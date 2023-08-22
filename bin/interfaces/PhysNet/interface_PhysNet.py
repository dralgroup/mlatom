#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_PhysNet: Interface between PhysNet and MLatom                   ! 
  ! Implementations by: Fuchun Ge and Max Pinheiro Jr                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os, sys, subprocess, time, shutil, re, math, random
import stopper
from args_class import ArgsBase

filedir = os.path.dirname(__file__)

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'learningCurve','useMLmodel','CVopt','CVtest'
            ],
            bool
        )
        self.add_default_dict_args([
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin','setname'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodeltype': 'PhysNet',
            'mlmodelout': "PhysNet",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat",
            'lcNtrains': [],
            'natom': 0,
            'atype': []
        })
        self.parse_input_content([
            'physnet.batch_sizes=0',
            'physnet.earlystopping=0',
            'physnet.threshold=0.0001',
            'physnet.patience=60',
            'physnet.restart=0',
            'physnet.num_features=128',
            'physnet.num_basis=64',
            'physnet.num_blocks=5',
            'physnet.num_residual_atomic=2',
            'physnet.num_residual_interaction=3',
            'physnet.num_residual_output=1',
            'physnet.cutoff=10.0',
            'physnet.use_electrostatic=0',
            'physnet.use_dispersion=0',
            'physnet.grimme_s6=0.5',
            'physnet.grimme_s8=0.2130',
            'physnet.grimme_a1=0.0',
            'physnet.grimme_a2=6.0519',
            'physnet.num_train=8',
            'physnet.num_valid=2',
            'physnet.seed=42',
            'physnet.max_steps=10000000',
            'physnet.learning_rate=0.0008',
            'physnet.max_norm=1000.0',
            'physnet.ema_decay=0.999',
            'physnet.keep_prob=1.0',
            'physnet.l2lambda=0.0',
            'physnet.nhlambda=0.01',
            'physnet.decay_steps=10000000',
            'physnet.decay_rate=0.1',
            'physnet.batch_size=12',
            'physnet.valid_batch_size=2',
            'physnet.force_weight=52.91772105638412',
            'physnet.charge_weight=0',
            'physnet.dipole_weight=0',
            'physnet.summary_interval=0',
            'physnet.validation_interval=0',
            'physnet.save_interval=0',
            'physnet.record_run_metadata=0'
            ])
    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        with open(self.xyzfile,'r') as f:
            self.natom = int(f.readline())
            exec('f.readline()')
            self.atype = [f.readline().split()[0] for i in range(self.natom)]
        if not self.mlmodelin:
            self.mlmodelin = self.mlmodelout
        if not self.ygradxyzfile:
            self.physnet.force_weight=0
        
        if self.learningcurve:
            self.lcNtrains = [int(i) for i in str(self.lcNtrains).split(',')]
            if self.physnet.batch_sizes:
                self.physnet.batch_sizes = [int(i) for i in self.lcNtrains.split(',')]
                self.physnet.batch_size = self.physnet.batch_sizes[self.lcNtrains.index(self.ntrain)]
        if not self.physnet.summary_interval:
            try:
                self.physnet.summary_interval=int(0.8*self.ntrain//self.physnet.batch_size)
            except:
                self.physnet.summary_interval=10
        if not self.physnet.validation_interval:
            try:
                self.physnet.validation_interval=int(0.8*self.ntrain//self.physnet.batch_size)
            except:
                self.physnet.validation_interval=10
        if not self.physnet.save_interval:
            try:
                self.physnet.save_interval=int(0.8*self.ntrain//self.physnet.batch_size)
            except:
                self.physnet.save_interval=10

class PhysNetCls(object):
    dataConverted = False

    @classmethod
    def load(cls):
        loaded=False
        if not loaded:
            
            sp_z = {'X': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uut': 113, 'Fl': 114, 'Uup': 115, 'Lv': 116, 'Uus': 117, 'Uuo': 118}

            z_sp = {v:k for k,v in sp_z.items()}
            try:
                from . import PhysNet_train
                from . import PhysNet_predict
            except:
                stopper.stopMLatom('Please specify PhysNet installation dir in $PhysNet')

            globals()['z_sp'] = z_sp
            globals()['sp_z'] = sp_z
            globals()['PhysNet_train'] = PhysNet_train
            globals()['PhysNet_predict'] = PhysNet_predict
            loaded=True


    def __init__(self, argsPhysNet = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)
    
    @classmethod
    def convertdata(cls, argsPhysNet, subdatasets):
        cls.load()
        def convert(fileout, setname, yorgrad=False):
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
            with open(coordfile ,'r') as fxyz:
                natom = int(fxyz.readline())
                fxyz.seek(0)
                data={}
                data['R'] = np.array([])
                data['Z'] = np.array([])
                data['N'] = np.array([])
                if args.yfile and yorgrad: 
                    fy =  open(yfile,'r')
                    data['E'] = np.array([])
                if args.ygradxyzfile and yorgrad: 
                    fgrad = open(gradfile,'r')
                    data['F'] = np.array([])
                for line in fxyz:
                    # 1st line
                    data['N'] = np.append(data['N'], int(line)).astype('int')
                    if args.ygradxyzfile and yorgrad: fgrad.readline()
                    if args.yfile and yorgrad: data['E'] =np.append(data['E'], float(fy.readline()))
                    # 2nd line
                    fxyz.readline()
                    if args.ygradxyzfile and yorgrad: fgrad.readline()
                    # rest lines
                    for i in range(natom):
                        ln = fxyz.readline()
                        data['Z'] = np.append(data['Z'], sp_z[ln.split()[0].capitalize()])
                        data['R'] = np.append(data['R'], [float(i) for i in ln.split()[-3:]])
                        if args.ygradxyzfile and yorgrad: data['F'] = np.append(data['F'], [-1*float(i) for i in fgrad.readline().split()[-3:]]) 

                if args.yfile and yorgrad: fy.close()
                if args.ygradxyzfile and yorgrad: fgrad.close()
                data['R'] = data['R'].reshape(-1,natom,3)
                data['Z'] = data['Z'].reshape(-1,natom).astype('int')
                if args.ygradxyzfile and yorgrad: data['F'] = data['F'].reshape(-1,natom,3)
                
                np.savez(fileout, **data)

            
        args = Args()
        args.parse(argsPhysNet)
        print('\n Converting data...\n\n')
        if subdatasets:
            if 'Subtrain'  in subdatasets:
                for i in subdatasets:
                    if i in ['Subtrain', 'Validate']: convert('PhysNet_'+i.lower()+'.npz', i.lower(),True)
                    else: convert('PhysNet_'+i.lower()+'.npz', i.lower())
            else:
                for i in subdatasets:
                    if i == 'Train': convert('PhysNet_'+i.lower()+'.npz', i.lower(),True)
                    else: convert('PhysNet_'+i.lower()+'.npz', i.lower())
        else:   convert('PhysNet.npz', '')

    @classmethod
    def createMLmodel(cls, argsPhysNet, subdatasets):
        cls.load()
        PNargs = Args()
        PNargs.parse(argsPhysNet)

        if not cls.dataConverted or PNargs.learningcurve  or PNargs.cvtest or PNargs.cvopt:
            cls.convertdata(argsPhysNet, subdatasets)
            cls.dataConverted = True
        
        starttime = time.time()

        PhysNet_train.train(PNargs)

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock
        
    @classmethod
    def useMLmodel(cls, argsPhysNet, subdatasets):
        cls.load()
        PNargs = Args()
        PNargs.parse(argsPhysNet)

        if not cls.dataConverted and not PNargs.learningcurve: 
            cls.convertdata(argsPhysNet, subdatasets)
            cls.dataConverted = True
        if PNargs.useMLmodel:
            cls.dataConverted = False
        starttime = time.time()
        PhysNet_predict.predict(PNargs)
        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

def printHelp():
    helpText = __doc__ + '''
  To use Interface_PhysNet, please define $PhysNet to where PhysNet is located

  Arguments with their default values:
    MLprog=PhysNet             enables this interface
    MLmodelType=PhysNet        requests PhysNet model
    
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
    physnet.summary_interval=5 interval for summary
    physnet.validation_interval=5 
                               interval for validation
    physnet.save_interval=10   interval for model saving

  Cite PhysNet:
    O. T. Unke, M. Meuwly, J. Chem. Theory Comput. 2019, 15, 3678
'''
    print(helpText)

if __name__ == '__main__':
    PhysNetCls()
