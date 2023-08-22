from args_class import ArgsBase 
import numpy as np 
from functions3D import *
import stopper

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.args2pass = [] 
        self.add_default_dict_args([ 

            ],
            ""
        )

        self.add_dict_args({
            'gaukw':'',
            'spin':1,
            'charge':0,
            'XYZfile':'' ,
            'nthreads':1,
            'link0': 0,
        })

    def parse(self,argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['',None])

class GaussianCls(object):
    #def __init__(self,argsGaussian):
    #    args = Args() 
    #    args.parse(argsGaussian)

    @classmethod 
    def calculate(cls,argsGaussian):
        args = Args() 
        args.parse(argsGaussian)
        if args.link0 == 0:
            link0 = False 
        else: 
            link0 = True

        Gaussianbin, version = cls.check_gaussian()
        gaukw = args.gaukw.split('_')
        gaukw = ' '.join(gaukw)


        atoms,coords = readXYZs(args.XYZfile)
        Nmol = len(atoms)

        for imol in range(Nmol):
            fname = str(imol).zfill(6)
            writeCom(args.nthreads,gaukw,fname,args.charge,args.spin,atoms[imol],coords[imol],link0)
            cls.run_gaussian(fname,Gaussianbin)


    @classmethod
    def check_gaussian(cls):
        status = os.popen('echo $GAUSS_EXEDIR').read().strip() 
        if len(status) != 0:
            Gaussianroot = status.split('bsd')[0]
            if 'g16' in Gaussianroot:
                version = 'g16'
            elif 'g09' in Gaussianroot:
                version='g09'
            Gaussianbin = Gaussianroot + version 
        else:
             stopper.stopMLatom('Can not find Gaussian software in the environment variable, $GAUSS_EXEDIR variable does not exist')
        version = version.replace('g','')
        return Gaussianbin, version

    @classmethod 
    def run_gaussian(cls,com,Gaussianbin):
        fcom = com+'.com'
        os.system(Gaussianbin + ' '+ com)
