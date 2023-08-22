#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_sGDML: Interface between sGDML and MLatom                       ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''
from typing import Tuple
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
            'mlmodeltype'
            ],
            'sgdml'
        )
        self.add_default_dict_args([
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin','setname'
            ],
            ""
        )        
        self.add_dict_args({
            'nthreads': 1,
            'mlmodelout': "sGDML",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat"
        })
        self.parse_input_content([
            'sGDML.torch=false ',
            'sGDML.gdml=false ',
            'sGDML.cprsn=false ',
            'sGDML.no_E=false ',
            'sGDML.s=false'
        ])

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        with open(self.xyzfile,'r') as f:
            sGDMLCls.natom = int(f.readline())
            exec('f.readline()')
            sGDMLCls.atype = [f.readline().split()[0] for i in range(sGDMLCls.natom)]
        
        # for estAccMLmodel that MLmodelIn may not be provided
        if self.mlmodelout[-4:] != '.npz': self.mlmodelout += '.npz'
        if not self.mlmodelin:
            self.mlmodelin = self.mlmodelout

        if self.sGDML.s:
            sigmas = self.sGDML.s.split(',')
            self.sGDML.s = []
            for i in range(len(sigmas)):
                self.sGDML.s.append(sigmas[i])
            self.sGDML.s.reverse()
        if self.mlmodeltype.lower()=='gdml':
            self.sGDML.gdml = True

            

class sGDMLCls(object):
    setname = ''
    atype = ''
    natom = 0
    nsubtrain = 0
    nvalidate = 0
    dataConverted = False
    @classmethod
    def load(cls):
        loaded=False
        if not loaded:
            try:
                sGDMLbin = os.environ['sGDML']
                from . import sgdml_dataset_from_extxyz
                from sgdml.predict import GDMLPredict
                from sgdml.utils import io
                globals()['sGDMLbin'] = sGDMLbin
                globals()['sgdml_dataset_from_extxyz'] = sgdml_dataset_from_extxyz
                globals()['GDMLPredict'] = GDMLPredict
                globals()['io'] = io
            except:
                stopper.stopMLatom('Please specify sGDML bin in $sGDML')
            loaded=True

    def __init__(self, argssGDML = sys.argv[1:]):
            print(' ___________________________________________________________\n\n%s' % __doc__)
    
    @classmethod
    def convertdata(cls, argssGDML, subdatasets):
        cls.load()
        def convert(fileout, setname, yorgrad=False):
            # convert data to .exyz format
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
            with open(fileout, 'w') as fo, open(coordfile ,'r') as fxyz:
                if args.yfile and yorgrad: fy =  open(yfile,'r')
                if args.ygradxyzfile and yorgrad: fgrad = open(gradfile,'r')
                for line in fxyz:
                    if setname == 'subtrain': cls.nsubtrain += 1
                    if setname == 'validate': cls.nvalidate += 1
                    natom = int(line)
                    fo.write(line)
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    if args.yfile and yorgrad: fo.write('%s '% fy.readline()[:-1])
                    fo.write('\n')
                    fxyz.readline()
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    for i in range(natom):
                        fo.write(fxyz.readline()[:-1])
                        if args.ygradxyzfile and yorgrad: 
                            fo.write('%24.15f %24.15f %24.15f' % tuple(-1*float(i) for i in fgrad.readline().split()[-3:]))
                        fo.write('\n')
                if args.yfile and yorgrad: fy.close()
                if args.ygradxyzfile and yorgrad: fgrad.close()
                
            if yorgrad: sgdml_dataset_from_extxyz.convert(fileout)

        args = Args()
        args.parse(argssGDML)
        print('\n Converting data...\n\n')
        if subdatasets:
            if 'Subtrain' in subdatasets:
                for i in subdatasets:
                    if i in ['Subtrain', 'Validate']: convert('sGDML_'+i.lower()+'.xyz', i.lower(),True)
                    else: convert('sGDML_'+i.lower()+'.xyz', i.lower())
            else:
                for i in subdatasets:
                    if i == 'Train': convert('sGDML_'+i.lower()+'.xyz', i.lower(),True)
                    else: convert('sGDML_'+i.lower()+'.xyz', i.lower())
        else:   convert('sGDML.xyz', '')

    @classmethod
    def createMLmodel(cls, argssGDML, subdatasets):
        cls.load()
        args = Args()
        args.parse(argssGDML)
        
        if not cls.dataConverted or args.learningcurve or args.cvtest or args.cvopt:
            cls.convertdata(argssGDML, subdatasets)
            cls.dataConverted = True

        cmds = [
            [sGDMLbin, 'create', '-p', str(args.nthreads), '-v', 'sGDML_validate.npz', '--task_dir', 'sGDMLtask', 'sGDML_subtrain.npz', str(cls.nsubtrain), str(cls.nvalidate)],
            [sGDMLbin, 'train', '-p', str(args.nthreads), 'sGDMLtask'],
            [sGDMLbin, 'validate', '-p', str(args.nthreads), 'sGDMLtask', 'sGDML_validate.npz'], 
            [sGDMLbin, 'select', '-p', str(args.nthreads), '--model_file', args.mlmodelout, 'sGDMLtask'], 
            [sGDMLbin, 'show', '-o', '-p', str(args.nthreads), args.mlmodelout]
        ]
        
        if args.sGDML.E_cstr: cmds[0].insert(6, '--E_cstr')
        if args.sGDML.no_E: cmds[0].insert(6, '--no_E')
        if args.sGDML.cprsn: cmds[0].insert(6, '--cprsn')
        if args.sGDML.gdml: cmds[0].insert(6, '--gdml')
        if args.sGDML.s: 
            for i in args.sGDML.s:
                cmds[0].insert(6, i)
            cmds[0].insert(6, '-s')
        if args.sGDML.torch: 
            cmds[0].insert(4, '--torch')
            cmds[1].insert(4, '--torch')
            cmds[2].insert(4, '--torch')
            cmds[3].insert(4, '--torch')
        
        cls.nsubtrain = 0
        cls.nvalidate = 0

        FNULL = open(os.devnull, 'w')
        
        starttime = time.time()

        for cmd in cmds:
            print('> sgdml '+' '.join(cmd[1:]))
            sys.stdout.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE ,stderr=subprocess.PIPE)
            for line in iter(proc.stdout.readline, b''):
                print(line.decode('ascii').replace('\n',''))
                sys.stdout.flush()
            proc.stdout.close()  
        FNULL.close()     

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock


    @classmethod
    def useMLmodel(cls, argssGDML, subdatasets):
        cls.load()
        args = Args()
        args.parse(argssGDML)

        if not cls.dataConverted and not args.learningcurve: 
            cls.convertdata(argssGDML, subdatasets)
            cls.dataConverted = True
        if args.useMLmodel:
            cls.dataConverted = False


        if args.setname: args.setname='_'+args.setname

        starttime = time.time()

        model = np.load(args.mlmodelin)
        gdml = GDMLPredict(model)

        r,_ = io.read_xyz('sGDML'+args.setname+'.xyz') # 9 atoms
        e,f = gdml.predict(r)

        line="%d\n\n" % int(cls.natom)
        with open(args.yestfile,'wb') as ff:
            np.savetxt(ff, e, fmt='%12.6f', delimiter=" ")
        with open(args.ygradxyzestfile,'wb') as ff:
            for force in f:
                ff.write(line.encode('utf-8'))
                np.savetxt(ff, -1*force.reshape(-1,3), fmt='%12.6f', delimiter=" ")
                
        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock
def printHelp():
    helpText = __doc__ + '''
  To use Interface_sGDML, please set enviromental variable $sGDML
  to the path of the sGDML executable

  Arguments with their default values:
    MLprog=sGDML               enables this interface
    MLmodelType=S              requests model S
      sGDML                    [defaut]
      GDML
    sgdml.gdml=False           use GDML instead of sGDML
    sgdml.cprsn=False          compress kernel matrix along symmetric 
                               degrees of freedom
    sgdml.no_E=False           do not predict energies
    sgdml.E_cstr=False         include the energy constraints in the
                               kernel
    sgdml.s=<s1>[,<s2>[,...]]  set hyperparameter sigma
           =<start>:[<step>:]<stop>     
 
  Cite sGDML program:
    S. Chmiela, H. E. Sauceda, I. Poltavsky, K.-R. Müller, A. Tkatchenko,
    ‎Comput. Phys. Commun. 2019, 240, 38

  Cite GDML method, if you use it:
    S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K. T. Schütt,
    K.-R. Müller, Sci. Adv. 2017, 3, e1603015
    
  Cite sGDML method, if you use it:
    S. Chmiela, H. E. Sauceda, K.-R. Müller, A. Tkatchenko,
    Nat. Commun. 2018, 9, 3887
    
'''
    print(helpText)

if __name__ == '__main__':
    sGDMLCls()
