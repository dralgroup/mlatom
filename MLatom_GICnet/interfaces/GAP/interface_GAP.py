#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_GAP: Interface between GAP and MLatom                           ! 
  ! Implementations by: Fuchun Ge                                             ! 
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
            'mlmodeltype'
            ],
            'gap-soap'
        )
        self.add_default_dict_args([
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin','setname'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodelout': "GAPmodel.xml",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat"
        })
        self.parse_input_content('gapfit.gap.type=soap')
        self.parse_input_content('gapfit.gap.n_sparse=1000000')

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        GAPCls.gapdict.update(self.gapfit.gap)
        if not self.MLmodelIn:
            with open(self.xyzfile,'r') as f:
                natom=int(f.readline())
                nline=1
                for line in f:
                    nline +=1
            if int(GAPCls.gapdict['n_sparse']) > nline/(natom+2):
                GAPCls.gapdict['n_sparse'] = str(6*int(nline/(natom+2)))
        gap = '{'+' '.join([str(k)+'='+str(v) for k,v in GAPCls.gapdict.items()])[5:]+'}'
        GAPCls.gapfitdict.update(self.gapfit)
        if self.mlmodelout[-4:]!='.xml':
            self.mlmodelout+='.xml'
        GAPCls.gapfitdict.update({
            'gp_file': self.mlmodelout, 
            'gap': gap,
        })
        with open(self.xyzfile,'r') as f:
            GAPCls.natom = int(f.readline())
            exec('f.readline()')
            GAPCls.atype = [f.readline().split()[0] for i in range(GAPCls.natom)]

        if not self.mlmodelin:
                self.mlmodelin = self.mlmodelout
        if 'default_sigma_e' in GAPCls.gapfitdict.keys(): 
            GAPCls.gapfitdict['default_sigma']=GAPCls.gapfitdict['default_sigma'].replace(
                GAPCls.gapfitdict['default_sigma'][1:-1].split(',')[0],
                GAPCls.gapfitdict['default_sigma_e']
            )
            GAPCls.gapfitdict.pop('default_sigma_e')
        if 'default_sigma_f' in GAPCls.gapfitdict.keys(): 
            GAPCls.gapfitdict['default_sigma']=GAPCls.gapfitdict['default_sigma'].replace(
                GAPCls.gapfitdict['default_sigma'][1:-1].split(',')[1],
                GAPCls.gapfitdict['default_sigma_f']
            )
            GAPCls.gapfitdict.pop('default_sigma_f')
        GAPCls.gapfitdict['default_sigma'] = " ".join(GAPCls.gapfitdict['default_sigma'].replace(',',', ').split())
        GAPCls.gapfitdict['gp_file']=self.mlmodelout
        if self.yfile: GAPCls.gapfitdict['energy_parameter_name']='energy'
        if self.ygradxyzfile: GAPCls.gapfitdict['force_parameter_name']='forces'
        if self.setname: GAPCls.gapfitdict['at_file']='GAP.exyz_'+self.setname
        else: GAPCls.gapfitdict['at_file']='GAP.exyz_train'
        GAPCls.gapfit          = [GAPbin]
        for k, v in GAPCls.gapfitdict.items():
            GAPCls.gapfit.append(k+'='+v)

class GAPCls(object):
    dataConverted = False
    gapdict = {
        'type': 'soap',
        'l_max': '6',
        'n_max': '6',
        'atom_sigma': '0.5',
        'zeta': '4',
        'cutoff': '6',
        'cutoff_transition_width': '0.5',
        'n_sparse': '1000000',
        'delta': '1',
        'covariance_type': 'dot_product',   
        'sparse_method': 'cur_points',
        'add_species': 'T'
    }
    # gap = '{'+' '.join([k+'='+v for k,v in args.gapdic.items()])[5:]+'}'
    gapfitdict = {
        'at_file':'GAP.exyz_train',
        'default_sigma':'{0.0005,0.001,0.1,0.1}', 
        'gp_file': '', 
        'gap': '', 
        'e0_method':'average',
        'sparse_separate_file': 'F'
    }
    gapfit = []
    natom = 0 
    atype = ''

    @classmethod
    def load(cls):
        loaded=False
        if not loaded:
            
            try:
                GAPbin = os.environ['gap_fit']
            except:
                print('please set $gap_fit')
            try:
                QUIPbin = os.environ['quip']
            except:
                print('please set $quip')
            globals()['QUIPbin'] = QUIPbin
            globals()['GAPbin'] = GAPbin
            loaded=True

        

    def __init__(self, args = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)

    @classmethod
    def convertdata(cls, GAPargs, subdatasets):
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
                    natom = int(line)
                    fo.write(line)
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    if args.yfile and yorgrad: fo.write('energy=%s '% fy.readline()[:-1])
                    fo.write('pbc="T T T" Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3')
                    if args.ygradxyzfile and yorgrad: fo.write(':forces:R:3')
                    fo.write('\n')
                    fxyz.readline()
                    if args.ygradxyzfile and yorgrad: fgrad.readline()

                    for i in range(natom):
                        fo.write(fxyz.readline()[:-1])
                        if args.ygradxyzfile and yorgrad: 
                            fo.write('%24.15f%24.15f%24.15f' % tuple(-1*float(i) for i in fgrad.readline().split()[-3:]))
                        fo.write('\n')
                if args.yfile and yorgrad: fy.close()
                if args.ygradxyzfile and yorgrad: fgrad.close()
        args = Args()
        args.parse(GAPargs)
        print('\n Converting data...\n\n')
        if subdatasets:
            # if 'Subtrain' in subdatasets :
            for i in subdatasets:
                if 'rain' in i : convert('GAP.exyz_'+i.lower(), i.lower(),True)
                else: convert('GAP.exyz_'+i.lower(), i.lower())
            # else:
            #     for i in subdatasets:
            #         if i == 'Train': convert('GAP.exyz_'+i.lower(), i.lower(),True)
            #         else: convert('GAP.exyz_'+i.lower(), i.lower())
        else:   convert('GAP.exyz', '')

    @classmethod
    def createMLmodel(cls, GAPargs, subdatasets):        
        cls.load()

        args = Args()
        args.parse(GAPargs)

        if not cls.dataConverted or args.learningcurve  or args.cvtest or args.cvopt:
            cls.convertdata(GAPargs, subdatasets)
            cls.dataConverted = True
        if args.useMLmodel:
            cls.dataConverted = False
        
        FNULL = open(os.devnull, 'w')

        os.system('rm '+args.mlmodelout+'* > /dev/null 2>&1 ')
        # call gap_fitr
        starttime = time.time()
        print('> '+' '.join(cls.gapfit))
        sys.stdout.flush()

        subprocess.call(cls.gapfit, stdout=FNULL, stderr=FNULL)
        # subprocess.call(cls.gapfit)
        FNULL.close()
        endtime = time.time()
        wallclock = endtime - starttime

        return wallclock

    @classmethod
    def useMLmodel(cls, GAPargs, subdatasets):
        cls.load()

        args = Args()

        if not cls.dataConverted and not args.learningcurve: 
            cls.convertdata(GAPargs, subdatasets)
            cls.dataConverted = True

        args.parse(GAPargs)
        if args.setname: args.setname='_'+args.setname
        
        starttime = time.time()

        p1 = subprocess.Popen([
            QUIPbin,
            'E=T',
            'F=T',
            'atoms_filename=GAP.exyz'+args.setname,
            'param_filename='+args.mlmodelin,
        ], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(['grep','AT'], stdin=p1.stdout, stdout=subprocess.PIPE)
        (stdout, stderr) = p2.communicate() 
        p1.stdout.close()
        with open('GAPest.exyz'+args.setname,'w') as f: 
            f.write(stdout.decode('ascii'))

        endtime = time.time()
        wallclock = endtime - starttime
        
        # convert results back to MLatom format
        with open('GAPest.exyz'+args.setname,'r') as f, open(args.yestfile,'w') as fy, open(args.ygradxyzestfile,'w') as fgrad:
            for i, line in enumerate(f):
                if i%(cls.natom+2) == 0:
                    fgrad.write(line[3:])
                elif i%(cls.natom+2) ==1:
                    fgrad.write('\n')
                    fy.write(line.split()[1].split('=')[1]+'\n')
                else:
                    fgrad.write('%20s%20s%20s\n' % tuple([-1*float(i) for i in line.split()[-3:]]))
        sys.stdout.flush()
        os.system('rm GAP.exyz'+args.setname+'.idx > /dev/null 2>&1 ')
        args.setname='train'
        return wallclock


# class args(object):
#     # MLatom args
#     learningcurve       = False
#     xyzfile             = ""
#     yfile               = ""
#     ygradxyzfile            = ""
#     virialfile          = ""
#     ntrain              = 0
#     ntest               = 0
#     nsubtrain           = 0
#     nvalidate           = 0
#     sampling            = "random"
#     itrainin            = ""
#     itestin             = ""
#     isubtrainin         = ""
#     ivalidatein         = ""
#     mlmodelout          = "GAPmodel.xml"
#     mlmodelin           = ""
#     yestfile            = "enest.dat"
#     ygradxyzestfile     = "gradest.dat"

#     # args for this interface
#     setname             = ''
#     natom               = 0
#     atype               = []
#     gapdic              = {
#         'type': 'soap',
#         'l_max': '6',
#         'n_max': '6',
#         'atom_sigma': '0.5',
#         'zeta': '4',
#         'cutoff': '6',
#         'cutoff_transition_width': '0.5',
#         'n_sparse': '1000000',
#         'delta': '1',
#         'covariance_type': 'dot_product',   
#         'sparse_method': 'cur_points',
#         'add_species': 'T'
#     }
#     gap                 = '{'+' '.join([k+'='+v for k,v in gapdic.items()])[5:]+'}'
#     gapfitdic           = {
#         'at_file':'GAP.exyz_train',
#         'default_sigma':'{0.0005,0.001,0.1,0.1}', 
#         'gp_file': mlmodelout, 
#         'gap': gap, 
#         'e0_method':'average',
#         'sparse_separate_file': 'F'
#     }
#     gapfit               = []

    # @classmethod
    # def parse(cls,argsraw):
    #     if len(argsraw) == 0:
    #         printHelp()
    #         stopper.stopMLatom('At least one option should be provided')
    #     for arg in argsraw:
    #         if (arg.lower() == 'help'
    #           or arg.lower() == '-help'
    #           or arg.lower() == '-h'
    #           or arg.lower() == '--help'):
    #             printHelp()
    #             stopper.stopMLatom('')
    #         elif arg.lower().split('=')[0] == 'gapfit':
    #             exec(arg)
    #             print(gapfit)
    #             test=gapfit['gap']
    #         elif arg.lower().split('.')[0] in ['gap-soap', 'gap', 'gap_fit', 'gapfit']:
    #             if arg.lower().split('.')[1]=="gap":
    #                 exec('cls.gapdic[arg.split(".")[2].split("=")[0]]=arg.split("=",1)[1]')
    #                 cls.gap = '{'+' '.join([k+'='+v for k,v in cls.gapdic.items()])[5:]+'}'
    #                 GAPCls.gapfitdict['gap']=cls.gap
    #             else:
    #                 exec('GAPCls.gapfitdict[arg.split(".")[1].split("=")[0]]=arg.split("=",1)[1]')   
    #         elif len(arg.lower().split('=')) == 1:                             # parse boolean args
    #             try:
    #                 exec('cls.'+arg.lower())
    #                 exec('cls.'+arg.lower()+'=True')
    #             except: pass
    #         else:                                               # parse other args
    #             try:
    #                 exec('cls.'+arg.split('=')[0].lower())
    #                 if type(eval('cls.'+arg.split('=')[0].lower())) == str :
    #                     exec('cls.'+arg.split('=')[0].lower()+'='+"arg.split('=')[1]")
    #                 else:
    #                     exec('cls.'+arg.split('=')[0].lower()+'='+arg.split('=')[1])
    #             except:
    #                 pass


    #     # calc. & fix something args that useful and you don't want users to change...
    #     with open(cls.xyzfile,'r') as f:
    #         cls.natom = int(f.readline())
    #         exec('f.readline()')
    #         cls.atype = [f.readline().split()[0] for i in range(cls.natom)]
        
    #     # for estAccMLmodel that MLmodelIn may not be provided
    #     if not cls.mlmodelin:
    #         cls.mlmodelin = cls.mlmodelout
    #     if 'default_sigma_e' in GAPCls.gapfitdict.keys(): 
    #         GAPCls.gapfitdict['default_sigma']=GAPCls.gapfitdict['default_sigma'].replace(
    #             GAPCls.gapfitdict['default_sigma'][1:-1].split(',')[0],
    #             GAPCls.gapfitdict['default_sigma_e']
    #         )
    #         GAPCls.gapfitdict.pop('default_sigma_e')
    #     if 'default_sigma_f' in GAPCls.gapfitdict.keys(): 
    #         GAPCls.gapfitdict['default_sigma']=GAPCls.gapfitdict['default_sigma'].replace(
    #             GAPCls.gapfitdict['default_sigma'][1:-1].split(',')[1],
    #             GAPCls.gapfitdict['default_sigma_f']
    #         )
    #         GAPCls.gapfitdict.pop('default_sigma_f')
    #     GAPCls.gapfitdict['default_sigma'] = " ".join(GAPCls.gapfitdict['default_sigma'].replace(',',', ').split())
    #     GAPCls.gapfitdict['gp_file']=cls.mlmodelout
    #     if cls.yfile: GAPCls.gapfitdict['energy_parameter_name']='energy'
    #     if cls.ygradxyzfile: GAPCls.gapfitdict['force_parameter_name']='forces'
    #     if cls.setname: GAPCls.gapfitdict['at_file']='GAP.exyz_'+cls.setname
    #     cls.gapfit          = [GAPbin]
    #     for k, v in GAPCls.gapfitdict.items():
    #         cls.gapfit.append(k+'='+v)
        
            
def printHelp():
    helpText = __doc__ + '''
  To use Interface_GAP, please define $gap_fit and $quip
  to corresponding executables
  
  Arguments with their default values:
    MLprog=GAP                 enables this interface
    MLmodelType=GAP-SOAP       requests GAP-SOAP model
    
    gapfit.xxx=x               xxx could be any option for gap_fit
                               Note that at_file and gp_file are not required
    gapfit.gap.xxx=x           xxx could be any option for gap
    gapfit.default_sigma={0.0005,0.001,0,0}
                               sigmas for energies, forces, virals, Hessians
    gapfit.e0_method=average   method for determining e0
    gapfit.gap.type=soap       descriptor type
    gapfit.gap.l_max=6         max number of angular basis functions
    gapfit.gap.n_max=6         max number of radial  basis functions
    gapfit.gap.atom_sigma=0.5  Gaussian smearing of atom density hyperparameter
                                                  
    gapfit.gap.zeta=4          hyperparameter for kernel sensitivity              
    gapfit.gap.cutoff=6.0      cutoff radius of local environment
    gapfit.gap.cutoff_transition_width=0.5  
                               cutoff transition width
    gapfit.gap.delta=1         hyperparameter delta for kernel scaling

  Cite GAP method:
    A. P. Bartok, M. C. Payne, R. Konor, G. Csanyi,
    Phys. Rev. Lett. 2010, 104, 136403
    
  Cite SOAP descriptor:
    A. P. Bartok,              R. Konor, G. Csanyi,
    Phys. Rev. B     2013,  87, 184115
'''
    print(helpText)

if __name__ == '__main__':
    GAPCls()
