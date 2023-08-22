#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  ! Interface_DeePMDkit: Interface between DeePMD-kit and MLatom              ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os, sys, subprocess, time, shutil, re, math, random, json
import stopper
from args_class import ArgsBase
from args_class import AttributeDict

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
            'xyzfile', 'yfile', 'ygradxyzfile','itrainin','itestin','isubtrainin','ivalidatein','mlmodelin'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodeltype': 'DeepPot-SE',
            'mlmodelout': "graph.pb",
            'sampling': "random",
            'yestfile': "enest.dat",
            'ygradxyzestfile': "gradest.dat",
            'lcNtrains': [],
            'natom': 0,
            'atype': []
        })
        self.parse_input_content([
            'deepmd.earlystopping.threshold=0.0001',
            'deepmd.earlystopping.patience=60',
            'deepmd.earlystopping.enable=0',
            'deepmd.earlystopping.loss_id=1',
            'deepmd.input=%s/template.json'%filedir
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

        with open(self.deepmd.input, 'r') as f:
            DeePMDCls.deepmdarg= AttributeDict.dict_to_attributedict(json.load(f))
        DeePMDCls.deepmdarg.merge_dict(DeePMDCls.deepmdarg,self.deepmd.data)
        DeePMDCls.deepmdarg=iterdict(AttributeDict.normal_dict(DeePMDCls.deepmdarg))
        
        if self.learningcurve:
            self.lcNtrains = [int(i) for i in str(self.lcNtrains).split(',')]
            if self.deepmd.batch_sizes:
                self.deepmd.batch_sizes = [int(i) for i in self.lcNtrains.split(',')]
                DeePMDCls.deepmdarg = self.deepmd.batch_sizes[self.lcNtrains.index(self.ntrain)]

        if self.mlmodeltype.lower() in ['dpmd']:
            DeePMDCls.deepmdarg['model']['descriptor']['type'] = 'loc_frame'

        # if self.deepmd.earlystopping.enable:
        #     DeePMDCls.deepmdarg['training']['disp_freq']=int(self.Ntrain*0.8/DeePMDCls.deepmdarg['training']['batch_size'][0])
        #     DeePMDCls.deepmdarg['training']['save_freq']=10*DeePMDCls.deepmdarg['training']['disp_freq']
        DeePMDCls.deepmdarg['training']['set_prefix']='set'
        DeePMDCls.deepmdarg['training']['systems']=['./']
        DeePMDCls.deepmdarg["model"]["type_map"] = sorted(set(self.atype), key=lambda x: self.atype.index(x))
        if 'decay_rate' in DeePMDCls.deepmdarg['learning_rate'].keys(): 
            DeePMDCls.deepmdarg['learning_rate']['stop_lr']=DeePMDCls.deepmdarg['learning_rate']['start_lr']*DeePMDCls.deepmdarg['learning_rate']['decay_rate']**(DeePMDCls.deepmdarg['training']['stop_batch']//DeePMDCls.deepmdarg['learning_rate']['decay_steps'])
        if DeePMDCls.deepmdarg['model']['descriptor']['type'] != 'loc_frame':
            DeePMDCls.deepmdarg['model']['descriptor']['sel'] = [self.natom+1]*len(DeePMDCls.deepmdarg["model"]["type_map"])
        else:
            DeePMDCls.deepmdarg['model']['descriptor']['sel_a'] = [self.natom+1]*len(DeePMDCls.deepmdarg["model"]["type_map"])
            DeePMDCls.deepmdarg['model']['descriptor']['sel_r'] = [self.natom+1]*len(DeePMDCls.deepmdarg["model"]["type_map"])
            try:
                del DeePMDCls.deepmdarg['model']['descriptor']['sel'] 
            except: pass

        if not self.ygradxyzfile:
            DeePMDCls.deepmdarg['loss']['limit_pref_f']=0
            DeePMDCls.deepmdarg['loss']['start_pref_f']=0
        if not self.yfile:
            DeePMDCls.deepmdarg['loss']['limit_pref_e']=0
            DeePMDCls.deepmdarg['loss']['start_pref_e']=0
        
        intlist={'learning_rate':['decay_steps'],'training':['numb_test','batch_size','disp_freq','save_freq','stop_batch'],'model':{'fitting_net':['neuron'],'descriptor':['sel','sel_a','sel_r']}}


        def float2int(d1: dict, d2: dict):
            for k, v in d1.items():
                if k in d2.keys():
                    if type(d2[k]) == list:
                        for i in d2[k]:
                            try:
                                if type(d1[k][i]) == list:
                                    d1[k][i] = [int(j) for j in d1[k][i]]
                                elif type(d1[k][i]) == float:
                                    d1[k][i] = int(d1[k][i])
                            except:
                                pass
                    elif type(d2[k]) == dict:
                        float2int(d1[k],d2[k])
        float2int(DeePMDCls.deepmdarg,intlist)
        




class DeePMDCls(object):
    deepmdarg=AttributeDict()
    @classmethod
    def load(cls):
        loaded=False
        if not loaded:
            try:
                DeePMDdir = os.environ['DeePMDkit']
            except:
                print('please set $DeePMDkit')
            DeePMDbin = DeePMDdir+'/dp'
            
            globals()['DeePMDdir'] = DeePMDdir
            globals()['DeePMDbin'] = DeePMDbin

            loaded=True
    def __init__(self, argsDeePMD = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)

    @classmethod
    def createMLmodel(cls, argsDeePMD, subdatasets):
        cls.load()
        args=Args()
        args.parse(argsDeePMD)
        if args.deepmd.earlystopping.enable:
            earlyStop=earlyStopCls(patience = args.deepmd.earlystopping.patience, threshold=args.deepmd.earlystopping.threshold)
        # data conversion
        print('\n Converting data...\n\n')
        prefix = ''
        if args.learningcurve: prefix = '../'
        cls.convertdata('coord', 'subtrain', prefix+'xyz.dat_subtrain',args)
        cls.convertdata('coord', 'validate', prefix+'xyz.dat_validate',args)
        if args.yfile:
            cls.convertdata('en', 'subtrain', prefix+'y.dat_subtrain',args)
            cls.convertdata('en', 'validate', prefix+'y.dat_validate',args)
        if args.ygradxyzfile:
            cls.convertdata('force', 'subtrain', prefix+'grad.dat_subtrain',args)
            cls.convertdata('force', 'validate', prefix+'grad.dat_validate',args)
        # write dp input json
        with open("deepmdargs.json","w") as f:
            json.dump(cls.deepmdarg, f, indent=4)
        
        if os.path.exists(cls.deepmdarg['training']['disp_file']):
            os.system('rm '+ cls.deepmdarg['training']['disp_file']+' '+args.mlmodelout)
        
        # run dp train
        FNULL = open(os.devnull, 'w')
        starttime = time.time()
        print('\n\n> %s train deepmdargs.json' % DeePMDbin)
        sys.stdout.flush()

        proc = subprocess.Popen([DeePMDbin,"train","deepmdargs.json"], stdout=subprocess.PIPE ,stderr=FNULL, universal_newlines=True)
        # for line in iter(proc.stdout.readline, b''):
            # print(line.decode('ascii').replace('\n',''))
        for line in proc.stdout:
            print(line.replace('\n',''))
            if args.deepmd.earlystopping.enable:
                try:
                    lastline=subprocess.check_output(['tail', '-1', cls.deepmdarg['training']['disp_file']],stderr=FNULL).split()
                    loss = float(lastline[int(args.deepmd.earlystopping.loss_id)])
                    nbatch = int(lastline[0])
                    sys.stdout.flush()
                    if earlyStop.current(nbatch, loss):
                        if nbatch%cls.deepmdarg['training']['save_freq'] != 0:
                            print('met early-stopping conditions')
                            proc.terminate()
                except:
                        pass
            sys.stdout.flush()
        # proc.stdout.close()

        print('\n ___________________________________________________________\n\n Learning curve:\n')
        sys.stdout.flush()
        os.system('cat '+cls.deepmdarg['training']['disp_file'])
        sys.stdout.flush()
        # save MLmodel
        print('\n\n> %s freeze -o %s' % (DeePMDbin, args.mlmodelout))
        sys.stdout.flush()
        subprocess.call([DeePMDbin,"freeze","-o",args.mlmodelout],stdout=FNULL,stderr=FNULL)
        if os.path.exists(args.mlmodelout):
            print('model saved in %s' % args.mlmodelout)
        FNULL.close()

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

    @classmethod
    def useMLmodel(cls, argsDeePMD, subdatasets):
        cls.load()
        args=Args()
        args.parse(argsDeePMD)
        cls.convertdata('coord', 'test', args.xyzfile, args)
        
        starttime = time.time()

        FNULL = open(os.devnull, 'w')
        subprocess.call([DeePMDdir+"/python", filedir+"/DP_inference.py", 'testset/coord.npy', args.mlmodelin,  args.yestfile, args.ygradxyzestfile],stdout=FNULL , stderr=FNULL)
        FNULL.close()
        
        # if os.path.exists(args.yestfile) and os.path.exists(args.ygradxyzestfile):
        #     print('estimated values saved in '+args.yestfile+' and '+args.ygradxyzestfile)

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock
        
    @classmethod
    def convertdata(cls, datatype, dataset, filein, args):
        
        if dataset=='subtrain':
            outdir = 'set.000'
        elif dataset=='validate':
            outdir = 'set.001'
        elif dataset=='test':
            outdir = 'testset'
        
        if not os.path.isdir(outdir):
            os.system('mkdir '+outdir)

        # convert to coord.npy box.npy type.raw
        if datatype=='coord':
            dic = {cls.deepmdarg["model"]["type_map"][i]:i for i in range(len(cls.deepmdarg["model"]["type_map"]))}
            index = [dic[i] for i in args.atype]
            with open('type.raw','w') as ff:
                for i in index:
                    ff.writelines("%d " % i)

            with open(filein,'r') as fi:
                data = np.array([])
                for i, line in enumerate(fi):
                    if i%(args.natom+2) > 1:  
                        data = np.append(data,np.array(line.split()[-3:]).astype('float'))
                data = data.reshape(-1,3*args.natom)

            with open(outdir+'/coord.npy','wb') as fo:
                np.save(fo, data)

            with open(outdir+'/box.npy','wb') as fo:
                np.save(fo, np.repeat(np.diag(64 * np.ones(3)).reshape([1, -1]),len(data), axis=0))

        # convert to force.npy
        elif datatype=='force':
            with open(filein,'r') as fi:
                data = np.array([])
                for i, line in enumerate(fi):
                    if i%(args.natom+2) > 1:  
                        data = np.append(data,-1*np.array(line.split()[-3:]).astype('float'))
                data = data.reshape(-1,3*args.natom)

            with open(outdir+'/force.npy','wb') as fo:
                np.save(fo, data)

        # conver to energy.npy
        elif datatype=='en':
            with open(filein,'r') as fi:
                data = np.array([])
                for line in fi: 
                    data = np.append(data,np.array(line).astype('float'))

            with open(outdir+'/energy.npy','wb') as fo:
                np.save(fo, data)    
# home-made early stopping for dp
class earlyStopCls():
    def __init__(self, patience = 60, threshold = 0.0001 ):
        self.bestloss=1000
        self.bestbatch=0
        self.patience=patience
        self.threshold=threshold
    
    def current(self, nbatch, loss):
        stop = False
        if loss < (1 - self.threshold)*self.bestloss:
            self.bestloss = loss
            self.bestbatch = nbatch
        counter=int((nbatch - self.bestbatch)/DeePMDCls.deepmdarg['training']['disp_freq'])
        print('bestloss: %s    bestbatch: %s    early-stopping counter: %s/%s'% (self.bestloss, self.bestbatch, counter,self.patience))
        if counter > self.patience:
            stop = True
        return stop

# function for modifying args in dp input json
def mapdict(dic,arg):
    try:
        key,value=arg.split('.',1)[1].split("=")
        subdic=dic
        for i in key.split(".")[:-1]:
            subdic = subdic[i.lower()]
        #exec 'subdic[key.split(".")[-1].lower()]'
        subdic[key.split(".")[-1].lower()]=json.loads(value)
    except:
        pass

def iterdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        else:
            if type(v) == str:
                if v[0]=='[' and v[-1]==']':
                    v=json.loads(v)
            d.update({k: v})
    return d

def printHelp():
    helpText = __doc__ + '''
  To use Interface_DeePMDkit, please define environmental variable $DeePMDkit
  to where dp binary is located (e.g "/home/xxx/deepmd-kit-1.2/bin").

  Arguments with their default values:
    MLprog=DeePMD-kit          enables this interface
    MLmodelType=S              requests model S
      DeepPot-SE               [defaut]
      DPMD
      
    deepmd.xxx.xxx=X           specify arguments for DeePMD,
                               follows DeePMD-kit's json input file structure
      deepmd.training.stop_batch=4000000        
                               number of batches to be trained before stopping       
      deepmd.training.batch_size=32 
                               size of each batch
      deepmd.learning_rate.start_lr=0.001
                               initial learning rate
      deepmd.learning_rate.decay_steps=4000
                               number of batches for one decay
      deepmd.learning_rate.decay_rate=0.95
                               decay rate of each decay 
      deepmd.model.descriptor.rcut=6.0        
                               cutoff radius for local environment
      deepmd.model.fitting_net.neuron=80,80,80
                               NN structure of fitting network
        
    deepmd.input=S             file S with DeePMD input parameters
                               in json format (as a template)

  Cite DeePMD-kit:
    H. Wang, L. Zhang, J. Han, W. E, Comput. Phys. Commun. 2018, 228, 178
    
  Cite DeepPot-SE method, if you use it:
    L.F. Zhang, J.Q. Han, H. Wang, W.A. Saidi, R. Car, W.N. E,
    Adv. Neural. Inf. Process. Syst. 2018, 31, 4436
    
  Cite DPMD method, if you use it:
    L. Hang, J. Han, H. Wang, R. Car, W. E, Phys. Rev. Lett. 2018, 120, 143001
'''
    print(helpText)

if __name__ == '__main__':
    DeePMDCls()
