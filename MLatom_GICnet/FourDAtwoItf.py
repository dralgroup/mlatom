from gettext import find
from utils import *
from functions4Dtf import *
import sys
import time
import json
from args_class import ArgsBase
from pyh5md import File, element
import stopper

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'create4Dmodel','use4Dmodel','estAcc4Dmodel','MLmodelIn','reactionPath','MD','activeLearning'
            ],
            bool
        )
        self.add_default_dict_args([
            'trajsList',
            'trajEpot',
            'ICidx',
            'mlmodelin',
            'mlmodelsin'
            ],
            ""
        )        
        self.add_dict_args({
            'mlmodelout': "4D_model",
            'nthreads': None,
            'initXYZ':'',
            'initVXYZ':'',
            'trun':1000,
            'finalXYZ':'',
            'initVout':'',
            'reactionTime':10,
            'trajXYZout':'traj.xyz',
            'trajVXYZout':'traj.vxyz',
            'trajTime':'traj.t',
            'trajH5MDout':'traj.h5',
            'trajEpot':'traj.epot',
            'trajEkin':'traj.ekin',
            'trajEtot':'traj.etot',
            'trajDescr':'traj.y',
            'tc':10,
            'tSegm':'tc',
            'dt':'tc',
            'Trajs':[]
        })
        self.parse_input_content([
            "FourD.reOrient=0",
            "FourD.ICdict=0",
            'FourD.tmax=0',
            'FourD.use3D=1',
            'FourD.reuseData=0',
            'FourD.reuse3D=0',
            'FourD.reuse4D=0',
            'FourD.tc=0',
            'Nsubtrain=0.9',
            'Nvalidate=0.05',
            'NNvalidate=0',
            'NNsubtrain=0',
            'FourD.initFreq=tc',
            'FourD.batchSize3D=16',
            'FourD.batchSize4D=16',
            'FourD.maxEpoch3D=4096',
            'FourD.maxEpoch4D=4096',
            'FourD.MD.subDirOut=4DMD',
            'FourD.reactionPath.subDirOut=4Dreaction',
            "FourD.m2vbar=0",
            "FourD.m2dm=0",
            "FourD.logisticateT=0",
            "FourD.normalizeT=1",
            'FourD.Descriptor=ic',
            'FourD.Descriptor3D=0',
            'FourD.xList=Descriptor',
            'FourD.yList=Descriptor,ep',
            'FourD.monarchWings=0',
            'FourD.adaptSeg=0',
            'FourD.forceEtot=0',
            'FourD.learnSeg=0',
            'FourD.checkEk=0',
            'FourD.checkV0=0',
            'FourD.ekbias=0',
            'FourD.epbias=0',
            ])
        self.meta={
            "descriptor":{},
            "data":{},
            "3Dmodel":{},
            "4Dmodel":{}
        }
    def parse(self, argsraw):
        self.args2pass=argsraw
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        self.FourD.ICdict={}
        if self.MLmodelsIn:
            self.MLmodelIn=self.MLmodelsIn.split(',')[0]
        if self.MLmodelIn and not self.activeLearning:
            with open(f'{self.MLmodelIn}/meta.json') as f: self.meta=json.load(f)
            self.FourD.tc=self.meta['4Dmodel']['Tc']
            self.FourD.Descriptor=self.meta['descriptor']['type']
            self.FourD.ICdict=self.meta['descriptor']['ICdict']
            self.FourD.M2dM=self.meta['descriptor']['M2dM']
            self.FourD.M2Vbar=self.meta['descriptor']['M2Vbar']
            self.FourD.logisticateT=self.meta['descriptor']['logisticateT']
            self.FourD.normalizeT=self.meta['descriptor']['normalizeT']
            try:
                self.FourD.Descriptor3D=self.meta['descriptor3D']['type']
            except:
                pass

        if not self.FourD.tc: self.FourD.tc=self.tc
        if self.FourD.initFreq=='tc':
            self.FourD.initFreq=self.FourD.tc
        self.ICidx=self.ICidx.split(',')
        if len(self.ICidx)==1:
            self.ICidx=self.ICidx*4
        if self.FourD.Descriptor.lower()=='ic' and not self.FourD.ICdict: self.FourD.ICdict=readIC(self.ICidx[0]) 
        
        if self.tSegm=='tc':
            self.tSegm=self.FourD.tc
        if self.dt=='tc':
            self.dt=self.FourD.tc
        
        if self.trajsList:
            with open(self.trajsList) as f:
                for line in f:
                    self.Trajs.append(line.strip())
        elif self.Trajs:
            self.Trajs=self.Trajs.split(',')

        if not self.nthreads:
            self.nthreads=os.cpu_count()

        if self.create4Dmodel:
            if self.FourD.reuseData or self.FourD.reuse3D or self.FourD.reuse4D:
                with open(f'{self.MLmodelOut}/meta.json') as f:
                    self.meta=json.load(f)
            else: 
                os.system(f'mkdir {self.mlmodelout}')
               
            self.writeMeta()

    def writeMeta(self):
        if not self.FourD.reuseData:
            self.meta['descriptor']['type']=self.FourD.Descriptor
            self.meta['descriptor']['xList']=self.FourD.xList.split(',')
            self.meta['descriptor']['yList']=self.FourD.yList.split(',')
            self.meta['descriptor']['ICdict']=dict(self.FourD.ICdict)
            self.meta['descriptor']['3D']=self.FourD.Descriptor3D
            self.meta['data']['trajs']=self.Trajs
            self.meta['data']['Nsubtrain']=self.Nsubtrain
            self.meta['data']['Nvalidate']=self.Nvalidate
            self.meta['data']['Tmax']=self.FourD.tmax
        if not self.FourD.reuse3D:
            self.meta['3Dmodel']['use']=self.FourD.use3D
            if self.meta['3Dmodel']['use']: self.meta['3Dmodel']['batchSize']=self.FourD.batchSize3D
        if not self.FourD.reuse4D:
            self.meta['4Dmodel']['batchSize']=self.FourD.batchSize4D
            self.meta['4Dmodel']['Tc']=self.FourD.tc
        self.meta['descriptor']['M2Vbar']=self.FourD.M2Vbar
        self.meta['descriptor']['M2dM']=self.FourD.M2dM
        self.meta['descriptor']['logisticateT']=self.FourD.logisticateT
        self.meta['descriptor']['normalizeT']=self.FourD.normalizeT

        with open(f'{self.mlmodelout}/meta.json','w') as f:
            json.dump(self.meta,f,indent=4)
                
            
class FourDcls(object):
    def __init__(self, args4D,devices=None,msg='') -> None:
        self.args = Args()
        self.args.parse(args4D)
        self.model3D=None
        self.model=None
        self.models=[]
        self.x=None
        self.y=None
        self.xEx=None
        self.yEx=None
        
        self.x3D=None
        self.y3D=None
        self.xEx3D=None
        self.yEx3D=None
        self.masses=None
        
        self.loss3D=np.inf
        self.loss4D=np.inf
        self.losses4D=None

        self.epoch3D=0
        self.epoch4D=0

        self.normalizeTfactor=0.1
        self.msg=msg
        self.strategy = tf.distribute.MirroredStrategy(devices)
        

        self.init_lr=0.001
        self.decay_steps=8192*2
        self.decay_rate=0.99
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)

        if self.args.MLmodelIn:
            self.masse=self.args.meta['masses']
            if self.args.meta['3Dmodel']['use']:
                self.model3D= tf.keras.models.load_model(f'{self.args.mlmodelin}/3D_model',compile=False)  
                self.epoch3D=self.args.meta['3Dmodel']['Epoch']
                self.loss3D=self.args.meta['3Dmodel']['loss']
            self.model=tf.keras.models.load_model(f'{self.args.mlmodelin}/4D_model',compile=False)
            self.epoch4D=self.args.meta['4Dmodel']['Epoch']
            self.losses4D=tuple(self.args.meta['4Dmodel']['losses'])
            self.loss4D=self.losses4D[-1]

        if self.args.MLmodelsIn:
            self.model3Ds=[]
            self.models=[]
            for model in self.args.MLmodelsIn.split(','):
                self.model3Ds.append(tf.keras.models.load_model(f'{model}/3D_model',compile=False))
                self.models.append(tf.keras.models.load_model(f'{model}/4D_model',compile=False))
            self.model3D=lambda x: tf.reduce_mean([m3D(x) for m3D in self.model3Ds],axis=0)

    
    def prepareData(self, shuffle=True):
        args=self.args
        global getDataFromTraj

        def getDataFromTraj(traj):
            if traj[-4:]=='.npz':
                dataset=dict(np.load(traj))
            elif traj[-3:]=='.h5':
                with File(traj,'r') as f:
                    dataset={}
                    dataset['xyz']=f['particles/all/position/value'][()]
                    dataset['t']=f['particles/all/position/time'][()].reshape(-1,1)
                    dataset['v']=f['particles/all/velocities/value'][()]
                    dataset['ep']=f['observables/potential_energy/value'][()].reshape(-1,1)
                    dataset['ek']=f['observables/kinetic_energy/value'][()].reshape(-1,1)
                    steps=min(len(dataset['xyz']),len(dataset['v']),len(dataset['ep']))
                    for k,v in dataset.items():
                        dataset[k]=v[:steps]
            else:
                print('unknown traj file type')
                return
            
            if args.FourD.tmax:
                tmax=args.FourD.tmax
            else:
                tmax=np.max(dataset['t'])
            
            mask=dataset['t'][:,0]<=tmax
            for k,v in dataset.items():
                try: 
                    dataset[k]=v[mask]
                except: 
                    pass
            
            if 'et' in args.FourD.xlist+args.FourD.ylist:
                dataset['et']=dataset['ek']+dataset['ep']
            if 'deltaE' in args.FourD.xlist+args.FourD.ylist:
                dataset['deltaE']=np.zeros_like(dataset['ek'])
            if args.FourD.Descriptor.lower()=='ic':
                dataset['Descriptor']=Describe(dataset['xyz'],dataset['v'],args.FourD.ICdict,m=self.masses)
                dataset['Descriptor']=correctAngle(dataset['Descriptor'])
            elif args.FourD.Descriptor.lower()=='xyz':
                dataset['Descriptor']=np.concatenate((dataset['xyz'],dataset['v']),axis=1).reshape(dataset['xyz'].shape[0],-1)
            x,y=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*args.Nsubtrain,0,step=1)
            xEx,yEx=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*(args.Nsubtrain+args.Nvalidate),tmax*args.Nsubtrain,step=1)
            if 'deltaE' in args.FourD.xlist+args.FourD.ylist:
                ek0=getEks(dataset['v'],dataset['m'][0])
                dataset['v']+=np.random.normal(0,0.01,dataset['v'].shape)
                ek1=getEks(dataset['v'],dataset['m'][0])
                dataset['deltaE']=ek1-ek0
                xx,yy=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*args.Nsubtrain,0,step=1)
                xxEx,yyEx=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*(args.Nsubtrain+args.Nvalidate),tmax*args.Nsubtrain,step=1)
                x=np.concatenate((x,xx),axis=0)
                y=np.concatenate((y,yy),axis=0)
                xEx=np.concatenate((xEx,xxEx),axis=0)
                yEx=np.concatenate((yEx,yyEx),axis=0)
            if not args.FourD.Descriptor3D:
                x3D=dataset['Descriptor'][:,:dataset['Descriptor'].shape[1]//2]
            elif args.FourD.Descriptor3D.lower()=='id':
                x3D=IDescr(dataset['xyz'])
            y3D=dataset['ep']


            
            print('described')
            xs[traj]=x
            ys[traj]=y
            xExs[traj]=xEx
            yExs[traj]=yEx
            x3Ds[traj]=x3D
            y3Ds[traj]=y3D
        if args.Trajs[0][-4:]=='.npz':
            self.masses=dict(np.load(args.Trajs[0]))['m'][0]
        elif args.Trajs[0][-3:]=='.h5':
            with File(args.Trajs[0],'r') as f:
                self.masses=f['particles/all/mass'][()]
        args.meta['masses']=list(self.masses.astype(float))
        args.meta['descriptor']['length']=len(args.meta['masses'])*6
        args.writeMeta()

        manager = Manager()
        xs=manager.dict()
        ys=manager.dict()
        xExs=manager.dict()
        yExs=manager.dict()
        x3Ds=manager.dict()
        y3Ds=manager.dict()

        print(' preparing data...')
        pool = Pool(args.nthreads)
        pool.map(getDataFromTraj,args.Trajs)
        pool.close()
        x=np.concatenate([xs[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        y=np.concatenate([ys[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        xEx=np.concatenate([xExs[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        yEx=np.concatenate([yExs[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        x3D=np.concatenate([x3Ds[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        y3D=np.concatenate([y3Ds[traj] for traj in args.Trajs], axis=0).astype(np.float32)
        if shuffle:
            x,y=shuffle(x,y)
            xEx,yEx=shuffle(xEx,yEx)
            x3D,y3D=shuffle(x3D,y3D)

        if args.NNvalidate:
            xEx,yEx=xEx[:args.NNvalidate],yEx[:args.NNvalidate]
        if args.NNsubtrain:
            x,y=x[:args.NNsubtrain],y[:args.NNsubtrain]

        l=args.meta['descriptor']['length']//2

        xEx3D=x3D[y.shape[0]:]
        yEx3D=y3D[y.shape[0]:]
        x3D=x3D[:y.shape[0]]
        y3D=y3D[:y.shape[0]]
        
        if args.FourD.reuse3D or args.estacc4Dmodel:
            m_shift=args.meta['3Dmodel']['m_shift']
            m_scale=args.meta['3Dmodel']['m_scale']
            ep_shift=args.meta['3Dmodel']['ep_shift']
            ep_scale=args.meta['3Dmodel']['ep_scale']
        else:
            m_shift=np.mean(x3D,axis=0)
            m_scale=np.std(x3D,axis=0)
            ep_shift=np.mean(y3D)
            ep_scale=np.std(y3D)
            args.meta['3Dmodel']['m_shift']=list(m_shift.astype(float))
            args.meta['3Dmodel']['m_scale']=list(m_scale.astype(float))
            args.meta['3Dmodel']['ep_shift']=float(ep_shift)
            args.meta['3Dmodel']['ep_scale']=float(ep_scale)
            args.writeMeta()

        x3D-=m_shift
        x3D/=m_scale
        y3D-=ep_shift
        y3D/=ep_scale
        xEx3D-=m_shift
        xEx3D/=m_scale
        yEx3D-=ep_shift
        yEx3D/=ep_scale

        if args.FourD.normalizeT:
            print("     normalized by time")
            x,y=normalizeT(x,y,l=l,a=self.normalizeTfactor)
            xEx,yEx=normalizeT(xEx,yEx,l=l,a=self.normalizeTfactor)

        elif args.FourD.m2vbar:
            print('     learning average velocities of descriptors')
            x,y=M2Vbar(x,y)
            xEx,yEx=M2Vbar(xEx,yEx)

        elif args.FourD.m2dm:
            print('     learning differences of descriptors')
            x,y=M2dM(x,y)
            xEx,yEx=M2dM(xEx,yEx)

        elif args.FourD.logisticatet:
            print("     time normalized with logistic function")
            x,y=logisticateT(x,y)
            xEx,yEx=logisticateT(xEx,yEx)

        if args.FourD.Descriptor.lower()=='ic':
            la=int(l/3)
            ld=la

            print("     correcting angles...")
            theOppositeWayToCorrectAngle(x,idx=list(range(ld,l)))
            theOppositeWayToCorrectAngle(xEx,idx=list(range(ld,l)))
            theOppositeWayToCorrectAngle(x3D,idx=list(range(ld,l)))
            theOppositeWayToCorrectAngle(xEx3D,idx=list(range(ld,l)))

        if args.FourD.reuse4D:
            x_shift=args.meta['4Dmodel']['x_shift']
            x_scale=args.meta['4Dmodel']['x_scale']
            y_shift=args.meta['4Dmodel']['y_shift']
        else:
            x_shift=np.mean(x,axis=0)
            x_scale=np.std(x,axis=0)
            y_shift=np.mean(y,axis=0)

            
            x_shift[:]=0
            x_scale[:]=1
            y_shift[:]=0
            args.meta['4Dmodel']['x_shift']=list(x_shift.astype(float))
            args.meta['4Dmodel']['x_scale']=list(x_scale.astype(float))
            args.meta['4Dmodel']['y_shift']=list(y_shift.astype(float))
            args.writeMeta()

        x-=x_shift
        x/=x_scale
        y-=y_shift
        xEx-=x_shift
        xEx/=x_scale
        yEx-=y_shift

        self.m_shift=m_shift
        self.m_scale=m_scale
        self.ep_shift=ep_shift
        self.ep_scale=ep_scale
        self.x_shift=x_shift
        self.x_scale=x_scale
        self.y_shift=y_shift

        self.x,self.y=x,y
        self.xEx,self.yEx=xEx,yEx
        self.x3D,self.y3D=x3D,y3D
        self.xEx3D,self.yEx3D=xEx3D,yEx3D

    def appendData(self,traj):
        args=self.args
        l=args.meta['descriptor']['length']//2
        m_shift=self.m_shift
        m_scale=self.m_scale
        ep_shift=self.ep_shift
        ep_scale=self.ep_scale
        x_shift=self.x_shift
        x_scale=self.x_scale
        y_shift=self.y_shift

        if traj[-4:]=='.npz':
            dataset=dict(np.load(traj))
        elif traj[-3:]=='.h5':
            with File(traj,'r') as f:
                dataset={}
                dataset['xyz']=f['particles/all/position/value'][()]
                dataset['t']=f['particles/all/position/time'][()]
                dataset['v']=f['particles/all/velocities/value'][()]
                dataset['ep']=f['observables/potential_energy/value'][()]
                dataset['ek']=f['observables/kinetic_energy/value'][()]
                steps=min(len(dataset['xyz']),len(dataset['v']),len(dataset['ep']))
                for k,v in dataset.items():
                    dataset[k]=v[:steps]
        else:
            print('unknown traj file type')
            return
        if args.FourD.Descriptor.lower()=='ic':
            dataset['Descriptor']=Describe(dataset['xyz'],dataset['v'],args.FourD.ICdict,m=self.masses)
        elif args.FourD.Descriptor.lower()=='xyz':
            dataset['Descriptor']=np.concatenate((dataset['xyz'],dataset['v']),axis=1).reshape(dataset['xyz'].shape[0],-1)
        tmax=np.max(dataset['t'])
        x,y=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*args.Nsubtrain,0,step=1,mode='dense')
        xEx,yEx=getData(dataset,self.xlist,self.ylist,args.FourD.tc,0,tmax*(args.Nsubtrain+args.Nvalidate),tmax*args.Nsubtrain,step=1,mode='dense')
        
        l=args.meta['descriptor']['length']//2
        ya=np.concatenate((y,yEx))
        ya,_=shuffle(ya,ya)
        x3D=ya[:y.shape[0],:l]
        y3D=ya[:y.shape[0],[-1]]
        xEx3D=ya[y.shape[0]:,:l]
        yEx3D=ya[y.shape[0]:,[-1]]
        x3D-=m_shift
        x3D/=m_scale
        y3D-=ep_shift
        y3D/=ep_scale
        xEx3D-=m_shift
        xEx3D/=m_scale
        yEx3D-=ep_shift
        yEx3D/=ep_scale

        if args.FourD.normalizeT:
            print("     normalized by time")
            x,y=normalizeT(x,y,l=l,a=self.normalizeTfactor)
            xEx,yEx=normalizeT(xEx,yEx,l=l,a=self.normalizeTfactor)

        elif args.FourD.m2vbar:
            print('     learning average velocities of descriptors')
            x,y=M2Vbar(x,y)
            xEx,yEx=M2Vbar(xEx,yEx)

        elif args.FourD.m2dm:
            print('     learning differences of descriptors')
            x,y=M2dM(x,y)
            xEx,yEx=M2dM(xEx,yEx)

        elif args.FourD.logisticatet:
            print("     time normalized with logistic function")
            x,y=logisticateT(x,y)
            xEx,yEx=logisticateT(xEx,yEx)

        x-=x_shift
        x/=x_scale
        y-=y_shift
        xEx-=x_shift
        xEx/=x_scale
        yEx-=y_shift

        x=np.concatenate((self.x,x)).astype(np.float32)
        xEx=np.concatenate((self.xEx,xEx)).astype(np.float32)
        x3D=np.concatenate((self.x3D,x3D)).astype(np.float32)
        xEx3D=np.concatenate((self.xEx3D,xEx3D)).astype(np.float32)
        y=np.concatenate((self.y,y)).astype(np.float32)
        yEx=np.concatenate((self.yEx,yEx)).astype(np.float32)
        y3D=np.concatenate((self.y3D,y3D)).astype(np.float32)
        yEx3D=np.concatenate((self.yEx3D,yEx3D)).astype(np.float32)

        x,y=shuffle(x,y)
        xEx,yEx=shuffle(xEx,yEx)
        x3D,y3D=shuffle(x3D,y3D)
        xEx3D,yEx3D=shuffle(xEx3D,yEx3D)


        self.x,self.y=x,y
        self.xEx,self.yEx=xEx,yEx
        self.x3D,self.y3D=x3D,y3D
        self.xEx3D,self.yEx3D=xEx3D,yEx3D


    def create3Dmodel(self):
        args=self.args
            
        with self.strategy.scope():
            if args.FourD.reuse3D:
                self.model3D=tf.keras.models.load_model(f'{args.mlmodelout}/3D_model',compile=False)  
                self.epoch3D=self.args.meta['3Dmodel']['Epoch']
                self.loss3D=self.args.meta['3Dmodel']['loss']
                if self.epoch3D + 2 > args.FourD.maxEpoch3D:
                    return
            else:
                self.model3D = tf.keras.Sequential([
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation=gelu),
                    tf.keras.layers.Dense(256, activation=gelu),
                    tf.keras.layers.Dense(128, activation=gelu),
                    tf.keras.layers.Dense(64, activation=gelu),
                    tf.keras.layers.Dense(1,activation='linear')
                ])
        
        self.train3D()

    def train3D(self):
        args=self.args
        ntrain=self.x3D.shape[0]
        m_shift=self.m_shift
        m_scale=self.m_scale
        ep_shift=self.ep_shift
        ep_scale=self.ep_scale
        
        with self.strategy.scope():
            self.optimizer3D = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        training_set3D = tf.data.Dataset.from_tensor_slices((self.x3D, self.y3D)).shuffle(256).batch(args.FourD.batchSize3D)
        dist_training_set3D=self.strategy.experimental_distribute_dataset(training_set3D)
        test_set3D = tf.data.Dataset.from_tensor_slices((self.xEx3D, self.yEx3D)).shuffle(256).batch(args.FourD.batchSize3D)
        

        @tf.function
        def loss_3D(model,xx,yy):
            return tf.sqrt(tf.reduce_mean(tf.math.square(model(xx)-yy)))
            
        # @tf.function
        def validate_3D(model,dataset):
            se=0.0
            count=0
            for data in dataset:
                se+=tf.square(loss_3D(model,*data))*data[0].shape[0]
                count+=data[0].shape[0]
            return tf.sqrt(se/count)

        @tf.function
        def training_step3D(data):
            xx,yy=data
            with tf.GradientTape() as tape:
                tape.watch(xx)
                loss=loss_3D(self.model3D,xx,yy)
            grad = tape.gradient(loss, self.model3D.trainable_variables)
            self.optimizer3D.apply_gradients(zip(grad, self.model3D.trainable_variables))
            return loss
        @tf.function
        def dist_training_step3D(dist_inputs):
            per_replica_losses = self.strategy.run(training_step3D, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,axis=None)

        def train(training_set):
            print('----------------------------------------------------')
            epoch_start= time.time()
            for step, data in enumerate(training_set):

                loss_train=dist_training_step3D(data)


            ex_loss=validate_3D(self.model3D,test_set3D)*ep_scale

            if ex_loss < self.loss3D:
                self.model3D.save(f'{args.mlmodelout}/3D_model')
                self.loss3D=ex_loss
                args.meta['3Dmodel']['Epoch']=self.epoch3D
                args.meta['3Dmodel']['loss']=self.loss3D.numpy().astype(float)
                args.writeMeta()

            
            now=time.time()
            t_epoch=now-epoch_start
            print('\repoch %-6d %5d/%-5d         t_epoch: %10.1f       ' % (self.epoch3D,step,ntrain//args.FourD.batchSize3D,t_epoch))

            print('validate:  %-10.6f'% ex_loss)
       
            print('best:      %-10.6f'% self.loss3D)
            sys.stdout.flush()
            self.epoch3D+=1
        
        print(' training 3D model')        
        print("     Ntrain: ",self.y3D.shape[0])

        while True:
            train(dist_training_set3D)
            if self.epoch3D >= args.FourD.maxEpoch3D: 
                break

        print(' 3D model trained')

    def create4Dmodel(self):
        args=self.args
        
        self.xlist=self.args.meta['descriptor']['xList']
        self.ylist=self.args.meta['descriptor']['yList']

        self.prepareData()
        
        l=args.meta['descriptor']['length']//2

        if args.FourD.use3D:
            self.create3Dmodel()
        with self.strategy.scope():
            if args.FourD.reuse4D:
                self.model=tf.keras.models.load_model(f'{args.mlmodelout}/4D_model',compile=False)
                self.epoch4D=self.args.meta['4Dmodel']['Epoch']
                self.losses4D=tuple(self.args.meta['4Dmodel']['losses'])
                self.loss4D=self.losses4D[-1]
                if self.epoch4D + 2 > args.FourD.maxEpoch4D:
                    return
            else:
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation=gelu),
                    tf.keras.layers.Dense(512, activation=gelu),
                    tf.keras.layers.Dense(256, activation=gelu),
                    tf.keras.layers.Dense(128, activation=gelu),
                    tf.keras.layers.Dense(64, activation=gelu),
                    tf.keras.layers.Dense(l,activation='linear')
                ])
        
        self.train4D()

    def train4D(self):
        args=self.args
        l=args.meta['descriptor']['length']//2
        if args.FourD.Descriptor.lower()=='ic':
            icidx=tf.constant(args.meta["descriptor"]["ICdict"]['idx'],dtype=tf.int32)
        m=tf.constant(args.meta['masses'],dtype=tf.float32)
        ntrain=self.x.shape[0]
        m_shift=self.m_shift
        m_scale=self.m_scale
        ep_shift=self.ep_shift
        ep_scale=self.ep_scale
        x_shift=self.x_shift
        x_scale=self.x_scale
        y_shift=self.y_shift

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        training_set = tf.data.Dataset.from_tensor_slices((self.x, self.y)).shuffle(256).batch(args.FourD.batchSize4D)
        test_set = tf.data.Dataset.from_tensor_slices((self.xEx, self.yEx)).shuffle(256).batch(args.FourD.batchSize4D)
        dist_training_set=self.strategy.experimental_distribute_dataset(training_set)

        @tf.function
        def loss_fn_ic(model,x,y,model3D=None):   
            ekloss=0
            eploss=0
            w=1

            with tf.GradientTape()as tape:
                tape.watch(x)
                yest=model(x)+y_shift[:l]
            vp=tape.batch_jacobian(yest,x)[:,:l,-1]/x_scale[-1]
            yy=y+y_shift
            mp=yest[:,:l]
            vt=yy[:,l:2*l]
            mt=yy[:,:l]

            if args.FourD.checkV0:
                x0=tf.concat((x[:,:-1],x[:,-1:]*0),axis=1)
                v0p=(model(x0)+y_shift[:l])*10
                v0t=x0[:,l:2*l]
                v0loss=tf.sqrt(tf.reduce_mean(tf.square(v0t-v0p)*w))
                ekloss+=v0loss

            if args.FourD.normalizeT:
                xx=x*x_scale+x_shift
                t=xx[:,-1][:,tf.newaxis]
                a=self.normalizeTfactor
                ft=(1-tf.exp(-(t/a)**2))
                dftdt=2*t/a**2*tf.exp(-(t/a)**2)
                mpic=mp*ft+xx[:,:l]+xx[:,l:2*l]*t*(1-ft)
                mtic=mt*ft+xx[:,:l]+xx[:,l:2*l]*t*(1-ft)
                if args.FourD.checkEk:
                    vpic=vp*ft+dftdt*mp+xx[:,l:2*l]*(1-dftdt*t-ft)
                    vpic=vt*ft+dftdt*mp+xx[:,l:2*l]*(1-dftdt*t-ft)

                w=tf.exp(-t)+1
            else:
                mpic=mp
                mtic=mt
                vpic=vp
                vtic=vp

            if args.FourD.checkEk:
                mpxyz,vpxyz=unDescribeTF(tf.concat((mpic,vpic),1),icidx)
                mtxyz,vtxyz=unDescribeTF(tf.concat((mtic,vtic),1),icidx)
                Ekp=getEksTF(vpxyz,m)
                Ekt=yy[:,-2] if 'ek' in args.FourD.ylist else getEksTF(vtxyz,m)
                ekloss+=tf.sqrt(tf.reduce_mean(tf.square(Ekp-Ekt)*w))

            # deltaE=tf.exp(-10*tf.abs(xx[:,-2][:,tf.newaxis]))
            # w=xx[:,-1][:,tf.newaxis]/10*(tf.sqrt(deltaE)-deltaE)+deltaE

            if model3D:
                epest=model3D(((tf.math.floormod(mpic+semidivisor,2*semidivisor)-semidivisor)-m_shift)/m_scale)
                epref=model3D(((tf.math.floormod(mtic+semidivisor,2*semidivisor)-semidivisor)-m_shift)/m_scale)
                eperr=(epest-epref)*ep_scale
                eploss=tf.sqrt(tf.reduce_mean(tf.square(eperr-args.FourD.epbias/627.5)*w))


            mloss=tf.sqrt(tf.reduce_mean(tf.square(mt-mp)*w,axis=0))
            vloss=tf.sqrt(tf.reduce_mean(tf.square(vt-vp)*w,axis=0))
            Dloss=tf.reduce_mean(mloss[1:ld])
            Aloss=tf.reduce_mean(mloss[ld+2:ld+la])
            DAloss=tf.reduce_mean(mloss[ld+la+3:])
            vDloss=tf.reduce_mean(vloss[1:ld])
            vAloss=tf.reduce_mean(vloss[ld+2:ld+la])
            vDAloss=tf.reduce_mean(vloss[ld+la+3:])

            if args.FourD.ekbias:
                ekloss+=tf.reduce_mean(vp**2-vt**2)*args.FourD.ekbias
                

            loss=1.0*Dloss+2.0*Aloss+4.0*DAloss+16.0*vDloss+20.0*vAloss+24.0*vDAloss+1.0*eploss+1*ekloss
            return Dloss,Aloss,DAloss,eploss,vDloss,vAloss,vDAloss,ekloss,loss

        @tf.function
        def loss_fn_xyz(model,x,y,model3D=None):
            with tf.GradientTape()as tape:
                tape.watch(x)
                yest=model(x)+y_shift[:l]
            vp=tape.batch_jacobian(yest,x)[:,:,-1]/x_scale[-1]
            yy=y+y_shift
            mp=yest[:,:l]
            vt=yy[:,l:2*l]
            mt=yy[:,:l]
            mloss=tf.sqrt(tf.reduce_mean(tf.square(mt-mp)))
            vloss=tf.sqrt(tf.reduce_mean(tf.square(vt-vp)))

            ekloss=0
            eploss=0

            if args.FourD.normalizeT:
                xx=x*x_scale+x_shift
                t=xx[:,-1][:,tf.newaxis]
                ft=1-tf.exp(-10*t)
                dftdt=10*tf.exp(-10*t)
                mpxyz=mp*ft+xx[:,:l]
                if args.FourD.checkEk:
                    vpxyz=vp*ft+dftdt*mp
            else:
                mpxyz=mp
                vpxyz=vp

            mpxyz=tf.reshape(mpxyz,[mpxyz.shape[0],-1,3])
            vpxyz=tf.reshape(vpxyz,[vpxyz.shape[0],-1,3])

            if model3D:
                x3D=((IDescrTF(mpxyz)-m_shift)/m_scale)
                eploss=tf.sqrt(tf.reduce_mean(tf.square(model3D(x3D)*ep_scale+ep_shift-yy[:,-1:])))
            if args.FourD.checkEk:
                vtxyz=tf.reshape(vt,[vt.shape[0],-1,3])
                Ekp=getEksTF(vpxyz,m)
                Ekt=yy[:,-2] if 'ek' in args.FourD.ylist else getEksTF(vtxyz,m)
                ekloss=tf.sqrt(tf.reduce_mean(tf.square(Ekp-Ekt)))
            

            loss=1.0*mloss+8.0*vloss+1.0*eploss+1.0*ekloss
            return mloss,vloss,eploss,ekloss,loss

        if args.FourD.Descriptor.lower()=='ic':
            la=l//3
            ld=la
            lda=la
            semidivisor=tf.constant([100]*ld+[np.pi]*(la+lda))
            loss_fn=loss_fn_ic
        elif args.FourD.Descriptor.lower()=='xyz':
            loss_fn=loss_fn_xyz

        def validate(model,dataset):
            se=np.zeros(9 if args.FourD.Descriptor.lower()=='ic' else 5)
            count=0
            for data in dataset:
                n=data[0].shape[0]
                se+=np.square(loss_fn(model,*data,model3D=self.model3D if args.FourD.use3D else None))*n
                count+=n
            return tuple(np.sqrt(se/count))

        @tf.function
        def training_step(data):
            xx,yy=data
            with tf.GradientTape() as tape:
                tape.watch(xx)
                losses=loss_fn(self.model,xx,yy,model3D=self.model3D if args.FourD.use3D else None)
                loss=losses[-1]
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            return losses

        @tf.function
        def dist_training_step(dist_inputs):
            per_replica_losses = self.strategy.run(training_step, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,axis=None)

        def train(training_data):
            print('----------------------------------------------------')
            epoch_start= time.time()
            for step, data in enumerate(training_data):
                losses=dist_training_step(data)

            train_losses=validate(self.model,training_set)
            ex_losses=validate(self.model,test_set)
            if ex_losses[-1] < self.loss4D:
                self.model.save(f'{args.mlmodelout}/4D_model')
                self.loss4D=ex_losses[-1]
                self.losses4D=ex_losses
                args.meta['4Dmodel']['Epoch']=self.epoch4D
                args.meta['4Dmodel']['losses']=[loss.astype(float) for loss in self.losses4D]
                args.writeMeta()

            # print(self.optimizer._decayed_lr(tf.float32))
            # print(self.losses4D)
            now=time.time()
            t_epoch=now-epoch_start
            print('\repoch %-6d %5d/%-5d         t_epoch: %10.1f       ' % (self.epoch4D,step,ntrain//args.FourD.batchSize4D,t_epoch))
            if args.FourD.Descriptor.lower()=='ic':
                print('%-10s %-10s %-10s %-10s %-10s'% (self.msg,"D","A","DA","Ep/Ek"))
                print('train:     %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% train_losses)
                print('validate:  %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% ex_losses)            
                print('best:      %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% self.losses4D)
            elif args.FourD.Descriptor.lower()=='xyz':
                print('           %-10s %-10s %-10s %-10s'% ("M","v","Ep","Ek"))
                print('train:     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% train_losses)   
                print('validate:  %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% ex_losses)            
                print('best:      %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% self.losses4D)
            sys.stdout.flush()
            self.epoch4D+=1
            
        
        print('training 4D model')
        print("     Ntrain: ",self.y.shape[0])
        while True:
            train(dist_training_set)
            if self.epoch4D >= args.FourD.maxEpoch4D: 
                break
        print('4D model trained')

    def use4Dmodel(self):
        args=self.args
        
        icdict=args.meta["descriptor"]["ICdict"]
        m=np.array(args.meta['masses'])
        if self.model3D:
            m_shift=args.meta['3Dmodel']['m_shift']
            m_scale=args.meta['3Dmodel']['m_scale']
            ep_shift=args.meta['3Dmodel']['ep_shift']
            ep_scale=args.meta['3Dmodel']['ep_scale']

        x_shift=np.array(args.meta['4Dmodel']['x_shift']).astype(np.float32)
        x_scale=np.array(args.meta['4Dmodel']['x_scale']).astype(np.float32)
        y_shift=np.array(args.meta['4Dmodel']['y_shift']).astype(np.float32)

        l=args.meta['descriptor']['length']//2
        la=l//3
        ld=la
        lda=la

        tm=args.trun

        
        MaxSegm=args.tSegm
        dt=args.dt

        xyzs,[sp,*_] =loadXYZ(args.initXYZ,list)
        vs, _ = loadXYZ(args.initVXYZ,list,getsp=False)

        i=0
        vs[i]=adjV(xyzs[i], vs[i], m)
        xyzoffset=getCoM(xyzs[i],m)
        voffset=getCoM(vs[i],m)
        # print(voffset)
        x0=Describe(xyzs[i][np.newaxis],vs[i][np.newaxis],icdict,m=m).astype(np.float32)
        init_transrot=x0[0,[0,1,2,la,la+1,2*la,l+0,l+1,l+2,l+la,l+la+1,l+2*la]]
        # print(x0)
        ts=np.arange(0,tm,dt)+dt
        ts[-1]=tm
        t0=0

        xyzfile=args.trajXYZout
        vxyzfile=args.trajVXYZout
        tfile=args.trajTime
        ft=open(args.trajTime,'w')
        fek=open(args.trajEkin,'w')
        fy=open(args.trajDescr,'w')
        fep=open(args.trajEpot,'w')
        fet=open(args.trajEtot,'w')

        ek=getEk(vs[i],m)[np.newaxis]
        if self.model3D:
            ep=self.model3D((x0[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
        else:
            ep=np.zeros_like(ek)
        etot=(ep+ek)[0]
        P0=getLinM(vs[i],m)
        L0=getAngM(xyzs[i],vs[i],m)
        Eerror=0
        np.savetxt(fek,ek)
        np.savetxt(fep,ep)
        np.savetxt(fet,(ep+ek))
        np.savetxt(fy,x0)
        saveXYZ(xyzfile,xyzs[i],sp,'w',msg=f't=0.0fs')
        saveXYZ(vxyzfile,vs[i],sp,'w',msg=f't=0.0fs')
        
        with open(tfile,'w') as f:
            f.write('0.0\n')

        def getY(x, model):
#            t_start=time.time()
            y=differentiation(model,x,x_shift=x_shift,x_scale=x_scale,y_shift=y_shift,l=l).numpy()

            if np.sum(np.isnan(y)) > 0:
                stopper.stopMLatom('prediction failed')
                
            if args.FourD.normalizeT:
                x,y=unnormalizeT(x,y,l=l,a=self.normalizeTfactor)
            elif args.FourD.m2vbar:
                x,y=Vbar2M(x,y)
            elif args.FourD.m2dm:
                x,y=dM2M(x,y)
            elif args.FourD.logisticatet:
                x,y=unlogisticateT(x,y)
            theOppositeWayToCorrectAngle(y,idx=list(range(ld,l)))
            return y
        
        def getAll(x, model, t=0, adjv=False):
            y=getY(x, model)
            xyz,v=unDescribeParaTF(y,icdict['idx'])
            xyz=xyz+xyzoffset[np.newaxis,:]+voffset[np.newaxis,:]*t
            v=v-getCoM(v,m)[:,np.newaxis,:]+voffset[np.newaxis,:]
            if adjv:
                for i in range(len(v)):
                    v[i]=adjV(xyz[i],v[i],m,P0,L0)
            ek=getEks(v,m)
            ep=np.zeros_like(ek)
            if self.model3D:
                ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            if args.FourD.forceEtot:
                v=v*np.sqrt((etot-ep)/ek)[:,np.newaxis,np.newaxis]
                ek=getEks(v,m)
            Eerror=(ep+ek)-etot
            return y, xyz, v, ek ,ep, Eerror#, x0

        def monarchWings(x0):
            t=np.append((np.arange(90)+10)/10,10.)
            # t=np.append((np.arange(100))/10,10.)
            # t=(np.arange(61)+20)/10.
            x=np.concatenate((np.repeat(x0,len(t),axis=0),t[:,np.newaxis]),axis=1)
            y=getY(x, self.model)
            x[:-1]=np.concatenate((y[:-1],(t[-1]-t[:-1])[:,np.newaxis]),axis=1)
            # print(x[:,[0,-1]])
            y[:-1]=getY(x[:-1], self.model)

            xyz,v=unDescribePara(y,icdict['idx'])
            v=v-getCoM(v,m)[:,np.newaxis,:]+voffset[np.newaxis,:]
            ek=getEks(v,m)
            ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            et=ek+ep
            eterr=np.abs(et-etot)
            print(np.min(eterr)*627.5,t[np.argmin(eterr)])
            return x[np.argmin(eterr),:-1], t[np.argmin(eterr)]

        def findRoot(x0,t0,tt,y0=None,threshold=0.01/627.5,i=0):
            print('finding root between %.4f and %.4f'% (t0,tt))
            i+=1
            tm=(t0+tt)/2
            ts=[[tm]] if y0 else [[t0],[tm]] 
            # ts=[[t0],[tm],[tt]]
            x=np.concatenate((np.repeat(x0,len(ts),axis=0),ts),axis=1)
            y=getY(x, self.model)
            _,v=unDescribePara(y,icdict['idx'])
            v=v-getCoM(v,m)[:,np.newaxis,:]+voffset[np.newaxis,:]
            ek=getEks(v,m)
            ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            et=ek+ep
            eterr=et-etot
            if np.abs(eterr[-1])<threshold or i>8:
                return tm 
            y0=y0 if y0 else eterr[0]
            # print(y0*eterr[-1]<0,y0*eterr[-2]<0)
            if y0*eterr[-1]<0:
                return findRoot(x0,t0,tm,y0,i=i)
            else:
                return findRoot(x0,tm,tt,eterr[-1],i=i)

        def findNextT0(x0):
            t=np.append((np.arange(90)+10)/10,10.)
            # t=np.append((np.arange(100))/10,10.)
            x=np.concatenate((np.repeat(x0,len(t),axis=0),t[:,np.newaxis]),axis=1)
            y=getY(x, self.model)

            xyz,v=unDescribePara(y,icdict['idx'])
            xyz=xyz+xyzoffset[np.newaxis,:]+voffset[np.newaxis,:]*(t[:,np.newaxis,np.newaxis]+t0)
            v=v-getCoM(v,m)[:,np.newaxis,:]+voffset[np.newaxis,:]
            ek=getEks(v,m)
            ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            et=ek+ep
            eterr=et-etot
            # print(eterr)
            for i in range(len(eterr)-1,0,-1):
                if eterr[i]*eterr[i-1]<0:
                    print(eterr[i-1],eterr[i])
                    return findRoot(x0,t[i-1],t[i],eterr[i-1])
            print(' root not found, using point with least Etot error for the next step...')
            return t[np.argmin(np.abs(eterr))]

        def saveResult(result,times, modelID):
            tstamps=['%.3f fs %s' % (times[i], 'model'+str(modelID[i]) if self.args.MLmodelsIn else '') for i in range(len(times))]
            y, xyz, v, ek ,ep, _ =result
            np.savetxt(fep,ep)
            np.savetxt(fet,(ep+ek))
            saveXYZs(xyzfile,xyz,sp,'a',msgs=tstamps)
            saveXYZs(vxyzfile,v,sp,'a',msgs=tstamps)

            np.savetxt(fek,ek)
            np.savetxt(fy,y)
            # print(y[:,56])
            for tstamp in tstamps:
                ft.write(tstamp+'\n')
            return

        Segms=[]
        pdict={0:0}
        tbreak=0
        threshold=1
        t0s=[0]
        md_start=time.time()
        while t0 < tm:
            if args.FourD.adaptSeg and not self.args.MLmodelsIn:
                tSegm=findNextT0(x0)
            else:
                tSegm=args.tSegm

            if t0 not in pdict.keys(): pdict[t0]=0
            if t0 > tbreak: threshold=1
            
            t=ts[(ts>t0+0.000001)&(ts<t0+tSegm+0.000001)]-t0
            x=np.concatenate((np.repeat(x0,len(t),axis=0),t[:,np.newaxis]),axis=1)
            xSegm=np.concatenate((x0,[[tSegm]]),axis=1)
            if args.FourD.monarchWings:
                xx, tt= monarchWings(x0)
                x[x[:,-1]>tt+0.001,:-1]=xx
                x[x[:,-1]>tt+0.001,-1]-=tt

            if not self.args.MLmodelsIn:
                x0, *_, Eerror =getAll(xSegm, self.model)
                # print(Eerror)
                segm=[getAll(x, self.model,(t[:,np.newaxis,np.newaxis]+t0)),[i for i in t+t0],[0 for i in t+t0],x0]
            elif args.FourD.adaptSeg:
                results=[]
                for model in self.models:
                    results.append(getAll(x, model,(t[:,np.newaxis,np.newaxis]+t0)))
                Eerrors=np.abs([result[-1] for result in results])/np.sqrt(t)
                sort=np.dstack(np.unravel_index(Eerrors.ravel().argsort(), Eerrors.shape))[0]
                segm=[result[:sort[pdict[t0],1]+1] for result in results[sort[pdict[t0],0]]]
                x0=segm[0][[-1]]
                Eerror=segm[-1]
                tSegm=t[sort[pdict[t0],1]]
                if np.abs(Eerror[-1])*627.5>threshold or pdict[t0]>=len(sort)-1:
                    print(t0,pdict[t0],len(Segms),Eerror[-1]*627.5,threshold)
                    tbreak=max(t0,tbreak)
                    if len(Segms)>1:
                        Segms.pop(-1)
                        t0s.pop(-1)
                        t0=t0s[-1]
                        x0=Segms[-1][-1]
                        pdict[t0]+=1
                    else:
                        threshold*=2
                        pdict[t0]=0
                    for k in pdict.keys():
                        if k>t0:
                            pdict[k]=0
                    continue

                print(f'model: {sort[pdict[t0],0]}, tSegm: {tSegm}')
                segm=[segm,[i for i in t+t0],[sort[pdict[t0]] for i in t+t0],x0]
            else:
                results=[]
                for model in self.models:
                    results.append(getAll(xSegm, model, adjv=True))
                Eerrors=np.abs([result[-1][-1] for result in results])
                sort=np.argsort(Eerrors)
                print(sort[pdict[t0]])
                x0, *_, Eerror=results[sort[pdict[t0]]]
                if np.abs(Eerror[-1])*627.5>threshold or pdict[t0]>=len(sort)-1:
                    print(t0,pdict[t0],len(Segms))
                    tbreak=max(t0,tbreak)
                    if len(Segms)>1:
                        Segms.pop(-1)
                        t0-=tSegm
                        x0=Segms[-1][-1]
                        pdict[t0]+=1
                    else:
                        x0=Segms[-1][-1]
                        if args.FourD.reOrient:
                            x0[0,[0,1,2,la,la+1,2*la,l+0,l+1,l+2,l+la,l+la+1,l+2*la]]=init_transrot
                        threshold*=2
                        pdict[t0]=0
                    for k in pdict.keys():
                        if k>t0:
                            pdict[k]=0
                    continue
                if not self.args.FourD.reOrient:
                    segm=[x,t+t0,[sort[pdict[t0]] for _ in t],x0]
                else:
                    segm=[getAll(x, self.models[sort[pdict[t0]]],(t[:,np.newaxis,np.newaxis]+t0),adjv=True),t+t0,[sort[pdict[t0]] for _ in t],x0]

            Segms.append(segm)
            if self.args.FourD.reOrient:
                if len(Segms)>10:
                    saveResult(*Segms[0][:-1])
                    Segms.pop(0)
            
            print(' %.2f-%.2f fs'% (t0,t0+tSegm))
            print(f'Etot error: {(Eerror[-1])*627.5} kcal/mol')
            
            sys.stdout.flush()
            t0+=tSegm
            t0s.append(t0)
        
        if self.args.MLmodelsIn and not self.args.FourD.reOrient:
            models_segms=[]
            models_ts=[]
            models_ids=[]
            for i in range(len(self.models)):
                if [True for Segm in Segms if Segm[2][0]==i]:
#                    t_start=time.time()
                    model_xs=np.concatenate([Segm[0] for Segm in Segms if Segm[2][0]==i],axis=0)
                    model_ts=np.array([Segm[1] for Segm in Segms if Segm[2][0]==i]).flatten()
                    models_ts.append(model_ts)
                    models_ids.append(np.array([Segm[2] for Segm in Segms if Segm[2][0]==i]).flatten())
                    print(f'predict model{i}', model_xs.shape)
                    models_segms.append(getAll(model_xs, self.models[i],model_ts[:,np.newaxis,np.newaxis],adjv=True))
#                    print('predict time: ', time.time()-t_start)
            order=np.concatenate(models_ts).argsort()
            models_segms=[np.concatenate(result,0)[order] for result in zip(*models_segms)]
            t_start=time.time()
            saveResult(models_segms,np.concatenate(models_ts,0)[order],np.concatenate(models_ids,0)[order])
            print('write time: ', time.time()-t_start)
            
            print('MD time: ', time.time()-md_start)

        else:
            for Segm in Segms:
                saveResult(*Segm[:-1])
        
        ft.close()
        fek.close()
        fy.close()
        fep.close()
        fet.close()
    def estAcc4Dmodel(self):
        args=self.args
        
        self.xlist=self.args.meta['descriptor']['xList']
        self.ylist=self.args.meta['descriptor']['yList']


        l=args.meta['descriptor']['length']//2
        m=tf.constant(args.meta['masses'],dtype=tf.float32)
        la=l//3
        ld=la
        lda=la
        semidivisor=np.array([100]*ld+[np.pi]*(la+lda))

        print(self.model3D)
    
        m_shift=args.meta['3Dmodel']['m_shift']
        m_scale=args.meta['3Dmodel']['m_scale']
        ep_shift=args.meta['3Dmodel']['ep_shift']
        ep_scale=args.meta['3Dmodel']['ep_scale']
        x_shift=args.meta['4Dmodel']['x_shift']
        x_scale=args.meta['4Dmodel']['x_scale']
        y_shift=args.meta['4Dmodel']['y_shift']
        tc=args.meta['4Dmodel']['Tc']
        icidx=tf.constant(args.meta["descriptor"]["ICdict"]['idx'],dtype=tf.int32)
        icdict=args.meta["descriptor"]["ICdict"]

        # nsegm=49
        
        nsegm={20:49, 10:99, 1:990, 5:198, 30:33, 50:19}[tc]
        print(tc)
        self.args.Nsubtrain=0
        self.args.Nvalidate=1
        self.args.NNvalidate=(20*tc+1)*nsegm if args.FourD.normalizeT else (20*tc)*nsegm
        self.prepareData(shuffle=False)
        print(self.xEx.shape, self.xEx[:,-1])
        ep=self.model3D((self.xEx[:1,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
        xyz,v=unDescribeParaTF(self.xEx[:1,:-1],icdict['idx'])
        ek=getEk(v,m)[np.newaxis]
        etot=ek+ep
        def getY(x, model):
#            t_start=time.time()
            y=differentiation(model,x,x_shift=x_shift,x_scale=x_scale,y_shift=y_shift,l=l).numpy()
            if np.sum(np.isnan(y)) > 0:
                stopper.stopMLatom('prediction failed')
                
            if args.FourD.normalizeT:
                
#                t_start=time.time()
                x,y=unnormalizeT(x,y,l=l,a=self.normalizeTfactor)
            elif args.FourD.m2vbar:
                x,y=Vbar2M(x,y)
            elif args.FourD.m2dm:
                x,y=dM2M(x,y)
            elif args.FourD.logisticatet:
                x,y=unlogisticateT(x,y)
            
            theOppositeWayToCorrectAngle(y,idx=list(range(ld,l)))
            return y
        
        def getAll(x, model, t=0, adjv=False,):
            y=getY(x, model)
            xyz,v=unDescribeParaTF(y,icdict['idx'])
            if adjv:
                t_start=time.time()
                for i in range(len(v)):
                    v[i]=adjV(xyz[i],v[i],m)
                print('    adjV time: ', time.time()-t_start)
            ek=getEks(v,m)
            ep=np.zeros_like(ek)
            if self.model3D:
                ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            if args.FourD.forceEtot:
                v=v*np.sqrt((etot-ep)/ek)[:,np.newaxis,np.newaxis]
                ek=getEks(v,m)
            Eerror=(ep+ek)-etot
            
            return y, xyz, v, ek ,ep, Eerror

        if args.FourD.normalizeT:
            _, yt=unnormalizeT(self.xEx,self.yEx,l=l,a=self.normalizeTfactor)
        else:
            yt = self.yEx
        theOppositeWayToCorrectAngle(yt,idx=list(range(ld,l)))
        mt=yt[:,:l]
        vt=yt[:,l:2*l]
        losses=[]
        print(etot)
        yp=[]
        Eerror=[]
        for isegm in range(nsegm):
            print(isegm)
            Eerrors=[]
            for i, model in enumerate(self.models):
                Eerrors.append(getAll(self.xEx[[(isegm+1)*(20*tc)-1]], model)[-1])
            bestidx=np.argmin(Eerrors)
            print(Eerrors[bestidx]*627.5)
            yest,_,_,_,_,etshift=getAll(self.xEx[(isegm)*(20*tc):(isegm+1)*(20*tc)], self.models[bestidx], adjv=True)
            yp.append(yest)
            Eerror.append(etshift)
        yp=np.concatenate(yp,axis=0)
        Eerror=np.concatenate(Eerror,axis=0)
        np.savetxt('error',Eerror)
        mp=yp[:,:l]
        vp=yp[:,l:]
        print(self.xEx.shape, self.yEx.shape,mt.shape, mp.shape)
        print((mt-mp).max())
        merr=mt-mp
        theOppositeWayToCorrectAngle(merr,idx=list(range(ld,l)))
        mloss=np.sqrt(np.mean(np.square(merr),axis=0))
        # print(mt[0],mp[0])
        vloss=np.sqrt(np.mean(np.square(vt-vp),axis=0))
        Dloss=np.mean(mloss[1:ld])
        Aloss=np.mean(mloss[ld+2:ld+la])
        DAloss=np.mean(mloss[ld+la+3:])
        vDloss=np.mean(vloss[1:ld])
        vAloss=np.mean(vloss[ld+2:ld+la])
        vDAloss=np.mean(vloss[ld+la+3:])
        etloss=np.sqrt(np.mean(np.square(Eerror)))
        losses.append([Dloss,Aloss,DAloss,vDloss,vAloss,vDAloss,etloss])
        print(np.array(losses))
        print("%.6f %.6f %.6f %.6f %.6f %.6f %.6f"% tuple(list(np.mean(losses,axis=0))))


    def activeLearning(self):
        args=self.args
        os.system(f'mkdir {args.MLmodelOut}')
        if not os.path.isdir(f'{args.MLmodelOut}/4DMD'): os.system(f'mkdir {args.MLmodelOut}/4DMD')
        os.system(f'cp {args.initxyz} {args.MLmodelOut}/4DMD/init.xyz')
        os.system(f'cp {args.initvxyz} {args.MLmodelOut}/4DMD/init.vxyz')

        Nmodel=len(args.ICidx)
        gpus=[x.name.split(':',1)[1] for x in  tf.config.get_visible_devices(device_type='GPU')]

        models=[]
        for i in range(Nmodel):
            newargs=addReplaceArg('activeLearning','create4Dmodel',args.args2pass)
            newargs=addReplaceArg('ICidx',f'ICidx={args.ICidx[i]}',newargs)
            newargs=addReplaceArg('MLmodelOut',f'MLmodelOut={args.MLmodelOut}/ALmodel{i}',newargs)
            newargs=addReplaceArg('initXYZ',f'initXYZ={args.MLmodelOut}/4DMD/init.xyz',newargs)
            newargs=addReplaceArg('initVXYZ',f'initVXYZ={args.MLmodelOut}/4DMD/init.vxyz',newargs)
            newargs=addReplaceArg('trajXYZout',f'trajXYZout={args.MLmodelOut}/ALmodel{i}/traj.xyz',newargs)
            newargs=addReplaceArg('trajVXYZout',f'trajVXYZout={args.MLmodelOut}/ALmodel{i}/traj.vxyz',newargs)
            newargs=addReplaceArg('trajTime',f'trajTime={args.MLmodelOut}/ALmodel{i}/traj.t',newargs)
            newargs=addReplaceArg('trajEpot',f'trajEpot={args.MLmodelOut}/ALmodel{i}/traj.epot',newargs)
            newargs=addReplaceArg('trajEkin',f'trajEkin={args.MLmodelOut}/ALmodel{i}/traj.ekin',newargs)
            newargs=addReplaceArg('trajEtot',f'trajEtot={args.MLmodelOut}/ALmodel{i}/traj.etot',newargs)
            newargs=addReplaceArg('trajH5MDout',f'trajH5MDout={args.MLmodelOut}/ALmodel{i}/traj.h5',newargs)
            
            models.append(FourDcls(newargs,devices=[gpus[i]] if gpus else None, msg=f'AL_{i}'))
        for i in range(Nmodel):
            if i>0 and args.ICidx[i]==args.ICidx[-1] and not args.FourD.reuse3D and not args.FourD.reuse4D:
                os.system(f'rm -rf {args.MLmodelOut}/ALmodel{i}; cp -r {args.MLmodelOut}/ALmodel{i-1} {args.MLmodelOut}/ALmodel{i}')
                newargs=addReplaceArg('FourD.reuseData','FourD.reuseData=1',newargs)
            print(f" training ALmodel{i}...")
            models[i].create4Dmodel()
        self.Niter=0
        self.Nretrain=0
        while True:
            path=f'{args.MLmodelOut}/4DMD/{self.Niter}_{self.Nretrain}'
            os.system(f'mkdir {path}')
            xyzts=[]
            vts=[]
            yts=[]
            deltaE=[]
            for i in range(Nmodel):
                print(f" running 4DMD with ALmodel{i}...")
                xyzt,vt,yt,sp,ek,ep=models[i].use4Dmodel()
                os.system(f'mkdir {path}/{i}')
                os.system(f'cp {args.MLmodelOut}/ALmodel{i}/traj.* {path}/{i}')
                xyzts.append(xyzt)
                vts.append(vt)
                yts.append(yt)
                deltaE.append(np.abs((ek+ep)[1]-(ek+ep)[0]))
            # print(np.array(xyzts))
            sdxyz=np.std(np.array(xyzts),axis=0)
            sdv=np.std(np.array(vts),axis=0)
            sdy=np.std(np.array(yts),axis=0)
            print(f' SD of finial geometries:\n{sdxyz}')
            print(f' SD of finial velocities:\n{sdv}')
            print(f' SD of finial descriptors:\n{sdy.reshape(2,3,-1).transpose(0,2,1)}')
            print(f' energy shift {(ek+ep)[1]-(ek+ep)[0]}')
            # sd=np.mean(sdxyz)+np.mean(sdv)
            sd=np.mean(sdxyz)
            print(f' score of SD: {sd}')
            sys.stdout.flush()
            if sd >0.1 or np.isnan(sd) or np.max(deltaE)>0.01:
                if not self.Nretrain:
                    newtraj=self.run3DMD()
                for i in range(Nmodel):
                    models[i].appendData(newtraj)
                    print(f' retraining ALmodel{i}...')
                    models[i].train3D()
                    models[i].train4D()
                self.Nretrain+=1
                continue
            else:
                xyz0=np.mean(np.array(xyzts),axis=0)
                v0=np.mean(np.array(vts),axis=0)
                saveXYZ(f'{args.MLmodelOut}/4DMD/init.xyz',xyz0,sp)
                saveXYZ(f'{args.MLmodelOut}/4DMD/init.vxyz',v0,sp)
                self.Nretrain=0
                self.Niter+=1
    
    def run3DMD(self):
        import ThreeDMD
        args=self.args
        if not os.path.isdir(f'{args.MLmodelOut}/3DMD'): os.system(f'mkdir {args.MLmodelOut}/3DMD')
        path=f'{args.MLmodelOut}/3DMD/{self.Niter}'
        os.system(f'mkdir {path}')
        os.system(f'cp {args.MLmodelOut}/4DMD/init.xyz {path}/init.xyz')
        os.system(f'cp {args.MLmodelOut}/4DMD/init.vxyz {path}/init.vxyz')
        args3D=addReplaceArg('activeLearning','create4Dmodel',args.args2pass)
        args3D=addReplaceArg('device',f'device=cpu',args3D)
        args3D=addReplaceArg('initXYZ',f'initXYZ={path}/init.xyz',args3D)
        args3D=addReplaceArg('initVXYZ',f'initVXYZ={path}/init.vxyz',args3D)
        args3D=addReplaceArg('trajXYZout',f'trajXYZout={path}/traj.xyz',args3D)
        args3D=addReplaceArg('trajVXYZout',f'trajVXYZout={path}/traj.vxyz',args3D)
        args3D=addReplaceArg('trajTime',f'trajTime={path}/traj.t',args3D)
        args3D=addReplaceArg('trajEpot',f'trajEpot={path}/traj.epot',args3D)
        args3D=addReplaceArg('trajEkin',f'trajEkin={path}/traj.ekin',args3D)
        args3D=addReplaceArg('trajEtot',f'trajEtot={path}/traj.etot',args3D)
        args3D=addReplaceArg('trajH5MDout',f'trajH5MDout={path}/traj.h5',args3D)
        print(" models didn't agree with each other, runing 3DMD...")
        ThreeDMD.ThreeDcls.dynamics(args3D)
        return f'{path}/traj.h5'

    def reactionPath(self):
        args=self.args
        from scipy.optimize import dual_annealing
        
        icdict=args.meta["descriptor"]["ICdict"]
        m=np.array(args.meta['masses'])
        if self.model3D:
            m_shift=args.meta['3Dmodel']['m_shift']
            m_scale=args.meta['3Dmodel']['m_scale']
            ep_shift=args.meta['3Dmodel']['ep_shift']
            ep_scale=args.meta['3Dmodel']['ep_scale']

        x_shift=np.array(args.meta['4Dmodel']['x_shift']).astype(np.float32)
        x_scale=np.array(args.meta['4Dmodel']['x_scale']).astype(np.float32)
        y_shift=np.array(args.meta['4Dmodel']['y_shift']).astype(np.float32)

        l=args.meta['descriptor']['length']//2
        la=l//3
        ld=la
        lda=la
        args=self.args

        # path=args.FourD.reactionPath.dirOut
        # os.system(f'mkdir {path}')
        step=0.5

        def findInit(xyz0,t):
            def predict(xyz, v, t):
                xyzoffset=getCoM(xyz,m)
                voffset=getCoM(v,m)
                x0=Describe(xyz[np.newaxis],v[np.newaxis],icdict,m=m).astype(np.float32)
                etot=(getEk(v,m)[np.newaxis]+self.model3D((x0[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift)[0]
                def segment(x):
                    y=differentiation(self.model,x,x_shift=x_shift,x_scale=x_scale,y_shift=y_shift,l=l).numpy()
                    x,y=unnormalizeT(x,y,l=l,a=self.normalizeTfactor)
                    theOppositeWayToCorrectAngle(y,idx=list(range(ld,l)))
                    xyz,v=unDescribePara(y,icdict['idx'])
                    for i in range(len(v)):
                        v[i]=adjV(xyz[i],v[i],m)
                    ek=getEks(v,m)
                    ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
                    Eerror=(ep+ek)[0]-etot
                    return y, xyz, v, ek[0] ,ep[0], Eerror

                while True:
                    if t > self.args.tc:
                        x=np.concatenate((x0,np.array([[self.args.tc]])),axis=1)
                        x0, *_ = segment(x)
                        t = t - self.args.tc
                    else:
                        return segment(x)
            global best
            best=np.inf
            def getDifference(vxyz0):
                global best
                try:
                    vxyz0=vxyz0.reshape(-1,3)
                    vxyz0=adjV(xyz0,vxyz0,m)
                    y, [xyzt,*_], [vxyzt,*_], ek ,ep, Eerror = predict(xyz0, vxyz0, t)
                    ek0=getEk(vxyz0,m)
                    ek=getEk(vxyzt,m)
                    Eerror=np.abs(Eerror)*627.5
                    DAerror=np.abs(y[0,56]-np.pi)%(2*np.pi)
                    loss= DAerror + Eerror/100
                except:
                    loss = np.inf
                    DAerror = np.nan
                    Eerror =np.nan
                if best > loss:
                    best=loss
                    saveXYZ("inits/%.3f_%.3f_%.3f.vxyz" %(loss, DAerror, Eerror), vxyz0,sp)
                    saveXYZ("finals/%.3f_%.3f_%.3f.xyz" %(loss, DAerror, Eerror), xyzt,sp)
                    # print(xyz)
                    # print(v)
                    print(t,ek, ep)
                    # print(x)
                print(f'current/best loss: {loss}/{best}, ({DAerror}/{Eerror}）')
                sys.stdout.flush()
                return loss
            res=dual_annealing(getDifference,tuple([(-.06,.06)]*l), x0=loadXYZ('inits/0.186_0.089_9.728.vxyz')[0].flatten(),maxiter=10000,maxfun=100000)
            return res

        reactants,sps =loadXYZ(args.initXYZ)
        # products,sps =loadXYZ(args.finalXYZ)
        
        for i in range(len(reactants)):
            xyz0=reactants[i]
            # xyzt=products[i]
            res=findInit(xyz0,args.reactionTime)
            v0=res.x.reshape(-1,3)
            # saveXYZ(f'{path}/mol{i}_v0.xyz', v0,sp)
            
            saveXYZ(args.initVout, v0,sp)

def gelu(features, approximate=False, name=None):
    features = tf.convert_to_tensor(features, name="features")
    if approximate:
        coeff = tf.cast(0.044715, features.dtype)
        return 0.5 * features * (
            1.0 + tf.tanh(0.7978845608028654 *
                                (features + coeff * tf.math.pow(features, 3))))
    else:
        return 0.5 * features * (1.0 + tf.math.erf(
            features / tf.cast(1.4142135623730951, features.dtype)))  