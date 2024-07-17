
from .utils import *
from .functionsGICnet import *
import sys
import time
import json
import uuid
from pyh5md import File, element
from . import stopper

from . import data
from . import models
from .decorators import doc_inherit
              
            
class GICnet(object):
    hyperparameters = models.hyperparameters({
        'batchSize3D':           models.hyperparameter(value=16),
        'batchSize4D':           models.hyperparameter(value=16),
        'maxEpoch3D':           models.hyperparameter(value=1024),
        'maxEpoch4D':           models.hyperparameter(value=1024),
        'ICidx':           models.hyperparameter(value=''),
        'ICdict':           models.hyperparameter(value={}),
        'use3D':           models.hyperparameter(value=1),
        'tc':           models.hyperparameter(value=10),
        'tMax':           models.hyperparameter(value=0),
        'M2Vbar':           models.hyperparameter(value=0),
        'M2dM':           models.hyperparameter(value=0),
        'logisticateT':           models.hyperparameter(value=0),
        'normalizeT':           models.hyperparameter(value=1),
        'xList':           models.hyperparameter(value='Descriptor'),
        'yList':           models.hyperparameter(value='Descriptor,ep'),
        'Descriptor':           models.hyperparameter(value='ic'),
        'Descriptor3D':           models.hyperparameter(value=0),
        'adaptSeg':           models.hyperparameter(value=0),
        'forceEtot':           models.hyperparameter(value=0),
        'checkEk':           models.hyperparameter(value=0),
        'checkV0':           models.hyperparameter(value=0),
        'reuseData':           models.hyperparameter(value=0),
        'reuse3D':           models.hyperparameter(value=0),
        'reuse4D':           models.hyperparameter(value=0),
        'Nsubtrain':           models.hyperparameter(value=0.95),
        'Nvalidate':           models.hyperparameter(value=0.05),
        'NsubtrainMax':           models.hyperparameter(value=0),
        'NvalidateMax':           models.hyperparameter(value=0),
    })
    def __init__(self, model_file=None, devices=None, hyperparameters={},) -> None:
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        
        self.meta={
            "descriptor":{},
            "data":{},
            "3Dmodel":{},
            "4Dmodel":{}
        }
        
        self.model3D=None
        self.model=None
        self.model3Ds=[]
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

        self.normalizeTfactor=0.32
        # self.strategy = tf.distribute.MirroredStrategy(devices)
        

        self.init_lr=0.001
        self.decay_steps=8192
        self.decay_rate=0.99
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.init_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate)
        
        if model_file: 
            if type(model_file)==list or os.path.isdir(model_file):
                self.load(model_file)
            self.model_file = model_file
        else:
            self.model_file = f'GICnet_{str(uuid.uuid4())}'
            
    def load(self, model_file):
        if type(model_file) == str:
            model_file = [model_file]
        with open(f'{model_file[0]}/meta.json') as f: self.meta=json.load(f)
        self.masses=self.meta['masses']
        self.hyperparameters.tc=self.meta['4Dmodel']['Tc']
        self.hyperparameters.Descriptor=self.meta['descriptor']['type']
        self.hyperparameters.ICdict=self.meta['descriptor']['ICdict']
        self.hyperparameters.M2dM=self.meta['descriptor']['M2dM']
        self.hyperparameters.M2Vbar=self.meta['descriptor']['M2Vbar']
        self.hyperparameters.logisticateT=self.meta['descriptor']['logisticateT']
        self.hyperparameters.normalizeT=self.meta['descriptor']['normalizeT']
        self.hyperparameters.Descriptor3D=self.meta['descriptor']['3D']

        if len(model_file) == 1:
            if self.meta['3Dmodel']['use']:
                self.model3D= tf.keras.models.load_model(f'{model_file[0]}/3D_model.keras',compile=False)  
                self.epoch3D=self.meta['3Dmodel']['Epoch']
                self.loss3D=self.meta['3Dmodel']['loss']
            self.model=tf.keras.models.load_model(f'{model_file[0]}/4D_model.keras',compile=False)
            self.epoch4D=self.meta['4Dmodel']['Epoch']
            self.losses4D=tuple(self.meta['4Dmodel']['losses'])
            self.loss4D=self.losses4D[-1]
        else:
            for model in model_file:
                if self.meta['3Dmodel']['use']:
                    self.model3Ds.append(tf.keras.models.load_model(f'{model}/3D_model.keras',compile=False))
                self.models.append(tf.keras.models.load_model(f'{model}/4D_model.keras',compile=False))
            self.model3D=lambda x: tf.reduce_mean([m3D(x) for m3D in self.model3Ds],axis=0)
    
    def train(self, trajectories):
        if self.hyperparameters.Descriptor.lower()=='ic' and not self.hyperparameters.ICdict: self.hyperparameters.ICdict=readIC(self.hyperparameters.ICidx) 
        if self.hyperparameters.reuseData or self.hyperparameters.reuse3D or self.hyperparameters.reuse4D:
            with open(f'{self.model_file}/meta.json') as f:
                self.meta=json.load(f)
        else: 
            os.system(f'mkdir {self.model_file}')
            
        self.writeMeta()
        self.create4Dmodel(trajectories)

    def writeMeta(self):
        self.meta['descriptor']['type']=self.hyperparameters.Descriptor
        self.meta['descriptor']['xList']=self.hyperparameters.xList.split(',')
        self.meta['descriptor']['yList']=self.hyperparameters.yList.split(',')
        self.meta['descriptor']['ICdict']=dict(self.hyperparameters.ICdict)
        self.meta['descriptor']['3D']=self.hyperparameters.Descriptor3D
        self.meta['3Dmodel']['use']=self.hyperparameters.use3D
        if self.meta['3Dmodel']['use']: self.meta['3Dmodel']['batchSize']=self.hyperparameters.batchSize3D
        self.meta['4Dmodel']['batchSize']=self.hyperparameters.batchSize4D
        self.meta['4Dmodel']['Tc']=self.hyperparameters.tc
        self.meta['descriptor']['M2Vbar']=self.hyperparameters.M2Vbar
        self.meta['descriptor']['M2dM']=self.hyperparameters.M2dM
        self.meta['descriptor']['logisticateT']=self.hyperparameters.logisticateT
        self.meta['descriptor']['normalizeT']=self.hyperparameters.normalizeT
        self.meta['data']['Nsubtrain']=self.hyperparameters.Nsubtrain
        self.meta['data']['Nvalidate']=self.hyperparameters.Nvalidate
        self.meta['data']['tMax']=self.hyperparameters.tMax
        
        with open(f'{self.model_file}/meta.json','w') as f:
            json.dump(self.meta,f,indent=4)
    
    def prepareData(self, trajs):
        args=self.hyperparameters
        global getDataFromTraj

        def getDataFromTraj(itraj):
            traj = trajs[itraj]
            if isinstance(traj, data.molecular_trajectory):
                mdb = data.molecular_database([step.molecule for step in traj.steps])
                dataset={}
                dataset['xyz']=mdb.xyz_coordinates
                dataset['t']=np.array([step.time for step in traj.steps]).reshape(-1,1)
                dataset['v']=mdb.get_xyz_vectorial_properties("xyz_velocities")
                dataset['ep']=mdb.get_properties('energy').reshape(-1,1)
                dataset['ek']=mdb.get_properties('kinetic_energy').reshape(-1,1)
            elif traj[-4:]=='.npz':
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
            
            if args.tMax:
                tMax=args.tMax
            else:
                tMax=np.max(dataset['t'])
            
            mask=dataset['t'][:,0]<=tMax
            for k,v in dataset.items():
                try: 
                    dataset[k]=v[mask]
                except: 
                    # print(f" problem with {traj}'s {k}")
                    pass
            
            if 'et' in args.xList+args.yList:
                dataset['et']=dataset['ek']+dataset['ep']
            if 'deltaE' in args.xList+args.yList:
                dataset['deltaE']=np.zeros_like(dataset['ek'])
            if args.Descriptor.lower()=='ic':
                dataset['Descriptor']=Describe(dataset['xyz'],dataset['v'],args.ICdict,m=self.masses)
            elif args.Descriptor.lower()=='xyz':
                dataset['Descriptor']=np.concatenate((dataset['xyz'],dataset['v']),axis=1).reshape(dataset['xyz'].shape[0],-1)
            x,y=getData(dataset,self.xList,self.yList,args.tc,0,tMax*args.Nsubtrain,0,step=1)
            # xEx,yEx=getData(dataset,xList,yList,args.tc,0,tMax*(args.Nsubtrain+args.Nvalidate),tMax*args.Nsubtrain,step=1)
            xEx,yEx=getData(dataset,self.xList,self.yList,args.tc,0,tMax*(args.Nsubtrain+args.Nvalidate),tMax*args.Nsubtrain,step=1)
            if 'deltaE' in args.xList+args.yList:
                ek0=getEks(dataset['v'],dataset['m'][0])
                dataset['v']+=np.random.normal(0,0.01,dataset['v'].shape)
                ek1=getEks(dataset['v'],dataset['m'][0])
                dataset['deltaE']=ek1-ek0
                xx,yy=getData(dataset,self.xList,self.yList,args.tc,0,tMax*args.Nsubtrain,0,step=1)
                xxEx,yyEx=getData(dataset,self.xList,self.yList,args.tc,0,tMax*(args.Nsubtrain+args.Nvalidate),tMax*args.Nsubtrain,step=1)
                x=np.concatenate((x,xx),axis=0)
                y=np.concatenate((y,yy),axis=0)
                xEx=np.concatenate((xEx,xxEx),axis=0)
                yEx=np.concatenate((yEx,yyEx),axis=0)
            if not args.Descriptor3D:
                x3D=dataset['Descriptor'][:,:dataset['Descriptor'].shape[1]//2]
            elif args.Descriptor3D.lower()=='id':
                x3D=IDescr(dataset['xyz'])
            y3D=dataset['ep']

            xs[itraj]=x
            ys[itraj]=y
            xExs[itraj]=xEx
            yExs[itraj]=yEx
            x3Ds[itraj]=x3D
            y3Ds[itraj]=y3D

        if args.reuseData:
            x=np.load(f'{self.model_file}/x.npy')
            y=np.load(f'{self.model_file}/y.npy')
            xEx=np.load(f'{self.model_file}/xEx.npy')
            yEx=np.load(f'{self.model_file}/yEx.npy')
            x3D=np.load(f'{self.model_file}/x3D.npy')
            y3D=np.load(f'{self.model_file}/y3D.npy')
            
        else:
            if isinstance(trajs[0], data.molecular_trajectory):
                self.masses = trajs[0].steps[0].molecule.nuclear_masses
            elif trajs[0][-4:]=='.npz':
                self.masses=dict(np.load(trajs[0]))['m'][0]
            elif trajs[0][-3:]=='.h5':
                with File(trajs[0],'r') as f:
                    self.masses=f['particles/all/mass'][()]
            self.meta['masses']=list(self.masses.astype(float))
            self.meta['descriptor']['length']=len(self.meta['masses'])*6
            self.writeMeta()

            manager = Manager()
            xs=manager.dict()
            ys=manager.dict()
            xExs=manager.dict()
            yExs=manager.dict()
            x3Ds=manager.dict()
            y3Ds=manager.dict()

            print(' preparing data...')
            pool = Pool(os.cpu_count())
            pool.map(getDataFromTraj, range(len(trajs)))
            pool.close()
            x=np.concatenate([xs[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            y=np.concatenate([ys[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            xEx=np.concatenate([xExs[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            yEx=np.concatenate([yExs[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            x3D=np.concatenate([x3Ds[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            y3D=np.concatenate([y3Ds[traj] for traj in range(len(trajs))], axis=0).astype(np.float32)
            np.save(f'{self.model_file}/x.npy',x)
            np.save(f'{self.model_file}/y.npy',y)
            np.save(f'{self.model_file}/xEx.npy',xEx)
            np.save(f'{self.model_file}/yEx.npy',yEx)
            np.save(f'{self.model_file}/x3D.npy',x3D)
            np.save(f'{self.model_file}/y3D.npy',y3D)


        x,y=shuffle(x,y)
        xEx,yEx=shuffle(xEx,yEx)
        x3D,y3D=shuffle(x3D,y3D)

        if args.NvalidateMax:
            xEx,yEx=xEx[:args.NvalidateMax],yEx[:args.NvalidateMax]
        if args.NsubtrainMax:
            x,y=x[:args.NsubtrainMax],y[:args.NsubtrainMax]

        l=self.meta['descriptor']['length']//2

        xEx3D=x3D[y.shape[0]:]
        yEx3D=y3D[y.shape[0]:]
        x3D=x3D[:y.shape[0]]
        y3D=y3D[:y.shape[0]]
        
        if args.reuse3D:
            m_shift=self.meta['3Dmodel']['m_shift']
            m_scale=self.meta['3Dmodel']['m_scale']
            ep_shift=self.meta['3Dmodel']['ep_shift']
            ep_scale=self.meta['3Dmodel']['ep_scale']
        else:
            m_shift=np.mean(x3D,axis=0)
            m_scale=np.std(x3D,axis=0)
            # m_shift=np.zeros_like(m_shift)
            # m_scale=np.ones_like(m_scale)
            ep_shift=np.mean(y3D)
            ep_scale=np.std(y3D)
            self.meta['3Dmodel']['m_shift']=list(m_shift.astype(float))
            self.meta['3Dmodel']['m_scale']=list(m_scale.astype(float))
            self.meta['3Dmodel']['ep_shift']=float(ep_shift)
            self.meta['3Dmodel']['ep_scale']=float(ep_scale)
            self.writeMeta()

        x3D-=m_shift
        x3D/=m_scale
        y3D-=ep_shift
        y3D/=ep_scale
        xEx3D-=m_shift
        xEx3D/=m_scale
        yEx3D-=ep_shift
        yEx3D/=ep_scale


        if args.Descriptor.lower()=='ic':
            la=int(l/3)
            ld=la

            print("     correcting angles...")
            correctAngle(x,y,idx=list(range(ld,l)))
            correctAngle(xEx,yEx,idx=list(range(ld,l)))
            
        if args.normalizeT:
            print("     normalized by time")
            x,y=normalizeT(x,y,l=l,a=self.normalizeTfactor)
            xEx,yEx=normalizeT(xEx,yEx,l=l,a=self.normalizeTfactor)

        elif args.m2vbar:
            print('     learning average velocities of descriptors')
            x,y=M2Vbar(x,y)
            xEx,yEx=M2Vbar(xEx,yEx)

        elif args.M2dM:
            print('     learning differences of descriptors')
            x,y=M2dM(x,y)
            xEx,yEx=M2dM(xEx,yEx)

        elif args.logisticatet:
            print("     time normalized with logistic function")
            x,y=logisticateT(x,y)
            xEx,yEx=logisticateT(xEx,yEx)
        if args.reuse4D:
            x_shift=self.meta['4Dmodel']['x_shift']
            x_scale=self.meta['4Dmodel']['x_scale']
            y_shift=self.meta['4Dmodel']['y_shift']
        else:
            x_shift=np.mean(x,axis=0)
            x_scale=np.std(x,axis=0)
            y_shift=np.mean(y,axis=0)
            # if args.M2dM or args.normalizeT:
            #     x_shift[:]=0
            #     x_scale[:]=1
            #     y_shift[:]=0
            
            x_shift[:]=0
            x_scale[:]=1
            y_shift[:]=0
            self.meta['4Dmodel']['x_shift']=list(x_shift.astype(float))
            self.meta['4Dmodel']['x_scale']=list(x_scale.astype(float))
            self.meta['4Dmodel']['y_shift']=list(y_shift.astype(float))
            self.writeMeta()

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

    def create3Dmodel(self):
        args=self.hyperparameters
            
        # with self.strategy.scope():
        if args.reuse3D:
            self.model3D=tf.keras.models.load_model(f'{self.model_file}/3D_model',compile=False)  
            self.epoch3D=self.meta['3Dmodel']['Epoch']
            self.loss3D=self.meta['3Dmodel']['loss']
            if self.epoch3D + 2 > args.maxEpoch3D:
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
        args=self.hyperparameters
        ntrain=self.x3D.shape[0]
        m_shift=self.m_shift
        m_scale=self.m_scale
        ep_shift=self.ep_shift
        ep_scale=self.ep_scale
        
        # with self.strategy.scope():
        self.optimizer3D = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        training_set3D = tf.data.Dataset.from_tensor_slices((self.x3D, self.y3D)).shuffle(256).batch(args.batchSize3D)
        # dist_training_set3D=self.strategy.experimental_distribute_dataset(training_set3D)
        test_set3D = tf.data.Dataset.from_tensor_slices((self.xEx3D, self.yEx3D)).shuffle(256).batch(args.batchSize3D)
        

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
        
        # global bestloss, bestlosses

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
                # step_start = time.time()
                loss_train=training_step3D(data)

            # train_loss=validate_3D(self.model3D,training_set3D)*ep_scale
            ex_loss=validate_3D(self.model3D,test_set3D)*ep_scale
            # print(self.optimizer3D._decayed_lr(tf.float32).numpy())
            # ex_loss1=validate_3D(model3D,test_set3D1)*ep_scale
            if ex_loss < self.loss3D:
                self.model3D.save(f'{self.model_file}/3D_model.keras')
                self.loss3D=ex_loss
                self.meta['3Dmodel']['Epoch']=self.epoch3D
                self.meta['3Dmodel']['loss']=self.loss3D.numpy().astype(float)
                self.writeMeta()

            
            now=time.time()
            t_epoch=now-epoch_start
            print('\repoch %-6d %5d/%-5d         t_epoch: %10.1f       ' % (self.epoch3D,step,ntrain//args.batchSize3D,t_epoch))
            # print('train:     %-10.6f'% train_loss)
            print('validate:  %-10.6f'% ex_loss)
            # print('validate:  %-10.6f'% ex_loss1)            
            print('best:      %-10.6f'% self.loss3D)
            sys.stdout.flush()
            self.epoch3D+=1
        
        print(' training 3D model')        
        print("     Ntrain: ",self.y3D.shape[0])

        while True:
            train(training_set3D)
            if self.epoch3D >= args.maxEpoch3D: 
                break

        print(' 3D model trained')

    def create4Dmodel(self, trajs):
        args=self.hyperparameters
        
        self.xList=self.meta['descriptor']['xList']
        self.yList=self.meta['descriptor']['yList']

        self.prepareData(trajs)
        
        l=self.meta['descriptor']['length']//2

        if args.use3D:
            self.create3Dmodel()
        # with self.strategy.scope():
        if args.reuse4D:
            self.model=tf.keras.models.load_model(f'{self.model_file}/4D_model',compile=False)
            self.epoch4D=self.meta['4Dmodel']['Epoch']
            self.losses4D=tuple(self.meta['4Dmodel']['losses'])
            self.loss4D=self.losses4D[-1]
            if self.epoch4D + 2 > args.maxEpoch4D:
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
        args=self.hyperparameters
        l=self.meta['descriptor']['length']//2
        if args.Descriptor.lower()=='ic':
            icidx=tf.constant(self.meta["descriptor"]["ICdict"]['idx'],dtype=tf.int32)
        m=tf.constant(self.meta['masses'],dtype=tf.float32)
        ntrain=self.x.shape[0]
        m_shift=self.m_shift
        m_scale=self.m_scale
        ep_shift=self.ep_shift
        ep_scale=self.ep_scale
        x_shift=self.x_shift
        x_scale=self.x_scale
        y_shift=self.y_shift

        # with self.strategy.scope():
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        
        training_set = tf.data.Dataset.from_tensor_slices((self.x, self.y)).shuffle(256).batch(args.batchSize4D)
        test_set = tf.data.Dataset.from_tensor_slices((self.xEx, self.yEx)).shuffle(256).batch(args.batchSize4D)
        # dist_training_set=self.strategy.experimental_distribute_dataset(training_set)

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

            if args.checkV0:
                x0=tf.concat((x[:,:-1],x[:,-1:]*0),axis=1)
                v0p=(model(x0)+y_shift[:l])*10
                v0t=x0[:,l:2*l]
                v0loss=tf.sqrt(tf.reduce_mean(tf.square(v0t-v0p)*w))
                ekloss+=v0loss

            if args.normalizeT:
                xx=x*x_scale+x_shift
                t=xx[:,-1][:,tf.newaxis]
                a=self.normalizeTfactor
                ft=(1-tf.exp(-(t/a)**2))
                dftdt=2*t/a**2*tf.exp(-(t/a)**2)
                mpic=mp*ft+xx[:,:l]+xx[:,l:2*l]*t*(1-ft)
                mtic=mt*ft+xx[:,:l]+xx[:,l:2*l]*t*(1-ft)
                if args.checkEk:
                    vpic=vp*ft+dftdt*mp+xx[:,l:2*l]*(1-dftdt*t-ft)
                    vpic=vt*ft+dftdt*mp+xx[:,l:2*l]*(1-dftdt*t-ft)
                w=tf.exp(-t)+tf.exp(t-args.tc)+1
            else:
                mpic=mp
                mtic=mt
                vpic=vp
                vtic=vp

            if args.checkEk:
                mpxyz,vpxyz=unDescribeTF(tf.concat((mpic,vpic),1),icidx)
                mtxyz,vtxyz=unDescribeTF(tf.concat((mtic,vtic),1),icidx)
                Ekp=getEksTF(vpxyz,m)
                Ekt=yy[:,-2] if 'ek' in args.yList else getEksTF(vtxyz,m)
                ekloss+=tf.sqrt(tf.reduce_mean(tf.square(Ekp-Ekt)*w))

            if model3D:
                epest=model3D(((tf.math.floormod(mpic+semidivisor,2*semidivisor)-semidivisor)-m_shift)/m_scale)
                epref=model3D(((tf.math.floormod(mtic+semidivisor,2*semidivisor)-semidivisor)-m_shift)/m_scale)
                eperr=(epest-epref)*ep_scale
                eploss=tf.sqrt(tf.reduce_mean(tf.square(eperr)*w))


            mloss=tf.sqrt(tf.reduce_mean(tf.square(mt-mp)*w,axis=0))
            vloss=tf.sqrt(tf.reduce_mean(tf.square(vt-vp)*w,axis=0))
            Dloss=tf.reduce_mean(mloss[:ld])
            Aloss=tf.reduce_mean(mloss[ld:ld+la])
            DAloss=tf.reduce_mean(mloss[ld+la:])
            vDloss=tf.reduce_mean(vloss[:ld])
            vAloss=tf.reduce_mean(vloss[ld:ld+la])
            vDAloss=tf.reduce_mean(vloss[ld+la:])

            loss=1.0*Dloss+2.0*Aloss+4.0*DAloss+16.0*vDloss+20.0*vAloss+24.0*vDAloss+1.0*eploss+16*ekloss
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

            if args.normalizeT:
                xx=x*x_scale+x_shift
                t=xx[:,-1][:,tf.newaxis]
                ft=1-tf.exp(-10*t)
                dftdt=10*tf.exp(-10*t)
                mpxyz=mp*ft+xx[:,:l]
                if args.checkEk:
                    vpxyz=vp*ft+dftdt*mp
            else:
                mpxyz=mp
                vpxyz=vp

            mpxyz=tf.reshape(mpxyz,[mpxyz.shape[0],-1,3])
            vpxyz=tf.reshape(vpxyz,[vpxyz.shape[0],-1,3])

            if model3D:
                x3D=((IDescrTF(mpxyz)-m_shift)/m_scale)
                eploss=tf.sqrt(tf.reduce_mean(tf.square(model3D(x3D)*ep_scale+ep_shift-yy[:,-1:])))
            if args.checkEk:
                vtxyz=tf.reshape(vt,[vt.shape[0],-1,3])
                Ekp=getEksTF(vpxyz,m)
                Ekt=yy[:,-2] if 'ek' in args.yList else getEksTF(vtxyz,m)
                ekloss=tf.sqrt(tf.reduce_mean(tf.square(Ekp-Ekt)))
            

            loss=1.0*mloss+8.0*vloss+1.0*eploss+1.0*ekloss
            return mloss,vloss,eploss,ekloss,loss

        if args.Descriptor.lower()=='ic':
            la=l//3
            ld=la
            lda=la
            semidivisor=tf.constant([100]*ld+[np.pi]*(la+lda))
            loss_fn=loss_fn_ic
        elif args.Descriptor.lower()=='xyz':
            loss_fn=loss_fn_xyz

        def validate(model,dataset):
            se=np.zeros(9 if args.Descriptor.lower()=='ic' else 5)
            count=0
            for data in dataset:
                n=data[0].shape[0]
                se+=np.square(loss_fn(model,*data,model3D=self.model3D if args.use3D else None))*n
                count+=n
            return tuple(np.sqrt(se/count))

        @tf.function
        def training_step(data):
            xx,yy=data
            with tf.GradientTape() as tape:
                tape.watch(xx)
                losses=loss_fn(self.model,xx,yy,model3D=self.model3D if args.use3D else None)
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
                losses=training_step(data)

            train_losses=validate(self.model,training_set)
            ex_losses=validate(self.model,test_set)
            if ex_losses[-1] < self.loss4D:
                self.model.save(f'{self.model_file}/4D_model.keras')
                self.loss4D=ex_losses[-1]
                self.losses4D=ex_losses
                self.meta['4Dmodel']['Epoch']=self.epoch4D
                self.meta['4Dmodel']['losses']=[loss.astype(float) for loss in self.losses4D]
                self.writeMeta()
                
            now=time.time()
            t_epoch=now-epoch_start
            print('\repoch %-6d %5d/%-5d         t_epoch: %10.1f       ' % (self.epoch4D,step,ntrain//args.batchSize4D,t_epoch))
            if args.Descriptor.lower()=='ic':
                print('%-10s %-10s %-10s %-10s %-10s'% ("","D","A","DA","Ep/Ek"))
                print('train:     %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% train_losses)
                print('validate:  %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% ex_losses)            
                print('best:      %-10.6f %-10.6f %-10.6f %-10.6f\n  d/dt     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% self.losses4D)
            elif args.Descriptor.lower()=='xyz':
                print('           %-10s %-10s %-10s %-10s'% ("M","v","Ep","Ek"))
                # print('train:     %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% train_losses)   
                print('validate:  %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% ex_losses)            
                print('best:      %-10.6f %-10.6f %-10.6f %-10.6f\n                                 Loss       %-10.6f'% self.losses4D)
            sys.stdout.flush()
            self.epoch4D+=1
            
        
        print('training 4D model')
        print("     Ntrain: ",self.y.shape[0])
        while True:
            train(training_set)
            if self.epoch4D >= args.maxEpoch4D: 
                break
        print('4D model trained')

    def propagate(self, molecule, maximum_propagation_time=1000.0, time_step=1.0, time_segment='tc'):
        args=self.hyperparameters
        time_segment=args.tc if time_segment=='tc' else time_segment
            
        icdict=self.meta["descriptor"]["ICdict"]
        m=np.array(self.meta['masses'])
        if self.model3D:
            m_shift=self.meta['3Dmodel']['m_shift']
            m_scale=self.meta['3Dmodel']['m_scale']
            ep_shift=self.meta['3Dmodel']['ep_shift']
            ep_scale=self.meta['3Dmodel']['ep_scale']

        x_shift=np.array(self.meta['4Dmodel']['x_shift']).astype(np.float32)
        x_scale=np.array(self.meta['4Dmodel']['x_scale']).astype(np.float32)
        y_shift=np.array(self.meta['4Dmodel']['y_shift']).astype(np.float32)

        l=self.meta['descriptor']['length']//2
        la=l//3
        ld=la
        lda=la
        tm=maximum_propagation_time

        molecular_trajectory = data.molecular_trajectory()

        xyz = molecule.xyz_coordinates
        v = molecule.get_xyz_vectorial_properties("xyz_velocities")
        sp = molecule.element_symbols


        xyzoffset=getCoM(xyz,m)
        voffset=getCoM(v,m)
        x0=Describe(xyz[np.newaxis],v[np.newaxis],icdict,m=m).astype(np.float32)
        ts=np.arange(0,tm,time_step)+time_step
        ts[-1]=tm
        t0=0

        ek=getEk(v,m)[np.newaxis]
        if self.model3D:
            ep=self.model3D((x0[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
        else:
            ep=np.zeros_like(ek)
        etot=(ep+ek)[0]
        P0=getLinM(v,m)
        L0=getAngM(xyz,v,m)
        Eerror=0
                
        mole = molecule.copy()
        mole.energy =  ep[0]
        mole.total_energy = mole.energy + mole.kinetic_energy
        trajectory_step = data.molecular_trajectory_step()
        trajectory_step.step = 0
        trajectory_step.time = 0.0
        trajectory_step.molecule = mole 
        molecular_trajectory.steps.append(trajectory_step)
        

        def getY(x, model):
            y=differentiation(model,x,x_shift=x_shift,x_scale=x_scale,y_shift=y_shift,l=l).numpy()
            if np.sum(np.isnan(y)) > 0:
                stopper.stopMLatom('prediction failed')
                
            if args.normalizeT:
                x,y=unnormalizeT(x,y,l=l,a=self.normalizeTfactor)
            elif args.M2Vbar:
                x,y=Vbar2M(x,y)
            elif args.M2dM:
                x,y=dM2M(x,y)
            elif args.logisticatet:
                x,y=unlogisticateT(x,y)
            theOppositeWayToCorrectAngle(y,idx=list(range(ld,l)))
            return y
        
        def getAll(x, model):
            y=getY(x, model)
            xyz,v=unDescribe(y,icdict['idx'])
            xyz=xyz+xyzoffset[np.newaxis,:]+voffset[np.newaxis,:]*(t[:,np.newaxis,np.newaxis]+t0)
            v=v-getCoM(v,m)[:,np.newaxis,:]+voffset[np.newaxis,:]
            # x0=y[[-1]]
            for i in range(len(v)):
                v[i]=adjV(xyz[i],v[i],m,P0,L0)
                # x0=Describe(xyz[[-1]],v[[-1]],icdict,m=m).astype(np.float32)
            ek=getEks(v,m)
            ep=np.zeros_like(ek)
            if self.model3D:
                ep=self.model3D((y[:,:l]-m_shift)/m_scale).numpy().flatten()*ep_scale+ep_shift
            
            if args.forceEtot:
                v=v*np.sqrt((etot-ep)/ek)[:,np.newaxis,np.newaxis]
                ek=getEks(v,m)
                # x0=Describe(xyz[[-1]],v[[-1]],icdict,m=m).astype(np.float32)

            Eerror=(ep+ek)-etot
            
            return y, xyz, v, ek ,ep, Eerror#, x0

        def findRoot(x0,t0,tt,y0=None,threshold=0.001/627.5,i=0):
            print('finding root between %.4f and %.4f'% (t0,tt))
            i+=1
            tm=(t0+tt)/2
            ts=[[tm]] if y0 else [[t0],[tm]] 
            # ts=[[t0],[tm],[tt]]
            x=np.concatenate((np.repeat(x0,len(ts),axis=0),ts),axis=1)
            y=getY(x, self.model)
            _,v=unDescribe(y,icdict['idx'])
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

            xyz,v=unDescribe(y,icdict['idx'])
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
            # print(np.min(eterr)*627.5,eterr[np.argmin(eterr)],ek[np.argmin(eterr)],ep[np.argmin(eterr)],etot,t[np.argmin(eterr)])
            print(' root not found, using point with least Etot error for the next step...')
            # print(np.min(eterr)*627.5,t[np.argmin(eterr)])
            return t[np.argmin(np.abs(eterr))]

        def saveResult(result, tstamps):
            y, xyz, v, ek ,ep, _ =result
            for i in range(len(tstamps)):
                mole = molecule.copy()
                mole.xyz_coordinates =  xyz[i]
                mole.add_xyz_vectorial_property(v[i], 'xyz_velocities')
                mole.energy =  ep[i]
                mole.total_energy = mole.energy + mole.kinetic_energy
                trajectory_step = data.molecular_trajectory_step()
                trajectory_step.step = len(molecular_trajectory.steps)
                trajectory_step.time = tstamps[i]
                trajectory_step.molecule = mole 
                molecular_trajectory.steps.append(trajectory_step)

        Segms=[]
        pdict={0:0}
        tbreak=0
        threshold=1
        while t0 < tm:
            if args.adaptSeg:
                time_segment=findNextT0(x0)

            if t0 not in pdict.keys(): pdict[t0]=0
            if t0 > tbreak: threshold=10
            t=ts[(ts>t0+0.000001)&(ts<t0+time_segment+0.000001)]-t0
            x=np.concatenate((np.repeat(x0,len(t),axis=0),t[:,np.newaxis]),axis=1)
            xSegm=np.concatenate((x0,[[time_segment]]),axis=1)

            if self.models:
                results=[]
                for model in self.models:
                    results.append(getAll(xSegm, model))
                Eerrors=np.abs([result[-1][-1] for result in results])
                sort=np.argsort(Eerrors)
                # print(sort[pdict[t0]])
                self.model=self.models[sort[pdict[t0]]]
                x0, *_, Eerror=results[sort[pdict[t0]]]
                if np.abs(Eerror[-1])*627.5>threshold or pdict[t0]>=len(sort)-1:
                    # print(t0,pdict[t0],len(Segms))
                    tbreak=max(t0,tbreak)
                    if len(Segms)>1:
                        Segms.pop(-1)
                        t0-=time_segment
                        x0=Segms[-1][-1]
                        pdict[t0]+=1
                    else:
                        threshold*=2
                        pdict[t0]=0
                    for k in pdict.keys():
                        if k>t0:
                            pdict[k]=0
                    continue

            else:
                x0, *_, Eerror =getAll(xSegm, self.model)
            Segms.append([getAll(x, self.model),['%.3f fs %s' % (i, 'model'+str(sort[pdict[t0]]) if self.models else '') for i in t+t0],x0])
            if len(Segms)>10:
                saveResult(*Segms[0][:-1])
                Segms.pop(0)
            
            print(' %.2f-%.2f fs'% (t0,t0+time_segment))
            print(f'Etot error: {(Eerror[-1])*627.5} kcal/mol')
            
            sys.stdout.flush()
            t0+=time_segment
        
        for Segm in Segms:
            saveResult(*Segm[:-1])
        return molecular_trajectory


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