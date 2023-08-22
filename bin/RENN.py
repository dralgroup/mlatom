import os, sys, time
import numpy as np
import torch
import random
from  args_class import ArgsBase
from hyperopt import hp, fmin, tpe


filedir = os.path.dirname(__file__)

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'learningCurve', 'useMLmodel', 'CVopt', 'CVtest'
        ],
        bool
        )
        self.add_default_dict_args([
        'xyzfile', 'xfile','yfile', 'itrainin', 'itestin', 'isubtrainin','ivalidatein','mlmodelin','setname'
        ],
        ""
        )
        self.add_dict_args({
        'MLmodelType': 'RENN',
        'MLmodelout': 'RENNbestmodel.pt',
        'sampling': 'random',
        'yestfile': 'enest.dat',
        'nthreads': None
        })
        self.parse_input_content([
        'renn.learning_rate=0.001',
        'renn.batch_size=32',
        'renn.max_epochs=1000000',
        'renn.lr_decay_steps=16',
        'renn.lr_decay_rate=0.99',
        'renn.patience=200',
        'renn.delta=0',
        'renn.neurons=512,256,128',
        'renn.transfer_learning_fixed_layer=na'
        ])

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        if self.renn.transfer_learning_fixed_layer == 'na':
            self.renn.transfer_learning_fixed_layer = None
        else:
            self.renn.transfer_learning_fixed_layer=[int(i) for i in self.renn.transfer_learning_fixed_layer.split(',')]

        self.renn.neurons=[int(i) for i in self.renn.neurons.split(',')]
        
        if not self.mlmodelin:
            self.mlmodelin = self.mlmodelout
     
class RENNCls(object):
    dataConverted = False

    @classmethod
    def __init__(self, argsRENN = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)

    @classmethod
    def converdata(cls, argsRENN, subdatasets):
        def convert(fileout, setname, yorgrad=False):
            #convert xyz to re
            prefix = ''
            if args.learningcurve:
                prefix = '../'
            if setname:
                coordfile = prefix + 'xyz.dat_' + setname
                yfile = prefix + 'y.dat_' + setname
            else:
                coordfile = args.xyzfile
                if args.yfile and yorgrad:
                    yfile = args.yfile
            
            if args.xfile:
                pass
            else:
                mlatomdir=os.path.dirname(__file__)
                mlatomfbin="%s/MLatomF" % mlatomdir
                os.system(f'{mlatomfbin} XYZ2X XYZfile={coordfile} XfileOut={fileout}')

        args = Args()
        args.parse(argsRENN)
        print('\n Converting data...\n\n')
        if subdatasets:
            if 'Subtrain' in subdatasets:
                for i in subdatasets:
                    if i in ['Subtrain', 'Validate']: 
                        convert('RENN_'+i.lower()+'.re', i.lower(),True)

                    else: 
                        convert('RENN_'+i.lower()+'.re', i.lower())
            else:
                for i in subdatasets:
                    if i == 'Train': 
                        convert('RENN_'+i.lower()+'.re', i.lower(),True)
                    else: convert('RENN_'+i.lower()+'.re', i.lower())
        else:   
            convert('RENN.re', '')

    @classmethod
    def createMLmodel(cls, argsRENN, subdatasets):
        args = Args()
        args.parse(argsRENN)

        if not cls.dataConverted or args.learningcurve or args.cvopt or args.cvtest:
            cls.converdata(argsRENN, subdatasets)

        prefix = '../' if args.learningcurve else ''
        train_set = MyDataset(xfile='./RENN_subtrain.re', yfile=prefix+'y.dat_subtrain')
        valid_set = MyDataset(xfile='./RENN_validate.re', yfile=prefix+'y.dat_validate')
        
        starttime = time.time()
        model=MLPregression([train_set.x.shape[-1]]+args.renn.neurons+[1])
        model.setNorm(torch.mean(train_set.y,dim=0),torch.std(train_set.y,dim=0))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.renn.batch_size, 
                                            num_workers = 0, shuffle = False)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = args.renn.batch_size, 
                                            num_workers = 0, shuffle = False)

        train(model,train_loader,valid_loader,args.renn.max_epochs,args.renn.learning_rate, 
              path=args.mlmodelout, patience=args.renn.patience, delta=args.renn.delta, lr_decay_rate=args.renn.lr_decay_rate, lr_decay_steps=args.renn.lr_decay_steps)

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

    @classmethod
    def useMLmodel(cls, argsRENN, subdatasets):
        args = Args()
        args.parse(argsRENN)
        cls.dataConverted = True
        if not cls.dataConverted or args.learningcurve:
            cls.converdata(argsRENN, subdatasets)
            cls.dataConverted = True
        if args.useMLmodel:
            cls.dataConverted = False

        if args.setname: args.setname='_'+args.setname
        x = torch.from_numpy(np.genfromtxt(f'./RENN{args.setname}.re')).float()
        
        starttime = time.time()
        model = torch.load(args.mlmodelin)
        yest=predict(model, x)
        np.savetxt(args.yestfile, yest, fmt='%.16f')

        endtime = time.time()
        wallclock = endtime - starttime
        return wallclock

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, xfile, yfile):
        super(MyDataset).__init__()
        self.x = torch.from_numpy(np.genfromtxt(xfile)).float()
        self.y = torch.from_numpy(np.genfromtxt(yfile)[:,np.newaxis]).float()
    def __len__(self):
        return (len(self.y))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        return x.float(), y.float()

class MLPregression(torch.nn.Module):
    def __init__(self, neurons, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(MLPregression, self).__init__()
        self.device=device
        self.setNorm()
        layers = []
        for i in range(len(neurons) - 1):
            layers.append(torch.nn.Linear(neurons[i], neurons[i + 1]))
            # layers.append(torch.nn.BatchNorm1d(neurons[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.pop()

        self.model = torch.nn.Sequential(*layers)

    def setNorm(self,mY=0,sY=0):
        self.sY=torch.tensor(sY).to(self.device)
        self.mY=torch.tensor(mY).to(self.device)

    def forward(self, x):
        yhat = self.model(x)
        yhat = yhat * self.sY + self.mY
        return yhat

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, labels, preds):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(preds, labels))
        return loss

class EarlyStopping:
    def __init__(self, patience = 200, verbose = False, delta = 0, path = './RENNbestmodel.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.epoch = None
    def __call__(self, val_loss, train_loss, model, epoch):

        score = val_loss
        self.epoch = epoch

        if score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        # if self.verbose:
            # print('Validation RMSE: {:.8f} at epoch {}'.format(val_loss, epoch + 1))
            # print('Saving model ...')

        torch.save(model, self.path)
        self.val_loss_min = val_loss


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# generator = torch.Generator()
# generator.manual_seed(1024)

def train(model, train_loader, valid_loader, max_epochs, lr, path, patience, delta, lr_decay_rate, lr_decay_steps):
    Lambda1 = lambda epoch: lr_decay_rate**(epoch//lr_decay_steps)
    # lossfunc = RMSELoss()
    lossfunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                  lr_lambda = [Lambda1], 
                                                  last_epoch = -1)

    
    early_stopping = EarlyStopping(verbose = True, path=path, patience=patience, delta=delta)
    
    for epoch in range(max_epochs):
        model.train()
        train_se = 0
        n = 0
        for x, y in train_loader:   
            n += x.shape[0]        
            pred = model(x)
            loss = lossfunc(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_se += loss.item() * x.shape[0]
        scheduler.step()
            
        train_loss = np.sqrt(train_se/n)
        model.eval()
        valid_se = 0
        # best_loss = np.Inf
        n = 0
        with torch.no_grad():
            for x, y in valid_loader:
                n += x.shape[0]
                pred = model(x)
                loss = lossfunc(pred, y)
                valid_se += loss.item() * x.shape[0]
            
            valid_loss = np.sqrt(valid_se/n)
            # if best_loss > valid_loss:
            #     best_loss = loss
        print('Sub-training RMSE: {:.8f} Validation RMSE: {:.8f} at epoch {}'.format(train_loss, valid_loss, epoch + 1))

        # print(' at epoch {}'.format(valid_loss, epoch + 1))
        print('Learning rate: {:.18f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        # testMLP()
        early_stopping(valid_loss, train_loss, model, epoch)
        best_loss = early_stopping.val_loss_min

        if early_stopping.early_stop:
            print('early stop')
            # print('Sub-training RMSE: {}    Validation RMSE: {} at {} epoch'.format(train_loss, best_loss, early_stopping.epoch) )
            break

    return best_loss

def predict(model, x):
    model.eval()
    with torch.no_grad():
        out = model(x)
    return out.detach().numpy()

if __name__ == '__main__':
    RENNCls()