import numpy as np 
import os 

from julia import Main
dirname = os.path.dirname(__file__)
Main.include(f"{dirname}/julia/krr.jl")

class KRR_julia():
    def __init__(self):
        self.model = Main.krr()
        

    def train(self,X,Y,kernel=None,prior=0.0,**kwargs):
        self.clean_up()
        # if type(prior) == str:
        #     if prior.casefold() == 'mean'.casefold():
        #         prior = np.mean(Y)
        self.model.setter("X",self.array_py2f_dim2(X))
        self.model.setter("Y",Y)
        self.model.setter("prior",prior)

        if kernel.casefold() == 'Gaussian'.casefold():
            self.model.setter("kernel","Gaussian")
            self.model.setter("sigma",kwargs['sigma'])
            self.model.setter("lambdav",kwargs['lmbd'])
        elif kernel.casefold() == 'periodic_Gaussian'.casefold():
            self.model.setter("kernel","periodic_Gaussian")
            self.model.setter("sigma",kwargs['sigma'])
            self.model.setter("lambdav",kwargs['lmbd'])
            self.model.setter("period",kwargs['period'])
        elif kernel.casefold() == 'decaying_periodic_Gaussian'.casefold():
            self.model.setter("kernel","decaying_periodic_Gaussian")
            self.model.setter("sigma",kwargs['sigma'])
            self.model.setter("lambdav",kwargs['lmbd'])
            self.model.setter("period",kwargs['period'])
            self.model.setter("sigmap",kwargs['sigmap'])
        elif kernel.casefold() == 'Matern'.casefold():
            self.model.setter("kernel","Matern")
            self.model.setter("sigma",kwargs['sigma'])
            self.model.setter("lambdav",kwargs['lmbd'])
            self.model.setter("nn",kwargs['nn'])
        else:
            stopper.stopMLatom(f"Unsupported kernel function type: {kernel}")
        self.model.train()
        self.alpha = self.model.alpha

        # print(self.alpha)

    def predict(self,Xpredict,calcVal):
        Yest = self.model.predict(self.array_py2f_dim2(Xpredict),None,calcVal=calcVal)
        return Yest

    def save_model(self,filename):
        keys = ["sigma", "sigmap", "period", "nn", # Hyperparameters 
                "NtrVal","NtrGr","NtrGrXYZ",
                "X",
                "kernel",
                #'XYZ', # @Yifan - why is it needed?
                "alpha",
                "prior"]
        model_dict = {}
        for key in keys:
            value = self.model.getter(key)
            if not value is None:
                model_dict[key] = value
        np.savez(filename,**model_dict)

    def load_model(self,filename):
        model = np.load(filename)
        model_keys = model.files
        for key in model_keys:
            # print(key)
            # print(model[key])
            # print(type(model[key]))
            if model[key].ndim == 0:
                self.model.setter(key,model[key].item())
            else:
                self.model.setter(key,model[key])

    def array_py2f_dim2(self,array):
        return array.T 
    
    def array_f2py_dim2(self,array):
        return array.T

    def array_py2f_dim3(self,array):
        shape = array.shape
        return np.concatenate([each.T.reshape(shape[2],shape[1],1) for each in array],axis=2)
    
    def array_f2py_dim3(self,array):
        shape = array.shape
        tmp = [array[:,:,ii].T.reshape(1,shape[1],shape[0]) for ii in range(shape[2])]
        return np.array([each[0] for each in tmp])
        

    def clean_up(self):
        self.__init__()