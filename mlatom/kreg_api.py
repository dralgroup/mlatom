#!/usr/bin/env python3
import mkl
import numpy as np 
from . import models
from .fortran import KREG

class KREG_API(models.model):
    def __init__(self):
        # Hyperparameters
        self.sigma         = None # Kernel width in Gaussian kernel function
        self.lambdav       = None # 
        self.lambdagradxyz = None 
        # 
        self.Natoms        = None # Number of atoms in the molecule
        self.Xsize         = None # Size of RE desciptor
        self.Ntrain        = None
        self.NtrVal        = None # Number of reference values
        self.NtrGrXYZ       = None # Number of reference gradients
        self.Ksize         = None # Size of kernel matrix
        self.ac2dArray     = None # 
        self.Req           = None # Equilibrium geometry
        self.Xeq           = None # 
        # Training
        self.prior         = 0.0
        self.XYZ           = None 
        self.X             = None # RE descriptors
        self.K             = None # Kernel matrix
        self.alpha         = None # Regression coefficients
        self.Yref          = None # Reference values
        self.YgradXYZref   = None # Reference gradients
        self.Ytrain        = None # Concatenation of values and gradients
        self.nthreads      = None


    def train(self,sigma,lambdav,lambdagradxyz,molecular_database,equilibrium_molecule,property_to_learn=None,xyz_derivative_property_to_learn=None,prior=None,shutup=True):
        # Overwrite the old model
        self.clean_up()
        KREG.mathutils.shutup=shutup
        # Preparing inputs
        self.sigma = sigma 
        self.lambdav = lambdav 
        self.lambdagradxyz = lambdagradxyz
        self.XYZ = molecular_database.xyz_coordinates
        self.Ntrain = len(molecular_database.molecules)
        self.Natoms = len(molecular_database.molecules[0].atoms)
        self.Xsize = self.Natoms * (self.Natoms-1) // 2
        self.Ytrain = np.array([])
        if property_to_learn != None:
            self.NtrVal = self.Ntrain
            self.Yref = molecular_database.get_properties(property_to_learn)
            if prior == None:
                self.prior = 0.0
            elif type(prior) == str:
                if prior.casefold() == 'mean'.casefold():
                    self.prior = np.mean(self.Yref)
            else:
                self.prior = prior
            self.Ytrain = np.concatenate([self.Ytrain,self.Yref-self.prior])

        else:
            self.NtrVal = 0
        if xyz_derivative_property_to_learn != None:
            self.NtrGrXYZ = 3 * self.Natoms * self.Ntrain
            self.YgradXYZref = molecular_database.get_xyz_vectorial_properties(xyz_derivative_property_to_learn)
            self.Ytrain = np.concatenate([self.Ytrain,self.YgradXYZref.reshape(self.NtrGrXYZ)])
        else:
            self.NtrGrXYZ = 0
        self.Ksize = self.NtrVal + self.NtrGrXYZ
        self.ac2dArray = np.asfortranarray(np.zeros((self.Natoms,self.Natoms),dtype=np.int32))
        KREG.kreg.get_ac2darray(xsize=self.Xsize,ac2darray=self.ac2dArray)
        self.Req = equilibrium_molecule.xyz_coordinates
        # Calculate interatomic distances of Req
        self.Xeq = np.zeros(self.Xsize)
        self.Xeq = KREG.kreg.get_xeq(xsize=self.Xsize,ac2darray=self.ac2dArray,req=self.array_py2f_dim2(self.Req))
        # Calculate RE descriptors
        self.X = np.zeros((self.Xsize,self.Ntrain))
        self.X = KREG.kreg.calc_re_descriptors(ac2darray=self.ac2dArray,xyz=self.array_py2f_dim3(self.XYZ),xeq=self.Xeq)
        self.X = self.array_f2py_dim2(self.X)

        # Train 
        self.K = np.zeros((self.Ksize,self.Ksize))
        self.alpha = np.zeros((self.Ksize,1))
        self.K,self.alpha = KREG.kreg.train(ntrval=self.NtrVal,ntrgrxyz=self.NtrGrXYZ,ac2darray=self.ac2dArray,xyz=self.array_py2f_dim3(self.XYZ),x=self.array_py2f_dim2(self.X),ytrain=self.Ytrain,sigma=self.sigma,lambdav=self.lambdav,lambdagradxyz=self.lambdagradxyz,calckernel=True)
        self.alpha = self.array_f2py_dim2(self.alpha)



    def predict(self, molecular_database=None,molecule=None,property_to_predict=None, xyz_derivative_property_to_predict=None):
        # Assuming that you have trained or loaded a model
        if property_to_predict != None:
            calculate_energy = True 
        else:
            calculate_energy = False 
        if xyz_derivative_property_to_predict != None:
            calculate_energy_gradients = True 
        else:
            calculate_energy_gradients = False
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)
        XYZpredict = molDB.xyz_coordinates
        Npredict = len(XYZpredict)
        #print(self.ac2dArray)
        Xpredict = KREG.kreg.calc_re_descriptors(ac2darray=self.ac2dArray,xyz=self.array_py2f_dim3(XYZpredict),xeq=self.Xeq)
        Xpredict = self.array_f2py_dim2(Xpredict)

        # Predict
        Yest = np.zeros(Npredict)
        YgradXYZest = np.zeros((3,self.Natoms,Npredict))
        if self.nthreads != None:
            import os
            os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
            os.environ["MKL_NUM_THREADS"] = str(self.nthreads)
        Yest, YgradXYZest = KREG.kreg.predict(ntrval=self.NtrVal,ntrgrxyz=self.NtrGrXYZ,ac2darray=self.ac2dArray,x=self.array_py2f_dim2(self.X),xyz=self.array_py2f_dim3(self.XYZ),xpredict=self.array_py2f_dim2(Xpredict),xyzpredict=self.array_py2f_dim3(XYZpredict),alpha=self.array_f2py_dim2(self.alpha),calcval=calculate_energy,calcgradxyz=calculate_energy_gradients,sigma=self.sigma)
        Yest = Yest + self.prior
        YgradXYZest = self.array_f2py_dim3(YgradXYZest)
        # Get predictions
        if calculate_energy:
            molDB.add_scalar_properties(Yest,property_name=property_to_predict)
        if calculate_energy_gradients:
            molDB.add_xyz_vectorial_properties(YgradXYZest,xyz_vectorial_property=xyz_derivative_property_to_predict)


    def save_model(self,filename):
        keys = ['sigma', 'Natoms', 'Xsize', 'Ntrain', 'NtrVal', 'NtrGrXYZ',
                'lambdav', 'lambdagradxyz', # strictly speeking, not needed
                'X',
                #'Ksize', 'K', # Not needed
                'XYZ', # @Yifan - why is it needed?
                #'Yref', 'YgradXYZref', 'Ytrain', 
                'ac2dArray', # can be recovered
                'Req', 'Xeq',
                'alpha',
                'prior']
        model_dict = {}
        for key in keys:
            if key in self.__dict__:
                if type(self.__dict__[key]) != type(None):
                    model_dict[key] = self.__dict__[key]
        np.savez(filename,**model_dict)

    def load_model(self,filename):
        model = np.load(filename)
        model_keys = model.files
        for key in model_keys:
            self.__dict__[key] = model[key]

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

    
    