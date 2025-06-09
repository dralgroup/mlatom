import mlatom as ml
import timeit 
import numpy as np

# from julia.api import Julia 
# jl = Julia(compiled_modules=False)
from julia import Main 
t0 = timeit.default_timer()
Main.include("KREG.jl")
t1 = timeit.default_timer() 
print(f"Loading time: {t1-t0} s")

class KREG_julia():
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
        # self.ac2dArray = np.asfortranarray(np.zeros((self.Natoms,self.Natoms),dtype=np.int32))
        self.ac2dArray = Main.get_ac2dArray(self.Natoms)
        # print(f"DEBUG ac2darray: {self.ac2dArray}")
        self.Req = equilibrium_molecule.xyz_coordinates
        # Calculate interatomic distances of Req
        # self.Xeq = np.zeros(self.Xsize)
        self.Xeq = Main.get_Xeq(self.Natoms,self.ac2dArray,self.array_py2f_dim2(self.Req))
        # print(f"DEBUG Xeq: {self.Xeq}")
        # Calculate RE descriptors
        # self.X = np.zeros((self.Xsize,self.Ntrain))
        self.X = Main.calc_RE_descriptors_wrap(self.Natoms,self.ac2dArray,self.array_py2f_dim3(self.XYZ),self.Xeq)
        self.X = self.array_f2py_dim2(self.X)
        # print(f"DEBUG X: {self.X[1]}")

        # Train 
        # self.K = np.zeros((self.Ksize,self.Ksize))
        # self.alpha = np.zeros((self.Ksize,1))
        self.K,self.alpha = Main.train(self.Natoms,self.Ntrain,self.NtrVal,self.NtrGrXYZ,self.ac2dArray,self.array_py2f_dim3(self.XYZ),self.array_py2f_dim2(self.X),self.Ytrain,self.sigma,self.lambdav,self.lambdagradxyz,True)
        # self.alpha = self.array_f2py_dim2(self.alpha)
        # print(f"DEBUG K: {self.K[:4,:4]}")
        # np.save('K.npy',self.K)
        # print(self.alpha[:10])



    def predict(self, molecular_database=None,molecule=None,property_to_predict=None, xyz_derivative_property_to_predict=None):
        t0 = timeit.default_timer()
        # Assuming that you have trained or loaded a model
        if property_to_predict != None:
            calculate_energy = True 
        else:
            calculate_energy = False 
        if xyz_derivative_property_to_predict != None:
            calculate_energy_gradients = True 
        else:
            calculate_energy_gradients = False
        if molecular_database != None:
            molDB = molecular_database 
        elif molecule != None:
            molDB = ml.data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            raise ValueError(errmsg)
        XYZpredict = molDB.xyz_coordinates
        Npredict = len(XYZpredict)
        #print(self.ac2dArray)
        t1 = timeit.default_timer()
        Xpredict = Main.calc_RE_descriptors_wrap(self.Natoms,self.ac2dArray,self.array_py2f_dim3(XYZpredict),self.Xeq)        
        Xpredict = self.array_f2py_dim2(Xpredict)
        

        # Predict
        # Yest = np.zeros(Npredict)
        # YgradXYZest = np.zeros((3,self.Natoms,Npredict))
        if self.nthreads != None:
            import os
            os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
            os.environ["MKL_NUM_THREADS"] = str(self.nthreads)
        t2 = timeit.default_timer()
        Yest, YgradXYZest = Main.predict_wrap(self.Natoms,self.Ntrain,Npredict,self.NtrVal,self.NtrGrXYZ,self.ac2dArray,self.array_py2f_dim2(self.X),self.array_py2f_dim3(self.XYZ),self.array_py2f_dim2(Xpredict),self.array_py2f_dim3(XYZpredict),self.alpha,calculate_energy,calculate_energy_gradients,self.sigma)
        t3 = timeit.default_timer()
        Yest = Yest + self.prior
        YgradXYZest = self.array_f2py_dim3(YgradXYZest)
        
        # Get predictions
        if calculate_energy:
            molDB.add_scalar_properties(Yest,property_name=property_to_predict)
        if calculate_energy_gradients:
            molDB.add_xyz_vectorial_properties(YgradXYZest,xyz_vectorial_property=xyz_derivative_property_to_predict)
        t4 = timeit.default_timer() 
        print(t1-t0,t2-t1,t3-t2,t4-t3)


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

    def debug(self):
        print("First time:")
        t0 = timeit.default_timer()
        # ethanol = ml.data.molecule.from_xyz_file('ethanol.xyz')
        # print(Main.distance2([1,1],[2,2]))
        mol = ml.data.molecule.from_xyz_file('ethanol.xyz')
        moldb = ml.data.molecular_database.from_xyz_file('xyz.dat')
        moldb.add_scalar_properties_from_file('en.dat','reference_energy')
        moldb.add_xyz_vectorial_properties_from_file('grad.dat','reference_derivatives')
        # moldb.molecules.append(mol)
        self.train(sigma=1.0,lambdav=0.0001,lambdagradxyz=0.0001,molecular_database=moldb,equilibrium_molecule=mol,property_to_learn='reference_energy',prior=0.0)
        t1 = timeit.default_timer()
        print(f"Training time: {t1-t0} s")

        t0 = timeit.default_timer()
        self.predict(molecular_database=moldb,property_to_predict='energy')
        # energies = moldb.get_properties('energy')
        # print(energies[:5])
        t1 = timeit.default_timer()
        print(f"Predicting time: {t1-t0} s")


        print('Second time:')
        t0 = timeit.default_timer()
        # ethanol = ml.data.molecule.from_xyz_file('ethanol.xyz')
        # print(Main.distance2([1,1],[2,2]))
        mol = ml.data.molecule.from_xyz_file('ethanol.xyz')
        moldb = ml.data.molecular_database.from_xyz_file('xyz.dat')
        moldb.add_scalar_properties_from_file('en.dat','reference_energy')
        moldb.add_xyz_vectorial_properties_from_file('grad.dat','reference_derivatives')
        # moldb.molecules.append(mol)
        self.train(sigma=1.0,lambdav=0.0001,lambdagradxyz=0.0001,molecular_database=moldb,equilibrium_molecule=mol,property_to_learn='reference_energy',prior=0.0)
        t1 = timeit.default_timer()
        print(f"Training time: {t1-t0} s")

        t0 = timeit.default_timer()
        self.predict(molecular_database=moldb,property_to_predict='energy')
        # energies = moldb.get_properties('energy')
        # print(energies[:5])
        t1 = timeit.default_timer()
        print(f"Predicting time: {t1-t0} s")

if __name__ == '__main__':
    KREG_julia().debug()