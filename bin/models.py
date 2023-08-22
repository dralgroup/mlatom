#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! models: Module with models                                                ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  ! To-do:                                                                    ! 
# Pre-defined (pre-set, popular) ML models, e.g., KREG
# Delta learning models
# hML models
# Custom (generic) models, KRR, NN.
  !---------------------------------------------------------------------------! 
'''
import os, tempfile, uuid
import numpy as np
import torch
pythonpackage = True
try:
    from . import data, stats, stopper
except:
    import data, stats, stopper
    pythonpackage = False

class methods():
    def __init__(self, method=None, program=None, **kwargs):
        # !!! IMPORTANT !!! 
        # It is neccesary to save all the argments in the model, otherwise it would not be dumped correctly!
        self.method  = method
        self.program = program
        if kwargs != {}: self.kwargs = kwargs
        try:
            from .aiqm1 import aiqm1
            from .interfaces.torchani_interface import ani_methods
            from .interfaces.mndo import mndo_methods
            from .interfaces.sparrow import sparrow_methods
            from .interfaces.gaussian import gaussian_methods
            from .interfaces.xtb import xtb_methods
            from .interfaces.orca import orca_methods
            from .interfaces.columbus import columbus_methods
        except:
            from aiqm1 import aiqm1
            from interfaces.torchani_interface import ani_methods
            from interfaces.mndo import mndo_methods
            from interfaces.sparrow import sparrow_methods
            from interfaces.gaussian import gaussian_methods
            from interfaces.xtb import xtb_methods
            from interfaces.orca import orca_methods
            from interfaces.columbus import columbus_methods
        
        if program != None:
            if   program.casefold() == 'mndo'.casefold():     self.interface = mndo_methods(method=method, **kwargs)
            elif program.casefold() == 'Sparrow'.casefold():  self.interface = sparrow_methods(method=method)
            elif program.casefold() == 'Gaussian'.casefold(): self.interface = gaussian_methods(method=method, **kwargs)
            elif program.casefold() == 'orca'.casefold():     self.interface = orca_methods(method=method, **kwargs)
            elif program.casefold() == 'COLUMBUS'.casefold(): self.interface = columbus_methods(**kwargs)
        elif self.method.casefold() in [mm.casefold() for mm in aiqm1.available_methods]:
            self.interface = aiqm1(method=method, **kwargs)      
        elif self.method.casefold() in [mm.casefold() for mm in ani_methods.available_methods]:
            self.interface = ani_methods(method=method)
        elif self.method.casefold() in [mm.casefold() for mm in mndo_methods.available_methods] or self.method in [mm.casefold() for mm in sparrow_methods.available_methods]:
            if self.method.casefold() in [mm.casefold() for mm in mndo_methods.available_methods] and 'mndobin' in os.environ:
                self.interface = mndo_methods(method=method, **kwargs)
            elif self.method in [mm.casefold() for mm in sparrow_methods.available_methods] and 'sparrowbin' in os.environ:
                self.interface = sparrow_methods(method=method)
            else:
                errmsg = 'Can find appropriate program for the requested method, please set the environment variable: export mndobin=... or export sparrowbin=...'
                if pythonpackage: raise ValueError(errmsg)
                else: stopper.stopMLatom(errmsg)
        elif self.method.casefold() in [mm.casefold() for mm in xtb_methods.available_methods]:
            self.interface = xtb_methods(method=method, **kwargs)
        elif self.method.casefold() == 'D4'.casefold():
            try: from .interfaces.dftd4 import dftd4_methods
            except: from interfaces.dftd4 import dftd4_methods
            self.interface = dftd4_methods(**kwargs)
        
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False):
        self.interface.predict(molecular_database=molecular_database, molecule=molecule,
                                    calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian)
        
    @classmethod
    def is_known_method(cls, method=None):
        try:
            from .aiqm1 import aiqm1
            from .interfaces.torchani_interface import ani_methods
            from .interfaces.mndo import mndo_methods
            from .interfaces.sparrow import sparrow_methods
            from .interfaces.xtb import xtb_methods
        except:
            from aiqm1 import aiqm1
            from interfaces.torchani_interface import ani_methods
            from interfaces.mndo import mndo_methods
            from interfaces.sparrow import sparrow_methods
            from interfaces.xtb import xtb_methods
        methodcasefold = (aiqm1.available_methods + 
                          ani_methods.available_methods +
                          mndo_methods.available_methods +
                          sparrow_methods.available_methods + 
                          xtb_methods.available_methods)
        methodcasefold = [mm.casefold() for mm in methodcasefold]
        if method.casefold() in methodcasefold: return True
        else: return False
        
    def dump(self, filename=None, format='json'):
        if format == 'json':
            import json
            dd = {}
            for key in self.__dict__:
                tt = type(self.__dict__[key])
                if tt in [str, dict]:
                    dd[key] = self.__dict__[key]
            with open(filename, 'w') as fjson:
                json.dump(dd, fjson)

# Parent model class
class ml_model():
    # X or XYZ
    # Y
    # algorithm
    # createMLmodel
    #   train
    #   do not do here hyperparameter optimization
    #   do not do here splitting into sub-training and validation (random split .., cross-validation)
    #   define loss
    #   learn derivatives
    #   learn multiple properties
    #   learn tensorial properties (dipole moments, etc.)
    # predict
    # save model
    # load model
    # define how to optimize hyperparameters - which algorithm grid search, random search, TPE
    # loss training
    # validation loss
    def optimize_hyperparameters(self,
                                 hyperparameters=None,
                                 training_kwargs=None,
                                 prediction_kwargs=None,
                                 cv_splits_molecular_databases=None,
                                 subtraining_molecular_database=None, validation_molecular_database=None,
                                 optimization_algorithm=None,
                                 maximum_evaluations=10000,
                                 ):
        
        property_to_learn = self.get_property_to_learn(training_kwargs)
        property_to_predict = self.get_property_to_predict(prediction_kwargs)
        
        def validation_loss(current_hyperparameters):
            for ii in range(len(current_hyperparameters)):
                self.hyperparameters[hyperparameters[ii]] = current_hyperparameters[ii]
            if type(cv_splits_molecular_databases) == type(None):
                self.holdout_validation(subtraining_molecular_database=subtraining_molecular_database,
                                        validation_molecular_database=validation_molecular_database,
                                        training_kwargs=training_kwargs,
                                        prediction_kwargs=prediction_kwargs)
                y = validation_molecular_database.get_properties(property=property_to_learn)
                estimated_y = validation_molecular_database.get_properties(property=property_to_predict)
            else:
                self.cross_validation(cv_splits_molecular_databases=cv_splits_molecular_databases,
                                        training_kwargs=training_kwargs,
                                        prediction_kwargs=prediction_kwargs)
                training_molecular_database = data.molecular_database()
                for CVsplit in cv_splits_molecular_databases:
                    training_molecular_database.molecules += CVsplit.molecules
                y = training_molecular_database.get_properties(property=property_to_learn)
                estimated_y = training_molecular_database.get_properties(property=property_to_predict)
            return stats.rmse(estimated_y, y)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_name = self.model_file
            self.model_file = f'{tmpdirname}/{saved_name}'
            if optimization_algorithm.casefold() in [mm.casefold() for mm in ['Nelder-Mead', 'BFGS', 'L-BFGS-B', 'Powell', 'CG', 'Newton-CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-krylov', 'trust-exact']]:
                import scipy.optimize
                import numpy as np
                initial_hyperparameters = np.array([self.hyperparameters[key] for key in hyperparameters])
                bounds = np.array([[self.hyperparameters_min[key], self.hyperparameters_max[key]] for key in hyperparameters])
                
                res = scipy.optimize.minimize(validation_loss, initial_hyperparameters, method=optimization_algorithm, bounds=bounds,
                            options={'xatol': 1e-8, 'disp': True, 'maxiter': maximum_evaluations})
                for ii in range(len(res.x)):
                    self.hyperparameters[hyperparameters[ii]] = res.x[ii]
                    
            elif optimization_algorithm.casefold() in [mm.casefold() for mm in ['grid', 'brute']]:
                import scipy.optimize
                import numpy as np
                grid_slices = [list(np.linspace(self.hyperparameters_min[key], self.hyperparameters_max[key], num=9)) for key in hyperparameters]
                grid_slices = [list(np.logspace(np.log(self.hyperparameters_min[key]), np.log(self.hyperparameters_max[key]), num=9, base=np.exp(1))) for key in hyperparameters]
                params, _ = optimize_grid(validation_loss, grid_slices)
                for ii in range(len(params)):
                    self.hyperparameters[hyperparameters[ii]] = params[ii]

            elif optimization_algorithm.lower() == 'tpe':
                import hyperopt
                import numpy as np
                validation_loss_wraper_for_hyperopt = lambda d: validation_loss([d[k] for k in hyperparameters])
                initial_hyperparameters = [{key: self.hyperparameters[key] for key in hyperparameters}]
                space_mapping = {'linear': hyperopt.hp.uniform, 'log': hyperopt.hp.loguniform, 'normal': hyperopt.hp.normal, 'lognormal': hyperopt.hp.lognormal, 'discrete': hyperopt.hp.quniform, 'discretelog': hyperopt.hp.qloguniform, 'discretelognormal': hyperopt.hp.qlognormal, 'choices': hyperopt.hp.choice}
                def get_space(key):
                    space_type = self.hyperparameter_optimization_space[key]
                    if space_type in ['linear', 'log']:
                        args = [np.log(self.hyperparameters_min[key]), np.log(self.hyperparameters_max[key])]
                    else:
                        raise NotImplementedError
                    return space_mapping[space_type](key, *args)
                
                space = {key: get_space(key) for key in hyperparameters}
                res = hyperopt.fmin(fn=validation_loss_wraper_for_hyperopt, space=space, algo=hyperopt.tpe.suggest, max_evals=maximum_evaluations, show_progressbar=True, points_to_evaluate=initial_hyperparameters)
                for k, v in res.items():
                    self.hyperparameters[k] = v
                
            self.model_file = saved_name

    def holdout_validation(self, subtraining_molecular_database=None, validation_molecular_database=None,
                     training_kwargs=None, prediction_kwargs=None):
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        self.train(molecular_database=subtraining_molecular_database, **training_kwargs)
        self.predict(molecular_database = validation_molecular_database, **prediction_kwargs)
        if os.path.exists(self.model_file): os.remove(self.model_file)

    def cross_validation(self, cv_splits_molecular_databases=None,
                     training_kwargs=None, prediction_kwargs=None):
        
        if type(training_kwargs) == type(None): training_kwargs = {}
        if type(prediction_kwargs) == type(None): prediction_kwargs = {}
        
        nsplits = len(cv_splits_molecular_databases)
        for ii in range(nsplits):
            subtraining_molecular_database = data.molecular_database()
            for jj in range(nsplits):
                if ii != jj: subtraining_molecular_database.molecules += cv_splits_molecular_databases[jj].molecules
            validation_molecular_database = cv_splits_molecular_databases[ii]
            self.train(molecular_database=subtraining_molecular_database, **training_kwargs)
            self.predict(molecular_database=validation_molecular_database, **prediction_kwargs)
            if os.path.exists(self.model_file): os.remove(self.model_file)
        
    
    def get_property_to_learn(self, training_kwargs=None):
        if type(training_kwargs) == type(None):
            property_to_learn = 'y'
        else:
            if 'property_to_learn' in training_kwargs:
                property_to_learn = training_kwargs['property_to_learn']
            else:
                property_to_learn = 'y'
        return property_to_learn
    
    def get_property_to_predict(self, prediction_kwargs=None):
        if type(prediction_kwargs) != type(None):
            if 'property_to_predict' in prediction_kwargs:
                property_to_predict = prediction_kwargs['property_to_predict']
            else:
                if 'calculate_energy' in prediction_kwargs:
                    property_to_predict = 'estimated_energy'
                else:
                    property_to_predict = 'estimated_y'
        else:
            property_to_predict = 'estimated_y'
        return property_to_predict

def optimize_grid(func, grid):
    last = True
    for ii in grid[:-1]:
        if len(ii) != 1:
            last = False
            break
    if last:
        other_params = [jj[0] for jj in grid[:-1]]
        opt_param = grid[-1][0]
        min_val = func(other_params + [opt_param])
        for param in grid[-1][1:]:
            val = func(other_params + [param])
            if val < min_val:
                opt_param = param
                min_val = val
        return other_params + [opt_param], min_val
    else:
        for kk in range(len(grid))[:-1]:
            ii = grid[kk]
            if len(ii) != 1:
                other_params_left = grid[:kk]
                opt_param = grid[kk][0]
                other_params_right = grid[kk+1:]
                opt_params, min_val = optimize_grid(func,other_params_left + [[opt_param]] + other_params_right)
                for param in grid[kk][1:]:
                    params, val = optimize_grid(func,other_params_left + [[param]] + other_params_right)
                    if val < min_val:
                        opt_params = params
                        min_val = val
            break
        return opt_params, min_val
 
class krr(ml_model):
    def train(self, molecular_database=None,
              property_to_learn='y',
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None,
              kernel_function_kwargs=None):
        #xx = np.array([mol.descriptor for mol in molecular_database.molecules]).astype(float)
        xyz = np.array([mol.get_xyz_coordinates() for mol in molecular_database.molecules]).astype(float)
        yy = molecular_database.get_properties(property=property_to_learn)
        #self.train_x = xx
        self.Ntrain = len(xyz) 
        self.train_xyz = xyz 
        self.kernel_function = self.gaussian_kernel_function
        self.kernel_function_kwargs = kernel_function_kwargs
        self.kernel_matrix_size = 0 
        Natoms = len(xyz[0])
        self.Natoms = Natoms 

        self.train_property = False 
        self.train_xyz_derivative_property = False 
        yyref = np.array([])
        if property_to_learn != None:
            self.train_property = True
            self.kernel_matrix_size += len(yy)
            yyref = np.concatenate((yyref,yy))
        if xyz_derivative_property_to_learn != None:
            self.train_xyz_derivative_property = True
            self.kernel_matrix_size += 3*Natoms*len(yy)
            yygrad = molecular_database.get_xyz_derivative_properties().reshape(3*Natoms*len(yy))
            yyref = np.concatenate((yyref,yygrad))

        kernel_matrix = np.zeros((self.kernel_matrix_size,self.kernel_matrix_size))


        # for ii in range(len(yy)):
        #     kernel_matrix[ii][ii] = self.kernel_function(xx[ii], xx[ii]) + self.hyperparameters['lambda']
        #     for jj in range(ii+1, len(yy)):
        #         kernel_matrix[ii][jj] = self.kernel_function(xx[ii], xx[jj])
        #         kernel_matrix[jj][ii] = kernel_matrix[ii][jj]

        kernel_matrix = kernel_matrix + np.identity(self.kernel_matrix_size)*self.hyperparameters['lambda']
        for ii in range(len(yy)):
            value_and_derivatives = self.kernel_function(xyz[ii],xyz[ii],calculate_value=self.train_property,calculate_gradients=self.train_xyz_derivative_property,calculate_Hessian=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
            kernel_matrix[ii][ii] += value_and_derivatives['value']
            if self.train_xyz_derivative_property:
                kernel_matrix[ii,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['gradients'].reshape(3*Natoms)
                kernel_matrix[len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['Hessian']
            for jj in range(ii+1,len(yy)):
                value_and_derivatives = self.kernel_function(xyz[ii],xyz[jj],calculate_value=self.train_property,calculate_gradients=self.train_xyz_derivative_property,calculate_Hessian=self.train_xyz_derivative_property,calculate_gradients_j=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                kernel_matrix[ii][jj] += value_and_derivatives['value']
                kernel_matrix[jj][ii] += value_and_derivatives['value']
                if self.train_xyz_derivative_property:
                    kernel_matrix[jj,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] = value_and_derivatives['gradients'].reshape(3*Natoms)
                    kernel_matrix[ii,len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms] = value_and_derivatives['gradients_j'].reshape(3*Natoms)
                    kernel_matrix[len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms,len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms] += value_and_derivatives['Hessian']
                    kernel_matrix[len(yy)+jj*3*Natoms:len(yy)+(jj+1)*3*Natoms,len(yy)+ii*3*Natoms:len(yy)+(ii+1)*3*Natoms] += value_and_derivatives['Hessian']
        if self.train_xyz_derivative_property:
            kernel_matrix[len(yy):,:len(yy)] = kernel_matrix[:len(yy),len(yy):].T


        if invert_matrix:
            self.alphas = np.dot(np.linalg.inv(kernel_matrix), yy)
        else:
            from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
            if matrix_decomposition==None:
                try:
                    c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = cho_solve((c, low), yy, check_finite=False)
                except:
                    c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = lu_solve((c, low), yy, check_finite=False)
            elif matrix_decomposition.casefold()=='Cholesky'.casefold():
                c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = cho_solve((c, low), yy, check_finite=False)
            elif matrix_decomposition.casefold()=='LU'.casefold():
                c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = lu_solve((c, low), yy, check_finite=False)
            
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False, # calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
                predict_property = True, property_to_predict = None,
                predict_xyz_derivative_property = False, xyz_derivative_property_to_predict = None):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)
        
        if  calculate_energy:
            predict_property = True
            property_to_predict = 'energy'
        elif predict_property and property_to_predict == None:
            property_to_predict = 'estimated_y'
        elif property_to_predict != None:
            predict_property = True
        
        if calculate_energy_gradients:
            predict_xyz_derivative_property = True
            xyz_derivative_property_to_predict = 'energy_gradients'
        elif predict_xyz_derivative_property and xyz_derivative_property_to_predict == None:
            xyz_derivative_property_to_predict = 'estimated_xyz_derivatives'
        elif xyz_derivative_property_to_predict != None:
            predict_xyz_derivative_property = True

        Natoms = len(molDB.molecules[0].atoms)
        kk_size = 0 
        if self.train_property:
            kk_size += self.Ntrain 
        if self.train_xyz_derivative_property:
            kk_size += self.Ntrain * Natoms*3
            
        # for mol in molDB.molecules:
        #     xx = mol.descriptor
        #     kk = [self.kernel_function(xx, self.train_x[ii]) for ii in range(len(self.train_x))]
        #     mol.__dict__[property_to_predict] = np.sum(np.multiply(self.alphas, kk))

        for mol in molDB.molecules:
            kk = np.zeros(kk_size)
            kk_der = np.zeros((kk_size,3*Natoms))
            for ii in range(self.Ntrain):
                value_and_derivatives = self.kernel_function(self.train_xyz[ii],mol.get_xyz_coordinates(),calculate_gradients=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                kk[ii] = value_and_derivatives['value']
                if self.train_xyz_derivative_property:
                    #print(self.Ntrain+ii*3*Natoms,self.Ntrain+(ii+1)*3*Natoms)
                    kk[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms] = value_and_derivatives['gradients'].reshape(3*Natoms)

                if predict_xyz_derivative_property:
                    value_and_derivatives = self.kernel_function(mol.get_xyz_coordinates(),self.train_xyz[ii],calculate_gradients=predict_xyz_derivative_property,calculate_Hessian=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                    kk_der[ii] = value_and_derivatives['gradients'].reshape(3*Natoms)
                if self.train_xyz_derivative_property:
                    kk_der[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms,:] = value_and_derivatives['Hessian'].T
                #kk_der.append(value_and_derivatives['gradients'])
            #kk_der = np.array(kk_der).reshape(kk_size,3*Natoms)
            gradients = np.matmul(self.alphas.reshape(1,len(self.alphas)),kk_der)[0]
            #kk = [self.kernel_function(xx, self.train_x[ii])['value'] for ii in range(len(self.train_x))]
            mol.__dict__[property_to_predict] = np.sum(np.multiply(self.alphas, kk))
            # print(mol.__dict__[property_to_predict])
            # print(gradients)
            for iatom in range(len(mol.atoms)):
                mol.atoms[iatom].__dict__[xyz_derivative_property_to_predict] = gradients[3*iatom:3*iatom+3]
        
    # def gaussian_kernel_function(self, xi, xj):
    #     return np.exp(np.sum(np.square(xi - xj))/(-2*self.hyperparameters['sigma']**2))
    
    def gaussian_kernel_function(self,coordi,coordj,calculate_value=True,calculate_gradients=False,calculate_gradients_j=False,calculate_Hessian=False,**kwargs):
        if 'Req' in kwargs:
            Req = kwargs['Req']
        if calculate_Hessian:
            calculate_gradients = True
            calculate_value = True 
        elif calculate_gradients:
            calculate_value = True
        coordi_tensor = torch.tensor(coordi,requires_grad=True)
        coordj_tensor = torch.tensor(coordj,requires_grad=True)
        value = None 
        gradients = None 
        gradients_j = None
        Hessian = None
        xi_tensor = self.RE_descriptor_tensor(coordi_tensor,Req)
        xj_tensor = self.RE_descriptor_tensor(coordj_tensor,Req)
        if calculate_value:
            value_tensor = torch.exp(torch.sum(torch.square(xi_tensor - xj_tensor))/(-2*self.hyperparameters['sigma']**2))
            value = value_tensor.detach().numpy()
        if calculate_gradients:
            gradients_tensor = torch.autograd.grad(value_tensor,coordi_tensor,create_graph=True,retain_graph=True)[0]
            gradients = gradients_tensor.detach().numpy()
        if calculate_gradients_j:
            gradients_j_tensor = torch.autograd.grad(value_tensor,coordj_tensor,create_graph=True,retain_graph=True)[0]
            gradients_j = gradients_j_tensor.detach().numpy()
        if calculate_Hessian:
            Hessian = []
            gradients_tensor = gradients_tensor.reshape(len(gradients_tensor)*3)
            for ii in range(len(gradients_tensor)):
                Hessian.append(torch.autograd.grad(gradients_tensor[ii],coordj_tensor,create_graph=True,retain_graph=True)[0].detach().numpy().reshape((len(gradients_tensor))))
            Hessian = np.array(Hessian)
        output = {'value':value,'gradients':gradients,'gradients_j':gradients_j,'Hessian':Hessian}
        return output
    
    def RE_descriptor_tensor(self,coord_tensor,Req):
        Natoms = len(coord_tensor)
        icount = 0 
        for iatom in range(Natoms):
            for jatom in range(iatom+1,Natoms):
                output = Req[icount] / self.distance_tensor(coord_tensor[iatom],coord_tensor[jatom])
                if icount == 0:
                    descriptor = output.reshape(1)
                else:
                    descriptor = torch.cat(descriptor,output.reshape(1))
                icount += 1
        return descriptor
    
    def distance_tensor(self, atomi,atomj):
        return torch.sqrt(self.distance_squared_tensor(atomi,atomj))
    
    def distance_squared_tensor(self, atomi,atomj):
        return torch.sum(torch.square(atomi-atomj))

class kreg(krr):    
    def __init__(self, model_file = None, use_mlatomf=True, equilibrium_molecule=None):
        self.model_file = model_file
        self.use_mlatomf=use_mlatomf
        self.equilibrium_molecule = equilibrium_molecule
        if use_mlatomf:
            try:
                from . import interface_MLatomF
            except:
                import interface_MLatomF
            self.interface_mlatomf = interface_MLatomF
        self.hyperparameters = {
            'lambda':          0.0,
            'lambdaGradXYZ':   0.0,
            'sigma':           1.0
            }
        self.hyperparameters_min = {
            'lambda':        2**-35,
            'lambdaGradXYZ': 2**-35,
            'sigma':         2**-5
            }
        self.hyperparameters_max = {
            'lambda':         1.0,
            'lambdaGradXYZ':  1.0,
            'sigma':         2**9
            }
        self.hyperparameter_optimization_space = {
            'lambda':        'log',
            'lambdaGradXYZ': 'log',
            'sigma':         'log'
            }
    
    def get_descriptor(self, molecule=None, molecular_database=None, descriptor_name='re', equilibrium_molecule=None):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)
        
        if 'reference_distance_matrix' in self.__dict__:
            eq_distmat = self.reference_distance_matrix
        else:
            if equilibrium_molecule == None:
                if 'energy' in molDB.molecules[0].__dict__:
                    energies = molDB.get_properties(property='energy')
                    equilibrium_molecule = molDB.molecules[np.argmin(energies)]
                else:
                    errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
                    if pythonpackage: raise ValueError(errmsg)
                    else: stopper.stopMLatom(errmsg)
            eq_distmat = equilibrium_molecule.get_internuclear_distance_matrix()
            self.reference_distance_matrix = eq_distmat
        natoms = len(molDB.molecules[0].atoms)
        for mol in molDB.molecules:
            descriptor = np.zeros(int(natoms*(natoms-1)/2))
            distmat = mol.get_internuclear_distance_matrix()
            ii = -1
            for iatomind in range(natoms):
                for jatomind in range(iatomind+1,natoms):
                    ii += 1
                    descriptor[ii] = eq_distmat[iatomind][jatomind]/distmat[iatomind][jatomind]
            mol.__dict__[descriptor_name] = descriptor
            mol.descriptor = descriptor

    def get_equilibrium_distances(self,molecular_database=None):
        molDB = molecular_database
        if self.equilibrium_molecule == None:
            if 'energy' in molDB.molecules[0].__dict__:
                energies = molDB.get_properties(property='energy')
                self.equilibrium_molecule = molDB.molecules[np.argmin(energies)]
            else:
                errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
                if pythonpackage: raise ValueError(errmsg)
                else: stopper.stopMLatom(errmsg)
        eq_distmat = self.equilibrium_molecule.get_internuclear_distance_matrix()
        Req = np.array([])
        for ii in range(len(eq_distmat)):
            Req = np.concatenate((Req,eq_distmat[ii][ii+1:]))
        return Req
    
    def train(self, molecular_database=None,
              property_to_learn=None,
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None):
        if self.use_mlatomf:
            mlatomfargs = ['createMLmodel'] + ['%s=%.14f' % (param, self.hyperparameters[param]) for param in self.hyperparameters.keys()]
            if save_model:
                if self.model_file == None:
                    self.model_file =f'kreg_{str(uuid.uuid4())}.unf'
            with tempfile.TemporaryDirectory() as tmpdirname:
                molecular_database.write_file_with_xyz_coordinates(filename = f'{tmpdirname}/train.xyz')
                mlatomfargs.append(f'XYZfile={tmpdirname}/train.xyz')
                if property_to_learn != None:
                    molecular_database.write_file_with_properties(filename = f'{tmpdirname}/y.dat', property_to_write = property_to_learn)
                    mlatomfargs.append(f'Yfile={tmpdirname}/y.dat')
                if xyz_derivative_property_to_learn != None:
                    molecular_database.write_file_with_xyz_derivative_properties(filename = f'{tmpdirname}/ygrad.xyz', xyz_derivative_property_to_write = xyz_derivative_property_to_learn)
                    mlatomfargs.append(f'YgradXYZfile={tmpdirname}/ygrad.xyz')
                mlatomfargs.append(f'MLmodelOut={self.model_file}')
                self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
        else:
            if 'reference_distance_matrix' in self.__dict__: del self.__dict__['reference_distance_matrix']
            #self.get_descriptor(molecular_database=molecular_database)
            Req = self.get_equilibrium_distances(molecular_database=molecular_database)
            #print(Req)
            super().train(molecular_database=molecular_database, property_to_learn=property_to_learn,
              invert_matrix=invert_matrix,
              matrix_decomposition=matrix_decomposition,
              kernel_function_kwargs={'Req':Req})
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False, # calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
                predict_property = True, property_to_predict = None,
                predict_xyz_derivative_property = False, xyz_derivative_property_to_predict = None):
        if not self.use_mlatomf:
            #self.get_descriptor(molecule=molecule, molecular_database=molecular_database)
            super().predict(molecular_database=molecular_database, molecule=molecule,
                            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients,
                            predict_property=predict_property, property_to_predict=property_to_predict,
                            predict_xyz_derivative_property = predict_xyz_derivative_property, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict)
            return
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            if pythonpackage: raise ValueError(errmsg)
            else: stopper.stopMLatom(errmsg)
        
        if  calculate_energy:
            predict_property = True
            property_to_predict = 'energy'
        elif predict_property and property_to_predict == None:
            property_to_predict = 'estimated_y'
        elif property_to_predict != None:
            predict_property = True
        
        if calculate_energy_gradients:
            predict_xyz_derivative_property = True
            xyz_derivative_property_to_predict = 'energy_gradients'
        elif predict_xyz_derivative_property and xyz_derivative_property_to_predict == None:
            xyz_derivative_property_to_predict = 'estimated_xyz_derivatives'
        elif xyz_derivative_property_to_predict != None:
            predict_xyz_derivative_property = True
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            molDB.write_file_with_xyz_coordinates(filename = f'{tmpdirname}/predict.xyz')
            mlatomfargs = ['useMLmodel', 'MLmodelIn=%s' % self.model_file]
            mlatomfargs.append(f'XYZfile={tmpdirname}/predict.xyz')
            if predict_property: mlatomfargs.append(f'YestFile={tmpdirname}/yest.dat')
            if predict_xyz_derivative_property: mlatomfargs.append(f'YgradXYZestFile={tmpdirname}/ygradest.xyz')
            self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
            # read yest.dat and update database
            if property_to_predict != None: molDB.add_scalar_properties_from_file(filename = f'{tmpdirname}/yest.dat', property = property_to_predict)
            if xyz_derivative_property_to_predict != None: molDB.add_xyz_derivative_properties_from_file(filename = f'{tmpdirname}/ygradest.xyz', xyz_derivative_property = xyz_derivative_property_to_predict)

def ani(**kwargs):
    try:
        from .interfaces.torchani_interface import ani
    except: 
        from interfaces.torchani_interface import ani
    return ani(**kwargs)

def sgdml(**kwargs):
    try:
        from .interfaces.sgdml_interface import sgdml
    except: 
        from interfaces.sgdml_interface import sgdml
    return sgdml(**kwargs)

class DeltaML():
    pass
    #baseline
    #MLmodel

class model_tree_node():
    def __init__(self, name=None, parent=None, children=None, operator=None, model=None):
        self.name = name
        self.parent = parent
        self.children = children
        if self.parent != None:
            if self.parent.children == None: self.parent.children = []   
            if not self in self.parent.children:
                self.parent.children.append(self)
        if self.children != None:
            for child in self.children:
                child.parent=self
        self.operator = operator
        self.model = model
    
    def predict(self, **kwargs):
        if 'molecular_database' in kwargs:
            molDB = kwargs['molecular_database']
        if 'molecule' in kwargs:
            molDB = data.molecular_database()
            molDB.molecules.append(kwargs['molecule'])
            
        if 'calculate_energy' in kwargs: calculate_energy = kwargs['calculate_energy']
        else: calculate_energy = True
        if 'calculate_energy_gradients' in kwargs: calculate_energy_gradients = kwargs['calculate_energy_gradients']
        else: calculate_energy_gradients = False
        if 'calculate_hessian' in kwargs: calculate_hessian = kwargs['calculate_hessian']
        else: calculate_hessian = False

        properties = [] ; atomic_properties = []
        if calculate_energy: properties.append('energy')
        if calculate_energy_gradients: atomic_properties.append('energy_gradients')
        if calculate_hessian: properties.append('hessian')

        for mol in molDB.molecules:
            if not self.name in mol.__dict__:
                parent = None
                if self.parent != None:
                    if self.parent.name in mol.__dict__:
                        parent = mol.__dict__[self.parent.name]
                children = None
                if self.children != None:
                    for child in self.children:
                        if child.name in mol.__dict__:
                            if children == None: children = []
                            children.append(mol.__dict__[child.name])
                mol.__dict__[self.name] = data.properties_tree_node(name=self.name, parent=parent, children=children)
        
        if self.children == None and self.operator == 'predict':
            self.model.predict(**kwargs)
            for mol in molDB.molecules:
                self.get_properties_from_molecule(mol, properties, atomic_properties)
        else:
            for child in self.children:
                if not child.name in molDB.molecules[0].__dict__: child.predict(**kwargs)

            if self.operator == 'sum':
                for mol in molDB.molecules:
                    mol.__dict__[self.name].sum(properties+atomic_properties)
            if self.operator == 'average':
                for mol in molDB.molecules:
                    mol.__dict__[self.name].average(properties+atomic_properties)
                    
        if self.parent == None:
            self.update_molecular_properties(molecular_database=molDB, properties=properties, atomic_properties=atomic_properties)
        
    def get_properties_from_molecule(self, molecule, properties=[], atomic_properties=[]):
        property_values = molecule.__dict__[self.name].__dict__
        for property in properties:
            if property in molecule.__dict__: property_values[property] = molecule.__dict__[property]
        for property in atomic_properties:
            property_values[property] = []
            for atom in molecule.atoms:
                property_values[property].append(atom.__dict__[property])
            property_values[property] = np.array(property_values[property]).astype(float)
    
    def update_molecular_properties(self, molecular_database=None, molecule=None, properties=[], atomic_properties=[]):
        molDB = molecular_database
        if molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)

        for mol in molDB.molecules:
            for property in properties:
                mol.__dict__[property] = mol.__dict__[self.name].__dict__[property]
            for property in atomic_properties:
                for iatom in range(len(mol.atoms)):
                    mol.atoms[iatom].__dict__[property] = mol.__dict__[self.name].__dict__[property][iatom]

if __name__ == '__main__':
    pass