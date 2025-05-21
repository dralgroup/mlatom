#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! models: Module with models                                                ! 
  ! Implementations by: Pavlo O. Dral, Fuchun Ge, Yi-Fan Hou, Yuxinxin Chen,  !
  !                     Peikun Zheng                                          ! 
  !---------------------------------------------------------------------------! 
'''
from __future__ import annotations
from typing import Any, Union, Dict
import os, tempfile, uuid, sys
import numpy as np

from . import data, stopper
from .model_cls import model, ml_model, OMP_model, MKL_model, hyperparameter, hyperparameters, model_tree_node
class krr(ml_model):
    def train(self, molecular_database=None,
              property_to_learn='y',
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None,
              kernel_function_kwargs=None,prior=None):
        
        xyz = np.array([mol.xyz_coordinates for mol in molecular_database.molecules]).astype(float)
        yy = molecular_database.get_properties(property_name=property_to_learn)
        if prior == None:
            self.prior = 0.0
        elif type(prior) == float or type(prior) == int:
            self.prior = prior 
        elif prior.casefold() == 'mean'.casefold():
            self.prior = np.mean(yy)
        else:
            stopper.stopMLatom(f'Unsupported prior: {prior}')
        yy = yy - self.prior
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

        kernel_matrix = kernel_matrix + np.identity(self.kernel_matrix_size)*self.hyperparameters['lambda'].value
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
            self.alphas = np.dot(np.linalg.inv(kernel_matrix), yyref)
        else:
            from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve
            if matrix_decomposition==None:
                try:
                    c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = cho_solve((c, low), yyref, check_finite=False)
                except:
                    c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                    self.alphas = lu_solve((c, low), yyref, check_finite=False)
            elif matrix_decomposition.casefold()=='Cholesky'.casefold():
                c, low = cho_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = cho_solve((c, low), yyref, check_finite=False)
            elif matrix_decomposition.casefold()=='LU'.casefold():
                c, low = lu_factor(kernel_matrix, overwrite_a=True, check_finite=False)
                self.alphas = lu_solve((c, low), yyref, check_finite=False)
            
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False,  calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
               property_to_predict = None, xyz_derivative_property_to_predict = None,
                hessian_to_predict=None,):
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)

        Natoms = len(molDB.molecules[0].atoms)
        kk_size = 0 
        if self.train_property:
            kk_size += self.Ntrain 
        if self.train_xyz_derivative_property:
            kk_size += self.Ntrain * Natoms*3

        for mol in molDB.molecules:
            kk = np.zeros(kk_size)
            kk_der = np.zeros((kk_size,3*Natoms))
            for ii in range(self.Ntrain):
                value_and_derivatives = self.kernel_function(self.train_xyz[ii],mol.xyz_coordinates,calculate_gradients=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                kk[ii] = value_and_derivatives['value']
                if self.train_xyz_derivative_property:
                    kk[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms] = value_and_derivatives['gradients'].reshape(3*Natoms)

                if xyz_derivative_property_to_predict:
                    value_and_derivatives = self.kernel_function(mol.xyz_coordinates,self.train_xyz[ii],calculate_gradients=bool(xyz_derivative_property_to_predict),calculate_Hessian=self.train_xyz_derivative_property,**self.kernel_function_kwargs)
                    kk_der[ii] = value_and_derivatives['gradients'].reshape(3*Natoms)
                if self.train_xyz_derivative_property:
                    kk_der[self.Ntrain+ii*3*Natoms:self.Ntrain+(ii+1)*3*Natoms,:] = value_and_derivatives['Hessian'].T
            gradients = np.matmul(self.alphas.reshape(1,len(self.alphas)),kk_der)[0]
            mol.__dict__[property_to_predict] = np.sum(np.multiply(self.alphas, kk))+self.prior
            for iatom in range(len(mol.atoms)):
                mol.atoms[iatom].__dict__[xyz_derivative_property_to_predict] = gradients[3*iatom:3*iatom+3]
    
    def gaussian_kernel_function(self,coordi,coordj,calculate_value=True,calculate_gradients=False,calculate_gradients_j=False,calculate_Hessian=False,**kwargs):
        import torch
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
            value_tensor = torch.exp(torch.sum(torch.square(xi_tensor - xj_tensor))/(-2*self.hyperparameters['sigma'].value**2))
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
        import torch
        Natoms = len(coord_tensor)
        icount = 0 
        for iatom in range(Natoms):
            for jatom in range(iatom+1,Natoms):
                output = Req[icount] / self.distance_tensor(coord_tensor[iatom],coord_tensor[jatom])
                if icount == 0:
                    descriptor = output.reshape(1)
                else:
                    descriptor = torch.cat((descriptor,output.reshape(1)))
                icount += 1
        return descriptor
    
    def distance_tensor(self, atomi,atomj):
        import torch
        return torch.sqrt(self.distance_squared_tensor(atomi,atomj))
    
    def distance_squared_tensor(self, atomi,atomj):
        import torch
        return torch.sum(torch.square(atomi-atomj))

class kreg(krr, OMP_model, MKL_model):
    '''
    Create a KREG model object.

    Arguments:
        model_file (str, optional): The name of the file where the model should be dumped to or loaded from.
        ml_program (str, optional): Specify which ML program to use. Avaliable options: ``'KREG_API'``, ``'MLatomF``.
        equilibrium_molecule (:class:`mlatom.data.molecule` | None): Specify the equilibrium geometry to be used to generate RE descriptor. The geometry with lowest energy/value will be selected if set to ``None``.
        prior (default - None): the prior can be 'mean', None (0.0), or any floating point number.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
    '''
    hyperparameters = hyperparameters({'lambda': hyperparameter(value=2**-35, 
                                                         minval=2**-35, 
                                                         maxval=1.0, 
                                                         optimization_space='log',
                                                         name='lambda'),
                                'sigma':  hyperparameter(value=1.0,
                                                         minval=2**-5,
                                                         maxval=2**9,
                                                         optimization_space='log',
                                                         name='sigma')}) 
    def __init__(self, model_file: Union[str, None] = None, ml_program: str = 'KREG_API', equilibrium_molecule: Union[data.molecule, None] = None, prior: float = 0, nthreads: Union[int, None] = None, hyperparameters: Union[Dict[str,Any], hyperparameters]={}):
        self.model_file = model_file
        self.equilibrium_molecule = equilibrium_molecule
        self.ml_program = ml_program
        if self.ml_program.casefold() == 'KREG_API'.casefold():
            from .kreg_api import KREG_API
            if self.model_file != None:
                if self.model_file[-4:] != '.npz':
                    self.model_file += '.npz'
                if os.path.exists(self.model_file):
                    self.kreg_api = KREG_API()
                    self.kreg_api.load_model(self.model_file)
        if self.ml_program.casefold() == 'MLatomF'.casefold():
            from . import interface_MLatomF
            self.interface_mlatomf = interface_MLatomF
        
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
            
        self.nthreads = nthreads

    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                if args.data[hyperparam].lower() == 'opt':
                    self.hyperparameters[hyperparam].dtype = str
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
        if args.eqXYZfileIn:
            self.equilibrium_molecule = data.molecule.from_xyz_file(args.eqXYZfileIn)
    
    def generate_model_dict(self):
        model_dict = super().generate_model_dict()
        model_dict['kwargs']['ml_program'] =  self.ml_program
        return model_dict
    
    def get_descriptor(self, molecule=None, molecular_database=None, descriptor_name='re', equilibrium_molecule=None):
        if molecular_database != None:
            molDB = molecular_database
        elif molecule != None:
            molDB = data.molecular_database()
            molDB.molecules.append(molecule)
        else:
            errmsg = 'Either molecule or molecular_database should be provided in input'
            raise ValueError(errmsg)
        
        if 'reference_distance_matrix' in self.__dict__:
            eq_distmat = self.reference_distance_matrix
        else:
            if equilibrium_molecule == None:
                if 'energy' in molDB.molecules[0].__dict__:
                    energies = molDB.get_properties(property_name='energy')
                    equilibrium_molecule = molDB.molecules[np.argmin(energies)]
                elif 'y' in molecular_database.molecules[0].__dict__:
                    y = molecular_database.get_properties(property_name='y')
                    equilibrium_molecule = molecular_database.molecules[np.argmin(y)]
                else:
                    errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
                    raise ValueError(errmsg)
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
        if self.equilibrium_molecule == None: self.equilibrium_molecule = self.get_equilibrium_molecule(molecular_database=molecular_database)
        eq_distmat = self.equilibrium_molecule.get_internuclear_distance_matrix()
        Req = np.array([])
        for ii in range(len(eq_distmat)):
            Req = np.concatenate((Req,eq_distmat[ii][ii+1:]))
        return Req
    
    def get_equilibrium_molecule(self,molecular_database=None):
        if 'energy' in molecular_database.molecules[0].__dict__:
            energies = molecular_database.get_properties(property_name='energy')
            equilibrium_molecule = molecular_database.molecules[np.argmin(energies)]
        elif 'y' in molecular_database.molecules[0].__dict__:
            y = molecular_database.get_properties(property_name='y')
            equilibrium_molecule = molecular_database.molecules[np.argmin(y)]
        else:
            errmsg = 'equilibrium molecule is not provided and no energies are found in molecular database'
            raise ValueError(errmsg)
        return equilibrium_molecule
    
    def train(self, molecular_database=None,
              property_to_learn=None,
              xyz_derivative_property_to_learn = None,
              save_model=True,
              invert_matrix=False,
              matrix_decomposition=None,
              prior=None,
              hyperparameters: Union[Dict[str,Any], hyperparameters] = {},):
        '''
        Train the KREG model with molecular database provided.

        Arguments:
            molecular_database (:class:`mlatom.data.molecular_database`): The database of molecules for training.
            property_to_learn (str, optional): The label of property to be learned in model training.
            xyz_derivative_property_to_learn (str, optional): The label of XYZ derivative property to be learned.
            prior (str or float or int, optional): default zero prior. It can also be 'mean' and any user-defined number.
        '''
        self.hyperparameters.update(hyperparameters)
        if self.ml_program.casefold() == 'MLatomF'.casefold():
            mlatomfargs = ['createMLmodel'] + ['%s=%s' % (param, self.hyperparameters[param].value) for param in self.hyperparameters.keys()]
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
                if prior != None:
                    mlatomfargs.append(f'prior={prior}')
                mlatomfargs.append(f'MLmodelOut={self.model_file}')
                if 'additional_mlatomf_args' in self.__dict__:
                    mlatomfargs += self.additional_mlatomf_args
                self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
        elif self.ml_program.casefold() == 'KREG_API'.casefold():
            from .kreg_api import KREG_API
            if save_model:
                if self.model_file == None:
                    self.model_file = f'kreg_{str(uuid.uuid4())}.npz'
                else:
                    if self.model_file[-4:] != '.npz':
                        self.model_file += '.npz'
            self.kreg_api = KREG_API()
            self.kreg_api.nthreads = self.nthreads
            if self.equilibrium_molecule == None: self.equilibrium_molecule = self.get_equilibrium_molecule(molecular_database=molecular_database)
            if not 'lambdaGradXYZ' in self.hyperparameters.keys(): lambdaGradXYZ = self.hyperparameters['lambda'].value
            else: lambdaGradXYZ = self.hyperparameters['lambdaGradXYZ'].value
            if 'prior' in self.hyperparameters.keys(): prior = self.hyperparameters['prior'].value
            self.kreg_api.train(self.hyperparameters['sigma'].value,self.hyperparameters['lambda'].value,lambdaGradXYZ,molecular_database,self.equilibrium_molecule,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn,prior=prior)
            if save_model:
                self.kreg_api.save_model(self.model_file)

        else:
            if 'reference_distance_matrix' in self.__dict__: del self.__dict__['reference_distance_matrix']
            Req = self.get_equilibrium_distances(molecular_database=molecular_database)
            super().train(molecular_database=molecular_database, property_to_learn=property_to_learn,
              invert_matrix=invert_matrix,
              matrix_decomposition=matrix_decomposition,
              kernel_function_kwargs={'Req':Req},prior=prior)
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=False, calculate_energy_gradients=False, calculate_hessian=False, # arguments if KREG is used as MLP ; hessian not implemented (possible with numerical differentiation)
                property_to_predict = None, xyz_derivative_property_to_predict = None, hessian_to_predict = None,):
        if self.ml_program.casefold() != 'MLatomF'.casefold() and self.ml_program.casefold() != 'KREG_API'.casefold():
            super().predict(molecular_database=molecular_database, molecule=molecule,
                            calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, property_to_predict=property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict)
        else:
            molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super(krr, self).predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)

            if self.ml_program.casefold() == 'KREG_API'.casefold():
                if 'kreg_api' in dir(self):
                    self.kreg_api.nthreads = self.nthreads
                    self.kreg_api.predict(molecular_database=molecular_database,molecule=molecule,property_to_predict=property_to_predict,xyz_derivative_property_to_predict=xyz_derivative_property_to_predict)
                else:
                    stopper.stopMLatom('KREG_API model not found')
            elif self.ml_program.casefold() == 'MLatomF'.casefold():
                with tempfile.TemporaryDirectory() as tmpdirname:
                    molDB.write_file_with_xyz_coordinates(filename = f'{tmpdirname}/predict.xyz')
                    mlatomfargs = ['useMLmodel', 'MLmodelIn=%s' % self.model_file]
                    mlatomfargs.append(f'XYZfile={tmpdirname}/predict.xyz')
                    if property_to_predict: mlatomfargs.append(f'YestFile={tmpdirname}/yest.dat')
                    if xyz_derivative_property_to_predict: mlatomfargs.append(f'YgradXYZestFile={tmpdirname}/ygradest.xyz')
                    self.interface_mlatomf.ifMLatomCls.run(mlatomfargs, shutup=True)
                    if not os.path.exists(f'{tmpdirname}/yest.dat'):
                        import time
                        time.sleep(0.0000001) # Sometimes program needs more time to write file yest.dat / P.O.D., 2023-04-18
                        os.system(f'cp -r {tmpdirname} .')
                    if property_to_predict != None: molDB.add_scalar_properties_from_file(filename = f'{tmpdirname}/yest.dat', property_name = property_to_predict)
                    if xyz_derivative_property_to_predict != None: molDB.add_xyz_derivative_properties_from_file(filename = f'{tmpdirname}/ygradest.xyz', xyz_derivative_property = xyz_derivative_property_to_predict)

from .interfaces.torchani_interface import ani, msani

def dpmd(**kwargs):
    '''
    Returns a DPMD model object (see :class:`mlatom.interfaces.dpmd_interface.dpmd`).
    '''
    from .interfaces.dpmd_interface import dpmd
    return dpmd(**kwargs)

def gap(**kwargs):
    '''
    Returns a GAP model object (see :class:`mlatom.interfaces.gap_interface.gap`).
    '''
    from .interfaces.gap_interface import gap
    return gap(**kwargs)

def physnet(**kwargs):
    '''
    Returns a PhysNet model object (see :class:`mlatom.interfaces.physnet_interface.physnet`).
    '''
    from .interfaces.physnet_interface import physnet
    return physnet(**kwargs)

def sgdml(**kwargs):
    '''
    Returns an sGDML model object (see :class:`mlatom.interfaces.sgdml_interface.sgdml`).
    '''
    from .interfaces.sgdml_interface import sgdml
    return sgdml(**kwargs)

def mace(**kwargs):
    '''
    Returns an MACE model object (see :class:`mlatom.interfaces.mace_interface.mace`).
    '''
    from .interfaces.mace_interface import mace
    return mace(**kwargs)

from .addons.uaiqm.uaiqm             import uaiqm
from .addons.omnip2x.omnip2x         import omnip2x
from .addons.omnip2x.vecmsani        import vecmsani

from .aiqm1                          import aiqm1
from .aiqm2                          import aiqm2
from .dens                           import dens
# from .aimnet2                        import aimnet2_methods
from .interfaces.aimnet2_interface   import aimnet2_methods
from .interfaces.torchani_interface  import ani_methods
from .interfaces.gaussian_interface  import gaussian_methods
from .interfaces.pyscf_interface     import pyscf_methods
from .interfaces.orca_interface      import orca_methods
from .interfaces.turbomole_interface import turbomole_methods
from .interfaces.mndo_interface      import mndo_methods
from .interfaces.sparrow_interface   import sparrow_methods
from .interfaces.xtb_interface       import xtb_methods
from .interfaces.dftbplus_interface  import dftbplus_methods
from .interfaces.columbus_interface  import columbus_methods
from .composite_methods              import ccsdtstarcbs_legacy as ccsdtstarcbs
from .interfaces.dftd3_interface     import dftd3_methods
from .interfaces.dftd4_interface     import dftd4_methods

# The order of classes determines the defaults (i.e., whatever first works, is used)
known_classes = [aiqm1, aiqm2, dens, ani_methods, aimnet2_methods, ccsdtstarcbs, gaussian_methods, pyscf_methods, orca_methods, turbomole_methods, mndo_methods, sparrow_methods, xtb_methods, dftbplus_methods, columbus_methods, dftd3_methods, dftd4_methods, uaiqm, omnip2x]
    
def methods(method: str = None, program: str = None, **kwargs):
    '''
    Create a model object with a specified method.

    Arguments:
        method (str): Specify the method. Available methods are listed in the section below.
        program (str, optional): Specify the program to use.
        **kwargs: Other method and program-specific options

    **Available Methods:**

        ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'AM1'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'CCSD(T)*/CBS'``, ``'CNDO/2'``, ``'D4'``, ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'GFN2-xTB'``, ``'MINDO/3'``, ``'MNDO'``, ``'MNDO/H'``, ``'MNDO/d'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM2*'``, ``'ODM3'``, ``'ODM3*'``, ``'OM1'``, ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'PM6'``, ``'RM1'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``.

        Methods listed above can be accepted without specifying a program.
        The required programs still have to be installed though as described in the installation manual.
    
    **Available Programs and Their Corresponding Methods:** 

        .. table::
            :align: center

            ===============  ==========================================================================================================================================================================
            Program          Methods                                                                                                                                                                   
            ===============  ==========================================================================================================================================================================
            TorchANI         ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'ANI-1ccx'``, ``'ANI-1x'``, ``'ANI-1x-D4'``, ``'ANI-2x'``, ``'ANI-2x-D4'``, ``'ANI-1xnr'``                                              
            dftd4            ``'AIQM1'``, ``'AIQM1@DFT'``, ``'ANI-1x-D4'``, ``'ANI-2x-D4'``, ``'D4'``                                                                                                  
            MNDO or Sparrow  ``'AIQM1'``, ``'AIQM1@DFT'``, ``'AIQM1@DFT*'``, ``'MNDO'``, ``'MNDO/d'``, ``'ODM2*'``, ``'ODM3*'``,  ``'OM2'``, ``'OM3'``, ``'PM3'``, ``'SCC-DFTB'``, ``'SCC-DFTB-heats'``
            MNDO             ``'CNDO/2'``, ``'MINDO/3'``, ``'MNDO/H'``, ``'MNDO/dH'``, ``'MNDOC'``, ``'ODM2'``, ``'ODM3'``, ``'OM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                  
            Sparrow          ``'DFTB0'``, ``'DFTB2'``, ``'DFTB3'``, ``'PM6'``, ``'RM1'``, semiempirical OMx, DFTB, NDDO-type methods                                                                                                              
            xTB              ``'GFN2-xTB'``, semiempirical GFNx-TB methods                                                                                                                                                           
            Orca             ``'CCSD(T)*/CBS'``, DFT                                                                                                                                                      
            Gaussian         ab initio methods, DFT
            PySCF            ab initio methods, DFT
            ===============  ==========================================================================================================================================================================
    
    '''

    method_instance = None
    
    if program is not None:
        program_method = f'{program.casefold()}_methods'
        if not program_method in globals():
            raise ValueError(f'The {program} is not recognized. Check the spelling or it might be not implemented.')
        program_method = globals()[program_method]
        if method is not None:
            method_instance = program_method(method=method, **kwargs)
        else:
            method_instance = program_method(**kwargs)
    elif method is not None:
        for method_class in known_classes:
            if not hasattr(method_class, 'is_method_supported'): continue
            if hasattr(method_class, 'is_program_found'):
                if method_class.is_program_found() is False: continue
            if method_class.is_method_supported(method):
                method_instance = method_class(method=method, **kwargs)
                break
        else:
            raise ValueError('''This method is not detected in any of the interfaces MLatom could find.
    Possible reasons:
    1. You might have misspelled method's name.
    2. MLatom could not find the required program on your device (check installation instructions).
    3. You might need to specify program, e.g., mymethod = mlatom.methods(method=[your method], program=[required program]), if you are sure that this program supports this method and the program is interfaced and found by MLatom. MLatom does not know all the methods available in each interfaced program.
    ''')
    
    return method_instance

def known_methods():
    supported_methods = []
    for method_class in known_classes:
        if 'supported_methods' in method_class.__dict__:
            supported_methods += method_class.supported_methods
    return supported_methods

def load(filename, format=None):
    '''
    Load a saved model object.
    '''
    if filename[-5:] == '.json' or format == 'json':
        return load_json(filename)  
    elif filename[-5:] == '.pkl' or format == 'pkl':
        return load_pickle(filename)

def load_json(filename):
    import json
    
    with open(filename) as f:
        model_dict = json.load(f)
    return load_dict(model_dict)

def load_pickle(filename):
    import pickle
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_dict(model_dict):
    type = model_dict.pop('type')
    nthreads = model_dict.pop('nthreads') if 'nthreads' in model_dict else 0
    if type == 'method':
        kwargs = {}
        if 'kwargs' in model_dict:
            kwargs = model_dict.pop('kwargs')
        model = methods(**model_dict, **kwargs)

    if type == 'ml_model':
        model = globals()[model_dict['ml_model_type'].split('.')[-1]](**model_dict['kwargs'])

    if type == 'model_tree_node':
        children = [load_dict(child_dict) for child_dict in model_dict['children']] if model_dict['children'] else None
        name = model_dict['name']
        operator = model_dict['operator']
        model = load_dict(model_dict['model']) if model_dict['model'] else None
        weight = model_dict['weight'] if 'weight' in model_dict else None 
        model = model_tree_node(name=name, children=children, operator=operator, model=model)
        if weight:
            model.weight = weight

    if type not in ['method', 'ml_model', 'model_tree_node']:
        moduleinfo = model_dict.pop('module')
        module_import = import_from_path(moduleinfo['name'], moduleinfo['path'])
        modelcls = module_import.__dict__[type]
        import inspect
        init_signature = inspect.signature(modelcls.__init__)
        init_params = init_signature.parameters
        init_kwargs = {}
        other_kwargs = {}
        for key in model_dict.keys():
            if key in init_params:
                init_kwargs[key] = model_dict[key]
            else:
                other_kwargs[key] = model_dict[key]
        if 'init_kwargs' in model_dict.keys():
            init_kwargs = model_dict['init_kwargs']
        model = modelcls(**init_kwargs)
        for key in other_kwargs.keys():
            model.__dict__[key] = other_kwargs[key]
        
    model.set_num_threads(nthreads)
    return model

def import_from_path(module_name, file_path):
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    import importlib
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    pass