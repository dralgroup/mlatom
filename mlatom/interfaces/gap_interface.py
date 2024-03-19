'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_GAP: Interface between GAP and MLatom                           ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''
from __future__ import annotations
from typing import Any, Union, Dict, Callable
import os, sys, uuid, time, tempfile, subprocess
import numpy as np
from .. import constants
from .. import data
from .. import models
from .. import stopper
from ..decorators import doc_inherit


GAPbin = os.environ['gap_fit'] if 'gap_fit' in os.environ else ''
if not GAPbin:
    print('please set $gap_fit')
    pass

QUIPbin = os.environ['quip'] if 'quip' in os.environ else ''
if not QUIPbin:
    print('please set $quip')
    pass

def molDB2extendedXYZ(file_name, molDB, 
                      property_to_learn=None,
                      xyz_derivative_property_to_learn=None):
    with open(file_name, 'w') as file:
        for mol in molDB:
            file.write(f'{len(mol)}\n')
            if property_to_learn:
                file.write(f'energy={mol.__dict__[property_to_learn]}')
            file.write(' pbc="T T T" Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" Properties=species:S:1:pos:R:3')
            if xyz_derivative_property_to_learn: 
                file.write(':forces:R:3')
            file.write('\n')

            for atom in mol:
                file.write(atom.element_symbol)
                file.write(' %12.8f %12.8f %12.8f' % tuple(atom.xyz_coordinates))
                if xyz_derivative_property_to_learn: 
                    file.write(' %12.8f %12.8f %12.8f' % tuple(-1 * atom.__dict__[xyz_derivative_property_to_learn]))
                file.write('\n')

class gap(models.ml_model):
    '''
    Create an `GAP <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.136403>`_-`SOAP <https://doi.org/10.1103/PhysRevB.87.184115>`_ model object. 
    
    Interfaces to `QUIP <https://github.com/libAtoms/QUIP>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''
    
    hyperparameters = models.hyperparameters({
        #gap
        'type':                     models.hyperparameter(value='soap'),
        'l_max':                    models.hyperparameter(value=6, minval=1, maxval=10, optimization_space='linear', dtype=int),
        'n_max':                    models.hyperparameter(value=6, minval=1, maxval=10, optimization_space='linear', dtype=int),
        'atom_sigma':               models.hyperparameter(value=0.5, minval=0.1, maxval=4, optimization_space='linear', dtype=float),
        'zeta':                     models.hyperparameter(value=4, minval=1, maxval=10, optimization_space='linear', dtype=float),
        'cutoff':                   models.hyperparameter(value=6, minval=1, maxval=10, optimization_space='linear', dtype=float),
        'cutoff_transition_width':  models.hyperparameter(value=0.5, minval=0.1, maxval=2, optimization_space='linear', dtype=float),
        'n_sparse':                 models.hyperparameter(value=1000000, minval=1, maxval=1000000, optimization_space='log', dtype=int),
        'delta':                    models.hyperparameter(value=1, minval=0.1, maxval=10, optimization_space='linear', dtype=float),
        'covariance_type':          models.hyperparameter(value='dot_product', optimization_space='choice', dtype=str), 
        'sparse_method':            models.hyperparameter(value='cur_points', optimization_space='choice', dtype=str),
        'add_species':              models.hyperparameter(value='T', choices=['T', 'F'], optimization_space='choice', dtype=str),
        #gapfit
        'default_sigma_e':          models.hyperparameter(value= 0.0005, minval=0.0001, maxval=10, optimization_space='log', dtype=float),
        'default_sigma_f':          models.hyperparameter(value= 0.001, minval=0.0001, maxval=10, optimization_space='log', dtype=float),
        'default_sigma_v':          models.hyperparameter(value= 0.1, minval=0.0001, maxval=10, optimization_space='log', dtype=float),
        'default_sigma_h':          models.hyperparameter(value= 0.1, minval=0.0001, maxval=10, optimization_space='log', dtype=float),
        'e0_method':                models.hyperparameter(value='average',  optimization_space='choice', dtype=str),
        'sparse_separate_file':     models.hyperparameter(value='F', choices=['T', 'F'], optimization_space='choice', dtype=str)
    })
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
    gapfitdict = {
        'at_file': '',
        'default_sigma': '{0.0005, 0.001, 0.1, 0.1}', 
        'gp_file': '', 
        'gap': '', 
        'e0_method': 'average',
        'sparse_separate_file': 'F',
        'verbosity': 'NORMAL',
    }

    gap = '{}'
    gapfit = []

    property_name = 'y'
    program = 'QUIP'
    meta_data = {
        "genre": "kernel method"
    }
    model_file = None
    model = None
    verbose = 1

    def __init__(self, model_file=None, hyperparameters={}, verbose=1,):
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.verbose = verbose
        if model_file: 
            self.model_file = model_file
        else:
            self.model_file = f'GAPmodel_{str(uuid.uuid4())}.xml'
    
    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'gapfit' in args.data and hyperparam in args.gapfit.data:
                self.hyperparameters[hyperparam].value = args.gapfit.data[hyperparam]
            elif 'gapfit' in args.data and 'gap' in args.gapfit.data and hyperparam in args.gapfit.gap.data:
                self.hyperparameters[hyperparam].value = args.gapfit.gap.data[hyperparam]

    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        hyperparameters: Union[Dict[str,Any], models.hyperparameters] = {},
        stdout=None,
        stderr=None,
    ):
        
        if not self.verbose:
            FNULL = open(os.devnull, 'w')
            if not stdout:
                stdout = FNULL
            if not stderr:
                stderr = FNULL
        else:
            stdout = sys.stdout
            stderr = sys.stderr

        if os.path.exists(self.model_file):
            self.model_file = f'GAPmodel_{str(uuid.uuid4())}.xml'

        if self.verbose:
            print(f'Trained GAP model will be saved in {self.model_file}.')

        with tempfile.TemporaryDirectory() as tmpdirname:
            molDB2extendedXYZ(f'{tmpdirname}/training_set.exyz', molecular_database, property_to_learn, xyz_derivative_property_to_learn)

            self.hyperparameters.update(hyperparameters)
            for k, v in self.hyperparameters.items():
                if k in self.gapdict:
                    self.gapdict[k] = v.value
                if k in self.gapfitdict:
                    self.gapfitdict[k] = v.value

            if self.gapdict['n_sparse'] > 6 * len(molecular_database):
                self.gapdict['n_sparse'] = 6 * len(molecular_database)

            self.gapfitdict['default_sigma'] = f'{{{self.hyperparameters.default_sigma_e}, {self.hyperparameters.default_sigma_f}, {self.hyperparameters.default_sigma_v}, {self.hyperparameters.default_sigma_h}}}'

            if property_to_learn:
                self.gapfitdict['energy_parameter_name'] = 'energy'
            if xyz_derivative_property_to_learn:
                self.gapfitdict['force_parameter_name'] = 'forces'
            self.gap = '{' + ' '.join([str(k)+'='+str(v) for k, v in self.gapdict.items()])[5:] + '}'
            self.gapfitdict.update({
                'at_file':  f'{tmpdirname}/training_set.exyz',
                'gp_file':  self.model_file,
                'gap':      self.gap
            })
            
            self.gapfit = [GAPbin]
            for k, v in self.gapfitdict.items():
                self.gapfit.append(k + '=' + v)
                
            if self.verbose:
                print('> '+' '.join(self.gapfit))
                sys.stdout.flush()
            subprocess.call(self.gapfit, stdout=stdout, stderr=stderr)
        
        if not self.verbose:
            FNULL.close()
    
    @doc_inherit
    def predict(
        self, 
        molecular_database: data.molecular_database = None, 
        molecule: data.molecule = None,
        calculate_energy: bool = False,
        calculate_energy_gradients: bool = False, 
        calculate_hessian: bool = False,
        property_to_predict: Union[str, None] = 'estimated_y', 
        xyz_derivative_property_to_predict: Union[str, None] = None, 
        hessian_to_predict: Union[str, None] = None,
    ) -> None:
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)
 
        with tempfile.TemporaryDirectory() as tmpdirname:
            molDB2extendedXYZ(f'{tmpdirname}/geometries.exyz', molDB)

            p1 = subprocess.Popen([
                QUIPbin,
                'E=T' if property_to_predict else 'E=F',
                'F=T' if xyz_derivative_property_to_predict else 'F=F',
                f'atoms_filename={tmpdirname}/geometries.exyz',
                f'param_filename={self.model_file}',
            ], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(['grep','AT'], stdin=p1.stdout, stdout=subprocess.PIPE)
            (stdout, stderr) = p2.communicate() 
            p1.stdout.close()

            result = stdout.decode('ascii').strip().split('\n')

            if property_to_predict:
                enest = np.array([float(line.split()[1].split('=')[1]) for line in result[1::len(molDB[0])+2]])
                molDB.add_scalar_properties(enest, property_to_predict)

            if xyz_derivative_property_to_predict:
                gradest = np.empty((len(molDB),len(molDB[0]),3))
                for i in range(len(molDB[0])):
                    gradest[:, i] = -1 * np.array([line.split()[-3:] for line in result[2+i::len(molDB[0])+2]]).astype(float)
                
                molDB.add_xyz_vectorial_properties(gradest, xyz_derivative_property_to_predict)

def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
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