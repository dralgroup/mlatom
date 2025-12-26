'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! ciopt: Module for minimum-energy conical intersection optimization        ! 
  ! Implementations by: Mikolaj Martyka, 4 November 2025                      ! 
  !---------------------------------------------------------------------------! 
'''
import os
from . import data
from .simulations import optimize_geometry
from .model_cls import model 
class model_for_ciopt(model):
    def __init__(self, model=None,state_0 = 0, state_1 = 1, sigma=3.5, alpha=0.02):
        self.model = model
        self.state_0 = state_0
        self.state_1 = state_1
        self.sigma = sigma
        self.alpha = alpha
        self.nstates=state_1+1
    def predict(self, molecule=None, current_state=None, calculate_energy=True,
            calculate_energy_gradients=None, nstates=None, **kwargs):


        current_state = self.state_0
        nstates = self.nstates
        calculate_energy_gradients = [True] * nstates
    
        self.model._predict_geomopt(
            molecule=molecule,
            current_state=current_state,
            calculate_energy=calculate_energy,
            calculate_energy_gradients=calculate_energy_gradients,
            nstates=nstates,
            **kwargs
        )
        energy_lower = molecule.electronic_states[self.state_0].energy
        energy_upper = molecule.electronic_states[self.state_1].energy
        gradient_lower = molecule.electronic_states[self.state_0].get_energy_gradients()
        gradient_upper = molecule.electronic_states[self.state_1].get_energy_gradients()
        E_avg = (energy_upper+energy_lower)/2
        grad_avg = (gradient_upper+gradient_lower)/2
        E_diff = energy_upper-energy_lower
        grad_diff = gradient_upper - gradient_lower
        Epen = self.sigma * E_diff**2/(E_diff+self.alpha)
        Gpen = self.sigma*(E_diff**2+2*self.alpha*E_diff)/((E_diff+self.alpha)**2) * grad_diff
        molecule.energy = E_avg+Epen
        molecule.energy_gradients = (grad_avg+Gpen)

class ci_opt():
    """
    Perform a geometry optimization of a molecule with a two-state penalty scheme,
    following the method of penalty-constrained optimization described by
    Levine et al. (J. Phys. Chem. B 112(2):405-413, 2008. DOI: https://doi.org/10.1021/jp0761618).

    Example:

    .. code-block:: python
    
        mol = ml.data.molecule()
        mol.read_from_smiles_string("C=C")
        odm2 = ml.models.methods(method='ODM2', read_keywords_from_file='mndokw')
        ml.ci_opt(molecule=mol, model=odm2, maximum_number_of_steps=400)

    Parameters:

        model (:class:`mlatom.models.model`):
            Model providing energies and gradients for the molecule.
        model_predict_kwargs (dict, optional):
            Additional keyword arguments passed to the model's prediction routine.
        initial_molecule : :class:`mlatom.data.molecule`, optional
            Starting geometry (cannot be used with `molecule`).
        molecule : :class:`mlatom.data.molecule`, optional
            Molecule to be optimized in place (cannot be used with `initial_molecule`).
        maximum_number_of_steps : int, optional
            Maximum number of optimization steps (default 200).
        working_directory : str, optional
            Directory for output files (default current directory).
        print_properties : list or None, optional
            Properties to print at each step; enables trajectory dumping if set.
        dump_trajectory_interval : int or None, optional
            Interval for writing trajectory frames (1 or None supported).
        filename : str, optional
            Trajectory file name; generated automatically if omitted.
        format : {'json', 'h5md'}, default='json'
            Format for trajectory output.
        program : {'geometric', 'scipy'}, default='geometric'
            Optimization backend. Raises an error if unsupported.
        program_kwargs : dict, optional
            Additional keyword arguments for the optimization backend.
        state_0 : int, default=0
            Index of the lower electronic state.
        state_1 : int, default=1
            Index of the upper electronic state.
        sigma : float, default=3.5
            Dimensionless scaling factor for the CI penalty term.
        alpha : float, default=0.02
            Energy-scale regularization parameter (same units as energy).
    """
    
    def __init__(self, model=None,  model_predict_kwargs={}, initial_molecule=None, molecule=None, maximum_number_of_steps=None, working_directory=None, 
    print_properties=None,
    dump_trajectory_interval=None,program=None, program_kwargs=None,convergence_criterion_for_forces=None,
    filename=None, format='json', state_0 = 0, state_1 = 1, sigma=3.5, alpha=0.02): 
        if model != None:
            self.model = model
        self.print_properties = print_properties
        self.model_predict_kwargs = model_predict_kwargs

        if not initial_molecule is None and not molecule is None:
            raise ValueError('molecule and initial_molecule cannot be used at the same time')
        if not initial_molecule is None:
            self.initial_molecule = initial_molecule.copy()
        if not molecule is None:
            self.initial_molecule = molecule
        
        self.alpha = alpha
        self.sigma = sigma
        self.state_0 = state_0
        self.state_1 = state_1
        if maximum_number_of_steps != None: self.maximum_number_of_steps = maximum_number_of_steps
        else: self.maximum_number_of_steps = 200
        

        if working_directory != None:
            self.working_directory = working_directory
        else:
            self.working_directory = '.'
        
        self.dump_trajectory_interval = dump_trajectory_interval
        if program != None:
            self.program = program
        else:
            self.program = 'geometric'
        if self.program.lower() not in ('geometric', 'scipy'):
            raise ValueError(
                f"Unsupported optimization program '{program}'. "
                "Allowed options are 'geometric' or 'scipy'."
            )

        if self.program.casefold() == 'geometric'.casefold(): self.dump_trajectory_interval = 1
        self.filename = filename
        self.format = format
        if not program_kwargs: self.program_kwargs = {}
        else: self.program_kwargs = program_kwargs
        if self.print_properties != None and self.dump_trajectory_interval == None:
            self.dump_trajectory_interval = 1
        if self.dump_trajectory_interval != None:
            self.format = format
            if format == 'h5md': ext = '.h5'
            elif format == 'json': ext = '.json'
            if self.filename == None:
                import uuid
                self.filename = str(uuid.uuid4()) + ext 
            # Dump trajectory every step
            self.optimization_trajectory = data.molecular_trajectory()
            self.optimization_trajectory.dump(filename=os.path.join(self.working_directory,self.filename), format=self.format) 
        
        # Pack the required geomopt-related kwargs into the model kwargs
        self.model_predict_kwargs['return_string'] = False
        self.model_predict_kwargs['dump_trajectory_interval'] = self.dump_trajectory_interval
        self.model_predict_kwargs['filename'] = self.filename
        self.model_predict_kwargs['format'] = self.format
        self.model_predict_kwargs['print_properties'] = self.print_properties
        
        optmodel = model_for_ciopt(model=self.model, state_0=self.state_0, state_1 =self.state_1, sigma=self.sigma, alpha=self.alpha)
        _ = optimize_geometry(molecule=self.initial_molecule, model=optmodel, program=self.program, model_predict_kwargs=self.model_predict_kwargs, program_kwargs=self.program_kwargs, maximum_number_of_steps=self.maximum_number_of_steps, working_directory=self.working_directory, print_properties=self.print_properties,
        dump_trajectory_interval=self.dump_trajectory_interval, filename=self.filename, format=self.format)
        


