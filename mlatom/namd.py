#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! namd: Module for nonadiabatic molecular dynamics                          ! 
  ! Implementations by: Lina Zhang & Pavlo O. Dral                            ! 
  ! Implementation of FSSH by: Jakub Martinka                                 ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os
from collections import Counter
from . import data
from . import constants
from .md import md as md

def generate_random_seed():
    return int(np.random.randint(0, 2**31 - 1))

class surface_hopping_md():
    '''
    Surface-hopping molecular dynamics

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces (and couplings).
        model_predict_kwargs (Dict, optional): Kwargs of model prediction
        molecule_with_initial_conditions (:class:`data.molecule`): The molecule with initial conditions.
        molecule (:class:`data.molecule`): Work the same as molecule_with_initial_conditions
        ensemble (str, optional): Which kind of ensemble to use.
        thermostat (:class:`thermostat.Thermostat`): The thermostat applied to the system.
        time_step (float): Time step in femtoseconds.
        time_step_tdse (float): Sub-time step in femtoseconds.
        integrator (str, optional): Integrator of semiclassical time dependent Schrödinger equation.
        decoherence_model (str, optional): Decoherence correction type
        decoherence_SDM_decay (float, optional): Decay used in Simplified Decay of Mixing decoherence correction.
        coupling_calc_threshold (float, optional): Calculate NACs/TDBAs only, when dE < coupling_calc_threshold (units: eV; typical values are 0.5-0.6 eV). By default, NACs/TDBA are calculated for all time steps.
        maximum_propagation_time (float): Maximum propagation time in femtoseconds.
        dump_trajectory_interval (int, optional): Dump trajectory at which interval. Set to ``None`` to disable dumping.
        filename (str, optional): The file that saves the dumped trajectory
        format (str, optional): Format in which the dumped trajectory is saved
        stop_function (any, optional): User-defined function that stops MD before ``maximum_propagation_time``
        stop_function_kwargs (Dict, optional): Kwargs of ``stop_function``
        hopping_algorithm (str, optional): Surface hopping algorithm (default: 'LZBL', also supported: 'FSSH')
        nstates (int): Number of states 
        initial_state (int): Initial state
        random_seed (int): Random seed 
        random_numbers (list): List of predefined random numbers.
        prevent_back_hop (bool, optional): Whether to prevent back hopping
        rescale_velocity_direction (string, optional): Rescale velocity direction (defaults: 'momentum' for LZBL, 'nacv' for FSSH, also available: 'gradient difference')
        reduce_kinetic_energy (bool, optional): Whether to reduce kinetic energy


    .. table:: 
       :align: center

       =====================  ================================================
        ensemble                description
       =====================  ================================================
        ``'NVE'`` (default)     Microcanonical (NVE) ensemble
        ``'NVT'``               Canonical (NVT) ensemble
       =====================  ================================================

    .. table::
        :align: center

        =======================================  ==============================
         thermostat                                    description
        =======================================  ==============================
         :class:`ml.md.Andersen_thermostat`         Andersen thermostat
         :class:`ml.md.Nose_Hoover_thermostat`      Hose-Hoover thermostat
         ``None`` (default)                         No thermostat is applied
        =======================================  ==============================

    For theoretical details, see and cite original paper:
    Lina Zhang, Sebastian V. Pios, Mikołaj Martyka, Fuchun Ge, Yi-Fan Hou, Yuxinxin Chen, Lipeng Chen, Joanna Jankowska, Mario Barbatti, and `Pavlo O. Dral <http://dr-dral.com>`__. MLatom Software Ecosystem for Surface Hopping Dynamics in Python with Quantum Mechanical and Machine Learning Methods. Journal of Chemical Theory and Computation 2024 20 (12), 5043-5057. DOI: 10.1021/acs.jctc.4c00468 

    For FSSH implementation details
    Jakub Martinka, Lina Zhang, Yi-Fan Hou, Mikołaj Martyka, Jiří Pittner, Mario Barbatti, and `Pavlo O. Dral <http://dr-dral.com>`__. A Descriptor Is All You Need: Accurate Machine Learning of Nonadiabatic Coupling Vectors. **2025**. Preprint on arXiv: https://arxiv.org/abs/2505.23344 (2025.05.29).

    Examples:

    .. code-block:: python
    
        # Propagate multiple surface-hopping trajectories in parallel
        # .. setup LZBL dynamics calculations
        namd_kwargs = {
                    'model': aiqm1,
                    'time_step': 0.25,
                    'maximum_propagation_time': 5,
                    'hopping_algorithm': 'LZBL',
                    'nstates': 3,
                    'initial_state': 2,
                    }

        # .. setup FSSH dynamics calculations
        namd_kwargs = {
                    'model': model,
                    'time_step': 0.1,
                    'time_step_tdse': 0.005,
                    'maximum_propagation_time': 5,
                    'hopping_algorithm': 'FSSH',
                    # the lines commented out are the defaults
                    #'decoherence_model': 'SDM',
                    #'rescale_velocity_direction': 'nacv',
                    #'prevent_back_hop': False,
                    'nstates': 3,
                    'initial_state': 2
                    }

        # .. setup TDBA dynamics calculations
        namd_kwargs = {
                    'model': model,
                    'time_step': 0.1,
                    'time_step_tdse': 0.005,
                    'maximum_propagation_time': 5,
                    'hopping_algorithm': 'TDBA',
                    # the lines commented out are the defaults
                    #'decoherence_model': 'SDM',
                    #'rescale_velocity_direction': 'momentum',
                    #'prevent_back_hop': False,
                    'nstates': 3,
                    'initial_state': 2
                    }

        # .. run trajectories in parallel
        dyns = ml.simulations.run_in_parallel(molecular_database=init_cond_db,
                                            task=ml.namd.surface_hopping_md,
                                            task_kwargs=namd_kwargs,
                                            create_and_keep_temp_directories=True)
        trajs = [d.molecular_trajectory for d in dyns]

        # Dump the trajectories
        itraj=0 
        for traj in trajs:
            itraj+=1 
            traj.dump(filename=f"traj{itraj}.h5",format='h5md')

        # Analyze the result of trajectories and make the population plot
        ml.namd.analyze_trajs(trajectories=trajs, maximum_propagation_time=5)
        ml.namd.plot_population(trajectories=trajs, time_step=0.25, 
                            max_propagation_time=5, nstates=3, filename=f'pop.png',
                            pop_filename='pop.txt')
                            
    .. note::

        Trajectory is saved in ``ml.md.molecular_trajectory``, which is a :class:`ml.data.molecular_trajectory` class

    .. warning:: 

        In MLatom, energy unit is Hartree and distance unit is Angstrom. Make sure that the units in your model are consistent.
        
    '''
    def __init__(self, model=None,
                 model_predict_kwargs={},
                 molecule_with_initial_conditions=None,
                 molecule=None,
                 ensemble='NVE',
                 thermostat=None,
                 time_step=0.1,
                 time_step_tdse=None,
                 integrator='Butcher',
                 decoherence_model='SDM',
                 decoherence_SDM_decay=0.1,
                 coupling_calc_threshold=None,
                 maximum_propagation_time=100,
                 dump_trajectory_interval=None,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None,
                 hopping_algorithm='LZBL',
                 nstates=None, initial_state=None,
                 random_seed=generate_random_seed,
                 prevent_back_hop=False, 
                 reduce_memory_usage = False,
                 rescale_velocity_direction=None,
                 insufficient_energy_action=None,
                 reduce_kinetic_energy=False):
        self.model = model
        self.model_predict_kwargs = model_predict_kwargs
        if not molecule_with_initial_conditions is None:
            self.molecule_with_initial_conditions = molecule_with_initial_conditions 
        if not molecule is None:
            self.molecule_with_initial_conditions = molecule
        self.ensemble = ensemble
        if thermostat != None:
            self.thermostat = thermostat
        self.time_step = time_step
        self.maximum_propagation_time = maximum_propagation_time
        self.reduce_memory_usage = reduce_memory_usage
        
        self.dump_trajectory_interval = dump_trajectory_interval
        self.format = format
        self.filename = filename 

        if dump_trajectory_interval != None:
            if format == 'h5md': ext = '.h5'
            elif format == 'json': ext = '.json'
            if filename == None:
                import uuid
                filename = str(uuid.uuid4()) + ext
                self.filename = filename 


        self.stop_function = stop_function
        self.stop_function_kwargs = stop_function_kwargs
        
        if hopping_algorithm.casefold() == 'fssh':
            self.hopping_algorithm = 'FSSH'
        elif hopping_algorithm.casefold() in ['lzbl', 'lzsh']:
            self.hopping_algorithm = 'LZBL'
        elif hopping_algorithm.casefold() == 'tdba':
            self.hopping_algorithm = 'TDBA'
        else:
            raise ValueError("Invalid hopping_algorithm. Possible values are: 'FSSH', 'TDBA' and 'LZBL' (default).")
        self.nstates=nstates
        self.initial_state = initial_state
        if self.initial_state is None:
            if 'current_state' in self.molecule_with_initial_conditions.__dict__:
                self.initial_state = self.molecule_with_initial_conditions.current_state
            else:
                self.initial_state = self.nstates - 1
        self.random_seed = random_seed
        self.prevent_back_hop = prevent_back_hop
        if type(rescale_velocity_direction) is str:
            if rescale_velocity_direction not in ['momentum', 'along velocities', 'nacv', 'nacs', 'gradient difference']:
                raise ValueError("Invalid rescale_velocity_direction. Possible values are: 'momentum' (same as 'along velocities'), 'nacv' (same as 'nacs'), 'gradient difference'.")
            else:
                self.rescale_velocity_direction = rescale_velocity_direction
        elif rescale_velocity_direction is None:
            if self.hopping_algorithm == 'FSSH':
                self.rescale_velocity_direction = 'nacv'
            else:
                self.rescale_velocity_direction = 'momentum'
        else:
            raise ValueError("Invalid rescale_velocity_direction. Possible values are: 'momentum' (same as 'along velocities'), 'nacv' (same as 'nacs'), 'gradient difference'.")
        if type(insufficient_energy_action) is str:
            if insufficient_energy_action not in ['do not change velocities', 'zero velocities', 'raise error']:
                raise ValueError("Invalide insufficient_energy_action. Possible values are: 'do not change velocities', 'zero velocities', 'raise error'.")
            else:
                self.insufficient_energy_action = insufficient_energy_action
        elif insufficient_energy_action is None:
            #if self.hopping_algorithm == 'FSSH':
                self.insufficient_energy_action = 'do not change velocities'
            #else:
            #    self.insufficient_energy_action = 'zero velocities'
        else:
            raise ValueError("Invalide insufficient_energy_action. Possible values are: 'do not change velocities', 'zero velocities', 'raise error'.")
        self.reduce_kinetic_energy = reduce_kinetic_energy
        if self.reduce_kinetic_energy:
            if self.molecule_with_initial_conditions.is_it_linear():
                self.degrees_of_freedom = 3 * len(self.molecule_with_initial_conditions.atoms) - 5
            else:
                self.degrees_of_freedom = 3 * len(self.molecule_with_initial_conditions.atoms) - 6
            self.reduce_kinetic_energy_factor = self.degrees_of_freedom
        else:
            self.reduce_kinetic_energy_factor = 1
        if self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
            if time_step_tdse == None:
                self.time_step_tdse = time_step / 20
            else:
                if round(1e6*time_step) % round(1e6*time_step_tdse) != 0:
                    raise ValueError("time_step must be divisible by time_step_tdse.")
                self.time_step_tdse = time_step_tdse
            if integrator not in ['RK4', 'AM5', 'Butcher']:
                raise ValueError("Invalid integrator. Possible values are: 'RK4', 'AM5', and 'Butcher'.")
            else:
                self.integrator = integrator
            self.decoherence_model = decoherence_model
            self.decoherence_SDM_decay = decoherence_SDM_decay
            self.coupling_calc_threshold = coupling_calc_threshold
        self.propagate()

    def propagate(self):
        self.molecular_trajectory = data.molecular_trajectory()

        istep = 0
        stop = False
        if callable(self.random_seed):
            np.random.seed(self.random_seed())
        else:
            np.random.seed(self.random_seed)
        self.current_state = self.initial_state
        # if self.model_predict_kwargs == {}:
        #     calculate_energy_gradients = [False] * self.nstates
        #     calculate_energy_gradients[self.current_state] = True
        #     self.model_predict_kwargs={'nstates':self.nstates, 
        #                                'current_state':self.current_state,
        #                                'calculate_energy':True,
        #                                'calculate_energy_gradients':calculate_energy_gradients}
        if self.model_predict_kwargs == {}:
            if self.hopping_algorithm == 'FSSH':
                self.model_predict_kwargs={'nstates':self.nstates, 
                                           'current_state':self.current_state,
                                           'calculate_energy':True,
                                           'calculate_energy_gradients':[True] * self.nstates,
                                           'calculate_nacv':True}
            else:
                self.model_predict_kwargs={'nstates':self.nstates, 
                                           'current_state':self.current_state,
                                           'calculate_energy':True,
                                           'calculate_energy_gradients':[True] * self.nstates}
        else:
            self.model_predict_kwargs['nstates'] = self.nstates
            self.model_predict_kwargs['current_state'] = self.current_state
            self.model_predict_kwargs['calculate_energy'] = True
            if 'calculate_energy_gradients' in self.model_predict_kwargs:
                if self.model_predict_kwargs['calculate_energy_gradients'] != True:
                    if not isinstance(self.model_predict_kwargs['calculate_energy_gradients'], list):
                        self.model_predict_kwargs['calculate_energy_gradients'] = [False] * self.nstates
                    self.model_predict_kwargs['calculate_energy_gradients'][self.current_state] = True
            else:
                if self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                    # All gradients might be needed in some cases like rescaling velocities along the gradient difference or when the gradient difference is needed for as descriptor for ML-NAC model
                    self.model_predict_kwargs['calculate_energy_gradients'] = [True] * self.nstates
                else:
                    self.model_predict_kwargs['calculate_energy_gradients'] = [False] * self.nstates
                    self.model_predict_kwargs['calculate_energy_gradients'][self.current_state] = True
            if self.hopping_algorithm == 'FSSH':
                self.model_predict_kwargs['calculate_nacv'] = True

        one_step_propagation = True

        while not stop:
            if istep == 0:
                molecule = self.molecule_with_initial_conditions.copy(atomic_labels=['xyz_coordinates','xyz_velocities'], molecular_labels=[])
            else:
                molecule = self.molecular_trajectory.steps[-1].molecule.copy(atomic_labels=['xyz_coordinates','xyz_velocities'], molecular_labels=[])
            
            self.model_predict_kwargs['current_state'] = self.current_state
            # self.model_predict_kwargs['calculate_energy_gradients'][self.initial_state] = self.model_predict_kwargs['calculate_energy_gradients'][self.current_state]
            # self.model_predict_kwargs['calculate_energy_gradients'][self.current_state] = True

            if one_step_propagation:
                dyn = md(model=self.model,
                         model_predict_kwargs=self.model_predict_kwargs,
                         molecule_with_initial_conditions=molecule,
                         ensemble='NVE',
                         thermostat=None,
                         time_step=self.time_step,
                         maximum_propagation_time=self.time_step,
                         dump_trajectory_interval=None,
                         filename=None, format='h5md')
                if istep == 0:
                    self.molecular_trajectory.steps.extend(dyn.molecular_trajectory.steps)
                else:
                    self.molecular_trajectory.steps.append(dyn.molecular_trajectory.steps[-1])
                    self.molecular_trajectory.steps[-1].step = istep + 1
                    self.molecular_trajectory.steps[-1].time = (istep + 1) * self.time_step
            if istep == 0:
                self.molecular_trajectory.steps[istep].current_state = self.current_state
                if self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                    self.molecular_trajectory.steps[istep].state_coefficients = np.zeros(self.nstates, dtype=complex)
                    self.molecular_trajectory.steps[istep].state_coefficients[self.initial_state] = 1
            # fssh/lzsh/znsh: prob list
            if self.hopping_algorithm == 'LZBL':
                random_number = np.random.random()
                hopping_probabilities = self.lzsh(istep=istep)
            elif self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                hopping_probabilities, random_number = self.fssh(istep=istep)
            self.molecular_trajectory.steps[-2].random_number = random_number
            self.molecular_trajectory.steps[-2].hopping_probabilities = hopping_probabilities
            max_prob = max(hopping_probabilities)
            if max_prob > random_number:
                self.initial_state = self.current_state
                self.current_state = hopping_probabilities.index(max_prob)

                # fssh/lzsh/znsh: rescale_velocity; change en grad in molecular_trajectory; change ekin etot
                mol_istep_plus1 = self.molecular_trajectory.steps[-2].molecule
                kinetic_energy_change = mol_istep_plus1.electronic_states[self.current_state].energy - mol_istep_plus1.electronic_states[self.initial_state].energy
                vector = None
                if self.rescale_velocity_direction.casefold() in ['nacv', 'nacs']:
                    vector = np.array(mol_istep_plus1.nacv[self.current_state][self.initial_state]) / constants.Angstrom2Bohr
                elif self.rescale_velocity_direction.casefold() in ['momentum', 'along velocities']:
                    vector = 'velocities'
                elif self.rescale_velocity_direction == 'gradient difference':
                    vector = np.array(mol_istep_plus1.electronic_states[self.initial_state].get_energy_gradients() - mol_istep_plus1.electronic_states[self.current_state].get_energy_gradients()) / constants.Angstrom2Bohr
                mol_istep_plus1.rescale_velocities(
                    kinetic_energy_change=kinetic_energy_change,
                    vector=vector,
                    if_not_enough_kinetic_energy=self.insufficient_energy_action)
                self.change_properties_of_hopping_step()
                if self.hopping_algorithm == 'LZBL' or self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                    del self.molecular_trajectory.steps[-1]
                    one_step_propagation = True
                    self.molecular_trajectory.steps[-1].current_state = self.current_state
                    if type(self.stop_function) != type(None):
                        if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                        if 'stop_check' not in locals():
                            stop_check = False
                        stop, stop_check = self.stop_function(stop_check=stop_check,
                                                              mol=self.molecular_trajectory.steps[-1].molecule, 
                                                              current_state=self.current_state, 
                                                              **self.stop_function_kwargs)
                    if stop:
                        del self.molecular_trajectory.steps[-1]
                        if self.reduce_memory_usage: 
                            self.molecular_trajectory.dump(filename=self.filename, format=self.format)
            elif self.hopping_algorithm == 'LZBL' or self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                one_step_propagation = False
                self.molecular_trajectory.steps[-2].current_state = self.current_state

                if type(self.stop_function) != type(None):
                    if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                    if 'stop_check' not in locals():
                        stop_check = False
                    stop, stop_check = self.stop_function(stop_check=stop_check,
                                                          mol=self.molecular_trajectory.steps[-2].molecule, 
                                                          current_state=self.current_state, 
                                                          **self.stop_function_kwargs)
                    if stop:
                        del self.molecular_trajectory.steps[-1]
                        if self.reduce_memory_usage: 
                            self.molecular_trajectory.dump(filename=self.filename, format=self.format)

            # Dump trajectory at some interval
            if self.dump_trajectory_interval != None:
                if istep % self.dump_trajectory_interval == 0 and istep !=0:
                    if self.format == 'h5md':
                        temp_traj_dump = data.molecular_trajectory()
                        temp_traj_dump.steps.append(self.molecular_trajectory.steps[-1])
                        if self.reduce_memory_usage:
                            temp_traj = data.molecular_trajectory()
                            if self.hopping_algorithm == 'LZBL':
                                temp_traj.steps.append(self.molecular_trajectory.steps[-2])
                                temp_traj.steps.append(self.molecular_trajectory.steps[-1])
                                del self.molecular_trajectory.steps[-2:] 
                                self.molecular_trajectory.dump(filename=self.filename, format=self.format)
                                del self.molecular_trajectory
                                self.molecular_trajectory = temp_traj
                            if self.hopping_algorithm == 'FSSH' or self.hopping_algorithm == 'TDBA':
                                temp_traj.steps.append(self.molecular_trajectory.steps[-4])
                                temp_traj.steps.append(self.molecular_trajectory.steps[-3])
                                temp_traj.steps.append(self.molecular_trajectory.steps[-2])
                                temp_traj.steps.append(self.molecular_trajectory.steps[-1])
                                del self.molecular_trajectory.steps[-4:] 
                                self.molecular_trajectory.dump(filename=self.filename, format=self.format)
                                del self.molecular_trajectory
                                self.molecular_trajectory = temp_traj
                        else:
                            temp_traj_dump.dump(filename=self.filename, format=self.format)
                    elif self.format == 'json':
                        temp_traj_dump = self.molecular_trajectory
                        temp_traj_dump.dump(filename=self.filename, format=self.format)
            istep += 1

            # if self.dump_trajectory_interval != None:
            #     if (istep - 1) == 0:
            #         if self.format == 'h5md':
            #             temp_traj = data.molecular_trajectory()
            #             temp_traj.steps.append(self.molecular_trajectory.steps[0])
            #         elif self.format == 'json':
            #             temp_traj.steps.append(self.molecular_trajectory.steps[0])
            #     if istep % self.dump_trajectory_interval == 0:
            #         if self.format == 'h5md':
            #             if (istep - 1) != 0:
            #                 temp_traj = data.molecular_trajectory()
            #             temp_traj.steps.append(self.molecular_trajectory.steps[istep])
            #         elif self.format == 'json':
            #             temp_traj.steps.append(self.molecular_trajectory.steps[istep])
            #         temp_traj.dump(filename=self.filename, format=self.format)

            if istep * self.time_step + 1e-6 > self.maximum_propagation_time:
                stop = True

        if float(f"{self.molecular_trajectory.steps[-1].time:.6f}") > self.maximum_propagation_time:
            del self.molecular_trajectory.steps[-1]
            if self.reduce_memory_usage or self.filename: 
                self.molecular_trajectory.dump(filename=self.filename, format=self.format)
                
    def lzsh(self, istep=None):
        self.model_predict_kwargs['current_state'] = self.current_state
        dyn = md(model=self.model,
                model_predict_kwargs=self.model_predict_kwargs,
                molecule_with_initial_conditions=self.molecular_trajectory.steps[-1].molecule.copy(atomic_labels=['xyz_coordinates','xyz_velocities'], molecular_labels=[]),
                ensemble='NVE',
                thermostat=None,
                time_step=self.time_step,
                maximum_propagation_time=self.time_step,
                dump_trajectory_interval=None,
                filename=None, format='h5md')
        self.molecular_trajectory.steps.append(dyn.molecular_trajectory.steps[-1])
        self.molecular_trajectory.steps[-1].step = istep + 2
        self.molecular_trajectory.steps[-1].time = (istep + 2) * self.time_step
        
        hopping_probabilities = []
        for stat in range(self.nstates):
            gap_per_stat = []
            if stat == self.current_state:
                prob = -1.0      
            else:
                for iistep in [-3, -2, -1]:
                    gap_per_stat.append(abs(self.molecular_trajectory.steps[iistep].molecule.electronic_states[self.current_state].energy
                                        -self.molecular_trajectory.steps[iistep].molecule.electronic_states[stat].energy))
                if (gap_per_stat[0] > gap_per_stat[1]) and (gap_per_stat[2] > gap_per_stat[1]):
                    if not self.prevent_back_hop:
                        #if (stat > self.current_state) and (self.molecular_trajectory.steps[istep+1].molecule.kinetic_energy < gap_per_stat[1]):
                        if ((stat > self.current_state) and 
                        ((self.molecular_trajectory.steps[-1].molecule.kinetic_energy/(self.reduce_kinetic_energy_factor)) 
                         < gap_per_stat[1])):
                            prob = -1.0
                        else:
                            prob = self.lz_prob(gap_per_stat)
                    else:
                        if stat > self.current_state:
                            prob = -1.0
                        else:
                            prob = self.lz_prob(gap_per_stat)
                else:
                    prob = -1.0
            hopping_probabilities.append(prob)
        return hopping_probabilities

    def lz_prob(self, gap_per_stat):
        gap = gap_per_stat[1]
        gap_sotd = ((gap_per_stat[2] + gap_per_stat[0] - 2 * gap) / (self.time_step * constants.fs2au)**2)
        return np.exp((-np.pi/2.0) * np.sqrt(abs(gap)**3 / abs(gap_sotd)))
    
    def fssh(self, istep=None):
        self.model_predict_kwargs['current_state'] = self.current_state
        dyn = md(model=self.model,
                model_predict_kwargs=self.model_predict_kwargs,
                molecule_with_initial_conditions=self.molecular_trajectory.steps[-1].molecule.copy(atomic_labels=['xyz_coordinates','xyz_velocities'], molecular_labels=[]),
                ensemble='NVE',
                thermostat=None,
                time_step=self.time_step,
                maximum_propagation_time=self.time_step,
                dump_trajectory_interval=None,
                filename=None, format='h5md')
        self.molecular_trajectory.steps.append(dyn.molecular_trajectory.steps[-1])
        self.molecular_trajectory.steps[-1].step = istep + 2
        self.molecular_trajectory.steps[-1].time = (istep + 2) * self.time_step

        n_substeps = int(self.time_step / self.time_step_tdse)

        # Load energies and interpolate
        energies = [np.array([self.molecular_trajectory.steps[-2].molecule.electronic_states[iState].energy for iState in range(self.nstates)]), np.array([self.molecular_trajectory.steps[-3].molecule.electronic_states[stat].energy for stat in range(self.nstates)])]
        if istep == 1:
            energies.append(np.array([self.molecular_trajectory.steps[-4].molecule.electronic_states[iState].energy for iState in range(self.nstates)]))
        elif istep > 1:
            energies.append(np.array([self.molecular_trajectory.steps[-4].molecule.electronic_states[iState].energy for iState in range(self.nstates)]))
            energies.append(np.array([self.molecular_trajectory.steps[-5].molecule.electronic_states[iState].energy for iState in range(self.nstates)]))
        energies_interpolated = [list(ienergy) for ienergy in zip(*self.interpolate_energy(energies, n_substeps))]

        # Load velocities
        velocity_initial = np.array(self.molecular_trajectory.steps[-3].molecule.get_xyz_vectorial_properties('xyz_velocities'))*constants.Angstrom2Bohr/constants.fs2au
        velocity_final = np.array(self.molecular_trajectory.steps[-2].molecule.get_xyz_vectorial_properties('xyz_velocities'))*constants.Angstrom2Bohr/constants.fs2au

        if self.hopping_algorithm == 'FSSH':
            # Load NACs
            if istep > 1:
                nac_initial0 = np.array(self.molecular_trajectory.steps[-4].molecule.nacv)/constants.Angstrom2Bohr
            nac_initial = np.array(self.molecular_trajectory.steps[-3].molecule.nacv)/constants.Angstrom2Bohr
            nac_final = np.array(self.molecular_trajectory.steps[-2].molecule.nacv)/constants.Angstrom2Bohr
            for iState in range(self.nstates):
                for jState in range(iState):
                    # Phase alignment within trajectory based on the dot product of NAC(t) and NAC(t+dt)
                    if istep == 1:
                        if np.vdot(nac_initial[iState][jState], nac_final[iState][jState]) < 0:
                            self.molecular_trajectory.steps[-2].molecule.nacv[iState][jState] *= -1
                            self.molecular_trajectory.steps[-2].molecule.nacv[jState][iState] *= -1
                            nac_final[iState][jState] *= -1
                            nac_final[jState][iState] *= -1
                    # Phase alignment within trajectory based on the dot product of NAC(t-dt), NAC(t) and NAC(t+dt)
                    elif istep > 1:
                        h_extr = 2 * nac_initial[iState][jState] - nac_initial0[iState][jState]
                        if np.vdot(h_extr, nac_final[iState][jState])/(np.linalg.norm(h_extr[iState][jState])*np.linalg.norm(nac_final[iState][jState])) < 0:
                            self.molecular_trajectory.steps[-2].molecule.nacv[iState][jState] *= -1
                            self.molecular_trajectory.steps[-2].molecule.nacv[jState][iState] *= -1
                            nac_final[iState][jState] *= -1
                            nac_final[jState][iState] *= -1
        elif self.hopping_algorithm == 'TDBA':
            dt_initial = self.get_tdba_dt_coupling(istep, self.coupling_calc_threshold)
            dt_final = self.get_tdba_dt_coupling(istep, self.coupling_calc_threshold, next_step=True)

        if istep == 0:
            ss_E = [energies_interpolated[0]]
            ss_v = [velocity_initial]
            if self.hopping_algorithm == 'FSSH':
                ss_NAC = [nac_initial]
                ss_dt = [self.get_dt_coupling(velocity_initial, nac_initial)]
            elif self.hopping_algorithm == 'TDBA':
                ss_dt = [dt_initial]
            ss_ph = [np.zeros((self.nstates, self.nstates))]
            ss_c = [self.molecular_trajectory.steps[istep].state_coefficients]
            ss_cdot = [self.coeff_dot(ss_c[-1], ss_dt[-1], ss_ph[-1])]
        else:
            ss_E = self.molecular_trajectory.steps[-4].substep_potential_energy[-4:]
            ss_v = self.molecular_trajectory.steps[-4].substep_velocities[-4:]
            if self.hopping_algorithm == 'FSSH':
                ss_NAC = self.molecular_trajectory.steps[-4].substep_nonadiabatic_coupling_vectors[-4:]
            ss_dt = self.molecular_trajectory.steps[-4].substep_time_derivative_coupling[-4:]
            ss_c = self.molecular_trajectory.steps[-4].substep_state_coefficients[-4:]
            ss_cdot = self.molecular_trajectory.steps[-4].substep_state_coefficients_dot[-4:]
            ss_ph = self.molecular_trajectory.steps[-4].substep_phase[-4:]
        ss_random_numbers = []
        ss_hopping_probabilities = []

        hop_occured = False
        for iSubStep in range(n_substeps):
            # Interpolation
            ss_E.append(energies_interpolated[iSubStep+1])
            ss_v.append(velocity_initial + (iSubStep+1)/n_substeps * (velocity_final - velocity_initial))
            if self.hopping_algorithm == 'FSSH':
                ss_NAC.append(nac_initial + (iSubStep+1)/n_substeps * (nac_final - nac_initial))
                ss_dt.append(self.get_dt_coupling(ss_v[-1], ss_NAC[-1]))
            elif self.hopping_algorithm == 'TDBA':
                ss_dt.append(dt_initial + (iSubStep+1)/n_substeps * (dt_final - dt_initial))

            if (istep == 0 and iSubStep == 0):
                ss_ph.append(self.evolve_phase(ss_ph[0], ss_E[-2:]))
                ss_c.append(self.evolve_coeff_0(ss_c[0], ss_cdot[0], ss_dt[-1], ss_ph[-1]))
                ss_cdot.append(self.coeff_dot(ss_c[-1], ss_dt[-1], ss_ph[-1]))

                # Calculate hopping probabilities
                hopping_probabilities = []
                for stat in range(self.nstates):
                    if self.hopping_algorithm == 'FSSH':
                        if self.rescale_velocity_direction == 'momentum':
                            frustrated_hop = self.is_hop_frustrated(ss_v[-1], ss_v[-1], ss_E[-1], stat)
                        else: # 'nacv' and 'gradient difference' (not interpolated)
                            frustrated_hop = self.is_hop_frustrated(ss_NAC[-1][stat][self.current_state], ss_v[-1], ss_E[-1], stat)
                    elif self.hopping_algorithm == 'TDBA':
                        frustrated_hop = self.is_hop_frustrated(ss_v[-1], ss_v[-1], ss_E[-1], stat)
                    if stat == self.current_state:
                        prob = -1.0
                    elif stat > self.current_state and (self.prevent_back_hop or frustrated_hop):
                            prob = -1.0
                    else:
                        prob0 = -2*np.real(ss_c[0][self.current_state]*np.conj(ss_c[0][stat])*np.exp(0+1j*ss_ph[0][stat][self.current_state]))*ss_dt[0][stat][self.current_state]
                        prob1 = -2*np.real(ss_c[1][self.current_state]*np.conj(ss_c[1][stat])*np.exp(0+1j*ss_ph[1][stat][self.current_state]))*ss_dt[1][stat][self.current_state]
                        prob = (prob0+prob1)/abs(ss_c[1][self.current_state]**2)*self.time_step_tdse*constants.fs2au/2
                        if prob < 0:
                            prob = 0.0
                    hopping_probabilities.append(prob)
            else:
                ss_ph.append(self.evolve_phase(ss_ph[-1], ss_E[-3:]))
                if self.integrator == 'RK4':
                    ss_c.append(self.evolve_coeff_rk4(ss_c[-2], ss_cdot[-2], ss_dt[-2:], ss_ph[-2:]))
                elif self.integrator == 'AM5':
                    if (istep == 0) and iSubStep <= 2:
                        ss_c.append(self.evolve_coeff_rk4(ss_c[-2], ss_cdot[-2], ss_dt[-2:], ss_ph[-2:]))
                    else:
                        ss_c.append(self.evolve_coeff_am5(ss_c[-1], ss_cdot[-1], ss_cdot[-2], ss_cdot[-3], ss_cdot[-4], ss_dt[-1], ss_ph[-1]))
                elif self.integrator == 'Butcher':
                    if (istep == 0) and iSubStep <= 2:
                        ss_c.append(self.evolve_coeff_rk4(ss_c[-2], ss_cdot[-2], ss_dt[-2:], ss_ph[-2:]))
                    else:
                        ss_c.append(self.evolve_coeff_butcher(ss_c[-1], ss_c[-2], ss_cdot[-1], ss_cdot[-2], ss_dt[-1], ss_dt[-2], ss_ph[-1]))
                ss_cdot.append(self.coeff_dot(ss_c[-1], ss_dt[-1], ss_ph[-1]))

                # Calculate hopping probabilities
                if not hop_occured:
                    hopping_probabilities = []
                    for stat in range(self.nstates):
                        if self.hopping_algorithm == 'FSSH':
                            if self.rescale_velocity_direction == 'momentum':
                                frustrated_hop = self.is_hop_frustrated(ss_v[-1], ss_v[-1], ss_E[-1], stat)
                            else: # 'nacv' and 'gradient difference' (not interpolated)
                                frustrated_hop = self.is_hop_frustrated(ss_NAC[-1][stat][self.current_state], ss_v[-1], ss_E[-1], stat)
                        elif self.hopping_algorithm == 'TDBA':
                            frustrated_hop = self.is_hop_frustrated(ss_v[-1], ss_v[-1], ss_E[-1], stat)
                        if stat == self.current_state:
                            prob = -1.0
                        elif stat > self.current_state and (self.prevent_back_hop or frustrated_hop):
                                prob = -1.0
                        else:
                            prob0 = -2*np.real(ss_c[-1][self.current_state]*np.conj(ss_c[-1][stat])*np.exp(0+1j*ss_ph[-1][stat][self.current_state]))*ss_dt[-1][stat][self.current_state]
                            prob1 = -2*np.real(ss_c[-2][self.current_state]*np.conj(ss_c[-2][stat])*np.exp(0+1j*ss_ph[-2][stat][self.current_state]))*ss_dt[-2][stat][self.current_state]
                            prob2 = -2*np.real(ss_c[-3][self.current_state]*np.conj(ss_c[-3][stat])*np.exp(0+1j*ss_ph[-3][stat][self.current_state]))*ss_dt[-3][stat][self.current_state]                            
                            prob = self.time_step_tdse*constants.fs2au*(-prob2/12 + 5*prob0/12 + 2*prob1/3)/(abs(ss_c[-1][self.current_state]**2))
                            if prob < 0:
                                prob = 0.0
                        hopping_probabilities.append(prob)

            # Generate random number
            if not hop_occured:
                random_number = np.random.random()
                if max(hopping_probabilities) > random_number:
                    hop_occured = True

            ss_c[-1] = self.decoherence_correction(-2, ss_c[-1])
            
            ss_hopping_probabilities.append(hopping_probabilities)
            ss_random_numbers.append(random_number)

        self.molecular_trajectory.steps[-2].state_coefficients = ss_c[-1]

        # Saving substep info: last element of istep is the same as first element of istep+1
        if istep == 0:
            self.molecular_trajectory.steps[istep].substep_potential_energy = ss_E
            self.molecular_trajectory.steps[istep].substep_velocities = ss_v
            if self.hopping_algorithm == 'FSSH':
                self.molecular_trajectory.steps[istep].substep_nonadiabatic_coupling_vectors = ss_NAC
            self.molecular_trajectory.steps[istep].substep_time_derivative_coupling = ss_dt
            self.molecular_trajectory.steps[istep].substep_state_coefficients = ss_c
            self.molecular_trajectory.steps[istep].substep_state_coefficients_dot = ss_cdot
            self.molecular_trajectory.steps[istep].substep_phase = ss_ph
        else:
            self.molecular_trajectory.steps[-3].substep_potential_energy = ss_E[3:]
            self.molecular_trajectory.steps[-3].substep_velocities = ss_v[3:]
            if self.hopping_algorithm == 'FSSH':
                self.molecular_trajectory.steps[-3].substep_nonadiabatic_coupling_vectors = ss_NAC[3:]
            self.molecular_trajectory.steps[-3].substep_time_derivative_coupling = ss_dt[3:]
            self.molecular_trajectory.steps[-3].substep_state_coefficients = ss_c[3:]
            self.molecular_trajectory.steps[-3].substep_state_coefficients_dot = ss_cdot[3:]
            self.molecular_trajectory.steps[-3].substep_phase = ss_ph[3:]
        self.molecular_trajectory.steps[-3].substep_random_numbers = ss_random_numbers
        self.molecular_trajectory.steps[-3].substep_hopping_probabilities = ss_hopping_probabilities

        return hopping_probabilities, random_number

    def is_hop_frustrated(self, nac, velocity, energies, iState):
        a = 0.5 * np.sum(np.sum(nac * nac, axis=1) / (self.molecular_trajectory.steps[0].molecule.nuclear_masses * constants.amu2kg / constants.au2kg))
        b = np.sum(velocity * nac)
        c = energies[iState] - energies[self.current_state]
        if b**2 - 4 * a * c < 0:
            return True
        else:
            return False

    def interpolate_energy(self, energies, Nsteps):
        from scipy.interpolate import interp1d
        x = [i for i in range(len(energies))]
        x_interp = np.linspace(x[-2], x[-1], Nsteps+1)
        E_interp = []
        for iState in range(len(energies[0])):
            if len(energies) == 2:
                E = [energies[1][iState], energies[0][iState]]
                interp = interp1d(x, E, kind='linear')
            elif len(energies) == 3:
                E = [energies[2][iState], energies[1][iState], energies[0][iState]]
                interp = interp1d(x, E, kind='quadratic')
            elif len(energies) == 4:
                E = [energies[3][iState], energies[2][iState], energies[1][iState], energies[0][iState]]
                interp = interp1d(x, E, kind='cubic')
            E_interp.append(interp(x_interp))
        return E_interp

    def get_dt_coupling(self, velocity, nac):
        dt_coupling = np.zeros((self.nstates, self.nstates))
        for iState in range(self.nstates):
            for jState in range(iState):
                dt_coupling[iState][jState] = np.vdot(velocity, nac[iState][jState])
                dt_coupling[jState][iState] = -dt_coupling[iState][jState]
        return dt_coupling

    def get_tdba_dt_coupling(self, istep, coupling_calc_threshold=None, next_step=False):
        # Energy formula implementation, as in 10.12688/openreseurope.13624.2 and 10.1021/acs.jctc.1c01080
        dt_coupling = np.zeros((self.nstates, self.nstates))
        if istep < 2:
            return dt_coupling
        if coupling_calc_threshold is None:
            coupling_calc_threshold = 1000
        for iState in range(self.nstates):
            for jState in range(iState):
                if (self.molecular_trajectory.steps[-3].molecule.electronic_states[iState].energy-self.molecular_trajectory.steps[-3].molecule.electronic_states[jState].energy) * constants.Hartree2eV > coupling_calc_threshold:
                    dt_coupling[iState][jState] = 0
                    dt_coupling[jState][iState] = 0
                else:
                    if istep == 2:
                        gap_per_stat = []
                        for iistep in [-4, -3, -2]:
                            if next_step:
                                iistep += 1
                            gap_per_stat.append(self.molecular_trajectory.steps[iistep].molecule.electronic_states[iState].energy-self.molecular_trajectory.steps[iistep].molecule.electronic_states[jState].energy)
                        gap = gap_per_stat[-1]
                        gap_sotd = (gap - 2*gap_per_stat[-2] + gap_per_stat[-3]) / (self.time_step*constants.fs2au)**2
                    else:
                        gap_per_stat = []
                        for iistep in [-5, -4, -3, -2]:
                            if next_step:
                                iistep += 1
                            gap_per_stat.append(self.molecular_trajectory.steps[iistep].molecule.electronic_states[iState].energy-self.molecular_trajectory.steps[iistep].molecule.electronic_states[jState].energy)
                        gap = gap_per_stat[-1]
                        gap_sotd = (2*gap - 5*gap_per_stat[-2] + 4*gap_per_stat[-3] - gap_per_stat[-4]) / (self.time_step*constants.fs2au)**2
                    if gap_sotd / gap <= 0:
                        dt_coupling[iState][jState] = 0
                        dt_coupling[jState][iState] = 0
                    else:
                        dt_coupling[iState][jState] = np.sign(gap) * np.sqrt(gap_sotd / gap) / 2
                        dt_coupling[jState][iState] = -dt_coupling[iState][jState]
        return dt_coupling

    def coeff_dot(self, coeff, dt_coupling, phase):
        return - dt_coupling * np.exp(1j * phase) @ coeff

    def evolve_phase(self, phase, E):
        dt = self.time_step_tdse*constants.fs2au
        ph = np.zeros((self.nstates, self.nstates))
        for iState in range(self.nstates):
            for jState in range(self.nstates):
                if len(E) == 2:
                    ph[iState][jState] = phase[iState][jState] + 0.5*(E[1][iState] - E[1][jState] + E[0][iState] - E[0][jState])*dt
                else:
                    ph[iState][jState] = phase[iState][jState] + 5*(E[2][iState] - E[2][jState])*dt/12 + 2*(E[1][iState] - E[1][jState])*dt/3 - (E[0][iState] - E[0][jState])*dt/12
        return ph
    
    def evolve_coeff_0(self, c_o, cdot_o, dt_coupling, phase):
        dt = self.time_step_tdse*constants.fs2au
        c_temp = c_o + cdot_o * dt
        cdot_temp = self.coeff_dot(c_temp, dt_coupling, phase)
        c = c_o + 0.5*(cdot_temp + cdot_o)*dt
        return c

    def evolve_coeff_butcher(self, c_o, c_oo, cdot_o, cdot_oo, dt_coupling, dt_coupling_o, phase):
        """Butcher 5th order"""
        dt = self.time_step_tdse*constants.fs2au
        c_temp = c_oo + 0.125*(9*cdot_o + 3*cdot_oo)*dt
        cdot_temp = self.coeff_dot(c_temp, 0.5*(dt_coupling + dt_coupling_o), phase)
        c_temp2 = 0.2*(28*c_o - 23*c_oo) + (32*cdot_temp - 60*cdot_o - 26*cdot_oo)*dt/15
        cdot_temp2 = self.coeff_dot(c_temp2, dt_coupling, phase)
        c = (32*c_o - c_oo)/31 + (64*cdot_temp + 15*cdot_temp2 + 12*cdot_o - cdot_oo)*dt/93
        return c

    def evolve_coeff_am5(self, c_o, cdot_o, cdot_oo, cdot_ooo, cdot_oooo, dt_coupling, phase):
        """Adams Moulton 5th order"""
        dt = self.time_step_tdse*constants.fs2au
        c_temp = c_o + 0.0625*(-9*cdot_oooo + 37*cdot_ooo - 59*cdot_oo + 55*cdot_o)*dt
        cdot_temp = self.coeff_dot(c_temp, dt_coupling, phase)
        c = c_o + 0.0625*(cdot_ooo - 5*cdot_oo + 19*cdot_o + 9*cdot_temp)*dt/24
        return c

    def evolve_coeff_rk4(self, c_oo, cdot_oo, dt_coupling, phase):
        """Runge-Kutta 4th order"""
        dt = self.time_step_tdse*constants.fs2au
        c_temp = c_oo + cdot_oo * dt
        cdot_temp = self.coeff_dot(c_temp, dt_coupling[0], phase[0])
        cdot_temp_bk = cdot_temp.copy()
        c_temp = c_oo + cdot_temp * dt
        cdot_temp = self.coeff_dot(c_temp, dt_coupling[0], phase[0])
        c = c_oo + cdot_temp*2*dt
        cdot = self.coeff_dot(c, dt_coupling[1], phase[1])
        c = c_oo + (cdot + 2*(cdot_temp + cdot_temp_bk) + cdot_oo)*dt/3
        return c

    def decoherence_correction(self, istep, coeff):
        if self.decoherence_model == 'SDM':
            tau = np.zeros(self.nstates)
            coeff_sum = 0
            for iState in range(self.nstates):
                if iState != self.current_state:
                    dE = self.molecular_trajectory.steps[istep].molecule.electronic_states[iState].energy - self.molecular_trajectory.steps[istep].molecule.electronic_states[self.current_state].energy
                    tau[iState] = (1 + self.decoherence_SDM_decay / self.molecular_trajectory.steps[istep].molecule.kinetic_energy) / np.abs(dE)
                    coeff[iState] *= np.exp(-self.time_step_tdse*constants.fs2au / tau[iState])
                    coeff_sum += np.abs(coeff[iState])**2
            coeff[self.current_state] *= np.sqrt((1-coeff_sum)/(np.abs(coeff[self.current_state])**2))
        return coeff
    
    def change_properties_of_hopping_step(self):
        new_epot = self.molecular_trajectory.steps[-2].molecule.electronic_states[self.current_state].energy
        self.molecular_trajectory.steps[-2].molecule.energy = new_epot
        # for atom in self.molecular_trajectory.steps[step].molecule.atoms:
        #     atom.energy_gradients = atom.state_gradients[self.current_state]
        new_grad = self.molecular_trajectory.steps[-2].molecule.electronic_states[self.current_state].get_energy_gradients()
        self.molecular_trajectory.steps[-2].molecule.add_xyz_derivative_property(new_grad, 'energy', 'energy_gradients')
        #self.molecular_trajectory.steps[step].molecule.calculate_kinetic_energy()
        new_ekin = self.molecular_trajectory.steps[-2].molecule.kinetic_energy
        new_etot = new_epot + new_ekin
        self.molecular_trajectory.steps[-2].molecule.total_energy = new_etot
    
def analyze_trajs(trajectories=None, maximum_propagation_time=100.0):
    print('Start analyzing trajectories.') # debug

    traj_status_list = []
    for i in range(len(trajectories)):
        traj_status = {}
        try:
            if float(f"{trajectories[i].steps[-1].time:.6f}") == maximum_propagation_time:
                traj_status['status'] = 1
            else:
                traj_status['status'] = 0
        except:
            traj_status['status'] = 0
        if traj_status:
            try:
                final_time = float(f"{trajectories[i].steps[-1].time:.6f}")
                traj_status.update({"final time": final_time})
            except:
                traj_status.update({"final time": 0.0})
        traj_status_list.append(traj_status)

    print('%d trajectories ends normally.' % sum(1 for traj_status in traj_status_list if traj_status["status"] == 1))
    print('%d trajectories ends abnormally.' % sum(1 for traj_status in traj_status_list if traj_status["status"] == 0))
    for i in range(len(trajectories)):
        print("TRAJ%d ends %s at %.3f fs." % (i+1, ("normally" if traj_status_list[i]["status"] == 1 else "abnormally"), traj_status_list[i]["final time"]))

    print('Finish analyzing trajectories.') # debug

def analyze_trajs_from_disk(ntraj=1, max_propagation_time=100.0, dirname="job_surface_hopping_md_", traj_filename="traj.h5"):
    print('Start analyzing trajectories.') # debug
    traj_status_list = []
    for i in range(1,ntraj+1):
        print(i)
        traj_status = {}
        try:
            traj= data.molecular_trajectory()
            traj.load(dirname+str(i)+'/'+traj_filename, format="h5md")
            if float(f"{traj.steps[-1].time:.6f}") == max_propagation_time:
                traj_status['status'] = 1
            else:
                traj_status['status'] = 0
        except:
            traj_status['status'] = 0
        if traj_status:
            try:
                final_time = float(f"{traj.steps[-1].time:.6f}")
                traj_status.update({"final time": final_time})
            except:
                traj_status.update({"final time": 0.0})
        traj_status_list.append(traj_status)
    print('%d trajectories ends normally.' % sum(1 for traj_status in traj_status_list if traj_status["status"] == 1))
    print('%d trajectories ends abnormally.' % sum(1 for traj_status in traj_status_list if traj_status["status"] == 0))
    for i in range(ntraj):
        print("TRAJ%d ends %s at %.3f fs." % (i+1, ("normally" if traj_status_list[i]["status"] == 1 else "abnormally"), traj_status_list[i]["final time"]))
    print('Finish analyzing trajectories.') # debug

def plot_population(trajectories=None, time_step=0.1, max_propagation_time=100.0, nstates=3, filename='population.png', ref_pop_filename='ref_pop.txt', pop_filename='pop.txt'):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    time_list = list(np.arange(0.0, (max_propagation_time*100 + time_step*100)/100, time_step))
    pes_all_timestep = []
    population_all_timestep = []
    population_plot = []

    for i in range(len(time_list)):
        pes_per_timestep = []
        for j in range(len(trajectories)):
            try:
                pes_per_timestep.append(trajectories[j].steps[i].current_state+1)
            except:
                pes_per_timestep.append(None)
        Count_pes = Counter()
        for pes in pes_per_timestep:
            Count_pes[pes] += 1
        population_all_timestep.append([time_list[i]] + list(map(lambda x: Count_pes[x] / (len(pes_per_timestep) - pes_per_timestep.count(None))
                                                                    if (pes_per_timestep.count(None) != len(pes_per_timestep))
                                                                    else Count_pes[x] / len(pes_per_timestep), range(1, nstates + 1))))
        pes_all_timestep.append(pes_per_timestep)

    with open(pop_filename, "w") as file:
        for sublist in population_all_timestep:
            formatted_sublist = [f"{sublist[0]:.3f}"] + list(map(str, sublist[1:]))
            line = " ".join(formatted_sublist)
            file.write(line + "\n")

    for i in range(1, nstates + 1 + 1):
        population_plot.append(
            [population_all_timestep[j][i-1] for j in range(len(population_all_timestep))])

    if os.path.exists(ref_pop_filename):
        ref_population_all_timestep = []
        ref_population_plot = []

        with open('%s' % ref_pop_filename) as f_refpop:
            refpop_data = f_refpop.read().splitlines()
        for line in refpop_data:
            ref_population_all_timestep.append(
                list(map(float, line.split())))

        for i in range(1, nstates + 1 + 1):
            ref_population_plot.append(
                [ref_population_all_timestep[j][i-1] for j in range(len(ref_population_all_timestep))])

    plt.clf()

    plt.xlabel('Time (fs)')
    plt.ylabel('Population')

    plt.xlim([0, max_propagation_time])
    plt.ylim([0.0, 1.0])
    num_major_xticks = int(max_propagation_time / 10) + 1
    plt.xticks(np.linspace(0.0, max_propagation_time, num_major_xticks))
    num_major_yticks = int(1.0 / 0.25) + 1
    plt.yticks(np.linspace(0.0, 1.0, num_major_yticks))

    x = population_plot[0]
    if os.path.exists(ref_pop_filename):
        x_ref = ref_population_plot[0]

    for i in range(1, nstates + 1):
        y = population_plot[i]
        plt.plot(x, y, color=list(mcolors.TABLEAU_COLORS.keys())[
                    i-1], label='S%d' % (i-1))
        if os.path.exists(ref_pop_filename):
            y_ref = ref_population_plot[i]
            plt.plot(x_ref, y_ref, color=list(mcolors.TABLEAU_COLORS.keys())[
                    i-1], label='%s-S%d' % (ref_pop_filename,i-1), linestyle='dashed')
            
    plt.legend(loc='best', frameon=False, prop={'size': 10})

    plt.savefig(filename, bbox_inches='tight', dpi=300)

def plot_population_from_disk(time_step=0.1, max_propagation_time=100.0, nstates=3, filename='population.png', ref_pop_filename='ref_pop.txt', pop_filename='pop.txt', dirname="job_surface_hopping_md_", ntraj=1, traj_filename="traj.h5"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    time_list = np.arange(0, max_propagation_time+time_step, time_step)
    popArray = np.zeros((len(time_list),nstates))
    for i in range(1,ntraj+1):
        traj= data.molecular_trajectory()
        traj.load(dirname+str(i)+'/'+traj_filename, format="h5md")
        for idx, step in enumerate(traj.steps):
            popArray[idx][int(step.current_state)]+=1.0
    pop_norm = np.sum(popArray, axis=1)
    for i in range(len(time_list)):
        for j in range(nstates):
            popArray[i,j] = popArray[i,j]/pop_norm[i]

    if os.path.exists(ref_pop_filename):
        ref_population_all_timestep = []
        ref_population_plot = []

        with open('%s' % ref_pop_filename) as f_refpop:
            refpop_data = f_refpop.read().splitlines()
        for line in refpop_data:
            ref_population_all_timestep.append(
                list(map(float, line.split())))
        x_ref = ref_population_plot[0]
        for i in range(1, nstates + 1 + 1):
            ref_population_plot.append([ref_population_all_timestep[j][i-1] for j in range(len(ref_population_all_timestep))])    
    plt.clf()
    plt.xlabel('Time (fs)')
    plt.ylabel('Population')
    plt.xlim([0, max_propagation_time])
    plt.ylim([0.0, 1.0])
    
 
    for i in range(nstates):
        plt.plot(time_list, popArray[:,i], color=list(mcolors.TABLEAU_COLORS.keys())[i], label='S%d' % (i))
        if os.path.exists(ref_pop_filename):
            y_ref = ref_population_plot[i]
            plt.plot(x_ref, y_ref, color=list(mcolors.TABLEAU_COLORS.keys())[i-1], label='%s-S%d' % (ref_pop_filename,i-1), linestyle='dashed')
    
    plt.legend(loc='best', frameon=False, prop={'size': 10})
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    np.savetxt(pop_filename, popArray)

def plot_pes(traj, ax=None):
    """
    Plot potential energy surfaces (PESs) and current state along a trajectory.

    Parameters
    ----------
    traj : object
        Molecular trajectory object containing steps.
    ax : matplotlib.axes.Axes, optional
        Axis object used for plotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 3))
    nstates = len(traj.steps[0].molecule.electronic_states)
    colors = [plt.cm.tab20(i) for i in range(nstates)]
    time_step = traj.steps[1].time - traj.steps[0].time
    time = list(np.arange(traj.steps[0].time, len(traj.steps) * time_step, time_step))
    energies = []
    for iState in range(nstates):
        energies.append(np.array([istep.molecule.electronic_states[iState].energy for istep in traj.steps]))
    en_current = np.array([istep.molecule.electronic_states[int(istep.current_state)].energy for istep in traj.steps])
    en_kin = np.array([istep.molecule.kinetic_energy for istep in traj.steps]) * constants.Hartree2eV
    emin = min(energies[0])
    for iState in range(nstates):
        ax.plot(time, (energies[iState]-emin) * constants.Hartree2eV, c=colors[iState], label=f"$S_{iState}$")
    ax.scatter(time, (en_current-emin) * constants.Hartree2eV, c=en_kin, cmap=plt.cm.viridis, s=9, label="Current", zorder=2)
    ax.set_ylabel("Energy [eV]", fontsize=10)
    ax.set_xlim([time[0], time[-1]])
    ax.legend(ncols=nstates+1, fontsize=10)

def plot_nacs(traj, ax=None):
    """
    Plot the Frobenius norms of nonadiabatic coupling vectors (NACs) over time.

    Parameters
    ----------
    traj : object
        Molecular trajectory object containing steps.
    ax : matplotlib.axes.Axes, optional
        Axis object used for plotting.

    Notes
    -----
    - If predicted NACs are available in `traj.steps[i].molecule.nacv_predicted`,
      they are overlaid with hollow markers for comparison.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 3))
    nstates = len(traj.steps[0].molecule.electronic_states)
    colors = [plt.cm.tab20(i) for i in range(int(nstates*(nstates-1)/2))]
    nsteps = len(traj.steps)
    time_step = traj.steps[1].time - traj.steps[0].time
    time = list(np.arange(traj.steps[0].time, nsteps * time_step, time_step))
    norm_nacs = np.zeros((nsteps, nstates, nstates))
    norm_nacs_pred = np.zeros((nsteps, nstates, nstates))
    has_pred = hasattr(traj.steps[0].molecule, 'nacv_predicted')
    for istep, step in enumerate(traj.steps):
        for iState in range(nstates):
            for jState in range(iState):
                norm_nacs[istep, iState, jState] = np.linalg.norm(step.molecule.nacv[iState][jState], ord='fro')
                if has_pred:
                    norm_nacs_pred[istep, iState, jState] = np.linalg.norm(step.molecule.nacv_predicted[iState][jState], ord='fro')
    icolor = 0
    for iState in range(nstates):
        for jState in range(iState):
            ax.plot(time, norm_nacs[:, iState, jState], c=colors[icolor], label=f"$S_{iState}-S_{jState}$")
            if has_pred:
                ax.plot(time, norm_nacs_pred[:, iState, jState], 'o', mfc='none', c=colors[icolor], label=f"$S_{iState}-S_{jState}$")
            icolor += 1
    ax.set_ylabel("norm of NACs [1/Å]", fontsize=10)
    ax.set_xlim([time[0], time[-1]])
    ax.legend(fontsize=10)

def plot_tdc(traj, ax=None):
    """
    Plot the time-derivative couplings (TDCs) between each pair of electronic states over time.

    Parameters
    ----------
    traj : object
        Molecular trajectory object containing steps.
    ax : matplotlib.axes.Axes, optional
        Axis object used for plotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 3))
    nstates = len(traj.steps[0].molecule.electronic_states)
    colors = [plt.cm.tab20(i) for i in range(int(nstates*(nstates-1)/2))]
    nsteps = len(traj.steps)
    time_step = traj.steps[1].time - traj.steps[0].time
    time = np.arange(traj.steps[0].time, nsteps * time_step, time_step)
    tdcs = np.zeros((nsteps, nstates, nstates))
    for istep_idx, istep in enumerate(traj.steps):
        for iState in range(nstates):
            for jState in range(iState):
                tdcs[istep_idx, iState, jState] = istep.substep_time_derivative_coupling[0][iState][jState]
    icolor = 0
    for iState in range(nstates):
        for jState in range(iState):
            ax.plot(time, tdcs[:, iState, jState], c=colors[icolor], label=f"$S_{iState}-S_{jState}$")
            icolor += 1
    ax.set_ylabel("TDCs [a.u.]", fontsize=10)
    ax.set_xlim([time[0], time[-1]])
    ax.legend(ncols=icolor//5+1, fontsize=10)

def plot_pop(traj, ax=None):
    """
    Plot the populations of electronic states over time.

    Parameters
    ----------
    traj : object
        Molecular trajectory object containing steps.
    ax : matplotlib.axes.Axes, optional
        Axis object used for plotting.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 3))
    nstates = len(traj.steps[0].molecule.electronic_states)
    time_step = traj.steps[1].time - traj.steps[0].time
    time_substep = time_step / (len(traj.steps[0].substep_state_coefficients) - 1)
    colors = [plt.cm.tab20(i) for i in range(nstates)]
    for iState in range(nstates):
        times = []
        state_pop = []
        for istep, step in enumerate(traj.steps):
            if istep == len(traj.steps) - 1:
                continue
            nsub = len(step.substep_state_coefficients) - 1
            sub_times = istep * time_step + np.arange(nsub) * time_substep
            sub_pop = [abs(step.substep_state_coefficients[isub][iState])**2 for isub in range(nsub)]
            times.extend(sub_times)
            state_pop.extend(sub_pop)
        ax.plot(times, state_pop, c=colors[iState], label=f"$S_{iState}$")
    ax.set_xlabel("Time [fs]", fontsize=10)
    ax.set_ylabel("$|c|^2$", fontsize=10)
    ax.set_xlim([times[0], times[-1]])
    ax.legend(fontsize=10)

def plot_dist(traj, ax=None, geom_params=[[0, 1]], left_axis=False):
    """
    Plot degrees of freedom defined by 2(distance), 3(angle) or 4(dihedral angle) atoms over time.

    Parameters
    ----------
    traj : object
        Molecular trajectory object containing steps.
    ax : matplotlib.axes.Axes, optional
        Axis object used for plotting.
    geom_params : list of lists, optional
        List of degrees of freedom to plot. Each element is:
        - [i, j] for bond distances between atoms i and j
        - [i, j, k] for bond angles between atoms i-j-k
        - [i, j, k, l] for dihedral angles between atoms i-j-k-l.
    left_axis : bool, optional
        If True, shift the primary axis to the left and update the right y-axis position.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8, 3))
    time_step = traj.steps[1].time - traj.steps[0].time
    time = np.arange(traj.steps[0].time, len(traj.steps) * time_step, time_step)
    colors = [plt.cm.gist_ncar(i / len(geom_params)) for i in range(len(geom_params))]
    plot_types = []
    for d in geom_params:
        if len(d) == 2 and "r" not in plot_types: plot_types.append("r")
        elif len(d) == 3 and "a" not in plot_types: plot_types.append("a")
        elif len(d) == 4 and "d" not in plot_types: plot_types.append("d")
    axes = {plot_types[0]: ax}
    extra_axes_count = -1 if left_axis else 0
    for p in plot_types[1:]:
        extra_axes_count += 1
        ax_new = ax.twinx()
        ax_new.spines["right"].set_position(("outward", 45 * extra_axes_count))
        ax_new.tick_params(axis="y", labelsize=10)
        axes[p] = ax_new
    for i, d in enumerate(geom_params):
        if len(d) == 2: axes["r"].plot(time, [s.molecule.bond_length(*d) for s in traj.steps], c=colors[i], label=fr"$r({d[0]},{d[1]})$")
        elif len(d) == 3: axes["a"].plot(time, [s.molecule.bond_angle(*d, degrees=True) for s in traj.steps], c=colors[i], label=fr"$\alpha({d[0]},{d[1]},{d[2]})$")
        elif len(d) == 4: axes["d"].plot(time, [s.molecule.dihedral_angle(*d, degrees=True) for s in traj.steps], c=colors[i], label=fr"$\delta({d[0]},{d[1]},{d[2]},{d[3]})$")
    handles, labels = [], []
    for p in plot_types:
        h, l = axes[p].get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax.legend(handles, labels, ncols=len(geom_params), fontsize=10)
    ax.set_xlabel("Time [fs]", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim([time[0], time[-1]])
    if "r" in axes: axes["r"].set_ylabel(r"Distance $r$ [Å]", fontsize=10)
    if "a" in axes: axes["a"].set_ylabel(r"Angle $\alpha$ [°]", fontsize=10)
    if "d" in axes: axes["d"].set_ylabel(r"Dihedral angle $\delta$ [°]", fontsize=10)

def plot_trajs(trajectories=None, geom_params=[[0,1]], show_tdc=False, only_energy_params=False, filename=None):
    """
    Plot multiple aspects of molecular trajectories including PES, NACs/TDCs, populations, and degrees of freedom.

    Parameters
    ----------
    trajectories : list
        List of molecular trajectory objects.
    geom_params : list of lists, optional
        List of degrees of freedom to plot. Each element is:
        - [i, j] for bond distances between atoms i and j
        - [i, j, k] for bond angles between atoms i-j-k
        - [i, j, k, l] for dihedral angles between atoms i-j-k-l.
    show_tdc : bool, optional
        If True, plot TDCs instead of NACs when available.
    only_energy_params : bool, optional
        If True, adjust extra y-axis positions to match Landau-Zener surface hopping plotting style.
    filename : str, optional
        File path to save the figure. If None, the figure is not saved.
    """
    import matplotlib.pyplot as plt
    for itraj, traj in enumerate(trajectories):
        has_tdc = hasattr(traj.steps[0], 'substep_time_derivative_coupling')
        has_nacv = hasattr(traj.steps[0], 'substep_nonadiabatic_coupling_vectors')
        if has_tdc and not only_energy_params:
            fig = plt.figure(figsize=(12, 4))
            gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
            axs = gs.subplots()
            plot_pes(traj, axs[0][0])
            if has_nacv and not show_tdc:
                plot_type = 'nacs'
                plot_nacs(traj, axs[0][1])
            else:
                plot_type = 'tdba'
                plot_tdc(traj, axs[0][1])
            plot_pop(traj, axs[1][0])
            plot_dist(traj, axs[1][1], geom_params, left_axis=False)
            for ax in [axs[0][1], axs[1][1]]:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            for ax in [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]:
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, which='major', linestyle='--', color='lightgray', linewidth=0.8)
            for ax in [axs[0][0], axs[0][1]]:
                ax.set_xticklabels([])
            labels = [item.get_text() for item in axs[1][1].get_xticklabels()]
            labels[0] = ''
            axs[1][1].set_xticklabels(labels)
        else:
            plot_type = 'lzsh'
            fig = plt.figure(figsize=(6, 4))
            gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
            axs = gs.subplots()
            plot_pes(traj, axs[0])
            plot_dist(traj, axs[1], geom_params, left_axis=True)
            for ax in axs:
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, which='major', linestyle='--', color='lightgray', linewidth=0.8)
            axs[0].set_xticklabels([])
        plt.tight_layout()
        if filename:
            plt.savefig(f'{filename}_{plot_type}_{itraj}', bbox_inches='tight', dpi=1200)
        plt.show()

def internal_consistency_check(trajectories=None, filename=None):
    """
    Compare classical state occupations with average adiabatic populations to check internal consistency.

    Parameters
    ----------
    trajectories : list
        List of molecular trajectory objects.
    filename : bool, optional
        File path to save the figure. If None, the figure is not saved.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    nstates = len(trajectories[0].steps[0].molecule.electronic_states)
    colors = [plt.cm.tab20(i) for i in range(nstates)]
    time_step = trajectories[0].steps[1].time - trajectories[0].steps[0].time
    Ntrajs = []
    pop_cs = []
    pop_qm = []
    for traj in trajectories:
        for istep, step in enumerate(traj.steps):
            if istep == len(Ntrajs):
                Ntrajs.append(0)
                pop_cs.append(np.zeros(nstates))
                pop_qm.append(np.zeros(nstates))
            Ntrajs[istep] += 1
            pop_cs[istep][int(step.current_state)] += 1
            pop_qm[istep] += np.abs(step.substep_state_coefficients[0])**2
    time = np.arange(trajectories[0].steps[0].time, len(Ntrajs) * time_step, time_step)
    fig, ax = plt.subplots(1, figsize=(8, 5))
    for iState in range(nstates):
        ax.plot(time, np.array(pop_cs)[:, iState] / np.array(Ntrajs), c=colors[iState], lw=2, label=rf"Occupation S$_{iState}$")
        ax.plot(time, np.array(pop_qm)[:, iState] / np.array(Ntrajs), ':', c=colors[iState], lw=2, label=rf"Adiabatic S$_{iState}$")
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel("Population")
    ax.set_xlim([time[0], time[-1]])
    ax.grid(linestyle='--', dashes=(5, 9), linewidth=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=1200)
    plt.show()