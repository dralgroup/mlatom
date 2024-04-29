#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! namd: Module for nonadiabatic molecular dynamics                          ! 
  ! Implementations by: Lina Zhang & Pavlo O. Dral                            ! 
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
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces.
        model_predict_kwargs (Dict, optional): Kwargs of model prediction
        molecule_with_initial_conditions (:class:`data.molecule`): The molecule with initial conditions.
        molecule (:class:`data.molecule`): Work the same as molecule_with_initial_conditions
        ensemble (str, optional): Which kind of ensemble to use.
        thermostat (:class:`thermostat.Thermostat`): The thermostat applied to the system.
        time_step (float): Time step in femtoseconds.
        maximum_propagation_time (float): Maximum propagation time in femtoseconds.
        dump_trajectory_interval (int, optional): Dump trajectory at which interval. Set to ``None`` to disable dumping.
        filename (str, optional): The file that saves the dumped trajectory
        format (str, optional): Format in which the dumped trajectory is saved
        stop_function (any, optional): User-defined function that stops MD before ``maximum_propagation_time``
        stop_function_kwargs (Dict, optional): Kwargs of ``stop_function``
        hopping_algorithm (str, optional): Surface hopping algorithm
        nstates (int): Number of states 
        initial_state (int): Initial state
        random_seed (int): Random seed 
        prevent_back_hop (bool, optional): Whether to prevent back hopping
        rescale_velocity_direction (string, optional): Rescale velocity direction 
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

    For theoretical details, see and cite original paper (to be submitted).
    
    * Lina Zhang, Sebastian Pios, Mikołaj Martyka, Fuchun Ge, Yi-Fan Hou, Yuxinxin Chen, Joanna Jankowska, Lipeng Chen, Mario Barbatti, `Pavlo O. Dral <http://dr-dral.com>`__. MLatom software ecosystem for surface hopping dynamics in Python with quantum mechanical and machine learning methods. **2024**, *to be submitted*. Preprint on *arXiv*: https://arxiv.org/abs/2404.06189.

    Examples:

    .. code-block:: python
    
        # Propagate multiple LZBL surface-hopping trajectories in parallel
        # .. setup dynamics calculations
        namd_kwargs = {
                    'model': aiqm1,
                    'time_step': 0.25,
                    'maximum_propagation_time': 5,
                    'hopping_algorithm': 'LZBL',
                    'nstates': 3,
                    'initial_state': 2,
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
                 maximum_propagation_time=100,
                 dump_trajectory_interval=None,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None,
                 hopping_algorithm='LZBL',
                 nstates=None, initial_state=None,
                 random_seed=generate_random_seed, 
                 prevent_back_hop=False, 
                 rescale_velocity_direction='along velocities',
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
        
        self.dump_trajectory_interval = dump_trajectory_interval
        if dump_trajectory_interval != None:
            self.format = format
            if format == 'h5md': ext = '.h5'
            elif format == 'json': ext = '.json'
            if filename == None:
                import uuid
                filename = str(uuid.uuid4()) + ext
            self.filename = filename 
        
        self.stop_function = stop_function
        self.stop_function_kwargs = stop_function_kwargs
        
        self.hopping_algorithm=hopping_algorithm
        self.nstates=nstates
        self.initial_state = initial_state
        if self.initial_state is None:
            if 'current_state' in self.molecule_with_initial_conditions.__dict__:
                self.initial_state = self.molecule_with_initial_conditions.current_state
            else:
                self.initial_state = self.nstates - 1
        self.random_seed = random_seed
        self.prevent_back_hop = prevent_back_hop
        self.rescale_velocity_direction = rescale_velocity_direction
        self.reduce_kinetic_energy = reduce_kinetic_energy
        if self.reduce_kinetic_energy:
            if self.molecule_with_initial_conditions.is_it_linear():
                self.degrees_of_freedom = 3 * len(self.molecule_with_initial_conditions.atoms) - 5
            else:
                self.degrees_of_freedom = 3 * len(self.molecule_with_initial_conditions.atoms) - 6
            self.reduce_kinetic_energy_factor = self.degrees_of_freedom
        else:
            self.reduce_kinetic_energy_factor = 1
        
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
            calculate_energy_gradients = [True] * self.nstates
            self.model_predict_kwargs={'nstates':self.nstates, 
                                       'current_state':self.current_state,
                                       'calculate_energy':True,
                                       'calculate_energy_gradients':calculate_energy_gradients}
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
                self.model_predict_kwargs['calculate_energy_gradients'] = [False] * self.nstates
                self.model_predict_kwargs['calculate_energy_gradients'][self.current_state] = True

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

            random_number = np.random.random()
            self.molecular_trajectory.steps[istep+1].random_number = random_number
            if istep == 0:
                self.molecular_trajectory.steps[istep].current_state = self.current_state
            # fssh/lzsh/znsh: prob list
            if self.hopping_algorithm == 'LZBL':
                hopping_probabilities = self.lzsh(istep=istep)
            self.molecular_trajectory.steps[istep+1].hopping_probabilities = hopping_probabilities
            max_prob = max(hopping_probabilities)
            if max_prob > random_number:
                max_prob_stat = hopping_probabilities.index(max_prob)
                self.initial_state = self.current_state
                self.current_state = max_prob_stat
                # fssh/lzsh/znsh: rescale_velocity; change en grad in molecular_trajectory; change ekin etot
                # hopping_gap = (self.molecular_trajectory.steps[istep+1].molecule.state_energies[self.current_state]
                #                 -self.molecular_trajectory.steps[istep+1].molecule.state_energies[self.initial_state])
                hopping_gap = (self.molecular_trajectory.steps[istep+1].molecule.electronic_states[self.current_state].energy
                               -self.molecular_trajectory.steps[istep+1].molecule.electronic_states[self.initial_state].energy)
                if self.rescale_velocity_direction == 'along velocities':
                    self.molecular_trajectory.steps[istep+1].molecule.rescale_velocities(kinetic_energy_change=-hopping_gap)
                self.change_properties_of_hopping_step(step=istep+1) 
                if self.hopping_algorithm == 'LZBL':
                    del self.molecular_trajectory.steps[-1]
                    one_step_propagation = True
            elif self.hopping_algorithm == 'LZBL':
                one_step_propagation = False
            self.molecular_trajectory.steps[istep+1].current_state = self.current_state

            if type(self.stop_function) != type(None):
                if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                if 'stop_check' not in locals():
                    stop_check = False
                stop, stop_check = self.stop_function(stop_check=stop_check,
                                                      mol=self.molecular_trajectory.steps[istep+1].molecule, 
                                                      current_state=self.current_state, 
                                                      **self.stop_function_kwargs)
                if stop:
                    del self.molecular_trajectory.steps[-1]

            # Dump trajectory at some interval
            if self.dump_trajectory_interval != None:
                
                if istep % self.dump_trajectory_interval == 0:
                    if self.format == 'h5md':
                        temp_traj = data.molecular_trajectory()
                        temp_traj.steps.append(self.molecular_trajectory.steps[-1])
                    elif self.format == 'json':
                        temp_traj = self.molecular_trajectory
                    temp_traj.dump(filename=self.filename, format=self.format)

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
                for iistep in [istep, istep+1, istep+2]:
                    gap_per_stat.append(abs(self.molecular_trajectory.steps[iistep].molecule.electronic_states[self.current_state].energy
                                        -self.molecular_trajectory.steps[iistep].molecule.electronic_states[stat].energy))
                if (gap_per_stat[0] > gap_per_stat[1]) and (gap_per_stat[2] > gap_per_stat[1]):
                    if not self.prevent_back_hop:
                        #if (stat > self.current_state) and (self.molecular_trajectory.steps[istep+1].molecule.kinetic_energy < gap_per_stat[1]):
                        if ((stat > self.current_state) and 
                        ((self.molecular_trajectory.steps[istep+1].molecule.kinetic_energy/(self.reduce_kinetic_energy_factor)) 
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

    def change_properties_of_hopping_step(self, step):
        new_epot = self.molecular_trajectory.steps[step].molecule.electronic_states[self.current_state].energy
        self.molecular_trajectory.steps[step].molecule.energy = new_epot
        # for atom in self.molecular_trajectory.steps[step].molecule.atoms:
        #     atom.energy_gradients = atom.state_gradients[self.current_state]
        new_grad = self.molecular_trajectory.steps[step].molecule.electronic_states[self.current_state].get_energy_gradients()
        self.molecular_trajectory.steps[step].molecule.add_xyz_derivative_property(new_grad, 'energy', 'energy_gradients')
        #self.molecular_trajectory.steps[step].molecule.calculate_kinetic_energy()
        new_ekin = self.molecular_trajectory.steps[step].molecule.kinetic_energy
        new_etot = new_epot + new_ekin
        self.molecular_trajectory.steps[step].molecule.total_energy = new_etot
    
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
