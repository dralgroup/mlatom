#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! MD: Module for molecular dynamics                                         ! 
  ! Implementations by: Yi-Fan Hou & Pavlo O. Dral                            ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
from . import data
from . import constants
from .thermostat import Andersen_thermostat, Nose_Hoover_thermostat
from . import stopper
import time

class md_parallel():
    '''
    MD object

    Initialize and propagate MD

    Arguments:
        model (:class:`mlatom.models.model` or :class:`mlatom.models.methods`): Any model or method which provides energies and forces.
        molecule_with_initial_conditions (:class:`data.molecule`): The molecule with initial conditions.
        ensemble (str, optional): Which kind of ensemble to use.
        thermostat (:class:`thermostat.Thermostat`): The thermostat applied to the system.
        time_step (float): Time step in femtoseconds.
        maximum_propagation_time (float): Maximum propagation time in femtoseconds.
        dump_trajectory_interval (int, optional): Dump trajectory at which interval. Set to ``None`` to disable dumping.
        filename (str, optional): The file that saves the dumped trajectory
        format (str, optional): Format in which the dumped trajectory is saved
        stop_function (any, optional): User-defined function that stops MD before ``maximum_propagation_time``
        stop_function_kwargs (Dict, optional): Kwargs of ``stop_function``

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

    Examples:

    .. code-block:: python

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Initialize methods
        aiqm1 = ml.models.methods(method='AIQM1')
        # User-defined initial condition
        init_cond_db = ml.generate_initial_conditions(molecule = mol,
                                                      generation_method = 'user-defined',
                                                      file_with_initial_xyz_coordinates = 'ethanol.xyz',
                                                      file_with_initial_xyz_velocities  = 'ethanol.vxyz')
        init_mol = init_cond_db.molecules[0]
        # Initialize thermostat
        nose_hoover = ml.md.Nose_Hoover_thermostat(temperature=300,molecule=init_mol,degrees_of_freedom=-6)
        # Run dynamics
        dyn = ml.md(model=aiqm1,
                    molecule_with_initial_conditions = init_mol,
                    ensemble='NVT',
                    thermostat=nose_hoover,
                    time_step=0.5,
                    maximum_propagation_time = 10.0)
        # Dump trajectory
        traj = dyn.molecular_trajectory
        traj.dump(filename='traj', format='plain_text')
        traj.dump(filename='traj.h5', format='h5md')
        

    .. note::

        Trajectory is saved in ``ml.md.molecular_trajectory``, which is a :class:`ml.data.molecular_trajectory` class

    .. warning:: 

        In MLatom, energy unit is Hartree and distance unit is Angstrom. Make sure that the units in your model are consistent.
        
    '''
    Andersen_thermostat = Andersen_thermostat
    Nose_Hoover_thermostat = Nose_Hoover_thermostat
    def __init__(self, model=None,
                 molecular_database=None,
                 molecule_with_initial_conditions=None,
                 molecule=None,
                 ensemble='NVE',
                 thermostat=None,
                 time_step=0.1,
                 maximum_propagation_time=1000,
                 excess_energy=0,
                 dump_trajectory_interval=None,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None):
        self.model = model
        if not molecule_with_initial_conditions is None and not molecule is None:
            stopper.stopMLatom('molecule and molecule_with_initial_conditions cannot be used at the same time')
        if not molecule_with_initial_conditions is None:
            molecular_database = molecule_with_initial_conditions
        if not molecule is None:
            molecular_database = molecule
        self.molecular_database = data.molecular_database(molecular_database)
        self.ensemble = ensemble
        if thermostat != None:
            self.thermostat = thermostat
        self.time_step = time_step
        self.maximum_propagation_time = maximum_propagation_time

        if excess_energy == 0:
            self.excess_energy = [0]*len(self.molecular_database)
        else:
            self.excess_energy = excess_energy
        self.excess_energy_per_step = [each/(self.maximum_propagation_time/self.time_step) for each in self.excess_energy]

        self.Natoms = self.molecular_database.number_of_atoms
        self.masses = self.molecular_database.nuclear_masses
        # self.mass = self.masses.reshape(self.Natoms,1)
        
        if self.ensemble.upper() == 'NVE':
            self.propagation_algorithm = nve()
        elif self.ensemble.upper() == 'NVT':
            self.propagation_algorithm = self.thermostat
        
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
        self.propagate()
        
    def propagate(self):
        self.molecular_trajectory = data.molecular_trajectory()
        self.molecular_trajectory.traj_len = {}
        ntraj = len(self.molecular_database)
        stop_array = np.array([False for ii in range(ntraj)])
        
        istep = 0
        stop = False
        while not stop:
            # trajectory_step = data.molecular_trajectory_step()
            if istep == 0:
                
                molecular_database = self.molecular_database.copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=[])
                for ii in range(ntraj):
                    molecular_database[ii].index = ii
                self.model.predict(molecular_database=molecular_database,
                                calculate_energy = True,
                                calculate_energy_gradients = True)
                coord = molecular_database.xyz_coordinates
                forces = -molecular_database.get_xyz_vectorial_properties('energy_gradients')
                acceleration = forces / self.masses[...,np.newaxis] / constants.ram2au * (constants.Bohr2Angstrom**2) * constants.fs2au**2 #/ MLenergyUnits
            else:
                previous_molecular_database = molecular_database
                molecular_database = data.molecular_database()
                for ii in range(len(previous_molecular_database)):
                    if not stop_array[previous_molecular_database[ii].index]:
                        mol = previous_molecular_database[ii].copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=['index'])
                        molecular_database.append(mol)

                # ensemble and/or thermostat
                self.propagation_algorithm.update_velocities_first_half_step(molecular_database = molecular_database,
                                                                                       time_step = self.time_step)

                # Coordinate update 
                coord = coord + velocity*self.time_step+acceleration*self.time_step**2 * 0.5

                # Velocity update half step 
                velocity = velocity + acceleration * self.time_step * 0.5

                # Calculate forces
                molecular_database.xyz_coordinates = coord
                self.model.predict(molecular_database=molecular_database,
                                   calculate_energy = True,
                                   calculate_energy_gradients = True)
                forces = -molecular_database.get_xyz_vectorial_properties('energy_gradients')
                acceleration = forces / self.masses[...,np.newaxis] / constants.ram2au * (constants.Bohr2Angstrom**2) * constants.fs2au**2 #/ MLenergyUnits

                # Velocity update half step
                velocity = velocity + acceleration*self.time_step*0.5
                
                molecular_database.xyz_coordinates = coord
                molecular_database.add_xyz_vectorial_properties(velocity,'xyz_velocities')

                # thermostat
                self.propagation_algorithm.update_velocities_second_half_step(molecular_database = molecular_database,
                                                                                        time_step = self.time_step)
            # Rescale velocities according to the excess energy           
            icount = 0
            for ii in range(ntraj):
                if not stop_array[ii]:
                    self.rescale_velocity(molecular_database[icount],molecular_database[icount].kinetic_energy+self.excess_energy_per_step[ii])
                    icount += 1
            
            velocity = molecular_database.get_xyz_vectorial_properties('xyz_velocities')
            
            molecular_database.add_scalar_properties(molecular_database.get_properties('energy') + molecular_database.get_properties('kinetic_energy'), 'total_energy')

            molecular_database_saved = data.molecular_database()
            icount = 0
            for ii in range(ntraj):
                if stop_array[ii]:
                    molecular_database_saved.molecules.append(None)
                else:
                    molecular_database_saved.append(molecular_database[icount])
                    icount += 1

            self.molecular_trajectory.steps.append(molecular_database_saved)
            # Stop function
            if type(self.stop_function) != type(None):
                index_array = []
                if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                for ii in range(len(molecular_database)):
                    stop_ii = self.stop_function(molecular_database[ii], **self.stop_function_kwargs)
                    stop_array[molecular_database[ii].index] = stop_ii
                    if not stop_ii:
                        index_array.append(ii)
                    else:
                        self.molecular_trajectory.traj_len[molecular_database[ii].index] = istep
                acceleration = acceleration[index_array]
                coord = coord[index_array]
                velocity = velocity[index_array]
                self.masses = self.masses[index_array]



            if istep*self.time_step >= self.maximum_propagation_time or stop_array.all():
                # print("???")
                stop = True

                for ii in range(ntraj):
                    if not stop_array[ii]:
                        self.molecular_trajectory.traj_len[ii] = istep

            istep += 1

    def rescale_velocity(self,mol,target_ekin):
        rescale_factor = np.sqrt(target_ekin / mol.kinetic_energy)
        for each in mol.atoms:
            each.xyz_velocities *= rescale_factor

class nve():
    def __init__(self):
        pass 

    def update_velocities_first_half_step(self,**kwargs):
        if 'molecular_database' in kwargs:
            molecular_database = kwargs['molecular_database']
        return molecular_database

    def update_velocities_second_half_step(self,**kwargs):
        if 'molecular_database' in kwargs:
            molecular_database = kwargs['molecular_database']
        return molecular_database

