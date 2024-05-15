#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! md: Module for molecular dynamics                                         ! 
  ! Implementations by: Yi-Fan Hou & Pavlo O. Dral                            ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
from . import data
from . import constants
from .thermostat import Andersen_thermostat, Nose_Hoover_thermostat
from . import stopper

class md():
    '''
    Molecular dynamics

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

    For theoretical details, see and cite original `paper <https://doi.org/10.1039/D3CP03515H>`__.

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
                 model_predict_kwargs={},
                 molecule_with_initial_conditions=None,
                 molecule=None,
                 ensemble='NVE',
                 thermostat=None,
                 time_step=0.1,
                 maximum_propagation_time=1000,
                 dump_trajectory_interval=None,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None):
        self.model = model
        self.model_predict_kwargs ={'calculate_energy':True, 'calculate_energy_gradients':True}
        self.model_predict_kwargs.update(model_predict_kwargs)
        if not molecule_with_initial_conditions is None and not molecule is None:
            stopper.stopMLatom('molecule and molecule_with_initial_conditions cannot be used at the same time')
        if not molecule_with_initial_conditions is None:
            self.molecule_with_initial_conditions = molecule_with_initial_conditions 
        if not molecule is None:
            self.molecule_with_initial_conditions = molecule
        self.ensemble = ensemble
        if thermostat != None:
            self.thermostat = thermostat
        self.time_step = time_step
        self.maximum_propagation_time = maximum_propagation_time

        self.Natoms = len(self.molecule_with_initial_conditions.atoms)
        self.masses = self.molecule_with_initial_conditions.get_nuclear_masses()
        self.mass = self.masses.reshape(self.Natoms,1)
        
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

        self.linearity = self.molecule_with_initial_conditions.is_it_linear()
        if self.linearity:
            self.degrees_of_freedom = 3 * self.Natoms - 5
        else:
            self.degrees_of_freedom = 3 * self.Natoms - 6
        self.propagate()
        
    def propagate(self):
        self.molecular_trajectory = data.molecular_trajectory()
        temp_traj = data.molecular_trajectory()

        istep = 0
        stop = False
        while not stop:
            trajectory_step = data.molecular_trajectory_step()
            if istep == 0:
                molecule = self.molecule_with_initial_conditions.copy()
                if not 'energy_gradients' in molecule.atoms[0].__dict__:
                    self.model.predict(molecule=molecule,
                                    **self.model_predict_kwargs)
                forces = -np.copy(molecule.get_energy_gradients())
                acceleration = forces / self.mass / constants.ram2au * (constants.Bohr2Angstrom**2) * constants.fs2au**2 #/ MLenergyUnits
                pass 
            else:
                previous_molecule = molecule
                molecule = self.molecule_with_initial_conditions.copy()
                molecule.xyz_coordinates=previous_molecule.xyz_coordinates
                velocity = previous_molecule.get_xyz_vectorial_properties('xyz_velocities')
                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_velocities = np.copy(velocity[iatom])

                # ensemble and/or thermostat
                self.propagation_algorithm.update_velocities_first_half_step(molecule = molecule,
                                                                                       time_step = self.time_step)

                coord = np.copy(molecule.xyz_coordinates)
                velocity = np.copy(molecule.get_xyz_vectorial_properties('xyz_velocities'))

                # Coordinate update 
                coord = coord + velocity*self.time_step+acceleration*self.time_step**2 * 0.5

                # Velocity update half step 
                velocity = velocity + acceleration * self.time_step * 0.5

                # Calculate forces
                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_coordinates = np.copy(coord[iatom])
                self.model.predict(molecule=molecule,
                                   **self.model_predict_kwargs)
                forces = -np.copy(molecule.get_energy_gradients())
                acceleration = forces / self.mass / constants.ram2au * (constants.Bohr2Angstrom**2) * constants.fs2au**2 #/ MLenergyUnits
                
                # Velocity update half step
                velocity = velocity + acceleration*self.time_step*0.5

                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_coordinates = np.copy(coord[iatom])
                    molecule.atoms[iatom].xyz_velocities = np.copy(velocity[iatom])

                self.propagation_algorithm.update_velocities_second_half_step(molecule = molecule,
                                                                                        time_step = self.time_step)
            velocity = np.copy(molecule.get_xyz_vectorial_properties('xyz_velocities'))
            
            molecule.total_energy = molecule.energy + molecule.kinetic_energy
            molecule.temperature = molecule.kinetic_energy / (constants.kB_in_Hartree*self.degrees_of_freedom/2)
            trajectory_step.step = istep 
            trajectory_step.time = istep * self.time_step
            trajectory_step.molecule = molecule 
            self.molecular_trajectory.steps.append(trajectory_step)
            # Stop function
            if type(self.stop_function) != type(None):
                if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                stop = self.stop_function(molecule, **self.stop_function_kwargs)
            if istep*self.time_step >= self.maximum_propagation_time:
                stop = True
            # Dump trajectory at some interval
            if self.dump_trajectory_interval != None:
                
                if istep % self.dump_trajectory_interval == 0:
                    if self.format == 'h5md':
                        temp_traj = data.molecular_trajectory()
                        temp_traj.steps.append(trajectory_step)
                    elif self.format == 'json':
                        temp_traj.steps.append(trajectory_step)
                    temp_traj.dump(filename=self.filename, format=self.format)

            istep += 1

    def dump_md_checkpoint(self,filename='md_chk'):
        with open(filename,'w') as f:
            mol = self.molecular_trajectory.steps[-1].molecule
            f.write('xyz_coordinates\n')
            for iatom in mol.atoms:
                f.write('%-3s %25.13f %25.13f %25.13f\n'%(iatom.element_symbol,iatom.xyz_coordinates[0],iatom.xyz_coordinates[1],iatom.xyz_coordinates[2]))
            f.write('xyz_velocities\n')
            for iatom in mol.atoms:
                f.write('%25.13f %25.13f %25.13f\n'%(iatom.xyz_velocities[0],iatom.xyz_velocities[1],iatom.xyz_velocities[2]))
            f.write('ensemble=%s\n'%self.ensemble.upper())
            if 'thermostat' in self.__dict__.keys():
                f.write('thermostat=%s\n'%type(self.thermostat))
                for key in self.thermostat.__dict__.keys():
                    f.write('%s=%s\n'%(key,str(self.thermostat.__dict__[key])))

class nve():
    def __init__(self):
        pass 

    def update_velocities_first_half_step(self,**kwargs):
        if 'molecule' in kwargs:
            molecule = kwargs['molecule']
        return molecule

    def update_velocities_second_half_step(self,**kwargs):
        if 'molecule' in kwargs:
            molecule = kwargs['molecule']
        return molecule

