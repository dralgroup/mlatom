#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! MD: Module for molecular dynamics                                         ! 
  ! Implementations by: Yi-Fan Hou & Pavlo O. Dral                            ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
try:
    from . import data
    from . import constants
    from .thermostat import Nose_Hoover_thermostat
except:
    import data
    import constants
    from thermostat import Nose_Hoover_thermostat

class md():
    Nose_Hoover_thermostat = Nose_Hoover_thermostat
    def __init__(self, model=None,
                 molecule_with_initial_conditions=None,
                 ensemble='NVE',
                 thermostat=None,
                 time_step=0.1,
                 maximum_propagation_time=1000,
                 dump_trajectory_every_time_step=False,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None):
        self.model = model
        self.molecule_with_initial_conditions = molecule_with_initial_conditions
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
        
        self.dump_trajectory_every_time_step = dump_trajectory_every_time_step
        if dump_trajectory_every_time_step:
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

        istep = 0
        stop = False
        while not stop:
            trajectory_step = data.molecular_trajectory_step()
            if istep == 0:
                molecule = self.molecule_with_initial_conditions.copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=[])
                self.model.predict(molecule=molecule,
                                   calculate_energy = True,
                                   calculate_energy_gradients = True)
                forces = -np.copy(molecule.get_energy_gradients())
                acceleration = forces / self.mass / 1822.888515 * (constants.Bohr2Angstrom**2) * (100.0/2.4188432)**2 #/ MLenergyUnits
                # energy, forces = 
                # acceleration = forces / atom_mass / 1822.888515 * (MLdistanceUnits**2) * (100.0/2.4188432)**2 / MLenergyUnits
                # kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0
                pass 
            else:
                previous_molecule = molecule
                molecule = self.molecule_with_initial_conditions.copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=[])
                molecule.update_xyz_coordinates(xyz_coordinates=previous_molecule.get_xyz_coordinates())
                velocity = previous_molecule.get_xyz_vectorial_properties('xyz_velocities')
                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_velocities = np.copy(velocity[iatom])

                # ensemble and/or thermostat
                #kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0 # Raw kinetic energy
                #KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,args.Nc,YSlist,NHC_Q,Natoms,args.temp,MLdistanceUnits,DOF)
                # molecule = self.propagation_algorithm.update_velocities_first_half_step(molecule = molecule,
                #                                                                        time_step = self.time_step)
                self.propagation_algorithm.update_velocities_first_half_step(molecule = molecule,
                                                                                       time_step = self.time_step)

                coord = np.copy(molecule.get_xyz_coordinates())
                velocity = np.copy(molecule.get_xyz_vectorial_properties('xyz_velocities'))

                # Coordinate update 
                coord = coord + velocity*self.time_step+acceleration*self.time_step**2 * 0.5

                # Velocity update half step 
                velocity = velocity + acceleration * self.time_step * 0.5

                # Calculate forces
                #print(molecule.get_xyz_coordinates())
                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_coordinates = np.copy(coord[iatom])
                #print(molecule.get_xyz_coordinates())
                self.model.predict(molecule=molecule,
                                   calculate_energy = True,
                                   calculate_energy_gradients = True)
                forces = -np.copy(molecule.get_energy_gradients())
                acceleration = forces / self.mass / 1822.888515 * (constants.Bohr2Angstrom**2) * (100.0/2.4188432)**2 #/ MLenergyUnits
                
                # Velocity update half step
                velocity = velocity + acceleration*self.time_step*0.5

                for iatom in range(self.Natoms):
                    molecule.atoms[iatom].xyz_coordinates = np.copy(coord[iatom])
                    molecule.atoms[iatom].xyz_velocities = np.copy(velocity[iatom])

                # thermostat
                # kin_en_raw = np.sum(velocity**2 * self.mass) / 2.0 # Raw kinetic energy (Unit: relative_mass MLdistanceUnits^2 / fs^2)
                # # KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,args.Nc,YSlist,NHC_Q,Natoms,args.temp,MLdistanceUnits,DOF)
                # molecule = self.propagation_algorithm.update_velocities_second_half_step(molecule = molecule,
                #                                                                         time_step = self.time_step)
                self.propagation_algorithm.update_velocities_second_half_step(molecule = molecule,
                                                                                        time_step = self.time_step)
            velocity = np.copy(molecule.get_xyz_vectorial_properties('xyz_velocities'))
            kinetic_energy = np.sum(velocity**2 * self.mass) / 2.0 * 1822.888515 * (0.024188432 / constants.Bohr2Angstrom)**2 #* MLenergyUnits
            #print(molecule.energy)
            molecule.kinetic_energy = kinetic_energy
            molecule.total_energy = molecule.energy + molecule.kinetic_energy
            trajectory_step.step = istep 
            trajectory_step.time = istep * self.time_step
            trajectory_step.molecule = molecule 
            self.molecular_trajectory.steps.append(trajectory_step)
            if self.dump_trajectory_every_time_step:
                if self.format == 'h5md':
                    temp_traj = data.molecular_trajectory()
                    temp_traj.steps.append(trajectory_step)
                elif self.format == 'json':
                    temp_traj = self.molecular_trajectory
                temp_traj.dump(filename=self.filename, format=self.format)
            istep += 1
            if type(self.stop_function) != type(None):
                if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                stop = self.stop_function(molecule, **self.stop_function_kwargs)
            if istep*self.time_step > self.maximum_propagation_time:
                stop = True

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

