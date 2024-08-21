#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! gap_md: Module for gap-driven dynamics                                    ! 
  ! Implementations by: Pavlo O. Dral & Mikołaj Martyka                       ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os
import random
from . import data
from .md import md as md

class gap_model():
    def __init__(self, model=None, surface0=0, surface1=1, current_surface=1, gap_threshold=0.03, init_Ekin=0.0, nstates=2):
        self.model = model
        self.surface0 = surface0
        self.surface1 = surface1
        self.current_surface = current_surface
        self.gap_threshold = gap_threshold
        self.init_Ekin = init_Ekin
        self.nstates = nstates
        
    def predict(self, molecule=None, calculate_energy=True, calculate_energy_gradients=True):
        self.model.predict(molecule=molecule,nstates=self.nstates,calculate_energy=True, calculate_energy_gradients=[True]*self.nstates)
        energy_gap = molecule.electronic_states[self.surface1].energy - molecule.electronic_states[self.surface0].energy
        molecule.energy = molecule.electronic_states[self.current_surface].energy
        if not 'initial_energy' in self.__dict__: self.initial_energy = molecule.electronic_states[self.current_surface].energy
        if energy_gap > self.gap_threshold: gap = True
        else: gap = False
        if molecule.electronic_states[self.current_surface].energy - self.initial_energy > self.init_Ekin: gap = False
        for idx, atom in enumerate(molecule):
            if gap: atom.energy_gradients = molecule.electronic_states[self.surface1].atoms[idx].energy_gradients - molecule.electronic_states[self.surface0].atoms[idx].energy_gradients
            else:   atom.energy_gradients = molecule.electronic_states[self.current_surface].atoms[idx].energy_gradients

class gap_md():
    def __init__(self, model=None,
                 molecule_with_initial_conditions= None,
                 molecule = None,
                 ensemble='NVE',
                 thermostat=None,
                 time_step=0.1,
                 lower_state = 0,
                 upper_state = 1,
                 current_state=1, 
                 gap_threshold = 0.03,
                 init_Ekin = None,
                 nstates=2,
                 maximum_propagation_time=1000,
                 dump_trajectory_interval=None,
                 filename=None, format='h5md',
                 stop_function=None, stop_function_kwargs=None, reduce_memory_usage=False):
        self.current_state= current_state
        if not molecule_with_initial_conditions is None:
            self.molecule_with_initial_conditions = molecule_with_initial_conditions
        if not molecule is None:
            self.molecule_with_initial_conditions = molecule
        if isinstance(model, gap_model):
            self.model = model
        else:
            if init_Ekin == None:
                self.init_Ekin = self.molecule_with_initial_conditions.calculate_kinetic_energy()*random.random()
            else:
                self.init_Ekin = init_Ekin
            self.model = gap_model(model=model, surface0=lower_state, surface1=upper_state, current_surface=current_state, gap_threshold=0.03, init_Ekin=self.init_Ekin,nstates=nstates)      
        self.ensemble = ensemble
        if thermostat != None:
            self.thermostat = thermostat
        self.time_step = time_step
        self.maximum_propagation_time = maximum_propagation_time
        self.reduce_memory_usage = reduce_memory_usage
        
        self.filename = filename 
        self.format = format

        self.dump_trajectory_interval = dump_trajectory_interval
        if dump_trajectory_interval:
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
        temp_traj = data.molecular_trajectory()
        
        istep = 0
        stop = False
        init_mol = self.molecule_with_initial_conditions
        while not stop:
            dyn = md(model=self.model,
                        molecule_with_initial_conditions = init_mol,
                        ensemble=self.ensemble,
                        time_step=self.time_step,
                        maximum_propagation_time = self.time_step,
                        dump_trajectory_interval=None)
            if istep == 0:
                self.molecular_trajectory = dyn.molecular_trajectory
                Etot_ref = (dyn.molecular_trajectory.steps[0].molecule.energy + dyn.molecular_trajectory.steps[0].molecule.kinetic_energy)
            else:
                dyn.molecular_trajectory.steps[1].time = self.molecular_trajectory.steps[-1].time + self.time_step
                self.molecular_trajectory.steps += [dyn.molecular_trajectory.steps[1]]
                self.molecular_trajectory.steps[-1].step = istep
            excess_energy = (self.molecular_trajectory.steps[-1].molecule.energy + self.molecular_trajectory.steps[-1].molecule.kinetic_energy) - Etot_ref
            Ekin = self.molecular_trajectory.steps[-1].molecule.kinetic_energy
            if Ekin < excess_energy: ww = 0.0
            else: ww = (1-excess_energy / Ekin)**0.5
            init_mol = self.molecular_trajectory.steps[-1].molecule#.copy()
            for atom in init_mol.atoms:
                atom.xyz_velocities *= ww
            
            if self.dump_trajectory_interval != None:
                if self.dump_trajectory_interval % self.time_step == 0:
                    if self.format == 'h5md':
                        if istep ==0:
                            temp_traj_dump = data.molecular_trajectory()
                            temp_traj_dump = self.molecular_trajectory
                        elif self.reduce_memory_usage:
                            temp_traj = data.molecular_trajectory()
                            temp_traj.steps.append(self.molecular_trajectory.steps[-2])
                            temp_traj.steps.append(self.molecular_trajectory.steps[-1])
                            del self.molecular_trajectory.steps[-2:] 
                            self.molecular_trajectory.dump(filename=self.filename, format=self.format)
                            del self.molecular_trajectory
                            self.molecular_trajectory = temp_traj
                        else:
                            #pass
                            temp_traj_dump = data.molecular_trajectory()
                            temp_traj_dump.steps.append(self.molecular_trajectory.steps[-1])
                    elif self.format == 'json':
                        temp_traj_dump.steps.append(self.molecular_trajectory.steps[-1]) 
                #temp_traj_dump.dump(filename=self.filename, format=self.format)
            
            if istep == 0: istep += 1
            istep += 1
            if type(self.stop_function) != type(None):
                if self.stop_function_kwargs == None: self.stop_function_kwargs = {}
                if 'stop_check' not in locals():
                    stop_check = False
                stop, stop_check = self.stop_function(stop_check=stop_check,
                                                      mol=self.molecular_trajectory.steps[-1].molecule, 
                                                      current_state=self.current_state, 
                                                      **self.stop_function_kwargs)
                if stop:
                    if self.reduce_memory_usage: 
                        self.molecular_trajectory.dump(filename=self.filename, format=self.format)
                
            if (istep*(self.time_step))+ 1e-6 > self.maximum_propagation_time:
                stop = True
                if self.reduce_memory_usage: 
                    self.molecular_trajectory.dump(filename=self.filename, format=self.format)
