#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! ase: interface to the ASE package                                         ! 
  ! Implementations by: Peikung Zheng and Pavlo O. Dral                       ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import copy, tempfile
import warnings
warnings.filterwarnings("ignore")
from .. import constants
from .. import data
from .. import stopper
    

import ase
from ase import io
from ase.calculators.calculator import Calculator, all_changes
from ase.thermochemistry import IdealGasThermo
import ase.units as units

def optimize_geometry(initial_molecule, model, convergence_criterion_for_forces, maximum_number_of_steps, optimization_algorithm='LBFGS', **kwargs):
    if optimization_algorithm == None:
        optimization_algorithm = 'LBFGS'
    
    if 'model_predict_kwargs' in kwargs:
        model_predict_kwargs = kwargs['model_predict_kwargs']
    else:
        model_predict_kwargs = {}
    
    optimization_trajectory = data.molecular_trajectory()
    globals()['initial_molecule'] = initial_molecule
    globals()['optimization_trajectory'] = optimization_trajectory
    
    # Ugly solution because no time to understand what 'atoms' in ASE is exactly (probably corresponds to MLatom's 'molecular_database')
    # more like 'molecule', the 'atoms' object defines a collection of atoms
    with tempfile.TemporaryDirectory() as tmpdirname: # todo: new atoms object directly, with something like molecule.ase()
        xyzfilename = f'{tmpdirname}/tmp.xyz'
        initial_molecule.write_file_with_xyz_coordinates(filename=xyzfilename)
        atoms = io.read(xyzfilename, index=':', format='xyz')[0]
    if 'constraints' in kwargs:
        constraints = kwargs['constraints']
        from ase.constraints import FixInternals 
        
        all_constraints = []
        # Fix bond lengths
        if 'bond_length' in constraints:
            from ase.constraints import FixBondLengths
            bond_lengths_constraints = FixBondLengths(constraints['bond_length'])
            all_constraints.append(bond_lengths_constraints)
        # Fix internal coordinates
        if 'bonds' in constraints:
            bonds_constraints = constraints['bonds']
        else:
            bonds_constraints = None 
        if 'angles' in constraints:
            angles_constraints = constraints['angles']
        else:
            angles_constraints = None 
        if 'dihedrals' in constraints:
            dihedrals_constraints = constraints['dihedrals']
        else:
            dihedrals_constraints = None 
        internal_constraints = FixInternals(bonds=bonds_constraints,angles_deg=angles_constraints,dihedrals_deg=dihedrals_constraints)
        all_constraints.append(internal_constraints)

        atoms.set_constraint(all_constraints)
            
    # atoms.set_calculator(MLatomCalculator(model=model, save_optimization_trajectory=True))
    # atoms.set_calculator() is deprecated
    atoms.calc = MLatomCalculator(model=model, save_optimization_trajectory=True, model_predict_kwargs=model_predict_kwargs)
    
    from ase import optimize
    opt = optimize.__dict__[optimization_algorithm](atoms)
    opt.run(fmax=convergence_criterion_for_forces, steps=maximum_number_of_steps)
    
    # For some reason ASE dumps the same energy twice. Here we remove the repeated value.
    if len(optimization_trajectory.steps) > 1:
        if abs(optimization_trajectory.steps[1].molecule.energy - optimization_trajectory.steps[0].molecule.energy) < 1e-13:
            for istep in range(2,len(optimization_trajectory.steps)):
                optimization_trajectory.steps[istep].step -= 1
            del optimization_trajectory.steps[1]
    
    return optimization_trajectory

def transition_state(initial_molecule, model, 
                     convergence_criterion_for_forces,
                     maximum_number_of_steps,  
                     optimization_algorithm='dimer', 
                     **kwargs):
    if optimization_algorithm == None:
        optimization_algorithm = 'dimer'

    if 'model_predict_kwargs' in kwargs:
        # model_predict_kwargs = kwargs['model_predict_kwargs']
        model_predict_kwargs = kwargs.pop('model_predict_kwargs')
    else:
        model_predict_kwargs = {}

    if optimization_algorithm.casefold() == 'dimer'.casefold():
        return dimer_method(initial_molecule, model, 
                            model_predict_kwargs,
                            convergence_criterion_for_forces,
                            maximum_number_of_steps, **kwargs)
    elif optimization_algorithm.casefold() == 'NEB'.casefold():
        return nudged_elastic_band(initial_molecule, kwargs.pop('final_molecule'), model, 
                                   model_predict_kwargs,
                                   convergence_criterion_for_forces,
                                   maximum_number_of_steps,  **kwargs)

def dimer_method(initial_molecule, model, 
                 model_predict_kwargs,
                 convergence_criterion_for_forces,
                 maximum_number_of_steps,  **kwargs):
    optimization_trajectory = data.molecular_trajectory()
    globals()['initial_molecule'] = initial_molecule
    globals()['optimization_trajectory'] = optimization_trajectory

    with tempfile.TemporaryDirectory() as tmpdirname: # todo: new atoms object directly
        xyzfilename = f'{tmpdirname}/tmp.xyz'
        initial_molecule.write_file_with_xyz_coordinates(filename=xyzfilename)
        atoms = io.read(xyzfilename, index=':', format='xyz')[0]

    atoms.calc = MLatomCalculator(model=model,  model_predict_kwargs= model_predict_kwargs, save_optimization_trajectory=True)

    from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate

    with DimerControl(**kwargs) as d_control:
        d_atoms = MinModeAtoms(atoms, d_control, random_seed = kwargs['random_seed'] if 'random_seed' in kwargs else 0)
        d_atoms.displace()
        with MinModeTranslate(d_atoms) as dim_rlx:
            dim_rlx.run(fmax=convergence_criterion_for_forces,
                        steps=maximum_number_of_steps)
    
    return optimization_trajectory

def nudged_elastic_band(initial_molecule, final_molecule, model, 
                        model_predict_kwargs,
                        convergence_criterion_for_forces,
                        maximum_number_of_steps,
                        number_of_middle_images=3, **kwargs):
    optimization_trajectory = data.molecular_trajectory()
    globals()['initial_molecule'] = initial_molecule
    globals()['optimization_trajectory'] = optimization_trajectory

    with tempfile.TemporaryDirectory() as tmpdirname: # todo: new atoms object directly
        xyzfilename = f'{tmpdirname}/tmp_init.xyz'
        initial_molecule.write_file_with_xyz_coordinates(filename=xyzfilename)
        initial = io.read(xyzfilename, index=':', format='xyz')[0]
        xyzfilename = f'{tmpdirname}/tmp_final.xyz'
        final_molecule.write_file_with_xyz_coordinates(filename=xyzfilename)
        final = io.read(xyzfilename, index=':', format='xyz')[0]

    from ase.neb import NEB
    from ase.optimize import MDMin
    images = [initial]
    images += [initial.copy() for _ in range(number_of_middle_images)]
    images += [final]
    neb = NEB(images, **kwargs)
    neb.interpolate()
    for image in images[1:number_of_middle_images+1]:
        image.calc = MLatomCalculator(model=model,  model_predict_kwargs= model_predict_kwargs, save_optimization_trajectory=True)
    optimizer = MDMin(neb, trajectory='A2B.traj')
    optimizer.run(fmax=convergence_criterion_for_forces,
                  steps=maximum_number_of_steps)
    return optimization_trajectory


class MLatomCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, model,  model_predict_kwargs, save_optimization_trajectory = False):
        super(MLatomCalculator, self).__init__()
        self.model = model
        self.model_predict_kwargs =  model_predict_kwargs
        self.save_optimization_trajectory = save_optimization_trajectory

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super(MLatomCalculator, self).calculate(atoms, properties, system_changes)
        global initial_molecule
        # Ugly solution because no time to understand what 'atoms' in ASE is exactly (probably corresponds to MLatom's 'molecular_database')
        with tempfile.TemporaryDirectory() as tmpdirname:
            xyzfilename = f'{tmpdirname}/tmp.xyz'
            io.write(xyzfilename, self.atoms, format='extxyz', plain=True)
            
            current_molecule = initial_molecule.copy()
            mol_from_file = data.molecule()
            mol_from_file.read_from_xyz_file(filename=xyzfilename)
            coordinates = mol_from_file.xyz_coordinates
            current_molecule.xyz_coordinates = coordinates
            
            self.model._predict_geomopt(molecule=current_molecule, calculate_energy=True, calculate_energy_gradients=True, **self.model_predict_kwargs)
            if not 'energy' in current_molecule.__dict__:
                raise ValueError('model did not return any energy')
            
            if self.save_optimization_trajectory:
                global optimization_trajectory
                istep = len(optimization_trajectory.steps)
                optimization_trajectory.steps.append(data.molecular_trajectory_step(step=istep, molecule=current_molecule))
            
            energy = current_molecule.energy
            forces = -current_molecule.get_energy_gradients()

            energy *= ase.units.Hartree
            forces *= ase.units.Hartree

            self.results['energy'] = energy

            if 'forces' in properties:
                self.results['forces'] = forces

def thermochemistry(molecule):
    energy = molecule.energy * units.Hartree
    cm2ev = 100.0 * units._c * units._hplanck / units._e
    ev2kcal = units.mol / units.kcal
    freqs = copy.deepcopy(molecule.frequencies)
    # Check negative frequencies 
    nnegative = 0
    for ii in range(len(freqs)):
        if freqs[ii] < 1:
            nnegative += 1
        else:
            break 
    if nnegative == len(freqs):
        print('* warning * All the frequencies are negative, skip thermochemistry calculation')
        return 
    elif nnegative > 0:
        print(f'* warning * {nnegative} negative frequencies found, remove them before thermochemistry calculation')
        freqs = freqs[nnegative:]
    # print(freqs)
    if 'shape' not in molecule.__dict__.keys():
        molecule.shape = 'nonlinear'
    if molecule.shape.lower() == 'linear': add_freqs = np.zeros(5)
    else: add_freqs = np.zeros(6)
    vib_energies = freqs * cm2ev
    geometry = 'nonlinear'
    if molecule.shape.lower() == 'linear': geometry = 'linear'
    spin = (molecule.multiplicity - 1) / 2
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        xyzfilename = f'{tmpdirname}/tmp.xyz'
        molecule.write_file_with_xyz_coordinates(filename=xyzfilename)
        mol = io.read(xyzfilename, index=':', format='xyz')[0]
    
    if 'symmetry_number' not in molecule.__dict__.keys():
        molecule.symmetry_number = 1
    thermo = IdealGasThermo(vib_energies=vib_energies,
                            potentialenergy=energy,
                            atoms=mol,
                            geometry=geometry,
                            symmetrynumber=molecule.symmetry_number,
                            spin=spin)
    
    molecule.G = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.) * ev2kcal * constants.kcalpermol2Hartree
    molecule.H = thermo.get_enthalpy(temperature=298.15, verbose=True) * ev2kcal * constants.kcalpermol2Hartree
    molecule.H0 = thermo.get_enthalpy(temperature=0.0, verbose=True) * ev2kcal * constants.kcalpermol2Hartree
    molecule.U0 = molecule.H0
    molecule.ZPE = thermo.get_ZPE_correction() * ev2kcal * constants.kcalpermol2Hartree
    