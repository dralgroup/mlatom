import time 
import re, sys, os 
from .args_class import ArgsBase 
from .stopper import stopMLatom 
from .initial_conditions import generate_initial_conditions 
from .md import md
from . import constants
from . import data
from . import simulations
from .interfaces import gaussian_interface

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.args2pass = []

        self.add_dict_args({
            'dt':0.1,                      # Time step; Unit: fs
            'trun':1000,                   # Length of trajectory; Unit: fs
            'initXYZ':'',                  # File containing initial geometry
            'initVXYZ':'',                 # File containing initial velocity
            'initConditions':'',           # How to generate initial condition
            'normalModefile':'',           # Gaussian ouput file containing normal modes
            'trajH5MDout':'traj.h5',       # Output file: H5MD format
            'trajTextOut':'traj',          # Output file
            'MLenergyUnits':'',            # Energy unit in ML model
            'MLdistanceUnits':'',          # Distance unit in ML model
            'randomSeed':'',               # Random seed for initial condition sampling
            'ensemble':'nve',
            'thermostat':'',               # Thermostat
            'gamma':0.2,                   # Option for Anderson thermostat
            'initTemperature':300,         # Initial temperature 
            'temperature':300,             # Thermostat temperature
            'initVXYZout':'',              # Output file containing initial velocity
            'initXYZout':'',               # Output file containing initial geometry
            'NHClength':3,                 # Nose-Hoover chain length
            'Nc':3,                        # Multiple time step
            'Nys':7,                       # Number of Yoshida Suzuki steps used in NHC (1,3,5,7)
            'NHCfreq':1.0/16,
            'noang':1,
            'DOF':-6,
            'linear':0,

        })


    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['', None])

class MD_CMD():
    def __init__(self,args3D):
        args = Args()
        args.parse(args3D)

    @classmethod 
    def dynamics(cls,args3D,method):
        args = Args() 
        args.parse(args3D)

        # Deal with Units
        # By default, use Hartree and Angstrom
        # MLenergyUnits and MLdistanceUnits are not valid, at least, in this version.
        if args.MLenergyUnits == '' or args.MLenergyUnits.lower() == 'hartree':
            MLenergyUnits = constants.Hartree2kcalpermol
        elif args.MLenergyUnits.lower() == 'kcal/mol':
            MLenergyUnits = 1.0
        else:
            stopMLatom('Unknown MLenergyUnits: %s'%(args.MLenergyUnits))
        if args.MLdistanceUnits == '' or args.MLdistanceUnits.lower() == 'angstrom':
            MLdistanceUnits = constants.Bohr2Angstrom 
        elif args.MLdistanceUnits.lower() == 'bohr':
            MLdistanceUnits = 1.0
        else: 
            stopMLatom('Unknown MLdistanceUnits: %s'%(args.MLenergyUnits))

        if args.randomSeed == '':
            random_seed = None 
        else:
            random_seed = int(args.randomSeed)

        print("  Propagating molecular dynamics...")

        # Deal with initial conditions 
        # Check file and get initial molecule
        if args.initConditions == '' or args.initConditions.lower() == 'user-defined' or args.initConditions.lower() == 'random':
            if args.initXYZ == '':
                stopMLatom('Please provide initial XYZ file')
            if not os.path.exists(args.initXYZ):
                stopMLatom('User-defined initial XYZ file %s does not exist'%(args.initXYZ))
            mol = data.molecule()
            mol.read_from_xyz_file(args.initXYZ)
        else:
            mol = data.molecule()
            mol.load(args.normalModefile)

        # Deal with degrees of freedom
        if args.initConditions == '' or args.initConditions.lower() == 'user-defined' or args.initConditions.lower() == 'random':
            if mol.is_it_linear():
                linear = 1 
                print('    Linear molecule detected')
            else:
                linear = 0
            Natoms = len(mol.atoms)
            DOF = 3*Natoms-6+linear 
        else:
            if mol.is_it_linear():
                linear = 1 
                print('    Linear molecule detected')
            else:
                linear = 0
            Natoms = len(mol.atoms)
            DOF = 3*Natoms-6+linear 
        
        # print('    Degrees of freedom: %d'%(DOF))

        # Generate initial conditions
        if args.initConditions == '' or args.initConditions.lower() == 'user-defined':
            print('    Use user-defined initial condition')
            print(f'      Initial XYZ coordinates file: {args.initXYZ}')
            print(f'      Initial XYZ velocities file : {args.initVXYZ}')
            # Check file 
            if args.initVXYZ == '':
                stopMLatom('Please provide initial VXYZ file')
            if not os.path.exists(args.initVXYZ):
                stopMLatom('User-defined initial VXYZ file %s does not exist'%(args.initVXYZ))
            init_cond_db = generate_initial_conditions(molecule = mol,
                                                          generation_method = 'user-defined',
                                                          file_with_initial_xyz_coordinates = args.initXYZ,
                                                          file_with_initial_xyz_velocities  = args.initVXYZ)
        elif args.initConditions.lower() == 'random':
            initTemperature = args.initTemperature
            print('    Use random sampling to generate initial condition')
            print(f'      Initial XYZ coordinates file: {args.initXYZ}')
            print(f'      Initial instantaneous temperature: {initTemperature}')
            init_cond_db = generate_initial_conditions(molecule=mol,
                                                       generation_method='random',
                                                       degrees_of_freedom = DOF,
                                                       initial_temperature = initTemperature,
                                                       random_seed = random_seed)
        elif args.initConditions.lower() == 'wigner':
            print('    Use Wigner sampling to generate initial condition')
            if args.normalModefile != '' and os.path.exists(args.normalModefile):
                try:
                    if args.normalModefile[-4:] == 'json':
                        mol = data.molecule() 
                        mol.load(args.normalModefile,format='json')
                    else:
                        mol = data.molecule.from_xyz_file(args.normalModefile)
                except:
                    stopMLatom(f'Failed to open normal model file {args.normalModefile}')
            elif args.normalModefile == '':
                stopMLatom('Please provide file with normal modes')
            elif not os.path.exists(args.normalModefile):
                stopMLatom(f'Normal model file {args.normalMode} does not exist')
            init_cond_db = generate_initial_conditions(molecule=mol,
                                                       generation_method='wigner',
                                                       initial_temperature = args.initTemperature,
                                                       random_seed = random_seed)
        elif args.initConditions.lower() == 'harmonic-quantum-boltzmann':
            print('    Use harmonic quantum Boltzmann distribution to generate initial condition')
            if args.normalModefile != '' and os.path.exists(args.normalModefile):
                try:
                    if args.normalModefile[-4:] == 'json':
                        mol = data.molecule() 
                        mol.load(args.normalModefile,format='json')
                    else:
                        mol = data.molecule.from_xyz_file(args.normalModefile)
                except:
                    stopMLatom(f'Failed to open normal model file {args.normalModefile}')
            elif args.normalModefile == '':
                stopMLatom('Please provide file with normal modes')
            elif not os.path.exists(args.normalModefile):
                stopMLatom(f'Normal model file {args.normalMode} does not exist')
            init_cond_db = generate_initial_conditions(molecule=mol,
                                                       generation_method='harmonic-quantum-boltzmann',
                                                       initial_temperature = args.initTemperature,
                                                       random_seed = random_seed)

        init_mol = init_cond_db.molecules[0]
        print_initial_condition(init_mol)
        if args.initXYZout != '':
            init_mol.write_file_with_xyz_coordinates(filename=args.initXYZout)
        if args.initVXYZout != '':
            write_velocities(args.initVXYZout,init_mol)

        # Deal with thermostat 
        if args.ensemble.lower() == 'nve': # NVE ensemble 
            if args.thermostat == '': 
                thermostat = None
            else:
                stopMLatom(f'Unrecognized thermostat for NVE: {args.thermostat}')
        elif args.ensemble.lower() == 'nvt':
            if args.thermostat.lower() == 'andersen':
                thermostat = md.Andersen_thermostat(gamma=args.gamma,temperature=args.temperature)
            elif args.thermostat.lower() == 'nose-hoover':
                thermostat = md.Nose_Hoover_thermostat(nose_hoover_chain_length=args.NHClength,
                                                       multiple_time_steps=args.Nc,
                                                       number_of_yoshida_suzuki_steps=args.Nys,
                                                       nose_hoover_chain_frequency=args.NHCfreq,
                                                       temperature=args.temperature,
                                                       molecule=init_mol,
                                                       degrees_of_freedom=DOF)

        # Deal with dynamics 
        dyn = md(model=method,
                    molecule_with_initial_conditions = init_mol,
                    ensemble=args.ensemble,
                    thermostat=thermostat,
                    time_step=args.dt,
                    maximum_propagation_time = args.trun,
                    dump_trajectory_interval=1,
                    filename=args.trajH5MDout, format='h5md')

        # Deal with outputs
        dyn.molecular_trajectory.dump(filename=args.trajTextOut, format='plain_text')

def write_velocities(filename,mol):
    velocities = mol.get_xyz_vectorial_properties('xyz_velocities')
    Natoms = len(mol.atoms)
    with open(filename,'w') as f:
        f.write(f'{Natoms}\n\n')
        for iatom in range(Natoms):
            f.write('%25.13f %25.13f %25.13f\n'%(velocities[iatom][0],velocities[iatom][1],velocities[iatom][2]))

def print_initial_condition(mol):
    Natoms = len(mol.atoms)
    coordinates = mol.xyz_coordinates
    velocities = mol.get_xyz_vectorial_properties('xyz_velocities')
    symbols = mol.get_element_symbols()
    print('      Initial XYZ coordinates: ')
    for iatom in range(Natoms):
        print('        %s %25.13f %25.13f %25.13f'%(symbols[iatom],coordinates[iatom][0],coordinates[iatom][1],coordinates[iatom][2]))
    print('      Initial XYZ velocities: ')
    for iatom in range(Natoms):
        print('        %s %25.13f %25.13f %25.13f'%(symbols[iatom],velocities[iatom][0],velocities[iatom][1],velocities[iatom][2]))