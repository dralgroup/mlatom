#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! initial_conditions: Module for generating initial conditions              ! 
  ! Implementations by: Yi-Fan Hou & Pavlo O. Dral                            ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np 
from . import data
from . import stopper
from . import constants

def generate_initial_conditions(molecule=None, generation_method=None, number_of_initial_conditions=1,
                                file_with_initial_xyz_coordinates=None, file_with_initial_xyz_velocities=None,
                                eliminate_angular_momentum=True,
                                degrees_of_freedom=None,
                                initial_temperature=None, initial_kinetic_energy=None):
    '''
    Generate initial conditions

    Arguments:
        molecule (:class:`data.molecule`): Molecule with necessary information
        generation_method (str): Initial condition generation method 
        number_of_initial_conditions (int): Number of initial conditions to generate, 1 by default
        file_with_initial_xyz_coordinates (str): File with initial xyz coordinates, only valid for ``generation_method='user-defined'``
        file_with_initial_xyz_velocities (str): File with initial xyz velocities, only valid for ``generation_method='user-defined'``
        eliminate_angular_momentum (bool): Remove angular momentum from velocities, valid for ``generation_method='random'``
        degrees_of_freedom (int): Degrees of freedom of the molecule, by default remove translational and rotational degrees of freedom. It can be a negative value, which means that some value is subtracted from 3*Natoms
        initial_temperature (float): Initial temperature in Kelvin, control random initial velocities
        initial_kinetic_energy (float): Initial temperature in Hartree, control random initial velocities

    .. table::
        :align: center

        =============================  =============================================
        generation_method              description
        =============================  =============================================
        ``'user-defined'`` (default)   Use user-defined initial conditions
        ``'random'``                   Generate random velocities
        =============================  =============================================

        
    Returns:
        A molecular database (:class:`ml.data.molecular_database`) with ``number_of_initial_conditions`` initial conditions

    Examples:

    .. code-block:: python

        # Use user-defined initial conditions
        init_cond_db = ml.generate_initial_conditions(molecule = mol,
                                                      generation_method = 'user-defined',
                                                      file_with_initial_xyz_coordinates = 'ethanol.xyz',
                                                      file_with_initial_xyz_velocities  = 'ethanol.vxyz',
                                                      number_of_initial_conditions = 1)
        # Generate random velocities 
        init_cond_db = ml.generate_initial_conditions(molecule = mol,
                                                      generation_method = 'random',
                                                      initial_temperature = 300,
                                                      number_of_initial_conditions = 1)
        
    '''
    if initial_temperature != None and initial_kinetic_energy != None:
        stopper.stopMLatom('Cannot use initial_temperature and initial_kinetic_energy at the same time')
    if initial_temperature == None and initial_kinetic_energy == None:
        initial_temperature = 300
    if degrees_of_freedom != None:
        if degrees_of_freedom <= 0:
            degrees_of_freedom = 3 * len(molecule.atoms) + degrees_of_freedom 
    else:
        if eliminate_angular_momentum:
            degrees_of_freedom = max(1, 3 * len(molecule.atoms) - 6)
        else:
            degrees_of_freedom = max(1, 3 * len(molecule.atoms) - 3)

    init_cond_db = data.molecular_database()
    Natoms = len(molecule.atoms)

    iteration = 0
    target_number_of_initial_conditions = number_of_initial_conditions
    while len(init_cond_db) < target_number_of_initial_conditions:
        init_cond_db = data.molecular_database()
        iteration += 1
        if generation_method.casefold() == 'user-defined'.casefold():
            init_cond_db.read_from_xyz_file(file_with_initial_xyz_coordinates)
            init_cond_db.add_xyz_vectorial_properties_from_file(file_with_initial_xyz_velocities, xyz_vectorial_property='xyz_velocities')
        elif generation_method.casefold() == 'random'.casefold():
            for irepeat in range(number_of_initial_conditions):
                new_molecule = molecule.copy(atomic_labels = ['xyz_coordinates'], molecular_labels = [])
                velocities = generate_random_velocities(new_molecule,eliminate_angular_momentum,degrees_of_freedom,temp=initial_temperature,ekin=initial_kinetic_energy)
                for iatom in range(Natoms):
                    new_molecule.atoms[iatom].xyz_velocities = velocities[iatom]
                init_cond_db.molecules.append(new_molecule)
    
    if len(init_cond_db) > target_number_of_initial_conditions:
       init_cond_db = init_cond_db[:target_number_of_initial_conditions]

    return init_cond_db

def read_velocities_from_file(filename):
    velocities = []
    with open(filename, 'r') as fxyz:
        nlines = 0
        natoms = 0
        for line in fxyz:
            nlines += 1
            if nlines == 1:
                natoms = int(line)
            elif nlines > 2 and nlines <= 2 + natoms:
                yy = line.split()
                velocities.append([float(xx) for xx in yy[-3:]])
                if nlines == 2 + natoms:
                    break
    return velocities

def generate_random_velocities(molecule,noang,dof,temp=None,ekin=None):
    np.random.seed()
    Natoms = len(molecule.atoms)
    randnum = np.random.randn(Natoms,3)
    coord = np.array([each.xyz_coordinates for each in molecule.atoms])
    
    mass_ = np.array([each.nuclear_mass for each in molecule.atoms])
    mass = np.array(mass_).reshape(Natoms,1)
    
    if temp != None:
        # kb = 1.380649E-23  # Unit: J/K
        # hartree = 27.21070 * 1.602176565E-19 # Unit: J/Hartree
        # kb = kb / hartree # Unit: Hartree/K
        kb = constants.kB_in_Hartree # Unit: Hartree/K
        kinetic_energy = dof/2.0 * kb * temp
    else:
        kinetic_energy = ekin
    
    linearity = molecule.is_it_linear()
    # Eliminate total angular momentum
    if noang:
        if linearity:
            rand_velocity = generate_random_velocities_for_linear_molecule(molecule)
        else:
            #rand_velocity = randnum / np.sqrt(mass*1822)
            rand_velocity = randnum / np.sqrt(mass*constants.ram2au)
            rand_velocity = getridofang(coord,rand_velocity,mass_)
    else:
        rand_velocity = randnum / np.sqrt(mass*constants.ram2au)

    #print(rand_velocity)

    # Raise warning if degrees of freedom are not compatible to the linear molecule
    if linearity:
        if dof != 3*Natoms-5:
            print(f'WARNING: Linear molecule detected, but degrees of freedom used is {dof} instead of {3*Natoms-5}')


    # Eliminate total linear momentum
    total_mass = sum(mass)[0]
    v_cm = sum(rand_velocity*mass)/total_mass
    rand_velocity = rand_velocity - v_cm
    
    #rand_energy = np.sum((rand_velocity**2) * (mass*1822)) / 2.0  
    rand_energy = np.sum((rand_velocity**2) * (mass*constants.ram2au)) / 2.0
    ratio = rand_energy / kinetic_energy 
    velocity = rand_velocity / np.sqrt(ratio) # Unit: a.u.
    
    # bohr = 0.5291772083 # 1 Bohr = 0.5291772083 Angstrom
    # time = 2.4189E-2 # Time Unit: 1 a.u. = 2.4189E-17 s = 2.4189E-2 fs
    # velocity = velocity * bohr / time # Unit: Angstrom / fs
    velocity = velocity * constants.Bohr2Angstrom / constants.au2fs # Unit: Angstrom / fs
    return velocity

def getridofang(coord,vel,mass):
    omega = getAngV(coord,vel,mass)
    coord_ = coord - getCoM(coord,mass)
    vel = vel - np.cross(omega,coord)

    return vel 

def getAngV(xyz,v,m,center=None):
    L=getAngM(xyz,v,m,center)
    I=getMomentOfInertiaTensor(xyz,m,center)
    omega=np.linalg.inv(I).dot(L)
    return omega

def getCoM(xyz,m=None):
    if m is None:
        m=np.ones(xyz.shape[-2])
    return np.sum(xyz*m[:,np.newaxis],axis=-2)/np.sum(m)

def getAngM(xyz,v,m,center=None):
    if center is None:
        centered=xyz-getCoM(xyz,m)
    else:
        centered=xyz-center
    L=np.sum(m[:,np.newaxis]*np.cross(centered,v),axis=0)
    return L

def getMomentOfInertiaTensor(xyz,m,center=None):
    if center is None:
        center=getCoM(xyz,m)
    centered=xyz-center
    I=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(len(m)):
                I[i,j]+=m[k]*(np.sum(centered[k]**2)-centered[k,i]*centered[k,j]) if i==j else m[k]*(-centered[k,i]*centered[k,j])
    return I
    
# generate random velocities without angular momentum for linear molecule
def generate_random_velocities_for_linear_molecule(molecule):
    np.random.seed()
    Natoms = len(molecule.atoms)
    coord = molecule.xyz_coordinates
    randnum = np.random.randn(Natoms)
    avgnum = np.average(randnum)
    randnum = randnum - avgnum 
    vec = coord[1] - coord[0]
    rand_velocities = [vec*each for each in randnum]
    return rand_velocities

    