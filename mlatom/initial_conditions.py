#!/usr/bin/env python3
from . import simulations
from . import constants
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! initial_conditions: Module for generating initial conditions              ! 
  ! Implementations by: Yi-Fan Hou, Lina Zhang, Fuchun Ge, Pavlo O. Dral      ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np 
from . import data
from . import stopper
from . import constants

def excitation_energy_window_filter(molecular_database=None,
                                    model=None,
                                    model_predict_kwargs={},
                                    target_excitation_energy=1,
                                    window_half_width=0.1,
                                    f_max=None,
                                    random_seed=None):
    molDB = molecular_database
    if not random_seed is None:
        np.random.seed(random_seed)
        random_number = np.random.random()
    else:
        random_number = np.random.random()
    init_cond_db = data.molecular_database()

    check_os=True
    if check_os and (type(f_max) == type(None)):
        f_list = []
        for mol in molDB.molecules:
            model.predict(molecule=mol, calculate_energy=True, **model_predict_kwargs)
            for i in range(1, len(mol.electronic_states)):
                f_list.append(mol.oscillator_strengths[i-1])
        f_max = np.max(np.array(f_list))
    else:
        for mol in molDB.molecules:
            model.predict(molecule=mol, calculate_energy=True, **model_predict_kwargs)

    for mol in molDB.molecules:
        for i in range(1, len(mol.electronic_states)):
            excitation_energy = (mol.electronic_states[i].energy - mol.electronic_states[0].energy) * constants.Hartree2eV
            if abs(excitation_energy - target_excitation_energy) <= window_half_width:
                if check_os:
                    if mol.oscillator_strengths[i-1]/f_max >= random_number:
                        mol.current_state = i
                        mol.energy == mol.electronic_states[i].energy
                        init_cond_db.molecules.append(mol)
                else:
                    mol.current_state = i
                    init_cond_db.molecules.append(mol)
    if check_os:
        return check_os, f_max, init_cond_db
    else:
        return check_os, init_cond_db

def generate_initial_conditions(molecule=None, generation_method=None, number_of_initial_conditions=1,
                                file_with_initial_xyz_coordinates=None, file_with_initial_xyz_velocities=None,
                                eliminate_angular_momentum=True,
                                degrees_of_freedom=None,
                                initial_temperature=None, initial_kinetic_energy=None,
                                use_hessian=False,
                                reaction_coordinate_momentum=True,
                                filter_by_energy_window=False,
                                window_filter_kwargs={},
                                random_seed=None):
    '''
    Generate initial conditions

    Arguments:
        molecule (:class:`data.molecule`): molecule with necessary information
        generation_method (str): initial condition generation method, see below the table
        number_of_initial_conditions (int): number of initial conditions to generate, 1 by default
        file_with_initial_xyz_coordinates (str): file with initial xyz coordinates, only valid for ``generation_method='user-defined'``
        file_with_initial_xyz_velocities (str): file with initial xyz velocities, only valid for ``generation_method='user-defined'``
        eliminate_angular_momentum (bool): remove angular momentum from velocities, valid for ``generation_method='random'`` and ``generation_method='wigner'``
        degrees_of_freedom (int): degrees of freedom of the molecule, by default remove translational and rotational degrees of freedom. It can be a negative value, which means that some value is subtracted from 3*Natoms
        initial_temperature (float): initial temperature in Kelvin, control random initial velocities
        initial_kinetic_energy (float): initial kinetic energy in Hartree, control random initial velocities
        random_seed (int): random seed for numpy random number generator (do not use unless you want to obtain the same results every time)
        filter_by_energy_window (bool): filter by excitation energy window
        window_filter_kwargs (dict): keyword arguments for filtering the energy window, see below the table

    .. table::
        :align: center

        =============================  =============================================
        generation_method              description
        =============================  =============================================
        ``'user-defined'`` (default)   use user-defined initial conditions
        ``'random'``                   generate random velocities
        ``'maxwell-boltzmann'``        randomly generate initial velocities from Maxwell-Boltzmann distribution
        ``'wigner'``                   use Wigner sampling as implemented in `Newton-X <https://doi.org/10.1021/acs.jctc.2c00804>`__
        =============================  =============================================

        
    .. table::
        :align: center
        
        ================================  ================================================================================================================
        window_filter_kwargs              description
        ================================  ================================================================================================================
        model                             model or method that can calculate excitation energies and oscillator strengths
        model_predict_kwargs              keyword arguments for above model, typically ``nstates`` specifying how many states to calculate
        target_excitation_energy (float)  in eV
        window_half_width (float)         in eV
        random_seed (int)                 random seed for numpy random number generator (do not use unless you want to obtain the same results every time)
        ================================  ================================================================================================================
        
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
        # Use Wigner sampling  
        init_cond_db = ml.generate_initial_conditions(molecule = mol,
                                                      generation_method = 'wigner',
                                                      number_of_initial_conditions = 1)

        # Sample with filtering by excitation energy window. Requires the model for calculating excitation energies and oscillator strengths.
        model = ml.models.methods(method='AIQM1') 
        model_predict_kwargs={'nstates':9} # requests calculation of 9 electronic states
        window_filter_kwargs={'model':model,
                              'model_predict_kwargs':model_predict_kwargs, 
                              'target_excitation_energy':5.7, # eV
                              'window_half_width':0.1, # eV}
        init_cond_db = ml.generate_initial_conditions(molecule=mol,
                                                    generation_method='wigner',
                                                    number_of_initial_conditions=5,
                                                    initial_temperature=0,
                                                    random_seed=0,
                                                    use_hessian=False,
                                                    filter_by_energy_window=True,
                                                    window_filter_kwargs=window_filter_kwargs)
        
    .. note::

        Wigner sampling needs Hessian matrix. You can use ``ml.models.methods.predict(molecule=mol,calculate_hessian=True)`` to get Hessian matrix.
        
    '''
    if not random_seed is None:
        np.random.seed(random_seed)

    if initial_temperature != None and initial_kinetic_energy != None:
        stopper.stopMLatom('Cannot use initial_temperature and initial_kinetic_energy at the same time')
    if initial_temperature == None and initial_kinetic_energy == None:
        if generation_method.casefold() == 'wigner'.casefold():
            initial_temperature = 0
        else:
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
        elif generation_method.casefold() == 'maxwell-boltzmann'.casefold():
            init_cond_db.read_from_numpy(coordinates=np.repeat([molecule.xyz_coordinates], number_of_initial_conditions, axis=0), species=np.repeat([molecule.element_symbols], number_of_initial_conditions, axis=0))
            velocities = np.random.randn(*init_cond_db.xyz_coordinates.shape) * np.sqrt(initial_temperature * constants.kB / (init_cond_db.nuclear_masses / 1000 / constants.Avogadro_constant))[...,np.newaxis] / 1E5
            init_cond_db.add_xyz_vectorial_properties(velocities, xyz_vectorial_property='xyz_velocities')
        elif generation_method.casefold() == 'wigner'.casefold():
            coordinates_all, velocities_all = wigner_sampling.sample(number_of_initial_conditions,molecule,temperature=initial_temperature,use_hessian=use_hessian,reaction_coordinate_momentum=reaction_coordinate_momentum)
            mass_ = np.array([each.nuclear_mass for each in molecule.atoms])
            mass = mass_.reshape(Natoms,1)
            total_mass = np.sum(mass_)
            for irepeat in range(number_of_initial_conditions):
                if eliminate_angular_momentum and not molecule.is_it_linear():
                    velocities_all[irepeat] = getridofang(coordinates_all[irepeat],velocities_all[irepeat],mass_)
                    v_cm = sum(velocities_all[irepeat]*mass)/total_mass
                    velocities_all[irepeat] -= v_cm
            for irepeat in range(number_of_initial_conditions):
                new_molecule = molecule.copy(atomic_labels = [], molecular_labels = []) 
                for iatom in range(Natoms):
                    new_molecule.atoms[iatom].xyz_coordinates = coordinates_all[irepeat][iatom]
                    new_molecule.atoms[iatom].xyz_velocities = velocities_all[irepeat][iatom]
                init_cond_db.molecules.append(new_molecule)
        elif generation_method.casefold() == 'harmonic-quantum-boltzmann'.casefold():
            init_cond_db = harmonic_quantum_Boltzmann_sampling.sample(npoints=number_of_initial_conditions,molecule=molecule,temperature=initial_temperature,use_hessian=use_hessian)

        if filter_by_energy_window:
            if iteration == 1:
                result = excitation_energy_window_filter(init_cond_db,**window_filter_kwargs)
                if result[0]:
                    check_os, f_max, init_cond_db = result
                else:
                    check_os, init_cond_db = result
                filtered_ratio = len(init_cond_db)/number_of_initial_conditions
                if filtered_ratio == 0:
                    number_of_initial_conditions = int((target_number_of_initial_conditions - len(init_cond_db))/0.5)
                else:
                    number_of_initial_conditions = int((target_number_of_initial_conditions - len(init_cond_db))/filtered_ratio)
                previous_init_cond_db = init_cond_db
            else:
                if check_os:
                    result = excitation_energy_window_filter(molecular_database=init_cond_db,f_max=f_max,**window_filter_kwargs)
                    check_os, f_max, init_cond_db = result
                    filtered_ratio = len(init_cond_db)/number_of_initial_conditions
                    init_cond_db += previous_init_cond_db
                    previous_init_cond_db = init_cond_db
                else:
                    result = excitation_energy_window_filter(molecular_database=init_cond_db,**window_filter_kwargs)
                    check_os, init_cond_db = result
                    filtered_ratio = len(init_cond_db)/number_of_initial_conditions
                    init_cond_db += previous_init_cond_db
                    previous_init_cond_db = init_cond_db
                if filtered_ratio == 0:
                    number_of_initial_conditions = int((target_number_of_initial_conditions - len(init_cond_db))/0.5)
                else:
                    number_of_initial_conditions = int((target_number_of_initial_conditions - len(init_cond_db))/filtered_ratio)
    
    if len(init_cond_db) > target_number_of_initial_conditions:
       init_cond_db = init_cond_db[:target_number_of_initial_conditions]

    # Change the random seed so that the user-defined one only affects initial conditions sampling
    if not random_seed is None:
        np.random.seed()

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
    Natoms = len(molecule.atoms)
    randnum = np.random.randn(Natoms,3)
    coord = np.array([each.xyz_coordinates for each in molecule.atoms])
    
    mass_ = np.array([each.nuclear_mass for each in molecule.atoms])
    mass = np.array(mass_).reshape(Natoms,1)
    
    if temp != None:
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
            rand_velocity = randnum / np.sqrt(mass*constants.ram2au)
            rand_velocity = getridofang(coord,rand_velocity,mass_)
    else:
        rand_velocity = randnum / np.sqrt(mass*constants.ram2au)

    # Raise warning if degrees of freedom are not compatible to the linear molecule
    if linearity:
        if dof != 3*Natoms-5:
            print(f'WARNING: Linear molecule detected, but degrees of freedom used is {dof} instead of {3*Natoms-5}')


    # Eliminate total linear momentum
    total_mass = sum(mass)[0]
    v_cm = sum(rand_velocity*mass)/total_mass
    rand_velocity = rand_velocity - v_cm
    
    rand_energy = np.sum((rand_velocity**2) * (mass*constants.ram2au)) / 2.0
    ratio = rand_energy / kinetic_energy 
    velocity = rand_velocity / np.sqrt(ratio) # Unit: a.u.
    
    velocity = velocity * constants.Bohr2Angstrom / constants.au2fs # Unit: Angstrom / fs
    return velocity

def getridofang(coord,vel,mass):
    omega = getAngV(coord,vel,mass)
    coord_ = coord - getCoM(coord,mass)
    vel = vel - np.cross(omega,coord_)

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

class wigner_sampling():
    # Copyright (C) 2022  Light and Molecules under the GNU license
    #
    # Functions in class wigner_sampling are modified from NewtonX-2.4-B06 by Yi-Fan Hou
    # cite Newton-X when using it (M. Barbatti, M. Bondanza, R. Crespo-Otero, B. Demoulin, P. O. Dral, G. Granucci, F. Kossoski, H. Lischka, B. Mennucci, S. Mukherjee, M. Pederzoli, M. Persico, M. Pinheiro Jr, J. Pittner, F. Plasser, E. Sangiogo Gil, and L. Stojanovic. Newton-X Platform: New Software Developments for Surface Hopping and Nuclear Ensembles. J. Chem. Theory Comput. 18, 6851 (2022).)
    def __init__(self):
        pass 

    @classmethod 
    def sample(cls,npoints,molecule,temperature=0,use_hessian=True,reaction_coordinate_momentum=True):
        qlist = [] 
        vlist = []
        geomEq = np.array([each.xyz_coordinates for each in molecule.atoms])
        mass = molecule.get_nuclear_masses() 
        mass = mass.reshape(1,len(mass))
         

        linear = molecule.is_it_linear()
        Natoms = len(molecule.atoms)
        if linear:
            ntriv = 5 
        else:
            ntriv = 6

        if not use_hessian:
            if not ('frequencies' in molecule.__dict__ and 'normal_modes' in molecule[0].__dict__):
                print('Frequencies and normal modes not found, try to calculate them from Hessian matrix')
                if not 'hessian' in molecule.__dict__.keys():
                    stopper.stopMLatom('Hessian matrix not found -- cannot do wigner sampling')
                simulations.freq.freq_modified_from_TorchANI(molecule=molecule,normal_mode_normalization='mass deweighted normalized')
        else:
            if not 'hessian' in molecule.__dict__.keys():
                stopper.stopMLatom('Hessian matrix not found -- cannot do wigner sampling')
            simulations.freq.freq_modified_from_TorchANI(molecule=molecule,normal_mode_normalization='mass deweighted normalized')

        freq = np.array(molecule.frequencies)
        nm = np.zeros((3*Natoms,Natoms,3))

        # Remove normal modes with negative frequencies
        nnegative = 0
        while freq[nnegative] < 0:
            nnegative += 1 
        freq = freq[nnegative:]
        ntriv += nnegative
        if nnegative > 0:
            print(f"Wigner sampling: Removed {nnegative} negative frequencies")

        for itriv in range(ntriv):
            nm[itriv] = 1.0 / np.sqrt(3*Natoms)
        for imode in range(3*Natoms-ntriv):
            for iatom in range(Natoms):
                nm[ntriv+imode][iatom] = molecule[iatom].normal_modes[imode]

        numcoo = len(freq)
        atom_mass = [each.nuclear_mass for each in molecule.atoms]
        _, w_nmode = cls.nm2cart(nm,atom_mass)

        geomEq_au = np.array(geomEq) / constants.Bohr2Angstrom

        anq, amp, freq_au, atmau = cls.rdmol(atom_mass,numcoo,freq) 

        # Get Gaussian parameters (exponents and shifts)
        broad = np.ones((2,numcoo))
        shift = np.zeros((2,numcoo))
        if temperature != 0:
            broad[0,:] = np.tanh(constants.planck_constant*constants.speed_of_light*100.0*freq/(2.0*constants.kB*temperature))


        for ipoint in range(npoints):
            q,v = cls.inqp(geomEq_au,w_nmode[ntriv:],freq_au,anq,amp,numcoo,atmau,shift,broad)
            qlist.append(q)
            vlist.append(v)
        
        # Transform from a.u. to Angstrom & fs
        qlist = np.array(qlist) * constants.Bohr2Angstrom
        vlist = np.array(vlist) * constants.Bohr2Angstrom / constants.au2fs

        if reaction_coordinate_momentum and nnegative > 0:
            print("Adding velocity to reaction coordinate")
            reaction_nm = np.zeros((nnegative,Natoms,3))
            for ii in range(nnegative):
                for iatom in range(Natoms):
                    reaction_nm[ii][iatom] = molecule[iatom].normal_modes[ii]
            reaction_nm_masses = molecule.reduced_masses[:nnegative]
            for ii in range(nnegative):
                # Normalize the normal modes 
                reaction_nm[ii] = reaction_nm[ii] / np.sqrt(np.sum(reaction_nm[ii]**2))
                for ipoint in range(npoints):
                    scale = np.random.choice([-1,1])*np.sqrt(-2*constants.kB_in_Hartree*temperature*np.log(1-np.random.random())) / np.sqrt(reaction_nm_masses[ii] * constants.ram2au)/ constants.au2fs * constants.Bohr2Angstrom
                    vlist[ipoint] += scale * reaction_nm[ii]

        return qlist, vlist

    @classmethod 
    def nm2cart(cls,nms,atom_mass):
        Natoms = len(nms[0])
        cart_nms = np.copy(nms)
        
        for inm in range(len(nms)):
            # reduced mass
            reduced_amu = 0.0
            for iatom in range(Natoms):
                reduced_amu += np.sum(nms[inm][iatom]**2*atom_mass[iatom])
            # cartesian normal modes (au)
            for iatom in range(Natoms):
                cart_nms[inm][iatom] /= np.sqrt(reduced_amu*constants.ram2au)

        # Mass weighted normal modes:
        w_nmode = np.copy(cart_nms)
        for inm in range(len(nms)):
            for iatom in range(Natoms):
                w_nmode[inm][iatom] *= np.sqrt(atom_mass[iatom]*constants.ram2au)

        return cart_nms, w_nmode
        
    @classmethod 
    def herm2(cls,xarg,x,nn,cost):
        h = np.zeros(55)
        h[0] = 1.0 
        h[1] = xarg + xarg 
        if (nn>=2):
            for mm in range(1,nn+1):
                ff = xarg*h[mm] - (mm-1)*h[mm-1]
                h[mm+1] = ff+ff 
        ff = h[nn]
        herm1 = ff*cost*np.exp(-x*x*0.5)
        value = herm1*herm1 

        return value 

    @classmethod
    # NewtonX-2.4-B06 initcond/sample.f90  
    def sample_coor(cls,amp,ivib,quant,shift,broad):
        turnpo = np.sqrt(ivib*2.0+1.0)
        xmin = -turnpo - quant 
        xdel = -xmin - xmin 

        # Initial conditions in the vibrational ground state 
        if ivib == 0:
            while True:
                x1 = np.random.random() 
                xx = xmin + xdel*x1 
                fsval = np.exp(-xx*xx)
                x2 = np.random.random() 
                if (x2 < fsval):
                    break 
            xarg = xx / np.sqrt(broad) + shift 
            coor = xarg*amp 
            return coor 
        
        # Initial condition in vibrational excited states
        nstep = 50*ivib 
        dlt = 1.0/nstep
        fzb = -1000.0 
        k = 0 
        xarg = 0.0
        while (xarg < -xmin):
            k = k+1 
            xx = xmin+dlt*k
            xarg = xx / np.sqrt(broad) + shift 
            fz = cls.herm2(xarg,xx,ivib,1.0)
            if (fz >= fzb): fzb = fz 
        
        fmax = fzb 
        cost = np.sqrt(1.0/fmax)

        while True:
            x1 = np.random.random() 
            xx = xmin + xdel*x1
            xarg = xx / np.sqrt(broad) + shift 
            fsval = cls.herm2(xarg,xx,ivib,cost)
            x2 = np.random.random() 
            if (x2<= fsval): break 

        coor = xarg*amp 
        return coor

    @classmethod
    # NewtonX-2.4-B06 initcond/initqp.f90 
    # !! the len of all normal mode related array is numcoo
    def rdmol(cls,atom_mass,numcoo,freq):
        eps = 1.0E-15
        atmau = np.array(atom_mass)*constants.ram2au#*1822.888515
        Natoms = len(atom_mass)
        # numcoo = 3*Natoms - 6 + linear 
        anq = np.zeros(numcoo) # vibrational quantum numbers (all zero)

        freq_au = np.array(freq) * 4.55633539E-6 # cm 2 au

        # factor used for quantum distribution
        amp = np.zeros(3*Natoms)
        for i in range(numcoo):
            amp[i] = 1.0 / np.sqrt(freq_au[i])

        return anq, amp, freq_au, atmau
    
    @classmethod
    # NewtonX-2.4-B06 initcond/inqp.f90 
    def inqp(cls,geomEq,cn,w,anq,amp,numcoo,atmau,shift,broad):
        cint = []
        dcint = []
        dum0 = 0.0
        dum1 = 0.0
        Natoms = len(atmau)
        samp_points = np.zeros((2,numcoo))
        q = np.array(geomEq) # size: (Natoms,3)
        v = np.zeros((Natoms,3))
        for i in range(numcoo):
            wwai = amp[i]*w[i]
            ivb = int(abs(anq[i])+0.5)
            quant = 4.0
            cint.append(cls.sample_coor(amp[i],ivb,quant,shift[0,i],broad[0,i]))
            dcint.append(cls.sample_coor(wwai,ivb,quant,shift[1,i],broad[1,i]))
            enmi = w[i]*(ivb+0.5)         # Vibrational energies
            epot = 0.5*(w[i]*cint[i])**2  # Potential energies
            samp_points[0][i] = cint[i]/amp[i]
            samp_points[1][i] = dcint[i]/wwai 

            dum0 += epot 
            dum1 += enmi
        
        for iatom in range(Natoms):
            for icoord in range(3):
                fac = 1.0/np.sqrt(atmau[iatom])
                for i in range(numcoo):
                    q[iatom][icoord] += cn[i][iatom][icoord]*cint[i]*fac 
                    v[iatom][icoord] += cn[i][iatom][icoord]*dcint[i]*fac 
        return q,v # Unit: a.u.

class harmonic_quantum_Boltzmann_sampling():
    #
    # The harmonic quantum Boltzmann sampling in this class follows what is shown in the following paper (it is called thermal sampling in VENUS manual):
    # J. Phys. Chem. A 1998, 102, 3648-3658
    #
    # The quanta ni of each normal mode is first sampled from the harmonic quantum Boltzman distribution function:
    #
    #   p(ni) = exp(-ni * h * vi / kB / T) * (1 - (exp(-h * vi / kB / T))) 
    #
    # where vi is the frequency of normal mode i, h is the Planck constant, kB is the Boltzmann constant and T is the temperature
    # 
    # The energy of normal mode i is calculated as Ei = (ni + 0.5) * h * vi
    #
    # The mass weighted normal mode coordinates Qi and momenta Pi are
    #
    #   Qi = Ai * cos(2 * pi * Ri)
    #   Pi = -Ai * wi * sin(2 * pi * Ri)
    #
    # where wi = 2 * pi * vi, Ai = sqrt(2 * Ei) / wi and Ri is a uniform random number on [0,1]
    #
    # The mass weighted momentum Prc of reaction coordinate is chosed from a thermal distribution 
    # 
    #   Prc = ± sqrt(-2 * kB * T * ln(1 - R))
    #
    # where R is a uniform random number on [0,1]
    #

    def __init__(self):
        pass 

    @classmethod 
    def sample(cls,npoints,molecule,temperature,use_hessian):
        if not use_hessian:
            if not ('frequencies' in molecule.__dict__ and 'normal_modes' in molecule[0].__dict__):
                print('Frequencies and normal modes not found, try to calculate them from Hessian matrix')
                if not 'hessian' in molecule.__dict__.keys():
                    stopper.stopMLatom('Hessian matrix not found -- cannot do wigner sampling')
                simulations.freq.freq_modified_from_TorchANI(molecule=molecule,normal_mode_normalization='mass deweighted normalized')
        else:
            if not 'hessian' in molecule.__dict__.keys():
                stopper.stopMLatom('Hessian matrix not found -- cannot do wigner sampling')
            simulations.freq.freq_modified_from_TorchANI(molecule=molecule,normal_mode_normalization='mass deweighted normalized')
        freq = molecule.frequencies 
        nm_masses = molecule.reduced_masses
        Natoms = len(molecule)
        
        nnegative = 0 
        while freq[nnegative] < 0:
            nnegative += 1 
        # freq = freq[nnegative:]
        nm = np.zeros((len(freq),Natoms,3))

        for imode in range(len(freq)):
            for iatom in range(Natoms):
                nm[imode][iatom] = molecule[iatom].normal_modes[imode]
            nm[imode] /= np.sqrt(np.sum(nm[imode]**2))

        q_list = np.array([molecule.xyz_coordinates]*npoints)
        v_list = np.zeros((npoints,Natoms,3))

        for imode in range(nnegative,len(freq)):
            qq,vv = cls.get_vq(temperature,freq[imode],nm_masses[imode],npoints)
            for isample in range(npoints):
                # print(isample)
                q_list[isample] += nm[imode]*qq[isample]
                # print(nm[imode]*vv[isample])
                v_list[isample] += nm[imode]*vv[isample]

        # Deal with reaction coordinate
        for ii in range(nnegative):
            for isample in range(npoints):
                scale = np.random.choice([-1,1])*np.sqrt(-2*constants.kB_in_Hartree*temperature*np.log(1-np.random.random())) / np.sqrt(nm_masses[ii] * constants.ram2au)/ constants.au2fs * constants.Bohr2Angstrom
                v_list[isample] += scale * nm[ii]

        init_cond_db = data.molecular_database() 
        for isample in range(npoints):
            new_molecule = molecule.copy(atomic_labels=[],molecular_labels=[])
            for iatom in range(Natoms):
                new_molecule.atoms[iatom].xyz_coordinates = q_list[isample][iatom]
                new_molecule.atoms[iatom].xyz_velocities = v_list[isample][iatom]
            init_cond_db.molecules.append(new_molecule)

        return init_cond_db
    
    @classmethod 
    def sample_quanta(cls,temperature,freq,nsample):
        hvkT = constants.planck_constant*freq*constants.speed_of_light*100/constants.kB/temperature
        
        nn_len = 100
        nn = np.array([ii for ii in range(nn_len)])
        pp = np.array([np.exp(-ii*hvkT)*(1-np.exp(-hvkT)) for ii in nn])
        while 1.0-np.sum(pp) > 1e-10:
            nn_len += 50
            nn = np.array([ii for ii in range(nn_len)])
            pp = np.array([np.exp(-ii*hvkT)*(1-np.exp(-hvkT)) for ii in nn])
        rand = np.random.choice(nn,size=nsample,p=pp)
        return rand
    
    @classmethod 
    def get_energy(cls,rand,freq):
        return (rand+0.5)*constants.planck_constant*freq*constants.speed_of_light*100/1000*constants.Avogadro_constant*constants.kJpermol2Hartree # Hartree
    
    @classmethod 
    def get_vq(cls,temperature,freq,mass,nsample):
        energy = cls.get_energy(cls.sample_quanta(temperature,freq,nsample),freq)
        rand = np.random.random(len(energy))
        omega = freq * constants.speed_of_light*100*2*np.pi / 1.0E15 / constants.fs2au # au
        AA = np.sqrt(2*energy)/omega # au
        qq = AA * np.cos(2*np.pi*rand) / np.sqrt(mass*constants.ram2au) * constants.Bohr2Angstrom # Angstrom
        vv = -omega * AA * np.sin(2*np.pi*rand) / np.sqrt(mass*constants.ram2au) * constants.Bohr2Angstrom / constants.au2fs # Angstrom/fs 
        return qq,vv

# def wignersample(npoints,molecule):
#     qlist = [] 
#     vlist = []
#     geomEq = np.array([each.xyz_coordinates for each in molecule.atoms])
#     mass = molecule.get_nuclear_masses() 
#     mass = mass.reshape(1,len(mass))
#     # Calculate normal modes from Hessian matrix 
#     #nm,freq,ele,linear_int = readGaussianNM(nmfile)
#     if not 'hessian' in molecule.__dict__.keys():
#         stopper.stopMLatom('Hessian matrix not found -- cannot do wigner sampling')

#     linear = molecule.is_it_linear()
#     Natoms = len(molecule.atoms)
#     if linear:
#         ntriv = 5 
#         linear_int = 1
#     else:
#         ntriv = 6
#         linear_int = 0

#     # freq,nm,_,_ = vibrational_analysis(mass,molecule.hessian,mode_type='MDU')
#     simulations.freq.freq_modified_from_TorchANI(molecule=molecule,normal_mode_normalization='mass deweighted unnormalized')
#     freq = molecule.frequencies 
#     nm = np.zeros(3*Natoms,Natoms,3)
#     for itriv in range(ntriv):
#         nm[itriv] = 1.0 / np.sqrt(3*Natoms)
#     for imode in range(ntriv,3*Natoms):
#         for iatom in range(Natoms):
#             nm[ntriv+imode][iatom] = molecule[iatom].normal_modes[imode]

#     numcoo = len(freq)
#     #print(len(nm),len(freq),len(ele),linear)
#     atom_mass = [each.nuclear_mass for each in molecule.atoms]
#     cart_nms, w_nmode = nm2cart(nm,atom_mass)

#     geomEq_au = np.array(geomEq) / constants.Bohr2Angstrom

#     anq, amp, freq_au, atmau = rdmol(atom_mass,linear_int,freq) 

#     for ipoint in range(npoints):
#         q,v = inqp(geomEq_au,w_nmode[ntriv:],freq_au,anq,amp,numcoo,atmau)
#         qlist.append(q)
#         vlist.append(v)
    
#     # Transform from a.u. to Angstrom & fs
#     qlist = np.array(qlist) * constants.Bohr2Angstrom
#     vlist = np.array(vlist) * constants.Bohr2Angstrom / constants.au2fs

#     return qlist, vlist, linear_int
    
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

