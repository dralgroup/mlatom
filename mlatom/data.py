#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! data: Module for working with data                                        ! 
  ! Implementations by: Pavlo O. Dral, Fuchun Ge,                             !
  !                     Shuang Zhang, Yi-Fan Hou, Yanchi Ou                   !
  !---------------------------------------------------------------------------! 
'''

from __future__ import annotations
from typing import Any, Union, Dict, List, Optional, Iterable
import uuid, copy, os, json
import numpy as np
import h5py
import functools
from . import constants
from . import conversions

periodic_table = """ X
  H                                                                                                                           He
  Li  Be                                                                                                  B   C   N   O   F   Ne
  Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
  K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
  Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
  Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
  Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
""".strip().split()
atomic_number2element_symbol = {k: v for k, v in enumerate(periodic_table)}
element_symbol2atomic_number = {v: k for k,
                                v in atomic_number2element_symbol.items()}

class atom:
    '''
    Create an atom object.
    
    Arguments:
        nuclear_charge (int, optional): provide the nuclear charge to define the atom.
        atomic_number (int, optional): provide the atomic number to define the atom.
        element_symbol (int, optional): provide the element symbol to define the atom.
        nuclear_mass (int, optional): provide the nuclear mass to define the atom.
        xyz_coordinates (Array-like, optional): specify the location of atom in Cartesian coordinates.

    '''
    #xyz_coordinates = []  # list [x, y, z] with float numbers. Expected units: Angstrom

    def __init__(self, nuclear_charge: Union[int, None] = None, atomic_number: Union[int, None] = None, element_symbol: Union[str, None] = None, nuclear_mass: Union[float, None] = None, xyz_coordinates: Union[np.ndarray, List, None] = None):
        if nuclear_charge != None:
            self.nuclear_charge = nuclear_charge
            self.atomic_number = int(self.nuclear_charge)
            self.element_symbol = atomic_number2element_symbol[self.atomic_number]
        elif atomic_number != None:
            self.nuclear_charge = atomic_number
            self.atomic_number = self.nuclear_charge
            self.element_symbol = atomic_number2element_symbol[self.atomic_number]
        elif element_symbol != None:
            self.element_symbol = element_symbol
            self.atomic_number = element_symbol2atomic_number[self.element_symbol]
            self.nuclear_charge = self.atomic_number

        # Detect the correct isotope
        if nuclear_mass != None:
            most_similar_isotope = isotopes.get_most_similar_isotope_given_nuclear_charge_and_mass(self.nuclear_charge, nuclear_mass)
            for key in most_similar_isotope.__dict__.keys():
                self.__dict__[key] = most_similar_isotope.__dict__[key]
            self.nuclear_mass = nuclear_mass
        elif 'nuclear_charge' in self.__dict__:
            if self.nuclear_charge > 0:
                most_abundant_isotope = isotopes.get_most_abundant_with_given_nuclear_charge(
                    self.nuclear_charge)
                for key in most_abundant_isotope.__dict__.keys():
                    self.__dict__[key] = most_abundant_isotope.__dict__[key]
                self.nuclear_mass = self.relative_isotopic_mass

        if type(xyz_coordinates) != type(None):
            self.xyz_coordinates = xyz_coordinates

    def copy(self, atomic_labels=None) -> atom:
        '''
        Return a copy of the current atom object.
        '''
        if type(atomic_labels) == type(None):
            atomic_labels = []
        new_atom = atom(element_symbol=self.element_symbol)
        new_atom.nuclear_mass = self.nuclear_mass
        for each_label in atomic_labels:
            if each_label in self.__dict__:
                new_atom.__dict__[each_label] = np.copy(self.__dict__[each_label])

        return new_atom


def load_return_molecule(filename=None, format='json'):
    if format.casefold() == 'json'.casefold():
        jsonfile = open(filename, 'r')
        moldict = json.load(jsonfile)
        newmol = dict_to_molecule_class_instance(moldict)
    return newmol

def load_molecule(molobj, filename=None, format='json'):
    newmol = load_return_molecule(filename=filename, format=format)
    molobj.__dict__.update(newmol.__dict__)

class load_molecule_cls():
    def __get__(self, obj, objtype=None):
        if obj is not None:
            @functools.wraps(load_molecule)
            def _wrapperobj(*args, **kwargs):
                return load_molecule(obj, *args, **kwargs)
            return _wrapperobj
        else:
            return load_return_molecule

class molecule:
    '''
    Create a molecule object.

    Arguments:
        charge (float, optional): Specify the charge of the molecule.
        multiplicity (int, optional): Specify the multiplicity of the molecule.
        atoms (List[:class:`atom`], optional): Specify the atoms in the molecule.

    Examples:

        Select an atom inside with subscription:

        .. code-block:: python
           
           from mlatom.data import atom, molecule
           at = atom(element_symbol = 'C')
           mol = molecule(atoms = [at])
           print(id(at), id(mol[0]))

    Attributes:
        id: The unique ID for this molecule.
        charge: The electric charge of the molecule.
        multiplicity: The multiplicity of the molecule.
    
    load(filename: stringe, format: string):
        Load a molecule object from a dumped file.
        
        Updates a molecule object if initialized:
        
            ``mol = molecule(); mol.load(filename='mymol.json')``
        Returns a molecule object if called as class method:
        
            ``mol = molecule.load(filename='mymol.json')``
            
        Arguments:
            filename (str): filename or path
            
            format (str, optional): currently, only 'json' format is supported.
    '''
    load = load_molecule_cls()
    def __init__(self, charge: int = 0, multiplicity: int = 1, atoms: List[atom] = None, pbc: Optional[Union[np.ndarray, bool]] = None, cell: Optional[np.ndarray] = None): 
        self.id = str(uuid.uuid4())
        self.charge = charge
        self.multiplicity = multiplicity
        self.pbc = pbc
        self.cell = cell
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        
        self.electronic_states = []
    
    @property
    def pbc(self):
        '''
        The periodic boundary conditions of the molecule. Setting it with ``mol.pbc = True`` is equal to ``mol.pbc = [True, True, True]``.
        '''
        return self._pbc
    
    @pbc.setter
    def pbc(self, pbc):
        if pbc is not None:
            if isinstance(pbc, bool):
                pbc = [pbc] * 3
            pbc = np.array(pbc, bool)
            assert pbc.shape == (3,), 'please provide a valid pbc'
        self._pbc = pbc
    
    @property
    def cell(self):
        '''
        The matrix of 3 vectors that defines the unicell. The setter of it simply wraps `ase.geometry.cell.cellpar_to_cell() <https://wiki.fysik.dtu.dk/ase/ase/geometry.html#ase.geometry.cellpar_to_cell>`_.
        '''
        return self._cell
    
    @cell.setter
    def cell(self, cell):
        # reinvent the wheel with premade spokes from ASE
        if cell is not None:
            from ase.geometry.cell import cellpar_to_cell 
            cell = cellpar_to_cell(cell)
        self._cell = cell
        
    @property 
    def cell_coordinates(self) -> np.ndarray:
        '''
        The relative coordinates in the cell.
        '''
        assert self.cell is not None, 'make sure this molecule has a valid cell'
        inverse_cell = np.linalg.inv(self.cell)
        return self.xyz_coordinates @ inverse_cell
        
    @cell_coordinates.setter
    def cell_coordinates(self, value):
        assert self.cell is not None, 'make sure this molecule has a valid cell'
        self.xyz_coordinates = value @ self.cell
            
    def map_to_unicell(self):
        '''
        Map all atoms outside the unicell into it.
        '''
        if self.pbc is not None:
            self.cell_coordinates -= np.floor(self.cell_coordinates) * self.pbc
        

    def read_from_xyz_file(self, filename: str, format: Union[str, None] = None) -> molecule:
        '''
        Load molecular geometry from XYZ file.

        Data in standard XYZ format can be read if parameter ``format`` is not specified.

        Extra formats supoorted are:

            - ``'COLUMBUS'``

            - ``'NEWTON-X'`` or ``'NX'``

            - ``'turbomol'``
            
        Arguments:
            filename (str): The name of the file to be read.
            format (str, optional): The format of the file.
        '''
        fxyz = open(filename, 'r')
        string = fxyz.read()
        fxyz.close()
        return self.read_from_xyz_string(string, format=format)

    def read_from_xyz_string(self, string: str = None, format: Union[str, None] = None) -> molecule:
        '''
        Load molecular geometry from XYZ string.

        Data in standard XYZ format can be read if parameter ``format`` is not specified.

        Extra formats supoorted are:

            - ``'COLUMBUS'``

            - ``'NEWTON-X'`` or ``'NX'``

            - ``'turbomol'``
            
        Arguments:
            string (str): The string input.
            format (str, optional): The format of the string.
        '''
        self.atoms = []
        fxyz = string.split('\n')
        if format == None:
            nlines = 0
            natoms = 0
            for line in fxyz:
                nlines += 1
                if nlines == 1:
                    natoms = int(line)
                elif nlines == 2:
                    if line.strip() != '':
                        self.comment = line.strip()
                elif nlines > 2 and nlines <= 2 + natoms:
                    self.add_atom_from_xyz_string(line)
                    if nlines == 2 + natoms:
                        break
        elif format.casefold() in ['COLUMBUS'.casefold(), 'NEWTON-X'.casefold(), 'NX'.casefold()]:
            for line in fxyz:
                yy = line.split()
                coords = array([float(xx)*constants.Bohr2Angstrom for xx in yy[2:5]]).astype(float)
                self.atoms.append(atom(element_symbol=yy[0][0].upper() + yy[0][1:].lower(),
                                    nuclear_charge=float(yy[1]),
                                    xyz_coordinates=coords,
                                    nuclear_mass=float(yy[-1])))
        elif format.casefold() == 'turbomole':
            for line in fxyz:
                yy = line.split()
                if len(yy) != 4:
                    continue 
                else:
                    coords = array([float(xx)*constants.Bohr2Angstrom for xx in yy[0:3]]).astype(float)                    
                    self.atoms.append(atom(element_symbol=yy[3].capitalize(),
                                      xyz_coordinates=coords))
        return self

    def read_from_numpy(self, coordinates: np.ndarray, species: np.ndarray) -> molecule:
        '''
        Load molecular geometry from a numpy array of coordinates and another one for species.

        The shape of the input ``coordinates`` should be ``(N, 3)``, while ``(N,)`` for the input ``species``.

        Where the ``N`` is the number of atoms.
        '''
        self.atoms = []
        for i in range(coordinates.shape[0]):
            if np.issubdtype(species[i].dtype, np.integer):
                self.atoms.append(atom(atomic_number=species[i], xyz_coordinates=coordinates[i]))
            else:
                self.atoms.append(atom(element_symbol=species[i], xyz_coordinates=coordinates[i]))
        return self
    
    def read_from_smiles_string(self, smi_string: str) -> molecule:
        '''
        Generate molecular geometry from a SMILES string provided.

        The geometry will be generated and optimized with `Pybel's <https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_Pybel.html>`_ ``make3D()`` method.
        '''
        xyz_string = conversions.smi2xyz(smi_string)
        self.read_from_xyz_string(xyz_string)
        return self
    
    @classmethod
    def from_xyz_file(cls, filename: str, format: Union[str, None] = None) -> molecule:
        '''
        Classmethod wrapper for :meth:`molecule.read_from_xyz_file`, returns a :class:`molecule` object.
        '''
        return cls().read_from_xyz_file(filename, format=format)
    
    @classmethod
    def from_xyz_string(cls, string: str = None, format: Union[str, None] = None) -> molecule:
        '''
        Classmethod wrapper for :meth:`molecule.read_from_xyz_string`, returns a :class:`molecule` object.
        '''
        return cls().read_from_xyz_string(string, format=format)
    
    @classmethod
    def from_numpy(cls, coordinates: np.ndarray, species: np.ndarray) -> molecule:
        '''
        Classmethod wrapper for :meth:`molecule.read_from_numpy`, returns a :class:`molecule` object.
        '''
        return cls().read_from_numpy(coordinates, species)
    
    @classmethod
    def from_smiles_string(cls, smi_string: str) -> molecule:
        '''
        Classmethod wrapper for :meth:`molecule.read_from_smiles_string`, returns a :class:`molecule` object.
        '''
        return cls().read_from_smiles_string(smi_string)

    def add_atom_from_xyz_string(self, line: str) -> None:
        '''
        Add an atom to molecule from a string in XYZ format
        '''
        yy = line.split()
        coords = array([float(xx) for xx in yy[1:4]]).astype(float)
        if yy[0].isnumeric():
            self.atoms.append(
                atom(atomic_number=int(yy[0]), xyz_coordinates=coords))
        else:
            self.atoms.append(atom(element_symbol=yy[0][0].upper(
            ) + yy[0][1:].lower(), xyz_coordinates=coords))

    def add_scalar_property(self, scalar, property_name: str = 'y') -> None: # kind of redundant? mol.a = x does the samething
        '''
        Add a scalar property to the molecule. So the property can be called by molecule.<property_name>.

        Arguments:
            scalar: The scalar to be added.
            property_name (str, optional): The name assign to the scalar property.
        '''
        self.__dict__[property_name] = scalar

    def add_xyz_derivative_property(self, derivative, property_name: str = 'y', xyz_derivative_property: str = 'xyz_derivatives') -> None:
        '''
        Add a XYZ derivative property to the molecule.

        Arguments:
            derivative: The derivative property to be added.
            property_name (str, optional): The name of the associated non-derivative property.
            xyz_derivative_property (str, optional): the name assign to the derivative property.
            
        '''
        if not 'properties_and_their_derivatives' in self.__dict__.keys():
            self.properties_and_their_derivatives = {}
        self.properties_and_their_derivatives[property_name] = xyz_derivative_property
        self.add_xyz_vectorial_property(
            vector=derivative, xyz_vectorial_property=xyz_derivative_property)
    
    def add_hessian_property(self, hessian, hessian_propety='hessian'):
        self.add_scalar_property(hessian[:len(self)*3,:len(self)*3], property_name=hessian_propety)
    
    def add_xyz_vectorial_property(self, vector, xyz_vectorial_property: str = 'xyz_vector') -> None:
        '''
        Add a XYZ vectorial property to the molecule.

        Arguments:
            vector: The vector to be added.
            xyz_vectorial_property (str, optional): the name assign to the vectorial property.
            
        '''
        for j in range(len(self)):
            self.atoms[j].__dict__[xyz_vectorial_property] = vector[j]

    def write_file_with_xyz_coordinates(self, filename: str, format: Union[str, None] = None) -> None:
        '''
        Write the molecular geometry data into a file.
        Data in standard XYZ format can be read if parameter ``format`` is not specified.

        Extra formats supoorted are:

            - ``'COLUMBUS'``

            - ``'NEWTON-X'`` or ``'NX'``

            - ``'turbomol'``

        Arguments:
            filename (str): The name of the file to be written.
            format (str, optional): The format of the file.
        '''
        with open(filename, 'w') as fw:
            if format == None:
                fw.writelines('%d\n' % len(self.atoms))
                if 'comment' in self.__dict__.keys():
                    fw.writelines(f'{self.comment}\n')
                else:
                    fw.writelines('\n')
                for atom in self.atoms:
                    fw.writelines('%-3s %25.13f %25.13f %25.13f\n' % (atom.element_symbol,
                                atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2]))
            elif format.casefold() in ['COLUMBUS'.casefold(), 'NEWTON-X'.casefold(), 'NX'.casefold()]:
                for atom in self.atoms:
                    fw.writelines('%2s%8.1f%14.8f%14.8f%14.8f%14.8f\n' % (atom.element_symbol,
                                                                      atom.nuclear_charge,
                                                                      atom.xyz_coordinates[0] * constants.Angstrom2Bohr,
                                                                      atom.xyz_coordinates[1] * constants.Angstrom2Bohr,
                                                                      atom.xyz_coordinates[2] * constants.Angstrom2Bohr,
                                                                      atom.nuclear_mass))
            elif format.casefold() in ['TURBOMOLE'.casefold()]:
                fw.writelines('$coord\n')
                for atom in self.atoms:
                    fw.writelines('%25.13f %25.13f %25.13f %-3s \n' % (atom.xyz_coordinates[0] * constants.Angstrom2Bohr, atom.xyz_coordinates[1] * constants.Angstrom2Bohr, atom.xyz_coordinates[2] * constants.Angstrom2Bohr, atom.element_symbol))
                fw.writelines('$user-defined bonds\n')
                fw.writelines('$end\n')


    def get_xyz_string(self) -> str:
        '''
        Return the molecular geometry in a string of XYZ format.
        '''
        xyz_string = ''
        xyz_string += '%d\n' % len(self.atoms)
        if 'comment' in self.__dict__.keys():
                xyz_string += f'{self.comment}\n'
        else:
            xyz_string += '\n'
        for atom in self.atoms:
            xyz_string += '%-3s %25.13f %25.13f %25.13f\n' % (atom.element_symbol,
                atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2])
        return xyz_string

    def get_atomic_numbers(self) -> np.ndarray:
        return array([atom.atomic_number for atom in self.atoms]).astype(int)
    
    @property
    def atomic_numbers(self) -> np.ndarray:
        '''
        The atomic numbers of the atoms in the molecule.
        '''
        return self.get_atomic_numbers()

    def get_element_symbols(self) -> np.ndarray:
        return array([atom.element_symbol for atom in self.atoms])
    
    @property
    def element_symbols(self) -> np.ndarray:
        '''
        The element symbols of the atoms in the molecule.
        '''
        return self.get_element_symbols()
    
    @property
    def smiles(self) -> str:
        '''
        The SMILES representation of the molecule.
        '''
        return conversions.xyz2smi(self.get_xyz_string())
        
    def get_xyz_coordinates(self):
        return self.get_xyz_vectorial_properties('xyz_coordinates')
    
    @property 
    def xyz_coordinates(self) -> np.ndarray:
        '''
        The XYZ geometry of the molecule.
        '''
        return self.get_xyz_vectorial_properties('xyz_coordinates')
    
    @xyz_coordinates.setter
    def xyz_coordinates(self,value):
        for iatom in range(len(self.atoms)):
            self.atoms[iatom].xyz_coordinates = np.copy(value[iatom])

    def get_energy_gradients(self):
        return self.get_xyz_vectorial_properties('energy_gradients')
    
    @property
    def energy_gradients(self):
        return self.get_energy_gradients()
    
    @energy_gradients.setter
    def energy_gradients(self, value):
        self.add_xyz_derivative_property(value, property_name='energy', xyz_derivative_property='energy_gradients')

    def get_number_of_atoms(self):
        return len(self)
    
    def get_property(self, property_name):
        if property_name in self.__dict__:
            return self.__dict__[property_name] 
        elif property_name in self.__dir__() and isinstance(getattr(self.__class__, property_name), property):
            return getattr(self, property_name)
        else: 
            return np.nan

    def set_property(self, **kwargs):
        for property_name, value in kwargs.items():
            setattr(self, property_name, value)

    def get_xyz_vectorial_properties(self, property_name):
        vectorial_properties = []
        for atom in self.atoms: 
            vectorial_properties.append(atom.__dict__[property_name] if property_name in atom.__dict__ else np.full(3, np.nan))
        return array(np.copy(vectorial_properties)).astype(float)

    def get_nuclear_masses(self):
        return array([atom.nuclear_mass for atom in self.atoms])

    @property
    def nuclear_masses(self):
        return self.get_nuclear_masses()
    
    def calculate_kinetic_energy(self):
        velocity = np.copy(self.get_xyz_vectorial_properties('xyz_velocities'))
        Natoms = len(self.atoms)
        masses = self.nuclear_masses
        mass = masses.reshape(Natoms,1)
        return np.sum(velocity**2 * mass) / 2.0 * constants.ram2au * (constants.au2fs / constants.Bohr2Angstrom)**2 #
    
    @property 
    def kinetic_energy(self) -> float:
        '''
        Give out the kinetic energy (A.U.) based on the xyz_velocities.
        '''
        return self.calculate_kinetic_energy()

    def rescale_velocities(self, kinetic_energy_change=None, if_not_enough_kinetic_energy='zero velocities'):
        initial_kinetic_energy = self.kinetic_energy
        target_kinetic_energy = initial_kinetic_energy + kinetic_energy_change
        if target_kinetic_energy < 0:
            if if_not_enough_kinetic_energy == 'zero velocities':
                factor = 0.0
            elif if_not_enough_kinetic_energy == 'do not change velocities':
                factor = 1.0
            elif if_not_enough_kinetic_energy == 'raise error':
                raise ValueError('Not enough kinetic energy to rescale velocities to obtain the requested change in energy')
        else:
            factor = (target_kinetic_energy/initial_kinetic_energy)**0.5
        for atom in self.atoms:
            atom.xyz_velocities *= factor
    
    def update_xyz_vectorial_properties(self, property_name, vectorial_properties):
        for iatom in range(len(self.atoms)):
            self.atoms[iatom].__dict__[property_name] = vectorial_properties[iatom]

    def copy(self, atomic_labels=None, molecular_labels=None):
        '''
        Return a copy of current molecule object.
        '''
        if type(atomic_labels) != type(None) or type(molecular_labels) != type(None):
            new_molecule = molecule()
            new_molecule.multiplicity = self.multiplicity
            new_molecule.charge = self.charge
            if type(molecular_labels) != type(None):
                for each_label in molecular_labels:
                    if each_label in self.__dict__:
                        new_molecule.__dict__[each_label] = self.__dict__[each_label]
            else:
                for each_label in self.__dict__.keys():
                    if each_label == 'atoms': continue
                    new_molecule.__dict__[each_label] = self.__dict__[each_label]
            if type(atomic_labels) != type(None):
                for iatom in range(len(self.atoms)):
                    new_atom = self.atoms[iatom].copy(atomic_labels=atomic_labels)
                    new_molecule.atoms.append(new_atom)
        else:
            new_molecule = copy.deepcopy(self)
        new_molecule.id = str(uuid.uuid4())
        return new_molecule
    
    def proliferate(
            self, 
            shifts: Optional[Iterable] = None, 
            XYZshifts: Optional[Iterable] = None, 
            Xshifts: Optional[Iterable] = [0], 
            Yshifts: Optional[Iterable] = [0], 
            Zshifts: Optional[Iterable] = [0],
            PBC_constrained: bool = True,
        ) -> molecule: 
        '''
        Proliferate the unicell by specified shifts along cell vectors (called X/Y/Z here). 
        
        Returns a new :class:`molecule` object.
        
        Arguments:
            shifts (Iterable, optional): The list of shifts to perform. Each shift should be a 3D vector that indicates the coefficient applies to the corresponding cell vector.
            XYZshifts (Iterable, optional): Generate all possible shifts with given shift coefficients in all three directions when a list is specified. When a list of 3 lists is specified, it's equal to setting X/Y/Zshifts
            Xshifts (Iterable, optional): Specify all possible shift coefficients in the direction of the first cell vector.
            Yshifts (Iterable, optional): Specify all possible shift coefficients in the direction of the second cell vector.
            Zshifts (Iterable, optional): Specify all possible shift coefficients in the direction of the third cell vector.
            PBC_constrained (bool): Controls whether the shifts in some directions are disabled where corresponding PBC is false. Only applies to XYZshifts.
            
        .. note::
            
           Priorities for different types of shifts:
                ``shifts`` > ``XYZshifts`` > ``X/Y/Zshifts`` 
        
        Examples:

            Single H atom in the centre of a cubic cell (2x2x2):

            .. code-block:: python
                
                mol = ml.molecule.from_numpy(np.ones((1, 3)), np.array([1])) 
                mol.pbc = True 
                mol.cell = 2 
            
            Proliferate to get two periods in all three directions,
            with shifts:

            .. code-block:: python
                
                new_mol = mol.proliferate(
                    shifts = [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                )
            
            with XYZshifts:

            .. code-block:: python
                
                new_mol = mol.proliferate(XYZshifts=range(2))
                # or
                new_mol = mol.proliferate(XYZshifts=[range(2)]*3)
                
                
            with X/Y/Zshifts:

            .. code-block:: python
                
                new_mol = mol.proliferate(Xshifts=range(2), Yshifts=(0, 1), Zshifts=[0, 1]))
            
            All scripts above will make ``new_mol.xyz_coordinates`` be:
            
            .. code-block:: python
            
                array([[1., 1., 1.],
                       [3., 1., 1.],
                       [1., 3., 1.],
                       [3., 3., 1.],
                       [1., 1., 3.],
                       [3., 1., 3.],
                       [1., 3., 3.],
                       [3., 3., 3.]])
                
        '''
        if shifts is None:
            if XYZshifts is not None:
                XYZshifts = np.array(XYZshifts)
                if XYZshifts.ndim == 1:
                    XYZshifts = np.repeat(XYZshifts[np.newaxis], 3, 0)
                assert XYZshifts.ndim ==2, 'provide valid XYZshifts'
                Xshifts, Yshifts, Zshifts = XYZshifts
                if PBC_constrained:
                    Xshifts = Xshifts if self.pbc[0] else [0]
                    Yshifts = Yshifts if self.pbc[1] else [0]
                    Zshifts = Zshifts if self.pbc[2] else [0]
            shifts = [np.array([i, j, k]) for k in Zshifts for j in Yshifts  for i in Xshifts]
            
        xyzs = []
        for shift in shifts:
            xyzs.append(self.xyz_coordinates + shift @ self.cell)
        return self.from_numpy(np.concatenate(xyzs, 0), np.tile(self.atomic_numbers, len(shifts)))
    
    def dump(self, filename=None, format='json'):
        '''
        Dump the current molecule object into a file. Only in json format, which is supported now.
        '''
        if format.casefold() == 'json'.casefold():
            jsonfile = open(filename, 'w')
            json.dump(class_instance_to_dict(self), jsonfile, indent=4)
            jsonfile.close()

        if format.casefold() == 'gaussian'.casefold():
            write_gaussian_log(self, filename)
    
    def get_internuclear_distance_matrix(self):
        natoms = len(self.atoms)
        distmat = np.zeros((natoms, natoms))
        for iatomind in range(natoms):
            for jatomind in range(iatomind+1,natoms):
                distmat[iatomind][jatomind] = self.internuclear_distance(iatomind, jatomind)
                distmat[jatomind][iatomind] = distmat[iatomind][jatomind]
        return distmat
    
    def get_bonds(self):
        bonds = []
        natoms = len(self.atoms)
        for iatomind in range(natoms):
            for jatomind in range(iatomind+1,natoms):
                aa = self.atoms[iatomind]
                bb = self.atoms[jatomind]
                dist = self.internuclear_distance(iatomind, jatomind)
                an = aa.atomic_number ; bn = bb.atomic_number
                if an == 1 or bn == 1:
                    if dist < 1.2:
                        bonds.append([iatomind, jatomind])
                if (an > 1 and an < 10) and (bn > 1 and bn < 10):
                    if dist < 2.0:
                        bonds.append([iatomind, jatomind])
        return bonds
    
    def internuclear_distance(self, atom1_index, atom2_index):
        aa = self.atoms[atom1_index]
        bb = self.atoms[atom2_index]
        return np.sqrt(np.sum(np.square(aa.xyz_coordinates-bb.xyz_coordinates)))
    
    def is_it_linear(self):
        eps = 1.0E-8
        coord = self.xyz_coordinates
        if len(coord) == 2:
            return True 
        else:
            vec1 = coord[1] - coord[0]
            for ii in range(2,len(coord)):
                vec2 = coord[ii] - coord[0]
                nv = np.cross(vec1,vec2)
                if np.sum(nv**2) > eps: return False
            return True

    def rotate(self, axis=None, angle=None, pivot=np.zeros(3), matrix=None):
        if not matrix:
            matrix = np.eye(3)
        self.xyz_coordinates = self.xyz_coordinates.dot(matrix)

    def translate(self, vector):
        self.xyz_coordinates = self.xyz_coordinates + vector

    def scale(self, factor, pivot=np.zeros(3)):
        self.xyz_coordinates = (self.xyz_coordinates - pivot) * factor + pivot

    def align(self, ref, pivot='CoM'):
        pass
    
    def info(self, properties = 'all', return_string=False):
        printstrs = []
        printstrs += [f" Molecule with {len(self.get_element_symbols())} atom(s): {', '.join(self.get_element_symbols())}", '']
        if (properties == 'all' or 'xyz_coordinates' in properties) and 'xyz_coordinates' in self.atoms[0].__dict__:
            printstrs += [f' XYZ coordinates, Angstrom\n']
            iatom = 0
            for atom in self.atoms:
                iatom += 1
                printstrs += [' %-4d %-3s %18.6f %18.6f %18.6f' % (iatom, atom.element_symbol,
        atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2])]
            printstrs += ['']
        if (properties == 'all' or 'distance_matrix' in properties) and 'xyz_coordinates' in self.atoms[0].__dict__:
            printstrs += [f' Interatomic distance matrix, Angstrom\n']
            dist_mat = self.get_internuclear_distance_matrix()
            printstrs += [f'{dist_mat}']
            printstrs += ['']
        if (properties == 'all' or 'energy' in properties) and 'energy' in self.__dict__:
            printstrs += [' Energy: %18.6f Hartree\n' % self.energy]
        if (properties == 'all' or 'energy_gradients' in properties) and 'energy_gradients' in self.atoms[0].__dict__:
            printstrs += [f' Energy gradients, Hartree/Angstrom\n']
            iatom = 0
            for atom in self.atoms:
                iatom += 1
                printstrs += [' %-4d %-3s %18.6f %18.6f %18.6f' % (iatom, atom.element_symbol,
        atom.energy_gradients[0], atom.energy_gradients[1], atom.energy_gradients[2])]
            printstrs += ['']
            printstrs += [' Energy gradients norm: %18.6f Hartree/Angstrom\n' % np.linalg.norm(self.energy_gradients)]
        printstr = '\n'.join(printstrs)
        if return_string: return printstr
        else: print(printstr)

    def __add__(self, obj):
        if isinstance(obj, molecular_database):
            return molecular_database([self] + obj.molecules)
        if isinstance(obj, molecule):
            return molecular_database([self] + [obj])
        
    def __str__(self):
        printstr = self.info(properties = 'all', return_string=True)
        return printstr
   
    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def __len__(self):
        return len(self.atoms)
        
    def __getitem__(self, item):
        return self.atoms[item]

    @property 
    def state_energies(self) -> np.ndarray:
        '''
        The electronic state energies of the molecule.
        '''
        return np.array([state.energy for state in self.electronic_states])
        
    @property 
    def state_gradients(self) -> np.ndarray:
        '''
        The electronic state energy gradients of the molecule.
        '''
        return np.array([state.energy_gradients for state in self.electronic_states])
        
    @property 
    def energy_gaps(self) -> np.ndarray:
        '''
        The energy gaps of different states.
        '''
        return self.state_energies - self.state_energies[:, np.newaxis]
    
    @property 
    def excitation_energies(self) -> np.ndarray:
        '''
        The excitation energies of the molecule from ground state.
        '''
        if '_excitation_energies' in self.__dict__:
            return self._excitation_energies
        else:
            return self.state_energies[1:] - self.electronic_states[0].energy if len(self.electronic_states) > 1 else []

    @excitation_energies.setter
    def excitation_energies(self, excitation_energies=None):
        if excitation_energies is not None:
            elst = False
            if 'electronic_states' in self.__dict__:
                if self.electronic_states != []:
                    if 'energy' in self.electronic_states[0].__dict__:
                        elst = True
                        for ii in range(len(excitation_energies)):
                            self.electronic_states[ii+1].energy = self.electronic_states[0].energy + excitation_energies[ii]                        
            if not elst: self._excitation_energies = excitation_energies


    @property 
    def nstates(self) -> np.int:
        '''
        The number of electronic states.
        '''
        if 'electronic_states' in self.__dict__:
            if self.electronic_states != []:
                return len(self.electronic_states)
        if '_excitation_energies' in self.__dict__:
            return len(self.excitation_energies)+1
        return 0

    def get_xyzvib_string(self, normal_mode=0):
        '''
        Get the xyz string with geometries and displacements along the vibrational normal modes
        '''
        natoms = self.get_number_of_atoms()
        xyzvib = f'{natoms}\n\n'
        for iatom in range(natoms):
            coords = self.atoms[iatom].xyz_coordinates
            disp = self.atoms[iatom].normal_modes[normal_mode]
            xyzvib += self.atoms[iatom].element_symbol
            xyzvib += f" {coords[0]:25.13f} {coords[1]:25.13f} {coords[2]:25.13f}"
            xyzvib += f" {disp[0]:25.13f} {disp[1]:25.13f} {disp[2]:25.13f}\n"
        return xyzvib

    def view(self, normal_mode=None, slider=True):
        '''
        Visualize the molecule and its vibrations if requested. Uses ``py3Dmol``.
        Arguments:
            normal_mode (integer, optional): the index of a normal mode to visualize. Default: None.
            slider(bool, optional):          show interactive slider to choose the mode.
                                             Default: True (only works if normal_mode is not None).
        '''
        import py3Dmol
        def animate(mode):
            py3Dmolargs = []
            viewer = py3Dmol.view(width=400, height=300)
            if not normal_mode is None:
                xyzstr = self.get_xyzvib_string(normal_mode=mode)
                py3Dmolargs = [{'vibrate': {'frames':15,'amplitude':0.8}}]
            else:
                xyzstr = self.get_xyz_string()
            viewer.addModel(xyzstr, "xyz", *py3Dmolargs)
            viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
            if not normal_mode is None: viewer.animate({'loop': 'backAndForth'})
            viewer.show()
        if not normal_mode is None and slider:
            import ipywidgets
            _ = ipywidgets.interact(animate,
                        mode=ipywidgets.IntSlider(min=0, max=len(self.frequencies)-1, step=1, value=normal_mode))
        else:
            animate(normal_mode)

class properties_tree_node():
    def __init__(self, name=None, parent=None, children=None, properties=None):
        self.name = name
        self.parent = parent
        self.children = children
        if self.parent != None:
            if self.parent.children == None: self.parent.children = []   
            if not self in self.parent.children:
                self.parent.children.append(self)
        if self.children != None:
            for child in self.children:
                child.parent=self
            
    def sum(self, properties):
        for property_name in properties:
            property_values_list = []
            for child in self.children:
                property_values_list.append(child.__dict__[property_name])
            self.__dict__[property_name] = np.sum(property_values_list, axis=0)

    def weighted_sum(self, properties):
        for property_name in properties:
            property_values_list = []
            for child in self.children:
                if 'weight' not in child.__dict__.keys():
                    child.weight = 1
                property_values_list.append(child.__dict__[property_name]*child.weight)
            self.__dict__[property_name] = np.sum(property_values_list, axis=0)
    
    def average(self, properties):
        for property_name in properties:
            property_values_list = []
            for child in self.children:
                property_values_list.append(child.__dict__[property_name])
            self.__dict__[property_name] = np.mean(property_values_list, axis=0)
    
    def standard_deviation(self, properties):
        for property_name in properties:
            property_values_list = []
            for child in self.children:
                property_values_list.append(child.__dict__[property_name])
            self.__dict__[property_name + '_standard_deviation'] = np.std(property_values_list, axis=0)
            
class molecular_database:
    '''
    Create a database for molecule objects.

    Arguments:
        molecules (List[:class:`molecule`]): A list of molecule to be included in the molecular database.
    
    Examples:

        Select an atom inside with subscription:

        .. code-block:: python
           
           from mlatom.data import atom, molecule, molecular_database
           at = atom(element_symbol = 'C')
           mol = molecule(atoms = [at])
           molDB = molecular_database([mol])
           print(id(mol) == id(molDB[0]))
           # the output should be 'True'

        Slicing the database like a numpy array:
           
        .. code-block:: python

           from mlatom.data import molecular_database 
           molDB = molecular_database.from_xyz_file('devtests/al/h2_fci_db/xyz.dat')
           print(len(molDB))           # 451
           print(len(molDB[:100:4]))   # 25

           
    '''
    def __init__(self, molecules: List[molecule] = None):
        if type(molecules) == type(None):
            molecules = []
        elif isinstance(molecules, molecule):
            molecules = [molecules]
        elif isinstance(molecules, molecular_database):
            molecules = molecules.molecules

        self.molecules = molecules

    def read_from_xyz_file(self, filename: str, append: bool = False) -> molecular_database:
        '''
        Load molecular geometry from XYZ file.
            
        Arguments:
            filename (str): The name of the file to be read.
            append (bool, optional): Append to current database if True, otherwise clear the current database.
        '''
        with open(filename, 'r') as fxyz:
            string = fxyz.read()
        return self.read_from_xyz_string(string, append=append)
            
    def read_from_xyz_string(self, string: str, append=False) -> molecular_database:
        '''
        Load molecular geometry from XYZ string.
            
        Arguments:
            string (str): The name of the file to be read.
            append (bool, optional): Append to current database if True, otherwise clear the current database.
        '''
        xyz_strings = conversions.split_xyz_string(string)
        if not append: self.molecules = []
        for xyz_string in xyz_strings:
            self.molecules.append(molecule.from_xyz_string(xyz_string))
        return self
    
    def read_from_numpy(self, coordinates: np.ndarray, species: np.ndarray, append: bool = False) -> molecular_database:
        '''
        Load molecular geometries from a numpy array of coordinates and another one for species.

        The shape of the input ``coordinates`` should be ``(M, N, 3)``, while ``(M, N,)`` for the input ``species``.

        Where the ``N`` is the number of atoms and ``M`` is the number of molecules.
        '''
        if not append: self.molecules = []
        for i in range(coordinates.shape[0]):
            self.molecules.append(molecule.from_numpy(coordinates[i], species[i]))
        return self
    
    def read_from_smiles_file(self, smi_file: str, append: bool = False) -> molecular_database:
        '''
        Generate molecular geometries from a SMILES file provided.

        The geometries will be generated and optimized with `Pybel's <https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_Pybel.html>`_ ``make3D()`` method.
        '''
        with open(smi_file, 'r') as f:
            smi_string = f.read()
        return self.read_from_smiles_string(smi_string, append=append)

    def read_from_smiles_string(self, smi_string: str, append: bool = False) -> molecular_database:
        '''
        Generate molecular geometries from a SMILES string provided.

        The geometries will be generated and optimized with `Pybel's <https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_Pybel.html>`_ ``make3D()`` method.
        '''
        if not append: self.molecules = []
        xyz_string = conversions.smi2xyz(smi_string)
        self.read_from_xyz_string(xyz_string)
        return self
        
    def read_from_h5_file(self, 
        h5_file: str = '', 
        properties: list = None, 
        parallel: Union[bool,int,tuple] = False, 
        verbose:bool = False) -> molecular_database:
        '''
        Generate molecular database from formatted h5 file. The first level should be configurations (or ensemble of molecules with same number of atoms) and the second level should be conformations and their properties. 'species' and 'coordinates' are required to construct molecule. An example format of h5 file:

        ```
        /003                  dict
        /003/species          array (624, 3) [int8]
        /003/coordinates      array (624, 3, 3) [float32]
        /003/energies         array (624,) [float64]
        /003/property1        ['wb97x/def2tzvpp']
        /003/property2        array (624, 2) [int8]
        ```
        
        If the first two dimensions of the size of the value equals (number_of_configurations, number_of_atoms), the remaining dimension of the value will be assigned to each atom as xyz derivative properties. If the first dimensions of the size of the value equals to number of configurations, corresponging value will be assigned to each molecule. If only one value is provided for the property, it will be copied into each molecule.  
        For example, in the above case, the properties stored in each `molecule` object would be: {'energies': `float`, 'property1':'wb97x/def2tzvpp', 'property2': `numpy.ndarray` of size (2,0)}

        Arguments:
            h5file (str): path to h5 file. 
            properties (list): the properties to be stored in molecular database. By default all the properties presented in h5 file will be stored.
            parallel (int or tuple or bool): 
                - If `int` is provided, the value will be assigned to the number of workers, Batch size will be calculated automatically.
                - If `tuple` is provided, the first value will be assigned to the number of workers and the second value will be assigned to the batch size. 
                - If `bool` is provided, `True` means all the CPUs available will be used and batch size will be adjusted accordingly. 
            verbose (bool): whether to print the loading message.

        ''' 
        
        import sys
        import joblib 
        
        h5data = h5dataloader(h5_file)

        def configurations():
            for cc in h5data:
                yield cc

        def conformations():
            for m in configurations():
                species = m['species']
                coordinates = m['coordinates']
                (nc, na, _) = coordinates.shape
                for i in range(coordinates.shape[0]):
                    ret = {'species': species[i], 'coordinates': coordinates[i]}
                    for k in properties:
                        if k in m:
                            if len(m[k]) == nc:
                                ret[k] = m[k][i]
                            else:
                                ret[k] = m[k]
                    yield ret
        
        def pool(batch):
            empty_pool = []
            for cc in conformations():  
                empty_pool.append(cc)
                if len(empty_pool) >= batch:
                    yield empty_pool
                    empty_pool = []
            yield empty_pool

        def batch_load(b, properties):
            mols = []
            for cc in b:
                mol = molecule.from_numpy(species=cc['species'],coordinates=cc['coordinates'])
                na = cc['coordinates'].shape[0]
                if not properties:
                    properties = [*cc]
                    properties.remove('species')
                    properties.remove('coordinates')

                for prop in properties:
                    if type(cc[prop]) == np.ndarray and cc[prop].shape == (na, 3):
                        mol.add_xyz_derivative_property(cc[prop],prop,prop)
                    elif type(cc[prop]) in {list, np.ndarray} and len(cc[prop]) == 1:
                        mol.add_scalar_property(cc[prop][0],prop)
                    else:
                        mol.add_scalar_property(cc[prop],prop)
                mols.append(mol)
            return mols
                
        if parallel:
            if type(parallel) == int:
                njobs = parallel
                counts = h5data.size()
                batch = counts//njobs

            elif type(parallel) == tuple:
                njobs, batch = (parallel)
            else:
                njobs = -1
                counts = h5data.size()
                import multiprocessing as mp 
                ncpu = mp.cpu_count() 
                batch = counts//ncpu
        else:
            njobs = 1
            counts = h5data.size()
            batch = counts

        if verbose:
            print(f'{njobs} workers will be used and {batch} tasks are assigned to each worker.')
            if batch < 100 and njobs > 1:
                print('WARNING: batch size less than 100 is not recommended to use parallel loading.') # need to benchmark to set the threshold.
            sys.stdout.flush()

        mols = joblib.Parallel(n_jobs=njobs, verbose=10 if verbose else 0)(joblib.delayed(batch_load)(b, properties) for b in pool(batch))

        for mm in mols:
            self.molecules += mm

        return self 


    @classmethod
    def from_xyz_file(cls, filename: str) -> molecular_database:
        '''
        Classmethod wrapper for :meth:`molecular_database.read_from_xyz_file`, returns a :class:`molecular_database` object.
        '''
        return cls().read_from_xyz_file(filename)
    
    @classmethod
    def from_xyz_string(cls, string: str) -> molecular_database:
        '''
        Classmethod wrapper for :meth:`molecular_database.read_from_xyz_string`, returns a :class:`molecular_database` object.
        '''
        return cls().read_from_xyz_string(string)
    
    @classmethod
    def from_numpy(cls, coordinates: np.ndarray, species: np.ndarray) -> molecular_database:
        '''
        Classmethod wrapper for :meth:`molecular_database.read_from_numpy`, returns a :class:`molecular_database` object.
        '''
        return cls().read_from_numpy(coordinates, species)

    @classmethod
    def from_smiles_file(cls, smi_file: str) -> molecular_database:
        '''
        Classmethod wrapper for :meth:`molecular_database.read_from_smiles_file`, returns a :class:`molecular_database` object.
        '''
        return cls().read_from_smiles_file(smi_file)
    
    @classmethod
    def from_smiles_string(cls, smi_string: Union[str, List]) -> molecular_database:
        '''
        Classmethod wrapper for :meth:`molecular_database.read_from_smiles_string`, returns a :class:`molecular_database` object.
        '''
        return cls().read_from_smiles_string(smi_string)
    

    def add_scalar_properties(self, scalars, property_name: str = 'y') -> None: # kind of redundant? mol.a = x does the samething
        '''
        Add scalar properties to the molecules.

        Arguments:
            scalars: The scalar to be added.
            property_name (str, optional): The name assign to the scalar property.
        '''
        # for i in range(scalars.shape[0]):
        for i in range(len(scalars)):
            self.molecules[i].add_scalar_property(scalars[i], property_name=property_name)

    def add_scalar_properties_from_file(self, filename: str, property_name: str = 'y') -> None: # kind of redundant? mol.a = x does the samething
        '''
        Add scalar properties from a file to the molecules.

        Arguments:
            filename (str): Specify the text file that contains properties.
            property_name (str, optional): The name assign to the scalar property.
        '''
        with open(filename, 'r') as fy:
            ii = -1
            for line in fy:
                ii += 1
                yy = float(line)
                self.molecules[ii].__dict__[property_name] = yy

    def add_xyz_derivative_properties(self, derivatives, property_name: str = 'y', xyz_derivative_property: str = 'xyz_derivatives') -> None:
        '''
        Add a XYZ derivative property to the molecule.

        Arguments:
            derivatives: The derivatives to be added.
            property_name (str, optional): The name of the associated non-derivative property.
            xyz_derivative_property (str, optional): the name assign to the derivative property.
        '''
        if not 'properties_and_their_derivatives' in self.__dict__.keys():
            self.properties_and_their_derivatives = {}
        self.properties_and_their_derivatives[property_name] = xyz_derivative_property
        for i in range(derivatives.shape[0]):
            self.molecules[i].add_xyz_derivative_property(derivatives[i], property_name=property_name, xyz_derivative_property=xyz_derivative_property)

    def add_xyz_derivative_properties_from_file(self, filename: str, property_name: str = 'y', xyz_derivative_property: str = 'xyz_derivatives') -> None:
        '''
        Add a XYZ derivatives from a text file to the molecules.

        Arguments:
            filename (str): The filename that contains derivatives to be added.
            property_name (str, optional): The name of the associated non-derivative properties.
            xyz_derivative_property (str, optional): the name assign to the derivative properties.
            
        '''
        if not 'properties_and_their_derivatives' in self.__dict__.keys():
            self.properties_and_their_derivatives = {}
        self.properties_and_their_derivatives[property_name] = xyz_derivative_property
        self.add_xyz_vectorial_properties_from_file(
            filename=filename, xyz_vectorial_property=xyz_derivative_property)

    def add_hessian_properties(self, hessians, hessian_propety='hessian'):
        for i in range(len(hessians)):
            self.molecules[i].add_hessian_property(hessians[i], hessian_propety=hessian_propety)

    
    def add_xyz_vectorial_properties(self, vectors, xyz_vectorial_property: str = 'xyz_vector') -> None:
        '''
        Add a XYZ vectorial properties to the molecules.

        Arguments:
            vectors: The vectors to be added.
            xyz_vectorial_property (str, optional): the name assign to the vectorial properties.
            
        '''
        for i in range(vectors.shape[0]):
            self.molecules[i].add_xyz_vectorial_property(vectors[i], xyz_vectorial_property=xyz_vectorial_property)

    def add_xyz_vectorial_properties_from_string(self, string: str,  xyz_vectorial_property: str = 'xyz_vector') -> None:
        xyz_strings = conversions.split_xyz_string(string)
        for imol, xyz_string in enumerate(xyz_strings):
            fxyz = xyz_string.split('\n')
            natoms = int(fxyz.pop(0))
            fxyz.pop(0)
            assert natoms == len(self[imol]), 'the number of atom does not match'
            for line, atom in zip(fxyz, self[imol]):
                yy = line.split()[-3:]
                vector = array(yy).astype(float)
                setattr(atom, xyz_vectorial_property, vector)

    def add_xyz_vectorial_properties_from_file(self, filename: str, xyz_vectorial_property: str = 'xyz_vector') -> None:
        '''
        Add a XYZ derivatives from a text file to the molecules.

        Arguments:
            filename (str): The filename that contains vectorial properties to be added.
            xyz_vectorial_property (str, optional): the name assign to the vectorial properties.
        '''
        with open(filename, 'r') as fxyz:
            string = fxyz.read()
            self.add_xyz_vectorial_properties_from_string(string=string, xyz_vectorial_property=xyz_vectorial_property)

    def write_file_with_xyz_coordinates(self, filename: str) -> None:
        '''
        Write the molecular geometries into a file in XYZ format.

        Arguments:
            filename (str): The name of the file to be written.
        '''
        with open(filename, 'w') as fw:
            for mol in self.molecules:
                fw.writelines('%d\n' % len(mol.atoms))
                if 'comment' in mol.__dict__.keys():
                    fw.writelines(f'{mol.comment}\n')
                else:
                    fw.writelines('\n')
                for atom in mol.atoms:
                    fw.writelines('%-3s %25.13f %25.13f %25.13f\n' % (atom.element_symbol,
                                  atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2]))
    
    def get_xyz_string(self) -> None:
        '''
        Return a string in XYZ format for the molecules.
        '''
        xyz_string = ''
        for mol in self.molecules:
            xyz_string += mol.get_xyz_string()
        return xyz_string

    def write_file_with_properties(self, filename, property_to_write='y'): # to be rewrite
        '''
        Write a property of molecules to a text file.
        '''
        with open(filename, 'w') as fw:
            for mol in self.molecules:
                #fw.writelines('%25.13f\n' % mol.__dict__[property_to_write])
                fw.writelines('%25.13f\n' % eval(f'mol.{property_to_write}'))

    def get_number_of_atoms(self):
        return array([len(mol) for mol in self.molecules])
    
    @property
    def number_of_atoms(self):
        return self.get_number_of_atoms()
    
    def get_atomic_numbers(self):
        atomic_numbers = []
        for mol in self.molecules:
            atomic_numbers.append(mol.get_atomic_numbers())
        return array(atomic_numbers)
    
    @property
    def atomic_numbers(self) -> np.ndarray:
        '''
        The 2D array of the atomic numbers of each atom, for all molecules in the database.
        '''
        return self.get_atomic_numbers()
    
    def get_element_symbols(self):
        element_symbols = []
        for mol in self.molecules:
            element_symbols.append(mol.get_element_symbols())
        return array(element_symbols)
    
    @property
    def element_symbols(self) -> np.ndarray:
        '''
        The 2D array of the element symbols of each atom, for all molecules in the database.
        '''
        return self.get_element_symbols()
    
    @property
    def ids(self):
        '''
        The IDs of the molecules in the database.
        '''
        return self.get_properties(property_name='id')

    @property
    def smiles(self) -> str:
        '''
        The SMILES string of the molecules in the database.
        '''
        return conversions.xyz2smi(self.get_xyz_string())
    
    def write_file_with_smiles(self, filename):
        '''
        Write the SMILES of the molecules in the database to a file.
        '''
        with open(filename, 'w') as f:
            f.write(self.smiles)

    @property
    def nuclear_masses(self):
        '''
        The nuclear_masses of the molecules in the database.
        '''
        return self.get_properties(property_name='nuclear_masses')

    @property
    def charges(self):
        '''
        The electric charges of the molecules in the database.
        '''
        return self.get_properties(property_name='charge')
    
    @charges.setter
    def charges(self, charges):
        self.set_properties(charge=charges)

    @property
    def multiplicities(self):
        '''
        The multiplicities of the molecules in the database.
        '''
        return self.get_properties(property_name='multiplicity')
    
    @multiplicities.setter
    def multiplicities(self, multiplicities):
        self.set_properties(multiplicity=multiplicities)
        
    def get_properties(self, property_name='y',): # move to __getitem__
        '''
        Return the properties of the molecules by a given property name.
        '''
        properties = []
        for mol in self.molecules:
            properties.append(mol.get_property(property_name))
        return array(properties)
    
    def set_properties(self, **kwargs): # move to __setitem__
        '''
        Set properties of the molecules by given property name(s) as keyword(s).
        '''
        for property_name, values in kwargs.items():
            for i, mol in enumerate(self.molecules):
                mol.__dict__[property_name] = values[i]
    
    def get_xyz_derivative_properties(self, xyz_derivative_property='xyz_derivatives'):
        '''
        Return XYZ derivative properties by the name.
        '''
        return self.get_xyz_vectorial_properties(xyz_derivative_property)
    
    def get_xyz_vectorial_properties(self, property_name):
        '''
        Return XYZ vectorial properties by the name.
        '''
        coordinates = []
        for mol in self.molecules:
            coordinates.append(mol.get_xyz_vectorial_properties(property_name))
        return array(coordinates)
    
    def write_file_with_xyz_derivative_properties(self, filename, xyz_derivative_property_to_write='xyz_derivatives'):
        '''
        Write XYZ derivative properties into a file.
        '''
        self.write_file_with_xyz_vectorial_properties(
            filename=filename, xyz_vectorial_property_to_write=xyz_derivative_property_to_write)
    
    def write_file_energy_gradients(self, filename):
        '''
        Write energy gradients into a file.
        '''
        self.write_file_with_xyz_derivative_properties(filename=filename, xyz_derivative_property_to_write='energy_gradients')
     
    def write_file_with_xyz_vectorial_properties(self, filename, xyz_vectorial_property_to_write='xyz_vector'):
        '''
        Write XYZ vectorial properties into a file.
        '''
        with open(filename, 'w') as fw:
            for mol in self.molecules:
                fw.writelines('%d\n' % len(mol.atoms))
                fw.writelines('\n')
                for atom in mol.atoms:
                    fw.writelines(' %25.13f %25.13f %25.13f\n' % (atom.__dict__[xyz_vectorial_property_to_write][0], atom.__dict__[
                                  xyz_vectorial_property_to_write][1], atom.__dict__[xyz_vectorial_property_to_write][2]))

    def write_file_with_hessian(self, filename, hessian_property_to_write='hessian'):
        '''
        Write Hessians into a file.
        '''
        with open(filename, 'w') as fhess:
            for mol in self.molecules:
                fhess.write('%d\n\n' % len(mol.atoms))
                np.savetxt(fhess, mol.__dict__[hessian_property_to_write].flatten(), fmt='%25.13f')

    def sum_properties(self, **kwargs):
        if 'summed_property_label' in kwargs:
            summed_property_label = kwargs['summed_property_label']
            for mol in self.molecules:
                mol.__dict__[summed_property_label] = 0.0
        if 'properties_labels' in kwargs:
            properties_labels = kwargs['properties_labels']
            for property_name in properties_labels:
                for mol in self.molecules:
                    mol.__dict__[
                        summed_property_label] += mol.__dict__[property_name]

        if 'summed_xyz_derivative_property_label' in kwargs:
            summed_xyz_derivative_property_label = kwargs['summed_xyz_derivative_property_label']
            for mol in self.molecules:
                for atom in mol.atoms:
                    atom.__dict__[summed_xyz_derivative_property_label] = np.zeros(3)
        if 'xyz_derivative_properties_labels' in kwargs:
            xyz_derivative_properties_labels = kwargs['xyz_derivative_properties_labels']
            for property_name in xyz_derivative_properties_labels:
                for mol in self.molecules:
                    for atom in mol.atoms:
                        atom.__dict__[summed_xyz_derivative_property_label] += atom.__dict__[property_name]

        if 'summed_hessian_property_label' in kwargs:
            summed_hessian_property_label = kwargs['summed_hessian_property_label']
            for mol in self.molecules:
                ndim = len(mol.atoms)*3
                mol.__dict__[summed_hessian_property_label] = np.zeros((ndim, ndim))
        if 'hessian_properties_labels' in kwargs:
            hessian_properties_labels = kwargs['hessian_properties_labels']
            for property_name in hessian_properties_labels:
                for mol in self.molecules:
                    mol.__dict__[summed_hessian_property_label] += mol.__dict__[property_name]

    def append(self, obj):
        '''
        Append a molecule/molecular database.
        '''
        if isinstance(obj, molecular_database):
            self.molecules += obj.molecules
        if isinstance(obj, molecule):
            self.molecules += [obj]

    def copy(self, atomic_labels=None, molecular_labels=None, molecular_database_labels=None):
        '''
        Return a copy of the database.
        '''
        if type(atomic_labels) != type(None) or type(molecular_labels) != type(None) or type(molecular_database_labels) != type(None):
            new_molecular_database = molecular_database()
            if type(molecular_database_labels) != type(None):
                for each_label in molecular_database_labels:
                    if each_label in self.__dict__:
                        new_molecular_database.__dict__[each_label] = self.__dict__[each_label]
            else:
                for each_label in self.__dict__.keys():
                    if each_label == 'molecules': continue
                    new_molecular_database.__dict__[each_label] = self.__dict__[each_label]
            if type(molecular_labels) != type(None) or type(atomic_labels) != type(None):
                for imolecule in range(len(self.molecules)):
                    new_molecule = self.molecules[imolecule].copy(atomic_labels=atomic_labels,molecular_labels=molecular_labels)
                    new_molecular_database.molecules.append(new_molecule)
        else:
            new_molecular_database = copy.deepcopy(self)
        return new_molecular_database
    
    def filter_by_property(self, property_name):
        return molecular_database(self[~np.isnan(self.get_properties(property_name))])
    
    def proliferate(self, *args, **kwargs) -> molecular_database:
        '''
        Proliferate the unicell by specified shifts along cell vectors.
        
        Returns a new :class:`molecular_databse` object.
        
        Check :meth:`molecule.proliferate` for details on options.
        '''
        return molecular_database([mol.proliferate(*args, **kwargs) for mol in self])
    
    def dump(self, filename=None, format=None):
        '''
        Dump the molecular database to a file.
        '''
        if format.casefold() == 'json'.casefold():
            jsonfile = open(filename, 'w')
            json.dump(class_instance_to_dict(self), jsonfile, indent=4)
            jsonfile.close()
        if format.casefold() == 'npz'.casefold():
            np.savez(filename, **class_instance_to_dict(self))
            
    def _load(self, filename=None, format=None):
        if format.casefold() == 'json'.casefold():
            jsonfile = open(filename, 'r')
            data = json.load(jsonfile)
            self.molecules = []
            for molecule in data['molecules']:
                self.molecules.append(dict_to_molecule_class_instance(molecule))
        if format.casefold() == 'npz'.casefold():
            with np.load(filename, allow_pickle=True) as npz:
                data = dict(npz)
                self.molecules = []
                for molecule in data['molecules']:
                    self.molecules.append(dict_to_molecule_class_instance(molecule))
        return self
    
    @classmethod
    def load(cls, filename=None, format=None):
        '''
        Load a molecular database from a file.
        '''
        return cls()._load(filename=filename, format=format)
            
    def batches(self, batch_size):
        batch_id = -1
        for batch_id in range(len(self) // batch_size):
            yield self[batch_id * batch_size:(batch_id + 1)*batch_size]
        if len(self) % batch_size:
            yield self[(batch_id + 1)*batch_size:]
           
    def split(self, sampling='random', number_of_splits=2, split_equally=None, fraction_of_points_in_splits=None):
        '''
        Splits molecular database.
        
        Arguments:
            sampling (str, optional): default 'random'. Can be also 'none'.
            split_equally (bool, optinoal): default ``False``; if set to ``True`` splits 50:50.
            fraction_of_points_in_splits (list, optional): e.g., [0.8, 0.2] is the default one
            indices
        '''
        return sample(molecular_database_to_split=self, sampling=sampling, number_of_splits=number_of_splits, split_equally=split_equally, fraction_of_points_in_splits=fraction_of_points_in_splits)

    @property         
    def size(self):
        return len(self)
    
    def __add__(self, obj):
        if isinstance(obj, molecular_database):
            return molecular_database(self.molecules + obj.molecules)
        if isinstance(obj, molecule):
            return molecular_database(self.molecules + [obj])

    def __str__(self):
        return f"molecular database of {len(self)} molecule(s)"

    def __iter__(self):
        for mol in self.molecules:
            yield mol

    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, item):
        if item is None:
            return None
        if isinstance(item, list):
            return molecular_database([self.molecules[i] for i in item])
        if isinstance(item, np.ndarray):
            if item.dtype == 'bool':
                return molecular_database([self.molecules[i] for i in range(len(self)) if item[i]])
            else:
                return molecular_database([self.molecules[i] for i in item])
        if isinstance(item, slice):
            return molecular_database(self.molecules[item])
        else:
            return self.molecules[item]
        
    @property 
    def xyz_coordinates(self):
        '''
        The XYZ coordinates of each atom in every molecule.
        '''
        coordinates = []
        for mol in self.molecules:
            coordinates.append(mol.xyz_coordinates)
        return array(coordinates)

    @xyz_coordinates.setter
    def xyz_coordinates(self, value):
        for i, mol in enumerate(self):
            mol.xyz_coordinates = value[i]
    
    def _is_uniform_cell(self):
        cells = self.get_properties('cell')
        pbcs = self.get_properties('pbc')
        try:
            if set(cells) == {None} and set(pbcs) == {None}:
                return True
            else:
                return False
        except:
            try:
                if np.max(np.std(cells, 0)) == 0 and np.max(np.std(pbcs, 0)) == 0:
                    return True
                else:
                    return False
            except:
                return False
    
    @property
    def pbc(self):
        if self._is_uniform_cell():
            return self[0].pbc
            
    @pbc.setter
    def pbc(self, pbc):
        for mol in self:
            mol.pbc = pbc

    @property
    def cell(self):
        if self._is_uniform_cell():
            return self[0].cell
            
    @cell.setter
    def cell(self, cell):
        for mol in self:
            mol.cell = cell
            
    def view(self):
        '''
        Visualize the molecular database. Uses ``py3Dmol``.
        '''
        import py3Dmol
        xyzstr = self.get_xyz_string()
        viewer = py3Dmol.view(width=400, height=300)
        viewer.addModelsAsFrames(xyzstr, "xyz")
        viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
        viewer.zoomTo()
        viewer.animate({'loop': 'forward'})
        viewer.show()

def class_instance_to_dict(inst):
    dd = copy.deepcopy(inst.__dict__)
    for key in dd.keys():
        if type(dd[key]) == np.ndarray: dd[key] = dd[key].tolist()
        elif type(dd[key]) == np.float32: dd[key] = dd[key].item()
        elif type(dd[key]) == np.float64: dd[key] = dd[key].item()
        elif type(dd[key]) == np.int16: dd[key] = dd[key].item()
        elif type(dd[key]) == np.int32: dd[key] = dd[key].item()
        elif type(dd[key]) == np.int64: dd[key] = dd[key].item()
        elif type(dd[key]) == np.cfloat: dd[key] = dd[key].item()
        elif key == 'parent':
            if dd[key] != None: dd[key] = dd[key].name
        elif key == 'children':
            if dd[key] != None:
                for ii in range(len(dd[key])):
                    dd[key][ii] = dd[key][ii].name
        elif type(dd[key]) == list:
            for ii in range(len(dd[key])):
                if type(dd[key][ii]) == list:
                    for jj in range(len(dd[key][ii])):
                        if type(dd[key][ii][jj]) == np.ndarray: dd[key][ii][jj] = dd[key][ii][jj].tolist()
                elif type(dd[key][ii]) == np.ndarray: dd[key][ii] = dd[key][ii].tolist()
                elif hasattr(dd[key][ii], '__dict__'): dd[key][ii] = class_instance_to_dict(dd[key][ii])
        elif hasattr(dd[key], '__dict__'): dd[key] = class_instance_to_dict(dd[key])
    return dd

def dict_to_atom_class_instance(dd):
    aatom = atom()
    for key in dd.keys():
        if type(dd[key]) == list:
            if len(dd[key]) == 0:
                aatom.__dict__[key] = dd[key]
            elif type(dd[key][0]) == float:
                aatom.__dict__[key] = array(dd[key]).astype(float)
            else:
                aatom.__dict__[key] = dd[key]
        else:
            aatom.__dict__[key] = dd[key]
    return aatom

def dict_to_properties_tree_node_class_instance(original_dict, original_key, mol):
    node = properties_tree_node()
    dd = original_dict[original_key]
    name = dd['name']
    if name in mol.__dict__:
        return
    for key in dd.keys():
        if type(dd[key]) == list:
            if type(dd[key][0]) == float:
                node.__dict__[key] = array(dd[key]).astype(float)
            elif type(dd[key][0]) == list:
                if type(dd[key][0][0]) == float:
                    node.__dict__[key] = array(dd[key]).astype(float)
            elif key == 'children':
                node.children = []
                for ichild in range(len(dd[key])):
                    child = dd['children'][ichild]
                    if type(child) == str:
                        if child in mol.__dict__:
                            if type(mol.__dict__[child]) == properties_tree_node:
                                node.children.append(mol.__dict__[child])
                        else:
                            dict_to_properties_tree_node_class_instance(original_dict, child, mol)
                            node.children.append(mol.__dict__[child])
                        node.children[-1].parent = node
            else:
                node.__dict__[key] = dd[key]
        elif key == 'parent':
            if type(dd['parent']) == str:
                if dd['parent'] in mol.__dict__:
                    if type(mol.__dict__[dd['parent']]) == properties_tree_node:
                        node.parent = mol.__dict__[dd['parent']]
        else:
            node.__dict__[key] = dd[key]
    mol.__dict__[node.name] = node

def dict_to_molecule_class_instance(dd):
    mol = molecule()
    for key in dd.keys():
        if key == 'atoms':
            for aa in dd[key]:
                mol.atoms.append(dict_to_atom_class_instance(aa))
        elif key == 'electronic_states':
            mol.electronic_states = [dict_to_molecule_class_instance(state_dict) for state_dict in dd[key]]
        elif type(dd[key]) == dict:
            if 'parent' in dd[key].keys():
                dict_to_properties_tree_node_class_instance(dd, key, mol)    
            else:
                mol.__dict__[key] = dd[key]
        elif type(dd[key]) == list:
            try:
                mol.__dict__[key] = np.array(dd[key]).astype(float)
            except:
                mol.__dict__[key] = dd[key]
        else:
            mol.__dict__[key] = dd[key]
    return mol

def dict_to_reaction_step_class_instance(dd):
    reaction = reaction_step()
    for key in dd.keys():
        if key == 'molecules':
            for mol in dd[key]:
                reaction.molecules.append(dict_to_molecule_class_instance(mol))
        elif type(dd[key]) == dict:
            if 'parent' in dd[key].keys():
                dict_to_properties_tree_node_class_instance(dd, key, reaction)    
            else:
                reaction.__dict__[key] = dd[key]
        else:
            reaction.__dict__[key] = dd[key]
    return reaction

class molecular_trajectory():
    '''
    A class for storing/access molecular trajectory data, which is generated from a dynamics or a optimization task.
    '''
    def __init__(self, steps=None):
        # Meta-data: ensemble used, etc.
        if type(steps) != type(None): self.steps = steps
        else: self.steps = []  # List with instancies of molecular_trajectory_step

    def dump(self, filename=None, format=None):            
        '''
        Dump the molecular_trajectory object into a file.

        Available formats are:

        - ``'h5md'`` (requires python module ``h5py`` and ``pyh5md``)

        - ``'json'``

        - ``'plain_text'``
        '''
        if format.lower() == 'h5md':
            data = {'time':[],
                    'position':[],
                    'velocities':[],
                    'gradients':[],
                    'kinetic_energy':[],
                    'potential_energy':[],
                    'total_energy':[],
                    'mass':None,
                    'species':None,
                    }
            data['state_energies'] = []
            data['aux_state_energies'] = []
            data['state_gradients'] = []
            if 'nonadiabatic_coupling_vectors' in self.steps[0].molecule.atoms[0].__dict__: data['nonadiabatic_coupling_vectors'] = []
            data['random_number'] = []
            data['hopping_probabilities'] = []
            data['need_to_be_labeled'] = []
            if 'current_state' in self.steps[0].__dict__: data['current_state'] = []
            if 'uncertain' in self.steps[0].molecule.__dict__: data['uncertain'] = []

            #'state_gradients'
            dp_flag = True
            for istep in self.steps:
                if not 'dipole_moment' in istep.molecule.__dict__.keys():
                    dp_flag = False 
            if dp_flag:
                data['dipole_moment'] = []
            data['mass'] = self.steps[0].molecule.nuclear_masses
            data['species'] = self.steps[0].molecule.get_atomic_numbers()
            for istep in self.steps:
                
                data['time'].append(istep.time)
                data['position'].append(istep.molecule.xyz_coordinates)
                data['velocities'].append(istep.molecule.get_xyz_vectorial_properties('xyz_velocities'))
                data['gradients'].append(istep.molecule.get_energy_gradients())
                if 'uncertain' in data.keys():
                    if istep.molecule.uncertain == True:
                        data['uncertain'].append(1)
                    else:
                        data['uncertain'].append(0)
                if len(istep.molecule.electronic_states) > 1:
                    data['state_energies'].append(istep.molecule.state_energies)
                    state_gradients = []
                    for i in range(0, len(istep.molecule.electronic_states)):
                        if 'energy_gradients' in istep.molecule.electronic_states[i].atoms[0].__dict__:
                            state_gradients.append(istep.molecule.electronic_states[i].get_energy_gradients())
                        else:
                            state_gradients.append(None)
                    data['state_gradients'].append(np.array(state_gradients))
                    if 'aux_energy' in istep.molecule.electronic_states[0].__dict__.keys():
                        aux_state_energies = []
                        for i in range(0, len(istep.molecule.electronic_states)):
                            aux_state_energies.append(istep.molecule.electronic_states[i].aux_energy)
                        data['aux_state_energies'].append(np.array(aux_state_energies))
                        
                if 'nonadiabatic_coupling_vectors' in data.keys(): data['nonadiabatic_coupling_vectors'].append(istep.molecule.get_xyz_vectorial_properties('nonadiabatic_coupling_vectors'))
                data['kinetic_energy'].append(istep.molecule.kinetic_energy)
                data['potential_energy'].append(istep.molecule.energy)
                data['total_energy'].append(istep.molecule.kinetic_energy+istep.molecule.energy)
                if 'random_number' in data.keys():
                    try:
                        data['random_number'].append(istep.random_number)
                    except AttributeError:
                        data['random_number'].append(np.nan)
                if 'hopping_probabilities' in data.keys():
                    try:
                        data['hopping_probabilities'].append(max(istep.hopping_probabilities))
                    except AttributeError:
                        data['hopping_probabilities'].append(np.nan)
                if 'current_state' in data.keys():
                    try:
                        data['current_state'].append(istep.current_state)
                    except AttributeError:
                        data['current_state'].append(np.nan)
                if 'need_to_be_labeled' in data.keys():
                    try:
                        if istep.molecule.need_to_be_labeled == True:
                            data['need_to_be_labeled'].append(1)
                        elif istep.molecule.need_to_be_labeled == False:
                            data['need_to_be_labeled'].append(0)
                    except AttributeError:
                        data['need_to_be_labeled'].append(np.nan)
                if dp_flag:
                    data['dipole_moment'].append(istep.molecule.dipole_moment)
            with h5md(filename) as trajH5:
                trajH5.write(data)
        
        elif format.lower() == 'plain_text':
            moldb = molecular_database()
            for istep in self.steps:
                moldb.molecules.append(istep.molecule)
            moldb.write_file_with_xyz_coordinates(filename+'.xyz')
            moldb.write_file_with_xyz_vectorial_properties(filename+'.vxyz',xyz_vectorial_property_to_write='xyz_velocities')
            moldb.write_file_energy_gradients(filename+'.grad')
            moldb.write_file_with_properties(filename+'.ekin',property_to_write='kinetic_energy')
            moldb.write_file_with_properties(filename+'.epot',property_to_write='energy')
            moldb.write_file_with_properties(filename+'.etot',property_to_write='total_energy')
            moldb.write_file_with_properties(filename+'.temp',property_to_write='temperature')
            if 'dipole_moment' in moldb.molecules[0].__dict__.keys():
                with open(filename+'.dp','w') as dpf:
                    for imolecule in moldb.molecules:
                        dpf.write('%25.13f %25.13f %25.13f %25.13f\n'%(imolecule.dipole_moment[0],imolecule.dipole_moment[1],imolecule.dipole_moment[2],imolecule.dipole_moment[3]))
        
        elif format.casefold() == 'json'.casefold():
            jsonfile = open(filename, 'w')
            json.dump(class_instance_to_dict(self), jsonfile, indent=4)
            jsonfile.close()
        
    def load(self, filename: str = None, format: str =None):
        '''
        Load the previously dumped molecular_trajectory from file.
        '''
        self.steps = []
        if format.lower() == 'h5md':
            with h5md(filename) as trajH5:
                data = trajH5.export()
            imolecule = molecule()
            Natoms = len(data['species'])
            for iatom in range(Natoms):
                imolecule.atoms.append(atom(atomic_number=data['species'][iatom],nuclear_mass=data['mass'][iatom]))
            for istep in range(len(data['time'])):
                trajectory_step = molecular_trajectory_step()
                molecule_istep = imolecule.copy(atomic_labels=[])
                # position
                molecule_istep.xyz_coordinates = data['position'][istep]
                # velocities
                for iatom in range(Natoms):
                    molecule_istep.atoms[iatom].xyz_velocities = data['velocities'][istep][iatom]
                # gradients
                for iatom in range(Natoms):
                    molecule_istep.atoms[iatom].energy_gradients = data['gradients'][istep][iatom]
                if 'need_to_be_labeled' in data.keys():
                    if not np.isnan(data['need_to_be_labeled'][istep]):
                        if int(data['need_to_be_labeled'][istep]) == 1:
                            molecule_istep.need_to_be_labeled = True
                        else:
                            molecule_istep.need_to_be_labeled = False  
                if 'uncertain' in data.keys():
                    if not np.isnan(data['uncertain'][istep]):
                        if int(data['uncertain'][istep]) == 1:
                            molecule_istep.uncertain = True
                        else:
                            molecule_istep.uncertain = False   
                if 'state_energies' in data.keys():
                    molecule_istep.electronic_states=[]
                    molecule_istep.electronic_states.extend([molecule_istep.copy() for _ in range(len(data['state_energies'][istep]))])
                    for i in range(0, len(data['state_energies'][istep])):
                        molecule_istep.electronic_states[i].energy = data['state_energies'][istep][i]
                        if 'aux_state_energies' in data.keys():
                            molecule_istep.electronic_states[i].aux_energy = data['aux_state_energies'][istep][i]
                if 'state_gradients' in data.keys():
                    if not molecule_istep.electronic_states:
                        molecule_istep.electronic_states=[]
                        molecule_istep.electronic_states.extend([molecule_istep.copy() for _ in range(len(data['state_gradients'][istep]))])
                    for i in range(0, len(data['state_gradients'][istep])):
                        if data['state_gradients'][istep][i] is not None:
                            molecule_istep.electronic_states[i].add_xyz_derivative_property(np.array(data['state_gradients'][istep][i]).astype(float), 'energy', 'energy_gradients')
                     
                    
                if 'nonadiabatic_coupling_vectors' in data.keys():
                    for iatom in range(Natoms):
                        molecule_istep.atoms[iatom].nonadiabatic_coupling_vectors = data['nonadiabatic_coupling_vectors'][istep][iatom]
                # kinetic_energy 
                # molecule_istep.kinetic_energy = data['kinetic_energy'][istep]
                # potential_energy 
                molecule_istep.energy = data['potential_energy'][istep]
                # total_energy
                molecule_istep.total_energy = data['total_energy'][istep]
                # dipole_moment
                if 'dipole_moment' in data.keys():
                    molecule_istep.dipole_moment = data['dipole_moment'][istep]
                trajectory_step.molecule = molecule_istep
                trajectory_step.step = istep 
                trajectory_step.time = data['time'][istep]
                # random_number
                if 'random_number' in data.keys():
                    trajectory_step.random_number = data['random_number'][istep]
                # prob
                if 'hopping_probabilities' in data.keys():
                    trajectory_step.hopping_probabilities = data['hopping_probabilities'][istep]
                # current_state
                if 'current_state' in data.keys():
                    trajectory_step.current_state = data['current_state'][istep]
                self.steps.append(trajectory_step)
        
        elif format.casefold() == 'json'.casefold():
            jsonfile = open(filename, 'r')
            data = json.load(jsonfile)
            self.steps = []
            for step in data['steps']:
                self.steps.append(molecular_trajectory_step(step=step['step'],
                                                            molecule=dict_to_molecule_class_instance(step['molecule'])))
                for key in step.keys():
                    if not key in ['step', 'molecule']:
                        self.steps[-1].__dict__[key] = step[key]
    
   
    def get_xyz_string(self) -> str:
        '''
        Return the XYZ string of the molecules in the trajectory.
        '''
        xyz_string = ''
        for istep in self.steps:
            xyz_string += istep.molecule.get_xyz_string()
        return xyz_string
    
    def to_database(self) -> molecular_database:
        '''
        Return a molecular database comprising the molecules in the trajectory.
        '''
        return molecular_database([step.molecule for step in self.steps])
    
    def view(self):
        '''
        Visualize the molecular trajectory. Uses ``py3Dmol``.
        '''
        moldb = self.to_database()
        moldb.view()

class molecular_trajectory_step(object):
    def __init__(self, step=None, molecule=None):
        self.step = step
        self.molecule = molecule
        # self.time     = None # added only in MD trajectories but not in optimization trajectories
        # Also includes velocities, temperature, total energy, potential energy, kinetic energy...

class h5md():
    """
    Saving trajectory data to file in `H5MD <http://dx.doi.org/10.1016/j.cpc.2014.01.018>`_ format

    Arguments:
        filename (str): The filename of the h5md file output.
        data (Dict): The data to be stored (optional, if provided, the file will be closed after storing data).
        mode (str, optional): A string that controls the file processing mode (default value: 'w' for a new file, 'r+' for an exisiting file). The choices are listed in the table below which is consistent with ``pyh5md.File()`` 

    .. table:: 
       :align: center

       ========  ================================================
        r        Readonly, file must exist 
        r+       Read/write, file must exist
        w        Create file, truncate if exists
        w- or x  Create file, fail if exists
       ========  ================================================
          
    Examples:

    .. code-block:: python

       traj0 = h5md('traj.h5')  # open 'traj.h5'
       traj1 = h5md('/tmp/test.h5', mode='r')  # open an existing file in readonly mode
       traj2 = h5md('/tmp/traj2.h5', data={'time': 1.0, 'total_energy': -32.1, 'test': 8848}) # add some data to the file, then close the file
      
       traj0.write(data) # write data to opened file
       traj0(data) # an alternative way to write data

       data = traj0.export() # export the data in the opened file
       data = traj0() # an alternative way to export data
       with h5md('test.h5') as traj: # export with a with statement
           data = traj.export()
           

       traj0.close() # close the file
    
    .. note::

       the default data path in HDF5 file

           particles/all:
               'box', 'gradients', 'mass', 'nad', 'names', 'position', 'species', 'velocities'
    
           observables:
               'angular_momentum', 'generated_random_number', 'kinetic_energy', 'linear_momentum', 'nstatdyn', 'oscillator_strengths', 'populations', 'potential_energy', 'random_seed', 'sh_probabilities', 'total_energy', 'wavefunctions', 

               and any other keywords

    Attributes:
        h5: the HDF5 file object
    """

    particles_properties =  [
                            'position',
                            'velocities',
                            'accelerations',
                            'gradients',
                            'nad',
                            'names',
                            ]
    fix_properties =    [
                        'species',
                        'mass',
                        ]

    def __init__(self, filename: str, data: Dict[str, Any] = {}, mode: str = 'w',) -> None:
        from pyh5md import File
        if os.path.isfile(filename):
            mode = 'r+'
        self.h5 = File(filename, mode)
        self.part = self.h5.particles_group('all')
        self.observables = self.h5.require_group('observables')
        self.properties = {}

        if 'box' not in self.part:
            self.part.create_box(dimension=3, boundary=['none','none','none'])
            self.step = 0
        else:
            self.step = self.part['position/step'][-1] + 1

        if data:
            self.write(data)
            self.close()

    def add_properties(self, key, value, shape):
        from pyh5md import element
        if key in self.fix_properties:
            self.properties[key] = element(self.part, key, data=value, store='fixed')
        elif key in self.particles_properties:
            self.properties[key] = element(self.part, key, store='time', time=True, shape=shape)
        else:
            self.properties[key] = element(self.observables, key, store='time', time=True, shape=shape)
        self.properties[key].own_step=True

    
    def write(self, data: Dict[str, Any]) -> None:
        '''
        Write data to the opened H5 file. Data should be a dictionary-like object with 'time' in its keys().
        '''
        time = array(data['time'])
        shape_offset = 1 if time.shape else 0
        for key, value in data.items():
            value=array(value)
            if key == 'time' or not value.size: 
                continue
            if key not in self.properties.keys():
                self.add_properties(key, value, value.shape[shape_offset:])
            if shape_offset and key not in self.fix_properties:
                for i in range(time.size):
                    self.properties[key].append(value[i], self.step + i, time[i])
            else:
                self.properties[key].append(value, self.step, time)
        self.step+= time.size if shape_offset else 1

    def export(self) -> Dict[str, np.ndarray]:
        '''
        Export the data in the opened H5 file. 
        
        Returns:
            A dictionary of the trajectory data in the H5 file.
        '''        
        import h5py
        data = {'time': self.part['position/time'][()]}
        for key in self.part.keys():
            if key == 'box':
                pass
            elif isinstance(self.part[key], h5py._hl.dataset.Dataset):
                data[key] = self.part[key][()]
            else:
                data[key] = self.part[key+'/value'][()]
        for key in self.observables.keys():
            data[key] = self.observables[key+'/value'][()]
        return data

    def close(self) -> None:
        '''
        Close the opened file.
        '''
        self.h5.close()

    __call__ = export

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class h5dataloader:
    
    def __init__(self, store_file):
        
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator

        Iterate through all groups in all branches and return datasets in dicts)
        """
        
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                # data = {'path': path}
                data = {}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])

                        if isinstance(dataset, np.ndarray):
                            if dataset.size != 0:
                                if isinstance(dataset[0], np.bytes_):
                                    dataset = [a.decode('ascii')
                                               for a in dataset]
                                if isinstance(dataset[0], bytes):
                                    dataset = [a.decode('utf-8')
                                               for a in dataset]
                        data.update({k: dataset})
                yield data
            else:  # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)"""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    # def get_size(self):
    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g['coordinates'][:])
        return count

def sample(molecular_database_to_split=None, sampling='random', number_of_splits=2, split_equally=None, fraction_of_points_in_splits=None):
    molDB = molecular_database_to_split
    Ntot = len(molDB)
    if number_of_splits==2 and fraction_of_points_in_splits==None and split_equally==None:
        split_equally = False
    if number_of_splits==2 and fraction_of_points_in_splits==None and not split_equally:
        fraction_of_points_in_splits = [0.8, 0.2]
    elif split_equally or (number_of_splits>2 and split_equally==None and fraction_of_points_in_splits==None):
        split_equally = True
        fraction_of_points_in_splits = [1 / number_of_splits for ii in range(number_of_splits)]
        
    splits_DBs = []
    number_of_points_in_splits = []
    if sum(fraction_of_points_in_splits) > 1.0+1e-5:
        raise ValueError('sum of ratios of splits is more than one')
    for yy in fraction_of_points_in_splits:
        number_of_points_in_splits.append(round(Ntot * yy))

    sumofpoints = sum(number_of_points_in_splits)
    residual = Ntot - sumofpoints
    if residual > 0: residual_sign = 1
    else: residual_sign = -1
    for ii in range(abs(residual)):
        number_of_points_in_splits[ii] += residual_sign

    all_indices = [ii for ii in range(Ntot)]
    if sampling.casefold() == 'random'.casefold():
        import random
        random.shuffle(all_indices)
    elif sampling.casefold() == 'user-defined'.casefold():
        number_of_points_in_splits = []
        all_indices = []
        for index in indices:
            all_indices += index
            number_of_points_in_splits.append(len(index))
    elif sampling.casefold() != 'none'.casefold():
        raise ValueError('unsupported sampling type')

    split_indices = []
    istart = 0
    iend = 0
    for ii in number_of_points_in_splits:
        istart = iend
        iend = istart + ii
        split_indices.append(all_indices[istart:iend])

    for isplit in split_indices:
        splits_DBs.append(molecular_database())
        for ii in isplit:
            splits_DBs[-1].molecules.append(molDB.molecules[ii])

    return splits_DBs

def array(data, *args, **kwargs):
    try:
        return np.array(object=data, *args, **kwargs)
    except:
        return np.array(object=data, dtype=object, *args, **kwargs)
    
def read_y_file(filename=''):
    # Reads a file with scalar values.
    # Returns:
    #   Ys     - list with Ys (FP number)
    Ys = []
    with open(filename, 'r') as fy:
        for line in fy:
            Ys.append(float(line))
    return Ys

def write_gaussian_log(molecule, filename):

    def write_freq_block(f, s):
        f.write('             '+''.join(s) + '\n')
        symmetry_normal_modes = ['N']*len(s)
        frequency = [f'{molecule.frequencies[int(ii)-1]:21.4f}' for ii in s]
        reduced_mass = [f'{molecule.reduced_masses[int(ii)-1]:21.4f}' for ii in s]
        force_constants = [f'{molecule.force_constants[int(ii)-1]:21.4f}' for ii in s]
        if 'infrared_intensities' in molecule.__dict__:
            ir_density = [f'{molecule.infrared_intensities[int(ii)-1]:21.4f}' for ii in s]
        else:
            ir_density = [f'{0.0000:21.4f}']*len(s)
        if 'raman_intensities' in molecule.__dict__:
            raman_intensities = [f'{molecule.raman_intensities[int(ii)-1]:21.4f}' for ii in s]
            depolar = [f'{0.0000:21.4f}']*len(s)
        else:
            raman_intensities = [f'{0.0000:21.4f}']*len(s)
            depolar = [f'{0.0000:21.4f}']*len(s)
        ints = [int(ii)-1 for ii in s]
        f.write('                                 '+'                    '.join(symmetry_normal_modes) + '\n')
        f.write(' Frequencies --' + ''.join(frequency) + '\n')
        f.write(' Red. masses --' + ''.join(reduced_mass) + '\n')
        f.write(' Frc consts  --' + ''.join(force_constants) + '\n')
        f.write(' IR Inten    --' + ''.join(ir_density) + '\n')
        if 'raman_intensities' in molecule.__dict__:
            f.write(' Raman Activ --' + ''.join(raman_intensities) + '\n')
            f.write(' Depolar (P) --' + ''.join(depolar) + '\n')
            f.write(' Depolar (U) --' + ''.join(depolar) + '\n')
        f.write('  Atom  AN                ' + '      '.join(['X      Y      Z']*len(s)) + '\n')
        for iatom in range(len(molecule.atoms)): 
            normal_modes = []
            for ii in ints:
                nm_str = [f'{nm:7.2f}' for nm in molecule.atoms[iatom].normal_modes[ii]]
                normal_modes.append(''.join(nm_str))
            f.write(f'  '+f'{iatom+1:3.0f}  {molecule.atoms[iatom].atomic_number:3.0f}            ' + ''.join(normal_modes) + '\n')

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    f = open(filename, 'w')
    f.write(''' Entering Gaussian System
 Output generated using Gaussian format for frequencies and geometries with MLatom (for visualization purposes only)
 ----------------------------------------------------------------------
 #freq model/mlatom
 ----------------------------------------------------------------------
''')
    f.write('''
                         Standard orientation:                         
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
''')
    for iatom in range(len(molecule.atoms)):
        atom = molecule.atoms[iatom]
        f.write(' %6d %10d %10d       %11.6f %11.6f %11.6f\n' % (iatom+1, atom.atomic_number, 0,
                    atom.xyz_coordinates[0], atom.xyz_coordinates[1], atom.xyz_coordinates[2]))
    f.write(''' ---------------------------------------------------------------------\n''')
    f.write('\n')
    nnormal_modes = len(molecule.frequencies)
    splits = list(chunks(list(range(nnormal_modes)),3))
    for s in splits:
        s = [f'{ii+1:21.0f}' for ii in s]
        # s = [f'{ii:11.0f}' for ii in s]
        write_freq_block(f, s)
    f.close()

class isotope:
    nuclear_charge = 0   # units: elementary charge;      type: int
    relative_isotopic_mass = 0.0  # units: relative isotopic mass; type: float        isotope_abundance = 0.0  # units: percentage, %;          type: float
    nuclear_spin = 0.0  # type: float

    def __init__(self, nuclear_charge, relative_isotopic_mass, isotope_abundance, nuclear_spin, H0=None, multiplicity=None):
        self.nuclear_charge = nuclear_charge
        self.relative_isotopic_mass = relative_isotopic_mass
        self.isotope_abundance = isotope_abundance
        self.nuclear_spin = nuclear_spin
        if H0 != None: self.H0 = H0 # Enthalpy of formation at 0 K
        if multiplicity != None: self.multiplicity = multiplicity

# Masses and abundances from https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl, 2022-12-08
# Spins from https://nmr.wsu.edu/nmr-periodic-table/ and https://wwwndc.jaea.go.jp/NuC/index.html, 2022-12-08

class isotopes:
    isotopes = [isotope(nuclear_charge=1,   relative_isotopic_mass=1.00782503223,  isotope_abundance=99.9885,   nuclear_spin=0.5, H0 = 52.102*constants.kcalpermol2Hartree, multiplicity=2),
    isotope(nuclear_charge=1, relative_isotopic_mass=2.01410177812,        isotope_abundance=0.0115,    nuclear_spin=1.0),
    isotope(nuclear_charge=1,   relative_isotopic_mass=3.0160492779,        isotope_abundance=0.0,       nuclear_spin=0.5),
    isotope(nuclear_charge=2,   relative_isotopic_mass=3.0160293201,        isotope_abundance=0.000134,  nuclear_spin=0.5),
    isotope(nuclear_charge=2,   relative_isotopic_mass=4.00260325413,        isotope_abundance=99.999866, nuclear_spin=0.0),
    isotope(nuclear_charge=3,   relative_isotopic_mass=6.0151228874,        isotope_abundance=7.59,      nuclear_spin=1.0),
    isotope(nuclear_charge=3,   relative_isotopic_mass=7.0160034366,        isotope_abundance=92.41,     nuclear_spin=-1.5),
    isotope(nuclear_charge=4,   relative_isotopic_mass=9.012183065,        isotope_abundance=100,       nuclear_spin=-1.5),
    isotope(nuclear_charge=5,   relative_isotopic_mass=10.01293695,        isotope_abundance=19.9,      nuclear_spin=3.0),
    isotope(nuclear_charge=5,   relative_isotopic_mass=11.00930536,        isotope_abundance=80.1,      nuclear_spin=-1.5),
    isotope(nuclear_charge=6,   relative_isotopic_mass=12.0000000,        isotope_abundance=98.93,     nuclear_spin=0.0, H0 = 170.89*constants.kcalpermol2Hartree, multiplicity=3),
    isotope(nuclear_charge=6,   relative_isotopic_mass=13.00335483507,        isotope_abundance=1.07,      nuclear_spin=-0.5),
    isotope(nuclear_charge=6,   relative_isotopic_mass=14.0032419884,        isotope_abundance=0.0,       nuclear_spin=0.0),
    isotope(nuclear_charge=7,   relative_isotopic_mass=14.00307400443,        isotope_abundance=99.636,    nuclear_spin=1.0, H0 = 113.00*constants.kcalpermol2Hartree, multiplicity=4),
    isotope(nuclear_charge=7,   relative_isotopic_mass=15.00010889888,        isotope_abundance=0.364,     nuclear_spin=-0.5),
    isotope(nuclear_charge=8,   relative_isotopic_mass=15.99491461957,        isotope_abundance=99.757,    nuclear_spin=0.0, H0 = 59.559*constants.kcalpermol2Hartree, multiplicity=3),
    isotope(nuclear_charge=8,   relative_isotopic_mass=16.99913175650,        isotope_abundance=0.038,     nuclear_spin=2.5),
    isotope(nuclear_charge=8,   relative_isotopic_mass=17.99915961286,        isotope_abundance=0.205,     nuclear_spin=0.0),
    isotope(nuclear_charge=9,   relative_isotopic_mass=18.99840316273,        isotope_abundance=100,       nuclear_spin=0.5),
    isotope(nuclear_charge=10,  relative_isotopic_mass=19.9924401762,        isotope_abundance=90.48,     nuclear_spin=0.0),
    isotope(nuclear_charge=10,  relative_isotopic_mass=20.993846685,        isotope_abundance=0.27,      nuclear_spin=1.5),
    isotope(nuclear_charge=10,  relative_isotopic_mass=21.991385114,        isotope_abundance=9.25,      nuclear_spin=0.0),
    isotope(nuclear_charge=11,  relative_isotopic_mass=22.9897692820,        isotope_abundance=100,       nuclear_spin=1.5),
    isotope(nuclear_charge=12,  relative_isotopic_mass=23.985041697,        isotope_abundance=78.99,     nuclear_spin=0.0),
    isotope(nuclear_charge=12,  relative_isotopic_mass=24.985836976,        isotope_abundance=10.00,     nuclear_spin=2.5),
    isotope(nuclear_charge=12,  relative_isotopic_mass=25.982592968,        isotope_abundance=11.01,     nuclear_spin=0.0),
    isotope(nuclear_charge=13,  relative_isotopic_mass=26.98153853,        isotope_abundance=100,       nuclear_spin=2.5),
    isotope(nuclear_charge=14,  relative_isotopic_mass=27.97692653465,        isotope_abundance=92.223,    nuclear_spin=0.0),
    isotope(nuclear_charge=14,  relative_isotopic_mass=28.97649466490,        isotope_abundance=4.685,     nuclear_spin=0.5),
    isotope(nuclear_charge=14,  relative_isotopic_mass=29.973770136,        isotope_abundance=3.092,     nuclear_spin=0.0),
    isotope(nuclear_charge=15,  relative_isotopic_mass=30.97376199842,        isotope_abundance=100,       nuclear_spin=0.5),
    isotope(nuclear_charge=16,  relative_isotopic_mass=31.9720711744,        isotope_abundance=94.99,     nuclear_spin=0.0),
    isotope(nuclear_charge=16,  relative_isotopic_mass=32.9714589098,        isotope_abundance=0.75,      nuclear_spin=1.5),
    isotope(nuclear_charge=16,  relative_isotopic_mass=33.967867004,        isotope_abundance=4.25,      nuclear_spin=0.0),
    isotope(nuclear_charge=16,  relative_isotopic_mass=35.96708071,        isotope_abundance=0.01,      nuclear_spin=0.0),
    isotope(nuclear_charge=17,  relative_isotopic_mass=34.968852682,        isotope_abundance=75.76,     nuclear_spin=1.5),
    isotope(nuclear_charge=17,  relative_isotopic_mass=36.965902602,        isotope_abundance=24.24,     nuclear_spin=1.5),
    isotope(nuclear_charge=18,  relative_isotopic_mass=35.967545105,        isotope_abundance=0.3336,    nuclear_spin=0.0),
    isotope(nuclear_charge=18,  relative_isotopic_mass=37.96273211,        isotope_abundance=0.0629,    nuclear_spin=0.0),
    isotope(nuclear_charge=18,  relative_isotopic_mass=39.9623831237,        isotope_abundance=99.6035,   nuclear_spin=0.0),
    isotope(nuclear_charge=19,  relative_isotopic_mass=38.9637064864,        isotope_abundance=93.2581,   nuclear_spin=1.5),
    isotope(nuclear_charge=19,  relative_isotopic_mass=39.963998166,        isotope_abundance=0.0117,    nuclear_spin=-4.0),
    isotope(nuclear_charge=19,  relative_isotopic_mass=40.9618252579,        isotope_abundance=6.7302,    nuclear_spin=1.5),
    isotope(nuclear_charge=20,  relative_isotopic_mass=39.962590863,        isotope_abundance=96.941,    nuclear_spin=0.0),
    isotope(nuclear_charge=20,  relative_isotopic_mass=41.95861783,        isotope_abundance=0.647,     nuclear_spin=0.0),
    isotope(nuclear_charge=20,  relative_isotopic_mass=42.95876644,        isotope_abundance=0.135,     nuclear_spin=-3.5),
    isotope(nuclear_charge=20,  relative_isotopic_mass=43.95548156,        isotope_abundance=2.086,     nuclear_spin=0.0),
    isotope(nuclear_charge=20,  relative_isotopic_mass=45.9536890,        isotope_abundance=0.004,     nuclear_spin=0.0),
    isotope(nuclear_charge=20,  relative_isotopic_mass=47.95252276,        isotope_abundance=0.187,     nuclear_spin=0.0),
    isotope(nuclear_charge=21,  relative_isotopic_mass=44.95590828,        isotope_abundance=100,       nuclear_spin=-3.5),
    isotope(nuclear_charge=22,  relative_isotopic_mass=45.95262772,        isotope_abundance=8.25,      nuclear_spin=0.0),
    isotope(nuclear_charge=22,  relative_isotopic_mass=46.95175879,        isotope_abundance=7.44,      nuclear_spin=-2.5),
    isotope(nuclear_charge=22,  relative_isotopic_mass=47.94794198,        isotope_abundance=73.72,     nuclear_spin=0.0),
    isotope(nuclear_charge=22,  relative_isotopic_mass=48.94786568,        isotope_abundance=5.41,      nuclear_spin=-3.5),
    isotope(nuclear_charge=22,  relative_isotopic_mass=49.94478689,        isotope_abundance=5.18,      nuclear_spin=0.0),
    isotope(nuclear_charge=23,  relative_isotopic_mass=49.94715601,        isotope_abundance=0.250,     nuclear_spin=6.0),
    isotope(nuclear_charge=23,  relative_isotopic_mass=50.94395704,        isotope_abundance=99.750,    nuclear_spin=-3.5),
    isotope(nuclear_charge=24,  relative_isotopic_mass=49.94604183,        isotope_abundance=4.345,     nuclear_spin=0.0),
    isotope(nuclear_charge=24,  relative_isotopic_mass=51.94050623,        isotope_abundance=83.789,    nuclear_spin=0.0),
    isotope(nuclear_charge=24,  relative_isotopic_mass=52.94064815,        isotope_abundance=9.501,     nuclear_spin=-1.5),
    isotope(nuclear_charge=24,  relative_isotopic_mass=53.93887916,        isotope_abundance=2.365,     nuclear_spin=0.0),
    isotope(nuclear_charge=25,  relative_isotopic_mass=54.93804391,        isotope_abundance=100,       nuclear_spin=-2.5),
    isotope(nuclear_charge=26,  relative_isotopic_mass=53.93960899,        isotope_abundance=5.845,     nuclear_spin=0.0),
    isotope(nuclear_charge=26,  relative_isotopic_mass=55.93493633,        isotope_abundance=91.754,    nuclear_spin=0.0),
    isotope(nuclear_charge=26,  relative_isotopic_mass=56.93539284,        isotope_abundance=2.119,     nuclear_spin=-0.5),
    isotope(nuclear_charge=26,  relative_isotopic_mass=57.93327443,        isotope_abundance=0.282,     nuclear_spin=0.0),
    isotope(nuclear_charge=27,  relative_isotopic_mass=58.93319429,        isotope_abundance=100,       nuclear_spin=-3.5),
    isotope(nuclear_charge=28,  relative_isotopic_mass=57.93534241,        isotope_abundance=68.077,    nuclear_spin=0.0),
    isotope(nuclear_charge=28,  relative_isotopic_mass=59.93078588,        isotope_abundance=26.223,    nuclear_spin=0.0),
    isotope(nuclear_charge=28,  relative_isotopic_mass=60.93105557,        isotope_abundance=1.1399,    nuclear_spin=-1.5),
    isotope(nuclear_charge=28,  relative_isotopic_mass=61.92834537,        isotope_abundance=3.6346,    nuclear_spin=0.0),
    isotope(nuclear_charge=28,  relative_isotopic_mass=63.92796682,        isotope_abundance=0.9255,    nuclear_spin=0.0),
    isotope(nuclear_charge=29,  relative_isotopic_mass=62.92959772,        isotope_abundance=69.15,     nuclear_spin=-1.5),
    isotope(nuclear_charge=29,  relative_isotopic_mass=64.92778970,        isotope_abundance=30.85,     nuclear_spin=-1.5),
    isotope(nuclear_charge=30,  relative_isotopic_mass=63.92914201,        isotope_abundance=49.17,     nuclear_spin=0.0),
    isotope(nuclear_charge=30,  relative_isotopic_mass=65.92603381,        isotope_abundance=27.73,     nuclear_spin=0.0),
    isotope(nuclear_charge=30,  relative_isotopic_mass=66.92712775,        isotope_abundance=4.04,      nuclear_spin=-2.5),
    isotope(nuclear_charge=30,  relative_isotopic_mass=67.92484455,        isotope_abundance=18.45,     nuclear_spin=0.0),
    isotope(nuclear_charge=30,  relative_isotopic_mass=69.9253192,        isotope_abundance=0.61,      nuclear_spin=0.0),
    isotope(nuclear_charge=31,  relative_isotopic_mass=68.9255735,        isotope_abundance=60.108,    nuclear_spin=-1.5),
    isotope(nuclear_charge=31,  relative_isotopic_mass=70.92470258,        isotope_abundance=39.892,    nuclear_spin=-1.5),
    isotope(nuclear_charge=32,  relative_isotopic_mass=69.92424875,        isotope_abundance=20.57,     nuclear_spin=0.0),
    isotope(nuclear_charge=32,  relative_isotopic_mass=71.922075826,        isotope_abundance=27.45,     nuclear_spin=0.0),
    isotope(nuclear_charge=32,  relative_isotopic_mass=72.923458956,        isotope_abundance=7.75,      nuclear_spin=4.5),
    isotope(nuclear_charge=32,  relative_isotopic_mass=73.921177761,        isotope_abundance=36.50,     nuclear_spin=0.0),
    isotope(nuclear_charge=32,  relative_isotopic_mass=75.921402726,        isotope_abundance=7.73,      nuclear_spin=0.0),
    isotope(nuclear_charge=33,  relative_isotopic_mass=74.92159457,        isotope_abundance=100,       nuclear_spin=-1.5),
    isotope(nuclear_charge=34,  relative_isotopic_mass=73.922475934,        isotope_abundance=0.89,      nuclear_spin=0.0),
    isotope(nuclear_charge=34,  relative_isotopic_mass=75.919213704,        isotope_abundance=9.37,      nuclear_spin=0.0),
    isotope(nuclear_charge=34,  relative_isotopic_mass=76.919914154,        isotope_abundance=7.63,      nuclear_spin=-0.5),
    isotope(nuclear_charge=34,  relative_isotopic_mass=77.91730928,        isotope_abundance=23.77,     nuclear_spin=0.0),
    isotope(nuclear_charge=34,  relative_isotopic_mass=79.9165218,        isotope_abundance=49.61,     nuclear_spin=0.0),
    isotope(nuclear_charge=34,  relative_isotopic_mass=81.9166995,        isotope_abundance=8.73,      nuclear_spin=0.0),
    isotope(nuclear_charge=35,  relative_isotopic_mass=78.9183376,        isotope_abundance=50.69,     nuclear_spin=-1.5),
    isotope(nuclear_charge=35,  relative_isotopic_mass=80.9162897,        isotope_abundance=49.31,     nuclear_spin=-1.5),
    isotope(nuclear_charge=36,  relative_isotopic_mass=77.92036494,        isotope_abundance=0.355,     nuclear_spin=0.0),
    isotope(nuclear_charge=36,  relative_isotopic_mass=79.91637808,        isotope_abundance=2.286,     nuclear_spin=0.0),
    isotope(nuclear_charge=36,  relative_isotopic_mass=81.91348273,        isotope_abundance=11.593,    nuclear_spin=0.0),
    isotope(nuclear_charge=36,  relative_isotopic_mass=82.91412716,        isotope_abundance=11.500,    nuclear_spin=4.5),
    isotope(nuclear_charge=36,  relative_isotopic_mass=83.9114977282,        isotope_abundance=56.987,    nuclear_spin=0.0),
    isotope(nuclear_charge=36,  relative_isotopic_mass=85.9106106269,        isotope_abundance=17.279,    nuclear_spin=0.0),
    isotope(nuclear_charge=37,  relative_isotopic_mass=84.9117897379,        isotope_abundance=72.17,     nuclear_spin=-2.5),
    isotope(nuclear_charge=37,  relative_isotopic_mass=86.9091805310,        isotope_abundance=27.83,     nuclear_spin=-1.5),
    isotope(nuclear_charge=38,  relative_isotopic_mass=83.9134191,        isotope_abundance=0.56,      nuclear_spin=0.0),
    isotope(nuclear_charge=38,  relative_isotopic_mass=85.9092606,        isotope_abundance=9.86,      nuclear_spin=0.0),
    isotope(nuclear_charge=38,  relative_isotopic_mass=86.9088775,        isotope_abundance=7.00,      nuclear_spin=4.5),
    isotope(nuclear_charge=38,  relative_isotopic_mass=87.9056125,        isotope_abundance=82.58,     nuclear_spin=0.0),
    isotope(nuclear_charge=39,  relative_isotopic_mass=88.9058403,        isotope_abundance=100,       nuclear_spin=-0.5),
    isotope(nuclear_charge=40,  relative_isotopic_mass=89.9046977,        isotope_abundance=51.45,     nuclear_spin=0.0),
    isotope(nuclear_charge=40,  relative_isotopic_mass=90.9056396,        isotope_abundance=11.22,     nuclear_spin=2.5),
    isotope(nuclear_charge=40,  relative_isotopic_mass=91.9050347,        isotope_abundance=17.15,     nuclear_spin=0.0),
    isotope(nuclear_charge=40,  relative_isotopic_mass=93.9063108,        isotope_abundance=17.38,     nuclear_spin=0.0),
    isotope(nuclear_charge=40,  relative_isotopic_mass=95.9082714,        isotope_abundance=2.80,      nuclear_spin=0.0),
    isotope(nuclear_charge=41,  relative_isotopic_mass=92.9063730,        isotope_abundance=100,       nuclear_spin=4.5),
    isotope(nuclear_charge=42,  relative_isotopic_mass=91.90680796,        isotope_abundance=14.53,     nuclear_spin=0.0),
    isotope(nuclear_charge=42,  relative_isotopic_mass=93.90508490,        isotope_abundance=9.15,      nuclear_spin=0.0),
    isotope(nuclear_charge=42,  relative_isotopic_mass=94.90583877,        isotope_abundance=15.84,     nuclear_spin=2.5),
    isotope(nuclear_charge=42,  relative_isotopic_mass=95.90467612,        isotope_abundance=16.67,     nuclear_spin=0.0),
    isotope(nuclear_charge=42,  relative_isotopic_mass=96.90601812,        isotope_abundance=9.60,      nuclear_spin=2.5),
    isotope(nuclear_charge=42,  relative_isotopic_mass=97.90540482,        isotope_abundance=24.39,     nuclear_spin=0.0),
    isotope(nuclear_charge=42,  relative_isotopic_mass=99.9074718,        isotope_abundance=9.82,      nuclear_spin=0.0),
    isotope(nuclear_charge=43,  relative_isotopic_mass=96.9063667,        isotope_abundance=0.0,       nuclear_spin=4.5),
    isotope(nuclear_charge=43,  relative_isotopic_mass=97.9072124,        isotope_abundance=0.0,       nuclear_spin=6.0),
    isotope(nuclear_charge=43,  relative_isotopic_mass=98.9062508,        isotope_abundance=0.0,       nuclear_spin=4.5),
    isotope(nuclear_charge=44,  relative_isotopic_mass=95.90759025,        isotope_abundance=5.54,      nuclear_spin=0.0),
    isotope(nuclear_charge=44,  relative_isotopic_mass=97.9052868,        isotope_abundance=1.87,      nuclear_spin=0.0),
    isotope(nuclear_charge=44,  relative_isotopic_mass=98.9059341,        isotope_abundance=12.76,     nuclear_spin=2.5),
    isotope(nuclear_charge=44,  relative_isotopic_mass=99.9042143,        isotope_abundance=12.60,     nuclear_spin=0.0),
    isotope(nuclear_charge=44,  relative_isotopic_mass=100.9055769,        isotope_abundance=17.06,     nuclear_spin=2.5),
    isotope(nuclear_charge=44,  relative_isotopic_mass=101.9043441,        isotope_abundance=31.55,     nuclear_spin=0.0),
    isotope(nuclear_charge=44,  relative_isotopic_mass=103.9054275,        isotope_abundance=18.62,     nuclear_spin=0.0),
    isotope(nuclear_charge=45,  relative_isotopic_mass=102.9054980,        isotope_abundance=100,       nuclear_spin=-0.5),
    isotope(nuclear_charge=46,  relative_isotopic_mass=101.9056022,        isotope_abundance=1.02,      nuclear_spin=0.0),
    isotope(nuclear_charge=46,  relative_isotopic_mass=103.9040305,        isotope_abundance=11.14,     nuclear_spin=0.0),
    isotope(nuclear_charge=46,  relative_isotopic_mass=104.9050796,        isotope_abundance=22.33,     nuclear_spin=2.5),
    isotope(nuclear_charge=46,  relative_isotopic_mass=105.9034804,        isotope_abundance=27.33,     nuclear_spin=0.0),
    isotope(nuclear_charge=46,  relative_isotopic_mass=107.9038916,        isotope_abundance=26.46,     nuclear_spin=0.0),
    isotope(nuclear_charge=46,  relative_isotopic_mass=109.90517220,        isotope_abundance=11.72,     nuclear_spin=0.0),
    isotope(nuclear_charge=47,  relative_isotopic_mass=106.9050916,        isotope_abundance=51.839,    nuclear_spin=-0.5),
    isotope(nuclear_charge=47,  relative_isotopic_mass=108.9047553,        isotope_abundance=48.161,    nuclear_spin=-0.5),
    isotope(nuclear_charge=48,  relative_isotopic_mass=105.9064599,        isotope_abundance=1.25,      nuclear_spin=0.0),
    isotope(nuclear_charge=48,  relative_isotopic_mass=107.9041834,        isotope_abundance=0.89,      nuclear_spin=0.0),
    isotope(nuclear_charge=48,  relative_isotopic_mass=109.90300661,        isotope_abundance=12.49,     nuclear_spin=0.0),
    isotope(nuclear_charge=48,  relative_isotopic_mass=110.90418287,        isotope_abundance=12.80,     nuclear_spin=0.5),
    isotope(nuclear_charge=48,  relative_isotopic_mass=111.90276287,        isotope_abundance=24.13,     nuclear_spin=0.0),
    isotope(nuclear_charge=48,  relative_isotopic_mass=112.90440813,        isotope_abundance=12.22,     nuclear_spin=0.5),
    isotope(nuclear_charge=48,  relative_isotopic_mass=113.90336509,        isotope_abundance=28.73,     nuclear_spin=0.0),
    isotope(nuclear_charge=48,  relative_isotopic_mass=115.90476315,        isotope_abundance=7.49,      nuclear_spin=0.0),
    isotope(nuclear_charge=49,  relative_isotopic_mass=112.90406184,        isotope_abundance=4.29,      nuclear_spin=4.5),
    isotope(nuclear_charge=49,  relative_isotopic_mass=114.903878776,        isotope_abundance=95.71,     nuclear_spin=4.5),
    isotope(nuclear_charge=50,  relative_isotopic_mass=111.90482387,        isotope_abundance=0.97,      nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=113.9027827,        isotope_abundance=0.66,      nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=114.903344699,        isotope_abundance=0.34,      nuclear_spin=0.5),
    isotope(nuclear_charge=50,  relative_isotopic_mass=115.90174280,        isotope_abundance=14.54,     nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=116.90295398,        isotope_abundance=7.68,      nuclear_spin=0.5),
    isotope(nuclear_charge=50,  relative_isotopic_mass=117.90160657,        isotope_abundance=24.22,     nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=118.90331117,        isotope_abundance=8.59,      nuclear_spin=0.5),
    isotope(nuclear_charge=50,  relative_isotopic_mass=119.90220163,        isotope_abundance=32.58,     nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=121.9034438,        isotope_abundance=4.63,      nuclear_spin=0.0),
    isotope(nuclear_charge=50,  relative_isotopic_mass=123.9052766,        isotope_abundance=5.79,      nuclear_spin=0.0),
    isotope(nuclear_charge=51,  relative_isotopic_mass=120.903812,        isotope_abundance=57.21,     nuclear_spin=2.5),
    isotope(nuclear_charge=51,  relative_isotopic_mass=122.9042132,        isotope_abundance=42.79,     nuclear_spin=3.5),
    isotope(nuclear_charge=52,  relative_isotopic_mass=119.9040593,        isotope_abundance=0.09,      nuclear_spin=0.0),
    isotope(nuclear_charge=52,  relative_isotopic_mass=121.9030435,        isotope_abundance=2.55,      nuclear_spin=0.0),
    isotope(nuclear_charge=52,  relative_isotopic_mass=122.9042698,        isotope_abundance=0.89,      nuclear_spin=0.5),
    isotope(nuclear_charge=52,  relative_isotopic_mass=123.9028171,        isotope_abundance=4.74,      nuclear_spin=0.0),
    isotope(nuclear_charge=52,  relative_isotopic_mass=124.9044299,        isotope_abundance=7.07,      nuclear_spin=0.5),
    isotope(nuclear_charge=52,  relative_isotopic_mass=125.9033109,        isotope_abundance=18.84,     nuclear_spin=0.0),
    isotope(nuclear_charge=52,  relative_isotopic_mass=127.90446128,        isotope_abundance=31.74,     nuclear_spin=0.0),
    isotope(nuclear_charge=52,  relative_isotopic_mass=129.906222748,        isotope_abundance=34.08,     nuclear_spin=0.0),
    isotope(nuclear_charge=53,  relative_isotopic_mass=126.9044719,        isotope_abundance=100,       nuclear_spin=2.5),
    isotope(nuclear_charge=54,  relative_isotopic_mass=123.9058920,        isotope_abundance=0.0952,    nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=125.9042983,        isotope_abundance=0.0890,    nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=127.9035310,        isotope_abundance=1.9102,    nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=128.9047808611,        isotope_abundance=26.4006,   nuclear_spin=0.5),
    isotope(nuclear_charge=54,  relative_isotopic_mass=129.903509349,        isotope_abundance=4.0710,    nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=130.90508406,        isotope_abundance=21.2324,   nuclear_spin=1.5),
    isotope(nuclear_charge=54,  relative_isotopic_mass=131.9041550856,        isotope_abundance=26.9086,   nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=133.90539466,        isotope_abundance=10.4357,   nuclear_spin=0.0),
    isotope(nuclear_charge=54,  relative_isotopic_mass=135.907214484,        isotope_abundance=8.8573,    nuclear_spin=0.0),
    isotope(nuclear_charge=55,  relative_isotopic_mass=132.9054519610,        isotope_abundance=100,       nuclear_spin=3.5),
    isotope(nuclear_charge=56,  relative_isotopic_mass=129.9063207,        isotope_abundance=0.106,     nuclear_spin=0.0),
    isotope(nuclear_charge=56,  relative_isotopic_mass=131.9050611,        isotope_abundance=0.101,     nuclear_spin=0.0),
    isotope(nuclear_charge=56,  relative_isotopic_mass=133.90450818,        isotope_abundance=2.417,     nuclear_spin=0.0),
    isotope(nuclear_charge=56,  relative_isotopic_mass=134.90568838,        isotope_abundance=6.592,     nuclear_spin=1.5),
    isotope(nuclear_charge=56,  relative_isotopic_mass=135.90457573,        isotope_abundance=7.854,     nuclear_spin=0.0),
    isotope(nuclear_charge=56,  relative_isotopic_mass=136.90582714,        isotope_abundance=11.232,    nuclear_spin=1.5),
    isotope(nuclear_charge=56,  relative_isotopic_mass=137.90524700,        isotope_abundance=71.698,    nuclear_spin=0.0),
    isotope(nuclear_charge=57,  relative_isotopic_mass=137.9071149,        isotope_abundance=0.08881,   nuclear_spin=5.0),
    isotope(nuclear_charge=57,  relative_isotopic_mass=138.9063563,        isotope_abundance=99.91119,  nuclear_spin=3.5),
    isotope(nuclear_charge=58,  relative_isotopic_mass=135.90712921,        isotope_abundance=0.185,     nuclear_spin=0.0),
    isotope(nuclear_charge=58,  relative_isotopic_mass=137.905991,        isotope_abundance=0.251,     nuclear_spin=0.0),
    isotope(nuclear_charge=58,  relative_isotopic_mass=139.9054431,        isotope_abundance=88.450,    nuclear_spin=0.0),
    isotope(nuclear_charge=58,  relative_isotopic_mass=141.9092504,        isotope_abundance=11.114,    nuclear_spin=0.0),
    isotope(nuclear_charge=59,  relative_isotopic_mass=140.9076576,        isotope_abundance=100,       nuclear_spin=2.5),
    isotope(nuclear_charge=60,  relative_isotopic_mass=141.9077290,        isotope_abundance=27.152,    nuclear_spin=0.0),
    isotope(nuclear_charge=60,  relative_isotopic_mass=142.9098200,        isotope_abundance=12.174,    nuclear_spin=-3.5),
    isotope(nuclear_charge=60,  relative_isotopic_mass=143.9100930,        isotope_abundance=23.798,    nuclear_spin=0.0),
    isotope(nuclear_charge=60,  relative_isotopic_mass=144.9125793,        isotope_abundance=8.293,     nuclear_spin=-3.5),
    isotope(nuclear_charge=60,  relative_isotopic_mass=145.9131226,        isotope_abundance=17.189,    nuclear_spin=0.0),
    isotope(nuclear_charge=60,  relative_isotopic_mass=147.9168993,        isotope_abundance=5.756,     nuclear_spin=0.0),
    isotope(nuclear_charge=60,  relative_isotopic_mass=149.9209022,        isotope_abundance=5.638,     nuclear_spin=0.0),
    isotope(nuclear_charge=61,  relative_isotopic_mass=144.9127559,        isotope_abundance=0.0,       nuclear_spin=2.5),
    isotope(nuclear_charge=61,  relative_isotopic_mass=146.9151450,        isotope_abundance=0.0,       nuclear_spin=3.5),
    isotope(nuclear_charge=62,  relative_isotopic_mass=143.9120065,        isotope_abundance=3.07,      nuclear_spin=0.0),
    isotope(nuclear_charge=62,  relative_isotopic_mass=146.9149044,        isotope_abundance=14.99,     nuclear_spin=-3.5),
    isotope(nuclear_charge=62,  relative_isotopic_mass=147.9148292,        isotope_abundance=11.24,     nuclear_spin=0.0),
    isotope(nuclear_charge=62,  relative_isotopic_mass=148.9171921,        isotope_abundance=13.82,     nuclear_spin=-3.5),
    isotope(nuclear_charge=62,  relative_isotopic_mass=149.9172829,        isotope_abundance=7.38,      nuclear_spin=0.0),
    isotope(nuclear_charge=62,  relative_isotopic_mass=151.9197397,        isotope_abundance=26.75,     nuclear_spin=0.0),
    isotope(nuclear_charge=62,  relative_isotopic_mass=153.9222169,        isotope_abundance=22.75,     nuclear_spin=0.0),
    isotope(nuclear_charge=63,  relative_isotopic_mass=150.9198578,        isotope_abundance=47.81,     nuclear_spin=2.5),
    isotope(nuclear_charge=63,  relative_isotopic_mass=152.9212380,        isotope_abundance=52.19,     nuclear_spin=2.5),
    isotope(nuclear_charge=64,  relative_isotopic_mass=151.9197995,        isotope_abundance=0.20,      nuclear_spin=0.0),
    isotope(nuclear_charge=64,  relative_isotopic_mass=153.9208741,        isotope_abundance=2.18,      nuclear_spin=0.0),
    isotope(nuclear_charge=64,  relative_isotopic_mass=154.9226305,        isotope_abundance=14.80,     nuclear_spin=-1.5),
    isotope(nuclear_charge=64,  relative_isotopic_mass=155.9221312,        isotope_abundance=20.47,     nuclear_spin=0.0),
    isotope(nuclear_charge=64,  relative_isotopic_mass=156.9239686,        isotope_abundance=15.65,     nuclear_spin=-1.5),
    isotope(nuclear_charge=64,  relative_isotopic_mass=157.9241123,        isotope_abundance=24.84,     nuclear_spin=0.0),
    isotope(nuclear_charge=64,  relative_isotopic_mass=159.9270624,        isotope_abundance=21.86,     nuclear_spin=0.0),
    isotope(nuclear_charge=65,  relative_isotopic_mass=158.9253547,        isotope_abundance=100,       nuclear_spin=1.5),
    isotope(nuclear_charge=66,  relative_isotopic_mass=155.9242847,        isotope_abundance=0.056,     nuclear_spin=0.0),
    isotope(nuclear_charge=66,  relative_isotopic_mass=157.9244159,        isotope_abundance=0.095,     nuclear_spin=0.0),
    isotope(nuclear_charge=66,  relative_isotopic_mass=159.9252046,        isotope_abundance=2.329,     nuclear_spin=0.0),
    isotope(nuclear_charge=66,  relative_isotopic_mass=160.9269405,        isotope_abundance=18.889,    nuclear_spin=2.5),
    isotope(nuclear_charge=66,  relative_isotopic_mass=161.9268056,        isotope_abundance=25.475,    nuclear_spin=0.0),
    isotope(nuclear_charge=66,  relative_isotopic_mass=162.9287383,        isotope_abundance=24.896,    nuclear_spin=-2.5),
    isotope(nuclear_charge=66,  relative_isotopic_mass=163.9291819,        isotope_abundance=28.260,    nuclear_spin=0.0),
    isotope(nuclear_charge=67,  relative_isotopic_mass=164.9303288,        isotope_abundance=100,       nuclear_spin=-3.5),
    isotope(nuclear_charge=68,  relative_isotopic_mass=161.9287884,        isotope_abundance=0.139,     nuclear_spin=0.0),
    isotope(nuclear_charge=68,  relative_isotopic_mass=163.9292088,        isotope_abundance=1.601,     nuclear_spin=0.0),
    isotope(nuclear_charge=68,  relative_isotopic_mass=165.9302995,        isotope_abundance=33.503,    nuclear_spin=0.0),
    isotope(nuclear_charge=68,  relative_isotopic_mass=166.9320546,        isotope_abundance=22.869,    nuclear_spin=3.5),
    isotope(nuclear_charge=68,  relative_isotopic_mass=167.9323767,        isotope_abundance=26.978,    nuclear_spin=0.0),
    isotope(nuclear_charge=68,  relative_isotopic_mass=169.9354702,        isotope_abundance=14.910,    nuclear_spin=0.0),
    isotope(nuclear_charge=69,  relative_isotopic_mass=168.9342179,        isotope_abundance=100,       nuclear_spin=0.5),
    isotope(nuclear_charge=70,  relative_isotopic_mass=167.9338896,        isotope_abundance=0.123,     nuclear_spin=0.0),
    isotope(nuclear_charge=70,  relative_isotopic_mass=169.9347664,        isotope_abundance=2.982,     nuclear_spin=0.0),
    isotope(nuclear_charge=70,  relative_isotopic_mass=170.9363302,        isotope_abundance=14.09,     nuclear_spin=-0.5),
    isotope(nuclear_charge=70,  relative_isotopic_mass=171.9363859,        isotope_abundance=21.68,     nuclear_spin=0.0),
    isotope(nuclear_charge=70,  relative_isotopic_mass=172.9382151,        isotope_abundance=16.103,    nuclear_spin=-2.5),
    isotope(nuclear_charge=70,  relative_isotopic_mass=173.9388664,        isotope_abundance=32.026,    nuclear_spin=0.0),
    isotope(nuclear_charge=70,  relative_isotopic_mass=175.9425764,        isotope_abundance=12.996,    nuclear_spin=0.0),
    isotope(nuclear_charge=71,  relative_isotopic_mass=174.9407752,        isotope_abundance=97.401,    nuclear_spin=3.5),
    isotope(nuclear_charge=71,  relative_isotopic_mass=175.9426897,        isotope_abundance=2.599,     nuclear_spin=-7.0),
    isotope(nuclear_charge=72,  relative_isotopic_mass=173.9400461,        isotope_abundance=0.16,      nuclear_spin=0.0),
    isotope(nuclear_charge=72,  relative_isotopic_mass=175.9414076,        isotope_abundance=5.26,      nuclear_spin=0.0),
    isotope(nuclear_charge=72,  relative_isotopic_mass=176.9432277,        isotope_abundance=18.60,     nuclear_spin=-3.5),
    isotope(nuclear_charge=72,  relative_isotopic_mass=177.9437058,        isotope_abundance=27.28,     nuclear_spin=0.0),
    isotope(nuclear_charge=72,  relative_isotopic_mass=178.9458232,        isotope_abundance=13.62,     nuclear_spin=4.5),
    isotope(nuclear_charge=72,  relative_isotopic_mass=179.9465570,        isotope_abundance=35.08,     nuclear_spin=0.0),
    isotope(nuclear_charge=73,  relative_isotopic_mass=179.9474648,        isotope_abundance=0.01201,   nuclear_spin=-9.0),
    isotope(nuclear_charge=73,  relative_isotopic_mass=180.9479958,        isotope_abundance=99.98799,  nuclear_spin=3.5),
    isotope(nuclear_charge=74,  relative_isotopic_mass=179.9467108,        isotope_abundance=0.12,      nuclear_spin=0.0),
    isotope(nuclear_charge=74,  relative_isotopic_mass=181.94820394,        isotope_abundance=26.50,     nuclear_spin=0.0),
    isotope(nuclear_charge=74,  relative_isotopic_mass=182.95022275,        isotope_abundance=14.31,     nuclear_spin=-0.5),
    isotope(nuclear_charge=74,  relative_isotopic_mass=183.95093092,        isotope_abundance=30.64,     nuclear_spin=0.0),
    isotope(nuclear_charge=74,  relative_isotopic_mass=185.9543628,        isotope_abundance=28.43,     nuclear_spin=0.0),
    isotope(nuclear_charge=75,  relative_isotopic_mass=184.9529545,        isotope_abundance=37.40,     nuclear_spin=2.5),
    isotope(nuclear_charge=75,  relative_isotopic_mass=186.9557501,        isotope_abundance=62.60,     nuclear_spin=2.5),
    isotope(nuclear_charge=76,  relative_isotopic_mass=183.9524885,        isotope_abundance=0.02,      nuclear_spin=0.0),
    isotope(nuclear_charge=76,  relative_isotopic_mass=185.9538350,        isotope_abundance=1.59,      nuclear_spin=0.0),
    isotope(nuclear_charge=76,  relative_isotopic_mass=186.9557474,        isotope_abundance=1.96,      nuclear_spin=-0.5),
    isotope(nuclear_charge=76,  relative_isotopic_mass=187.9558352,        isotope_abundance=13.24,     nuclear_spin=0.0),
    isotope(nuclear_charge=76,  relative_isotopic_mass=188.9581442,        isotope_abundance=16.15,     nuclear_spin=-1.5),
    isotope(nuclear_charge=76,  relative_isotopic_mass=189.9584437,        isotope_abundance=26.26,     nuclear_spin=0.0),
    isotope(nuclear_charge=76,  relative_isotopic_mass=191.9614770,        isotope_abundance=40.78,     nuclear_spin=0.0),
    isotope(nuclear_charge=77,  relative_isotopic_mass=190.9605893,        isotope_abundance=37.3,      nuclear_spin=1.5),
    isotope(nuclear_charge=77,  relative_isotopic_mass=192.9629216,        isotope_abundance=62.7,      nuclear_spin=1.5),
    isotope(nuclear_charge=78,  relative_isotopic_mass=189.9599297,        isotope_abundance=0.012,     nuclear_spin=0.0),
    isotope(nuclear_charge=78,  relative_isotopic_mass=191.9610387,        isotope_abundance=0.782,     nuclear_spin=0.0),
    isotope(nuclear_charge=78,  relative_isotopic_mass=193.9626809,        isotope_abundance=32.86,     nuclear_spin=0.0),
    isotope(nuclear_charge=78,  relative_isotopic_mass=194.9647917,        isotope_abundance=33.78,     nuclear_spin=-0.5),
    isotope(nuclear_charge=78,  relative_isotopic_mass=195.96495209,        isotope_abundance=25.21,     nuclear_spin=0.0),
    isotope(nuclear_charge=78,  relative_isotopic_mass=197.9678949,        isotope_abundance=7.356,     nuclear_spin=0.0),
    isotope(nuclear_charge=79,  relative_isotopic_mass=196.96656879,        isotope_abundance=100,       nuclear_spin=1.5),
    isotope(nuclear_charge=80,  relative_isotopic_mass=195.9658326,        isotope_abundance=0.15,      nuclear_spin=0.0),
    isotope(nuclear_charge=80,  relative_isotopic_mass=197.96676860,        isotope_abundance=9.97,      nuclear_spin=0.0),
    isotope(nuclear_charge=80,  relative_isotopic_mass=198.96828064,        isotope_abundance=16.87,     nuclear_spin=-0.5),
    isotope(nuclear_charge=80,  relative_isotopic_mass=199.96832659,        isotope_abundance=23.10,     nuclear_spin=0.0),
    isotope(nuclear_charge=80,  relative_isotopic_mass=200.97030284,        isotope_abundance=13.18,     nuclear_spin=-1.5),
    isotope(nuclear_charge=80,  relative_isotopic_mass=201.97064340,        isotope_abundance=29.86,     nuclear_spin=0.0),
    isotope(nuclear_charge=80,  relative_isotopic_mass=203.97349398,        isotope_abundance=6.87,      nuclear_spin=0.0),
    isotope(nuclear_charge=81,  relative_isotopic_mass=202.9723446,        isotope_abundance=29.52,     nuclear_spin=0.5),
    isotope(nuclear_charge=81,  relative_isotopic_mass=204.9744278,        isotope_abundance=70.48,     nuclear_spin=0.5),
    isotope(nuclear_charge=82,  relative_isotopic_mass=203.9730440,        isotope_abundance=1.4,       nuclear_spin=0.0),
    isotope(nuclear_charge=82,  relative_isotopic_mass=205.9744657,        isotope_abundance=24.1,      nuclear_spin=0.0),
    isotope(nuclear_charge=82,  relative_isotopic_mass=206.9758973,        isotope_abundance=22.1,      nuclear_spin=-0.5),
    isotope(nuclear_charge=82,  relative_isotopic_mass=207.9766525,        isotope_abundance=52.4,      nuclear_spin=0.0),
    isotope(nuclear_charge=83,  relative_isotopic_mass=208.9803991,        isotope_abundance=100,       nuclear_spin=-4.5),
    isotope(nuclear_charge=84,  relative_isotopic_mass=208.9824308,        isotope_abundance=0.0,       nuclear_spin=-0.5),
    isotope(nuclear_charge=84,  relative_isotopic_mass=209.9828741,        isotope_abundance=0.0,       nuclear_spin=0.0),
    isotope(nuclear_charge=85,  relative_isotopic_mass=209.9871479,        isotope_abundance=0.0,       nuclear_spin=5.0),
    isotope(nuclear_charge=85,  relative_isotopic_mass=210.9874966,        isotope_abundance=0.0,     nuclear_spin=-4.5),
    isotope(nuclear_charge=86,  relative_isotopic_mass=210.9906011,        isotope_abundance=0.0,     nuclear_spin=-0.5),
    isotope(nuclear_charge=86,  relative_isotopic_mass=220.0113941,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=86,  relative_isotopic_mass=222.0175782,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=87,  relative_isotopic_mass=223.0197360,        isotope_abundance=0.0,     nuclear_spin=-1.5),
    isotope(nuclear_charge=88,  relative_isotopic_mass=223.0185023,        isotope_abundance=0.0,     nuclear_spin=1.5),
    isotope(nuclear_charge=88,  relative_isotopic_mass=224.0202120,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=88,  relative_isotopic_mass=226.0254103,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=88,  relative_isotopic_mass=228.0310707,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=89,  relative_isotopic_mass=227.0277523,        isotope_abundance=0.0,     nuclear_spin=-1.5),
    isotope(nuclear_charge=90,  relative_isotopic_mass=230.0331341,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=90,  relative_isotopic_mass=232.0380558,        isotope_abundance=100,     nuclear_spin=0.0),
    isotope(nuclear_charge=91,  relative_isotopic_mass=231.0358842,        isotope_abundance=100,     nuclear_spin=-1.5),
    isotope(nuclear_charge=92,  relative_isotopic_mass=233.0396355,        isotope_abundance=0.0,     nuclear_spin=2.5),
    isotope(nuclear_charge=92,  relative_isotopic_mass=234.0409523,        isotope_abundance=0.0054,  nuclear_spin=0.0),
    isotope(nuclear_charge=92,  relative_isotopic_mass=235.0439301,        isotope_abundance=0.7204,  nuclear_spin=-3.5),
    isotope(nuclear_charge=92,  relative_isotopic_mass=236.0455682,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=92,  relative_isotopic_mass=238.0507884,        isotope_abundance=99.2742, nuclear_spin=0.0),
    isotope(nuclear_charge=93,  relative_isotopic_mass=236.046570,        isotope_abundance=0.0,     nuclear_spin=-6.0),
    isotope(nuclear_charge=93,  relative_isotopic_mass=237.0481736,        isotope_abundance=0.0,     nuclear_spin=2.5),
    isotope(nuclear_charge=94,  relative_isotopic_mass=238.0495601,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=94,  relative_isotopic_mass=239.0521636,        isotope_abundance=0.0,     nuclear_spin=0.5),
    isotope(nuclear_charge=94,  relative_isotopic_mass=240.0538138,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=94,  relative_isotopic_mass=241.0568517,        isotope_abundance=0.0,     nuclear_spin=2.5),
    isotope(nuclear_charge=94,  relative_isotopic_mass=242.0587428,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=94,  relative_isotopic_mass=244.0642053,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=95,  relative_isotopic_mass=241.0568293,        isotope_abundance=0.0,     nuclear_spin=-2.5),
    isotope(nuclear_charge=95,  relative_isotopic_mass=243.0613813,        isotope_abundance=0.0,     nuclear_spin=-2.5),
    isotope(nuclear_charge=96,  relative_isotopic_mass=243.0613893,        isotope_abundance=0.0,     nuclear_spin=2.5),
    isotope(nuclear_charge=96,  relative_isotopic_mass=244.0627528,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=96,  relative_isotopic_mass=245.0654915,        isotope_abundance=0.0,     nuclear_spin=3.5),
    isotope(nuclear_charge=96,  relative_isotopic_mass=246.0672238,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=96,  relative_isotopic_mass=247.0703541,        isotope_abundance=0.0,     nuclear_spin=-4.5),
    isotope(nuclear_charge=96,  relative_isotopic_mass=248.0723499,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=97,  relative_isotopic_mass=247.070307,        isotope_abundance=0.0,     nuclear_spin=-1.5),
    isotope(nuclear_charge=97,  relative_isotopic_mass=249.0749877,        isotope_abundance=0.0,     nuclear_spin=3.5),
    isotope(nuclear_charge=98,  relative_isotopic_mass=249.0748539,        isotope_abundance=0.0,     nuclear_spin=-4.5),
    isotope(nuclear_charge=98,  relative_isotopic_mass=250.0764062,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=98,  relative_isotopic_mass=251.0795886,        isotope_abundance=0.0,     nuclear_spin=0.5),
    isotope(nuclear_charge=98,  relative_isotopic_mass=252.0816272,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=99,  relative_isotopic_mass=252.082980,        isotope_abundance=0.0,     nuclear_spin=-5.0),
    isotope(nuclear_charge=100, relative_isotopic_mass=257.0951061,        isotope_abundance=0.0,     nuclear_spin=4.5),
    isotope(nuclear_charge=101, relative_isotopic_mass=258.0984315,        isotope_abundance=0.0,     nuclear_spin=-8.0),
    isotope(nuclear_charge=101, relative_isotopic_mass=260.10365,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=102, relative_isotopic_mass=259.10103,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=103, relative_isotopic_mass=262.10961,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=104, relative_isotopic_mass=267.12179,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=105, relative_isotopic_mass=268.12567,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=106, relative_isotopic_mass=271.13393,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=107, relative_isotopic_mass=272.13826,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=108, relative_isotopic_mass=270.13429,        isotope_abundance=0.0,     nuclear_spin=0.0),
    isotope(nuclear_charge=109, relative_isotopic_mass=276.15159,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=110, relative_isotopic_mass=281.16451,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=111, relative_isotopic_mass=280.16514,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=112, relative_isotopic_mass=285.17712,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=113, relative_isotopic_mass=284.17873,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=114, relative_isotopic_mass=289.19042,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=115, relative_isotopic_mass=288.19274,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=116, relative_isotopic_mass=293.20449,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=117, relative_isotopic_mass=292.20746,        isotope_abundance=0.0,     nuclear_spin=None),
    isotope(nuclear_charge=118, relative_isotopic_mass=294.21392,      isotope_abundance=0.0,     nuclear_spin=0.0)]

    @classmethod
    def get_most_abundant_with_given_nuclear_charge(cls, nuclear_charge):
        # return cls.isotopes[0]
        most_abundant_isotope = None
        for ii in cls.isotopes:
            if ii.nuclear_charge == nuclear_charge:
                if most_abundant_isotope:
                    if ii.isotope_abundance > most_abundant_isotope.isotope_abundance:
                        most_abundant_isotope = ii
                else:
                    most_abundant_isotope = ii
        return most_abundant_isotope

    @classmethod
    def get_most_similar_isotope_given_nuclear_charge_and_mass(cls, nuclear_charge, nuclear_mass):
        # return cls.isotopes[0]
        most_similar_isotope = None
        for ii in cls.isotopes:
            if ii.nuclear_charge == nuclear_charge and abs(ii.relative_isotopic_mass - nuclear_mass) < 0.2:
                most_similar_isotope = ii
        return most_similar_isotope

    @classmethod 
    def get_isotopes_with_given_nuclear_charge(cls, nuclear_charge):
        return [each for each in cls.isotopes if each.nuculear_charge == nuclear_charge]
    
if __name__ == '__main__':
    print(__doc__)
