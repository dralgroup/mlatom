'''
  !---------------------------------------------------------------------------! 
  ! constants: physical constants and conversion factors                      ! 
  ! Implementations by: Pavlo O. Dral and Yi-Fan Hou                          !
  !---------------------------------------------------------------------------! 
'''
import numpy as np
from scipy.constants import physical_constants

Bohr2Angstrom =  0.52917721092# Peter J. Mohr, Barry N. Taylor, David B. Newell,
                              # CODATA Recommended Values of the
                              # Fundamental Physical Constants: 2010, NIST, 2012.
Angstrom2Bohr = 1.0 / Bohr2Angstrom

gas_constant = 8.3142 # Gas constant J K^-1 mol^-1
cal2Joule = 4.18585182085
Joule2cal = 1.0 / cal2Joule
Hartree2kcalpermol = 627.509474
kcalpermol2Hartree = 1.0 / Hartree2kcalpermol
kJpermol2Hartree = Joule2cal*kcalpermol2Hartree
J2hartree = 2.2937104486906E+17
Avogadro_constant = 6.02214076E23
kB = 1.380649E-23 # Unit: J/K https://en.wikipedia.org/wiki/Boltzmann_constant
planck_constant = 6.62607015E-34 # Unit: J·s
kB_in_Hartree = kB * Joule2cal / 1000.0 * kcalpermol2Hartree * Avogadro_constant # Hartree / K
ram2au = 1822.888515 # Relative atomics mass (it should be atomic mass constant)
au2ram = 1.0 / ram2au
au2fs = 2.4188843265857E-2 #https://physics.nist.gov/cgi-bin/cuu/Value?aut%7Csearch_for=time
fs2au = 1.0 / au2fs
Debye2au = 0.393456
electron_charge = 1.602176634E-19 # Unit: Coulomb, https://www.britannica.com/science/electron-charge
electron_mass    = 9.10938215E-31      # electron mass kg
eV2Joule = electron_charge 
Joule2eV = 1.0 / eV2Joule
eV2kJpermol = eV2Joule * Avogadro_constant / 1000
kJpermol2eV = 1.0 / eV2kJpermol
hartree2eV = 27.2114079527
Hartree2eV = hartree2eV
eV2hartree = 1.0 / hartree2eV
eV2Hartree = eV2hartree
eV2kcalpermol = eV2kJpermol * Joule2cal
kcalpermol2eV = 1.0 / eV2kcalpermol
speed_of_light = 2.99792458E8 # Unit: m/s, https://en.wikipedia.org/wiki/Speed_of_light 
eps0      = 8.854187817e-12     # vacuum permitivity C^2*s^2*kg^-1*m^-3
amu2kg = physical_constants["atomic mass constant"][0]
au2kg = amu2kg / ram2au
fine_structure_constant = physical_constants["fine-structure constant"][0]
Debye = 0.20819433442462576 # Unit: electron_charge / Angstrom
au2Angstrom4byamu = 1.065693E-31 * 8.077608721857700E+019 * 16605390400000.0

ev_nm_conversion_constant = planck_constant * Joule2eV * speed_of_light * 1E9
hartree_nm_conversion_constant = ev_nm_conversion_constant / hartree2eV
def eV2nm(eV):
  return ev_nm_conversion_constant / eV
def nm2eV(nm):
  return ev_nm_conversion_constant / nm
def hartree2nm(hartree):
  return hartree_nm_conversion_constant / hartree
def nm2hartree(nm):
  return hartree_nm_conversion_constant / nm