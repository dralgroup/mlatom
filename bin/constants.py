'''
  !---------------------------------------------------------------------------! 
  ! constants: physical constants and conversion factors                      ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 
'''

Bohr2Angstrom = 0.52917721092 # Peter J. Mohr, Barry N. Taylor, David B. Newell,
                              # CODATA Recommended Values of the
                              # Fundamental Physical Constants: 2010, NIST, 2012.
Angstrom2Bohr = 1.0 / Bohr2Angstrom

Gas_constant = 8.3142 # Gas constant J K^-1 mol^-1
Calorie2Joule = 4.18585182085
Joule2Calorie = 1.0 / Calorie2Joule
Hartree2kcalpermol = 627.509474
kcalpermol2Hartree = 1.0 / Hartree2kcalpermol
Avogadro_constant = 6.02214076E23
kb = 1.380649E-23 # J/K
planck_constant = 6.62607015E-34 # Unit: J·s
kb_in_Hartree = kb * Joule2Calorie / 1000.0 * kcalpermol2Hartree * Avogadro_constant # Hartree / K
ram2au = 1822.888515 # Relative atomics mass
au2ram = 1.0 / ram2au
au2fs = 2.418884254E-2
fs2au = 1.0 / au2fs