
  !---------------------------------------------------------------------------! 
  ! constants: physical constants and conversion factors                      ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module constants
  use precision, only : rprec
  implicit none

  real(kind=rprec), parameter :: Bohr2Angstrom = 0.52917721092_rprec             ! Peter J. Mohr, Barry N. Taylor, David B. Newell,
                                                                                 ! CODATA Recommended Values of the
                                                                                 ! Fundamental Physical Constants: 2010, NIST, 2012.
  real(kind=rprec), parameter :: Angstrom2Bohr = 1.0_rprec / Bohr2Angstrom
  real(kind=rprec), parameter :: pi = 4.0_rprec*atan(1.0_rprec)                  ! pi constant

end module constants
