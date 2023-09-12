
  !---------------------------------------------------------------------------! 
  ! D_rel2eq: RE (relative to equilibrium) descriptor construction            ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module D_ID
  use mathUtils,     only : Rij
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  implicit none

contains

subroutine calcID(Xsize, Nat, xyzAngstrom, Xvec, xyzSorted)
  use dataset,   only : NAtomsMax
  implicit none
  ! Arguments
  integer,          intent(in)              :: Xsize                        ! Size of the X array
  integer,          intent(in)              :: Nat                          ! Number of atoms
  real(kind=rprec), intent(in)              :: xyzAngstrom(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  real(kind=rprec), intent(inout)           :: Xvec(1:Xsize)                ! Input vector
  real(kind=rprec), intent(inout), optional :: xyzSorted(1:3,1:NAtomsMax)   ! Sorted XYZ coordinates
  ! Variables
  integer          :: ii, jj, itemp, Error
  
  ! Initialize
  Xvec = 0.0_rprec

  itemp = 0
  do ii=1,Nat-1
    do jj=ii+1,Nat
      itemp = itemp + 1
      Xvec(itemp) = 1.0_rprec / Rij(xyzAngstrom(:,ii),xyzAngstrom(:,jj))
    end do
  end do
  
end subroutine calcID

end module D_ID
