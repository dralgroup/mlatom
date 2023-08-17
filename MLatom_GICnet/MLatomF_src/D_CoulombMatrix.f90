
  !---------------------------------------------------------------------------! 
  ! D_CoulombMatrix: Coulomb matrix construction and vectorization            ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module D_CoulombMatrix
  use mathUtils, only : Rij
  use precision, only : rprec
  implicit none

contains

subroutine calcCM(Xsize, Nat, charges, xyzAngstrom, Xvec)
  use constants,     only : Angstrom2Bohr
  use dataset,       only : NAtomsMax
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer,          intent(in)    :: Xsize                        ! Size of the X array
  integer,          intent(in)    :: Nat                          ! Number of atoms
  integer,          intent(in)    :: charges(1:NAtomsMax)         ! Nuclear charges
  real(kind=rprec), intent(in)    :: xyzAngstrom(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  real(kind=rprec), intent(inout) :: Xvec(1:Xsize)                ! vectorized Coulomb matrix
  ! Variables
  integer          :: ii, jj, Error
  ! Arrays
  real(kind=rprec), allocatable :: CMat(:,:)    ! Coulomb matrix
  real(kind=rprec), allocatable :: xyzBohr(:,:) ! Nuclear coordinates in Bohr

  !Allocate arrays
  allocate(CMat(1:NAtomsMax,1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate space for CMat array')
  allocate(xyzBohr(1:3,1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate space for xyzBohr')

  CMat = 0.0_rprec  
  xyzBohr = xyzAngstrom * Angstrom2Bohr

!  if (option%gradForAtom == 0) then
    do ii=1,Nat
      CMat(ii,ii) = 0.5_rprec * (dble(charges(ii)) ** 2.4_rprec)
      do jj=ii+1,Nat
        CMat(ii,jj) = dble(charges(ii)) * dble(charges(jj)) / Rij(xyzBohr(:,ii),xyzBohr(:,jj))
        CMat(jj,ii) = CMat(ii,jj)
      end do
    end do
!  elseif (.false.) then ! Projected CM
!    do ii=1,Nat
!      CMat(ii,ii) = 0.5_rprec * (dble(charges(ii)) ** 2.4)
!      do jj=ii+1,Nat
!        dRij = sqrt( (xyzBohr(1,ii)-xyzBohr(1,jj)) ** 2 &
!                    +(xyzBohr(2,ii)-xyzBohr(2,jj)) ** 2 &
!                    +(xyzBohr(3,ii)-xyzBohr(3,jj)) ** 2)
!        CMat(ii,jj) = ((xyzAngstrom(option%gradForCoord,ii)-xyzAngstrom(option%gradForCoord,jj))/dRij) * dble(charges(ii)) * dble(charges(jj)) / dRij
!        CMat(jj,ii) = CMat(ii,jj)
!      end do
!    end do
!  else ! Projected CM fingerprint
!    do ii=1,Nat
!      CMat(ii,ii) = 0.0_rprec
!      do jj=ii+1,Nat
!        if (ii == option%gradForAtom) then
!          dRij = sqrt( (xyzBohr(1,ii)-xyzBohr(1,jj)) ** 2 &
!                      +(xyzBohr(2,ii)-xyzBohr(2,jj)) ** 2 &
!                      +(xyzBohr(3,ii)-xyzBohr(3,jj)) ** 2)
!          CMat(ii,jj) = ((xyzAngstrom(option%gradForCoord,ii)-xyzAngstrom(option%gradForCoord,jj))/dRij) * dble(charges(ii)) * dble(charges(jj)) / dRij
!        else
!          CMat(ii,jj) = 0.0_rprec
!        end if
!        CMat(jj,ii) = 0.0_rprec
!      end do
!    end do
!  endif
  
  ! Calculate norms and sort Coulomb matrices by norms
  if (option%molDescrType == 'sorted') call sortByNorm(CMat)
  
  ! Transform matrix form into the vector form
  call vectorizeCM(Xsize,CMat,Xvec)
  
  ! Free up space
  deallocate(CMat)
  deallocate(xyzBohr)

end subroutine calcCM

subroutine sortByNorm(CMat)
  use dataset,       only : NAtomsMax
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  real(kind=rprec), intent(inout) :: CMat(1:NAtomsMax,1:NAtomsMax) ! Coulomb matrix
  ! Local variables
  real(kind=rprec)                :: normsquare, temp
  integer                         :: ii, jj, itemp, Error
  ! Arrays
  real(kind=rprec), allocatable   :: norms(:)  ! Array with norms of rows of Coulomb matrices sorted from max to min
  real(kind=rprec), allocatable   :: CMtemp(:)

  ! Allocate array
  allocate(norms(1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate norms array')
  allocate(CMtemp(1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate CMtemp array')

  ! Initialize
  norms = 0.0_rprec
  
  ! Calculate norms
  do ii=1,NAtomsMax
    normsquare = 0.0_rprec
    do jj=1,NAtomsMax
      normsquare = normsquare + CMat(ii,jj) ** 2
    end do
    norms(ii) = sqrt(normsquare)
  end do
  
  ! Sort norms
  do ii=1,NAtomsMax-1
    itemp = ii
    do jj=ii+1,NAtomsMax
      if (norms(jj) > norms(itemp)) then
        itemp = jj
      end if
    end do
    if (ii /= itemp) then
      temp         = norms(itemp)
      norms(itemp) = norms(ii)
      norms(ii)    = temp
      ! First swap rows itemp and ii
      CMtemp(1:NAtomsMax)     = CMat(itemp,1:NAtomsMax)
      CMat(itemp,1:NAtomsMax) = CMat(ii,1:NAtomsMax)
      CMat(ii,1:NAtomsMax)    = CMtemp(1:NAtomsMax)
      ! Second swap columns itemp and ii
      CMtemp(1:NAtomsMax)     = CMat(1:NAtomsMax,itemp)
      CMat(1:NAtomsMax,itemp) = CMat(1:NAtomsMax,ii)
      CMat(1:NAtomsMax,ii)    = CMtemp(1:NAtomsMax)
    end if
  enddo

  ! Free up memory
  if(allocated(norms)) deallocate(norms)
  if(allocated(CMtemp)) deallocate(CMtemp)

end subroutine sortByNorm

subroutine vectorizeCM(Xsize,CMat,Xvec)
  use dataset,       only : NAtomsMax
  implicit none
  ! Arguments
  integer,          intent(in)    :: Xsize                         ! Size of the X array
  real(kind=rprec), intent(in)    :: CMat(1:NAtomsMax,1:NAtomsMax) ! Coulomb matrix
  real(kind=rprec), intent(inout) :: Xvec(1:Xsize)                 ! vectorized Coulomb matrix
  ! Variables
  integer          :: itemp
  integer          :: ii, jj

  ! Initialize
  Xvec  = 0.0_rprec
  
  ! Vectorize Coulomb matrices into X
  itemp = 0
  do ii=1, NAtomsMax
    do jj=1, NAtomsMax
      itemp = itemp + 1
      Xvec(itemp) = CMat(ii,jj)
    end do
  end do

end subroutine vectorizeCM

end module D_CoulombMatrix
