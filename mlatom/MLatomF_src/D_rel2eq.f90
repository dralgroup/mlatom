
  !---------------------------------------------------------------------------! 
  ! D_rel2eq: RE (relative to equilibrium) descriptor construction            ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module D_rel2eq
  use dataset,       only : ReqMod
  use mathUtils,     only : Rij
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  implicit none

contains

subroutine calcrel2eq(Xsize, Nat, charges, xyzAngstrom, Xvec, xyzSorted)
  use dataset,   only : NAtomsMax
  implicit none
  ! Arguments
  integer,          intent(in)              :: Xsize                        ! Size of the X array
  integer,          intent(in)              :: Nat                          ! Number of atoms
  integer,          intent(in)              :: charges(1:NAtomsMax)         ! Nuclear charges
  real(kind=rprec), intent(in)              :: xyzAngstrom(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  real(kind=rprec), intent(inout)           :: Xvec(1:Xsize)                ! Input vector
  real(kind=rprec), intent(inout), optional :: xyzSorted(1:3,1:NAtomsMax)   ! Sorted XYZ coordinates
  ! Variables
  integer          :: ii, jj, itemp, Error
  ! Arrays
  real(kind=rprec), allocatable :: xyz(:,:)           ! Nuclear coordinates, if necessary sorted by nuclear repulsion

  ! Allocate array
  allocate(xyz(1:3,1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate xyz array')
  
  ! Initialize
  Xvec = 0.0_rprec
  xyz = xyzAngstrom
  
  ! Sort nuclear coordinates by nuclear repulsions if requested
  if (option%molDescrType == 'sorted') call sortByNuclearRepulsion(xyzAngstrom, charges, xyz)

!  if (option%gradForAtom == 0) then
    itemp = 0
    do ii=1,Nat-1
      do jj=ii+1,Nat
        itemp = itemp + 1
        Xvec(itemp) = ReqMod(itemp) / Rij(xyz(:,ii),xyz(:,jj))
      end do
    end do
!  elseif (.false.) then ! Projected RE
!    itemp = 0
!    do ii=1,Nat-1
!      do jj=ii+1,Nat
!        itemp = itemp + 1
!        dRij = sqrt( (xyzAngstrom(1,ii)-xyzAngstrom(1,jj)) ** 2 &
!                    +(xyzAngstrom(2,ii)-xyzAngstrom(2,jj)) ** 2 &
!                    +(xyzAngstrom(3,ii)-xyzAngstrom(3,jj)) ** 2)
!        Xvec(itemp) = ((xyzAngstrom(option%gradForCoord,ii)-xyzAngstrom(option%gradForCoord,jj))/dRij) * Req(itemp) / dRij
!      end do
!    end do
!  else ! Projected RE fingerprint
!    itemp = 0
!    do ii=1,Nat-1
!      do jj=ii+1,Nat
!        itemp = itemp + 1
!        if (ii == option%gradForAtom) then
!          dRij = sqrt( (xyzAngstrom(1,ii)-xyzAngstrom(1,jj)) ** 2 &
!                      +(xyzAngstrom(2,ii)-xyzAngstrom(2,jj)) ** 2 &
!                      +(xyzAngstrom(3,ii)-xyzAngstrom(3,jj)) ** 2)
!          Xvec(itemp) = ((xyzAngstrom(option%gradForCoord,ii)-xyzAngstrom(option%gradForCoord,jj))/dRij) * Req(itemp) / dRij
!        else
!          Xvec(itemp) = 0.0_rprec
!        end if
!      end do
!    end do
!  endif

  if (present(xyzSorted)) xyzSorted = xyz

  if(allocated(xyz)) deallocate(xyz)
  
end subroutine calcrel2eq

subroutine getXYZeq(Zeq, XYZeq)
!==============================================================================
! Read the XYZ coordinates of the equilibrium structure from 'xyz.dat' file
!==============================================================================
  use dataset, only : NAtomsMax, parseXYZfile, readXYZ, readY, XYZ_A, Z, Y, NitemsTot
  implicit none
  ! Arguments
  integer,          intent(out) :: Zeq(1:NAtomsMax)       ! Nuclear charges, if necessary sorted by nuclear repulsion
  real(kind=rprec), intent(out) :: XYZeq(1:3,1:NAtomsMax) ! Nuclear coordinates, if necessary sorted by nuclear repulsion
  ! Local variables
  character(len=256) :: stmp     ! Temporary string
  integer :: itemp, ii, minii, Error, Nmols, NatMax
  real(kind=rprec) :: Ymin
  logical :: convert
  ! Arrays
  integer,          allocatable :: NatomsEq(:)  ! Number of atoms
  integer,          allocatable :: ZeqTemp(:,:) ! Nuclear charges
  real(kind=rprec), allocatable :: xyz(:,:,:)   ! Nuclear coordinates
  integer                       :: eqxyzunit
  
  ! Initialize
  convert = .false.
  
  ! Initialize
  XYZeq = 0.0_rprec
  Zeq = 0.0_rprec
  eqxyzunit=25
  
  if (option%eqXYZfileIn=='') option%eqXYZfileIn = 'eq.xyz'
  open(eqxyzunit,file=trim(option%eqXYZfileIn),action='read',iostat=Error)
  if (Error == 0) then 
    call parseXYZfile(trim(option%eqXYZfileIn), Nmols, NatMax, convert)
      
    ! Allocate arrays  
    allocate(NatomsEq(1:Nmols),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for Natoms array')
    allocate(ZeqTemp(1:NatMax,1:Nmols),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate ZeqTemp array')
    allocate(xyz(1:3,1:NatMax,1:Nmols),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate xyz array')

    call readXYZ(trim(option%eqXYZfileIn), Nmols, NatMax, convert, NatomsEq, ZeqTemp, xyz) 
    close(eqxyzunit)

    XYZeq(1:3,1:NatMax) = xyz(:,:,1)
    Zeq(1:NatMax) = ZeqTemp(:,1)

  else if (option%Yfile /= '') then 
      write(6,*) ''
      write(6,*) 'No eq.xyz is provided...'
      write(6,*) 'Geometry with the minimal reference value is chosen as eqXYZ'
      write(6,*) ''
      call readY(option%Yfile)
      do ii=1, NitemsTot
          if (ii==1) then 
            minii = ii 
            Ymin = Y(ii)
          else 
            if (Ymin > Y(ii)) then 
              Ymin = Y(ii)
              minii = ii 
            end if 
          end if 
        end do 
        XYZeq = XYZ_A(:,:,minii)
        Zeq = Z(:,minii)
  else 
        write(6,*) ''
        write(6,*) 'No file with equilibrium geometry or Yfile is provided...'
        write(6,*) 'Use the first geometry in the dataset as eqXYZ'
        write(6,*) ''
        XYZeq = XYZ_A(:,:,1)
        Zeq = Z(:,1)
  end if
  
  if (option%molDescrType == 'sorted') call sortByNuclearRepulsion(xyz(:,:,1), Zeq, XYZeq)

  write(6,*) ''
  write(6,*) '   Charge        X            Y            Z'
  do ii=1,NAtomsMax
    write(6,'(5X,I3,3X,F10.6,3X,F10.6,3X,F10.6)') Zeq(ii),XYZeq(1:3,ii)
  end do 
  write(6,*) ''

  ! Free up space
  if(allocated(Natomseq)) deallocate(Natomseq)
  if(allocated(ZeqTemp)) deallocate(ZeqTemp)
  if(allocated(xyz)) deallocate(xyz)
  
end subroutine getXYZeq

subroutine getReq(Nat,REsize,xyz,ReqLoc)
!==============================================================================
! Calculate the internuclear distances for all atom pairs in equilibrium structure
!==============================================================================
  use dataset, only : NAtomsMax, parseXYZfile, readXYZ
  implicit none
  ! Arguments
  integer,          intent(in)  :: Nat              ! Number of atoms
  integer,          intent(in)  :: REsize           ! Size of RE vector
  real(kind=rprec), intent(in)  :: xyz(1:3,1:Nat)   ! Nuclear coordinates of equilibrium structure
  real(kind=rprec), intent(out) :: ReqLoc(1:REsize) ! RE vector
  ! Local variables
  integer :: itemp, ii, jj

  write(6,*) ' Equilibrium distances: '
  ReqLoc = 0.0_rprec  
  itemp = 0
  do ii=1,Nat-1
    do jj=ii+1,Nat
      itemp = itemp + 1
      ReqLoc(itemp) = Rij(xyz(:,ii),xyz(:,jj))
      write(6,'(i0,5x,i0,"-",i0,f20.12)') itemp, ii, jj, ReqLoc(itemp)
    end do
  end do

end subroutine getReq

subroutine sortByNuclearRepulsion(xyz, charges, sortedXYZ)
  use dataset, only : NAtomsMax, permInvNuclei
  implicit none
  ! Arguments
  real(kind=rprec), intent(in)  :: xyz(1:3,1:NAtomsMax)       ! Nuclear coordinates
  integer,          intent(in)  :: charges(1:NAtomsMax)       ! Nuclear charges
  real(kind=rprec), intent(out) :: sortedXYZ(1:3,1:NAtomsMax) ! Nuclear coordinates sorted by nuclear repulsion
  ! Local variables
  integer                       :: Natoms2sort                ! Number of atoms to sort
  integer                       :: Error, ii, igroup, inucl
  ! Arrays
  integer, allocatable          :: sortedZ(:)                 ! Nuclear charges sorted by nuclear repulsion
  integer, allocatable          :: sortedIatoms(:)            ! Indices of atoms to sort

  ! Allocate array
  allocate(sortedZ(1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate sortedZ array')
  
  ! Initialize
  sortedZ   = charges
  sortedXYZ = xyz
  
  if (trim(option%permInvNuclei) /= '') then
    do igroup = 1, size(permInvNuclei)
      ! Define which atoms to sort
      Natoms2sort = size(permInvNuclei(igroup)%oneDintArr)
        
      ! Allocate array
      allocate(sortedIatoms(1:Natoms2sort),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate sortedIatoms array')

      ! Initialize
      do ii = 1, Natoms2sort
        sortedIatoms(ii) = permInvNuclei(igroup)%oneDintArr(ii)
      end do
        
      ! Sort coordinates by nulear repulsion
      call sortByNuclearRepulsionIndices(sortedZ, sortedXYZ, Natoms2sort, sortedIatoms)

      ! Free up memory
      if(allocated(sortedIatoms)) deallocate(sortedIatoms)
    end do
  else
    ! Define which atoms to sort
    Natoms2sort = NAtomsMax
    
    ! Allocate array
    allocate(sortedIatoms(1:Natoms2sort),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate sortedIatoms array')

    ! Initialize
    do ii = 1, NAtomsMax
      sortedIatoms(ii) = ii
    end do
    
    ! Sort coordinates by nulear repulsion
    call sortByNuclearRepulsionIndices(sortedZ, sortedXYZ, Natoms2sort, sortedIatoms)

    ! Free up memory
    if(allocated(sortedIatoms)) deallocate(sortedIatoms)
  end if
  
  ! Free up memory
  if(allocated(sortedZ)) deallocate(sortedZ)
  
end subroutine sortByNuclearRepulsion

subroutine sortByNuclearRepulsionIndices(sortedZ, sortedXYZ, Natoms2sort, sortedIatoms)
  use dataset, only : NAtomsMax, permInvNuclei
  implicit none
  ! Arguments
  integer,          intent(inout) :: sortedZ(1:NAtomsMax)        ! Nuclear charges
  real(kind=rprec), intent(inout) :: sortedXYZ(1:3,1:NAtomsMax)  ! Nuclear coordinates sorted by nuclear repulsion
  integer,          intent(in)    :: Natoms2sort                 ! Number of atoms to sort
  integer,          intent(inout) :: sortedIatoms(1:Natoms2sort) ! Indices of atoms to sort
  ! Local variables
  real(kind=rprec)                :: temp
  integer                         :: ii, jj, ir, jr, itemp, irtemp, Ztemp, Error, icount
  ! Arrays
  real(kind=rprec), allocatable   :: repulsions(:)               ! Array with repulsions
  real(kind=rprec), allocatable   :: inputZ(:), inputXYZ(:,:)    ! Temporary arrays with nuclear charges and coordinates provided to this subroutine

  ! Allocate array
  allocate(repulsions(1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate repulsions array')
  allocate(inputZ(1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate inputZ array')
  allocate(inputXYZ(1:3,1:NAtomsMax),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate inputXYZ array')

  ! Initialize
  repulsions = 0.0_rprec
  inputZ     = sortedZ
  inputXYZ   = sortedXYZ
  
  ! Calculate nuclear repulsions
  do ii=1,Natoms2sort
    ir = sortedIatoms(ii)
    do jr=1,NAtomsMax
      if (ir==jr) cycle
      repulsions(ii) = repulsions(ii) + dble(sortedZ(ir)) * dble(sortedZ(jr)) &
                                      / Rij(sortedXYZ(:,ir),sortedXYZ(:,jr))
    end do
  end do
  
  ! Sort coordinates by nulear repulsion
  do ii=1,Natoms2sort-1
    itemp = ii
    do jj=ii+1,Natoms2sort
      if (repulsions(jj) > repulsions(itemp)) then
        itemp = jj
      end if
    end do
    if (ii /= itemp) then
      temp                = repulsions(itemp)
      repulsions(itemp)   = repulsions(ii)
      repulsions(ii)      = temp
	  Ztemp               = sortedIatoms(itemp)
	  sortedIatoms(itemp) = sortedIatoms(ii)
	  sortedIatoms(ii)    = Ztemp
    end if
  enddo
  
  icount = 0
  do ir=minval(sortedIatoms),maxval(sortedIatoms)
    jr = ir
    if (any(sortedIatoms == ir)) then
	  icount = icount + 1
	  jr     = sortedIatoms(icount)
	end if
	sortedZ(ir)     = inputZ(jr)
	sortedXYZ(:,ir) = inputXYZ(:,jr)
  enddo

  ! Debug print
  if (option%debug) then
    write(6,'(A19)',advance='no') 'Sorted repulsions: '
    do ii=1,Natoms2sort-1
      write(6,'(F15.10)',advance='no') repulsions(ii)
    end do
    write(6,'(F15.10)',advance='yes') repulsions(Natoms2sort)
    write(6,'(A16)',advance='no') 'Sorted indices: '
    do ii=1,Natoms2sort-1
      write(6,'(I0,2X)',advance='no') sortedIatoms(ii)
    end do
    write(6,'(I0)',advance='yes') sortedIatoms(Natoms2sort)
  end if

  ! Free up memory
  if(allocated(repulsions)) deallocate(repulsions)
  if(allocated(inputZ))     deallocate(inputZ)
  if(allocated(inputXYZ))   deallocate(inputXYZ)

end subroutine sortByNuclearRepulsionIndices

end module D_rel2eq
