
  !---------------------------------------------------------------------------! 
  ! molDescr: generating molecular descriptors from XYZ coordinates           ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module molDescr
  use dataset,         only : XYZeq, Zeq, Req, ReqMod, XvecSize, NAtomsMax
  use dataset,         only : Nperm, permInvNuclei, permInvGroups
  use dataset,         only : permutedIatoms, NpermGroups, NpermNucl 
  use D_CoulombMatrix, only : calcCM
  use D_rel2eq,        only : getXYZeq, getReq, calcrel2eq
  use D_ID,            only : calcID
  use mathUtils,       only : factorial_rprec, permutations, permGroups, Rij
  use optionsModule,   only : option
  use precision,       only : rprec
  use stopper,         only : stopMLatomF
  use types,           only : arrayOfArrays, nuclGroups
  implicit none
  logical                       :: readEqXYZFlag = .True.

contains

subroutine getMolDescr()
  use dataset, only : NitemsTot, Natoms
  use dataset, only : XYZ_A, XYZ_A_sorted, X, XvecSize, Z
  use timing,  only : timeTaken
  implicit none
  ! Variables
  integer                       :: Nspecies, Error
  integer                       :: Xsize                  ! Size of the X array
  integer                       :: iperm, ii
  integer                       :: Natoms2perm
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date
  integer,          allocatable :: permutedZ(:)           ! Permuted nuclear charges
  real(kind=rprec), allocatable :: permutedXYZ(:,:)       ! Permuted nuclear coordinates in Angstrom

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  ! Calculate the size of the X vector
  if     (trim(option%molDescriptor) == 'CM') then
    XvecSize = NAtomsMax**2
  elseif (trim(option%molDescriptor) == 'RE' .or. option%molDescriptor == 'ID') then
    XvecSize = NAtomsMax*(NAtomsMax - 1) / 2
  end if
  if (trim(option%molDescrType) == 'permuted') XvecSize = XvecSize * Nperm

  ! Allocate arrays
  allocate(X(1:XvecSize,1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate space for X array')
  if (option%writeXYZsorted) then
    allocate(XYZ_A_sorted(1:3,1:NAtomsMax,1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ_A_sorted array')
  end if

  ! Initialize
  X = 0.0_rprec
  if (option%writeXYZsorted) XYZ_A_sorted = 0.0_rprec
  
  ! Get indices of permuted atoms
  if (trim(option%molDescrType) == 'permuted') call getPermIatoms()
  
  ! Get equilibrium geometry if necessary
  if (trim(option%molDescriptor) == 'RE' .or. &
      (trim(option%molDescrType) == 'permuted' .and. option%selectPerm)) then
    if (readEqXYZFlag) then 
      allocate(XYZeq(1:3,1:NAtomsMax),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ coordinates')
      allocate(Zeq(1:NAtomsMax),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for nuclear charges')
      call getXYZeq(Zeq,XYZeq)
    end if

  end if

  ! Read in coordinates of the equilibrium structure and get internuclear distances in it
  if (trim(option%molDescriptor) == 'RE') then
    allocate(Req(XvecSize),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for Req')
    if (trim(option%molDescrType) == 'permuted') then
      Xsize = XvecSize / Nperm
      allocate(ReqMod(Xsize),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for ReqMod')
      ReqMod = 0.0_rprec
      allocate(permutedZ(1:NAtomsMax),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for permutedZ array')
      allocate(permutedXYZ(1:3,1:NAtomsMax),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for permutedXYZ array')
      Natoms2perm = size(permutedIatoms(1)%oneDintArr)
      do iperm = 1, Nperm
        call permuteAtoms(Zeq, XYZeq, Natoms2perm, permutedIatoms(iperm)%oneDintArr, permutedZ, permutedXYZ)
        !write(*,*) NAtomsMax
        !write(*,*) permutedIatoms(iperm)%oneDintArr
        !do ii = 1, NAtomsMax
        !  write(*,'(I2,3F19.12)') permutedZ(ii), permutedXYZ(:,ii)
        !end do
        call getReq(NAtomsMax, Xsize, permutedXYZ, Req(1+Xsize*(iperm-1):Xsize+Xsize*(iperm-1)))
      end do
      deallocate(permutedZ)
      deallocate(permutedXYZ)
    else
      call getReq(NAtomsMax, XvecSize, XYZeq, Req)
      allocate(ReqMod(XvecSize),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for ReqMod')
      ReqMod = Req
    end if
  end if

  ! Calculate the input vectors
  if (option%writeXYZsorted) then
    do Nspecies=1,NitemsTot
      call calcX(Natoms(Nspecies), Z(:,Nspecies), XYZ_A(:,:,Nspecies), X(:,Nspecies), xyzSorted = XYZ_A_sorted(:,:,Nspecies))
    end do
  else
    do Nspecies=1,NitemsTot
      call calcX(Natoms(Nspecies), Z(:,Nspecies), XYZ_A(:,:,Nspecies), X(:,Nspecies))
    end do
  end if
  
  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Descriptor generation time:')

end subroutine getMolDescr

subroutine calcX(Nat, charges, xyzAngstrom, molDescr, xyzSorted)
  implicit none
  ! Arguments
  integer,          intent(in)              :: Nat                          ! Number of atoms
  integer,          intent(in)              :: charges(1:NAtomsMax)         ! Nuclear charges
  real(kind=rprec), intent(in)              :: xyzAngstrom(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  real(kind=rprec), intent(out)             :: molDescr(1:XvecSize)         ! Molecular descriptor
  real(kind=rprec), intent(inout), optional :: xyzSorted(1:3,1:NAtomsMax)   ! Sorted XYZ coordinates
  ! Variables
  integer                                   :: Xsize                        ! Size of the X array
  integer                                   :: iperm, Error
  integer                                   :: Natoms2perm
  ! Arrays
  integer,          allocatable             :: permutedZ(:)                 ! Permuted nuclear charges
  real(kind=rprec), allocatable             :: permutedXYZ(:,:)             ! Permuted nuclear coordinates in Angstrom

  if (present(xyzSorted)) then
    call calcMolDescr(XvecSize, Nat, charges, xyzAngstrom, molDescr, xyzSorted = xyzSorted)
  elseif (trim(option%molDescrType) == 'permuted') then
    Xsize = XvecSize / Nperm
    allocate(permutedZ(1:NAtomsMax),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permutedZ array')
    allocate(permutedXYZ(1:3,1:NAtomsMax),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permutedXYZ array')
    Natoms2perm = size(permutedIatoms(1)%oneDintArr)
    do iperm = 1, Nperm
      call permuteAtoms(charges, xyzAngstrom, Natoms2perm, permutedIatoms(iperm)%oneDintArr, permutedZ, permutedXYZ)
      if (trim(option%molDescriptor) == 'RE') ReqMod = Req(1+Xsize*(iperm-1):Xsize+Xsize*(iperm-1))
      call calcMolDescr(Xsize, Nat, permutedZ, permutedXYZ, molDescr(1+Xsize*(iperm-1):Xsize+Xsize*(iperm-1)))
    end do
    deallocate(permutedZ)
    deallocate(permutedXYZ)
  else
    call calcMolDescr(XvecSize, Nat, charges, xyzAngstrom, molDescr)
  end if

end subroutine calcX

subroutine calcMolDescr(Xsize, Nat, charges, xyzAngstrom, molDescr, xyzSorted)
  implicit none
  ! Arguments
  integer,          intent(in)              :: Xsize                        ! Size of the X array
  integer,          intent(in)              :: Nat                          ! Number of atoms
  integer,          intent(in)              :: charges(1:NAtomsMax)         ! Nuclear charges
  real(kind=rprec), intent(in)              :: xyzAngstrom(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  real(kind=rprec), intent(out)             :: molDescr(1:Xsize)            ! Molecular descriptor
  real(kind=rprec), intent(inout), optional :: xyzSorted(1:3,1:NAtomsMax)   ! Sorted XYZ coordinates
  ! Local variables
  integer                                   :: Error
  real(kind=rprec), allocatable             :: xyzLoc(:,:)                  ! Local XYZ coordinates


  allocate(xyzLoc(1:3,1:NAtomsMax),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for xyzLoc array')
  xyzLoc = xyzAngstrom

  if     (trim(option%molDescriptor) == 'CM') then
    call calcCM(Xsize, Nat, charges, xyzLoc, molDescr)
  elseif (trim(option%molDescriptor) == 'RE') then
    if (present(xyzSorted) .and. option%molDescrType == 'sorted') then
      call calcrel2eq(Xsize, Nat, charges, xyzAngstrom, molDescr, xyzSorted = xyzSorted)
    else
      call calcrel2eq(Xsize, Nat, charges, xyzLoc,      molDescr)
    end if
  elseif (trim(option%molDescriptor) == 'ID') then
    call calcID(Xsize, Nat, xyzLoc, molDescr)
  end if

end subroutine calcMolDescr

subroutine permuteAtoms(ZZ, xyz, Natoms2perm, permIatoms, permutedZ, permutedXYZ)
  implicit none
  ! Arguments
  integer,          intent(in)  :: ZZ(1:NAtomsMax)              ! Nuclear charges
  real(kind=rprec), intent(in)  :: xyz(1:3,1:NAtomsMax)         ! Nuclear coordinates in Angstrom
  integer,          intent(in)  :: Natoms2perm                  ! Number of atoms to permute
  integer,          intent(in)  :: permIatoms(1:Natoms2perm)    ! Indices of atoms to sort
  integer,          intent(out) :: permutedZ(1:NAtomsMax)       ! Nuclear charges
  real(kind=rprec), intent(out) :: permutedXYZ(1:3,1:NAtomsMax) ! Nuclear coordinates in Angstrom
  ! Variables
  integer                       :: icount, iperm, ii            ! Indices
  
  icount = 0
  do ii=1,NAtomsMax
    iperm = ii
    if (any(permIatoms == ii)) then
      icount = icount + 1
      iperm  = permIatoms(icount)
    end if
    permutedZ(ii)     = ZZ(iperm)
    permutedXYZ(:,ii) = xyz(:,iperm)
  enddo

end subroutine permuteAtoms

!==============================================================================
subroutine getPermIatoms()
!==============================================================================
! Parse option string into machine-readable integer arrays
!==============================================================================
  implicit none
  ! Local arguments
  integer :: ngroups, nsets, Natoms2perm, NallAtoms2perm
  integer :: ii, jj, kk, ll, igroup, iset, iperm, istart, iend, Error
  integer,             allocatable :: nElmntsArr(:), iAts(:), iAtstmp(:)
  type(arrayOfArrays), allocatable :: permIatomsGroups(:) ! Indices of permuted atoms
  type(arrayOfArrays), allocatable :: permIatomsNucl(:) ! Indices of permuted atoms
  type(nuclGroups),    allocatable :: permElmnts(:)

  if (trim(option%permInvNuclei) /= '') then
    ngroups = size(permInvNuclei)
    ! Get all possible atom permutations
    allocate(nElmntsArr(ngroups),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for nElmntsArr array')
    allocate(permElmnts(NpermNucl),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permElmnts array')
    NallAtoms2perm = 0
    do igroup = 1, ngroups
      Natoms2perm = size(permInvNuclei(igroup)%oneDintArr)
      nElmntsArr(igroup) = Natoms2perm
      NallAtoms2perm = NallAtoms2perm + Natoms2perm
    end do
    call permGroups(ngroups, nElmntsArr, NpermNucl, permElmnts)
    deallocate(nElmntsArr)
    
    ! Get arrays with permuted atoms
    allocate(permIatomsNucl(NpermNucl),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permIatomsNucl array')
    do iperm = 1, NpermNucl
      allocate(permIatomsNucl(iperm)%oneDintArr(NallAtoms2perm),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for permIatomsNucl(iperm)%oneDintArr array')
    end do
    istart = 0
    iend = 0
    do igroup = 1, ngroups
      istart = iend
      iend   = iend + size(permInvNuclei(igroup)%oneDintArr)
      do iperm = 1, NpermNucl
        do ii = 1, size(permInvNuclei(igroup)%oneDintArr)
          permIatomsNucl(iperm)%oneDintArr(istart+ii) = &
          permInvNuclei(igroup)%oneDintArr(permElmnts(iperm)%groups(igroup)%oneDintArr(ii))
        end do
      end do
    end do
    do iperm = 1, NpermNucl
      do igroup = 1, ngroups
        deallocate(permElmnts(iperm)%groups(igroup)%oneDintArr)
      end do
      deallocate(permElmnts(iperm)%groups)
    end do
    deallocate(permElmnts)
  end if

  if (trim(option%permInvGroups) /= '') then
    nsets = size(permInvGroups)
    ! Permute indices of set of groups
    allocate(nElmntsArr(nsets),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for nElmntsArr array')
    allocate(permElmnts(NpermGroups),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permElmnts array')
    NallAtoms2perm = 0
    do iset = 1, nsets
      ngroups = size(permInvGroups(iset)%groups)
      nElmntsArr(iset) = ngroups
      do igroup = 1, ngroups
        NallAtoms2perm = NallAtoms2perm + size(permInvGroups(iset)%groups(igroup)%oneDintArr)
      end do
    end do
    call permGroups(nsets, nElmntsArr, NpermGroups, permElmnts)
    deallocate(nElmntsArr)
    
    ! Get arrays with permuted atoms
    allocate(permIatomsGroups(NpermGroups),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permIatomsGroups array')
    do iperm = 1, NpermGroups
      allocate(permIatomsGroups(iperm)%oneDintArr(NallAtoms2perm),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for permIatomsGroups(iperm)%oneDintArr array')
    end do
    istart = 0
    iend = 0
    do iset = 1, nsets
      istart = iend
      kk = size(permInvGroups(iset)%groups)
      ll = size(permInvGroups(iset)%groups(1)%oneDintArr)
      iend   = iend + kk*ll
      do iperm = 1, NpermGroups
        do ii = 1, kk
          permIatomsGroups(iperm)%oneDintArr(istart+1+(ii-1)*ll:istart+ii*ll) = &
          permInvGroups(iset)%groups(permElmnts(iperm)%groups(iset)%oneDintArr(ii))%oneDintArr
        end do
      end do
    end do
    do iperm = 1, NpermGroups
      do iset = 1, nsets
        deallocate(permElmnts(iperm)%groups(iset)%oneDintArr)
      end do
      deallocate(permElmnts(iperm)%groups)
    end do
    deallocate(permElmnts)
  end if
  
  if (.not. option%usePermInd) then 
    allocate(permutedIatoms(Nperm),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms array')
  else 
    call readPermIndices(option%permIndIn)
  end if 

  if (trim(option%permInvGroups) /= '' .and. trim(option%permInvNuclei) /= '') then
    allocate(iAts(NAtomsMax),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for iAts array')
    allocate(iAtstmp(NAtomsMax),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for iAtstmp array')
    do ii = 1, NAtomsMax
      iAts(ii)    = ii
      iAtstmp(ii) = ii
    end do
    kk = size(permIatomsGroups(1)%oneDintArr)
    ll = size(permIatomsNucl(1)%oneDintArr)
    iperm = 0
    do ii = 1, NpermGroups
      call permuteIndices(NAtomsMax, iAts, kk, permIatomsGroups(ii)%oneDintArr, iAtstmp)
      do jj = 1, NpermNucl
        iperm = iperm + 1
        allocate(permutedIatoms(iperm)%oneDintArr(NAtomsMax),stat=Error)
        if(Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms(iperm)%oneDintArr array')
        permutedIatoms(iperm)%oneDintArr = 0
        call permuteIndices(NAtomsMax, iAtstmp, ll, permIatomsNucl(jj)%oneDintArr, permutedIatoms(iperm)%oneDintArr)
      end do
    end do
  elseif(trim(option%permInvGroups) /= '') then
    permutedIatoms = permIatomsGroups
  elseif(trim(option%permInvNuclei) /= '') then
    permutedIatoms = permIatomsNucl
  end if
  
  if (allocated(permIatomsNucl)) then
    do ii = 1, size(permIatomsNucl)
      if(allocated(permIatomsNucl(ii)%oneDintArr)) deallocate(permIatomsNucl(ii)%oneDintArr)
    end do
    deallocate(permIatomsNucl)
  end if
  if (allocated(permIatomsGroups)) then
    do ii = 1, size(permIatomsGroups)
      if(allocated(permIatomsGroups(ii)%oneDintArr)) deallocate(permIatomsGroups(ii)%oneDintArr)
    end do
    deallocate(permIatomsGroups)
  end if
 
end subroutine getPermIatoms
!==============================================================================

subroutine permuteIndices(Ninds, inds, Nperms, perms, permInds)
  implicit none
  ! Arguments
  integer,          intent(in)  :: Ninds, Nperms           
  integer,          intent(in)  :: inds(1:Ninds)   
  integer,          intent(in)  :: perms(1:Nperms)   
  integer,          intent(out) :: permInds(1:Ninds)
  ! Variables
  integer                       :: icount, iperm, ii             ! Indices
  
  icount = 0
  do ii=1,Ninds
    iperm = ii
    if (any(perms == ii)) then
      icount = icount + 1
      iperm  = perms(icount)
    end if
    permInds(ii) = inds(iperm)
  enddo

end subroutine permuteIndices

!==============================================================================
function dRMSD(xyz1, xyz2)
!==============================================================================
! Sort XYZ coordinates to minimize distance RMSD between this structure
! and equilibrium structure
!==============================================================================
  implicit none
  real(kind=rprec) :: dRMSD
  ! Arguments
  real(kind=rprec), intent(in) :: xyz1(1:3,1:NAtomsMax)
  real(kind=rprec), intent(in) :: xyz2(1:3,1:NAtomsMax)
  ! Local variables
  integer :: nn, ii, jj
  
  dRMSD = 0.0_rprec
  nn = 0
  do ii = 1, NAtomsMax
    do jj = ii+1, NAtomsMax
      nn = nn + 1
      dRMSD = dRMSD+(Rij(xyz1(:,ii),xyz1(:,jj))-Rij(xyz2(:,ii),xyz2(:,jj)))**2
    end do
  end do
  dRMSD = sqrt(dRMSD * 2.0_rprec / (nn * (nn-1)))

end function dRMSD
!==============================================================================

!==============================================================================
subroutine readPermIndices(filename)
  use dataset, only : modPermutedIatoms, permlen
  implicit none 
  ! Argument 
  character(len=256), intent(in) :: filename 
  ! Local variables
  integer :: fileunit
  integer :: Error 
  character(len=256) :: tempstr
  integer :: ii,nn
  integer :: Nline

  allocate(permutedIatoms(Nperm),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms array')

  if (option%readModPermInd) then 
    permutedIatoms = modPermutedIatoms
  else 
    if (option%permlen /= '') then 
      read(option%permlen,*) permlen
    else 
      call stopMLatomF('Please provide permlen')
    end if 
    fileunit = 60  
    open(fileunit,file=trim(filename),action='read',iostat=Error)
    if (Error/=0) call stopMLatomF('Failed to open file ' // trim(filename))
    do ii=1,Nperm
      allocate(permutedIatoms(ii)%oneDintArr(permlen),stat=Error)
      if (Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms(ii)%oneDintArr array')
      read(fileunit,'(a)',iostat=Error) tempstr
      read(tempstr,*) permutedIatoms(ii)%oneDintArr
    end do
    close(fileunit)
  end if 

end subroutine readPermIndices
!==============================================================================

end module molDescr
