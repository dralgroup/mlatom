
  !---------------------------------------------------------------------------! 
  ! mathUtils: collection of routines for basic mathematical operations       ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module mathUtils
  use precision, only : rprec
  implicit none

contains

!==============================================================================
subroutine invertMatrix(AA, nn)
!==============================================================================
! Subroutine return inverse of a matrix AA of nn*nn size
! and destroy passed matrix AA
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  ! Local variable
  integer :: Error

  Error = 0
  if (option%matDecomp == 'Cholesky') then
    call invertCholesky(AA, nn, Error)
    if (Error > 0) call invertBK(AA, nn)
  elseif (option%matDecomp == 'LU') then
    call invertLU(AA, nn)
  elseif (option%matDecomp == 'Bunch-Kaufman') then
    call invertBK(AA, nn)
  end if

end subroutine invertMatrix
!==============================================================================

!==============================================================================
subroutine invertLU(AA, nn)
!==============================================================================
! Uses LU factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  ! Local variables
  integer :: lwork, Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: pivotIndices(:)
  real(kind=rprec), allocatable   :: work(:)
  ! External MKL or Lapack subroutine
  integer, external :: ilaenv

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0

  ! Allocate arrays
  allocate(pivotIndices(1:nn),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate pivot indices')
  
  ! LU decomposition
  call dgetrf(nn, nn, AA, nn, pivotIndices, Error)

  if (error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'LU decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (error > 0) then
    write(6,'(a)') 'solveSysLinEqsLU: LU factorization could not be completed'
  elseif (error < 0) then
    call stopMLatomF('solveSysLinEqsLU: Argument had an illegal value')
  end if

  ! Determine optimal lwork size
  lwork = nn * ilaenv(1, 'DGETRI', '', nn, nn, -1, -1)

  ! Allocate arrays
  allocate(work(1:lwork),stat=Error)
  if(Error/=0)call stopMLatomF('invertLU: Unable to allocate pivot indices')

  ! Invert a matrix
  call dgetri(nn, AA, nn, pivotIndices, work, lwork, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Inverting time:')
  elseif (Error > 0) then
    call stopMLatomF('invertLU: The matrix is singular: impossible to inverse')
  elseif (Error < 0) then
    call stopMLatomF('invertLU: Argument had an illegal value')
  end if

  ! Free up memory
  if(allocated(pivotIndices)) deallocate(pivotIndices)
  if(allocated(work))         deallocate(work)

end subroutine invertLU
!==============================================================================

!==============================================================================
subroutine invertCholesky(AA, nn, Error)
!==============================================================================
! Subroutine return inverse of a matrix aInv and destroy passed matrix aInv
! It uses Cholesky factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  integer, intent(out)            :: Error
  ! Local variables
  integer :: ii, jj
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0
  
  ! Cholesky decomposition
  call dpotrf('U', nn, AA, nn, Error)

  if (Error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'Cholesky decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (Error > 0) then
    write(6,'(A)') 'invertCholesky: Cholesky decomposition failed. Attempting Bunch-Kaufman decomposition'
    return
  elseif (Error < 0) then
    call stopMLatomF('invertCholesky: Argument had an illegal value')
  end if

  ! Invert a matrix
  call dpotri('U', nn, AA, nn, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'invertCholesky: Inverting time:')
  elseif (Error > 0) then
    call stopMLatomF('invertCholesky: The matrix is singular: impossible to inverse')
  elseif (Error < 0) then
    call stopMLatomF('invertCholesky: Argument had an illegal value')
  end if
  
  do ii = 1, nn
    do jj = 1, ii-1
      AA(ii,jj) = AA(jj,ii)
    end do
  end do

end subroutine invertCholesky
!==============================================================================

!==============================================================================
subroutine invertBK(AA, nn)
!==============================================================================
! Uses Bunch-Kaufman factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  ! Local variables
  integer :: lwork, Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: pivotIndices(:)
  real(kind=rprec), allocatable   :: work(:)
  ! External MKL or Lapack subroutine
  integer, external :: ilaenv

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0

  ! Determine optimal lwork size
  lwork = nn * ilaenv(1, 'dsytrf', 'U', nn, nn, -1, -1)

  ! Allocate arrays
  allocate(pivotIndices(1:nn),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate pivot indices')
  allocate(work(1:lwork),stat=Error)
  if(Error/=0)call stopMLatomF('invertBK: Unable to allocate work array')
  
  !  Bunch-Kaufman decomposition
  call dsytrf('U', nn, AA, nn, pivotIndices, work, lwork, Error)

  if (error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'Bunch-Kaufman decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (error > 0) then
    write(6,'(a)') 'invertBK: Bunch-Kaufman factorization could not be completed'
  elseif (error < 0) then
    call stopMLatomF('invertBK: Argument had an illegal value')
  end if

  ! Invert a matrix
  call dsytri('U', nn, AA, nn, pivotIndices, work, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Inverting time:')
  elseif (Error > 0) then
    call stopMLatomF('invertBK: The matrix is singular: impossible to inverse')
  elseif (Error < 0) then
    call stopMLatomF('invertBK: Argument had an illegal value')
  end if

  ! Free up memory
  if(allocated(pivotIndices)) deallocate(pivotIndices)
  if(allocated(work))         deallocate(work)

end subroutine invertBK
!==============================================================================

!==============================================================================
subroutine solveSysLinEqs(nn, nrhs, AA, bb, xx)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  real(kind=rprec), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=rprec), intent(out)   :: xx(1:nn, 1:nrhs)
  ! Local variable
  integer :: Error

  Error = 0
  if (option%matDecomp == 'Cholesky') then
    call solveSysLinEqsCholesky(nn, nrhs, AA, bb, xx, Error)
    if (Error > 0) call solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
  elseif (option%matDecomp == 'LU') then
    call solveSysLinEqsLU(nn, nrhs, AA, bb, xx)
  elseif (option%matDecomp == 'Bunch-Kaufman') then
    call solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
  end if

end subroutine solveSysLinEqs
!==============================================================================

!==============================================================================
subroutine solveSysLinEqsCholesky(nn, nrhs, AA, bb, xx, Error)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
! It uses Cholesky factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  real(kind=rprec), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=rprec), intent(out)   :: xx(1:nn, 1:nrhs)
  integer, intent(out)            :: Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: iwork(:)
  real(kind=rprec), allocatable   :: Aoriginal(:,:), ferr(:), berr(:), work(:)

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0
  
  ! Allocate arrays
  if (option%refine) then
    allocate(Aoriginal(1:nn,1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate Aoriginal array')
    Aoriginal = AA
  end if
  
  ! Cholesky decomposition
  call dpotrf('U', nn, AA, nn, Error)

  if (Error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'Cholesky decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (Error > 0) then
    write(6,'(A)') ' Cholesky decomposition failed. Attempting Bunch-Kaufman decomposition'
    if(allocated(Aoriginal))    deallocate(Aoriginal)
    return
  elseif (Error < 0) then
    call stopMLatomF('solveSysLinEqsCholesky: Argument had an illegal value')
  end if

  ! Solve a system of linear equations
  xx = bb
  call dpotrs('U', nn, nrhs, AA, nn, xx, nn, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Time for solving a system of linear equations:')
  elseif (Error < 0) then
    call stopMLatomF('solveSysLinEqsCholesky: Argument had an illegal value')
  end if
  
  if (option%refine) then
    allocate(ferr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate ferr array')
    allocate(berr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate berr array')
    allocate(work(1:3*nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate work array')
    allocate(iwork(1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate iwork array')
    call dporfs('U', nn, nrhs, Aoriginal, nn, AA, nn, bb, nn, xx, nn, ferr, berr, work, iwork, Error)
  end if
  
  ! Free up memory
  if(allocated(Aoriginal))    deallocate(Aoriginal)
  if(allocated(ferr))         deallocate(ferr)
  if(allocated(berr))         deallocate(berr)
  if(allocated(work))         deallocate(work)
  if(allocated(iwork))        deallocate(iwork)

end subroutine solveSysLinEqsCholesky
!==============================================================================

!==============================================================================
subroutine solveSysLinEqsLU(nn, nrhs, AA, bb, xx)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
! It uses LU factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  real(kind=rprec), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=rprec), intent(out)   :: xx(1:nn, 1:nrhs)
  ! Local variables
  integer :: Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: pivotIndices(:), iwork(:)
  real(kind=rprec), allocatable   :: Aoriginal(:,:), ferr(:), berr(:), work(:)

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0
  
  ! Allocate arrays
  allocate(pivotIndices(1:nn),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate pivot indices')
  if (option%refine) then
    allocate(Aoriginal(1:nn,1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate Aoriginal array')
    Aoriginal = AA
  end if
  
  ! LU decomposition
  call dgetrf(nn, nn, AA, nn, pivotIndices, Error)

  if (error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'LU decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (error > 0) then
    write(6,'(a)') 'solveSysLinEqsLU: LU factorization could not be completed'
  elseif (error < 0) then
    call stopMLatomF('solveSysLinEqsLU: Argument had an illegal value')
  end if

  ! Solve a system of linear equations
  xx = bb
  call dgetrs('N', nn, nrhs, AA, nn, pivotIndices, xx, nn, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Time for solving a system of linear equations:')
  elseif (Error < 0) then
    call stopMLatomF('solveSysLinEqsLU: Argument had an illegal value')
  end if
  
  if (option%refine) then
    allocate(ferr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate ferr array')
    allocate(berr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate berr array')
    allocate(work(1:3*nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate work array')
    allocate(iwork(1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate iwork array')
    call dgerfs('N', nn, nrhs, Aoriginal, nn, AA, nn, pivotIndices, bb, nn, xx, nn, ferr, berr, work, iwork, Error)
  end if
  
  ! Free up memory
  if(allocated(pivotIndices)) deallocate(pivotIndices)
  if(allocated(Aoriginal))    deallocate(Aoriginal)
  if(allocated(ferr))         deallocate(ferr)
  if(allocated(berr))         deallocate(berr)
  if(allocated(work))         deallocate(work)
  if(allocated(iwork))        deallocate(iwork)

end subroutine solveSysLinEqsLU
!==============================================================================

!==============================================================================
subroutine solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
! It uses Bunch-Kaufman factorization and MKL/Lapack subroutines
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=rprec), intent(inout) :: AA(1:nn,1:nn)
  real(kind=rprec), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=rprec), intent(out)   :: xx(1:nn, 1:nrhs)
  ! Local variables
  integer :: ii, jj, lwork, Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: pivotIndices(:), iwork(:)
  real(kind=rprec), allocatable   :: Aoriginal(:,:), ferr(:), berr(:), work(:)
  ! External MKL or Lapack subroutine
  integer, external :: ilaenv

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  Error = 0
  
  ! Determine optimal lwork size
  lwork = nn * ilaenv(1, 'dsytrf', 'U', nn, nn, -1, -1)

  ! Allocate arrays
  allocate(pivotIndices(1:nn),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate pivot indices')
  allocate(work(1:lwork),stat=Error)
  if(Error/=0)call stopMLatomF('solveSysLinEqsBK: Unable to allocate work array')
  if (option%refine) then
    allocate(Aoriginal(1:nn,1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate Aoriginal array')
    Aoriginal = AA
  end if
  
  !  Bunch-Kaufman decomposition
  call dsytrf('U', nn, AA, nn, pivotIndices, work, lwork, Error)

  if (error == 0) then
    if (option%benchmark) then
      call timeTaken(dateStart,'Bunch-Kaufman decomposition time:')
      call date_and_time(values=dateStart)
    end if
  elseif (error > 0) then
    write(6,'(a)') 'solveSysLinEqsBK: Bunch-Kaufman factorization could not be completed'
  elseif (error < 0) then
    call stopMLatomF('solveSysLinEqsBK: Argument had an illegal value')
  end if

  ! Solve a system of linear equations
  xx = bb
  call dsytrs('U', nn, nrhs, AA, nn, pivotIndices, xx, nn, Error)

  if (Error == 0) then
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Time for solving a system of linear equations:')
  elseif (Error < 0) then
    call stopMLatomF('solveSysLinEqsBK: Argument had an illegal value')
  end if

  if (option%refine) then
    allocate(ferr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate ferr array')
    allocate(berr(1:nrhs),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate berr array')
    if(allocated(work))         deallocate(work)
    allocate(work(1:3*nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate work array')
    allocate(iwork(1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate iwork array')
    call dsyrfs('U', nn, nrhs, Aoriginal, nn, AA, nn, pivotIndices, bb, nn, xx, nn, ferr, berr, work, iwork, Error)
  end if

  ! Free up memory
  if(allocated(pivotIndices)) deallocate(pivotIndices)
  if(allocated(work))         deallocate(work)
  if(allocated(Aoriginal))    deallocate(Aoriginal)
  if(allocated(ferr))         deallocate(ferr)
  if(allocated(berr))         deallocate(berr)
  if(allocated(iwork))        deallocate(iwork)

end subroutine solveSysLinEqsBK
!==============================================================================

!==============================================================================
subroutine init_random_seed()
!==============================================================================
  use stopper, only : stopMLatomF
  implicit none
  integer, allocatable :: seed(:)
  integer              :: Nseed, currtime
  integer              :: i, error
          
  ! Get size of seed
  call random_seed(size = Nseed)
  
  ! Allocate seed
  allocate(seed(Nseed),stat=error)
  if(Error/=0)call stopMLatomF('Unable to allocate seed')

  call system_clock(count=currtime)

  seed = currtime + 37 * (/ (i, i = 0, Nseed - 1 ) /)

  call random_seed(put=seed)
 
  deallocate(seed)
 
end subroutine init_random_seed
!==============================================================================

!==============================================================================
function factorial(nn)
!==============================================================================
  implicit none
  integer :: factorial
  integer, intent(in) :: nn
  ! Local variables
  integer :: ii

  factorial = 1
  do ii = 1, nn
    factorial = factorial * ii
  end do
  
end function factorial
!==============================================================================

!==============================================================================
function factorial_rprec(nn)
!==============================================================================
  implicit none
  real(kind=rprec) :: factorial_rprec
  integer, intent(in) :: nn
  ! Local variables
  integer :: ii

  factorial_rprec = 1.0_rprec
  do ii = 1, nn
    factorial_rprec = factorial_rprec * ii
  end do
  
end function factorial_rprec
!==============================================================================

!==============================================================================
recursive function permutations(Nobj, Nperm, objs) result(perms)
!==============================================================================
  implicit none
  ! Return value
  integer             :: perms(1:Nobj,1:Nperm) ! Array with the permutations
  ! Arguments
  integer, intent(in) :: Nobj                  ! Number of objects to permute
  integer, intent(in) :: Nperm                 ! Number of permutations
  integer, intent(in) :: objs(1:Nobj)          ! Array with the objects
  ! Local variables
  integer :: ii, jj, istart, iend, numperm
  integer :: objtemp(1:Nobj)
  
  if (Nobj == 2) then
    perms(:,1) = objs(:)
    perms(1,2) = objs(2)
    perms(2,2) = objs(1)
  else
    numperm = factorial(Nobj-1)
    do ii = 1, Nobj
      objtemp = objs
      objtemp(1) = objs(ii)
      objtemp(ii) = objs(1)
      istart = 1+numperm*(ii-1)
      iend   = numperm*ii 
      do jj=istart,iend
        perms(:,jj) = objtemp(:)
      end do
      perms(2:Nobj,istart:iend) = permutations(Nobj-1, numperm, objtemp(2:Nobj))
    end do
  end if
  
end function permutations
!==============================================================================

!==============================================================================
recursive function choices(Narrays, arrays, Nchoices) result(choicesArray)
!==============================================================================
! Picks up one item from each of arrays and generates all possible choices
!==============================================================================
  use types, only : arrayOfArrays
  implicit none
  ! Return value
  integer :: choicesArray(1:Narrays, 1:Nchoices) ! Array with the permutations
  ! Arguments
  integer, intent(in) :: Narrays, Nchoices
  type(arrayOfArrays) :: arrays(Narrays)
  ! Local variables
  integer :: ii, jj, iarray, istart, iend, numchoices

  if (Narrays == 1) then
    do ii = 1, size(arrays(1)%oneDintArr)
      choicesArray(1, ii) = arrays(1)%oneDintArr(ii)
    end do
  else
    numchoices = Nchoices / size(arrays(1)%oneDintArr)
    do ii = 1, size(arrays(1)%oneDintArr)
      istart = 1+numchoices*(ii-1)
      iend   = numchoices*ii
      do jj=istart,iend
        choicesArray(1,jj) = arrays(1)%oneDintArr(ii)
      end do
      choicesArray(2:, istart:iend) = choices(Narrays-1, arrays(2:), numchoices)
    end do
  end if
  
end function choices
!==============================================================================

!==============================================================================
subroutine permGroups(nGroups, nElmntsArr, NpermGroups, permElmnts)
!==============================================================================
! Gets all permutations of nGroups and elements in the groups
! Example:
!   Input
!     nGroups    = 3
!     nElmntsArr = [3, 2, 2]
!     e.g. three groups of integers [1, 2, 3], [1, 2], and [1, 2]
!   Output
!     permElmnts(1)%groups(1)%oneDintArr = ['1','2','3']
!     permElmnts(1)%groups(2)%oneDintArr = [1, 2]
!     permElmnts(1)%groups(3)%oneDintArr = [1, 2]
!     permElmnts(2)%groups(1)%oneDintArr = ['1','2','3']
!     permElmnts(2)%groups(2)%oneDintArr = [1, 2]
!     permElmnts(2)%groups(3)%oneDintArr = [2, 1]
!     ...
!==============================================================================
  use stopper, only : stopMLatomF
  use types,   only : arrayOfArrays, nuclGroups
  implicit none
  ! Arguments
  integer, intent(in) :: nGroups, NpermGroups
  integer, intent(in) :: nElmntsArr(1:nGroups)
  type(nuclGroups), intent(out) :: permElmnts(NpermGroups)
  ! Local variables
  integer :: NpermLoc
  integer :: ii, igroup, iperm, Error
  integer, allocatable :: choicesArray(:,:) ! Array with the permutations
  integer, allocatable :: permGroupAtoms(:,:), permInds(:)
  type(arrayOfArrays), allocatable :: permArrays(:)

  ! Get all possible combinations of permuted groups
  allocate(permArrays(nGroups),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for permArrays array')
  do igroup = 1, nGroups
    NpermLoc = int(factorial_rprec(nElmntsArr(igroup)))
    allocate(permArrays(igroup)%oneDintArr(NpermLoc),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permArrays%oneDintArr array')
    do ii = 1, NpermLoc
      permArrays(igroup)%oneDintArr(ii) = ii
    end do
  end do
  allocate(choicesArray(1:nGroups, 1:NpermGroups),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for permArrays array')
  choicesArray = choices(nGroups, permArrays, NpermGroups)
  do igroup = 1, nGroups
    deallocate(permArrays(igroup)%oneDintArr)
  end do
  deallocate(permArrays)

  do iperm = 1, NpermGroups
    allocate(permElmnts(iperm)%groups(nGroups))
    do igroup = 1, nGroups
      allocate(permElmnts(iperm)%groups(igroup)%oneDintArr(nElmntsArr(igroup)))
    end do
  end do
  
  do igroup = 1, nGroups
    NpermLoc = int(factorial_rprec(nElmntsArr(igroup)))
    allocate(permGroupAtoms(nElmntsArr(igroup), NpermLoc),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permGroupAtoms array')
    allocate(permInds(nElmntsArr(igroup)),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for permInds array')
    do ii = 1, nElmntsArr(igroup)
      permInds(ii) = ii
    end do
    permGroupAtoms = permutations(nElmntsArr(igroup),NpermLoc,permInds)
    deallocate(permInds)
    do iperm = 1, NpermGroups
       permElmnts(iperm)%groups(igroup)%oneDintArr = permGroupAtoms(:,choicesArray(igroup,iperm))
    end do
    deallocate(permGroupAtoms)
  end do
  deallocate(choicesArray)
    
end subroutine permGroups
!==============================================================================

!==============================================================================
function Rij(xyzI, xyzJ)
!==============================================================================
! Calculates distance between two points i and j defined by
! Cartesian coordinates xyzI and xyzJ, respectively
!==============================================================================
  implicit none
  real(kind=rprec) :: Rij
  ! Arguments
  real(kind=rprec), intent(in) :: xyzI(1:3)
  real(kind=rprec), intent(in) :: xyzJ(1:3)
  ! Local variables
  integer :: nn, ii, jj
  
  Rij = sqrt((xyzI(1)-xyzJ(1))**2+(xyzI(2)-xyzJ(2))**2+(xyzI(3)-xyzJ(3))**2)

end function Rij
!==============================================================================

end module mathUtils
