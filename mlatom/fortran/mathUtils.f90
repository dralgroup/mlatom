module mathUtils 
!character(len=*) :: matDecomp
logical :: Cholesky = .true.
logical :: refine = .false.
logical :: shutup = .true.
contains 

subroutine solveSysLinEqs(nn, nrhs, AA, bb, xx)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
!==============================================================================
  use stopper,       only : stopKREG
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=8), intent(inout) :: AA(1:nn,1:nn)
  real(kind=8), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=8), intent(out)   :: xx(1:nn, 1:nrhs)
  ! Local variable
  integer :: Error

  Error = 0
  if (Cholesky) then
    call solveSysLinEqsCholesky(nn, nrhs, AA, bb, xx, Error)
    if (Error > 0) call solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
!   elseif (option%matDecomp == 'LU') then
!     call solveSysLinEqsLU(nn, nrhs, AA, bb, xx)
!   elseif (option%matDecomp == 'Bunch-Kaufman') then
!     call solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
  end if

end subroutine solveSysLinEqs


subroutine solveSysLinEqsCholesky(nn, nrhs, AA, bb, xx, Error)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
! It uses Cholesky factorization and MKL/Lapack subroutines
!==============================================================================
  use stopper,       only : stopKREG
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=8), intent(inout) :: AA(1:nn,1:nn)
  real(kind=8), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=8), intent(out)   :: xx(1:nn, 1:nrhs)
  integer, intent(out)            :: Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: iwork(:)
  real(kind=8), allocatable   :: Aoriginal(:,:), ferr(:), berr(:), work(:)

  Error = 0
  
  ! Allocate arrays
  if (refine) then
    allocate(Aoriginal(1:nn,1:nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate Aoriginal array')
    Aoriginal = AA
  end if
  
  ! Cholesky decomposition
  call dpotrf('U', nn, AA, nn, Error)

  if (Error > 0) then
    if (.not. shutup) then
      write(6,'(A)') ' Cholesky decomposition failed. Attempting Bunch-Kaufman decomposition'
    end if 
    if(allocated(Aoriginal))    deallocate(Aoriginal)
    return
  elseif (Error < 0) then
    call stopKREG('solveSysLinEqsCholesky: Argument had an illegal value')
  end if

  ! Solve a system of linear equations
  xx = bb
  call dpotrs('U', nn, nrhs, AA, nn, xx, nn, Error)

  if (Error < 0) then
    call stopKREG('solveSysLinEqsCholesky: Argument had an illegal value')
  end if
  
  if (refine) then
    allocate(ferr(1:nrhs),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate ferr array')
    allocate(berr(1:nrhs),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate berr array')
    allocate(work(1:3*nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate work array')
    allocate(iwork(1:nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate iwork array')
    call dporfs('U', nn, nrhs, Aoriginal, nn, AA, nn, bb, nn, xx, nn, ferr, berr, work, iwork, Error)
  end if
  
  ! Free up memory
  if(allocated(Aoriginal))    deallocate(Aoriginal)
  if(allocated(ferr))         deallocate(ferr)
  if(allocated(berr))         deallocate(berr)
  if(allocated(work))         deallocate(work)
  if(allocated(iwork))        deallocate(iwork)

end subroutine solveSysLinEqsCholesky

subroutine solveSysLinEqsBK(nn, nrhs, AA, bb, xx)
!==============================================================================
! Subroutine solves a system of linear equations AA*xx = bb
! It uses Bunch-Kaufman factorization and MKL/Lapack subroutines
!==============================================================================
  use stopper,       only : stopKREG
  implicit none
  ! Arguments
  integer, intent(in)             :: nn
  integer, intent(in)             :: nrhs
  real(kind=8), intent(inout) :: AA(1:nn,1:nn)
  real(kind=8), intent(in)    :: bb(1:nn, 1:nrhs)
  real(kind=8), intent(out)   :: xx(1:nn, 1:nrhs)
  ! Local variables
  integer :: ii, jj, lwork, Error
  ! Arrays
  integer*8                       :: dateStart(1:8)   ! Time and date, when MLatomF starts
  integer, allocatable            :: pivotIndices(:), iwork(:)
  real(kind=8), allocatable   :: Aoriginal(:,:), ferr(:), berr(:), work(:)
  ! External MKL or Lapack subroutine
  integer, external :: ilaenv


  Error = 0
  
  ! Determine optimal lwork size
  lwork = nn * ilaenv(1, 'dsytrf', 'U', nn, nn, -1, -1)

  ! Allocate arrays
  allocate(pivotIndices(1:nn),stat=Error)
  if(Error/=0)call stopKREG('Unable to allocate pivot indices')
  allocate(work(1:lwork),stat=Error)
  if(Error/=0)call stopKREG('solveSysLinEqsBK: Unable to allocate work array')
  if (refine) then
    allocate(Aoriginal(1:nn,1:nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate Aoriginal array')
    Aoriginal = AA
  end if
  
  !  Bunch-Kaufman decomposition
  call dsytrf('U', nn, AA, nn, pivotIndices, work, lwork, Error)

  if (error > 0) then
    if (.not. shutup) then 
      write(6,'(a)') 'solveSysLinEqsBK: Bunch-Kaufman factorization could not be completed'
    end if 
  elseif (error < 0) then
    call stopKREG('solveSysLinEqsBK: Argument had an illegal value')
  end if

  ! Solve a system of linear equations
  xx = bb
  call dsytrs('U', nn, nrhs, AA, nn, pivotIndices, xx, nn, Error)


  if (Error < 0) then
    call stopKREG('solveSysLinEqsBK: Argument had an illegal value')
  end if

  if (refine) then
    allocate(ferr(1:nrhs),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate ferr array')
    allocate(berr(1:nrhs),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate berr array')
    if(allocated(work))         deallocate(work)
    allocate(work(1:3*nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate work array')
    allocate(iwork(1:nn),stat=Error)
    if(Error/=0)call stopKREG('Unable to allocate iwork array')
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

end module mathUtils 