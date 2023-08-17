
  !---------------------------------------------------------------------------! 
  ! A_KRR: kernel ridge regression calculations                               ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !                     Yi-Fan Hou (co-implementation of derivatives)         !
  !---------------------------------------------------------------------------! 

module A_KRR
  use A_KRR_kernel,  only : NtrTot, NtrVal, NtrGrXYZ
  use A_KRR_kernel,  only :         itrval, itrgrxyz
  use dataset,       only : arrayOfArrays, Yprior, Int_const, Nprops, NgradXYZmax, NgradXYZ
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  implicit none
  ! Variables
  integer                          :: lgOptDepth     ! Depth of logarithmic optimization
  real(kind=rprec)                 :: lambda         ! Regularization parameter
  real(kind=rprec)                 :: lambdaGradXYZ  ! Regularization parameter for XYZ gradients
  integer                          :: lgLambdaPoints ! Number of points on a logarithmic grid for search optimal lambda
  real(kind=rprec)                 :: lgLambdaL      ! The lowest  value of log2(lambda) to try
  real(kind=rprec)                 :: lgLambdaH      ! The highest value of log2(lambda) to try
  ! Arrays
  real(kind=rprec),    allocatable :: alpha(:,:)     ! Regression coefficients
  real(kind=rprec),    allocatable :: Ytrain(:,:)    ! Reference values for training
  ! 
  real(kind=rprec)                 :: CVoptError

contains

!==============================================================================
subroutine init_KRR()
!==============================================================================
  use A_KRR_kernel,  only : getSigma, getPeriod, getSigmaP, getNN
  use A_KRR_kernel,  only : getC, getD
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  integer :: Error

  ! Get user-defined lambda or request for optimizing it
  call getLambda()
  if (trim(option%kernel) == 'polynomial') then
    ! Get user-defined c and d or request for optimizing them
    call getC()
    call getD()
  elseif ((trim(option%kernel) == 'Gaussian')  .or. &
          (trim(option%kernel) == 'Laplacian') .or. &
          (trim(option%kernel) == 'exponential')) then
    ! Get user-defined sigma or request for optimizing it
    call getSigma()
    if (option%periodKernel) call getPeriod
    if (option%decayKernel)  then
      call getSigmaP
      call getPeriod
    end if
  elseif (trim(option%kernel) == 'Matern') then
    ! Get user-defined c and d or request for optimizing them
    call getNN()
    call getSigma()
  end if
  
  select case (option%KRRtask)
    case ('learnVal')
      option%learnVal = .true.
      Nprops = 1
      option%calcVal = .true.
    case ('learnGradXYZ')
      option%learnGradXYZ = .true.
      Nprops = 1
      if (option%YestFile /= '' .or. option%Yfile /= '') then 
        option%calcVal = .true.
        option%prior = 'meandevi'
      end if 
      option%calcGradXYZ = .true.
      option%onTheFly = .true.
    case ('learnValGradXYZ')
      option%learnVal = .true.
      option%learnGradXYZ = .true.
      Nprops = 1
      option%calcVal = .true.
      option%calcGradXYZ = .true.
      option%onTheFly = .true.
  end select

end subroutine init_KRR
!==============================================================================

!==============================================================================
subroutine optHyper_KRR(hyperOptDataPt, indicesCVoptSplitsPt, doNotCalcStats)
!==============================================================================
  use A_KRR_kernel,  only : sigma, c, d
  use A_KRR_kernel,  only : K, Kvalidate, calcKernel, calc_Kvalidate
  use A_KRR_kernel,  only : selectPerm, NselectPerm
  use dataset,       only : Yest, YgradXYZest, YestCVopt, YgradXYZestCVopt
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use dataset,       only : NsubtrainGradXYZ, indicesSubtrainGradXYZ, NvalidateGradXYZ, indicesValidateGradXYZ
  use dataset,       only : Ntrain, indicesTrain, splitAndSample, indicesCVoptSplits
  use dataset,       only : CVoptFolds, trainTestData, optData
  use dataset,       only : Y, Yest
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none 
  ! Arguments
  type(optData),       intent(inout), target   :: hyperOptDataPt          ! Information necessary to hyperparmeter optimization
  type(arrayOfArrays), intent(in)              :: indicesCVoptSplitsPt(:) ! Indices of CV splits used for optimization of hyperparameters
  logical,             intent(in),    optional :: doNotCalcStats
  ! Arrays
  integer*8          :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  logical            :: doNotCalcStatsLocal
  integer            :: a, i, j, ip, mm, nn, nnsize, stsize, newsize, Nsets, Error
  integer            :: tt, bb, uu, ireal, jreal, icounter, jcounter
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  ! Prepare for optimizing parameters
  call getLgOptDepth()

  ! Split the training set for cross-validation search of optimal hyperparameters
  if (option%CVopt) then
    Nsets = size(indicesCVoptSplitsPt)
    if (option%onTheFly) Nsets = 1
    allocate(CVoptFolds(1:Nsets),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate CVoptFolds')
    allocate(K(1:Nsets),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate K')
    allocate(Kvalidate(1:Nsets),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate Kvalidate')
    if (.not. option%onTheFly) then
      do mm = 1, Nsets
        ! Get the sub-training and validation sets for the fold mm
        Nsubtrain       => CVoptFolds(mm)%Nsubtrain
        Nvalidate       => CVoptFolds(mm)%Nvalidate
        Nvalidate       = size(indicesCVoptSplitsPt(mm)%oneDintArr)
        Nsubtrain       = Ntrain - Nvalidate
        allocate(CVoptFolds(mm)%indicesValidate(1:Nvalidate),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate indicesValidate')
        allocate(CVoptFolds(mm)%indicesSubtrain(1:Nsubtrain),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate indicesSubtrain')
        indicesSubtrain => CVoptFolds(mm)%indicesSubtrain
        indicesValidate => CVoptFolds(mm)%indicesValidate
        indicesValidate(1:Nvalidate) = indicesCVoptSplitsPt(mm)%oneDintArr(1:Nvalidate)
        indicesSubtrain = 0
        stsize          = 0
        newsize         = 0
        do nn = 1, Nsets
          if (nn == mm) cycle
          nnsize = size(indicesCVoptSplitsPt(nn)%oneDintArr)
          newsize = stsize + nnsize
          indicesSubtrain(stsize+1:newsize) = indicesCVoptSplitsPt(nn)%oneDintArr(1:nnsize)
          stsize = newsize
        end do
        call getYtrain(Nsubtrain, indicesSubtrain, Nprops)
        allocate(K(mm)%twoDrArr(1:NtrTot,1:NtrTot),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate K(mm)%twoDrArr')
          allocate(Kvalidate(mm)%oneDrArr(1:Nvalidate*Nsubtrain),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate Kvalidate(mm)%oneDrArr')
      end do
    end if
  else
    ! Allocate arrays
    Nsubtrain       => hyperOptDataPt%Nsubtrain
    Nvalidate       => hyperOptDataPt%Nvalidate
    indicesSubtrain => hyperOptDataPt%indicesSubtrain
    indicesValidate => hyperOptDataPt%indicesValidate
      call getYtrain(Nsubtrain, indicesSubtrain, Nprops)
    allocate(K(1),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate K')
    allocate(K(1)%twoDrArr(1:NtrTot,1:NtrTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate K(1)%twoDrArr')
    if (.not. option%onTheFly) then
      allocate(Kvalidate(1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate Kvalidate')
        allocate(Kvalidate(1)%oneDrArr(1:Nvalidate*Nsubtrain),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate Kvalidate')
      end if 
    end if

  ! Check optimization of what hyper-parameters is requested
  if (option%c == 'OPT') then
    call optLog2Grid('c', hyperOptDataPt, indicesCVoptSplitsPt)
    if(option%lambda == 'OPT') write(6,'(a,F25.14)') ' Optimal value of lambda is ', lambda
    write(6,'(a,F25.14)') ' Optimal value of c is ', c
    if(option%d == 'OPT') then
      call optLog2Grid('d', hyperOptDataPt, indicesCVoptSplitsPt)
      write(6,'(a,F25.14)') ' Optimal value of d is ', d
    end if
  elseif (option%d == 'OPT') then
    call optLog2Grid('d', hyperOptDataPt, indicesCVoptSplitsPt)
    write(6,'(a,F25.14)') ' Optimal value of d is ', d
    if(option%lambda == 'OPT') write(6,'(a,F25.14)') ' Optimal value of lambda is ', lambda
  elseif (option%sigma == 'OPT') then
    call optLog2Grid('sigma', hyperOptDataPt, indicesCVoptSplitsPt)
    if(option%lambda == 'OPT') write(6,'(a,F25.14)') ' Optimal value of lambda is ', lambda
    write(6,'(a,F25.14)') ' Optimal value of sigma is ', sigma
  elseif (option%lambda == 'OPT') then
    if (option%CVopt .and. .not. option%onTheFly) then
      do mm = 1, Nsets
        call getYtrain(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, Nprops)
        call calcKernel(K(mm)%twoDrArr)
          call calc_Kvalidate(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, &
                              CVoptFolds(mm)%Nvalidate, &
                              CVoptFolds(mm)%indicesValidate, Kvalidate(mm)%oneDrArr)
      end do
    elseif (.not. option%CVopt) then
      ! Calculate kernel matrix
      call calcKernel(K(1)%twoDrArr)
      if (.not. option%onTheFly) then 
          call calc_Kvalidate(Nsubtrain, indicesSubtrain, &
                              Nvalidate, indicesValidate, Kvalidate(1)%oneDrArr)
      end if
    end if
    call optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
    write(6,'(a,F25.14)') ' Optimal value of lambda is ', lambda
  end if

  doNotCalcStatsLocal = .false.
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) doNotCalcStatsLocal = .true.
  end if
  if (.not. option%CVopt .and. .not. doNotCalcStatsLocal) then
    if (option%prior/='0') write(6,'(" Prior value = ",F25.13)') Yprior
    ! Predict values using ML model for its own sub-training set
    call calcEst_KRR(Nsubtrain, indicesSubtrain)
    ! Calculate errors of the predicted values
      write(stmp,'(I0," entries in the sub-training set")') Nsubtrain
      call calc_stat_measures(Nsubtrain, indicesSubtrain, comment=trim(stmp))
    write(6,'(a)') ''
    
    ! Predict values using ML model for its own validation set  
    if (option%onTheFly) then
      call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate)
    else
        call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                          indicesValidate, KvalidateMatrix=Kvalidate(1)%oneDrArr)
    end if
  end if
  
  ! Free up memory
  call cleanUp_KRR()
  if (allocated(CVoptFolds)) then
    do i = 1, size(CVoptFolds)
      if(allocated(CVoptFolds(i)%indicesSubtrain)) deallocate(CVoptFolds(i)%indicesSubtrain)
      if(allocated(CVoptFolds(i)%indicesValidate)) deallocate(CVoptFolds(i)%indicesValidate)
    end do
    deallocate(CVoptFolds)
  end if

  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Hyperparameter optimization time:')

end subroutine optHyper_KRR
!==============================================================================

!==============================================================================
subroutine getLgOptDepth()
!==============================================================================
  use optionsModule, only : option
  implicit none

  read(option%lgOptDepth,*) lgOptDepth

end subroutine getLgOptDepth
!==============================================================================

!==============================================================================
subroutine train_KRR(doNotCalcStats)
!==============================================================================
! Trains KRR on a vector of scalar reference values, i.e. one value per datum
!==============================================================================
  use A_KRR_kernel,  only : K, calcKernel, sigma
  use dataset,       only : Y, Yest, Ntrain, indicesTrain, permutedIatoms, Nperm
  use dataset,       only : NtrainGradXYZ, indicesTrainGradXYZ
  use dataset,       only : YgradXYZ, YgradXYZest
  use dataset,       only : XvecSize, Nperm
  use MLstatistics,  only : calc_stat_measures
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  logical, intent(in), optional :: doNotCalcStats
  ! Arrays
  integer*8 :: dateStart(1:8)  ! Time and date, when subroutine starts
  integer*8 :: dateStart2(1:8) ! Time and date
  real(kind=rprec), allocatable :: one(:)
  ! Local variables
  integer   :: a, i, j, ip, Error, Xsize
  integer   :: tt,ireal,jreal,bb,uu,icounter,jcounter
  character(len=256) :: stmp   ! Temporary variable

  ! Benchmark
  if(option%benchmark .and. .not. present(doNotCalcStats)) call date_and_time(values=dateStart)
  
  ! Get vector(s) with reference values for training
  call getYtrain(Ntrain, indicesTrain, Nprops)

  ! Allocate arrays
  allocate(K(1),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate K')
  allocate(K(1)%twoDrArr(1:NtrTot,1:NtrTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate K(1)%twoDrArr')

  ! Calculate kernel matrix
  call calcKernel(K(1)%twoDrArr)
  ! Calculate regression coefficients alpha
  call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
  if (option%kernel=='linear') call KRR2MLR()
  deallocate(Ytrain)
  if (option%prior/='0' .and. .not. present(doNotCalcStats)) write(6,'(" Prior value = ",F25.13)') Yprior
  
  ! Predict values using ML model for its own training set
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  
  if(option%benchmark .and. .not. present(doNotCalcStats)) call date_and_time(values=dateStart)
  
  call calcEst_KRR(Ntrain, indicesTrain)
  ! Calculate errors of the predicted values
  write(stmp,'(I0," entries in the training set")') Ntrain
  call calc_stat_measures(Ntrain, indicesTrain, comment=trim(stmp))
  write(6,'(a)') ''

  ! Benchmark
  if(option%benchmark .and. .not. present(doNotCalcStats)) call timeTaken(dateStart,'Training time:')

end subroutine train_KRR
!==============================================================================

!==============================================================================
subroutine KRR2MLR()
!==============================================================================
! Calculates the coefficients beta of multiple linear regression
! given KRR coefficients alpha and training points
!==============================================================================
  use dataset, only : X, XvecSize, Ntrain, indicesTrain
  implicit none
  integer :: ii, jj
  real(kind=rprec) :: MLRbeta(1:XvecSize)
  
  MLRbeta = 0.0_rprec
  write(6,'(/A)') ' Multiple linear regression coefficients:'
  do jj = 1, XvecSize
    do ii = 1, Ntrain
      MLRbeta(jj) = MLRbeta(jj) + alpha(ii,1) * X(jj,indicesTrain(ii))
    end do
    write(6,'(F25.13)') MLRbeta(jj)
  end do
  write(6,*) ''

end subroutine KRR2MLR

!==============================================================================
subroutine getYtrain(NtrainPoints,indicesTrainPoints,Nprops)
  use dataset,      only : Y, Ygrad, YgradXYZ, XvecSize, NAtomsMax, NgradForAtom
  use dataset,      only : gradForXel
  use A_KRR_kernel, only : NselectPerm, indicesSelectPerm
  implicit none
  ! Arguments
  integer, intent(in) :: NtrainPoints                       ! Number  of the training points
  integer, intent(in) :: indicesTrainPoints(1:NtrainPoints) ! Indices of the training points
  integer, intent(in) :: Nprops                             ! Number  of the properties
  ! Local variables
  integer :: ii, jj, ip, icounter, Error
  integer :: PP
  
  NtrTot = 0 ; NtrVal = 0 ; NtrGrXYZ = 0
  if (allocated(Ytrain)) deallocate(Ytrain)
  if (allocated(itrval)) deallocate(itrval)
  if (allocated(itrgrxyz)) deallocate(itrgrxyz)

  if (option%prior=='meandevi') then
    Int_const = 0.0_rprec 
    do ii=1, NtrainPoints
      Int_const = Int_const + Y(indicesTrainPoints(ii))
    end do 
    Int_const = Int_const / NtrainPoints
  end if 

  
  if (option%learnVal) then
      NtrVal = NtrainPoints
      allocate(itrval(1:NtrVal),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate itrval')
  end if
  if (option%learnGradXYZ) then
      !NtrGrXYZ = NtrainPoints * NgradXYZmax * 3
      NtrGrXYZ = 0
      do ip=1, NtrainPoints
        if (NgradXYZ(indicesTrainPoints(ip)) /= 0) NtrGrXYZ = NtrGrXYZ + 3 * NgradXYZ(indicesTrainPoints(ip))
      end do
      allocate(itrgrxyz(1:NtrGrXYZ,3),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate itrgrxyz')
  end if
  NtrTot = NtrVal + NtrGrXYZ
  allocate(Ytrain(1:NtrTot,1:Nprops),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate Ytrain')
  if (option%learnVal) then
      do ii = 1, NtrainPoints
        Ytrain(ii,1) = Y(indicesTrainPoints(ii))
        itrval(ii)   = indicesTrainPoints(ii)
      end do
  end if
  if (option%learnGradXYZ) then
    icounter = 0
      do ip=1,NtrainPoints
        if (NgradXYZ(indicesTrainPoints(ip)) /= 0) then
          do ii = 1, NgradXYZ(indicesTrainPoints(ip))
            do jj = 1, 3
              icounter = icounter + 1
              Ytrain(NtrVal+icounter, 1) = YgradXYZ(jj,ii,indicesTrainPoints(ip))
              itrgrxyz(icounter,1)             = indicesTrainPoints(ip)
              itrgrxyz(icounter,2)             = ii
              itrgrxyz(icounter,3)             = jj
            end do
          end do
        end if
      end do
  end if

end subroutine getYtrain
!==============================================================================

!==============================================================================
subroutine validate_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                        NvalidatePoints, indicesValidatePoints, &
                        KvalidateMatrix,                        & ! optional
                        doNotCalcStats, validationError)          ! optional
!==============================================================================
  use A_KRR_kernel,  only : Kvalidate
  use dataset,       only : Yest, YestCVopt, Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,          intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
  integer,          intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,          intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
  integer,          intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec), intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  logical,          intent(in),  optional :: doNotCalcStats
  real(kind=rprec), intent(out), optional :: validationError
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer            :: a, i
  character(len=256) :: stmp  ! Temporary variable
  
  select case (option%KRRtask)
    case ('learnVal')
      call validate1_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                         NvalidatePoints, indicesValidatePoints, &
                         KvalidateMatrix,                        &
                         doNotCalcStats, validationError) 
    case ('learnGradXYZ', 'learnValGradXYZ')
      call validateGradXYZ_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                               NvalidatePoints, indicesValidatePoints, &
                               KvalidateMatrix,                        &
                               doNotCalcStats, validationError)
  end select

end subroutine validate_KRR
!==============================================================================

!==============================================================================
subroutine validate1_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                        NvalidatePoints, indicesValidatePoints, &
                        KvalidateMatrix,                        & ! optional
                        doNotCalcStats, validationError)          ! optional
!==============================================================================
  use A_KRR_kernel,  only : Kvalidate
  use dataset,       only : Yest, YestCVopt, Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,          intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
  integer,          intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,          intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
  integer,          intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec), intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  logical,          intent(in),  optional :: doNotCalcStats
  real(kind=rprec), intent(out), optional :: validationError
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer            :: a, i
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if (.not. present(validationError) .and. .not. present(doNotCalcStats)) then
    if(option%benchmark) call date_and_time(values=dateStart)
  end if
  
  ! Predict values using ML model
  if (option%onTheFly) then
    call calcEst_KRR(NvalidatePoints, indicesValidatePoints)
  elseif (present(KvalidateMatrix)) then
    ! Predict values using ML model for its validation set
    Yest = Yprior
    !$OMP PARALLEL DO PRIVATE(a,i) &
    !$OMP SHARED(Yest,indicesValidatePoints,Kvalidate,alpha,NsubtrainPoints,NvalidatePoints) &
    !$OMP SCHEDULE(STATIC)
    do a=1, NvalidatePoints
      do i=1,NsubtrainPoints
        Yest(indicesValidatePoints(a)) = Yest(indicesValidatePoints(a)) &
                                       + alpha(i,1) * KvalidateMatrix((a-1)*NsubtrainPoints+i)
      end do
    end do
    !$OMP END PARALLEL DO
  end if
  
  if (option%CVopt) then
    do i = 1, NvalidatePoints
      YestCVopt(indicesValidatePoints(i)) = Yest(indicesValidatePoints(i))
    end do
  end if
  
  ! Calculate errors of predicted values
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  if (present(validationError)) then
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, returnValue=validationError)
  else
    write(stmp,'(I0," entries in the validation set")') NvalidatePoints
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, comment=trim(stmp))
    write(6,'(a)') ''
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Validating time:')
  end if

end subroutine validate1_KRR
!==============================================================================

!==============================================================================
subroutine validate2_KRR(NsubtrainPoints, indicesSubtrainPoints, &
  NvalidatePoints, indicesValidatePoints, &
  KvalidateMatrix,                        & ! optional
  doNotCalcStats, validationError)          ! optional
use A_KRR_kernel,  only : Kvalidate, NselectPerm
use dataset,       only : Yest, YestCVopt, Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
use MLstatistics,  only : calc_stat_measures
use optionsModule, only : option
use timing,        only : timeTaken
implicit none 
! Arguments
integer,          intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
integer,          intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
integer,          intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
integer,          intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
real(kind=rprec), intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints*NselectPerm) ! Array with extra kernel matrix elements
logical,          intent(in),  optional :: doNotCalcStats
real(kind=rprec), intent(out), optional :: validationError
! Arrays 
integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
! Local variables
integer            :: a, i
character(len=256) :: stmp  ! Temporary variable

! Benchmark 
if (.not. present(validationError) .and. .not. present(doNotCalcStats)) then 
if(option%benchmark) call date_and_time(values=dateStart) 
end if 

! Predict values using ML model
if (option%onTheFly .or. .not. present(KvalidateMatrix)) then 
call calcEst_KRR(NvalidatePoints, indicesValidatePoints) ! May have bugs when using onTheFly
else if (present(KvalidateMatrix)) then
! Predict values using ML model for its validation set
Yest = Yprior 
!$OMP PARALLEL DO PRIVATE(a,i) &
!$OMP SHARED(Yest,indicesValidatePoints,Kvalidate,alpha,NsubtrainPoints,NvalidatePoints) &
!$OMP SCHEDULE(STATIC)
do a=1, NvalidatePoints
do i=1,NsubtrainPoints*NselectPerm
Yest(indicesValidatePoints(a)) = Yest(indicesValidatePoints(a)) &
                 + alpha(i,1) * KvalidateMatrix((a-1)*NsubtrainPoints*NselectPerm+i)
end do
end do
!$OMP END PARALLEL DO
end if 

if (option%CVopt) then
do i = 1, NvalidatePoints
YestCVopt(indicesValidatePoints(i)) = Yest(indicesValidatePoints(i))
end do
end if

! Calculte errors of predicted values 
if (present(doNotCalcStats)) then 
if (doNotCalcStats) return 
end if 
if (present(validationError)) then 
call calc_stat_measures(NvalidatePoints, indicesValidatePoints, returnValue=validationError)
else
write(stmp,'(I0," entries in the validation set")') NvalidatePoints
call calc_stat_measures(NvalidatePoints, indicesValidatePoints, comment=trim(stmp))
write(6,'(a)') ''
! Benchmark
if(option%benchmark) call timeTaken(dateStart,'Validating time:')
end if

end subroutine validate2_KRR
!==============================================================================

!==============================================================================
subroutine validateGrad_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                        NvalidatePoints, indicesValidatePoints, &
                        KvalidateMatrix,                        & ! optional
                        doNotCalcStats, validationError)          ! optional
!==============================================================================
  use A_KRR_kernel,  only : Kvalidate
  use dataset,       only : YgradXYZest, YgradXYZestCVopt
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,             intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
  integer,             intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,             intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
  integer,             intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec),    intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  logical,             intent(in),  optional :: doNotCalcStats
  real(kind=rprec),    intent(out), optional :: validationError
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer            :: a, i, j, ip
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if (.not. present(validationError) .and. .not. present(doNotCalcStats)) then
    if(option%benchmark) call date_and_time(values=dateStart)
  end if
  
  ! Predict values using ML model
  if (option%onTheFly) then
    call calcEst_KRR(NvalidatePoints, indicesValidatePoints)
  end if
  
  ! Calculate errors of predicted values
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  if (present(validationError)) then
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, returnValue=validationError)
  else
    write(stmp,'(I0," entries in the validation set")') NvalidatePoints
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, comment=trim(stmp))
    write(6,'(a)') ''
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Validating time:')
  end if

end subroutine validateGrad_KRR
!==============================================================================

!==============================================================================
subroutine validateGradXYZ_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                        NvalidatePoints, indicesValidatePoints, &
                        KvalidateMatrix,                        & ! optional
                        doNotCalcStats, validationError)          ! optional
!==============================================================================
  use A_KRR_kernel,  only : Kvalidate
  use dataset,       only : YgradXYZest, YgradXYZestCVopt
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,             intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
  integer,             intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,             intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
  integer,             intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec),    intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  logical,             intent(in),  optional :: doNotCalcStats
  real(kind=rprec),    intent(out), optional :: validationError
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer            :: a, i, j, ip
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if (.not. present(validationError) .and. .not. present(doNotCalcStats)) then
    if(option%benchmark) call date_and_time(values=dateStart)
  end if
  
  ! Predict values using ML model
  if (option%onTheFly) then
    call calcEst_KRR(NvalidatePoints, indicesValidatePoints)
  end if
  
  if (option%CVopt) then
    do i = 1, 3
      do j = 1, NgradXYZmax
        do ip = 1, NvalidatePoints
          YgradXYZestCVopt(i,j,indicesValidatePoints(ip)) = YgradXYZest(i,j,indicesValidatePoints(ip))
        end do
      end do
    end do
  end if
  
  ! Calculate errors of predicted values
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  if (present(validationError)) then
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, returnValue=validationError)
  else
    write(stmp,'(I0," entries in the validation set")') NvalidatePoints
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, comment=trim(stmp))
    write(6,'(a)') ''
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Validating time:')
  end if

end subroutine validateGradXYZ_KRR
!==============================================================================

!==============================================================================
subroutine validateGradXYZCompnts_KRR(NsubtrainPoints, indicesSubtrainPoints, &
                        NvalidatePoints, indicesValidatePoints, &
                        KvalidateMatrix,                        & ! optional
                        doNotCalcStats, validationError)          ! optional
!==============================================================================
  use A_KRR_kernel,  only : Kvalidate
  use dataset,       only : YgradXYZest, YgradXYZestCVopt
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,             intent(in)            :: NsubtrainPoints                                    ! Number  of the training points
  integer,             intent(in)            :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,             intent(in)            :: NvalidatePoints                                    ! Number  of the validation points
  integer,             intent(in)            :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec),    intent(in),  optional :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  logical,             intent(in),  optional :: doNotCalcStats
  real(kind=rprec),    intent(out), optional :: validationError
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer            :: a, i, j, ip
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if (.not. present(validationError) .and. .not. present(doNotCalcStats)) then
    if(option%benchmark) call date_and_time(values=dateStart)
  end if
  
  ! Predict values using ML model
  if (option%onTheFly) then
    call calcEst_KRR(NvalidatePoints, indicesValidatePoints)
  elseif (present(KvalidateMatrix)) then
    ! Predict values using ML model for its validation set
    YgradXYZest = 0.0_rprec  
    !$OMP PARALLEL DO PRIVATE(a,i,j,ip) &
    !$OMP SHARED(YgradXYZest,indicesValidatePoints,Kvalidate,alpha,NsubtrainPoints,NvalidatePoints) &
    !$OMP SCHEDULE(STATIC)
    do i = 1, 3
      do j = 1, NgradXYZmax
        do a=1, NvalidatePoints
          YgradXYZest(i,j,indicesValidatePoints(a)) = 0.0_rprec
          do ip=1,NsubtrainPoints
            YgradXYZest(i,j,indicesValidatePoints(a)) = YgradXYZest(i,j,indicesValidatePoints(a)) &
             + alpha(ip,i+3*(j-1)) * KvalidateMatrix((a-1)*NsubtrainPoints+ip)
          end do
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end if
  
  if (option%CVopt) then
    do i = 1, 3
      do j = 1, NgradXYZmax
        do ip = 1, NvalidatePoints
          YgradXYZestCVopt(i,j,indicesValidatePoints(ip)) = YgradXYZest(i,j,indicesValidatePoints(ip))
        end do
      end do
    end do
  end if
  
  ! Calculate errors of predicted values
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  if (present(validationError)) then
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, returnValue=validationError)
  else
    write(stmp,'(I0," entries in the validation set")') NvalidatePoints*NgradXYZmax*3
    call calc_stat_measures(NvalidatePoints, indicesValidatePoints, comment=trim(stmp))
    write(6,'(a)') ''
    ! Benchmark
    if(option%benchmark) call timeTaken(dateStart,'Validating time:')
  end if

end subroutine validateGradXYZCompnts_KRR
!==============================================================================

!==============================================================================
subroutine test_KRR(doNotCalcStats)
!==============================================================================
  use dataset,       only : Ntrain, indicesTrain, Ntest, indicesTest
  use dataset,       only : YestCVtest, Yest
  use dataset,       only : YgradXYZest, YgradXYZestCVtest
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  logical, intent(in), optional :: doNotCalcStats
  ! Arrays
  integer*8 :: dateStart(1:8) ! Time and date, when MLatomF starts
  ! Local variables
  integer :: i, j, ip
  character(len=256) :: stmp  ! Temporary variable

  ! Benchmark
  if(option%benchmark .and. .not. present(doNotCalcStats)) call date_and_time(values=dateStart)
  
  ! Predict values using ML model (always on-the-fly)
  if (option%Yfile /= '') then
    call calcEst_KRR(Ntest, indicesTest)
    if (option%CVtest) then
      do i = 1, Ntest
        YestCVtest(indicesTest(i)) = Yest(indicesTest(i))
      end do
    end if
  end if
  if (option%YgradFile /= '') then
    call calcEst_KRR(Ntest, indicesTest)
  end if
  if (option%YgradXYZfile /= '') then
    call calcEst_KRR(Ntest, indicesTest)
    if (option%CVtest) then
      do i = 1, 3
        do j = 1, NgradXYZmax
          do ip = 1, Ntest
            YgradXYZestCVtest(i,j,indicesTest(ip)) = YgradXYZest(i,j,indicesTest(ip))
          end do
        end do
      end do
    end if
  end if
  
  ! Calculate errors of predicted values
  if (present(doNotCalcStats)) then
    if (doNotCalcStats) return
  end if
  write(stmp,'(I0," entries in the test set")') Ntest
  call calc_stat_measures(Ntest, indicesTest, comment=trim(stmp))
  write(6,'(a)') ''

  ! Benchmark
  if(option%benchmark .and. .not. present(doNotCalcStats)) call timeTaken(dateStart,'Test time:')

end subroutine test_KRR
!==============================================================================

!==============================================================================
subroutine calcAlpha(Nprops, Kmatrix, Yref)
!==============================================================================
! This subroutine calculates regression coefficients alpha
! by solving the system of equations (K+lambda*I)*alpha = Yref
! Input:  Kmatrix - K, kernel matrix
!         Yref    - Yref, vector(s) with reference value for various properties
!                   WARNING: may be modified if prior/=0 is requested
! Output: alpha  - alpha, the regression coefficients
!==============================================================================
  use dataset,       only : Y
  use mathUtils,     only : invertMatrix, solveSysLinEqs
  use optionsModule, only : option
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,          intent(in)  :: Nprops                                 ! Number  of the properties
  real(kind=rprec), intent(in)  :: Kmatrix(1:NtrTot,1:NtrTot) ! Indices of the training points
  real(kind=rprec), intent(inout) :: Yref(1:NtrTot,1:Nprops)        ! Reference values
  ! Variables
  integer                       :: jj, Error ! Loop index and error of (de)allocation
  real(kind=rprec)              :: lGrXYZloc ! Local lambdas for XYZ gradients
  ! Arrays
  integer*8                     :: dateStart(1:8) ! Time and date, when MLatomF starts
  real(kind=rprec), allocatable :: InvMat(:,:)    ! Inversed matrix

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  ! Allocate arrays
  if (allocated(alpha)) deallocate(alpha)
  allocate(alpha(1:NtrTot,Nprops),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate alpha')
  allocate(InvMat(1:NtrTot,1:NtrTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate inverse matrix')

  ! Add identity matrix multiplied by lambda to kernel matrix
  InvMat(1:NtrTot,1:NtrTot) = Kmatrix(1:NtrTot,1:NtrTot) ! To keep kernel matrix unmodified
  do jj=1,NtrVal
    InvMat(jj,jj) = InvMat(jj,jj) + lambda
  end do
  lGrXYZloc = lambda
  if (option%lambdaGradXYZ /= 'lambdaVal') lGrXYZloc = lambdaGradXYZ
  do jj=NtrVal+1,NtrTot
    InvMat(jj,jj) = InvMat(jj,jj) + lGrXYZloc
  end do
  
  ! Get prior (intercept)
  Yprior = 0.0_rprec  
  if (option%prior=='mean' .and. Nprops==1) then
    Yprior = 0.0_rprec
    do jj=1,NtrVal
      Yprior = Yprior + Yref(jj,1)
    end do
    Yprior = Yprior / NtrVal
    if (option%debug) write(6,'("   Mean prior = ",F25.13)') Yprior
  elseif (option%prior=='meandevi') then 
    Yprior = 0.0_rprec
  elseif (option%prior/='0') then
    if (Nprops > 1) call stopMLatomF('Only prior=0 can be used for these calculations')
    read(option%prior,*) Yprior    
  end if
  
  ! Center the data around prior
  if (option%prior/='0') then
    do jj=1,NtrVal
      Yref(jj,1) = Yref(jj,1) - Yprior
    end do
  end if

  ! Get vector with regression coefficients alpha:
  ! Solve the system of linear equations (K+lambda*I)*alpha = Yref
  if (option%invMatrix) then
    call invertMatrix(InvMat,NtrTot)
    alpha(:,1) = matmul(InvMat,Yref(:,1))
  else
    call solveSysLinEqs(NtrTot, Nprops, InvMat, Yref, alpha)
  end if

  call calcIntConst()

  ! Free up memory
  if(allocated(InvMat)) deallocate(InvMat)

  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Time for calculating regression coefficients:')
  
end subroutine calcAlpha
!==============================================================================

!==============================================================================
subroutine getLambda()
!==============================================================================
  use optionsModule, only : option
  implicit none

  if (option%lambda == 'OPT') then
    read(option%NlgLambda,*) lgLambdaPoints
    read(option%lgLambdaL,*) lgLambdaL
    read(option%lgLambdaH,*) lgLambdaH
    if(option%debug) write(6,'(a)') 'optimize lambda parameter'
    lambda=0.0_rprec
  else
    read(option%lambda,*) lambda
  end if
  if (option%lambdaGradXYZ /= 'lambdaVal') then
    read(option%lambdaGradXYZ,*) lambdaGradXYZ
  end if

end subroutine getLambda
!==============================================================================

!==============================================================================
subroutine calcEst_KRR(NestIndices, indicesForEst)
!==============================================================================
! This subroutine uses KRR model to make predictions
!==============================================================================
  use A_KRR_kernel,  only : Yest_KRR, YestGrad_KRR, YgradXYZest_KRR
  use A_KRR_kernel,  only : YgradXYZCompntsEst_KRR
  use A_KRR_kernel,  only : initializeAllArrays
  use dataset,       only : Yest, YestGrad, YgradXYZest, NgradXYZmax
  implicit none
  ! Arguments
  integer,          intent(in) :: NestIndices                        ! Size of indicesForEval
  integer,          intent(in) :: indicesForEst(1:NestIndices)       ! Indices for estimating
  ! Local variables
  integer :: i

  call initializeAllArrays()

  if (option%calcVal) then 
    Yest = Yprior
    !$OMP PARALLEL DO PRIVATE(i) SHARED(Yest,indicesForEst) SCHEDULE(STATIC)
    do i=1, NestIndices
      Yest(indicesForEst(i)) = Yprior + Yest_KRR(indicesForEst(i), alpha(:,1))
    end do
    !$OMP END PARALLEL DO
  end if
  
  if (option%calcGrad) then
    YestGrad = 0.0_rprec
    !$OMP PARALLEL DO PRIVATE(i) SHARED(YestGrad,indicesForEst) SCHEDULE(STATIC)
    do i=1, NestIndices
      YestGrad(:,indicesForEst(i)) = YestGrad_KRR(indicesForEst(i), alpha(:,1))
    end do
    !$OMP END PARALLEL DO
  end if
  
  if (option%calcGradXYZ) then
    YgradXYZest = 0.0_rprec
    !$OMP PARALLEL DO PRIVATE(i) SHARED(YgradXYZest,indicesForEst) SCHEDULE(STATIC)
    do i=1, NestIndices
      YgradXYZest(:,:,indicesForEst(i)) = YgradXYZest_KRR(indicesForEst(i), alpha(:,1))
    end do
    !$OMP END PARALLEL DO
  end if
  
end subroutine calcEst_KRR
!==============================================================================

subroutine calcIntConst()
  use dataset, only : Ntrain, indicesTrain
  use A_KRR_kernel,  only : Yest_KRR
  implicit none 
  integer :: i

  if (option%prior=='meandevi') then
    Yprior = 0.0_rprec
    do i=1, Ntrain
      Yprior = Yprior + Yest_KRR(indicesTrain(i), alpha(:,1))
    end do
    Yprior = Int_const - Yprior/Ntrain 
  end if

end subroutine calcIntConst

!==============================================================================
recursive subroutine optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
!==============================================================================
  use A_KRR_kernel,  only : K, calcKernel, Kvalidate
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use dataset,       only : Ntrain, indicesTrain, splitAndSample
  use dataset,       only : NsubtrainGradXYZ, indicesSubtrainGradXYZ
  use dataset,       only : NvalidateGradXYZ, indicesValidateGradXYZ
  use dataset,       only : Yest, YestCVopt, CVoptFolds, optData
  use dataset,       only : YgradXYZest, YgradXYZestCVopt
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  implicit none
  ! Arguments
  type(optData), intent(inout), target :: hyperOptDataPt ! Information necessary to hyperparmeter optimization
  type(arrayOfArrays), intent(in) :: indicesCVoptSplitsPt(:)  ! Indices of CV splits used for optimization of hyperparameters
  ! Local variables
  integer, save    :: Noptimized = 0
  integer          :: i, mm, nn, nnsize, stsize, newsize, Error
  real(kind=rprec) :: DlgLambda
  real(kind=rprec) :: ERR_temp, lgLambda_temp
  real(kind=rprec) :: ERR_min, lgLambda_min, lambda_min

  if (lgLambdaPoints == 1) then
    DlgLambda = 0
  else
    DlgLambda = (lgLambdaH - lgLambdaL) / dble(lgLambdaPoints - 1)
  end if

  do i=1,lgLambdaPoints
    lgLambda_temp = lgLambdaL + DlgLambda * (i - 1)
    lambda = 2 ** lgLambda_temp
    
    if (option%CVopt) then
      if (allocated(Yest))        Yest        = 0.0_rprec
      if (allocated(YgradXYZest)) YgradXYZest = 0.0_rprec
      if (option%onTheFly) then
        do mm = 1, size(indicesCVoptSplitsPt)
          ! Get the sub-training and validation sets for the fold mm
          if(allocated(hyperOptDataPt%indicesSubtrain)) deallocate(hyperOptDataPt%indicesSubtrain)
          if(allocated(hyperOptDataPt%indicesValidate)) deallocate(hyperOptDataPt%indicesValidate)
          Nsubtrain => hyperOptDataPt%Nsubtrain
          Nvalidate => hyperOptDataPt%Nvalidate
          Nvalidate = size(indicesCVoptSplitsPt(mm)%oneDintArr)
          Nsubtrain = Ntrain - Nvalidate
          allocate(hyperOptDataPt%indicesValidate(1:Nvalidate),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate indicesValidate')
          allocate(hyperOptDataPt%indicesSubtrain(1:Nsubtrain),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate indicesSubtrain')
          indicesSubtrain => hyperOptDataPt%indicesSubtrain
          indicesValidate => hyperOptDataPt%indicesValidate
          indicesValidate(1:Nvalidate) = indicesCVoptSplitsPt(mm)%oneDintArr(1:Nvalidate)
          indicesSubtrain = 0
          stsize          = 0
          newsize         = 0
          do nn = 1, size(indicesCVoptSplitsPt)
            if (nn == mm) cycle
            nnsize = size(indicesCVoptSplitsPt(nn)%oneDintArr)
            newsize = stsize + nnsize
            indicesSubtrain(stsize+1:newsize) = indicesCVoptSplitsPt(nn)%oneDintArr(1:nnsize)
            stsize = newsize
          end do
          call cleanUp_KRR()
          call getYtrain(Nsubtrain, indicesSubtrain, Nprops)  
          allocate(K(1),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate K')
          allocate(K(1)%twoDrArr(1:NtrTot,1:NtrTot),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate K(1)%twoDrArr')
          ! Train, i.e. calculate regression coefficients alpha
          call calcKernel(K(1)%twoDrArr)
          call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
          ! Predict values using ML model
          call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                            indicesValidate, doNotCalcStats=.true.)          
        end do
      else
        do mm = 1, size(indicesCVoptSplitsPt)
          ! Train, i.e. calculate regression coefficients alpha
          call getYtrain(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, Nprops)
          call calcAlpha(Nprops, K(mm)%twoDrArr, Ytrain)
          ! Predict values using ML model
          call validate_KRR(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, CVoptFolds(mm)%Nvalidate, &
                              CVoptFolds(mm)%indicesValidate, &
                              KvalidateMatrix=Kvalidate(mm)%oneDrArr, doNotCalcStats=.true.)
        end do
      end if
      ! Get error for the entire training set combined from size(indicesCVoptSplitsPt) validation sets
      if (allocated(Yest))        Yest        = YestCVopt
      if (allocated(YgradXYZest)) YgradXYZest = YgradXYZestCVopt
      call calc_stat_measures(Ntrain, indicesTrain, returnValue=ERR_temp)

    else
      ! Train, i.e. calculate regression coefficients alpha
      call getYtrain(Nsubtrain, indicesSubtrain, Nprops) ! Necessary to reset values if they are modified by prior
      call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
      ! Predict values using ML model
      if (option%onTheFly) then
        call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                          indicesValidate, validationError=ERR_temp)
      else
          call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                            indicesValidate, &
                            KvalidateMatrix=Kvalidate(1)%oneDrArr, validationError=ERR_temp)
      end if
    end if
    
    if (i ==1) then
      lambda_min   = lambda
      lgLambda_min = lgLambda_temp
      ERR_min      = ERR_temp
    elseif (ERR_temp < ERR_min) then
      lambda_min   = lambda
      lgLambda_min = lgLambda_temp
      ERR_min      = ERR_temp
    end if
    if(option%debug) write(6,'(" ERR(lambda of ",F25.13,") = ",F25.13)') lambda, ERR_temp
  end do
  
  Noptimized = Noptimized + 1

  lambda = lambda_min
  ! Recalculate regression coefficients alpha
  if (.not. option%CVopt) then
    call getYtrain(Nsubtrain, indicesSubtrain, Nprops) ! Necessary to reset values if they are modified by prior
    call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
  end if

  if(option%debug) write(6,'(" ERR(lambda of ",F25.13,") = ",F25.13)') lambda, ERR_min

  ! Make more dense grid around optimal value
  if (Noptimized < lgOptDepth) then
    lgLambdaL = lgLambda_min-DlgLambda
    lgLambdaH = lgLambda_min+DlgLambda
    call optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
  else ! Reset flag
    Noptimized = 0
  end if

end subroutine optLambdaLog2Grid
!==============================================================================

!==============================================================================
recursive subroutine optLog2Grid(hyperparameter, hyperOptDataPt, indicesCVoptSplitsPt)
!==============================================================================
  use A_KRR_kernel,  only : sigma, lgSigmaL, lgSigmaH, lgSigmaPoints
  use A_KRR_kernel,  only : c,     lgCL,     lgCH,     lgCPoints
  use A_KRR_kernel,  only : K, Kvalidate, calcKernel, calc_Kvalidate
  use dataset,       only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use dataset,       only : Ntrain, indicesTrain, splitAndSample
  use dataset,       only : NsubtrainGradXYZ, indicesSubtrainGradXYZ
  use dataset,       only : NvalidateGradXYZ, indicesValidateGradXYZ
  use dataset,       only : Yest, YestCVopt, CVoptFolds, optData
  use dataset,       only : YgradXYZest, YgradXYZestCVopt
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  implicit none
  ! Argument
  character(len=*), intent(in) :: hyperparameter
  type(optData), intent(inout), target :: hyperOptDataPt ! Information necessary to hyperparmeter optimization
  type(arrayOfArrays), intent(in) :: indicesCVoptSplitsPt(:)  ! Indices of CV splits used for optimization of hyperparameters
  ! Local variables
  integer, save    :: Noptimized = 0
  integer          :: i, mm, nn, nnsize, stsize, newsize, Error
  integer, pointer :: lgParamPoints
  real(kind=rprec), pointer :: Param, lgParamL, lgParamH
  real(kind=rprec) :: DlgParam
  real(kind=rprec) :: ERR_temp, lgParam_temp
  real(kind=rprec) :: ERR_min, lgParam_min, Param_min, lambda_min

  select case (hyperparameter)
    case ('sigma' )
    Param         => sigma
    lgParamL      => lgSigmaL
    lgParamH      => lgSigmaH
    lgParamPoints => lgSigmaPoints
    case ('c'     )
    Param         => c
    lgParamL      => lgCl
    lgParamH      => lgCh
    lgParamPoints => lgCPoints
  end select

  if (lgParamPoints == 1) then
    DlgParam = 0
  else
    DlgParam = (lgParamH - lgParamL) / dble(lgParamPoints - 1)
  end if

  do i=1,lgParamPoints
    lgParam_temp = lgParamL + DlgParam * (i - 1)
    Param = 2 ** lgParam_temp

    if (option%CVopt) then
      if (allocated(Yest)) Yest = 0.0_rprec
      if (allocated(YgradXYZest)) YgradXYZest = 0.0_rprec
      if (option%onTheFly) then
        if (option%lambda == 'OPT') then
          call getLambda() ! This call is necessary to reset lambda value to the default value
          call optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
        end if
        do mm = 1, size(indicesCVoptSplitsPt)
          ! Get the sub-training and validation sets for the fold mm
          if(allocated(hyperOptDataPt%indicesSubtrain)) deallocate(hyperOptDataPt%indicesSubtrain)
          if(allocated(hyperOptDataPt%indicesValidate)) deallocate(hyperOptDataPt%indicesValidate)
          Nsubtrain => hyperOptDataPt%Nsubtrain
          Nvalidate => hyperOptDataPt%Nvalidate
          Nvalidate    = size(indicesCVoptSplitsPt(mm)%oneDintArr)
          Nsubtrain    = Ntrain - Nvalidate
          allocate(hyperOptDataPt%indicesValidate(1:Nvalidate),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate indicesValidate')
          allocate(hyperOptDataPt%indicesSubtrain(1:Nsubtrain),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate indicesSubtrain')
          indicesSubtrain => hyperOptDataPt%indicesSubtrain
          indicesValidate => hyperOptDataPt%indicesValidate
          indicesValidate(1:Nvalidate) = indicesCVoptSplitsPt(mm)%oneDintArr(1:Nvalidate)
          indicesSubtrain = 0
          stsize          = 0
          newsize         = 0
          do nn = 1, size(indicesCVoptSplitsPt)
            if (nn == mm) cycle
            nnsize = size(indicesCVoptSplitsPt(nn)%oneDintArr)
            newsize = stsize + nnsize
            indicesSubtrain(stsize+1:newsize) = indicesCVoptSplitsPt(nn)%oneDintArr(1:nnsize)
            stsize = newsize
          end do
          call cleanUp_KRR()
          call getYtrain(Nsubtrain, indicesSubtrain, Nprops)
          allocate(K(1),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate K')
          allocate(K(1)%twoDrArr(1:NtrTot,1:NtrTot),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate K(1)%twoDrArr')
          ! Train, i.e. calculate regression coefficients alpha
          call calcKernel(K(1)%twoDrArr) 
          call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
          ! Predict values using ML model
          call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                            indicesValidate, doNotCalcStats=.true.)          
        end do
      else
        do mm = 1, size(indicesCVoptSplitsPt)
          call getYtrain(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, Nprops)
          call calcKernel(K(mm)%twoDrArr)
          call calc_Kvalidate(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, CVoptFolds(mm)%Nvalidate, &
                                CVoptFolds(mm)%indicesValidate, Kvalidate(mm)%oneDrArr)
        end do
        if (option%lambda == 'OPT') then
          call getLambda() ! This call is necessary to reset lambda value to the default value
          call optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
        end if
        do mm = 1, size(indicesCVoptSplitsPt)
          call getYtrain(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, Nprops)
          call calcAlpha(Nprops, K(mm)%twoDrArr, Ytrain)
          call validate_KRR(CVoptFolds(mm)%Nsubtrain, CVoptFolds(mm)%indicesSubtrain, CVoptFolds(mm)%Nvalidate, &
                              CVoptFolds(mm)%indicesValidate, &
                              KvalidateMatrix=Kvalidate(mm)%oneDrArr, doNotCalcStats=.true.)
        end do
      end if
      ! Get error for the entire training set combined from size(indicesCVoptSplitsPt) validation sets
      if (allocated(Yest))        Yest        = YestCVopt
      if (allocated(YgradXYZest)) YgradXYZest = YgradXYZestCVopt
      call calc_stat_measures(Ntrain, indicesTrain, returnValue=ERR_temp)

    else
      ! Calculate kernel matrix
      call getYtrain(Nsubtrain, indicesSubtrain, Nprops) ! Necessary to reset values if they are modified by prior
      call calcKernel(K(1)%twoDrArr)
      if (.not. option%onTheFly) then 
        call calc_Kvalidate(Nsubtrain, indicesSubtrain, &
                                                      Nvalidate, indicesValidate, &
                                                      Kvalidate(1)%oneDrArr)
      end if 
      ! Get optimal lambda
      if (option%lambda == 'OPT') then
        call getLambda() ! This call is necessary to reset lambda value to the default value
        call optLambdaLog2Grid(hyperOptDataPt, indicesCVoptSplitsPt)
      else
        ! Calculate regression coefficients alpha
        call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
      end if
      ! Predict values using ML model
      if (option%onTheFly) then
        call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                          indicesValidate, validationError=ERR_temp)
      else
        call validate_KRR(Nsubtrain, indicesSubtrain, Nvalidate, &
                            indicesValidate, &
                            KvalidateMatrix=Kvalidate(1)%oneDrArr, validationError=ERR_temp)
      end if
    end if

    if (i ==1) then
      lambda_min  = lambda
      lgParam_min = lgParam_temp
      Param_min   = Param
      ERR_min     = ERR_temp
    elseif (ERR_temp < ERR_min) then
      lambda_min  = lambda
      lgParam_min = lgParam_temp
      Param_min   = Param
      ERR_min     = ERR_temp
    end if
    if(option%debug) write(6,'(" ERR(",a, " of ",F25.13,") = ",F25.13)') hyperparameter, Param, ERR_temp
  end do
  
  Noptimized = Noptimized + 1
  
  lambda = lambda_min
  Param  = Param_min
  ! Recalculate regression coefficients alpha
  if (.not. option%CVopt) then
    call getYtrain(Nsubtrain, indicesSubtrain, Nprops) ! Necessary to reset values if they are modified by prior
    call calcKernel(K(1)%twoDrArr)
    if (.not. option%onTheFly) call calc_Kvalidate(Nsubtrain, &
      indicesSubtrain, Nvalidate, indicesValidate, Kvalidate(1)%oneDrArr)
    call calcAlpha(Nprops, K(1)%twoDrArr, Ytrain)
  end if
  
  if(option%debug) write(6,'(" ERR(",a, " of ",F25.13,") = ",F25.13)') hyperparameter, Param, ERR_min

  ! Make more dense grid around optimal value
  if (Noptimized < lgOptDepth) then
    lgParamL = lgParam_min-DlgParam
    lgParamH = lgParam_min+DlgParam
    call optLog2Grid(hyperparameter, hyperOptDataPt, indicesCVoptSplitsPt)
  else ! Reset flag
    Noptimized = 0
  end if

end subroutine optLog2Grid
!==============================================================================

!==============================================================================
subroutine selectPermutation_KRR()
  !==============================================================================
    implicit none 
    call mindRMSD_selectPermutation()  
  
end subroutine selectPermutation_KRR
!==============================================================================

!==============================================================================
subroutine mindRMSD_selectPermutation()
  use molDescr, only : dRMSD, permuteAtoms 
  use dataset,  only : XYZeq
  use dataset,  only : NitemsTot, NAtomsMax, Nperm
  use dataset,  only : Ntrain, indicesTrain
  use dataset,  only : XYZ_A, Z
  use dataset,  only : permutedIatoms
  use dataset,  only : minpermCount
  use A_KRR_kernel, only : indicesSelectPerm, NselectPerm
  ! Local variables
  integer  :: Nspecies 
  integer  :: iperm, miniperm, Natoms2perm
  real(kind=rprec) :: mindRMSD, dRMSDperm
  integer, allocatable :: permutedZ(:)
  real(kind=rprec), allocatable :: permutedXYZ(:,:) 
  integer :: Error
  integer :: count, ii
  integer, allocatable :: indicesSelectPermLoc(:)
  
  if(.not. allocated(minpermCount)) allocate(minpermCount(Nperm),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for minpermCount array')
  allocate(indicesSelectPermLoc(Nperm),stat=Error) 
  if(Error/=0) call stopMLatomF('Unable to allocate space for indicesSelectPermLoc')
  allocate(permutedZ(1:NAtomsMax),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for permutedZ array')
  allocate(permutedXYZ(1:3,1:NAtomsMax),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for permutedXYZ array')
  
  
  ! Initialize 
  minpermCount = 0

  do ii=1,Ntrain
    Nspecies = indicesTrain(ii)
    Natoms2perm = size(permutedIatoms(1)%oneDintArr)
    call permuteAtoms(Z(:,Nspecies),XYZ_A(:,:,Nspecies),Natoms2perm,permutedIatoms(1)%oneDintArr,permutedZ,permutedXYZ)
    mindRMSD = dRMSD(XYZeq,permutedXYZ)
    miniperm = 1 
    do iperm=2,Nperm 
      call permuteAtoms(Z(:,Nspecies),XYZ_A(:,:,Nspecies),Natoms2perm,permutedIatoms(iperm)%oneDintArr,permutedZ,permutedXYZ)
      dRMSDperm = dRMSD(XYZeq,permutedXYZ)
      if(dRMSDperm < mindRMSD) then 
        miniperm = iperm 
        mindRMSD = dRMSDperm
      end if 
    end do 
    minpermCount(miniperm) = minpermCount(miniperm) + 1
  end do 
  



  !do Nspecies=1,NitemsTot 
  !  Natoms2perm = size(permutedIatoms(1)%oneDintArr)
  !  call permuteAtoms(Z(:,Nspecies),XYZ_A(:,:,Nspecies),Natoms2perm,permutedIatoms(1)%oneDintArr,permutedZ,permutedXYZ)
  !  mindRMSD = dRMSD(XYZeq,permutedXYZ)
  !  miniperm = 1 
  !  do iperm=2,Nperm 
  !    call permuteAtoms(Z(:,Nspecies),XYZ_A(:,:,Nspecies),Natoms2perm,permutedIatoms(iperm)%oneDintArr,permutedZ,permutedXYZ)
  !    dRMSDperm = dRMSD(XYZeq,permutedXYZ)
  !    if(dRMSDperm < mindRMSD) then 
  !      miniperm = iperm 
  !      mindRMSD = dRMSDperm
  !    end if 
  !  end do 
  !  minpermCount(miniperm) = minpermCount(miniperm) + 1
  !end do 

  write(*,*) 'minpermCount = ',minpermCount

  count = 0
  do iperm=1, Nperm 
    if (minpermCount(iperm) > 0) then 
      count = count + 1 
      indicesSelectPermLoc(count) = iperm 
    end if 
  end do 
  
  allocate(indicesSelectPerm(count),stat=Error) 
  if(Error/=0) call stopMLatomF('Unable to allocate space for indicesSelectPerm')
  indicesSelectPerm = indicesSelectPermLoc(1:count)
  NselectPerm = count
  do ii=1,NselectPerm
    write(6,*) 'Selected permutation: '
    write(6,*) permutedIatoms(indicesSelectPerm(ii))%oneDintArr
  end do
  write(6,*) 'Indices of selected permutations = ', indicesSelectPerm
  
  call rearrangePermutations()
  
end subroutine mindRMSD_selectPermutation
!==============================================================================

!==============================================================================
! Change X, Nperm, XvecSize, permutedIatoms 
! according to the permutations selected
!==============================================================================
subroutine rearrangePermutations()
  use dataset,      only : X 
  use dataset,      only : permutedIatoms
  use dataset,      only : NitemsTot, XvecSize, Nperm, NAtomsMax
  use A_KRR_kernel, only : NselectPerm, indicesSelectPerm
  implicit none 
  ! Local variable
  integer :: Xsize 
  real(kind=rprec), allocatable :: X_loc(:,:)
  type(arrayOfArrays), allocatable :: permutedIatoms_loc(:)
  integer :: Error
  integer :: PP, imol

  Xsize = XvecSize / Nperm 

  ! Save the old arrays
  allocate(X_loc(1:XvecSize,1:NitemsTot),stat=Error)
  if (Error/=0) call stopMLatomF('Unable to allocate space for X_loc')
  allocate(permutedIatoms_loc(1:Nperm),stat=Error)
  if (Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms_loc')
  do PP=1,Nperm
    allocate(permutedIatoms_loc(PP)%oneDIntArr(1:NAtomsMax),stat=Error)
    if (Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms_loc(PP)%oneDIntArr')
  end do 

  X_loc = X 
  do PP=1,Nperm 
    permutedIatoms_loc(PP)%oneDIntArr = permutedIatoms(PP)%oneDIntArr 
  end do 

  ! Deallocate old arrays 
  deallocate(X)
  do PP=1,Nperm 
    deallocate(permutedIatoms(PP)%oneDIntArr)
  end do 
  deallocate(permutedIatoms)

  ! Allocate new arrays
  allocate(X(1:NselectPerm*Xsize,1:NitemsTot),stat=Error)
  if (Error/=0) call stopMLatomF('Unable to allocate space for X')
  allocate(permutedIatoms(1:NselectPerm),stat=Error)
  if (Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms')
  do PP=1, NselectPerm
    allocate(permutedIatoms(PP)%oneDIntArr(1:NAtomsMax),stat=Error)
    if (Error/=0) call stopMLatomF('Unable to allocate space for permutedIatoms(PP)%oneDIntArr')
  end do 
  
  ! Update new arrays
  do imol=1, NitemsTot
    do PP=1, NselectPerm 
      X((PP-1)*Xsize+1:PP*Xsize,imol) = X_loc((indicesSelectPerm(PP)-1)*Xsize+1:indicesSelectPerm(PP)*Xsize,imol)
    end do 
  end do 
  do PP=1, NselectPerm
    permutedIatoms(PP)%oneDIntArr = permutedIatoms_loc(indicesSelectPerm(PP))%oneDIntArr
  end do 
  Nperm = NselectPerm
  XvecSize = Xsize * NselectPerm


end subroutine rearrangePermutations
!==============================================================================
  
!==============================================================================
subroutine cleanUp_KRR()
!==============================================================================
  use A_KRR_kernel,  only : K, Kvalidate
  implicit none
  integer :: i, j

  ! Free up memory
  if (allocated(K)) then
    do i = 1, size(K)
      if(allocated(K(i)%twoDrArr)) deallocate(K(i)%twoDrArr)
    end do
    deallocate(K)
  end if
  if (allocated(Kvalidate)) then
    do i = 1, size(Kvalidate)
      if(allocated(Kvalidate(i)%oneDrArr)) deallocate(Kvalidate(i)%oneDrArr)
    end do
    deallocate(Kvalidate)
  end if
  if (allocated(alpha))    deallocate(alpha)
  if (allocated(Ytrain))   deallocate(Ytrain)
  if (allocated(itrval))   deallocate(itrval)
  if (allocated(itrgrxyz)) deallocate(itrgrxyz)

end subroutine cleanUp_KRR
!==============================================================================

end module A_KRR
