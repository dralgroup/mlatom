
  !---------------------------------------------------------------------------! 
  ! MLmodel: dealing with ML models                                           ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module MLmodel
  use A_KRR,         only : alpha, NtrTot, NtrVal, itrval
  use dataset,       only : Yprior
  use optionsModule, only : option
  use precision,     only : rprec
  implicit none
  integer                       :: Ntrain_MLmodel      ! Number of training molecules
  integer                       :: XvecSize_MLmodel    ! Length of X's vector
  integer                       :: NAtomsMax_MLmodel, NgradXYZmax_MLmodel
  real(kind=rprec), allocatable :: X_MLmodel(:,:)      ! Data set
  integer,          allocatable :: Z_MLmodel(:,:)      ! Nuclear charges
  integer,          allocatable :: Natoms_MLmodel(:)   ! Number of atoms in molecules 
  integer,          allocatable :: NgradXYZ_MLmodel(:)   ! Number of XYZ gradients in molecules 
  real(kind=rprec), allocatable :: XYZ_A_MLmodel(:,:,:)! Atomic coordinates in Angstrom
  character(len=6)              :: MLmodelFileVersion = '1.0'

contains

!==============================================================================
subroutine useMLmodel()
!==============================================================================
! Use the ML model
!==============================================================================
  use A_KRR,    only : calcEst_KRR
  use dataset,  only : writeYgradXYZest, YgradXYZest, NAtomsMax, NgradXYZmax
  use dataset,  only : NitemsTot, writeEst
  use dataset,  only : readXYZCoords, readX, XvecSize, Yest, YestGrad
  use dataset,  only : Npredict, indicesPredict
  use molDescr, only : getMolDescr
  use stopper,  only : stopMLatomF
  implicit none
  integer :: error

  ! Load ML model
  call readMLmodel()
  
  ! Read in or calculate X
  call getX()
  
  ! Merge sets for prediction and loaded ML model
  call mergeMLmodelAndX()
  
  if (option%YestFile /= '') then
    ! Allocate arrays
    allocate(Yest(1:Npredict),stat=error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector Yest with estimated values') 
    option%calcVal = .true.
  end if
  if (option%YgradEstFile /= '') then
    ! Allocate arrays
    allocate(YestGrad(1:XvecSize,1:Npredict),stat=error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YestGrad')
    option%calcGrad = .true.
  end if
  if (option%YgradXYZestFile /= '') then
    ! Allocate arrays
    allocate(YgradXYZest(1:3,1:NatomsMax,1:Npredict),stat=error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YgradXYZest')
    option%calcGradXYZ = .true.
  end if
  
  call calcEst_KRR(Npredict, indicesPredict)
  call writeEst(Npredict, indicesPredict)
  
  write (6,'(a)') ' ESTIMATED VALUES SAVED'
  write (6,'(a)') ''

end subroutine useMLmodel
!==============================================================================

!==============================================================================
subroutine createMLmodel()
!==============================================================================
! Create an ML model
!==============================================================================
  use A_KRR,        only : init_KRR, train_KRR, optHyper_KRR, test_KRR
  use A_KRR,        only : calcEst_KRR, cleanUp_KRR
  use A_KRR,        only : selectPermutation_KRR
  use dataset,      only : NitemsTot, indicesItemsTot, Ntrain, indicesTrain
  use dataset,      only : Ntest, indicesTest, trainTestData
  use dataset,      only : readY, readYgrad, readYgradXYZ, NgradXYZmax, YgradXYZest
  use dataset,      only : Yest, YestCVtest, getSubtrainValidate
  use dataset,      only : writeEst, YgradXYZestCVtest
  use dataset,      only : splitAndSample, indicesCVtestSplits
  use dataset,      only : CVtestFolds, hyperOptData, indicesCVoptSplits
  use MLstatistics, only : calc_stat_measures
  use sampling,     only : sample
  use stopper,      only : stopMLatomF
  implicit none
  integer :: i, mm
  character(len=256) :: stmp  ! Temporary variable

  ! Read in or calculate X
  call getX()

  ! Read in reference values dataset
  if (option%Yfile /= '')        call readY(option%Yfile)
  if (option%YgradFile /= '')    call readYgrad(option%YgradFile)
  if (option%YgradXYZfile /= '') call readYgradXYZ(option%YgradXYZfile)

  ! Split data set into the necessary subsets, e.g. training, test sets, etc.
  call sample()
  
  ! Get hyperparameters for KRR
  if (option%algorithm == 'KRR') then
    call init_KRR()
    if (option%selectPerm) then 
      call selectPermutation_KRR() 
    end if 
    if (option%hyperOpt) call optHyper_KRR(hyperOptData, indicesCVoptSplits)
  end if

  write (6,'(a/)') ' CREATE AND SAVE FINAL ML MODEL'
  if (option%algorithm == 'KRR') call train_KRR()
  call writeMLmodel()
  write (6,'(a/)') ' FINAL ML MODEL CREATED AND SAVED'
    
  ! Calculate and save estimated Yest values if requested
  if (option%writeEst) then
    call calcEst_KRR(NitemsTot, indicesItemsTot)
    call writeEst(NitemsTot, indicesItemsTot)
  end if  
  if (option%algorithm == 'KRR') call cleanUp_KRR()

end subroutine createMLmodel
!==============================================================================

!==============================================================================
subroutine estAccMLmodel()
!==============================================================================
! Create an ML model
!==============================================================================
  use A_KRR,        only : init_KRR, train_KRR, optHyper_KRR, test_KRR
  use A_KRR,        only : calcEst_KRR, cleanUp_KRR
  use A_KRR,        only : selectPermutation_KRR
  use dataset,      only : NitemsTot, indicesItemsTot, Ntrain, indicesTrain
  use dataset,      only : Ntest, indicesTest, trainTestData
  use dataset,      only : Nsubtrain, indicesSubtrain, Nvalidate, indicesValidate
  use dataset,      only : readY, readYgradXYZ, readYgrad, NgradXYZmax, YgradXYZest
  use dataset,      only : Yest, YestCVtest, getSubtrainValidate
  use dataset,      only : writeEst, YgradXYZestCVtest
  use dataset,      only : splitAndSample, indicesCVtestSplits
  use dataset,      only : CVtestFolds, hyperOptData, indicesCVoptSplits
  use dataset,      only : Npredict, indicesPredict
  use MLstatistics, only : calc_stat_measures
  use sampling,     only : sample
  use stopper,      only : stopMLatomF
  implicit none
  integer :: i, mm
  character(len=256) :: stmp  ! Temporary variable

  ! ---------------------------------------------------------------------------
  if (option%MLmodelFileIn /= '') then
    ! Load ML model
    call readMLmodel()  
    ! Read in or calculate X
    call getX()
    ! Read in reference values dataset
    if (option%Yfile /= '')        call readY(option%Yfile)
    if (option%YgradFile /= '')    call readYgrad(option%YgradFile)
    if (option%YgradXYZfile /= '') call readYgradXYZ(option%YgradXYZfile)
    ! Split data set into the necessary subsets, e.g. training, test sets, etc.
    call sample()
    ! Merge sets for prediction and loaded ML model
    call mergeMLmodelAndX()

    if (option%Yfile /= '') then
      write(6,'(a/)') ' Analysis for Y values with reference data from file ' // trim(option%Yfile)
      option%calcVal = .true.
    end if
    if (option%YgradFile /= '') then
      write(6,'(a/)') ' Analysis for Y gradients with reference data from file ' // trim(option%YgradFile)
      option%calcGrad = .true.
    end if
    if (option%YgradXYZfile /= '') then
        write(6,'(a/)') ' Analysis for XYZ gradients of Y with reference data from file ' // trim(option%YgradXYZfile)
        write(6,'(a/)') ' ... gradients are calculated as derivatives of ML model for Y'
        option%calcGradXYZ = .true.
    end if
    
    ! Calculate estimated values
    if (option%writeEst) call calcEst_KRR(Npredict, indicesPredict)
    if (Ntrain > 0) then
      if (.not. option%writeEst) call calcEst_KRR(Ntrain, indicesTrain)
      write(stmp,'(I0," entries in the training set")') Ntrain
      call calc_stat_measures(Ntrain, indicesTrain, comment=trim(stmp))
      write(6,'(a)') ''
    end if
    if (Ntest > 0) then
      if (.not. option%writeEst) call calcEst_KRR(Ntest, indicesTest)
      write(stmp,'(I0," entries in the test set")') Ntest
      call calc_stat_measures(Ntest, indicesTest, comment=trim(stmp))
      write(6,'(a)') ''  
    end if

    NitemsTot       = Npredict
    indicesItemsTot = indicesPredict
  ! ---------------------------------------------------------------------------
  else
  ! ---------------------------------------------------------------------------
  
    ! Read in or calculate X
    call getX()

    ! Read in reference values dataset
    if (option%Yfile /= '')        call readY(option%Yfile)
    if (option%YgradFile /= '')    call readYgrad(option%YgradFile)
    if (option%YgradXYZfile /= '') call readYgradXYZ(option%YgradXYZfile)

    ! Split data set into the necessary subsets, e.g. training, test sets, etc.
    call sample()
  
    ! Get hyperparameters for KRR
    if (option%algorithm == 'KRR') call init_KRR()
    
    ! Select permutation (using mindRMSD)
    if (option%selectPerm) then
      call selectPermutation_KRR()
    end if

    ! Estimate the predictin error using cross-validation
    if (option%CVtest) then
      do mm = 1, option%NcvTestFolds
        Ntrain       => CVtestFolds(mm)%Ntrain
        Ntest        => CVtestFolds(mm)%Ntest
        indicesTrain => CVtestFolds(mm)%indicesTrain
        indicesTest  => CVtestFolds(mm)%indicesTest
        if (option%Yfile /= '')    Yest        =  0.0_rprec
        if (option%YgradXYZfile /= '') YgradXYZest = 0.0_rprec
        call cleanUp_KRR()      
        
        ! Optimize hyperparameters
        if (option%hyperOpt) then
          ! Optimize hyperparameters
          if (option%algorithm == 'KRR') then
            call init_KRR() ! This call is necessary to reset all initial values of hyperparameters to the default values
            call optHyper_KRR(CVtestFolds(mm)%hyperOptData, CVtestFolds(mm)%indicesCVoptSplits, doNotCalcStats=.true.)
          end if
        end if

        ! Train and test
        if (option%algorithm == 'KRR') call train_KRR(doNotCalcStats=.true.)
        if (option%algorithm == 'KRR') call test_KRR( doNotCalcStats=.true.)
        
      end do
      
      ! Get error for the entire set combined from option%NcvTestFolds test sets
      if (option%Yfile /= '') then
        Yest = YestCVtest
        write(stmp,'(I0," entries in the combined cross-validation test set")') NitemsTot
      end if
      if (option%YgradXYZfile /= '') then
        YgradXYZest = YgradXYZestCVtest
        write(stmp,'(I0," entries in the combined cross-validation test set")') NitemsTot*NgradXYZmax*3
      end if
      call calc_stat_measures(NitemsTot, indicesItemsTot, comment=trim(stmp))
      write(6,'(a)') ''

    else

      ! Optimize hyperparameters
      if (option%hyperOpt) then
        if (option%algorithm == 'KRR') call optHyper_KRR(hyperOptData, indicesCVoptSplits)
      end if

      ! Train and test
      if (option%algorithm == 'KRR') call train_KRR()
      if (option%algorithm == 'KRR') call test_KRR()
      
      ! Calculate estimated Yest values
      if (option%writeEst) then
        call calcEst_KRR(NitemsTot, indicesItemsTot)
      end if

      ! Save the ML model if requested
      if (option%MLmodelFileOut /= '') then
        call writeMLmodel()
        write (6,'(a/)') ' ML MODEL SAVED'
      end if
      
    end if ! If for cross-validation 
    
    if (option%algorithm == 'KRR') then
      call cleanUp_KRR()
    end if
    
  end if ! If for loading ML model
  ! ---------------------------------------------------------------------------
  
  ! Write estimated Yest values
  if (option%writeEst) call writeEst(NitemsTot, indicesItemsTot)

end subroutine estAccMLmodel
!==============================================================================

!==============================================================================
subroutine getX()
!==============================================================================
! Read and/or calculate X vectors
!==============================================================================
  use dataset,  only : readXYZCoords, readX, writeX
  use dataset,  only : writeXYZsorted, getPermInvData
  use molDescr, only : getMolDescr
  implicit none
  integer :: i, Error
  
  ! Check whether permutational invariance was requested by the user
  if (trim(option%permInvGroups) /= '' .or. trim(option%permInvNuclei) /= '' &
      .or. trim(option%Nperm) /= '' .or. option%usePermInd) call getPermInvData()
  
  ! Read in X values or calculate them from a molecular descriptor
  if (option%readX) then
    call readX(option%XfileIn)
  else    
    ! Read in coordinates
    call readXYZCoords(option%XYZfile)
    ! Construct molecular descriptor
    call getMolDescr()
  end if
  
  ! Write file with X values
  if (option%writeX) then
    call writeX(option%XfileOut)
  end if

  ! Write file with sorted XYZ coordinates
  if (option%writeXYZsorted) then
    call writeXYZsorted(option%XYZsortedFileOut)
  end if

end subroutine getX
!==============================================================================

!==============================================================================
subroutine mergeMLmodelAndX()
!==============================================================================
! Merge loaded ML model with data for prediction
!==============================================================================
  use A_KRR_kernel, only : NtrTot, NtrVal, NtrGrXYZ, itrval, itrgrxyz
  use A_KRR_kernel, only : NselectPerm, indicesSelectPerm
  use dataset,      only : NitemsTot, Npredict, indicesPredict
  use dataset,      only : X, XvecSize
  use dataset,      only : XYZ_A, Z, NAtomsMax, Natoms
  use dataset,      only : NgradForAtom, gradForAtom, gradForXel
  use stopper,      only : stopMLatomF
  implicit none
  ! Local variables
  integer :: XvecSizePred, NAtomsMaxPred
  integer :: Nspecies, i, ip, ii, jj, icounter, PP
  integer :: Error
  ! Arrays
  real(kind=rprec), allocatable :: Xpred(:,:) ! Data set
  real(kind=rprec), allocatable :: XYZ_Apred(:,:,:), Zpred(:,:)
  integer,          allocatable :: NatomsPred(:)

  ! Get total number of points for prediction  
  Npredict = NitemsTot
  
  ! Allocate arrays
  allocate(Xpred(1:XvecSize,1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector with X values')
  if (NtrVal > 0) then
    allocate(itrval(1:NtrVal),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate itrval')
  end if
  if (NtrGrXYZ > 0) then
    allocate(itrgrxyz(1:NtrGrXYZ,3),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate itrgrxyz')
    allocate(NatomsPred(1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for NatomsPred')
    allocate(XYZ_Apred(1:3,1:NAtomsMax,1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ coordinates')
    allocate(Zpred(1:NAtomsMax,1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for nuclear charges')
    NatomsPred = Natoms
    XYZ_Apred = XYZ_A
    Zpred = Z
    NAtomsMaxPred = NAtomsMax
  end if
  allocate(indicesPredict(1:Npredict),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate indicesPredict')
  
  ! Save data for prediction
  XvecSizePred  = XvecSize
  Xpred         = X
  
  ! Extend X array
  NitemsTot = Ntrain_MLmodel + Npredict
  XvecSize  = max(XvecSize_MLmodel, XvecSizePred)
  deallocate(X)
  allocate(X(1:XvecSize,1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector with X values')
  if (NtrGrXYZ > 0) then
    NatomsMax = max(NAtomsMax_MLmodel,NAtomsMaxPred)
    deallocate(Natoms)
    allocate(Natoms(1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for Natoms')
    deallocate(XYZ_A)
    allocate(XYZ_A(1:3,1:NAtomsMax,1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ coordinates')
    deallocate(Z)
    allocate(Z(1:NAtomsMax,1:NitemsTot),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for nuclear charges')
    Natoms = 0
    Z = 0.0_rprec
    XYZ_A = 0.0_rprec
  end if
  
  ! Initialize with zeros all X values
  X = 0.0_rprec
  
  ! Merge X arrays from ML model and for prediction
  do Nspecies = 1, Npredict
    indicesPredict(Nspecies) = Nspecies
    do i = 1, XvecSizePred
      X(i,Nspecies) = Xpred(i,Nspecies)
    end do
  end do
  if (allocated(Zpred)) then
    do Nspecies = 1, Npredict
      Natoms(Nspecies) = NatomsPred(Nspecies)
      do i = 1, NAtomsMaxPred
        Z(i,Nspecies) = Zpred(i,Nspecies)
        XYZ_A(1:3,i,Nspecies) =  XYZ_Apred(1:3,i,Nspecies)
      end do
    end do
    deallocate(NatomsPred)
    deallocate(Zpred)
    deallocate(XYZ_Apred)
  end if
  do Nspecies = 1, Ntrain_MLmodel
    do i = 1, XvecSize_MLmodel
      X(i,Npredict + Nspecies) = X_MLmodel(i,Nspecies)
    end do
  end do
  if (allocated(Z_MLmodel)) then
    do Nspecies = 1, Ntrain_MLmodel
      Natoms(Npredict + Nspecies) = Natoms_MLmodel(Nspecies)
      do i = 1, NAtomsMax_MLmodel
        Z(i,Npredict + Nspecies) = Z_MLmodel(i,Nspecies)
        XYZ_A(1:3,i,Npredict + Nspecies) =  XYZ_A_MLmodel(1:3,i,Nspecies)
      end do
    end do
    deallocate(Natoms_MLmodel)
    deallocate(Z_MLmodel)
    deallocate(XYZ_A_MLmodel)
  end if
  if (NtrVal > 0) then

      do Nspecies = 1, Ntrain_MLmodel
        itrval(Nspecies) = Npredict + Nspecies
      end do
  end if
  if (NtrGrXYZ > 0) then
    icounter = 0
    do ip=1,Ntrain_MLmodel
        if (NgradXYZ_MLmodel(ip) /= 0) then
          do ii = 1, NgradXYZ_MLmodel(ip)
            do jj = 1, 3
              icounter = icounter + 1
              itrgrxyz(icounter,1)             = Npredict + ip
              itrgrxyz(icounter,2)             = ii
              itrgrxyz(icounter,3)             = jj
            end do
          end do
        end if
    end do
  end if
  
  if(allocated(Xpred))            deallocate(Xpred)
  if(allocated(X_MLmodel))        deallocate(X_MLmodel)
  if(allocated(NgradXYZ_MLmodel)) deallocate(NgradXYZ_MLmodel)

end subroutine mergeMLmodelAndX
!==============================================================================

!==============================================================================
subroutine readMLmodel()
!==============================================================================
! Read the ML model
!==============================================================================
  use A_KRR,         only : lambda, lambdaGradXYZ
  use A_KRR_kernel,  only : sigma,  nn, c, d, period, sigmap
  use A_KRR_kernel,  only : NtrTot, NtrVal, NtrGrXYZ, itrval, itrgrxyz
  use A_KRR_kernel,  only : NselectPerm, indicesSelectPerm
  use dataset,       only : Nperm, NgradXYZmax, NgradXYZ, XYZeq, Zeq, NAtomsMax
  use dataset,       only : permlen, modPermutedIatoms
  use molDescr,      only : readEqXYZFlag
  use stopper,       only : stopMLatomF
  implicit none
  ! Local variables
  integer :: MLmodelUnit
  integer :: Nspecies, i, j
  integer :: error
  integer :: eqxyzunit
  integer :: inttmp
  character(len=5) :: stmp
  character(len=10) :: descrtype
  character(len=10) :: kernel
  character(len=10) :: eqFileInfo
  real(kind=rprec)  :: dbtmp
  real(kind=rprec)  :: dbltmp(1:3)
  
  MLmodelUnit=51
  eqxyzunit = 25
  
  Ntrain_MLmodel = 0 ; NtrTot = 0 ; NtrVal = 0 ; NtrGrXYZ = 0
  
  open(MLmodelUnit,file=trim(option%MLmodelFileIn),access='stream',form='unformatted', &
       action='read',status='old',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to open file ' // trim(option%MLmodelFileIn))
  
  write (6,'(a)') ' ML model of the following type is read:'

  ! MLmodel file format
  read (MLmodelUnit) stmp
  read (MLmodelUnit) MLmodelFileVersion
  if ((trim(stmp) /= 'MLmod') .or. (trim(MLmodelFileVersion) /= '1.0')) then
    call stopMLatomF('MLatomF does not support format ' // trim(MLmodelFileVersion) // ' of file ' // trim(option%MLmodelFileIn))
  end if

  ! Molecular descriptor
  read (MLmodelUnit) descrtype
  if (descrtype == 'Xfpnumbers') then
    write(6,'(a)') '    User-defined X-values'
  elseif (descrtype == 'Xfpperminv') then
    option%permInvKernel = .true.
    read (MLmodelUnit) Nperm
    write(6,'(a)') '    User-defined X-values for permutationally invariant kernel'
	  write(6,'(a,i0)') '    Number of permutations: ', Nperm
  else 
    if (descrtype(3:3) == 'x') then 
      descrtype = descrtype(1:2) // 'p'// descrtype(4:10)
      option%readModPermInd = .true.
      option%usePermInd     = .true.
    end if 
    option%molDescriptor = descrtype(1:2)
    option%molDescrType  = descrtype(3:10)
    write(6,'(a)') '    Molecular descriptor:  ' // trim(option%molDescrType) // ' ' // trim(option%molDescriptor)
  endif
  if(.not. option%readX .and. trim(option%molDescrType) == 'permuted') then
    read (MLmodelUnit) option%permInvKernel
    if (option%permInvKernel) write(6,'(a)') '    Calculations with permutationally invariant kernel'
    read (MLmodelUnit) Nperm
    read (MLmodelUnit) option%permInvGroups
    read (MLmodelUnit) option%permInvNuclei
    write(6,'(a)') '    Number of permutations: ' // trim(option%Nperm)
    write(6,'(a)') '    Permutationally invariant groups: ' // trim(option%permInvGroups)
    write(6,'(a)') '    Permutationally invariant nuclei: ' // trim(option%permInvNuclei)
    if (option%readModPermInd) then 
      read(MLmodelUnit) permlen 
      allocate(modPermutedIatoms(Nperm),stat=error) 
      if (error /= 0) call stopMLatomF('Unable to allocate modPermutedIatoms')
      do i=1, Nperm 
        allocate(modPermutedIatoms(i)%oneDIntArr(permlen),stat=error)
        if (error /= 0) call stopMLatomF('Unable to allocate modPermutedIatoms(i)%oneDIntArr')
      end do 
      do Nspecies=1, Nperm 
        do i=1, permlen 
          read(MLmodelUnit) modPermutedIatoms(Nspecies)%oneDintArr(i)
        end do 
      end do 
      write(6,'(a)') '    Indices of permted atoms:'
      do Nspecies=1, Nperm  
        write(6,*) modPermutedIatoms(Nspecies)%oneDintArr 
      end do
    end if 
  end if

  ! Size of model
  read (MLmodelUnit) NtrVal
  write(6,'(a,I0,a)') '    Trained with:          ', NtrVal, ' entries'
  read (MLmodelUnit) XvecSize_MLmodel
  write(6,'(a,I0)') '    Max len. of X vectors: ', XvecSize_MLmodel
  allocate(X_MLmodel(1:XvecSize_MLmodel,1:NtrVal),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector with X values')
  do Nspecies = 1, NtrVal
    do i = 1, XvecSize_MLmodel
      read(MLmodelUnit) X_MLmodel(i,Nspecies)
    end do
  end do

  ! Algorithm specific data
  read (MLmodelUnit) option%algorithm
  if (option%algorithm == 'KRR') then
    option%prior = '0'
  elseif (option%algorithm == 'KRm') then
    option%algorithm = 'KRR'
    read (MLmodelUnit) option%prior
    write(6,'(a)') '    Prior type:           ' // trim(option%prior)
    read (MLmodelUnit) Yprior
    write(6,'(a,F25.13)') '    Prior value:           ', Yprior
  end if
  if (option%algorithm == 'KRR') then
    write(6,'(a)') '    ML algorithm:          ' // trim(option%algorithm)
    read (MLmodelUnit) kernel
    if (kernel == 'exponentia') then
      option%kernel = 'exponential'
    elseif (kernel == 'GaussianDe') then
      option%kernel = 'Gaussian'
      option%decayKernel = .true.
      read (MLmodelUnit) sigmap
      write(6,'(a,F25.13)') '        sigmap:            ', sigmap
      read (MLmodelUnit) period
      write(6,'(a,F25.13)') '        period:            ', period
    elseif (kernel == 'GaussianPe') then
      option%kernel = 'Gaussian'
      option%periodKernel = .true.
      read (MLmodelUnit) period
      write(6,'(a,F25.13)') '        period:            ', period
    else
      option%kernel = kernel
    end if
    write(6,'(a)') '    Kernel:                ' // trim(option%kernel)
    read (MLmodelUnit) lambda
    write(6,'(a,F25.13)')   '        lambda:            ', lambda
    if ((option%kernel == 'Gaussian')  .or. &
        (option%kernel == 'Laplacian') .or. &
        (option%kernel == 'exponential')) then
      read (MLmodelUnit) sigma
      write(6,'(a,F25.13)') '        sigma:             ', sigma
    elseif (option%kernel == 'Matern') then
      read (MLmodelUnit) nn
      write(6,'(a,I0)') '        nn:                ', nn
      read (MLmodelUnit) sigma
      write(6,'(a,F25.13)') '        sigma:             ', sigma
    elseif (option%kernel == 'polynomial') then
      read (MLmodelUnit) c
      read (MLmodelUnit) d
      write(6,'(a,F25.13)') '        c:                 ', c
      write(6,'(a,I0)')     '        d:                 ', d
    end if
    read (MLmodelUnit) Ntrain_MLmodel
    if (Ntrain_MLmodel == -3) then
      read (MLmodelUnit) Ntrain_MLmodel
      NtrVal = Ntrain_MLmodel
      NtrTot = NtrVal
      read (MLmodelUnit) NgradXYZmax
      allocate(alpha(1:NtrVal,1:3*NgradXYZmax),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate alpha')
      do i = 1, 3
        do j = 1, NgradXYZmax
          do Nspecies = 1, NtrVal
            read (MLmodelUnit) alpha(Nspecies,i+3*(j-1))
          end do
        end do
      end do
      if (option%debug) then
        do Nspecies = 1, NtrVal
          write(6,'(a,i0,a)') '        alpha(', Nspecies, '):'
          do j = 1, NgradXYZmax
            dbltmp = alpha(Nspecies,3*j-2:3*j)
            write(6,'(3F25.13)') dbltmp
          end do
        end do
      end if
    elseif (Ntrain_MLmodel == -4) then ! option%KRRtask == 'learnValGradXYZ' or learnGradXYZ - DEPRICATED, KEPT HERE TO READ OLD MODELS
      read (MLmodelUnit) Ntrain_MLmodel, NtrTot, NtrVal, inttmp, NtrGrXYZ
      read (MLmodelUnit) NgradXYZmax, NAtomsMax_MLmodel
      allocate(alpha(1:NtrTot,1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate alpha')
      allocate(Natoms_MLmodel(1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for Natoms_MLmodel array')
      allocate(NgradXYZ_MLmodel(1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for NgradXYZ_MLmodel array')
      NgradXYZ_MLmodel = NgradXYZmax
      allocate(XYZ_A_MLmodel(1:3,1:NAtomsMax_MLmodel,1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ_A_MLmodel array')
      allocate(Z_MLmodel(1:NAtomsMax_MLmodel,1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for Z_MLmodel array')
      do Nspecies = 1, Ntrain_MLmodel
        do i = 1, NAtomsMax_MLmodel
          read (MLmodelUnit) Natoms_MLmodel(Nspecies)
          read (MLmodelUnit) XYZ_A_MLmodel(1:3,i,Nspecies)
          read (MLmodelUnit) Z_MLmodel(i,Nspecies)
        end do
      end do   
      do i = 1, NtrTot
        read (MLmodelUnit) alpha(i,1)
      end do
      if (option%debug) then
        do Nspecies = 1, NtrTot
          write(6,'(a,i0,a,F25.13)') '        alpha(', Nspecies, '):             ', alpha(Nspecies,1)
        end do
      end if
    elseif (Ntrain_MLmodel == -9) then ! option%KRRtask == 'learnValGradXYZ' or learnGradXYZ
      read (MLmodelUnit) lambdaGradXYZ
      write(6,'(a,F25.13)')   '        lambdaGradXYZ:     ', lambdaGradXYZ
      read (MLmodelUnit) Ntrain_MLmodel, NtrTot, NtrVal, inttmp, NtrGrXYZ
      read (MLmodelUnit) NgradXYZmax_MLmodel, NAtomsMax_MLmodel
      allocate(alpha(1:NtrTot,1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate alpha')
      allocate(Natoms_MLmodel(1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for Natoms_MLmodel array')
      allocate(NgradXYZ_MLmodel(1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for NgradXYZ_MLmodel array')
      allocate(XYZ_A_MLmodel(1:3,1:NAtomsMax_MLmodel,1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ_A_MLmodel array')
      allocate(Z_MLmodel(1:NAtomsMax_MLmodel,1:Ntrain_MLmodel),stat=Error)
      if(Error/=0) call stopMLatomF('Unable to allocate space for Z_MLmodel array')
      do Nspecies = 1, Ntrain_MLmodel
        read (MLmodelUnit) Natoms_MLmodel(Nspecies)
        read (MLmodelUnit) NgradXYZ_MLmodel(Nspecies)
        do i = 1, NAtomsMax_MLmodel
          read (MLmodelUnit) XYZ_A_MLmodel(1:3,i,Nspecies)
          read (MLmodelUnit) Z_MLmodel(i,Nspecies)
        end do
      end do   
      do i = 1, NtrTot
        read (MLmodelUnit) alpha(i,1)
      end do
      if (option%debug) then
        do Nspecies = 1, NtrTot
          write(6,'(a,i0,a,F25.13)') '        alpha(', Nspecies, '):             ', alpha(Nspecies,1)
        end do
      end if
      
    elseif (Ntrain_MLmodel==-5) then 
      read (MLmodelUnit) Ntrain_MLmodel, NtrTot, NtrVal, inttmp 
      read (MLmodelUnit) NAtomsMax_MLmodel 
      allocate(alpha(1:NtrTot,1),stat=Error) 
      if (Error/=0) call stopMLatomF('Unable to allocate alpha')
      allocate(Natoms_MLmodel(1:Ntrain_MLmodel),stat=Error)
      if (Error/=0) call stopMLatomF('Unable to allocate space for Natoms_MLmodel array')
      

      do Nspecies = 1, Ntrain_MLmodel 
        do i = 1, NAtomsMax_MLmodel 
          read(MLmodelUnit) Natoms_MLmodel(Nspecies) 
        end do 
      end do 
      do i = 1, NtrTot 
        read (MLmodelUnit) alpha(i,1) 
      end do
      if (option%debug) then 
        do Nspecies = 1, NtrTot 
          write(6,'(a,i0,a,F25.13)') '      alpha(', Nspecies,'):             ', alpha(Nspecies,1)
        end do 
      end if 
    else
      NtrVal = Ntrain_MLmodel
      NtrTot = NtrVal
      !write(*,*) 'NtrVal = ', NtrVal
      if (option%selectPerm) then
        read (MLmodelUnit) NselectPerm 
        NtrVal = NtrVal * NselectPerm
        allocate(indicesSelectPerm(1:NselectPerm),stat=Error) 
        if (Error/=0) call stopMLatomF('Unable to allocate indicesSelectPerm')
        do Nspecies = 1, NselectPerm 
          read (MLmodelUnit) indicesSelectPerm(Nspecies)
        end do 
      end if 
      
      allocate(alpha(1:NtrVal,1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate alpha')
      do Nspecies = 1, NtrVal
        read (MLmodelUnit) alpha(Nspecies,1)
      end do
      if (option%debug) then
        do Nspecies = 1, NtrVal
            write(6,'(a,i0,a,F25.13)') '        alpha(', Nspecies, '):             ', alpha(Nspecies,1)
        end do
      end if
    end if
  end if
  
  write(6,'(a)') ''

  ! Equilibrium XYZ information: XYZeq
  read(MLmodelUnit,iostat=Error) NAtomsMax
  if (Error<0) then 
    write(6,'(a)') 'No XYZeq available in the model, reading from user-provided file'
    readEqXYZFlag = .True.
  else if (Error>0) then 
    call stopMLatomF('Error occurred when reading XYZeq')
  else
    readEqXYZFlag = .False.
    write(6,*) 'Use XYZeq saved in model'

    allocate(XYZeq(1:3,1:NAtomsMax),stat=Error)
    if (Error/=0) call stopMLatomF('Unable to allocate XYZeq')
    allocate(Zeq(1:NAtomsMax),stat=Error)
    if(Error/=0) call stopMLatomF('Unable to allocate space for nuclear charges')
    do Nspecies=1,NAtomsMax 
      do i=1, 3  
        read(MLmodelUnit) XYZeq(i,Nspecies)
      end do 
    end do
    do i=1, NAtomsMax 
      read(MLmodelUnit,iostat=Error) Zeq(i)
    end do
    if (Error /= 0) then
      Zeq = Z_MLmodel(:,1)
    end if
    if (option%debug) then 
      write(6,*) ''
      write(6,*) '   Near-equlibrium geometry:'
      write(6,*) ''
      write(6,*) '         X            Y            Z'
      do i=1,NAtomsMax
        write(6,'(3X,F10.6,3X,F10.6,3X,F10.6)') XYZeq(1:3,i)
      end do 
      write(6,*) ''
    end if 
  end if
  
  close(MLmodelUnit)

end subroutine readMLmodel
!==============================================================================

!==============================================================================
subroutine writeMLmodel()
!==============================================================================
! Save an ML model
!==============================================================================
  use A_KRR,         only : lambda, lambdaGradXYZ
  use A_KRR_kernel,  only : sigma, nn, c, d, period, sigmap
  use A_KRR_kernel,  only : NtrTot, NtrVal, NtrGrXYZ, itrval, itrgrxyz
  use A_KRR_kernel,  only : NselectPerm, indicesSelectPerm
  use dataset,       only : X, XvecSize, Ntrain, indicesTrain, Nperm, NgradXYZmax, NgradXYZ
  use dataset,       only : XYZ_A, Z, NAtomsMax, Natoms
  use dataset,       only : XYZeq, Zeq
  use dataset,       only : permutedIatoms, permlen
  use stopper,       only : stopMLatomF
  implicit none
  ! Local variables
  character(len=10) :: descrtype
  integer :: MLmodelUnit
  integer :: Nspecies, i, j
  integer :: error
  
  MLmodelUnit=52
  
  open(MLmodelUnit,file=trim(option%MLmodelFileOut),access='stream',form='unformatted', &
       action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(option%MLmodelFileOut))
  
  ! MLmodel file format
  write (MLmodelUnit) 'MLmod'
  write (MLmodelUnit) MLmodelFileVersion

  ! Molecular descriptor
  ! Must be EXACTLY 10 characters long
  if (option%readX) then
    if (option%permInvKernel) then
      write (MLmodelUnit) 'Xfpperminv'
      write (MLmodelUnit) Nperm
    else
      write (MLmodelUnit) 'Xfpnumbers'
    end if
  else
    if (option%usePermInd) then 
      descrtype = option%molDescriptor // 'x' //option%molDescrType(2:)
      write (MLmodelUnit) descrtype
    else 
      descrtype = option%molDescriptor // option%molDescrType
      write (MLmodelUnit) descrtype
    end if 
  end if
  if(.not. option%readX .and. trim(option%molDescrType) == 'permuted') then
    write (MLmodelUnit) option%permInvKernel
    write (MLmodelUnit) Nperm
    write (MLmodelUnit) option%permInvGroups
    write (MLmodelUnit) option%permInvNuclei
    if (option%usePermInd) then 
      write(MLmodelUnit) permlen
      do Nspecies = 1, Nperm 
        do i = 1, permlen
          write(MLmodelUnit) permutedIatoms(Nspecies)%oneDintArr(i)
        end do 
      end do 
    end if 
  end if
  write (MLmodelUnit) Ntrain
  write (MLmodelUnit) XvecSize
  do Nspecies = 1, Ntrain
    do i = 1, XvecSize
      write(MLmodelUnit) X(i,indicesTrain(Nspecies))
    end do
  end do

  ! Algorithm specific data
  if (option%algorithm == 'KRR') then
    if (option%prior/='0') then
      write (MLmodelUnit) 'KRm'
      write (MLmodelUnit) option%prior
      write (MLmodelUnit) Yprior
    else
      write (MLmodelUnit) 'KRR'
    end if
    if (option%decayKernel) then
      write (MLmodelUnit) 'GaussianDe'
      write (MLmodelUnit) sigmap
      write (MLmodelUnit) period
    elseif (option%periodKernel) then
      write (MLmodelUnit) 'GaussianPe'
      write (MLmodelUnit) period
    else
      write (MLmodelUnit) option%kernel(1:10)
    end if
    write (MLmodelUnit) lambda
    if ((option%kernel == 'Gaussian')  .or. &
        (option%kernel == 'Laplacian') .or. &
        (option%kernel == 'exponential')) then
      write (MLmodelUnit) sigma
    elseif (option%kernel == 'Matern') then
      write (MLmodelUnit) nn
      write (MLmodelUnit) sigma
    elseif (option%kernel == 'polynomial') then
      write (MLmodelUnit) c
      write (MLmodelUnit) d
    end if
    if (option%KRRtask == 'learnValGradXYZ' .or. &
            option%KRRtask == 'learnGradXYZ') then
      write (MLmodelUnit) -9
      write (MLmodelUnit) lambdaGradXYZ
      write (MLmodelUnit) Ntrain, NtrTot, NtrVal, 0, NtrGrXYZ
      write (MLmodelUnit) NgradXYZmax, NAtomsMax
      do Nspecies = 1, Ntrain
        write (MLmodelUnit) Natoms(indicesTrain(Nspecies))
        write (MLmodelUnit) NgradXYZ(indicesTrain(Nspecies))
        do i = 1, NAtomsMax
          write (MLmodelUnit) XYZ_A(1:3,i,indicesTrain(Nspecies))
          write (MLmodelUnit) Z(i,indicesTrain(Nspecies))
        end do
      end do
      do i = 1, NtrTot
        write (MLmodelUnit) alpha(i,1)
      end do

    else
      write (MLmodelUnit) Ntrain
      if (option%selectPerm) then
        write (MLmodelUnit) NselectPerm 
        do Nspecies = 1, NselectPerm 
          write(MLmodelUnit) indicesSelectPerm(Nspecies)
        end do 
      end if 
        do Nspecies = 1, Ntrain
          write (MLmodelUnit) alpha(Nspecies,1)
        end do
    end if
  end if

  ! Equilibrium XYZ information: XYZeq
  if (allocated(XYZeq)) then
    write(MLmodelUnit) NAtomsMax
    do Nspecies = 1, NAtomsMax 
      do i = 1, 3 
        write(MLmodelUnit) XYZeq(i,Nspecies)
      end do 
    end do 
    do i = 1, NAtomsMax 
      write(MLmodelUnit) Zeq(i)
    end do 
  end if 
  
  close(MLmodelUnit)

end subroutine writeMLmodel
!==============================================================================

end module MLmodel

