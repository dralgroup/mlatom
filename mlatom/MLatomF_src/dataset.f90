
  !---------------------------------------------------------------------------! 
  ! dataset: handling data sets                                               ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module dataset
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  use types,         only : arrayOfArrays, nuclGroups
  implicit none
  ! Types
  type :: optData
    integer               :: Nsubtrain
    integer               :: Nvalidate
    integer               :: NsubtrainGradXYZ
    integer               :: NvalidateGradXYZ
    integer, allocatable  :: indicesSubtrain(:)
    integer, allocatable  :: indicesValidate(:)
    integer, allocatable  :: indicesSubtrainGradXYZ(:)
    integer, allocatable  :: indicesValidateGradXYZ(:)
  end type optData
  type :: testData
    integer               :: Ntrain
    integer               :: Ntest
    integer               :: NtrainGradXYZ
    integer, allocatable  :: indicesTrain(:)
    integer, allocatable  :: indicesTest(:)
    integer, allocatable  :: indicesTrainGradXYZ(:)
    type(optData)         :: hyperOptData
    type(arrayOfArrays), allocatable :: indicesCVoptSplits(:)
  end type testData 
  ! Variables
  integer                               :: NitemsTot              ! Number of points in the entire   dataset
  integer, pointer                      :: Ntrain                 ! Number of points in the training     set
  integer, pointer                      :: Ntest                  ! Number of points in the test         set
  integer, pointer                      :: Nsubtrain              ! Number of points in the sub-training set
  integer, pointer                      :: Nvalidate              ! Number of points in the validation   set
  integer, pointer                      :: NtrainGradXYZ          ! Number of XYZ gradients in the training set
  integer, pointer                      :: NsubtrainGradXYZ       ! Number of XYZ gradients in the sub-training set 
  integer, pointer                      :: NvalidateGradXYZ       ! Number of XYZ gradients in the validation set
  integer                               :: Npredict               ! Number of points to be used for making predictions for using loaded ML model
  integer                               :: NAtomsMax              ! Maximum number of atoms
  integer                               :: NgradXYZmax            ! Maximum number of atoms in gradients file
  integer                               :: XvecSize               ! Length of X's vector
  integer                               :: Nprops                 ! Number of properties to learn simultaneously
  integer                               :: Nperm                  ! Number of permutations
  integer                               :: NpermGroups, NpermNucl ! Number of permutations originating from set of groups and from groups of nuclei
  integer                               :: permlen                ! size(permutedIatoms(1)%oneDIntArr)
  real(kind=rprec)                      :: Yprior                 ! Y prior 
  real(kind=rprec)                      :: Int_const              ! Integration constant for gradXYZ models
  ! Arrays
  integer,          allocatable         :: Natoms(:)              ! Number of atoms in species
  integer,          allocatable         :: NgradXYZ(:)            ! Number of atoms in file with gradients
  integer,          allocatable         :: indicesItemsTot(:)     ! Indices of all data points
  integer, pointer                      :: indicesTrain(:)        ! Indices of data points to be used as the training set for creating a final ML model to be saved
                                                                  ! or for estimating error of ML model on the test set
  integer, pointer                      :: indicesTest(:)         ! Indices of data points to be used as the test set for evaluating estimating error of ML model
  integer, pointer                      :: indicesSubtrain(:)     ! Indices of data points to be used as the sub-training set for optimization of tuning hyperparameters
  integer, pointer                      :: indicesValidate(:)     ! Indices of data points to be used as the validation set for optimization of tuning hyperparameters
  integer, pointer                      :: indicesTrainGradXYZ(:)     ! Indices of XYZ gradients to be used as the training set for creating a ML model
  integer, pointer                      :: indicesSubtrainGradXYZ(:)  ! Indices of XYZ gradients to be used as the sub-training set for optimization of tuning hyperparameters 
  integer, pointer                      :: indicesValidateGradXYZ(:)  ! Indices of XYZ gradients to be used as the validation set for optimization of tuning hyperparameters
  integer,          allocatable         :: indicesPredict(:)      ! Indices of data points to be used for making predictions for using loaded ML model
  integer,          allocatable         :: Zeq(:)                 ! Nuclear charges in equlibrium structure
  integer,          allocatable         :: Z(:,:)                 ! Nuclear charges
  real(kind=rprec), allocatable         :: X(:,:)                 ! Data set
  real(kind=rprec), allocatable         :: Y(:)                   ! Reference data
  real(kind=rprec), allocatable         :: Ygrad(:,:)             ! Reference gradients
  real(kind=rprec), allocatable         :: YgradXYZ(:,:,:)        ! Reference gradients in Cartesian coordinates
  real(kind=rprec), allocatable         :: Yest(:)                ! Estimated values
  real(kind=rprec), allocatable         :: YestGrad(:,:)          ! Estimated gradients YestGrad(1:XvecSize,Nspecies),
  real(kind=rprec), allocatable         :: YgradXYZest(:,:,:)     ! Estimated gradients YgradXYZest(1:3,1:Natoms,Nspecies)
  real(kind=rprec), allocatable         :: YestCVtest(:)          ! Estimated values
  real(kind=rprec), allocatable         :: YestCVopt(:)           ! Estimated values
  real(kind=rprec), allocatable         ::YgradXYZestCVtest(:,:,:)! Estimated values
  real(kind=rprec), allocatable         :: YgradXYZestCVopt(:,:,:)! Estimated values
  real(kind=rprec), allocatable, target :: XYZ_A(:,:,:)           ! Atomic coordinates in Angstrom
  real(kind=rprec), allocatable, target :: XYZ_B(:,:,:)           ! Atomic coordinates in Bohr
  real(kind=rprec), allocatable         :: XYZ_A_sorted(:,:,:)    ! Sorted coordinates in Angstrom
  real(kind=rprec), allocatable         :: XYZeq(:,:)             ! Nuclear coordinates in Angstrom in equlibrium structure
  real(kind=rprec), allocatable         :: Req(:)                 ! Internuclear distances in the equilibirum structure
  real(kind=rprec), allocatable         :: ReqMod(:) ! Internuclear distances in the equilibirum structure used in module D_rel2eq
  ! Arrays of arrays
  type(arrayOfArrays), allocatable      :: indicesCVtestSplits(:) ! Indices of CV splits used for testing
  type(arrayOfArrays), pointer          :: indicesCVoptSplits(:)  ! Indices of CV splits used for optimization of hyperparameters
  type(nuclGroups),    allocatable      :: permInvGroups(:)       ! Permutationally invariant groups
  type(arrayOfArrays), allocatable      :: permInvNuclei(:)       ! Permutationally invariant nuclei within the above groups  
  type(arrayOfArrays), allocatable      :: permutedIatoms(:)      ! Indices of permuted atoms
  type(arrayOfArrays), allocatable      :: modPermutedIatoms(:)   ! Same as permutedIatoms, but used in MLmodel
  ! Data for optimization of hyperparameters
  type(optData),                 target :: hyperOptData           ! Information necessary to hyperparmeter optimization
  type(optData),    allocatable, target :: CVoptFolds(:)          ! Information necessary to build cross-validation folds for CVopt
  ! Data for training and testing
  type(testData),                target :: trainTestData          ! Information necessary to build cross-validation folds for CVtest
  type(testData),   allocatable, target :: CVtestFolds(:)         ! Information necessary to build cross-validation folds for CVtest
  ! Select permutations by mindMRSD or minEuclidean 
  integer,          allocatable         :: minpermCount(:)
  ! Calculate gradient for user-defined atoms 
  integer,          allocatable         :: gradForAtom(:)
  integer                               :: NgradForAtom
  integer                               :: gradForXel
  
contains

!==============================================================================
subroutine getNitemsTot()
!==============================================================================
  implicit none
  
  if (option%Nuse /= 'ALL') then
    read(option%Nuse, *) NitemsTot
  end if

end subroutine getNitemsTot
!==============================================================================

!==============================================================================
subroutine getTotalTrainTest()
!==============================================================================
  implicit none
  real(kind=rprec) :: tempDble
  integer :: i, Nsets, Error
  
  nullify(Ntrain,Ntest,indicesTrain,indicesTest)
  nullify(NtrainGradXYZ, indicesTrainGradXYZ)
  
  allocate(indicesItemsTot(1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate indicesItemsTot')
  do i = 1, NitemsTot
    indicesItemsTot(i) = i
  end do
  
  if (option%CVtest) return

  ! Get number of entries to be used for training and test sets
  Ntrain => trainTestData%Ntrain
  Ntest  => trainTestData%Ntest
  NtrainGradXYZ => trainTestData%NtrainGradXYZ
  Ntrain = 0
  Ntest  = 0
  NtrainGradXYZ = 0
  if (option%createMLmodel .or. option%estAccMLmodel .or. &
      trim(option%iTrainOut) /= '') then
    read(option%Ntrain,*) tempDble
    if (tempDble < 1.0_rprec) then
      Ntrain = nint(tempDble * NitemsTot)
    else
      Ntrain = nint(tempDble)
      if (Ntrain > NitemsTot) call stopMLatomF('The training set cannot be larger than the entire dataset')
    end if
    if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
      read(option%Ntest,*) tempDble
      if (tempDble < 1.0_rprec) then
        Ntest = nint(tempDble * NitemsTot)
      else
        Ntest = nint(tempDble)
        if (Ntest > NitemsTot) call stopMLatomF('The test set cannot be larger than the entire dataset')
      end if
    end if
  end if
    
  if (Ntrain == 0) then
    if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
      if (Ntest == 0) then
        Ntrain = nint(0.8 * NitemsTot)
      else
        Ntrain = NitemsTot - Ntest
      end if
    else
      Ntrain = NitemsTot
    end if
  end if
  if (Ntest == 0) then
    Ntest = NitemsTot - Ntrain
  end if
    
  if (Ntrain + Ntest > NitemsTot) then
    call stopMLatomF('Number of entries in the training and test sets is larger than the total number of entries')
  elseif (Ntrain + Ntest /= NitemsTot) then
    write(6,'(a,I0,a,I0)') ' <!> Use only ', Ntrain + Ntest, ' data entries out of ', NitemsTot
    write(6,'(a)') ''
    !NitemsTot = Ntrain + Ntest
  end if

  ! Sample data points into the subsets
  call sampleTrainTest()
  
  ! Write files with indices, if requested
  if (trim(option%iTrainOut) /= '') then
    call writeIndices(trim(option%iTrainOut),  Ntrain,  indicesTrain)
  end if
  if (trim(option%iTestOut) /= '') then
    call writeIndices(trim(option%iTestOut),  Ntest,  indicesTest)
  end if

end subroutine getTotalTrainTest
!==============================================================================

subroutine sampleTrainTest()
!==============================================================================
  implicit none
  ! Local variables
  integer, allocatable :: tempIndArr(:)
  integer :: i, itemp, Error
  
  allocate(trainTestData%indicesTrain(1:Ntrain),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate indicesTrain')
  indicesTrain => trainTestData%indicesTrain
  if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
    allocate(trainTestData%indicesTest(1:Ntest),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate indicesTest')
    indicesTest => trainTestData%indicesTest
  end if
  
  if (trim(option%sampling) == 'none') then
    do i = 1, Ntrain
      indicesTrain(i) = i
    end do
    if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
      do i = 1, Ntest
        indicesTest(i) = Ntrain + i
      end do
    end if
  elseif (trim(option%sampling) == 'random') then
    ! Indices will be selected randomly during cross-validation, doing it here makes no sence and lead to small numerical differences when indices are read
    ! PD (17.04.2019): it may happen that the user samples Ntrain < NitemsTot randomly, let's live with small numerical differences
    !if ((option%createMLmodel .or. trim(option%iTrainOut) /= '') .and. option%CVopt) then
    !  do i = 1, Ntrain
    !    indicesTrain(i) = i
    !  end do
    !else
      allocate(tempIndArr(1:NitemsTot),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate tempIndArr')
      do i = 1, NitemsTot
        tempIndArr(i) = i
      end do
      call intArrShuffle(NitemsTot, tempIndArr)
      do i = 1, Ntrain
        indicesTrain(i) = tempIndArr(i)
      end do
      if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
        do i = 1, Ntest
          indicesTest(i) = tempIndArr(Ntrain + i)
        end do
      end if
      deallocate(tempIndArr)
    !end if
  elseif (trim(option%sampling) == 'structure-based' .or. &
          trim(option%sampling) == 'farthest-point') then
    allocate(tempIndArr(1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate tempIndArr')
    do i = 1, NitemsTot
      tempIndArr(i) = i
    end do
    call structureBasedSampling(NitemsTot, tempIndArr, Ntrain, indicesTrain)
    if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
      itemp = 0
      do i = 1, NitemsTot
        if (any(indicesTrain == i)) cycle
        itemp = itemp + 1
        if (itemp > Ntest) exit
        indicesTest(itemp) = i
      end do
    end if
    deallocate(tempIndArr)
  elseif (trim(option%sampling) == 'user-defined') then
    call readIndices(option%iTrainIn, Ntrain, indicesTrain)
    if (option%estAccMLmodel) call readIndices(option%iTestIn, Ntest, indicesTest)
  end if

end subroutine sampleTrainTest
!==============================================================================

subroutine getSubtrainValidate(hyperOptDataPt, CVtestPref)
!==============================================================================
  implicit none
  ! Arguments
  type(optData),    intent(inout), target :: hyperOptDataPt ! Information necessary to hyperparmeter optimization
  character(len=*), intent(in)            :: CVtestPref     ! Prefix of the file name with indices
  ! Local variables
  real(kind=rprec) :: tempDble
  integer :: i, Error

  ! Get number of entries to be used for sub-training and validation sets

  nullify(Nsubtrain,Nvalidate,indicesSubtrain,indicesValidate)
  nullify(NsubtrainGradXYZ, indicesSubtrainGradXYZ)
  nullify(NvalidateGradXYZ, indicesValidateGradXYZ)

  if (.not. (option%CVopt .and. .not. option%onTheFly)) then
    Nsubtrain => hyperOptDataPt%Nsubtrain
    Nvalidate => hyperOptDataPt%Nvalidate
  end if
  
  if (option%CVopt) return

  read(option%Nsubtrain,*) tempDble
  if (tempDble < 1.0_rprec) then
    Nsubtrain = nint(tempDble * Ntrain)
  else
    Nsubtrain = nint(tempDble)
    if (Nsubtrain > Ntrain) call stopMLatomF('The sub-training set cannot be larger than the training set')
  end if
  
  read(option%Nvalidate,*) tempDble
  if (tempDble < 1.0_rprec) then
    Nvalidate = nint(tempDble * Ntrain)
  else
    Nvalidate = nint(tempDble)
    if (Nvalidate > Ntrain) call stopMLatomF('The validation set cannot be larger than the training set')
  end if

  if (Nsubtrain == 0) then
    if (Nvalidate == 0) then
        Nsubtrain = nint(0.8 * Ntrain)
    else
      Nsubtrain = Ntrain - Nvalidate
    end if
  end if
  if (Nvalidate == 0) then
    Nvalidate = Ntrain - Nsubtrain
  end if


  ! Sample data points into the subsets
  call sampleSubtrainValidate(hyperOptDataPt, CVtestPref)
  
  ! Write files with indices, if requested
  if (trim(option%iSubtrainOut) /= '') then
    call writeIndices(trim(CVtestPref) // trim(option%iSubtrainOut),  Nsubtrain, indicesSubtrain)
  end if
  if (trim(option%iValidateOut) /= '') then
    call writeIndices(trim(CVtestPref) // trim(option%iValidateOut),  Nvalidate, indicesValidate)
  end if

end subroutine getSubtrainValidate
!==============================================================================

subroutine sampleSubtrainValidate(hyperOptDataPt, CVtestPref)
!==============================================================================
  implicit none
  ! Arguments
  type(optData),    intent(inout), target :: hyperOptDataPt ! Information necessary to hyperparmeter optimization
  character(len=*), intent(in)            :: CVtestPref     ! Prefix of the file name with indices
  ! Local variables
  integer, allocatable :: tempIndArr(:)
  integer :: i, Error
  
  allocate(hyperOptDataPt%indicesSubtrain(1:Nsubtrain),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate indicesSubtrain')
  allocate(hyperOptDataPt%indicesValidate(1:Nvalidate),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate indicesValidate')
  indicesSubtrain => hyperOptDataPt%indicesSubtrain
  indicesValidate => hyperOptDataPt%indicesValidate

  if (trim(option%sampling) == 'none') then
    do i = 1, Nsubtrain
      indicesSubtrain(i) = i
    end do
    do i = 1, Nvalidate
      indicesValidate(i) = Nsubtrain + i
    end do  
  elseif (trim(option%sampling) == 'random') then
    allocate(tempIndArr(1:Ntrain),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate tempIndArr')
    tempIndArr(:) = indicesTrain(:)
    call intArrShuffle(Ntrain, tempIndArr)
    do i = 1, Nsubtrain
      indicesSubtrain(i) = tempIndArr(i)
    end do
    do i = 1, Nvalidate
      indicesValidate(i) = tempIndArr(Nsubtrain + i)
    end do
    deallocate(tempIndArr)
  elseif (trim(option%sampling) == 'structure-based' .or. &
          trim(option%sampling) == 'farthest-point') then
    do i = 1, Nsubtrain
      indicesSubtrain(i) = indicesTrain(i)
    end do
    do i = 1, Nvalidate
      indicesValidate(i) = indicesTrain(Nsubtrain + i)
    end do
  elseif (trim(option%sampling) == 'user-defined') then
    call readIndices(trim(CVtestPref) // trim(option%iSubtrainIn), Nsubtrain, indicesSubtrain)
    call readIndices(trim(CVtestPref) // trim(option%iValidateIn), Nvalidate, indicesValidate)
  end if

end subroutine sampleSubtrainValidate
!==============================================================================

!==============================================================================
subroutine splitAndSample(Npoints, indicesPoints, Nsplits, indicesCVsplits, fName, IOiFile)
!==============================================================================
  implicit none
  ! Arguments
  integer, intent(in)                :: Npoints                    ! Number of points
  integer, intent(in)                :: indicesPoints(1:Npoints)   ! Indices of points
  integer, intent(in)                :: Nsplits                    ! Number of splits
  type(arrayOfArrays), intent(inout) :: indicesCVsplits(1:Nsplits) ! Indices of points in the splits
  character(len=*),    intent(in)    :: fName                      ! Prefix of the file name with indices
  character(len=5),    intent(in)    :: IOiFile                    ! read or write file with indices
  ! Local variables
  character(len=256) :: stmp
  integer, allocatable :: tempIndArr(:)
  integer :: NminSplit  ! minimal size of split
  integer :: Nremainder ! remaining items of set
  integer :: NsplitSize ! actual size of split
  integer :: i, Error, stsize, newsize

  if (trim(option%sampling) == 'random') then
    allocate(tempIndArr(1:Npoints),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate tempIndArr')
    tempIndArr(:) = indicesPoints(:)
    call intArrShuffle(Npoints, tempIndArr)
  end if

  NminSplit = Npoints / Nsplits
  Nremainder = Npoints - Nsplits * NminSplit
  stsize = 0
  newsize = 0

  do i = 1, Nsplits
    if (trim(IOiFile) /= 'no') then
      write(stmp, '(i0)') i
      stmp = trim(fName) // trim(stmp) // '.dat'
    end if
    NsplitSize = NminSplit + max(0,min(1,Nremainder))
    Nremainder = Nremainder - 1
    allocate(indicesCVsplits(i)%oneDintArr(1:NsplitSize),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate indicesCVsplits(i)%oneDintArr')
    indicesCVsplits(i)%oneDintArr(1:NsplitSize) = 0
    newsize = stsize + NsplitSize
    if (trim(option%sampling) == 'none') then
      indicesCVsplits(i)%oneDintArr(1:NsplitSize) = indicesPoints(stsize+1:newsize)
    elseif (trim(option%sampling) == 'random') then
      indicesCVsplits(i)%oneDintArr(1:NsplitSize) = tempIndArr(stsize+1:newsize)
    elseif (trim(option%sampling) == 'user-defined' .and. trim(IOiFile) == 'read') then
      call readIndices(stmp, NsplitSize, indicesCVsplits(i)%oneDintArr)
    end if
    stsize = newsize
    if (trim(IOiFile) == 'write') then
      call writeIndices(stmp, NsplitSize, indicesCVsplits(i)%oneDintArr)
    end if
  end do
  
  if (trim(option%sampling) == 'random') deallocate(tempIndArr)
  
!==============================================================================
end subroutine splitAndSample
!==============================================================================

!============================================================================== 
subroutine intArrShuffle(intArrSize, intArr)
!==============================================================================
!
! This subroutine implements the Knuth shuffle (the Fisher-Yates shuffle)
! with substitution (see http://rosettacode.org/wiki/Knuth_shuffle#Fortran)
!
!==============================================================================
  use mathUtils, only : init_random_seed
  implicit none
  ! Arguments
  integer, intent(in)    :: intArrSize
  integer, intent(inout) :: intArr(1:intArrSize)
  ! Local variables
  real(kind=rprec) :: randDble
  integer          :: i, randInt, itemp
  
  call init_random_seed()
 
  do i = intArrSize, 2, -1
    call random_number(randDble)
    randInt = int(randDble * i) + 1
    itemp = intArr(randInt)
    intArr(randInt) = intArr(i)
    intArr(i) = itemp
  end do
 
end subroutine intArrShuffle
!==============================================================================

!============================================================================== 
subroutine structureBasedSampling(Nall, allPoints, Nsampled, sampledPoints)
!==============================================================================
!
! This subroutine implements the structure-based sampling
!
!==============================================================================
  use timing, only : timeTaken
  implicit none
  ! Arguments
  integer, intent(in)            :: Nall
  integer, intent(inout), target :: allPoints(1:Nall)
  integer, intent(in)            :: Nsampled
  integer, intent(inout)         :: sampledPoints(1:Nsampled)
  ! Local variables
  integer, pointer               :: aP(:)
  real(kind=rprec), allocatable  :: distances(:,:), minDs(:), Xeq(:)
  integer*8                      :: dateStart(1:8) ! Time and date, when MLatomF starts
  integer,          allocatable  :: minIs(:), locindices(:)
  integer, allocatable, target   :: aPsorted(:)
  integer                        :: itemp(1), itempr2(2)
  real(kind=rprec)               :: dst, mindiff
  integer                        :: ii, jj, kk, kstart, icurrent, imin, Error

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  ! Initialization
  sampledPoints = 0
  aP            => allPoints
  
  ! Get the matrix of all distances
  allocate(distances(1:Nall,1:Nall),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate distances')
  allocate(minDs(1:Nall),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate minDs')
  allocate(minIs(1:Nall),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate minIs')
  allocate(locindices(1:Nall),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate locindices')
  
  ! Sort by distance to the equilibrium structure if requested
  if (trim(option%sampling) == 'structure-based') then
    distances = 0.0_rprec
    allocate(Xeq(1:XvecSize),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate Xeq')
    allocate(aPsorted(1:Nall),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate aPsorted')
    Xeq = 1.0_rprec ! Only works if molecular descriptor of the equilibrium structure is unity
    aPsorted = allPoints
    !$OMP PARALLEL DO PRIVATE(ii) SHARED(distances,aPsorted,X,Xeq) SCHEDULE(STATIC)
    do ii = 1, Nall
      distances(ii,1) = distance(XvecSize,X(:,aPsorted(ii)), Xeq(:))
    end do
    !$OMP END PARALLEL DO
    do ii=1,Nall-1
      imin = ii
      do jj=ii+1,Nall
        if (distances(aPsorted(jj),1) < distances(aPsorted(imin),1)) then
          imin = jj
        end if
      end do
      if (ii /= imin) then
        kk                          = aPsorted(imin)
        aPsorted(imin)              = aPsorted(ii)
        aPsorted(ii)                = kk
      end if
    end do    
    aP => aPsorted
  end if
  
  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Time for sorting to equilibrium structure:')
  if(option%benchmark) call date_and_time(values=dateStart)
  
  ! Get distances between all nuclear pairs
  distances = 0.0_rprec
  itempr2   = (/1, 2/)
  !$OMP PARALLEL DO PRIVATE(ii,jj) SHARED(distances) SCHEDULE(STATIC)
  do ii = 1, Nall
    distances(ii,ii) = 0.0_rprec
    do jj = ii + 1, Nall
      distances(ii,jj) = distance(XvecSize,X(:,aP(ii)), X(:,aP(jj)))
      distances(jj,ii) = distances(ii,jj)
    end do
  end do
  !$OMP END PARALLEL DO
  
  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Time for calculating all internuclear distances:')
  if(option%benchmark) call date_and_time(values=dateStart)
        
  ! Sample the first points
  if (trim(option%sampling) == 'structure-based') then
    ! The first point is always 1 (usuful for some applications)
    sampledPoints(1) = 1
    kstart = 2
  else ! trim(option%sampling) == 'farthest-point'
    ! Take pair of points with the largest distance between them as an initial seed for the training set
    itempr2 = MAXLOC(distances)
    if (itempr2(1) < itempr2(2)) then
      sampledPoints(1) = itempr2(1)
      sampledPoints(2) = itempr2(2)
    else
      sampledPoints(1) = itempr2(2)
      sampledPoints(2) = itempr2(1)
    end if
    kstart = 3
  end if
  
  ! Find all other points
  do kk = kstart, Nsampled
    minDs = 0.0_rprec
    minIs = 0
    locindices = 0
    icurrent = kk - 1
    do ii = 1, Nall
      if (any(sampledPoints == ii)) then
        locindices(ii) = 1
        cycle
      end if
      icurrent = icurrent + 1
      locindices(ii) = icurrent
    end do
    !$OMP PARALLEL DO PRIVATE(ii,mindiff,jj,dst) SHARED(sampledPoints,distances,minDs,minIs,locindices) SCHEDULE(STATIC)
    do ii = 1, Nall
      if (any(sampledPoints == ii)) cycle
      mindiff = distances(ii, sampledPoints(1))
      do jj = 2, kk - 1
        dst = distances(ii, sampledPoints(jj))
        if (dst < mindiff) then
          mindiff = dst
        end if
      end do
      minDs(locindices(ii)) = mindiff
      minIs(locindices(ii)) = ii
    end do
    !$OMP END PARALLEL DO
    itemp = maxloc(minDs(kk:Nall))
    sampledPoints(kk) = minIs(kk-1+itemp(1))
  end do
  
  ! Finally, translate local indices of points running from 1 to Nsampled to real indices
  do ii = 1, Nsampled
    sampledPoints(ii) = aP(sampledPoints(ii))
  end do
          
  deallocate(distances)
  deallocate(minDs)
  deallocate(minIs)
  deallocate(locindices)
  if(allocated(Xeq))      deallocate(Xeq)
  if(allocated(aPsorted)) deallocate(aPsorted)
  
  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Time for sampling:')
        
end subroutine structureBasedSampling
!==============================================================================

subroutine readX(filename)
  implicit none
  ! Parameters
  integer, parameter :: Nmax   = 100000
  integer, parameter :: maxLen = 1000000
  ! Argument
  character(len=*), intent(in) :: filename
  ! Arrays
  real(kind=rprec) :: rtemparray(1:Nmax)
  ! Local variables
  character(len=maxLen) :: stmp
  integer :: xunit, Error, Error2, i, Nlines
  logical :: first

  ! Initialize
  Error     = 0
  Error2    = 0
  Nlines    = 0
  XvecSize  = 0
  first     = .true.

  xunit = 26
  open(xunit,file=trim(filename),action='read',IOStat=Error)
  if (error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  ! Find number of entries and size of X vector 
  do while (.true.)
    read(xunit,'(a)',iostat=error) stmp
    if (error /= 0) exit
    Nlines = Nlines + 1
    if (first) then
      first = .false.
      do i = 1, Nmax
        read(stmp,*,iostat=error) rtemparray(1:i)
        if (error /= 0) exit
        XvecSize = i
      end do
    end if
  end do
  
  ! Check if we have enough data
  if (option%Nuse == 'ALL') then
    NitemsTot = Nlines
  else
    call getNitemsTot()
    if (NitemsTot > Nlines) then
      call stopMLatomF('File ' // trim(filename) // ' contains less data than requested')
    end if
  end if
  
  ! Rewind file
  rewind(xunit)
  
  ! Allocate arrays
  allocate(X(1:XvecSize,1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector with X values')
  
  ! Initialize
  X = 0.0_rprec

  do i=1,NitemsTot
    read(xunit,*,IOStat=Error2) X(1:XvecSize,i)
    if (Error2 /= 0) call stopMLatomF('Error while reading X values')
  end do

  close(xunit)

end subroutine readX

subroutine readXYZCoords(filename)
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  integer :: Nmols, Error
  logical :: convert

  ! Initialize  
  convert       = .false.
  NAtomsMax  = 0

  ! Check what kind of XYZ file we are dealing with  
  call parseXYZfile(filename, Nmols, NAtomsMax, convert)
  
  ! Check if we have enough data
  if (option%Nuse == 'ALL') then
    NitemsTot = Nmols
  else
    call getNitemsTot()
    if (NitemsTot > Nmols) then
      call stopMLatomF('File ' // trim(filename) // ' contains less data than requested')
    end if
  end if
  
  ! Allocate arrays  
  allocate(XYZ_A(1:3,1:NAtomsMax,1:NitemsTot),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for XYZ coordinates')
  allocate(Z(1:NAtomsMax,1:NitemsTot),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for nuclear charges')
  allocate(Natoms(1:NitemsTot),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for Natoms array')
  
  ! Initialize
  Z      = 0
  XYZ_A  = 0.0_rprec
  Natoms = 0
  
  call readXYZ(filename, NitemsTot, NAtomsMax, convert, Natoms, Z, XYZ_A) 
  
end subroutine readXYZCoords

subroutine parseXYZfile(filename, Nmols, NatMax, convert)
  implicit none
  ! Argument
  character(len=*), intent(in)  :: filename
  integer,          intent(out) :: Nmols
  integer,          intent(out) :: NatMax
  logical,          intent(out) :: convert
  ! Local variables
  character(len=256) :: stmp     ! Temporary string
  character(len=4)   :: symb     ! Symbol of the element
  real(kind=rprec)   :: xyz(1:3) ! Coordinates
  integer :: el             ! Atomic number
  integer :: Nspecies       ! Number of molecules
  integer :: Nlines         ! Number of lines
  integer :: NatMol         ! Number of atoms in current molecule
  integer :: xyzunit        ! File unit
  integer :: i              ! Loop index
  integer :: error, ioerror ! Errors for reading file
  ! Flags
  logical :: readNat
  logical :: checkXYZ

  ! Initialization
  Nmols    = 0
  NatMax   = 0
  convert  = .false.
  readNat  = .true.
  checkXYZ = .true.
  xyzunit  = 25
  Nlines   = 0  
    
  open(xyzunit,file=filename,action='read',iostat=Error)
  if (Error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  ! First define number of molecules and the largest number of atoms
  do while (.true.)
    if (readNat) then
      read(xyzunit,*,iostat=ioerror) NatMol
    else
      read(xyzunit,'(a)',iostat=ioerror) stmp
    end if

    if (ioerror /= 0) exit

    if (readNat) then
      Nmols    = Nmols + 1
      readNat = .false.
      if (NatMol > NatMax) then
        NatMax = NatMol
      end if
    else
      Nlines = Nlines + 1
      if (Nlines == (NatMol + 1)) then
        Nlines   = 0
        readNat  = .true.
      elseif ( (Nlines == 2) .and. checkXYZ ) then
        checkXYZ = .false.
        read(stmp,*,iostat=error) symb, xyz(1:3)
        if ( .not.( (ichar(symb(1:1)) >= ichar('0')) .and. &
                    (ichar(symb(1:1)) <= ichar('9')) ) ) then
          convert = .true.
        end if
      end if
    end if
    
  end do
  
  close(xyzunit)

end subroutine parseXYZfile

subroutine readXYZ(filename, Nmols, NatMax, convert, Nats, charges, coords) 
  implicit none
  ! Argument
  character(len=*), intent(in)  :: filename
  integer,          intent(in)  :: Nmols
  integer,          intent(in)  :: NatMax
  logical,          intent(in)  :: convert
  integer,          intent(out) :: Nats(1:Nmols)
  integer,          intent(out) :: charges(1:NatMax,1:Nmols)
  real(kind=rprec), intent(out) :: coords(1:3,1:NatMax,1:Nmols)
  ! Local variables
  character(len=256) :: stmp     ! Temporary string
  character(len=4)   :: symb     ! Symbol of the element
  real(kind=rprec)   :: xyz(1:3) ! Coordinates
  integer :: el             ! Atomic number
  integer :: Nspecies       ! Number of molecules
  integer :: Nlines         ! Number of lines
  integer :: Ncoords        ! Number of coordinates
  integer :: NatMol         ! Number of atoms in current molecule
  integer :: xyzunit        ! File unit
  integer :: error, ioerror ! Errors for reading file

  ! Initialize
  Nats     = 0
  charges  = 0
  coords   = 0.0_rprec  
  xyzunit  = 25
  Nspecies = 0
  Nlines   = 0  
    
  open(xyzunit,file=filename,action='read',iostat=Error)
  if (Error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  Nspecies = 0
  Nlines     = 0
  Ncoords    = 0
  do Nspecies = 1, Nmols
    read(xyzunit,*,iostat=ioerror) Nats(Nspecies)
    if (ioerror /= 0) call stopMLatomF('Error while reading coordinates')
    
    read(xyzunit,'(a)',iostat=ioerror) stmp
    if (ioerror /= 0) call stopMLatomF('Error while reading coordinates')
    
    do Ncoords = 1, Nats(Nspecies)
      if (convert) then
        read(xyzunit,*,iostat=ioerror) stmp, xyz(1:3)
      else
        read(xyzunit,*,iostat=ioerror) el, xyz(1:3)
      end if
      if (ioerror /= 0) call stopMLatomF('Error while reading coordinates')
      if (convert) then
        charges(Ncoords,Nspecies) = sym2at(trim(stmp))
      else
        charges(Ncoords,Nspecies) = el
      end if
      coords(1:3,Ncoords,Nspecies) = xyz(1:3)
    end do    
  end do

  close(xyzunit)

end subroutine readXYZ

subroutine readY(filename)
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  integer :: yunit, Error, Error2, i, Nlines

  ! Initialize
  Error     = 0
  Error2    = 0
  Nlines = 0

  yunit = 30
  open(yunit,file=trim(filename),action='read',IOStat=Error)
  if (error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  ! Find number of entries  
  do while (.true.)
    read(yunit,*,iostat=error)
    if (error /= 0) exit
    Nlines = Nlines + 1
  end do
  
  ! Check if we have enough data
  if     (NitemsTot > Nlines) then
    call stopMLatomF('File ' // trim(filename) // ' contains less data than requested')
  elseif (NitemsTot == 0) then
    NitemsTot = Nlines
  elseif ((Nlines > NitemsTot) .and. (option%Nuse == 'ALL')) then
    write (6,'(a,I0,a)') ' <!> File ' // trim(filename) // ' contains ', Nlines, ' reference values'
    write (6,'(a,I0,a)') '     but only first ', NitemsTot, ' will be used'
    write (6,'(a)')      ''
  end if
    
  ! Rewind file
  rewind(yunit)
  
  ! Allocate arrays
  if (.not. allocated(Y)) then 
    allocate(Y(1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector Y')
  end if 
  if (.not. allocated(Yest)) then
    allocate(Yest(1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector Yest')  
    Yest = 0.0_rprec
  end if
  if (option%CVtest) then
    allocate(YestCVtest(1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YestCVtest')  
    YestCVtest = 0.0_rprec
  end if
  if (option%CVopt) then
    allocate(YestCVopt(1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YestCVopt')  
    YestCVopt = 0.0_rprec
  end if
  
  ! Initialize
  Y    = 0.0_rprec
  do i=1,NitemsTot
    read(yunit,*,IOStat=Error2) Y(i)
    if (Error2 /= 0) call stopMLatomF('Error while reading reference data')
  end do

  close(yunit)

end subroutine readY

subroutine readYgrad(filename)
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  integer :: yunit, Error, Error2, i, Nlines

  ! Initialize
  Error     = 0
  Error2    = 0
  Nlines = 0

  yunit = 30
  open(yunit,file=trim(filename),action='read',IOStat=Error)
  if (error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  ! Find number of entries  
  do while (.true.)
    read(yunit,*,iostat=error)
    if (error /= 0) exit
    Nlines = Nlines + 1
  end do
  
  ! Check if we have enough data
  if     (NitemsTot > Nlines) then
    call stopMLatomF('File ' // trim(filename) // ' contains less data than requested')
  elseif (NitemsTot == 0) then
    NitemsTot = Nlines
  elseif ((Nlines > NitemsTot) .and. (option%Nuse == 'ALL')) then
    write (6,'(a,I0,a)') ' <!> File ' // trim(filename) // ' contains ', Nlines, ' reference values'
    write (6,'(a,I0,a)') '     but only first ', NitemsTot, ' will be used'
    write (6,'(a)')      ''
  end if
    
  ! Rewind file
  rewind(yunit)
  
  ! Allocate arrays
  allocate(Ygrad(1:XvecSize,1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector Ygrad')
  if (.not. allocated(YestGrad)) then
    allocate(YestGrad(1:XvecSize,1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YestGrad')  
    YestGrad = 0.0_rprec
  end if
  !if (option%CVtest) then
  !  allocate(YestCVtest(1:XvecSize,1:NitemsTot),stat=Error)
  !  if(Error/=0)call stopMLatomF('Unable to allocate vector YestCVtest')  
  !  YestCVtest = 0.0_rprec
  !end if
  !if (option%CVopt) then
  !  allocate(YestCVopt(1:XvecSize,1:NitemsTot),stat=Error)
  !  if(Error/=0)call stopMLatomF('Unable to allocate vector YestCVopt')  
  !  YestCVopt = 0.0_rprec
  !end if
  
  ! Initialize
  Ygrad    = 0.0_rprec
  do i=1,NitemsTot
    read(yunit,*,IOStat=Error2) Ygrad(1:XvecSize,i)
    if (Error2 /= 0) call stopMLatomF('Error while reading reference Ygrad data')
  end do

  close(yunit)

end subroutine readYgrad

!==============================================================================
subroutine readYgradXYZ(filename)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  character(len=256) :: stmp     ! Temporary string
  integer            :: ygradxyzunit, Nmols, Ncoords, ii, Error

  ! Initialize
  ygradxyzunit = 35
  NgradXYZmax  = 0

  ! Get essential information from the file
  open(ygradxyzunit,file=filename,action='read',iostat=Error)
  if (Error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  Nmols = 0
  Ncoords    = 0
  do while (.true.)
    read(ygradxyzunit,*,iostat=Error) Ncoords
    if (Error /= 0) exit
    Nmols = Nmols + 1
    if (Ncoords > NgradXYZmax) NgradXYZmax = Ncoords
    
    read(ygradxyzunit,'(a)',iostat=Error)
    if (Error /= 0) call stopMLatomF('Error while reading XYZ gradients')
    
    do ii = 1, Ncoords
      read(ygradxyzunit,*,iostat=Error)
      if (Error /= 0) call stopMLatomF('Error while reading XYZ gradients')
    end do    
  end do
  
  ! Check if we have enough data
  if     (NitemsTot > Nmols) then
    call stopMLatomF('File ' // trim(filename) // ' contains less data than geometries')
  elseif (NitemsTot == 0) then
    NitemsTot = Nmols
  elseif ((Nmols > NitemsTot) .and. (option%Nuse == 'ALL')) then
    write (6,'(a,I0,a)') ' <!> File ' // trim(filename) // ' contains ', Nmols, ' reference values'
    write (6,'(a,I0,a)') '     but only first ', NitemsTot, ' will be used'
    write (6,'(a)')      ''
  end if

  ! Rewind file
  rewind(unit=ygradxyzunit, iostat=Error)
  if (Error /= 0) call stopMLatomF('Error while rewinding file ' // trim(filename))
  
  ! Allocate arrays  
  allocate(YgradXYZ(1:3,1:NgradXYZmax,1:NitemsTot),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for YgradXYZ array')
  allocate(NgradXYZ(1:NitemsTot),stat=Error)
  if(Error/=0) call stopMLatomF('Unable to allocate space for NgradXYZ array')
  if (.not. allocated(YgradXYZest)) then
    allocate(YgradXYZest(1:3,1:NgradXYZmax,1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YgradXYZest')  
    YgradXYZest = 0.0_rprec
  end if
  if (option%CVtest) then
    allocate(YgradXYZestCVtest(1:3,1:NgradXYZmax,1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YgradXYZestCVtest')
    YgradXYZestCVtest = 0.0_rprec
  end if
  if (option%CVopt) then
    allocate(YgradXYZestCVopt(1:3,1:NgradXYZmax,1:NitemsTot),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate vector YgradXYZestCVopt')
    YgradXYZestCVopt = 0.0_rprec
  end if
  
  ! Initialize
  YgradXYZ    = 0.0_rprec
  NgradXYZ    = 0
  do ii = 1, NitemsTot
    read(ygradxyzunit,*,iostat=Error) NgradXYZ(ii)
    if (Error /= 0) call stopMLatomF('Error while reading XYZ gradients')
    
    read(ygradxyzunit,'(a)',iostat=Error) stmp
    if (Error /= 0) call stopMLatomF('Error while reading XYZ gradients')
    
    do Ncoords = 1, NgradXYZ(ii)
      read(ygradxyzunit,*,iostat=Error) YgradXYZ(1:3,Ncoords,ii)
      if (Error /= 0) call stopMLatomF('Error while reading XYZ gradients')
    end do
  end do

  close(ygradxyzunit)
  
end subroutine readYgradXYZ
!==============================================================================

function sym2at(sym)
  use strings, only : strUpper
  implicit none
  ! Argument
  character(len=*) :: sym
  ! Return value
  integer :: sym2at
  
  sym2at = 0
  
  select case (strUpper(sym))
    case ('X')
      sym2at = 0
    case ('H')
      sym2at = 1
    case ('HE')
      sym2at = 2
    case ('LI')
      sym2at = 3
    case ('BE')
      sym2at = 4
    case ('B')
      sym2at = 5
    case ('C')
      sym2at = 6
    case ('N')
      sym2at = 7
    case ('O')
      sym2at = 8
    case ('F')
      sym2at = 9
    case ('NE')
      sym2at = 10
    case ('NA')
      sym2at = 11
    case ('MG')
      sym2at = 12
    case ('AL')
      sym2at = 13
    case ('SI')
      sym2at = 14
    case ('P')
      sym2at = 15
    case ('S')
      sym2at = 16
    case ('CL')
      sym2at = 17
    case ('AR')
      sym2at = 18
    case ('K')
      sym2at = 19
    case ('CA')
      sym2at = 20
    case ('SC')
      sym2at = 21
    case ('TI')
      sym2at = 22
    case ('V')
      sym2at = 23
    case ('CR')
      sym2at = 24
    case ('MN')
      sym2at = 25
    case ('FE')
      sym2at = 26
    case ('CO')
      sym2at = 27
    case ('NI')
      sym2at = 28
    case ('CU')
      sym2at = 29
    case ('ZN')
      sym2at = 30
    case ('GA')
      sym2at = 31
    case ('GE')
      sym2at = 32
    case ('AS')
      sym2at = 33
    case ('SE')
      sym2at = 34
    case ('BR')
      sym2at = 35
    case ('KR')
      sym2at = 36
    case ('RB')
      sym2at = 37
    case ('SR')
      sym2at = 38
    case ('Y')
      sym2at = 39
    case ('ZR')
      sym2at = 40
    case ('NB')
      sym2at = 41
    case ('MO')
      sym2at = 42
    case ('TC')
      sym2at = 43
    case ('RU')
      sym2at = 44
    case ('RH')
      sym2at = 45
    case ('PD')
      sym2at = 46
    case ('AG')
      sym2at = 47
    case ('CD')
      sym2at = 48
    case ('IN')
      sym2at = 49
    case ('SN')
      sym2at = 50
    case ('SB')
      sym2at = 51
    case ('TE')
      sym2at = 52
    case ('I')
      sym2at = 53
    case ('XE')
      sym2at = 54
    case ('CS')
      sym2at = 55
    case ('BA')
      sym2at = 56
    case ('LA')
      sym2at = 57
    case ('CE')
      sym2at = 58
    case ('PR')
      sym2at = 59
    case ('ND')
      sym2at = 60
    case ('PM')
      sym2at = 61
    case ('SM')
      sym2at = 62
    case ('EU')
      sym2at = 63
    case ('GD')
      sym2at = 64
    case ('TB')
      sym2at = 65
    case ('DY')
      sym2at = 66
    case ('HO')
      sym2at = 67
    case ('ER')
      sym2at = 68
    case ('TM')
      sym2at = 69
    case ('YB')
      sym2at = 70
    case ('LU')
      sym2at = 71
    case ('HF')
      sym2at = 72
    case ('TA')
      sym2at = 73
    case ('W')
      sym2at = 74
    case ('RE')
      sym2at = 75
    case ('OS')
      sym2at = 76
    case ('IR')
      sym2at = 77
    case ('PT')
      sym2at = 78
    case ('AU')
      sym2at = 79
    case ('HG')
      sym2at = 80
    case ('TL')
      sym2at = 81
    case ('PB')
      sym2at = 82
    case ('BI')
      sym2at = 83
    case ('PO')
      sym2at = 84
    case ('AT')
      sym2at = 85
    case ('RN')
      sym2at = 86
    case ('FR')
      sym2at = 87
    case ('RA')
      sym2at = 88
    case ('AC')
      sym2at = 89
    case ('TH')
      sym2at = 90
    case ('PA')
      sym2at = 91
    case ('U')
      sym2at = 92
    case ('NP')
      sym2at = 93
    case ('PU')
      sym2at = 94
    case ('AM')
      sym2at = 95
    case ('CM')
      sym2at = 96
    case ('BK')
      sym2at = 97
    case ('CF')
      sym2at = 98
    case ('ES')
      sym2at = 99
    case ('FM')
      sym2at = 100
    case ('MD')
      sym2at = 101
    case ('NO')
      sym2at = 102
    case ('LR')
      sym2at = 103
    case ('RF')
      sym2at = 104
    case ('DB')
      sym2at = 105
    case ('SG')
      sym2at = 106
    case ('BH')
      sym2at = 107
    case ('HS')
      sym2at = 108
    case ('MT')
      sym2at = 109
    case ('DS')
      sym2at = 110
    case ('RG')
      sym2at = 111
    case ('CN')
      sym2at = 112
    case ('UUT')
      sym2at = 113
    case ('FL')
      sym2at = 114
    case ('UUP')
      sym2at = 115
    case ('LV')
      sym2at = 116
    case ('UUS')
      sym2at = 117
    case ('UUO')
      sym2at = 118
    case default
      call stopMLatomF('Unknown element ' // sym)
  end select
  
end function

function at2sym(at)
  implicit none
  ! Argument
  integer :: at  
  ! Return value
  character(len=3) :: at2sym
  
  at2sym = ''
  
  select case (at)
    case (0)
      at2sym = 'X  '
    case (1)
      at2sym = 'H  '
    case (2)
      at2sym = 'He '
    case (3)
      at2sym = 'Li '
    case (4)
      at2sym = 'Be '
    case (5)
      at2sym = 'B  '
    case (6)
      at2sym = 'C  '
    case (7)
      at2sym = 'N  '
    case (8)
      at2sym = 'O  '
    case (9)
      at2sym = 'F  '
    case (10)
      at2sym = 'Ne '
    case (11)
      at2sym = 'Na '
    case (12)
      at2sym = 'Mg '
    case (13)
      at2sym = 'Al '
    case (14)
      at2sym = 'Si '
    case (15)
      at2sym = 'P  '
    case (16)
      at2sym = 'S  '
    case (17)
      at2sym = 'Cl '
    case (18)
      at2sym = 'Ar '
    case (19)
      at2sym = 'K  '
    case (20)
      at2sym = 'Ca '
    case (21)
      at2sym = 'Sc '
    case (22)
      at2sym = 'Ti '
    case (23)
      at2sym = 'V  '
    case (24)
      at2sym = 'Cr '
    case (25)
      at2sym = 'Mn '
    case (26)
      at2sym = 'Fe '
    case (27)
      at2sym = 'Co '
    case (28)
      at2sym = 'Ni '
    case (29)
      at2sym = 'Cu '
    case (30)
      at2sym = 'Zn '
    case (31)
      at2sym = 'Ga '
    case (32)
      at2sym = 'Ge '
    case (33)
      at2sym = 'As '
    case (34)
      at2sym = 'Se '
    case (35)
      at2sym = 'Br '
    case (36)
      at2sym = 'Kr '
    case (37)
      at2sym = 'Rb '
    case (38)
      at2sym = 'Sr '
    case (39)
      at2sym = 'Y  '
    case (40)
      at2sym = 'Zr '
    case (41)
      at2sym = 'Nb '
    case (42)
      at2sym = 'Mo '
    case (43)
      at2sym = 'Tc '
    case (44)
      at2sym = 'Ru '
    case (45)
      at2sym = 'Rh '
    case (46)
      at2sym = 'Pd '
    case (47)
      at2sym = 'Ag '
    case (48)
      at2sym = 'Cd '
    case (49)
      at2sym = 'In '
    case (50)
      at2sym = 'Sn '
    case (51)
      at2sym = 'Sb '
    case (52)
      at2sym = 'Te '
    case (53)
      at2sym = 'I  '
    case (54)
      at2sym = 'Xe '
    case (55)
      at2sym = 'Cs '
    case (56)
      at2sym = 'Ba '
    case (57)
      at2sym = 'La '
    case (58)
      at2sym = 'Ce '
    case (59)
      at2sym = 'Pr '
    case (60)
      at2sym = 'Nd '
    case (61)
      at2sym = 'Pm '
    case (62)
      at2sym = 'Sm '
    case (63)
      at2sym = 'Eu '
    case (64)
      at2sym = 'Gd '
    case (65)
      at2sym = 'Tb '
    case (66)
      at2sym = 'Dy '
    case (67)
      at2sym = 'Ho '
    case (68)
      at2sym = 'Er '
    case (69)
      at2sym = 'Tm '
    case (70)
      at2sym = 'Yb '
    case (71)
      at2sym = 'Lu '
    case (72)
      at2sym = 'Hf '
    case (73)
      at2sym = 'Ta '
    case (74)
      at2sym = 'W  '
    case (75)
      at2sym = 'Re '
    case (76)
      at2sym = 'Os '
    case (77)
      at2sym = 'Ir '
    case (78)
      at2sym = 'Pt '
    case (79)
      at2sym = 'Au '
    case (80)
      at2sym = 'Hg '
    case (81)
      at2sym = 'Tl '
    case (82)
      at2sym = 'Pb '
    case (83)
      at2sym = 'Bi '
    case (84)
      at2sym = 'Po '
    case (85)
      at2sym = 'At '
    case (86)
      at2sym = 'Rn '
    case (87)
      at2sym = 'Fr '
    case (88)
      at2sym = 'Ra '
    case (89)
      at2sym = 'Ac '
    case (90)
      at2sym = 'Th '
    case (91)
      at2sym = 'Pa '
    case (92)
      at2sym = 'U  '
    case (93)
      at2sym = 'Np '
    case (94)
      at2sym = 'Pu '
    case (95)
      at2sym = 'Am '
    case (96)
      at2sym = 'Cm '
    case (97)
      at2sym = 'Bk '
    case (98)
      at2sym = 'Cf '
    case (99)
      at2sym = 'Es '
    case (100)
      at2sym = 'Fm '
    case (101)
      at2sym = 'Md '
    case (102)
      at2sym = 'No '
    case (103)
      at2sym = 'Lr '
    case (104)
      at2sym = 'Rf '
    case (105)
      at2sym = 'Db '
    case (106)
      at2sym = 'Sg '
    case (107)
      at2sym = 'Bh '
    case (108)
      at2sym = 'Hs '
    case (109)
      at2sym = 'Mt '
    case (110)
      at2sym = 'Ds '
    case (111)
      at2sym = 'Rg '
    case (112)
      at2sym = 'Cn '
    case (113)
      at2sym = 'Uut'
    case (114)
      at2sym = 'Fl '
    case (115)
      at2sym = 'Uup'
    case (116)
      at2sym = 'Lv '
    case (117)
      at2sym = 'Uus'
    case (118)
      at2sym = 'Uuo'
    case default
      write(at2sym,'(A3)') at
      call stopMLatomF('Unknown element with atomic number ' // at2sym)
  end select
  
end function

!==============================================================================
subroutine writeX(filename)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  integer :: xunit
  integer :: Nspecies, i
  integer :: error
  
  xunit=30
  
  open(xunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
  do Nspecies = 1, NitemsTot
    do i = 1, XvecSize - 1
      write(xunit,'(F20.12,A1)',advance='no') X(i,Nspecies), ' '
    end do
    write(xunit,'(F20.12)',advance='yes') X(XvecSize,Nspecies)
  end do
  
  close(xunit)

end subroutine writeX
!==============================================================================

!==============================================================================
subroutine writeEst(Nprint, indicesPrint)
!==============================================================================
  implicit none
  ! Argument
  integer, intent(in) :: Nprint                 ! Number  of points to write in the file
  integer, intent(in) :: indicesPrint(1:Nprint) ! Indices of points to write in the file
  
  if (trim(option%YestFile) /= '')        call writeYest(       option%YestFile,        Nprint, indicesPrint)
  if (trim(option%YgradEstFile) /= '')    call writeYestGrad(   option%YgradEstFile,    Nprint, indicesPrint)
  if (trim(option%YgradXYZestFile) /= '') call writeYgradXYZest(option%YgradXYZestFile, Nprint, indicesPrint)

end subroutine writeEst
!==============================================================================

!==============================================================================
subroutine writeYest(filename, Nprint, indicesPrint)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  integer,          intent(in) :: Nprint                 ! Number  of points to write in the file
  integer,          intent(in) :: indicesPrint(1:Nprint) ! Indices of points to write in the file
  ! Local variables
  integer :: yestunit
  integer :: Nspecies
  integer :: error
  
  yestunit = 31
  
  open(yestunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
    do Nspecies = 1, Nprint
      write(yestunit,'(F25.13)') Yest(indicesPrint(Nspecies))
    end do
  
  close(yestunit)

end subroutine writeYest
!==============================================================================

!==============================================================================
subroutine writeYestGrad(filename, Nprint, indicesPrint)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  integer,          intent(in) :: Nprint                 ! Number  of points to write in the file
  integer,          intent(in) :: indicesPrint(1:Nprint) ! Indices of points to write in the file
  ! Local variables
  integer :: yestunit, i, Nspecies, error
  
  yestunit = 32
  
  open(yestunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
  do Nspecies = 1, Nprint
    do i = 1, XvecSize - 1
      write(yestunit,'(F25.13)',advance='no') YestGrad(i,indicesPrint(Nspecies))
    end do
    write(yestunit,'(F25.13)',advance='yes') YestGrad(i,indicesPrint(Nspecies))
  end do
  
  close(yestunit)

end subroutine writeYestGrad
!==============================================================================

!==============================================================================
subroutine writeYgradXYZest(filename, Nprint, indicesPrint)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  integer,          intent(in) :: Nprint                 ! Number  of points to write in the file
  integer,          intent(in) :: indicesPrint(1:Nprint) ! Indices of points to write in the file
  ! Local variables
  integer :: yestunit, iatom, icoord, Nspecies, error
  
  yestunit = 33
  
  open(yestunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
  if (allocated(Natoms)) then
    do Nspecies = 1, Nprint
      write(yestunit,'(I0,/)',advance='yes') Natoms(indicesPrint(Nspecies))
      do iatom = 1, Natoms(indicesPrint(Nspecies))
        write(yestunit,'(3F25.13)',advance='yes') (YgradXYZest(icoord,iatom,indicesPrint(Nspecies)),icoord=1,3)
      end do
    end do
  elseif (allocated(NgradXYZ)) then
    do Nspecies = 1, Nprint
      write(yestunit,'(I0,/)',advance='yes') NgradXYZ(indicesPrint(Nspecies))
      do iatom = 1, NgradXYZ(indicesPrint(Nspecies))
        write(yestunit,'(3F25.13)',advance='yes') (YgradXYZest(icoord,iatom,indicesPrint(Nspecies)),icoord=1,3)
      end do
    end do
  else
    do Nspecies = 1, Nprint
      write(yestunit,'(I0,/)',advance='yes') NgradXYZmax
      do iatom = 1, NgradXYZmax
        write(yestunit,'(3F25.13)',advance='yes') (YgradXYZest(icoord,iatom,indicesPrint(Nspecies)),icoord=1,3)
      end do
    end do
  end if
  
  close(yestunit)

end subroutine writeYgradXYZest
!==============================================================================

!==============================================================================
subroutine writeXYZsorted(filename)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  ! Local variables
  integer :: yestunit, iatom, icoord, Nspecies, error
  
  yestunit = 34
  
  open(yestunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
  do Nspecies = 1, NitemsTot
    write(yestunit,'(I0,/)',advance='yes') Natoms(Nspecies)
    do iatom = 1, Natoms(Nspecies)
      !write(yestunit,'(A3,3F25.13)',advance='yes') at2sym(Z(iatom,Nspecies)),(XYZ_A_sorted(icoord,iatom,Nspecies),icoord=1,3)
      write(yestunit,'(A3,F25.15,2F26.15)',advance='yes') at2sym(Z(iatom,Nspecies)),(XYZ_A_sorted(icoord,iatom,Nspecies),icoord=1,3)
    end do
  end do
  
  close(yestunit)

end subroutine writeXYZsorted
!==============================================================================

!==============================================================================
subroutine writeIndices(filename, Nprint, indicesPrint)
!==============================================================================
  implicit none
  ! Argument
  character(len=*), intent(in) :: filename
  integer,          intent(in) :: Nprint                 ! Number  of points to write in the file
  integer,          intent(in) :: indicesPrint(1:Nprint) ! Indices of points to write in the file
  ! Local variables
  integer :: indunit, i, error
  
  indunit = 41
  
  open(indunit,file=trim(filename),action='write',status='replace',iostat=error)
  if (error /= 0) call stopMLatomF('Unable to create file ' // trim(filename))
  
  do i = 1, Nprint
    write(indunit,'(I0)') indicesPrint(i)
  end do
  
  close(indunit)

end subroutine writeIndices
!==============================================================================

subroutine readIndices(filename, Nread, indicesRead)
  implicit none
  ! Argument
  character(len=*), intent(in)    :: filename
  integer,          intent(in)    :: Nread                 ! Number  of points to write in the file
  integer,          intent(inout) :: indicesRead(1:Nread) ! Indices of points to write in the file
  ! Local variables
  integer :: iunit, Error, i, Nlines

  ! Initialize
  Error     = 0
  Nlines = 0

  iunit = 27
  open(iunit,file=trim(filename),action='read',IOStat=Error)
  if (error /= 0) call stopMLatomF('Failed to open file ' // trim(filename))

  ! Find number of entries  
  do while (.true.)
    read(iunit,*,iostat=error)
    if (error /= 0) exit
    Nlines = Nlines + 1
  end do
  
  ! Check if we have enough data
  if     (Nread > Nlines   ) then
    call stopMLatomF('File ' // trim(filename) // ' contains less data than requested')
  elseif (Nlines    > Nread) then
    write (6,'(a,I0,a)') ' <!> File ' // trim(filename) // ' contains ', Nlines, ' indices'
    write (6,'(a,I0,a)') '     but only first ', Nread, ' will be used'
    write (6,'(a)')      ''
  end if
    
  ! Rewind file
  rewind(iunit)
  
  ! Initialize
  indicesRead = 0

  do i=1,Nread
    read(iunit,*,IOStat=Error) indicesRead(i)
    if (Error /= 0) call stopMLatomF('Error while reading reference data')
  end do

  close(iunit)

end subroutine readIndices

!==============================================================================
subroutine getPermInvData()
!==============================================================================
! Parse option string into machine-readable integer arrays
!==============================================================================
  use mathUtils, only : factorial_rprec, permutations, choices
  use strings,   only : splitString
  implicit none
  ! Arrays
  character(len=len(trim(option%permInvGroups))), allocatable :: strGroupTypes(:)
  character(len=max(len(trim(option%permInvGroups)),len(trim(option%permInvNuclei)))), allocatable :: strNuclGroups(:)
  character(len=max(len(trim(option%permInvGroups)),len(trim(option%permInvNuclei)))), allocatable :: strNuclei(:)
  character(len=256) :: tempstr
  ! Local arguments
  integer :: ndots, nminuses, ncommas, ngroups, nsets, Natoms2perm, NallAtoms2perm
  integer :: ii, jj, kk, ll, igroup, iset, iperm, istart, iend, Error
  integer,             allocatable :: nElmntsArr(:), iAts(:), iAtstmp(:)
  type(arrayOfArrays), allocatable :: permIatomsGroups(:) ! Indices of permuted atoms
  type(arrayOfArrays), allocatable :: permIatomsNucl(:) ! Indices of permuted atoms
  type(nuclGroups),    allocatable :: permElmnts(:)
  
  if (trim(option%Nperm) /= '') then
    read(option%Nperm,*) Nperm
    return
  end if

  if (option%usePermInd) then 
    if (.not. option%readModPermInd .and. option%permIndIn /= '') then 
      Nperm = 0
      write(*,*) trim(option%permIndIn)
      open(60,file=trim(option%permIndIn),action='read',iostat=Error)
      if (Error/=0) call stopMLatomF('Failed to open file ' // trim(option%permIndIn))

      do while(.True.)
        read(60,*,iostat=Error) tempstr
        if (Error/=0) exit 
        Nperm = Nperm + 1
      end do 
      close(60)

    end if 

  else 
    NpermGroups = 1
    NpermNucl   = 1

    ! Get permutationally invariant groups
    if (trim(option%permInvGroups) /= '') then
      ! Get types of groups
      ndots = COUNT([(option%permInvGroups(ii:ii),ii=1,len(option%permInvGroups))] == '.')
      allocate(permInvGroups(ndots+1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate permInvGroups')
      allocate(strGroupTypes(ndots+1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate strGroupTypes')    
      call splitString(trim(option%permInvGroups), '.', ndots, strGroupTypes)
      NpermGroups = NpermGroups * int(factorial_rprec(ndots+1))
      ! Get groups
      do ll=1,ndots+1
        nminuses = COUNT([(strGroupTypes(ll)(ii:ii),ii=1,len(strGroupTypes(ll)))] == '-')
        allocate(permInvGroups(ll)%groups(nminuses+1),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate permInvGroups(ll)%groups')
        allocate(strNuclGroups(nminuses+1),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate strNuclGroups')    
        call splitString(trim(strGroupTypes(ll)), '-', nminuses, strNuclGroups)
        NpermGroups = NpermGroups * int(factorial_rprec(nminuses+1))
        ! Get nuclei
        do ii=1,nminuses+1
          ncommas = COUNT([(strNuclGroups(ii)(jj:jj),jj=1,len(strNuclGroups(ii)))] == ',')
          allocate(permInvGroups(ll)%groups(ii)%oneDintArr(ncommas+1),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate permInvGroups(ll)%groups(ii)%oneDintArr')
          allocate(strNuclei(ncommas+1),stat=Error)
          if(Error/=0)call stopMLatomF('Unable to allocate strNuclei')    
          call splitString(trim(strNuclGroups(ii)), ',', ncommas, strNuclei)
          do kk=1,ncommas+1
            read(strNuclei(kk), *) permInvGroups(ll)%groups(ii)%oneDintArr(kk)
          end do
          deallocate(strNuclei)
        end do
        deallocate(strNuclGroups)
      end do
      deallocate(strGroupTypes)
    end if

    ! Get permutationally invariant nuclei within the above groups
    if (trim(option%permInvNuclei) /= '') then
      ! Get groups
      ndots = COUNT([(option%permInvNuclei(ii:ii),ii=1,len(option%permInvNuclei))] == '.')
      allocate(permInvNuclei(ndots+1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate permInvNuclei')
      allocate(strNuclGroups(ndots+1),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate strNuclGroups')    
      call splitString(trim(option%permInvNuclei), '.', ndots, strNuclGroups)
      ! Get nuclei
      do ii=1,ndots+1
        nminuses = COUNT([(strNuclGroups(ii)(jj:jj),jj=1,len(strNuclGroups(ii)))] == '-')
        allocate(permInvNuclei(ii)%oneDintArr(nminuses+1),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate permInvNuclei(ii)%oneDintArr')
        allocate(strNuclei(nminuses+1),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate strNuclei')    
        call splitString(trim(strNuclGroups(ii)), '-', nminuses, strNuclei)
        NpermNucl = NpermNucl * int(factorial_rprec(nminuses+1))
        do kk=1,nminuses+1
          read(strNuclei(kk), *) permInvNuclei(ii)%oneDintArr(kk)
        end do
        deallocate(strNuclei)
      end do
      deallocate(strNuclGroups)
    end if
    Nperm = NpermGroups * NpermNucl
  end if 
  
  write(6,'(a,i0)') ' Number of permutations: ', Nperm

end subroutine getPermInvData
!==============================================================================

function distance(Xsize,Xi,Xj)
  implicit none
  real(kind=rprec)             :: distance
  integer,          intent(in) :: Xsize
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  ! Local variables
  integer :: kk

  distance = 0.0_rprec

  if (trim(option%kernel) == 'linear') then
    do kk=1,Xsize
      distance = distance + Xi(kk)*Xj(kk)
    end do
  elseif (trim(option%kernel) == 'polynomial') then
    do kk=1,Xsize
      distance = distance + Xi(kk)*Xj(kk)
    end do
  elseif (trim(option%kernel) == 'Gaussian') then
    do kk=1,Xsize
      distance = distance + (Xi(kk) - Xj(kk)) ** 2
    end do
  elseif (trim(option%kernel) == 'Laplacian') then
    do kk=1,Xsize
      distance = distance + abs(Xi(kk) - Xj(kk))
    end do
  elseif (trim(option%kernel) == 'exponential') then
    do kk=1,Xsize
      distance = distance + (Xi(kk) - Xj(kk)) ** 2
    end do
    distance = sqrt(distance)
  elseif (trim(option%kernel) == 'Matern') then
    do kk=1,Xsize
      distance = distance + (Xi(kk) - Xj(kk)) ** 2
    end do
    distance = sqrt(distance)
  end if

end function distance

!==============================================================================
subroutine cleanUp_dataset()
!==============================================================================
  implicit none
  integer :: i, j

  ! Free up memory
  if(allocated(indicesItemsTot))                             deallocate(indicesItemsTot)
  if(allocated(indicesPredict))                              deallocate(indicesPredict)
  if(allocated(Natoms))                                      deallocate(Natoms)
  if(allocated(NgradXYZ))                                    deallocate(NgradXYZ)
  if(allocated(X))                                           deallocate(X)
  if(allocated(Y))                                           deallocate(Y)
  if(allocated(Ygrad))                                       deallocate(Ygrad)
  if(allocated(YgradXYZ))                                    deallocate(YgradXYZ)
  if(allocated(Z))                                           deallocate(Z)
  if(allocated(Yest))                                        deallocate(Yest)
  if(allocated(YestGrad))                                    deallocate(YestGrad)
  if(allocated(YgradXYZest))                                 deallocate(YgradXYZest)
  if(allocated(YestCVtest))                                  deallocate(YestCVtest)
  if(allocated(YestCVopt))                                   deallocate(YestCVopt)
  if(allocated(XYZ_A))                                       deallocate(XYZ_A)
  if(allocated(XYZ_B))                                       deallocate(XYZ_B)
  if(allocated(XYZ_A_sorted))                                deallocate(XYZ_A_sorted)
  if(allocated(XYZeq))                                       deallocate(XYZeq)
  if(allocated(Zeq))                                         deallocate(Zeq)
  if(allocated(Req))                                         deallocate(Req)
  if(allocated(ReqMod))                                      deallocate(ReqMod)
  if(allocated(trainTestData%indicesTrain))                  deallocate(trainTestData%indicesTrain)
  if(allocated(trainTestData%indicesTest))                   deallocate(trainTestData%indicesTest)
  if(allocated(trainTestData%indicesCVoptSplits))            deallocate(trainTestData%indicesCVoptSplits)
  if(allocated(trainTestData%hyperOptData%indicesSubtrain))  deallocate(trainTestData%hyperOptData%indicesSubtrain)
  if(allocated(trainTestData%hyperOptData%indicesValidate))  deallocate(trainTestData%hyperOptData%indicesValidate)
  if(allocated(CVtestFolds))                                 deallocate(CVtestFolds)
  if (allocated(indicesCVtestSplits)) then
    do i = 1, size(indicesCVtestSplits)
      if(allocated(indicesCVtestSplits(i)%oneDintArr))       deallocate(indicesCVtestSplits(i)%oneDintArr)
    end do
                                                             deallocate(indicesCVtestSplits)
  end if
  if(allocated(hyperOptData%indicesSubtrain))                deallocate(hyperOptData%indicesSubtrain)
  if(allocated(hyperOptData%indicesValidate))                deallocate(hyperOptData%indicesValidate)
  if (allocated(CVoptFolds))          then
    do i = 1, size(CVoptFolds)
      if(allocated(CVoptFolds(i)%indicesSubtrain))           deallocate(CVoptFolds(i)%indicesSubtrain)
      if(allocated(CVoptFolds(i)%indicesValidate))           deallocate(CVoptFolds(i)%indicesValidate)
    end do
                                                             deallocate(CVoptFolds)
  end if
  if (allocated(permInvGroups))       then
    do i = 1, size(permInvGroups)
      do j = 1, size(permInvGroups(i)%groups)
        if(allocated(permInvGroups(i)%groups(j)%oneDintArr)) deallocate(permInvGroups(i)%groups(j)%oneDintArr)
      end do
      if(allocated(permInvGroups(i)%groups))                 deallocate(permInvGroups(i)%groups)
    end do
                                                             deallocate(permInvGroups)
  end if  
  if (allocated(permInvNuclei))       then
    do i = 1, size(permInvNuclei)
      if(allocated(permInvNuclei(i)%oneDintArr))             deallocate(permInvNuclei(i)%oneDintArr)
    end do
                                                             deallocate(permInvNuclei)
  end if
  if (allocated(permutedIatoms))      then
    do i = 1, size(permutedIatoms)
      if(allocated(permutedIatoms(i)%oneDintArr))            deallocate(permutedIatoms(i)%oneDintArr)
    end do
                                                             deallocate(permutedIatoms)
  end if
  
end subroutine cleanUp_dataset
!==============================================================================

end module dataset
