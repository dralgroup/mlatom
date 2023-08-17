
  !---------------------------------------------------------------------------! 
  ! sampling: sampling operations                                             ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

module sampling
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  implicit none

contains

!==============================================================================
subroutine sample()
!==============================================================================
! Sample data points from a data set
!==============================================================================
  use dataset, only : getTotalTrainTest, NitemsTot, indicesItemsTot
  use dataset, only : splitAndSample, indicesCVtestSplits, getSubtrainValidate
  use dataset, only : CVtestFolds, Ntrain, Ntest, indicesTrain, indicesTest
  use dataset, only : hyperOptData, indicesCVoptSplits, trainTestData
  implicit none
  integer :: mm, nn, stsize, newsize, nnsize, Error
  character(len=256) :: stmp, CVtestPref, CVoptPref, CVoptPrefGrad
  character(len=5) :: IOiFileTest, IOiFileOpt, IOiFileOptGrad
  
  ! Get number of entries to be used for training and test sets
  call getTotalTrainTest()
  
  ! Some preparations for files with indices
  CVtestPref  = ''
  IOiFileTest = 'no'
  if (trim(option%sampling) == 'user-defined' .and. trim(option%iCVtestPrefIn) /= '') then
    CVtestPref = option%iCVtestPrefIn
    IOiFileTest = 'read'
  end if
  if (trim(option%iCVtestPrefOut) /= '') then
    CVtestPref = option%iCVtestPrefOut
    IOiFileTest = 'write'
  end if
  
  CVoptPref  = ''
  IOiFileOpt = 'no'
  if (trim(option%sampling) == 'user-defined' .and. trim(option%iCVoptPrefIn) /= '') then
    CVoptPref = option%iCVoptPrefIn
    IOiFileOpt = 'read'
  end if
  if (trim(option%iCVoptPrefOut) /= '') then
    CVoptPref = option%iCVoptPrefOut
    IOiFileOpt = 'write'
  end if
  
  CVoptPrefGrad = '' 
  IOiFileOptGrad = 'no' 
  if (trim(option%sampling) == 'user-defined' .and. trim(option%iCVoptPrefGradIn) /= '') then
    CVoptPrefGrad = option%iCVoptPrefIn
    IOiFileOpt = 'read'
  end if

  ! Sample depending on task
  if (option%CVtest) then
    if (option%LOOtest) option%NcvTestFolds = NitemsTot
    allocate(indicesCVtestSplits(1:option%NcvTestFolds),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate indicesCVtestSplits')
    allocate(CVtestFolds(1:option%NcvTestFolds),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate CVtestFolds')
    call splitAndSample(NitemsTot, indicesItemsTot, option%NcvTestFolds, indicesCVtestSplits, trim(CVtestPref), IOiFileTest)
    do mm = 1, option%NcvTestFolds
      Ntrain => CVtestFolds(mm)%Ntrain
      Ntest  => CVtestFolds(mm)%Ntest
      Ntest  = size(indicesCVtestSplits(mm)%oneDintArr)
      Ntrain = NitemsTot - Ntest
      allocate(CVtestFolds(mm)%indicesTest(1:Ntest),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate indicesTest')
      allocate(CVtestFolds(mm)%indicesTrain(1:Ntrain),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate indicesTrain')
      indicesTrain         => CVtestFolds(mm)%indicesTrain
      indicesTest          => CVtestFolds(mm)%indicesTest
      indicesTest(1:Ntest) = indicesCVtestSplits(mm)%oneDintArr(1:Ntest)
      indicesTrain         = 0
      stsize               = 0
      newsize              = 0
      do nn = 1, option%NcvTestFolds
        if (nn == mm) cycle
        nnsize = size(indicesCVtestSplits(nn)%oneDintArr)
        newsize = stsize + nnsize
        indicesTrain(stsize+1:newsize) = indicesCVtestSplits(nn)%oneDintArr(1:nnsize)
        stsize = newsize
      end do
      
      ! If hyperparameter optimization is requested
      write(stmp, '(i0)') mm
      stmp = trim(CVtestPref) // trim(stmp)
      if (option%CVopt) then
        if (option%LOOopt) option%NcvOptFolds = Ntrain
        stmp = trim(stmp) // trim(CVoptPref) 
        allocate(CVtestFolds(mm)%indicesCVoptSplits(1:option%NcvOptFolds),stat=Error)
        if(Error/=0)call stopMLatomF('Unable to allocate indicesCVoptSplits')
        indicesCVoptSplits => CVtestFolds(mm)%indicesCVoptSplits
        call splitAndSample(Ntrain, indicesTrain, option%NcvOptFolds, indicesCVoptSplits, trim(stmp), IOiFileOpt)
      elseif (option%hyperOpt) then
        ! Get the subtraining and validation sets
        call getSubtrainValidate(CVtestFolds(mm)%hyperOptData, stmp) 
      end if
      
    end do
  else
    ! If hyperparameter optimization is requested
    if (option%CVopt) then
      if (option%LOOopt) option%NcvOptFolds = Ntrain
      allocate(trainTestData%indicesCVoptSplits(1:option%NcvOptFolds),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate indicesCVoptSplits')
      indicesCVoptSplits => trainTestData%indicesCVoptSplits
      call splitAndSample(Ntrain, indicesTrain, option%NcvOptFolds, indicesCVoptSplits, trim(CVoptPref), IOiFileOpt)
    elseif (option%hyperOpt .or. trim(option%iSubtrainOut) /= '' .or. trim(option%iValidateOut) /= '') then 
      ! Get the subtraining and validation sets
      call getSubtrainValidate(hyperOptData, '')
    end if
  end if

end subroutine sample
!==============================================================================

end module sampling
