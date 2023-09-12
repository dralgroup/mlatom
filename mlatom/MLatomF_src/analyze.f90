
  !---------------------------------------------------------------------------! 
  ! analysis: routines for statistical analysis of the data                   ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module analysis
  implicit none

contains

!==============================================================================
subroutine analyze()
!==============================================================================
! Analyze data sets
!==============================================================================
  use dataset,       only : Y, Ygrad, YgradXYZ, NAtomsMax, NgradXYZ
  use dataset,       only : NitemsTot, XvecSize, NgradXYZmax
  use dataset,       only : Yest, YestGrad, YgradXYZest
  use dataset,       only : readY, readYgrad, readYgradXYZ, cleanUp_dataset
  use MLstatistics,  only : calc_stat_measures
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  implicit none
  integer, allocatable :: indicesAll(:)
  integer              :: ii, Error
  character(len=256)   :: stmp
  
  if (option%Yfile /= '') then
    write(6,'(a/)') ' Analysis for Y values with reference data from file ' // trim(option%Yfile)
    call readY(option%YestFile)
    Yest = Y
    deallocate(Y)
    call readY(option%Yfile)
  end if
  if (option%YgradFile /= '') then
    write(6,'(a/)') ' Analysis for Y gradients with reference data from file ' // trim(option%YgradFile)
    call readYgrad(option%YgradEstFile)
    YestGrad = Ygrad
    deallocate(Ygrad)
    call readYgrad(option%YgradFile)
  end if
  if (option%YgradXYZfile /= '') then
    write(6,'(a/)') ' Analysis for XYZ gradients of Y with reference data from file ' // trim(option%YgradXYZfile)
    call readYgradXYZ(option%YgradXYZestFile)
    YgradXYZest = YgradXYZ
    deallocate(YgradXYZ)
    deallocate(NgradXYZ)
    call readYgradXYZ(option%YgradXYZfile)
  end if
  
  allocate(indicesAll(1:NitemsTot),stat=Error)
  if(Error/=0)call stopMLatomF('Unable to allocate vector indicesAll')
  do ii = 1, NitemsTot
    indicesAll(ii) = ii
  end do
  write(stmp,'(I0," entries in the set")') NitemsTot
  call calc_stat_measures(NitemsTot, indicesAll, comment=trim(stmp))
  write(6,'(a)') ''
  deallocate(indicesAll)
  call cleanUp_dataset()

end subroutine analyze
!==============================================================================

end module analysis

