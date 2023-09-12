
  !---------------------------------------------------------------------------! 
  ! MLatomFInfo: header of output files generated with MLatomF                ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------! 

module MLatomFInfo
  implicit none
  
contains

!==============================================================================
subroutine header()
!==============================================================================
  implicit none
  
  write(6,'(a)') ''
  write(6,'(a)') '  !---------------------------------------------------------------------------! '
  write(6,'(a)') '  ! MLatomF: Fortran + OpenMP part of MLatom                                  ! '
  write(6,'(a)') '  ! Implementations by: Pavlo O. Dral and Yi-Fan Hou                          ! '
  write(6,'(a)') '  !---------------------------------------------------------------------------! '
  write(6,'(a)') ''

end subroutine header
!==============================================================================

end module MLatomFInfo
