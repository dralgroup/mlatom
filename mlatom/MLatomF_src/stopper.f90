
  !---------------------------------------------------------------------------! 
  ! stopper: Stopping MLatom and writing an error message                     ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 

module stopper
  implicit none

contains

subroutine stopMLatomF(errorMsg)
  implicit none
  ! Argument
  character(len=*), intent(in) :: errorMsg

  if (len_trim(errorMsg) /= 0) then
    write(6,*) '  <!> ' // trim(errorMsg) // ' <!>'
  end if
  stop

end subroutine stopMLatomF

end module stopper
