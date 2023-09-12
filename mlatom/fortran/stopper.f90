module stopper 
  implicit none 

contains

subroutine stopKREG(errorMsg)
  implicit none 
  character(len=*), intent(in) :: errorMsg 

  if (len_trim(errorMsg) /= 0) then 
    write(*,*) ' <!> ' // trim(errorMsg) // ' <!>'
  end if 
  stop
end subroutine stopKREG

subroutine raiseWarning(warningMsg)
  implicit none 
  character(len=*), intent(in) :: warningMsg
  if (len_trim(warningMsg) /= 0) then 
    write(*,*) 'Warning: ' // trim(warningMsg) // ''
  end if 
end subroutine raiseWarning

end module stopper