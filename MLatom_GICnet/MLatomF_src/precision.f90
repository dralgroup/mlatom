
  !---------------------------------------------------------------------------! 
  ! precision: setting precision for calculations with floating point numbers ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

module precision
  implicit none
  integer, parameter :: rprec = selected_real_kind(15,307) !(7,38) !(17,308)

end module precision
