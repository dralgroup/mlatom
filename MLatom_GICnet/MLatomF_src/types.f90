
  !---------------------------------------------------------------------------! 
  ! types: user-defined types of variables                                    ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

module types
  use precision, only : rprec
  implicit none
  type :: arrayOfArrays
    integer,          allocatable :: oneDintArr(:)
    real(kind=rprec), allocatable :: oneDrArr(:)
    real(kind=rprec), allocatable :: twoDrArr(:,:)
  end type arrayOfArrays
  type :: nuclGroups
    type(arrayOfArrays), allocatable :: groups(:)
  end type nuclGroups

end module types
