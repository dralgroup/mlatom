
  !---------------------------------------------------------------------------! 
  ! strings: routines for dealing with strings                                ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

module strings
  use stopper, only : stopMLatomF
  implicit none

contains

function strUpper(str)
  implicit none
  ! Arguments
  character(len=*), intent(in) :: str
  ! Return value
  character(len=len(str)) :: strUpper
  ! Variables
  integer :: i

  do i=1,len(str)
    if(str(i:i) >= "a" .and. str(i:i) <= "z") then
      strUpper(i:i)=achar(iachar(str(i:i))-32)
    else
      strUpper(i:i)=str(i:i)
    endif
  end do
  
end function strUpper

!==============================================================================
subroutine splitString(string, delimiter, Ndelims, list)
!==============================================================================
! Split a string into a list given a delimiter
! Example:
!   Input
!     string    = '1,2,3'
!     delimiter = ','
!     Ndelims   = 2
!   Output
!     list      = ['1','2','3']
!==============================================================================
  implicit none
  ! Arguments
  character(len=*), intent(in)    :: string            ! String
  character(len=*), intent(in)    :: delimiter         ! Delimiter
  integer,          intent(in)    :: Ndelims           ! Number of delimiters in string
  character(len=*), intent(inout) :: list(1:Ndelims+1) ! Array with sub-strings split form string
  ! Arrays
  integer, allocatable :: iDelims(:) ! Indices of delimiters
  ! Local arguments
  integer :: nn, ii, Error
  
  ! Initialization
  list = ''
  
  if (Ndelims == 0) then
    list(1) = string
  else
    ! Split the string
    allocate(iDelims(Ndelims),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate iDelims')
    nn = 0
    do ii=1,len(string)
      if (string(ii:ii+len(delimiter)-1) == delimiter) then
        nn = nn + 1
        iDelims(nn) = ii
      end if
    end do
    
    nn = 1
    list(nn) = string(1:iDelims(nn)-1)
    do ii=1,Ndelims-1
      nn = nn + 1
      list(nn) = string(iDelims(ii)+len(delimiter):iDelims(ii+1)-1)
    end do
    nn = nn + 1
    list(nn) = string(iDelims(Ndelims)+len(delimiter):)
  end if
  
  if(allocated(iDelims)) deallocate(iDelims)
  
end subroutine splitString
!==============================================================================

end module strings
