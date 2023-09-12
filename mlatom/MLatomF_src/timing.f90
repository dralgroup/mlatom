
  !---------------------------------------------------------------------------! 
  ! timing: routines for time measurement                                     ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!
  
module timing
  use precision, only : rprec
  implicit none

contains

subroutine timeTaken(startTime,msg)
  implicit none
  ! Arguments
  integer*8, intent(in)        :: startTime(1:8)
  character(len=*), intent(in) :: msg
  ! Local variables
  integer*8                    :: secDiff  ! Difference in seconds
  real(kind=rprec)             :: timeDiff ! Total difference in seconds
  ! Arrays
  integer*8                    :: currentDate(1:8)

  call date_and_time(values=currentDate)

  if (currentDate(2) == startTime(2)) then
    secDiff = &
            + (currentDate(3) - startTime(3)) * 86400 & ! Difference in the days of the month
            + (currentDate(5) - startTime(5)) * 3600 &  ! Difference in the hours of the day
            + (currentDate(6) - startTime(6))  * 60 &   ! Difference in the minutes of the hour
            + (currentDate(7) - startTime(7))           ! Difference in the seconds of the minute
    timeDiff = dble(secDiff) + dble(currentDate(8) - startTime(8))/1000.0_rprec

    if (len_trim(msg) /= 0) then
      write(6,'(1x,a,1x,F10.2," sec")') msg, timeDiff
    end if
  else
    if (len_trim(msg) /= 0) then
      write(6,'(1x,a,1x,2(i2.2,"."),i4.4," at ",2(i2.2,":"),i2.2)') &
       msg // ' terminated on ',currentDate(3),currentDate(2),currentDate(1),currentDate(5),currentDate(6),currentDate(7)
      write(6,'(1x,a,1x,2(i2.2,"."),i4.4," at ",2(i2.2,":"),i2.2)') &
       msg // ' started on ',startTime(3),startTime(2),startTime(1),startTime(5),startTime(6),startTime(7)
    end if  
  end if

end subroutine timeTaken

end module timing

