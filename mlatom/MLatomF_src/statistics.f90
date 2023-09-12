
  !---------------------------------------------------------------------------! 
  ! MLstatistics: math routines for statistical analysis                      ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !---------------------------------------------------------------------------!

module MLstatistics
  use dataset,   only : NitemsTot, Y, Yest, NgradXYZ
  use dataset,   only : XvecSize, Ygrad, YestGrad, NAtomsMax
  use dataset,   only : NgradXYZmax, YgradXYZ, YgradXYZest
  use precision, only : rprec
  implicit none
  real(kind=rprec) :: MAEvalue       = 0.0_rprec ! mean absolute error
  real(kind=rprec) :: MSEvalue       = 0.0_rprec ! mean signed   error
  real(kind=rprec) :: RMSEvalue      = 0.0_rprec ! root-mean-square error
  real(kind=rprec) :: YestMEANvalue  = 0.0_rprec ! mean values
  real(kind=rprec) :: YMEANvalue     = 0.0_rprec ! mean values
  real(kind=rprec) :: LPOutlier      = 0.0_rprec ! largest positive outlier
  integer          :: LPOutlierIndex = 0         ! index of the largest positive outlier
  real(kind=rprec) :: LNOutlier      = 0.0_rprec ! largest negative outlier
  integer          :: LNOutlierIndex = 0         ! index of the largest negative outlier
  real(kind=rprec) :: CorrCoeff      = 0.0_rprec ! Pearson correlation coefficient
  ! Linear regression (y_est = a + b * y)
  real(kind=rprec) :: aValue         = 0.0_rprec ! a coefficient
  real(kind=rprec) :: bValue         = 0.0_rprec ! b coefficient
  real(kind=rprec) :: r_squared      = 0.0_rprec ! R squared value
  real(kind=rprec) :: SE_a           = 0.0_rprec ! standard error for a
  real(kind=rprec) :: SE_b           = 0.0_rprec ! standard error for b

contains

!==============================================================================
subroutine calc_stat_measures(Nindices, indicesForEval, comment, returnValue)
!==============================================================================
! Subroutine calculates requested measures of errors
!==============================================================================
  use optionsModule, only : option
  use stopper,       only : stopMLatomF
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,          intent(in)            :: Nindices                   ! Size of indicesForEval
  integer,          intent(in)            :: indicesForEval(1:Nindices) ! Indices for evaluation
  character(len=*), intent(in),  optional :: comment                    ! Comment
  real(kind=rprec), intent(out), optional :: returnValue                ! Contains value of the measure
  ! Arrays
  integer*8                     :: dateStart(1:8) ! Time and date, when subroutine starts
  integer,               allocatable :: indicesForEvalLoc(:) ! Local indices for evaluation
  real(kind=rprec), allocatable :: Yref(:)        ! Reference data
  real(kind=rprec), allocatable :: Yhat(:)        ! Estimated values
  ! Local variables
  integer          :: nn, ii, jj, ip, ipl, ind, Error, Nvalues
  real(kind=rprec) :: value, VALUEgeom
  character(len=1) :: stmp1, stmp2 ! Temporary string variables
  
  ! Benchmark
  if(option%benchmark .and. option%debug) call date_and_time(values=dateStart)

  VALUEgeom = 1.0_rprec
  Nvalues = 0
  if (option%Yfile /= '') then
    nn = Nindices
    allocate(Yref(1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate array Yref')
    allocate(Yhat(1:nn),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate array Yhat')
    do ii = 1, Nindices
      Yref(ii) = Y(   indicesForEval(ii))
      Yhat(ii) = Yest(indicesForEval(ii))
    end do
    if (present(comment)) write(6,'(a)') ' Analysis for values'
    call stats(nn, Yref, Yhat, indicesForEval, comment, returnValue)
    if (present(returnValue) .and. .not. option%learnGradXYZ) VALUEgeom = VALUEgeom * returnValue
    Nvalues = Nvalues + 1
    if (present(comment)) then
      if (option%calcLPOutlier) then
        if (LPOutlier > 0.0_rprec) then
          write(6,'(a)') '   largest positive outlier'
          write(6,'("     error = ",F25.13)') LPOutlier
          write(6,'("     index = ",I0)') indicesForEval(LPOutlierIndex)
          write(6,'("     estimated value = ",F25.13)') Yest(indicesForEval(LPOutlierIndex))
          write(6,'("     reference value = ",F25.13)') Y(indicesForEval(LPOutlierIndex))
        else
          write(6,'(a)') '   all errors are non-positive'
        end if
      end if
      if (option%calcLNOutlier) then
        if (LNOutlier < 0.0_rprec) then
          write(6,'(a)') '   largest negative outlier'
          write(6,'("     error = ",F25.13)') LNOutlier
          write(6,'("     index = ",I0)') indicesForEval(LNOutlierIndex)
          write(6,'("     estimated value = ",F25.13)') Yest(indicesForEval(LNOutlierIndex))
          write(6,'("     reference value = ",F25.13)') Y(indicesForEval(LNOutlierIndex))
        else
          write(6,'(a)') '   all errors are non-negative'
        end if
      end if
    end if
    deallocate(Yref)
    deallocate(Yhat)
  end if
  if (option%YgradFile /= '') then
      nn = Nindices * XvecSize
      if (nn == 0) then
        write(6,'(a)') ' <!> No entries to calculate statistics for <!>'
        return
      end if
      allocate(Yref(1:nn),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate array Yref')
      allocate(Yhat(1:nn),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate array Yhat')
      ind = 0
      do ii = 1, XvecSize
        do ip=1, Nindices
          ind = ind + 1
          Yref(ind) = Ygrad(   ii,indicesForEval(ip))
          Yhat(ind) = YestGrad(ii,indicesForEval(ip))
        end do
      end do
    if (present(comment)) write(6,'(a)') ' Analysis for gradients'
    call stats(nn, Yref, Yhat, indicesForEval, comment, returnValue)
    if (present(returnValue)) VALUEgeom = VALUEgeom * returnValue
    Nvalues = Nvalues + 1
    if (present(comment)) then
      if (option%calcLPOutlier) then
        if (LPOutlier > 0.0_rprec) then
          write(6,'(a)') '   largest positive outlier'
          write(6,'("     error = ",F25.13)') LPOutlier
          !write(6,'("     index = ",I0)') LPOutlierIndex!indicesForEval(1+int(LPOutlierIndex/3/NgradXYZmax))
          write(6,'("     estimated value = ",F25.13)') Yhat(LPOutlierIndex)
          write(6,'("     reference value = ",F25.13)') Yref(LPOutlierIndex)
        else
          write(6,'(a)') '   all errors are non-positive'
        end if
      end if
      if (option%calcLNOutlier) then
        if (LNOutlier < 0.0_rprec) then
          write(6,'(a)') '   largest negative outlier'
          write(6,'("     error = ",F25.13)') LNOutlier
          !write(6,'("     index = ",I0)') LNOutlierIndex
          write(6,'("     estimated value = ",F25.13)') Yhat(LNOutlierIndex)
          write(6,'("     reference value = ",F25.13)') Yref(LNOutlierIndex)
        else
          write(6,'(a)') '   all errors are non-negative'
        end if
      end if
    end if
    deallocate(Yref)
    deallocate(Yhat)
  end if
  if (option%YgradXYZfile /= '') then
      nn = 0
      ipl = 0
      do ip=1, Nindices
        if (NgradXYZ(indicesForEval(ip)) /= 0) then
          ipl = ipl + 1
          nn = nn + 3 * NgradXYZ(indicesForEval(ip))
        end if
      end do
      if (nn == 0) then
        write(6,'(a)') ' <!> No entries to calculate statistics for <!>'
        return
      end if
      allocate(indicesForEvalLoc(1:ipl),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate array indicesForEvalLoc')
      allocate(Yref(1:nn),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate array Yref')
      allocate(Yhat(1:nn),stat=Error)
      if(Error/=0)call stopMLatomF('Unable to allocate array Yhat')
      ind = 0
      ipl = 0
      do ip=1, Nindices
        if (NgradXYZ(indicesForEval(ip)) /= 0) then
          ipl = ipl + 1
          indicesForEvalLoc(ipl) = indicesForEval(ip)
          do jj = 1, NgradXYZ(indicesForEval(ip))
            do ii = 1, 3
              ind = ind + 1
              Yref(ind) = YgradXYZ(   ii,jj,indicesForEval(ip))
              Yhat(ind) = YgradXYZest(ii,jj,indicesForEval(ip))
            end do
          end do
        end if
      end do
    if (present(comment)) write(6,'(a)') ' Analysis for gradients in XYZ coordinates'
    call stats(nn, Yref, Yhat, indicesForEvalLoc, comment, returnValue)
    if (present(returnValue)) VALUEgeom = VALUEgeom * returnValue
    Nvalues = Nvalues + 1
    if (present(comment)) then
      if (option%calcLPOutlier) then
        if (LPOutlier > 0.0_rprec) then
          write(6,'(a)') '   largest positive outlier'
          write(6,'("     error = ",F25.13)') LPOutlier
          !write(6,'("     index = ",I0)') LPOutlierIndex!indicesForEval(1+int(LPOutlierIndex/3/NgradXYZmax))
          write(6,'("     estimated value = ",F25.13)') Yhat(LPOutlierIndex)
          write(6,'("     reference value = ",F25.13)') Yref(LPOutlierIndex)
        else
          write(6,'(a)') '   all errors are non-positive'
        end if
      end if
      if (option%calcLNOutlier) then
        if (LNOutlier < 0.0_rprec) then
          write(6,'(a)') '   largest negative outlier'
          write(6,'("     error = ",F25.13)') LNOutlier
          !write(6,'("     index = ",I0)') LNOutlierIndex
          write(6,'("     estimated value = ",F25.13)') Yhat(LNOutlierIndex)
          write(6,'("     reference value = ",F25.13)') Yref(LNOutlierIndex)
        else
          write(6,'(a)') '   all errors are non-negative'
        end if
      end if
    end if
    deallocate(indicesForEvalLoc)
    deallocate(Yref)
    deallocate(Yhat)
  end if
  
  if (present(returnValue)) then
    VALUEgeom = VALUEgeom**(1.0_rprec/float(Nvalues))
    returnValue = VALUEgeom
  end if
  if (present(comment) .and. present(returnValue)) write(6,'(a,F25.13)') '   Geometric mean of error measure:', VALUEgeom

  ! Benchmark
  if(option%benchmark .and. option%debug) call timeTaken(dateStart,'Time for statistical analysis:')

end subroutine calc_stat_measures
!==============================================================================

!==============================================================================
subroutine stats(nn, Yref, Yhat, indicesForEval, comment, returnValue)
!==============================================================================
! Subroutine calculates requested measures of errors
!==============================================================================
  use optionsModule, only : option
  implicit none
  ! Arguments
  integer,          intent(in)            :: nn          ! Size of arrays
  real(kind=rprec), intent(in)            :: Yref(1:nn)  ! reference values
  real(kind=rprec), intent(in)            :: Yhat(1:nn)  ! estimated values
  integer,          intent(in)            :: indicesForEval(1:nn) ! Indices for evaluation
  character(len=*), intent(in),  optional :: comment     ! Comment
  real(kind=rprec), intent(out), optional :: returnValue ! Contains value of the measure
  ! Local variables
  real(kind=rprec) :: value
  character(len=1) :: stmp1, stmp2 ! Temporary string variables
  
  if (nn == 0) then
    write(6,'(a)') ' <!> No entries to calculate statistics for <!>'
  else
    ! Calculate measure(s)
    if (option%calcMAE) then
      MAEvalue = MAE(nn, Yref, Yhat)
    end if
    if (option%calcMSE) then
      MSEvalue = MSE(nn, Yref, Yhat)
    end if
    if (option%calcRMSE) then
      RMSEvalue = RMSE(nn, Yref, Yhat)
    end if
    if (option%calcMEAN) then
      YestMEANvalue = mean(nn, Yhat)
      YMEANvalue    = mean(nn, Yref)
    end if
    if (option%calcLPOutlier) then
      call largest_positive_outlier(nn, Yref, Yhat, LPOutlier, LPOutlierIndex)
    end if
    if (option%calcLNOutlier) then
      call largest_negative_outlier(nn, Yref, Yhat, LNOutlier, LNOutlierIndex)
    end if
    if (option%calcCorrCoeff) then
      CorrCoeff = correlation_coefficient(nn, Yref, Yhat)
    end if
    if (option%calcLinReg) then
      call linear_regression(nn, Yref, Yhat, aValue, bValue, r_squared, SE_a, SE_b)
    end if
    
    if (present(comment)) then
      write(6,'(" Statistical analysis for ",a)') comment
      if (option%calcMAE) then
        write(6,'("   MAE = ",F25.13)') MAEvalue
      end if
      if (option%calcMSE) then
        write(6,'("   MSE = ",F25.13)') MSEvalue
      end if
      if (option%calcRMSE) then
        write(6,'("   RMSE = ",F25.13)') RMSEvalue
      end if
      if (option%calcMEAN) then
        write(6,'("   mean(Y) = ",F25.13)') YMEANvalue
        write(6,'("   mean(Yest) = ",F25.13)') YestMEANvalue
      end if
      if (option%calcCorrCoeff) then
        write(6,'("   correlation coefficient = ",F25.13)') CorrCoeff
      end if
      if (option%calcLinReg) then
        write(6,'(a)') '   linear regression of {y, y_est} by f(a,b) = a + b * y'
        write(6,'("     R^2 = ",F25.13)') r_squared
        write(6,'("     a = ",F25.13)')   aValue
        write(6,'("     b = ",F25.13)')   bValue
        write(6,'("     SE_a = ",F25.13)')   SE_a
        write(6,'("     SE_b = ",F25.13)')   SE_b
      end if
    end if  
  end if

  if (present(returnValue)) then
    select case (trim(option%minimizeError))
      case ('MAE')
        returnValue = MAEvalue
      case ('MSE')
        returnValue = MSEvalue
      case ('RMSE')
        returnValue = RMSEvalue
      case ('LPO')
        returnValue = LPOutlier
      case ('LNO')
        returnValue = LNOutlier
      case ('CORRCOEFF')
        returnValue = 1.0_rprec - CorrCoeff
      case ('R2')
        returnValue = 1.0_rprec - r_squared
    end select
  end if

end subroutine stats
!==============================================================================

!==============================================================================
function MAE(nn, Yref, Yhat)
!==============================================================================
! Function returns mean absolute error
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  real(kind=rprec)    :: MAE
  ! Local variables
  integer             :: i

  MAE = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i) SHARED(Yref,Yhat) SCHEDULE(STATIC) REDUCTION (+:MAE)
  do i=1, nn
    MAE = MAE + abs(Yhat(i) - Yref(i))
  end do
  !$OMP END PARALLEL DO
  
  MAE = MAE / dble(nn)

end function MAE
!==============================================================================

!==============================================================================
function MSE(nn, Yref, Yhat)
!==============================================================================
! Function returns mean signed error
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  real(kind=rprec)    :: MSE
  ! Local variables
  integer             :: i

  MSE = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i) SHARED(Yref,Yhat) SCHEDULE(STATIC) REDUCTION (+:MSE)
  do i=1, nn
    MSE = MSE + (Yhat(i) - Yref(i))
  end do
  !$OMP END PARALLEL DO

  MSE = MSE / dble(nn)

end function MSE
!==============================================================================

!==============================================================================
function RMSE(nn, Yref, Yhat)
!==============================================================================
! Function returns root-mean-square error
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  real(kind=rprec)    :: RMSE
  ! Local variables
  integer             :: i

  RMSE = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i) SHARED(Yref,Yhat) SCHEDULE(STATIC) REDUCTION (+:RMSE)
  do i=1, nn
    RMSE = RMSE + (Yhat(i) - Yref(i)) ** 2
  end do
  !$OMP END PARALLEL DO
  
  RMSE = sqrt(RMSE / dble(nn))

end function RMSE
!==============================================================================

!==============================================================================
function mean(arraySize, array)
!==============================================================================
! Function returns mean value of an array
!==============================================================================
  implicit none
  ! Arguments
  integer, intent(in) :: arraySize
  real(kind=rprec)    :: array(1:arraySize)
  ! Return value
  real(kind=rprec)    :: mean
  ! Local variables
  integer             :: i

  mean = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i) SHARED(array) SCHEDULE(STATIC) REDUCTION (+:mean)
  do i=1, arraySize
    mean = mean + array(i)
  end do
  !$OMP END PARALLEL DO
  
  mean = mean / dble(arraySize)

end function mean
!==============================================================================

!==============================================================================
subroutine largest_positive_outlier(nn, Yref, Yhat, value, valueIndex)
!==============================================================================
! Subroutine returns the error of the largest positive outlier and its index
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  integer,          intent(out) :: valueIndex
  real(kind=rprec), intent(out) :: value
  ! Local variables
  integer                       :: i
  real(kind=rprec)              :: temp

  valueIndex = 1
  value = Yhat(1) - Yref(1)

  do i=2, nn
    temp = Yhat(i) - Yref(i)
    if (temp > value) then
      value      = temp
      valueIndex = i
    end if
  end do

end subroutine largest_positive_outlier
!==============================================================================

!==============================================================================
subroutine largest_negative_outlier(nn, Yref, Yhat, value, valueIndex)
!==============================================================================
! Subroutine returns the error of the largest negative outlier and its index
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  integer,          intent(out) :: valueIndex
  real(kind=rprec), intent(out) :: value
  ! Local variables
  integer                       :: i
  real(kind=rprec)              :: temp

  valueIndex = 1
  value = Yhat(1) - Yref(1)

  do i=2, nn
    temp = Yhat(i) - Yref(i)
    if (temp < value) then
      value      = temp
      valueIndex = i
    end if
  end do

end subroutine largest_negative_outlier
!==============================================================================

!==============================================================================
function correlation_coefficient(nn, Yref, Yhat)
!==============================================================================
! Function returns Pearson correlation coefficient
!                  Note that squared correlation coefficient
!                  is the same as 
!                  R-squared in linear least squares regression
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  real(kind=rprec)    :: correlation_coefficient
  ! Local variables
  integer             :: i
  real(kind=rprec)    :: Xmean
  real(kind=rprec)    :: Ymean
  real(kind=rprec)    :: nominator, denominator1, denominator2

  correlation_coefficient = 0.0_rprec

  Xmean = mean(nn, Yref)
  Ymean = mean(nn, Yhat)

  nominator    = 0.0_rprec
  denominator1 = 0.0_rprec
  denominator2 = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i) SHARED(Yref,Yhat) SCHEDULE(STATIC) REDUCTION (+:nominator,denominator1,denominator2)
  do i=1, nn
    nominator    = nominator    + (Yref(i) - Xmean) * (Yhat(i) - Ymean)
    denominator1 = denominator1 + (Yref(i) - Xmean) ** 2
    denominator2 = denominator2 + (Yhat(i) - Ymean) ** 2
  end do
  !$OMP END PARALLEL DO
  
  correlation_coefficient = nominator / (sqrt(denominator1) * sqrt(denominator2))

end function correlation_coefficient
!==============================================================================

!==============================================================================
subroutine linear_regression(nn, Yref, Yhat, a, b, r_squared, SE_a, SE_b)
!==============================================================================
! Function returns regression coefficients a, b by least square
!                  fitting of {Yref, Yhat} by Yhat(a,b) = a + b * Yref
!                  Returns also R squared value and
!                  standard errors for a and b
!                  
!                  http://mathworld.wolfram.com/LeastSquaresFitting.html
!==============================================================================
  implicit none
  ! Arguments
  integer,          intent(in) :: nn         ! Size of arrays
  real(kind=rprec), intent(in) :: Yref(1:nn) ! reference values
  real(kind=rprec), intent(in) :: Yhat(1:nn) ! estimated values
  ! Return value
  real(kind=rprec), intent(out) :: a, b, r_squared, SE_a, SE_b
  ! Local variables
  integer             :: i
  real(kind=rprec)    :: Xmean, Ymean, ss_xx, ss_yy, ss_xy, s

  ! Initialization
  a         = 0.0_rprec
  b         = 0.0_rprec
  r_squared = 0.0_rprec
  SE_a      = 0.0_rprec
  SE_b      = 0.0_rprec

  Xmean = mean(nn, Yref)
  Ymean = mean(nn, Yhat)

  ss_xx = -nn * (Xmean ** 2)
  ss_yy = -nn * (Ymean ** 2)
  ss_xy = -nn * Xmean * Ymean

  !$OMP PARALLEL DO PRIVATE(i) SHARED(Yref,Yhat) SCHEDULE(STATIC) REDUCTION (+:ss_xx,ss_yy,ss_xy)
  do i=1, nn
    ss_xx = ss_xx + Yref(i) ** 2
    ss_yy = ss_yy + Yhat(i) ** 2
    ss_xy = ss_xy + Yref(i) * Yhat(i)
  end do
  !$OMP END PARALLEL DO

  if ((ss_xx == 0) .or. (ss_yy == 0)) then
    write(6,*) ' WARNING: UNABLE TO CALCULATE LINEAR REGRESSION'
    b         = 0.0_rprec
    a         = Ymean
    r_squared = 0.0_rprec
    s         = 0.0_rprec
    SE_a      = 0.0_rprec
    SE_b      = 0.0_rprec
  else
    b         = ss_xy / ss_xx
    a         = Ymean - b * Xmean
    r_squared = ss_xy ** 2 / (ss_xx * ss_yy)
    s         = sqrt((ss_yy - ( (ss_xy ** 2) / ss_xx) ) / float(nn - 2))
    SE_a      = s * sqrt(1.0_rprec / float(nn) + (Xmean ** 2) / ss_xx)
    SE_b      = s / sqrt(ss_xx)
  end if
  
end subroutine linear_regression
!==============================================================================
    
end module MLstatistics
