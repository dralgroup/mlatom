
  !---------------------------------------------------------------------------! 
  ! A_KRR_kernel: kernel-specific calculations                                ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !                     Yi-Fan Hou (co-implementation of derivatives)         !
  !---------------------------------------------------------------------------! 

module A_KRR_kernel
  ! This module is for kernel-specific calculations
  use dataset,       only : arrayOfArrays, NgradXYZmax, XvecSize, Nperm
  use optionsModule, only : option
  use precision,     only : rprec
  use stopper,       only : stopMLatomF
  implicit none
  ! Variables
  real(kind=rprec), target :: sigma = 100.0_rprec ! Length-scale
  real(kind=rprec), target :: sigmap = 100.0_rprec! Length-scale for the periodic part
  integer,          target :: lgSigmaPoints       ! Number of points on a logarithmic grid for search optimal sigma
  real(kind=rprec), target :: lgSigmaL            ! The lowest  value of log2(sigma) to try
  real(kind=rprec), target :: lgSigmaH            ! The highest value of log2(sigma) to try
  real(kind=rprec), target :: c = 100.0_rprec     ! addend
  integer,          target :: lgCPoints           ! Number of points on a logarithmic grid for search optimal c
  real(kind=rprec), target :: lgCL                ! The lowest  value of log2(c) to try
  real(kind=rprec), target :: lgCH                ! The highest value of log2(c) to try
  integer,          target :: d = 2               ! exponent
  integer,          target :: lgDPoints           ! Number of points on a logarithmic grid for search optimal d
  real(kind=rprec), target :: lgDL                ! The lowest  value of log2(d) to try
  real(kind=rprec), target :: lgDH                ! The highest value of log2(d) to try
  integer,          target :: nn = 2              ! n in the Matern kernel (nu = n + 1/2)
  real(kind=rprec), target :: period =  1.0_rprec ! Period
  integer                  :: NtrTot              ! Total number of training points
  integer                  :: NtrVal              ! Number of training points with values
  integer                  :: NtrGrXYZ            ! Number of training points with XYZ gradients
  integer, allocatable     :: itrval(:)           ! Indices of training points with values
  integer, allocatable     :: itrgrxyz(:,:)       ! Indices of training points with XYZ gradients
  integer                  :: NselectPerm
  integer, allocatable     :: indicesSelectPerm(:)
  logical                  :: selectPerm = .false.

  ! Arrays of arrays
  type(arrayOfArrays), allocatable :: K(:)        ! Kernel matrices (K(1:NkernelMatrices)%twoDrArr(1:Npoints,1:Npoints))
  type(arrayOfArrays), allocatable :: Kvalidate(:)! Extra elements of the kernel matrices for validation (Kvalidate(1:NKvalidateMatrices)%oneDrArr(1:Npoints))
  real(kind=rprec), allocatable :: temp_XYZ_B(:,:,:)
  logical :: allocated_B = .true.

  integer, allocatable          :: mod_permInds(:,:)
  integer, allocatable          :: mod_inversePermInds(:,:) 
  integer, allocatable          :: mod_ac2dArray(:,:)
  real(kind=rprec), allocatable :: mod_MaternConsts(:,:)
  type(arrayOfArrays), allocatable :: mod_permKvalue(:)
  type(arrayOfArrays) :: mod_distanceTimesX


contains

subroutine getSigma()
  implicit none

  if (option%sigma == 'OPT') then
    read(option%NlgSigma,*) lgSigmaPoints
    read(option%lgSigmaL,*) lgSigmaL
    read(option%lgSigmaH,*) lgSigmaH
    if(option%debug) write(6,'(a)') 'optimize sigma parameter'
    sigma=100.0_rprec
  else
    read(option%sigma,*) sigma
  end if

end subroutine getSigma

subroutine getPeriod()
  implicit none

  read(option%period,*) period

end subroutine getPeriod

subroutine getSigmaP()
  implicit none

  read(option%sigmap,*) sigmap

end subroutine getSigmaP

subroutine getC()
  implicit none

  if (option%c == 'OPT') then
    read(option%NlgC,*) lgCPoints
    read(option%lgCL,*) lgCL
    read(option%lgCH,*) lgCH
    if(option%debug) write(6,'(a)') 'optimize c parameter'
    c=100.0_rprec
  else
    read(option%c,*) c
  end if

end subroutine getC

subroutine getD()
  implicit none

!  if (option%d == 'OPT') then
!    read(option%NlgD,*) lgDPoints
!    read(option%lgDL,*) lgDL
!    read(option%lgDH,*) lgDH
!    if(option%debug) write(6,'(a)') 'optimize d parameter'
!    d=2.0_rprec
!  else
    read(option%d,*) d
!  end if

end subroutine getD

subroutine getNN()
  implicit none

!  if (option%d == 'OPT') then
!    read(option%NlgD,*) lgDPoints
!    read(option%lgDL,*) lgDL
!    read(option%lgDH,*) lgDH
!    if(option%debug) write(6,'(a)') 'optimize d parameter'
!    d=2.0_rprec
!  else
    read(option%nn,*) nn
!  end if

end subroutine getNN

subroutine calcKernel(Kmatrix)
  use dataset,       only : X, XvecSize, Nperm
  use timing,        only : timeTaken 
  implicit none
  ! Arguments
  real(kind=rprec), intent(inout) :: Kmatrix(1:NtrTot,1:NtrTot) ! Indices of the training points
  ! Local variables
  integer             :: ii, jj, PP
  ! Arrays
  integer*8           :: dateStart(1:8) ! Time and date, when MLatomF starts
  integer             :: Xsize

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  call initializeAllArrays()
  call initializeMod_permKvalue()

  ! Covariance between values
    !$OMP PARALLEL DO PRIVATE(ii,jj) &
    !$OMP SHARED(Kmatrix,itrval) SCHEDULE(STATIC)
    do ii=1,NtrVal
      Kmatrix(ii,ii) = kernelFunction(X(:,itrval(ii)),X(:,itrval(ii)))
      do jj=ii+1,NtrVal
        Kmatrix(ii,jj) = kernelFunction(X(:,itrval(ii)),X(:,itrval(jj)))
        Kmatrix(jj,ii) = Kmatrix(ii,jj)
      end do
    end do
    !$OMP END PARALLEL DO

  ! Covariance between values and XYZ gradients
  !$OMP PARALLEL DO PRIVATE(ii,jj) &
  !$OMP SHARED(Kmatrix,itrval,itrgrxyz) SCHEDULE(STATIC)
  do ii=1,NtrVal
    do jj=1,NtrGrXYZ
      Kmatrix(ii,NtrVal+jj) = dKdMiat_usepermKvalue(itrgrxyz(jj,1),itrval(ii),itrgrxyz(jj,2),itrgrxyz(jj,3))
      Kmatrix(NtrVal+jj,ii) = Kmatrix(ii,NtrVal+jj)
    end do
  end do
  !$OMP END PARALLEL DO
  
  ! Covariance between XYZ gradients
  !$OMP PARALLEL DO PRIVATE(ii,jj) &
  !$OMP SHARED(Kmatrix,itrgrxyz) SCHEDULE(STATIC)
  do ii=1,NtrGrXYZ
    Kmatrix(NtrVal+ii,NtrVal+ii) = &
    d2KdMiatdMjbu_usepermKvalue(itrgrxyz(ii,1),itrgrxyz(ii,1),itrgrxyz(ii,2),itrgrxyz(ii,3),itrgrxyz(ii,2),itrgrxyz(ii,3))
    do jj=ii+1,NtrGrXYZ
      Kmatrix(NtrVal+ii,NtrVal+jj) = &
      d2KdMiatdMjbu_usepermKvalue(itrgrxyz(ii,1),itrgrxyz(jj,1),itrgrxyz(ii,2),itrgrxyz(ii,3),itrgrxyz(jj,2),itrgrxyz(jj,3))
      Kmatrix(NtrVal+jj,NtrVal+ii) = Kmatrix(NtrVal+ii,NtrVal+jj)
    end do
  end do
  !$OMP END PARALLEL DO

  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Kernel matrix building time:')

end subroutine calcKernel

subroutine calc_Kvalidate(NsubtrainPoints, indicesSubtrainPoints, NvalidatePoints, indicesValidatePoints, KvalidateMatrix)
  use dataset,       only : Nsubtrain, Nvalidate, indicesSubtrain, indicesValidate, X
  use timing,        only : timeTaken
  implicit none
  ! Arguments
  integer,          intent(in)    :: NsubtrainPoints                                    ! Number  of the training points
  integer,          intent(in)    :: indicesSubtrainPoints(1:NsubtrainPoints)           ! Indices of the training points
  integer,          intent(in)    :: NvalidatePoints                                    ! Number  of the validation points
  integer,          intent(in)    :: indicesValidatePoints(1:NvalidatePoints)           ! Indices of the validation points
  real(kind=rprec), intent(inout) :: KvalidateMatrix(1:NvalidatePoints*NsubtrainPoints) ! Array with extra kernel matrix elements
  ! Local variables
  integer                       :: a, i, areal, ireal
  ! Arrays
  integer*8                     :: dateStart(1:8) ! Time and date, when MLatomF starts

  ! Benchmark
  if(option%benchmark) call date_and_time(values=dateStart)

  KvalidateMatrix = 0.0_rprec

  !$OMP PARALLEL DO PRIVATE(i,a,areal,ireal) SHARED(X,indicesValidatePoints,indicesSubtrainPoints,KvalidateMatrix) SCHEDULE(STATIC)
  do a=1,NvalidatePoints
    areal = indicesValidatePoints(a)
    do i=1,NsubtrainPoints
      ireal = indicesSubtrainPoints(i)
      KvalidateMatrix((a-1)*NsubtrainPoints+i) = kernelFunction(X(:,areal),X(:,ireal))
    end do
  end do
  !$OMP END PARALLEL DO

  ! Benchmark
  if(option%benchmark) call timeTaken(dateStart,'Extra elements of kernel matrix calculating time:')

end subroutine calc_Kvalidate

function Yest_KRR(iEst, alpha)
  use dataset, only : X, XvecSize, Nperm
  implicit none
  ! Arguments
  integer,          intent(in) :: iEst                               ! Real index of X array
  real(kind=rprec), intent(in) :: alpha(1:NtrTot)
  ! Return value
  real(kind=rprec) :: Yest_KRR
  ! Local variables
  integer          :: jj
  integer          :: Xsize

  Yest_KRR = 0.0_rprec
  do jj=1,NtrVal
    Yest_KRR = Yest_KRR + alpha(jj) * kernelFunction(X(:,iEst),X(:,itrval(jj)))
  end do
  do jj=1,NtrGrXYZ
    Yest_KRR = Yest_KRR + alpha(NtrVal+jj) * dKdMiat(itrgrxyz(jj,1),iEst,itrgrxyz(jj,2),itrgrxyz(jj,3))
  end do

end function Yest_KRR

function YestGrad_KRR(iEst, alpha)
  use dataset,       only : X, XvecSize
  implicit none
  ! Arguments
  integer,          intent(in) :: iEst                               ! Real index of X array
  real(kind=rprec), intent(in) :: alpha(1:NtrTot)
  ! Return value
  real(kind=rprec) :: YestGrad_KRR(1:XvecSize)
  ! Local variables
  integer          :: dd, jj
  real(kind=rprec) :: cmf, cmf1, cmf2
  real(kind=rprec) :: temp_Array1(1:XvecSize)
  real(kind=rprec) :: temp_Array2(1:XvecSize)

  YestGrad_KRR = 0.0_rprec
  temp_Array1 = 0.0_rprec
  temp_Array2 = 0.0_rprec

if (.not. option%permInvKernel .and. option%kernel == 'Gaussian' .and. .not. option%numDerivs &
    .and. .not. selectPerm) then 
  ! Optimized Code 
  do jj=1,Ntrval
    cmf = alpha(jj) * dKdXid_const2(XvecSize,X(:,iEst),X(:,itrval(jj)))
    do dd=1,XvecSize 
      temp_Array1(dd) = temp_Array1(dd) + (X(dd,itrval(jj)) - X(dd,iEst)) * cmf  
      !temp_Array1(dd) = temp_Array1(dd) + dKdXid_changing(XvecSize,X(:,iEst),X(:,itrval(jj)),dd) * cmf 
    end do 
  end do 
  temp_Array1 = temp_Array1 * dKdXid_const1()

  YestGrad_KRR = temp_Array1 + temp_Array2
else 

  ! Original Code
  YestGrad_KRR = 0.0_rprec
  do dd = 1, XvecSize
    do jj=1,NtrVal
      YestGrad_KRR(dd) = YestGrad_KRR(dd) + &
      alpha(jj) * dKdXid(XvecSize,X(:,iEst),X(:,itrval(jj)),dd)
    end do
  end do
end if 

end function YestGrad_KRR

function YgradXYZest_KRR(iEst, alpha)
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, XYZ_B, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  implicit none
  ! Arguments
  integer,          intent(in) :: iEst                               ! Real index of X array
  real(kind=rprec), intent(in) :: alpha(1:NtrTot)
  ! Return value
  real(kind=rprec) :: YgradXYZest_KRR(1:3,1:NAtomsMax)
  ! Local variables
  integer          :: jj, a, cc, tt, dd, bb, gg, ee, uu, Pdd, Pee
  real(kind=rprec) :: temp_Array1(1:3,1:NatomsMax), temp_Array2(1:3,1:NatomsMax)
  real(kind=rprec) :: in1_temp_Array(1:3,1:NatomsMax)
  real(kind=rprec) :: in1_temp_Array1(1:3), in1_temp_Array2(1:3)
  real(kind=rprec) :: in2_temp_Array(1:3,1:NatomsMax), in3_temp_Array(1:3,1:NatomsMax)
  real(kind=rprec) :: dXdMi, dXdMj, temp1, temp2, temp3, temp4 
  real(kind=rprec) :: cmf, cmf1, cmf2
  integer :: Xsize, PP, lb, ub 
  real(kind=rprec) :: KMiMj,KMiMi, KMjMj
  !real(kind=rprec) :: dKMiMjdMiat, dKMiMidMiat
  real(kind=rprec) :: dKMiMjdMiat_part(1:3,1:NatomsMax), dKMiMidMiat_part(1:3,1:NatomsMax)
  real(kind=rprec) :: dKMiMjdMjbu_part, dKMjMjdMjbu_part
  real(kind=rprec) :: d2KMiMjdMiatdMjbu_part(1:3,1:NatomsMax)


  YgradXYZest_KRR = 0.0_rprec
  temp_Array1 = 0.0_rprec
  temp_Array2 = 0.0_rprec
  in1_temp_Array1 = 0.0_rprec
  in1_temp_Array2 = 0.0_rprec
  in1_temp_Array = 0.0_rprec
  in2_temp_Array = 0.0_rprec
  in3_temp_Array = 0.0_rprec

  if (.not. option%permInvKernel .and. &
      (option%molDescriptor == 'RE' .or. option%molDescriptor == 'ID') &
      .and. option%kernel == 'Gaussian' .and. .not. option%numDerivs &
      .and. .not. selectPerm) then 

    ! Optimized Code(first derivatives)
    do jj=1,NtrVal 
      cmf = alpha(jj) * dKdXid_const2(XvecSize,X(:,iEst),X(:,itrval(jj)))
      do a = 1, Natoms(iEst) - 1 
        do cc = a + 1, Natoms(iEst)
          if (a /= cc) then
            dd = mod_ac2dArray(a,cc) 
            temp1 = (X(dd,itrval(jj)) - X(dd,iEst)) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
            !temp1 = dKdXid_changing(XvecSize,X(:,iEst),X(:,itrval(jj)),dd) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
            do tt = 1, 3 
              temp2 = XYZ_A(tt,cc,iEst)-XYZ_A(tt,a,iEst)
              in1_temp_Array(tt,a) = in1_temp_Array(tt,a) + temp2 * temp1    ! It seems that in1_temp_Array is not initialized.
              in1_temp_Array(tt,cc) = in1_temp_Array(tt,cc) - temp2 * temp1
            end do 
          end if 
        end do 
      end do 
      temp_Array1 = temp_Array1 + in1_temp_Array * cmf 
      in1_temp_Array = 0.0_rprec
    end do 
    temp_Array1 = temp_Array1 * dKdXid_const1()

    !do a = 1, Natoms(iEst)
    !  do tt = 1, 3
    !    do jj=1,NtrVal
    !      temp_Array1(tt,a) = temp_Array1(tt,a) + &
    !      alpha(jj) * dKdMiat(iEst,itrval(jj),a,tt)
    !    end do
    !  end do
    !end do


    ! Optimized Code
    do jj=1,NtrGrXYZ 
      cmf = alpha(NtrVal+jj) * d2KdXiddXje_const2(XvecSize,X(:,iEst),X(:,itrgrxyz(jj,1)))
      bb = itrgrxyz(jj,2) 
      uu = itrgrxyz(jj,3) 
      do gg = 1, Natoms(itrgrxyz(jj,1)) 
        if (bb /= gg) then 
          ee = mod_ac2dArray(bb,gg) 
          temp1 = X(ee,itrgrxyz(jj,1)) * (XYZ_A(uu,gg,itrgrxyz(jj,1))-XYZ_A(uu,bb,itrgrxyz(jj,1))) / Rij(XYZ_A(:,bb,itrgrxyz(jj,1)),XYZ_A(:,gg,itrgrxyz(jj,1))) ** 2
          do a = 1, Natoms(iEst) - 1
            do cc = a + 1, Natoms(iEst) 
              !if (a /= cc) then 
                dd = mod_ac2dArray(a,cc) 
                temp2 = X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
                temp3 = d2KdXiddXje_changing(X(dd,iEst),X(dd,itrgrxyz(jj,1)),X(ee,iEst),X(ee,itrgrxyz(jj,1))) * d2KdXiddXje_const1(XvecSize,X(:,iEst),X(:,itrgrxyz(jj,1)))
                do tt = 1, 3
                  temp4 = (XYZ_A(tt,cc,iEst)-XYZ_A(tt,a,iEst))
                  in1_temp_Array1(tt) = in1_temp_Array1(tt) + temp4
                  if (dd == ee) in1_temp_Array2(tt) = in1_temp_Array2(tt) + temp4 
                end do 
                in1_temp_Array1(:) = in1_temp_Array1(:) * temp2 * temp3 + in1_temp_Array2(:) * temp2
                in2_temp_Array(:,a) = in2_temp_Array(:,a) + in1_temp_Array1(:)! * temp2 * temp3 + in1_temp_Array2(:) * temp2
                in2_temp_Array(:,cc) = in2_temp_Array(:,cc) - in1_temp_Array1(:)! * temp2 * temp3 - in1_temp_Array2(:) * temp2
                in1_temp_Array1 = 0.0_rprec
                in1_temp_Array2 = 0.0_rprec
              !end if 
            end do 
          end do 
          in3_temp_Array = in3_temp_Array + in2_temp_Array * temp1 
          in2_temp_Array = 0.0_rprec
        end if 
      end do 
      temp_Array2 = temp_Array2 + in3_temp_Array * cmf 
      in3_temp_Array = 0.0_rprec
    end do 
    temp_Array2 = temp_Array2 * d2KdXiddXje_const3()



    !do a = 1, Natoms(iEst)
    !  do tt = 1, 3
    !    do jj=1,NtrGrXYZ
    !     temp_Array2(tt,a) = temp_Array2(tt,a) + &
    !      alpha(NtrVal+jj) * d2KdMiatdMjbu(iEst,itrgrxyz(jj,1),a,tt,itrgrxyz(jj,2),itrgrxyz(jj,3))
    !    end do
    !  end do
    !end do

    YgradXYZest_KRR = temp_Array1 + temp_Array2

  else if (option%permInvKernel .and.&
           (option%molDescriptor == 'RE' .or. option%molDescriptor == 'ID') &
           .and. option%kernel == 'Gaussian' .and. .not. option%numDerivs) then 
    !!!!! Calculate the first-order derivative !!!!!
    ! Initialize
    KMiMi = 0.0_rprec
    KMiMj = 0.0_rprec
    KMjMj = 0.0_rprec
    dKMiMjdMiat_part = 0.0_rprec
    dKMiMidMiat_part = 0.0_rprec
    Xsize = XvecSize / Nperm 
    ! Calculate KMiMi
    do PP=1, Nperm
      lb = 1+Xsize*(PP-1)
      ub = Xsize*PP
      KMiMi = KMiMi + Kfunk(Xsize,X(1:Xsize,iEst),X(lb:ub,iEst))
    end do 

    ! Calculate dKMiMidMiat_part

    do PP=1, Nperm 
      lb = Xsize*(PP-1) 
      cmf = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,iEst))
      do a=1, Natoms(iEst)-1 
        do cc=a+1, Natoms(iEst)
          !if (a /= cc) then 
            dd = mod_ac2dArray(a,cc) 
            Pdd = mod_ac2dArray(mod_inversePermInds(a,PP),mod_inversePermInds(cc,PP))
            temp1 = (X(lb+dd,iEst) + X(Pdd,iEst) - X(dd,iEst)*2) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
            do tt=1, 3 
              temp2 = XYZ_A(tt,cc,iEst) - XYZ_A(tt,a,iEst)
              in1_temp_Array(tt,a) = in1_temp_Array(tt,a) + temp1 * temp2
              in1_temp_Array(tt,cc) = in1_temp_Array(tt,cc) - temp1 * temp2       
            end do 
          !end if 
        end do 
      end do 
      dKMiMidMiat_part = dKMiMidMiat_part + cmf * in1_temp_Array
      in1_temp_Array = 0.0_rprec
    end do 
    dKMiMidMiat_part = dKMiMidMiat_part / (sigma ** 2)

    ! Calculate KMiMj & MjMj & dKMiMjdMiat_part
    do jj=1,NtrVal 
      ! Calculate KMiMj & KMjMj
      KMiMj = 0.0_rprec
      KMjMj = 0.0_rprec
      do PP=1, Nperm 
        lb = 1+Xsize*(PP-1) 
        ub = Xsize*PP 
        KMiMj = KMiMj + Kfunk(Xsize,X(1:Xsize,iEst),X(lb:ub,itrval(jj)))
        KMjMj = KMjMj + Kfunk(Xsize,X(1:Xsize,itrval(jj)),X(lb:ub,itrval(jj)))
      end do  

      ! Calculate dKMiMjdMiat_part 
      dKMiMjdMiat_part = 0.0_rprec


      cmf = alpha(jj) / sqrt(KMiMi*KMjMj)
      do PP=1, Nperm 
        lb = Xsize*(PP-1)
        cmf1 = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,itrval(jj)))
        do a=1, Natoms(iEst)-1 
          do cc=a+1, Natoms(iEst) 
            !if (a /= cc) then 
              dd = mod_ac2dArray(a,cc) 
              temp1 = (X(lb+dd,itrval(jj)) - X(dd,iEst)) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
              do tt=1, 3 
                temp2 = XYZ_A(tt,cc,iEst) - XYZ_A(tt,a,iEst)
                in1_temp_Array(tt,a) = in1_temp_Array(tt,a) + temp1 * temp2
                in1_temp_Array(tt,cc) = in1_temp_Array(tt,cc) - temp1 * temp2 
              end do 
            !end if 
          end do 
        end do 
        dKMiMjdMiat_part = dKMiMjdMiat_part + cmf1 * in1_temp_Array
        in1_temp_Array = 0.0_rprec
      end do 
      dKMiMjdMiat_part = dKMiMjdMiat_part / (sigma ** 2)

      ! Calculate temp_Array1
      temp_Array1 = temp_Array1 + cmf * (dKMiMjdMiat_part - dKMiMidMiat_part * KMiMj / KMiMi / 2.0_rprec)
    end do 


    !!!!! Calculate the second-order derivative !!!!!
    ! Initialize 
    KMiMi = 0.0_rprec
    KMiMj = 0.0_rprec
    KMjMj = 0.0_rprec
    dKMiMjdMiat_part = 0.0_rprec
    dKMiMidMiat_part = 0.0_rprec
    dKMiMjdMjbu_part = 0.0_rprec
    dKMjMjdMjbu_part = 0.0_rprec
    d2KMiMjdMiatdMjbu_part = 0.0_rprec

    ! Calculate KMiMi
    do PP=1, Nperm
      lb = 1+Xsize*(PP-1)
      ub = Xsize*PP
      KMiMi = KMiMi + Kfunk(Xsize,X(1:Xsize,iEst),X(lb:ub,iEst))
    end do 

    ! Calculate dKMiMidMiat_part
    do PP=1, Nperm 
      lb = Xsize*(PP-1) 
      cmf = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,iEst))
      do a=1, Natoms(iEst)-1 
        do cc=a+1, Natoms(iEst)
          !if (a /= cc) then 
            dd = mod_ac2dArray(a,cc) 
            Pdd = mod_ac2dArray(mod_inversePermInds(a,PP),mod_inversePermInds(cc,PP))
            temp1 = (X(lb+dd,iEst) + X(Pdd,iEst) - X(dd,iEst)*2) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
            do tt=1, 3 
              temp2 = XYZ_A(tt,cc,iEst) - XYZ_A(tt,a,iEst)
              in1_temp_Array(tt,a) = in1_temp_Array(tt,a) + temp1 * temp2
              in1_temp_Array(tt,cc) = in1_temp_Array(tt,cc) - temp1 * temp2       
            end do 
          !end if 
        end do 
      end do 
      dKMiMidMiat_part = dKMiMidMiat_part + cmf * in1_temp_Array
      in1_temp_Array = 0.0_rprec
    end do 
    dKMiMidMiat_part = dKMiMidMiat_part / (sigma ** 2)

    ! Calculate KMiMj & KMjMj & dKMiMjdMiat_part & dKMiMjdMjbu_part & dKMjMjdMjbu_part & d2KMiMjdMiatdMjbu_part
    do jj=1,NtrGrXYZ 
      bb = itrgrxyz(jj,2)
      uu = itrgrxyz(jj,3)
      ! Calculate KMiMj & KMjMj 
      KMiMj = 0.0_rprec
      KMjMj = 0.0_rprec
      do PP=1, Nperm 
        lb = 1+Xsize*(PP-1) 
        ub = Xsize*PP 
        KMiMj = KMiMj + Kfunk(Xsize,X(1:Xsize,iEst),X(lb:ub,itrgrxyz(jj,1)))
        KMjMj = KMjMj + Kfunk(Xsize,X(1:Xsize,itrgrxyz(jj,1)),X(lb:ub,itrgrxyz(jj,1)))
      end do 
      
      ! Calculate dKMiMjdMiat_part 
      dKMiMjdMiat_part = 0.0_rprec

      do PP=1, Nperm 
        lb = Xsize*(PP-1)
        cmf1 = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,itrgrxyz(jj,1)))
        do a=1, Natoms(iEst)
          do cc=a+1, Natoms(iEst)
            dd = mod_ac2dArray(a,cc) 
            temp1 = (X(lb+dd,itrgrxyz(jj,1)) - X(dd,iEst)) * X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
            do tt=1, 3 
              temp2 = XYZ_A(tt,cc,iEst) - XYZ_A(tt,a,iEst) 
              in1_temp_Array(tt,a) = in1_temp_Array(tt,a) + temp1 * temp2 
              in1_temp_Array(tt,cc) = in1_temp_Array(tt,cc) - temp1 * temp2 
            end do 
          end do 
        end do
        dKMiMjdMiat_part = dKMiMjdMiat_part + cmf1 * in1_temp_Array
        in1_temp_Array = 0.0_rprec
      end do  
      dKMiMjdMiat_part = dKMiMjdMiat_part / (sigma ** 2)


!      do PP=1, Nperm 
!        lb = Xsize*(PP-1)+1
!        ub = Xsize*PP 
!        do a=1, Natoms(iEst) 
!          do tt=1, 3 
!            dKMiMjdMiat_part(tt,a) = dKMiMjdMiat_part(tt,a) + dKdMiat_unpermuted(Xsize,X(1:Xsize,iEst),X(lb:ub,itrgrxyz(jj,1)),iEst,itrgrxyz(jj,1),a,tt) 
!          end do 
!        end do 
!      end do 


      ! Calculate dKMiMjdMjbu_part 
      dKMiMjdMjbu_part = 0.0_rprec


      do PP=1, Nperm 
        lb = Xsize*(PP-1) 
        cmf1 = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,itrgrxyz(jj,1)))
        temp1 = 0.0_rprec
        do gg=1, Natoms(iEst)
          if (bb /= gg) then 
            ee = mod_ac2dArray(bb,gg) 
            Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP)) 
            temp1 = temp1 + (XYZ_A(uu,gg,itrgrxyz(jj,1))-XYZ_A(uu,bb,itrgrxyz(jj,1))) &
                          * (X(Pee,iEst) - X(ee,itrgrxyz(jj,1))) * X(ee,itrgrxyz(jj,1)) &
                          / Rij(XYZ_A(:,bb,itrgrxyz(jj,1)),XYZ_A(:,gg,itrgrxyz(jj,1))) ** 2 
            !dKMiMjdMjbu_part = dKMiMjdMjbu_part + temp1 
          end if 
        end do 
       dKMiMjdMjbu_part = dKMiMjdMjbu_part + temp1 * cmf1
      end do 
      dKMiMjdMjbu_part = dKMiMjdMjbu_part / (sigma ** 2) 


!      do PP=1, Nperm 
!        lb = Xsize*(PP-1)+1
!        ub = Xsize*PP 
!        dKMiMjdMjbu_part = dKMiMjdMjbu_part + PdKdMjbu(Xsize,X(1:Xsize,iEst),X(1:Xsize,itrgrxyz(jj,1)),X(lb:ub,itrgrxyz(jj,1)),iEst,itrgrxyz(jj,1),bb,uu,PP)
!      end do 

      




      ! Calculate dKMjMjdMjbu_part 
      dKMjMjdMjbu_part = 0.0_rprec


      do PP=1, Nperm 
        lb = Xsize*(PP-1) 
        cmf1 = Kfunk(Xsize,X(1:Xsize,itrgrxyz(jj,1)),X(lb+1:lb+Xsize,itrgrxyz(jj,1)))
        temp1 = 0.0_rprec
        do gg=1, Natoms(iEst) 
          if (bb /= gg) then 
            ee = mod_ac2dArray(bb,gg) 
            Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))
            temp1 = temp1 + (XYZ_A(uu,gg,itrgrxyz(jj,1))-XYZ_A(uu,bb,itrgrxyz(jj,1))) &
                          * (X(lb+ee,itrgrxyz(jj,1)) + X(Pee,itrgrxyz(jj,1)) - X(ee,itrgrxyz(jj,1))*2) * X(ee,itrgrxyz(jj,1)) &
                          / Rij(XYZ_A(:,bb,itrgrxyz(jj,1)),XYZ_A(:,gg,itrgrxyz(jj,1))) ** 2 
          end if 
        end do 
        dKMjMjdMjbu_part = dKMjMjdMjbu_part + temp1 * cmf1 
      end do 
      dKMjMjdMjbu_part = dKMjMjdMjbu_part / (sigma ** 2)


!      do PP=1, Nperm 
!        lb = Xsize*(PP-1)+1
!        ub = Xsize*PP 
!        dKMjMjdMjbu_part = dKMjMjdMjbu_part + dKdMiat_unpermuted(Xsize,X(1:Xsize,itrgrxyz(jj,1)),X(lb:ub,itrgrxyz(jj,1)),itrgrxyz(jj,1),itrgrxyz(jj,1),bb,uu)
!        dKMjMjdMjbu_part = dKMjMjdMjbu_part + PdKdMjbu(Xsize,X(1:Xsize,itrgrxyz(jj,1)),X(1:Xsize,itrgrxyz(jj,1)),X(lb:ub,itrgrxyz(jj,1)),itrgrxyz(jj,1),itrgrxyz(jj,1),bb,uu,PP)
!      end do 




      ! Calculate d2KMiMjdMiatdMjbu_part 
      d2KMiMjdMiatdMjbu_part = 0.0_rprec
      do PP=1, Nperm 
        lb = Xsize*(PP-1) 
        cmf1 = Kfunk(Xsize,X(1:Xsize,iEst),X(lb+1:lb+Xsize,itrgrxyz(jj,1)))

        do gg=1, Natoms(iEst) 
          if (bb /= gg) then 
            ee = mod_ac2dArray(bb,gg) 
            Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))
            temp1 = X(ee,itrgrxyz(jj,1)) * (XYZ_A(uu,gg,itrgrxyz(jj,1))-XYZ_A(uu,bb,itrgrxyz(jj,1))) / Rij(XYZ_A(:,bb,itrgrxyz(jj,1)),XYZ_A(:,gg,itrgrxyz(jj,1)))**2

            do a=1, Natoms(iEst) - 1 
              do cc=a+1, Natoms(iEst) 
                dd = mod_ac2dArray(a,cc) 
                Pdd = mod_ac2dArray(mod_permInds(a,PP),mod_permInds(cc,PP))
                temp2 = X(dd,iEst) / Rij(XYZ_A(:,a,iEst),XYZ_A(:,cc,iEst)) ** 2
                !temp3 = (X(lb+dd,itrgrxyz(jj,1)) - X(dd,iEst) + X(Pee,iEst) - X(ee,itrgrxyz(jj,1))) / (sigma**2)
                temp3 = d2KdXiddXje_changing(X(dd,iEst),X(lb+dd,itrgrxyz(jj,1)),X(Pee,iEst),X(ee,itrgrxyz(jj,1))) / (sigma**2)
                do tt=1, 3 
                  temp4 = XYZ_A(tt,cc,iEst) - XYZ_A(tt,a,iEst)
                  in1_temp_Array1(tt) = in1_temp_Array1(tt) + temp4  
                  if (Pdd==ee) in1_temp_Array2(tt) = in1_temp_Array2(tt) + temp4
                end do 
                in1_temp_Array1(:) = in1_temp_Array1(:) * temp2 * temp3 + in1_temp_Array2(:) * temp2 
                in2_temp_Array(:,a) = in2_temp_Array(:,a) + in1_temp_Array1(:)
                in2_temp_Array(:,cc) = in2_temp_Array(:,cc) - in1_temp_Array1(:)
                in1_temp_Array1 = 0.0_rprec
                in1_temp_Array2 = 0.0_rprec
              end do 
            end do 
            in3_temp_Array = in3_temp_Array + in2_temp_Array * temp1
            in2_temp_Array = 0.0_rprec
          end if 
        end do 
        d2KMiMjdMiatdMjbu_part = d2KMiMjdMiatdMjbu_part + in3_temp_Array * cmf1
        in3_temp_Array = 0.0_rprec
      end do 
      d2KMiMjdMiatdMjbu_part = d2KMiMjdMiatdMjbu_part / (sigma ** 2)

!      d2KMiMjdMiatdMjbu_part = 0.0_rprec
!      do PP=1, Nperm 
!        lb = Xsize*(PP-1)+1
!        ub = Xsize*PP
!        do a=1, Natoms(iEst) 
!          do tt=1, 3 
!            d2KMiMjdMiatdMjbu_part(tt,a) = d2KMiMjdMiatdMjbu_part(tt,a) + d2KdMiatdMjbu_unpermuted(Xsize,X(1:Xsize,iEst),X(lb:ub,itrgrxyz(jj,1)),X(lb:ub,iEst),X(1:Xsize,itrgrxyz(jj,1)),iEst,itrgrxyz(jj,1),a,tt,bb,uu,PP)
!          end do
!        end do 
!      end do 

      ! Calculate temp_Array2 

      temp_Array2 = temp_Array2 + alpha(NtrVal+jj) * (d2KMiMjdMiatdMjbu_part &
                  -  dKMiMjdMjbu_part*dKMiMidMiat_part/(2*KMiMi) &
                  -  dKMiMjdMiat_part*dKMjMjdMjbu_part/(2*KMjMj) &
                  +  dKMiMidMiat_part*dKMjMjdMjbu_part*KMiMj/(4*KMiMi*KMjMj)) &
                  /  sqrt(KMiMi*KMjMj)




    end do


!    do a = 1, Natoms(iEst)
!      do tt = 1, 3
!        do jj=1,NtrGrXYZ
!         temp_Array2(tt,a) = temp_Array2(tt,a) + &
!          alpha(NtrVal+jj) * d2KdMiatdMjbu(iEst,itrgrxyz(jj,1),a,tt,itrgrxyz(jj,2),itrgrxyz(jj,3))
!        end do
!      end do
!    end do

    YgradXYZest_KRR = temp_Array1 + temp_Array2

  else
    YgradXYZest_KRR = 0.0_rprec
    do a = 1, Natoms(iEst)
      do tt = 1, 3
        do jj=1,NtrVal
          YgradXYZest_KRR(tt,a) = YgradXYZest_KRR(tt,a) + &
          alpha(jj) * dKdMiat(iEst,itrval(jj),a,tt)
        end do
        do jj=1,NtrGrXYZ
          YgradXYZest_KRR(tt,a) = YgradXYZest_KRR(tt,a) + &
          alpha(NtrVal+jj) * d2KdMiatdMjbu(iEst,itrgrxyz(jj,1),a,tt,itrgrxyz(jj,2),itrgrxyz(jj,3))
        end do
      end do
    end do

  end if 

end function YgradXYZest_KRR

function YgradXYZCompntsEst_KRR(iEst, alpha)
  use dataset, only : X, NgradXYZmax
  implicit none
  ! Arguments
  integer,             intent(in) :: iEst                                 ! Real index of X array
  real(kind=rprec), intent(in) :: alpha(1:NtrTot,1:3*NgradXYZmax) ! Arrays with regression coefficients
  ! Return value
  real(kind=rprec) :: YgradXYZCompntsEst_KRR(1:3,1:NgradXYZmax)
  ! Local variables
  integer          :: a, tt, jj
  real(kind=rprec) :: kk

  YgradXYZCompntsEst_KRR = 0.0_rprec
  do jj=1,NtrVal
    kk = kernelFunction(X(:,iEst),X(:,itrval(jj)))
    do a = 1, NgradXYZmax
      do tt = 1, 3
        YgradXYZCompntsEst_KRR(tt,a) = YgradXYZCompntsEst_KRR(tt,a) + alpha(jj,tt+3*(a-1)) * kk
      end do
    end do
  end do

end function YgradXYZCompntsEst_KRR

function kernelFunction(Xi,Xj)
  use dataset,       only : XvecSize, Nperm
  implicit none
  ! Return value
  real(kind=rprec) :: kernelFunction
  ! Arguments
  real(kind=rprec), intent(in) :: Xi(1:XvecSize), Xj(1:XvecSize) ! X arrays
  ! Local variables
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj
  integer :: Xsize, PP

  kernelFunction = 0.0_rprec

  if (option%permInvKernel) then
      ! Calculate permutation invariant kernel and normalize it
      KMiMj = 0.0_rprec
      KMiMi = 0.0_rprec
      KMjMj = 0.0_rprec
      Xsize = XvecSize / Nperm
      do PP = 1, Nperm
        KMiMj = KMiMj + Kfunk(Xsize,Xi(1:Xsize),Xj(1+Xsize*(PP-1):Xsize+Xsize*(PP-1)))
        KMiMi = KMiMi + Kfunk(Xsize,Xi(1:Xsize),Xi(1+Xsize*(PP-1):Xsize+Xsize*(PP-1)))
        KMjMj = KMjMj + Kfunk(Xsize,Xj(1:Xsize),Xj(1+Xsize*(PP-1):Xsize+Xsize*(PP-1)))
      end do
      kernelFunction = KMiMj / (sqrt(KMiMi) * sqrt(KMjMj))
  else if (selectPerm) then ! selectPerm & permInvKernel should not be used at the same time
    kernelFunction = kernelFunction_selectPerm(Xi,Xj)
  else
    kernelFunction = Kfunk(XvecSize,Xi,Xj)
  end if

end function kernelFunction

function Kfunk(Xsize,Xi,Xj)
  use constants,     only : pi
  use mathUtils,     only : factorial_rprec
  implicit none
  ! Return value
  real(kind=rprec) :: Kfunk
  ! Arguments
  integer,          intent(in) :: Xsize                    ! Size of the X arrays
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  ! Local variables
  real(kind=rprec) :: dist
  integer :: kk

  Kfunk = 0.0_rprec

  if (trim(option%kernel) == 'linear') then
    Kfunk = distance(Xsize,Xi,Xj)
  elseif (trim(option%kernel) == 'polynomial') then
    Kfunk = (distance(Xsize,Xi,Xj) + c) ** d
  elseif (trim(option%kernel) == 'Gaussian') then
    if (option%decayKernel) then
      Kfunk = exp(-1.0_rprec * distance(Xsize,Xi,Xj) / (2 * sigma ** 2) - 2.0_rprec * (sin(pi * sqrt(distance(Xsize,Xi,Xj)) / period))**2 / (sigmap ** 2))
    elseif (option%periodKernel) then
      Kfunk = exp(-2.0_rprec * (sin(pi * sqrt(distance(Xsize,Xi,Xj)) / period))**2 / (sigma ** 2))
    else
      Kfunk = exp(-1.0_rprec * distance(Xsize,Xi,Xj) / (2 * sigma ** 2))
    end if
  elseif (trim(option%kernel) == 'Laplacian') then
    Kfunk = exp(-1.0_rprec * distance(Xsize,Xi,Xj) / sigma)
  elseif (trim(option%kernel) == 'exponential') then
    Kfunk = exp(-1.0_rprec * distance(Xsize,Xi,Xj) / sigma)
  elseif (trim(option%kernel) == 'Matern') then
    dist = distance(Xsize,Xi,Xj)
    do kk = 0, nn
      Kfunk = Kfunk + factorial_rprec(nn + kk) &
                    / factorial_rprec(kk)      &
                    / factorial_rprec(nn - kk) &
                    * (2 * dist / sigma) ** (nn - kk)
    end do
    Kfunk = exp(-1.0_rprec * dist / sigma) &
          * factorial_rprec(nn)            &
          / factorial_rprec(2 * nn)        &
          * Kfunk
  end if

end function Kfunk
  
function kernelFunction_selectPerm(Xi,Xj)
  use dataset, only : Nperm, XvecSize
  implicit none 
  ! Arguments
  real(kind=rprec), intent(in) :: Xi(1:XvecSize), Xj(1:XvecSize)
  ! Return value 
  real(kind=rprec) :: kernelFunction_selectPerm
  ! Local variables 
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj
  integer          :: ii
  integer          :: lb, ub
  integer          :: Xsize 

  !Initialize
  KMiMj = 0.0_rprec
  KMiMi = 0.0_rprec
  KMjMj = 0.0_rprec
  Xsize = XvecSize / Nperm
  kernelFunction_selectPerm = 0.0_rprec

    do ii=1,size(indicesSelectPerm)
      lb = 1+Xsize*(indicesSelectPerm(ii)-1)
      ub = Xsize*indicesSelectPerm(ii)
      KMiMj = KMiMj + Kfunk(Xsize,Xi(1:Xsize),Xj(lb:ub))
      KMiMi = KMiMi + Kfunk(Xsize,Xi(1:Xsize),Xi(lb:ub))
      KMjMj = KMjMj + Kfunk(Xsize,Xj(1:Xsize),Xj(lb:ub))
    end do 
    kernelFunction_selectPerm = KMiMj / (sqrt(KMiMi)*sqrt(KMjMj))

end function kernelFunction_selectPerm

function dKdMiat_selectPerm(ireal,jreal,a,tt)
  use dataset,           only : X, XvecSize, Natoms, NAtomsMax, Nperm 
  implicit none 
  ! Arguments 
  integer, intent(in) :: ireal,jreal,a,tt
  ! Return value 
  real(kind=rprec)    :: dKdMiat_selectPerm 
  ! Local variables 
  integer             :: ii 
  integer             :: lb, ub 
  integer             :: Xsize 

  dKdMiat_selectPerm = 0.0_rprec
  Xsize = XvecSize / Nperm 
  do ii=1,NselectPerm 
    lb = 1+Xsize*(indicesSelectPerm(ii)-1) 
    ub = Xsize*indicesSelectPerm(ii) 
    dKdMiat_selectPerm = dKdMiat_selectPerm + &
                         dKdMiat_unpermuted(Xsize,X(lb:ub,ireal),X(lb:ub,jreal),ireal,jreal,a,tt)
  end do
  dKdMiat_selectPerm = dKdMiat_selectPerm / NselectPerm
  
end function dKdMiat_selectPerm

function d2KdMiatdMjbu_selectPerm(ireal,jreal,a,tt,bb,uu) 
  use dataset,       only : X, XvecSize, Nperm 
  implicit none 
  ! Arguments 
  integer, intent(in) :: ireal,jreal,a,tt,bb,uu 
  ! Return value 
  real(kind=rprec)    :: d2KdMiatdMjbu_selectPerm 
  ! Local variables 
  integer             :: ii 
  integer             :: lb, ub 
  integer             :: Xsize

  d2KdMiatdMjbu_selectPerm = 0.0_rprec
  Xsize = XvecSize / Nperm 
  do ii=1,NselectPerm 
    lb = 1+Xsize*(indicesSelectPerm(ii)-1) 
    ub = Xsize*indicesSelectPerm(ii)
    d2KdMiatdMjbu_selectPerm = d2KdMiatdMjbu_selectPerm + &
                               d2KdMiatdMjbu_unpermuted(Xsize,X(lb:ub,ireal),X(lb:ub,jreal),X(lb:ub,ireal),X(lb:ub,jreal),ireal,jreal,a,tt,bb,uu,1)
  end do 
  d2KdMiatdMjbu_selectPerm = d2KdMiatdMjbu_selectPerm / NselectPerm

end function d2KdMiatdMjbu_selectPerm

subroutine initializeAllArrays()
  use dataset,       only : X, XvecSize, Natoms, Nperm, NAtomsMax, permutedIatoms
  use stopper,       only : stopMLatomF
  use mathUtils,     only : factorial_rprec
  implicit none 
  ! Local variables
  integer :: Xsize, PP, ll
  integer :: ii, jj, kk
  real(kind=rprec) :: const

  call allocateAllArrays()

  if (option%permInvKernel .and. (trim(option%permInvGroups) /= '' .or. trim(option%permInvNuclei) /= '' .or. option%usePermInd)) then 
    ! Calculate mod_permInds & mod_inversePermInds
    do PP=1, Nperm
      ll = size(permutedIatoms(PP)%oneDintArr)
      call getPermIatoms(NatomsMax,ll,permutedIatoms(PP)%oneDintArr,mod_permInds(:,PP))
      call getInversePermIatoms(NatomsMax,NatomsMax,mod_permInds(:,PP),mod_inversePermInds(:,PP))
    end do     
  end if 

  ! Calculate mod_ac2dArray 
  !if (option%calcGrad .or. option%calcGradXYZ) then
    do ii=1,NatomsMax 
      mod_ac2dArray(ii,ii) = 0
    end do 
    do ii=1,NatomsMax-1 
      do jj=ii+1,NatomsMax 
        mod_ac2dArray(ii,jj) = (2*NatomsMax-ii)*(ii-1)/2 + jj-ii
        mod_ac2dArray(jj,ii) = mod_ac2dArray(ii,jj)
      end do 
    end do
  !end if

  ! Calculate mod_MaternConsts
  if(option%kernel == 'Matern') then
    const = factorial_rprec(nn)                  &
          / factorial_rprec(2*nn)                &
          / sigma ** 2                           &
          * 2.0_rprec
    do kk=0,nn-1
      mod_MaternConsts(kk+1,1) = factorial_rprec(nn + kk - 1)         &
                               / factorial_rprec(kk)                  &
                               / factorial_rprec(nn - kk)             &
                               * (nn - kk)                            &
                               * (2.0_rprec / sigma) ** (nn - kk - 1) &
                               * const
    end do
    const = factorial_rprec(nn)                  &
          / factorial_rprec(2*nn)                & 
          / sigma ** 4                           &
          * 4.0_rprec
    if (nn > 1) then
      do kk=0,nn-2 
        mod_MaternConsts(kk+1,2) = factorial_rprec(nn + kk - 2)         &
                                / factorial_rprec(kk)                  &
                                / factorial_rprec(nn - kk)             &
                                * (nn - kk)                            &
                                * (nn - kk - 1)                        &
                                * (2.0_rprec / sigma) ** (nn - kk - 2) &
                                * const
      end do
    end if 
  end if 

end subroutine initializeAllArrays

subroutine allocateAllArrays()
  use dataset,       only : Nperm, NatomsMax, permutedIatoms, NitemsTot
  use stopper,       only : stopMLatomF
  implicit none 
  ! Local variables
  integer :: modError=0
  integer :: PP

  ! Allocate mod_ac2dArray
  !if (option%calcGrad .or. option%calcGradXYZ) then
    if (allocated(mod_ac2dArray)) deallocate(mod_ac2dArray)
    allocate(mod_ac2dArray(NatomsMax,NatomsMax),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_ac2dArray array')
  !end if
  ! Allocate mod_MaternConsts
  if (option%kernel == 'Matern') then 
    if (allocated(mod_MaternConsts)) deallocate(mod_MaternConsts)
    allocate(mod_MaternConsts(nn,2),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_MaternConsts array')
  end if

  if (option%permInvKernel) then 
    ! Allocate mod_permInds & mod_inversePermInds
    if (allocated(mod_permInds)) deallocate(mod_permInds)
    allocate(mod_permInds(NatomsMax,Nperm),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_permInds array')
    if (allocated(mod_inversePermInds)) deallocate(mod_inversePermInds)
    allocate(mod_inversePermInds(NatomsMax,Nperm),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_inversePermInds array') 

  end if 

end subroutine allocateAllArrays

subroutine initializeMod_permKvalue()
  use dataset,       only : Nperm, NatomsMax, permutedIatoms, NitemsTot, XvecSize, X, XYZ_A
  use mathUtils,     only : Rij
  use stopper,       only : stopMLatomF
  implicit none 
  real(kind=rprec), allocatable :: temp2DArr(:,:)
  integer :: modError
  integer :: PP, Xsize, lb, ub 
  integer :: ii,jj,dd,a,bb

  if (option%permInvKernel .and. option%calcGradXYZ .and. (trim(option%permInvGroups) /= '' .or. trim(option%permInvNuclei) /= '' .or. option%usePermInd)) then 
    Xsize = XvecSize / Nperm 

    ! Allocate mod_distanceTimesX%OneDintArr & twoDrArr 
    if (.not. allocated(mod_distanceTimesX%oneDintArr)) allocate(mod_distanceTimesX%oneDintArr(NitemsTot),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_distanceTimesX%oneDintArr')
    if (allocated(mod_distanceTimesX%twoDrArr)) deallocate(mod_distanceTimesX%twoDrArr)
    allocate(mod_distanceTimesX%twoDrArr(Xsize,NtrGrXYZ/NatomsMax/3),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_distanceTimesX&twoDrArr')
    
    ! Allocate mod_permKvalue
    if (.not. allocated(mod_permKvalue)) allocate(mod_permKvalue(Nperm),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for mod_permKvalue')
    ! Allocate mod_permKvalue(PP)%oneDintArr & twoDrArr
    do PP=1, Nperm 
      if (.not. allocated(mod_permKvalue(PP)%oneDintArr)) allocate(mod_permKvalue(PP)%oneDintArr(1:NitemsTot),stat=modError)
      if (modError/=0) call stopMLatomF('Unable to allocate space for mod_permKvalue(PP)%oneDintArr')

      if (allocated(mod_permKvalue(PP)%twoDrArr)) deallocate(mod_permKvalue(PP)%twoDrArr)
      allocate(mod_permKvalue(PP)%twoDrArr(NtrGrXYZ/NatomsMax/3,NtrGrXYZ/NatomsMax/3),stat=modError)
      if (modError/=0) call stopMLatomF('Unable to allocate space for mod_permKvalue(PP)%twoDrArr')
    end do 

    allocate(temp2DArr(Xsize,NtrGrXYZ/NatomsMax/3),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for temp2DArr array')

    ! Calculate mod_distanceTimesX%OneDintArr 
    do jj=1, NtrGrXYZ/NatomsMax/3
      mod_distanceTimesX%oneDintArr(itrgrxyz(3*NatomsMax*(jj-1)+1,1)) = jj
    end do 
    ! Calculate mod_distanceTimesX%twoDrArr 
    !$OMP PARALLEL DO PRIVATE(a,bb,dd,ii) &
    !$OMP SHARED(temp2DArr,mod_ac2dArray,X) SCHEDULE(STATIC)
    do a=1, NatomsMax
      do bb=a+1,NatomsMax 
        dd = mod_ac2dArray(a,bb)
        do ii=1, NtrGrXYZ/NatomsMax/3 
          temp2DArr(dd,ii) = X(dd,ii) / Rij(XYZ_A(:,a,ii),XYZ_A(:,bb,ii))**2
        end do 
      end do 
    end do 
    !$OMP END PARALLEL DO
    mod_distanceTimesX%twoDrArr = temp2DArr
    deallocate(temp2DArr)


    allocate(temp2DArr(NtrGrXYZ/NatomsMax/3,NtrGrXYZ/NatomsMax/3),stat=modError)
    if (modError/=0) call stopMLatomF('Unable to allocate space for temp2DArr array')

    do PP=1, Nperm
      lb = 1+Xsize*(PP-1)
      ub = Xsize*PP 
      ! Calculate mod_permKvalue(PP)%oneDintArr 
      do jj=1, NtrGrXYZ/NatomsMax/3 
        !mod_permKvalue(PP)%oneDintArr(itrgrxyz(3*NatomsMax*(jj-1)+1,1)) = jj
        mod_permKvalue(PP)%oneDintArr = mod_distanceTimesX%oneDintArr
      end do
      ! Calculate mod_permKvalue(PP)%twoDrArr 
      mod_permKvalue(PP)%twoDrArr = 0.0_rprec
      !$OMP PARALLEL DO PRIVATE(ii,jj) &
      !$OMP SHARED(temp2DArr,itrgrxyz) SCHEDULE(STATIC)
      do ii=1, NtrGrXYZ/NatomsMax/3 
        temp2DArr(ii,ii) = Kfunk(Xsize,X(1:Xsize,itrgrxyz(3*NatomsMax*(ii-1)+1,1)),X(lb:ub,itrgrxyz(3*NatomsMax*(ii-1)+1,1)))
        !temp2DArr(ii,ii) = calcKMiMj(itrgrxyz(3*NatomsMax*(ii-1)+1,1),itrgrxyz(3*NatomsMax*(ii-1)+1,1))
        do jj=ii+1, NtrGrXYZ/NatomsMax/3 
          temp2DArr(ii,jj) = Kfunk(Xsize,X(1:Xsize,itrgrxyz(3*NatomsMax*(ii-1)+1,1)),X(lb:ub,itrgrxyz(3*NatomsMax*(jj-1)+1,1)))
          !temp2DArr(ii,jj) = calcKMiMj(itrgrxyz(3*NatomsMax*(ii-1)+1,1),itrgrxyz(3*NatomsMax*(jj-1)+1,1))
          temp2DArr(jj,ii) = temp2DArr(ii,jj)
        end do 
      end do
      !$OMP END PARALLEL DO
      mod_permKvalue(PP)%twoDrArr = temp2DArr
    end do 
    
    deallocate(temp2DArr)
  end if 

end subroutine initializeMod_permKvalue

subroutine getPermIatoms(Ninds,Nperms,perms,permInds)                  ! New subroutine
  implicit none 
  ! Arguments
  integer,     intent(in)    :: Ninds, Nperms
  integer,     intent(in)    :: perms(1:Nperms)
  integer,     intent(out)   :: permInds(1:Ninds)
  ! Local variables
  integer                    :: inds(1:Ninds)
  integer                    :: icount, iperm, ii, jj

  icount = 0
  do ii=1,Ninds 
    iperm = ii
    if (any(perms == ii)) then
      icount = icount + 1 
      iperm = perms(icount) 
    end if 
    permInds(ii) = iperm
  end do 

end subroutine getPermIatoms

subroutine getInversePermIatoms(Niinds,Njinds,permInds,inversePermInds)
  implicit none 
  ! Arguments
  integer, intent(in) :: Niinds,Njinds 
  integer, intent(in) :: permInds(Niinds) 
  integer, intent(out) :: inversePermInds(Njinds)
  ! Local variables 
  integer :: ii 

  do ii=1,Njinds
    inversePermInds(permInds(ii)) = ii 
  end do

end subroutine getInversePermIatoms

function calcKMiMj(ireal,jreal)
  use dataset,        only : X, XvecSize, Nperm 
  implicit none 
  ! Arguments
  integer, intent(in) :: ireal, jreal
  ! Return value 
  real(kind=rprec) :: calcKMiMj 
  ! Local variables
  integer :: PP, Xsize, lb, ub

  Xsize = XvecSize / Nperm 
  calcKMiMj = 0.0_rprec
  do PP=1, Nperm 
    lb = 1+Xsize*(PP-1) 
    ub = Xsize*PP 
    calcKMiMj = calcKMiMj + Kfunk(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal))
  end do
end function calcKMiMj

function dKdXid(Xsize,Xi,Xj,dd)
!==============================================================================
! Calculates partial first-order derivative of the kernel function dK(xi,xj)/dxi,d
! Derivatives of exponential and Matern kernels by Yifan Hou, 07.07.2020
!==============================================================================
  use dataset,       only : X, XvecSize, Natoms, Nperm, permutedIatoms
  use stopper,       only : stopMLatomF
  use mathUtils,     only : factorial_rprec
  implicit none
  ! Arguments
  integer,          intent(in) :: Xsize,dd
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  ! Return value
  real(kind=rprec) :: dKdXid
  ! Local variables
  integer :: kk
  real(kind=rprec) :: dist
  real(kind=rprec), allocatable :: xyzAngstrom(:,:) ! Nuclear coordinates in Angstrom
  real(kind=rprec) :: Xph(1:XvecSize), Xmh(1:XvecSize)
  real(kind=rprec) :: h, fxph, fxmh
  integer :: Error
  
  dKdXid = 0.0_rprec

  if (option%kernel == 'Gaussian' .and. .not. option%numDerivs) then
    dKdXid = Kfunk(Xsize,Xi,Xj) &
          * (Xj(dd)-Xi(dd)) / (sigma**2)
  else if (option%kernel == 'Matern' .and. nn > 0 .and. .not. option%numDerivs) then
    dist = distance(Xsize,Xi,Xj)
    do kk=0, nn-1 
      dKdXid = dKdXid + factorial_rprec(nn + kk - 1)        & 
                      / factorial_rprec(kk)                 &
                      / factorial_rprec(nn - kk)            & 
                      * (nn - kk)                           &
                      * (2.0_rprec * dist / sigma) ** (nn - kk - 1) &
                      * (Xj(dd)-Xi(dd)) 
    end do 
    dKdXid = dKdXid * exp(-1.0_rprec * dist / sigma) &
                    * factorial_rprec(nn)            &
                    / factorial_rprec(2*nn)          & 
                    / sigma ** 2                     & 
                    * 2.0_rprec
    !dKdXid = dKdXid_changing(Xsize,Xi,Xj,dd) * dKdXid_const2(Xsize,Xi,Xj) * dKdXid_const1()
  else
    h = sqrt(epsilon(Xi(dd)))
    if (abs(Xi(dd)) > 0.0_rprec+2.0_rprec*epsilon(Xi(dd))) h = h * Xi(dd)
    Xph = Xi
    Xph(dd) = Xph(dd) + h
    Xmh = Xi
    Xmh(dd) = Xmh(dd) - h
    fxph = Kfunk(Xsize,Xph,Xj)
    fxmh = Kfunk(Xsize,Xmh,Xj)
    dKdXid = (fxph - fxmh) / (2.0_rprec * h)
  end if 

end function dKdXid

function dKdXid_const1()
  use mathUtils,     only : factorial_rprec
  implicit none 
  ! Arguments
 
  ! Return value
  real(kind=rprec)                        :: dKdXid_const1

  dKdXid_const1 = 0.0_rprec

  if (option%kernel == 'Gaussian' .and. .not. option%numDerivs) then 
    dKdXid_const1 = 1.0_rprec / sigma**2
  else if (option%kernel == 'Matern' .and. nn > 0 .and. .not. option%numDerivs) then 
    dKdXid_const1 = 1.0_rprec 
  else 
    dKdXid_const1 = 1.0_rprec
  end if 
end function dKdXid_const1

function dKdXid_const2(Xsize,Xi,Xj)
  use mathUtils,     only : factorial_rprec
  implicit none 
  ! Arguments
  integer,          intent(in) :: Xsize
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  ! Return value 
  real(kind=rprec)             :: dKdXid_const2
  ! Local variables 
  integer :: kk 
  real(kind=rprec) :: dist

  dKdXid_const2 = 0.0_rprec

  if (option%kernel == 'Gaussian' .and. .not. option%numDerivs) then 
    dKdXid_const2 = Kfunk(Xsize,Xi,Xj) !/ sigma ** 2
  else if (option%kernel == 'Matern' .and. nn > 0 .and. .not. option%numDerivs) then 
    dist = distance(Xsize,Xi,Xj)
    do kk=0, nn-1 
      dKdXid_const2 = dKdXid_const2 + mod_MaternConsts(kk+1,1) &
                    * dist ** (nn - kk - 1) 
    end do 
    dKdXid_const2 = dKdXid_const2 * exp(-1.0_rprec * dist / sigma)
  else 
    dKdXid_const2 = 1.0_rprec
  end if 

end function dKdXid_const2 

function dKdXid_changing(Xsize,Xi,Xj,dd)
  implicit none 
  ! Arguments 
  integer,          intent(in) :: Xsize, dd
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  ! Return value 
  real(kind=rprec)             :: dKdXid_changing

  if (option%kernel == 'Gaussian' .and. .not. option%numDerivs) then 
    dKdXid_changing = Xj(dd) - Xi(dd)
  else if (option%kernel == 'Matern' .and. nn > 0 .and. .not. option%numDerivs) then 
    dKdXid_changing = Xj(dd) - Xi(dd)
  else 
    dKdXid_changing = dKdXid(Xsize,Xi,Xj,dd)
  end if 

  !dKdXid_changing = Xj(dd) - Xi(dd)

end function dKdXid_changing

function dKdMiat(ireal,jreal,a,tt)
!==============================================================================
! Calculates partial first-order derivative of the kernel function
! in Cartesian coordinates dk(x(Mi),x(Mj))/dMi,at
! Implemented only for analytical derivative of
! the Gaussian kernel function and unsorted RE descriptor
! Derivatives of exponential and Matern kernels by Yifan Hou, 08.07.2020
!==============================================================================
  use constants,     only : Angstrom2Bohr
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, XYZ_B, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer, intent(in) :: ireal,jreal,a,tt
  ! Return value
  real(kind=rprec) :: dKdMiat
  ! Local variables
  real(kind=rprec) :: Kdelta
  integer          :: dd, bb, at1, at2, at3, at4, Error, jj, kk, ll 
  integer          :: Xsize, PP, lb, ub
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj, dKMiMjdMiat, dKMiMidMiat
  
  dKdMiat = 0.0_rprec

  if (option%permInvKernel) then 
      ! Initialize
      KMiMj = 0.0_rprec
      KMiMi = 0.0_rprec
      KMjMj = 0.0_rprec
      dKMiMjdMiat = 0.0_rprec
      dKMiMidMiat = 0.0_rprec
      Xsize = Xvecsize / Nperm
      do PP=1, Nperm
        lb = 1+Xsize*(PP-1)
        ub = Xsize*PP
        KMiMj = KMiMj + Kfunk(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal))
        KMiMi = KMiMi + Kfunk(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal))
        KMjMj = KMjMj + Kfunk(Xsize,X(1:Xsize,jreal),X(lb:ub,jreal))
        dKMiMjdMiat = dKMiMjdMiat +   dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),ireal,jreal, a,tt)

        dKMiMidMiat = dKMiMidMiat + dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal, a,tt) &
                                  + PdKdMjbu(Xsize,X(1:Xsize,ireal),X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal,a,tt,PP)
        !dKMiMidMiat = dKMiMidMiat + 2*dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal, a,tt)
      end do
      dKdMiat = (dKMiMjdMiat - dKMiMidMiat * KMiMj / (2 * KMiMi)) / sqrt(KMiMi*KMjMj) 
  else if (selectPerm) then
    dKdMiat = dKdMiat_selectPerm(ireal,jreal,a,tt) 
  else
    dKdMiat = dKdMiat_unpermuted(XvecSize,X(:,ireal),X(:,jreal),ireal,jreal,a,tt)
  end if

end function dKdMiat

function dKdMiat_usepermKvalue(ireal,jreal,a,tt)
  use constants,     only : Angstrom2Bohr
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, XYZ_B, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer, intent(in) :: ireal,jreal,a,tt
  ! Return value
  real(kind=rprec) :: dKdMiat_usepermKvalue
  ! Local variables
  real(kind=rprec) :: Kdelta
  integer          :: dd, bb, at1, at2, at3, at4, Error, jj, kk, ll 
  integer          :: Xsize, PP, lb, ub
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj, dKMiMjdMiat, dKMiMidMiat
  
  dKdMiat_usepermKvalue = 0.0_rprec

  if (option%permInvKernel .and. option%kernel=='Gaussian' .and. &
      (option%molDescriptor=='RE' .or. option%molDescriptor=='ID') &
      .and. .not. option%numDerivs .and. .false.) then 
    ! Initialize
    KMiMj = 0.0_rprec
    KMiMi = 0.0_rprec
    KMjMj = 0.0_rprec
    dKMiMjdMiat = 0.0_rprec
    dKMiMidMiat = 0.0_rprec
    Xsize = Xvecsize / Nperm
    do PP=1, Nperm
      lb = 1+Xsize*(PP-1)
      ub = Xsize*PP
      KMiMj = KMiMj + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(jreal))
      KMiMi = KMiMi + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(ireal))
      KMjMj = KMjMj + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(jreal),mod_permKvalue(PP)%oneDintArr(jreal))
      dKMiMjdMiat = dKMiMjdMiat + PdKdMiat_useKvalue(Xsize,ireal,jreal,a,tt,PP)

      dKMiMidMiat = dKMiMidMiat + PdKdMiat_useKvalue(Xsize,ireal,ireal,a,tt,PP) &
                                + PdKdMjbu_useKvalue(Xsize,ireal,ireal,a,tt,PP)          
    end do
    dKdMiat_usepermKvalue = (dKMiMjdMiat - dKMiMidMiat * KMiMj / (2 * KMiMi)) / sqrt(KMiMi*KMjMj)
  else if (selectPerm) then
    dKdMiat_usepermKvalue = dKdMiat_selectPerm(ireal,jreal,a,tt)
  else 
    dKdMiat_usepermKvalue = dKdMiat(ireal,jreal,a,tt)
  end if

end function dKdMiat_usepermKvalue

function dKdMiat_unpermuted(Xsize,Xi,Xj,ireal,jreal,a,tt)
!==============================================================================
! Calculates partial first-order derivative of the kernel function
! in Cartesian coordinates dk(x(Mi),x(Mj))/dMi,at
! Implemented only for analytical derivative of
! the Gaussian kernel function and unsorted RE descriptor
! Derivatives of exponential and Matern kernels by Yifan Hou, 08.07.2020
!==============================================================================
  use constants,     only : Angstrom2Bohr
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, XYZ_B, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer,          intent(in) :: Xsize,ireal,jreal,a,tt
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  ! Return value
  real(kind=rprec) :: dKdMiat_unpermuted
  ! Local variables
  real(kind=rprec) :: Kdelta
  integer          :: dd, bb, Error
  real(kind=rprec) :: dist
  real(kind=rprec), allocatable :: xyzAngstrom(:,:) ! Nuclear coordinates in Angstrom
  real(kind=rprec) :: Xph(1:XvecSize), Xmh(1:XvecSize)
  real(kind=rprec) :: h, fxph, fxmh, maxCoord
  
  dKdMiat_unpermuted = 0.0_rprec

  if((option%molDescriptor == 'RE' .or. option%molDescriptor == 'ID') &
     .and. .not. option%numDerivs) then
    do bb=1,Natoms(ireal) 
      if(a /= bb) then 
        dd = mod_ac2dArray(a,bb) 
        dKdMiat_unpermuted = dKdMiat_unpermuted + dKdXid_changing(Xsize,Xi,Xj,dd) &
                           * Xi(dd)                                               & 
                           * (XYZ_A(tt,bb,ireal)-XYZ_A(tt,a,ireal))               &
                           / Rij(XYZ_A(:,a,ireal),XYZ_A(:,bb,ireal))**2 
      end if 
    end do 
    dKdMiat_unpermuted = dKdMiat_unpermuted * dKdXid_const2(Xsize,Xi,Xj) * dKdXid_const1()

  else if(option%molDescriptor == 'CM' .and. .not. option%numDerivs) then 
      do bb=1,Natoms(ireal) 
        if (a /= bb) then
          dd = (a-1)*Natoms(ireal)+bb
          dKdMiat_unpermuted = dKdMiat_unpermuted + dKdXid(Xsize,Xi,Xj,dd) &
                            * dble(Z(a,ireal)) * dble(Z(bb,ireal)) &
                            * (XYZ_A(tt,bb,ireal)-XYZ_A(tt,a,ireal)) &
                            / Rij(XYZ_A(:,a,ireal),XYZ_A(:,bb,ireal))**3
        end if 
      end do
      dKdMiat_unpermuted = dKdMiat_unpermuted * 2.0_rprec / Angstrom2Bohr
  else
    maxCoord = max(1.0_rprec,maxval(XYZ_A))
    h = sqrt(epsilon(maxCoord)) * maxCoord
    ! Allocate arrays
    allocate(xyzAngstrom(1:3,1:NAtomsMax),stat=Error)
    if(Error/=0)call stopMLatomF('Unable to allocate space for xyzAngstrom')
    xyzAngstrom = 0.0_rprec
    Xph  = 0.0_rprec
    xyzAngstrom = XYZ_A(:,:,ireal)
    xyzAngstrom(tt,a) = xyzAngstrom(tt,a) + h
    call calcX(Natoms(ireal), Z(:,ireal), xyzAngstrom, Xph)

    xyzAngstrom = 0.0_rprec
    Xmh  = 0.0_rprec
    xyzAngstrom = XYZ_A(:,:,ireal)
    xyzAngstrom(tt,a) = xyzAngstrom(tt,a) - h
    call calcX(Natoms(ireal), Z(:,ireal), xyzAngstrom, Xmh) 
    fxph = kernelFunction(Xph,Xi)
    fxmh = kernelFunction(Xmh,Xj)
    dKdMiat_unpermuted = (fxph - fxmh) / (2.0_rprec * h)
    ! Free up space
    deallocate(xyzAngstrom)
  end if

end function dKdMiat_unpermuted

function PdKdMiat_useKvalue(Xsize,ireal,jreal,a,tt,PP)
  use dataset,        only: X, Natoms, XYZ_A 
  use mathUtils,      only : Rij
  implicit none 
  ! Arguments 
  integer, intent(in) :: Xsize, ireal, jreal, a, tt, PP 
  ! Return value 
  real(kind=rprec) :: PdKdMiat_useKvalue
  ! Local variable 
  integer :: bb, dd

  PdKdMiat_useKvalue = 0.0_rprec
  do bb=1,Natoms(ireal) 
    if(a /= bb) then 
      dd = mod_ac2dArray(a,bb) 
!      PdKdMiat_useKvalue = PdKdMiat_useKvalue + (X(Xsize*(PP-1)+dd,jreal) - X(dd,ireal)) &
!                         * X(dd,ireal)                                                   & 
!                         * (XYZ_A(tt,bb,ireal)-XYZ_A(tt,a,ireal))                        &
!                         / Rij(XYZ_A(:,a,ireal),XYZ_A(:,bb,ireal))**2 
      PdKdMiat_useKvalue = PdKdMiat_useKvalue + (X(Xsize*(PP-1)+dd,jreal) - X(dd,ireal)) &
                         * mod_distanceTimesX%twoDrArr(dd,mod_distanceTimesX%oneDintArr(ireal))  & 
                         * (XYZ_A(tt,bb,ireal)-XYZ_A(tt,a,ireal))                        

    end if 
  end do 
  PdKdMiat_useKvalue = PdKdMiat_useKvalue * mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(jreal)) * dKdXid_const1()

end function PdKdMiat_useKvalue

function PdKdMjbu_useKvalue(Xsize,ireal,jreal,bb,uu,PP)
  use dataset,        only : X, Natoms, XYZ_A 
  use mathUtils,      only : Rij 
  implicit none 
  ! Arguments 
  integer, intent(in) :: Xsize, ireal, jreal, bb, uu, PP 
  ! Return value 
  real(kind=rprec) :: PdKdMjbu_useKvalue 
  ! Local variable 
  integer :: gg, ee, Pee 

  PdKdMjbu_useKvalue = 0.0_rprec
  do gg=1, Natoms(jreal) 
    if (bb /= gg) then 
      ee = mod_ac2dArray(bb,gg)
      Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))
!      PdKdMjbu_useKvalue = PdKdMjbu_useKvalue + (X(Pee,ireal) - X(ee,jreal)) &
!                         * X(ee,jreal)                                       &
!                         * (XYZ_A(uu,gg,jreal)-XYZ_A(uu,bb,jreal))           &
!                         / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal)) ** 2
      PdKdMjbu_useKvalue = PdKdMjbu_useKvalue + (X(Pee,ireal) - X(ee,jreal)) &
                         * mod_distanceTimesX%twoDrArr(ee,mod_distanceTimesX%oneDintArr(jreal))                                       &
                         * (XYZ_A(uu,gg,jreal)-XYZ_A(uu,bb,jreal))           
                         
    end if 
  end do 
  PdKdMjbu_useKvalue = PdKdMjbu_useKvalue * mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(jreal)) * dKdXid_const1()

end function PdKdMjbu_useKvalue

function PdKdMjbu(Xsize,Xi,Xj,PXj,ireal,jreal,bb,uu,PP)
  use constants,     only : Angstrom2Bohr
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, XYZ_B, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none 
  ! Arguments 
  integer, intent(in) :: Xsize, ireal, jreal, bb, uu, PP 
  real(kind=rprec), intent(in) :: Xi(1:Xsize)
  real(kind=rprec), intent(in) :: Xj(1:Xsize)
  real(kind=rprec), intent(in) :: PXj(1:Xsize)
  ! Return value 
  real(kind=rprec) :: PdKdMjbu 
  ! Local variables 
  integer :: gg, ee, Pee 
  PdKdMjbu = 0.0_rprec 
  if (option%molDescriptor == 'RE' .or. option%molDescriptor == 'ID') then 
    do gg=1,Natoms(jreal)
      if (bb /= gg) then 
        ee = mod_ac2dArray(bb,gg)
        Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))

        !PdKdMjbu = PdKdMjbu + dKdXid_changing(Xsize,PXj,Xi,Pee) &
        !         * Xj(ee) &
        !         * (XYZ_A(uu,gg,jreal)-XYZ_A(uu,bb,jreal)) &
        !         / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal)) ** 2
        PdKdMjbu = PdKdMjbu + dKdXid(Xsize,PXj,Xi,Pee) &
                 * Xj(ee) &
                 * (XYZ_A(uu,gg,jreal)-XYZ_A(uu,bb,jreal)) &
                 / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal)) ** 2

      end if
      !PdKdMjbu = PdKdMjbu * dKdXid_const2(Xsize,PXj,Xi) * dKdXid_const1()
    end do      
  end if  
end function PdKdMjbu

function d2KdXiddXje(Xsize,Xi,Xj,Xid,Xjd,Xie,Xje,dd,ee)
!==============================================================================
! Calculates partial second-order derivative of the kernel function
! in Cartesian coordinates d2k(xi,xj)/dXi,d dXj,e
! Implemented only for analytical derivative of
! the Gaussian kernel function
!==============================================================================
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec                                       
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer,          intent(in) :: Xsize,dd,ee
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  real(kind=rprec), intent(in) :: Xid,Xjd,Xie,Xje
  ! Return value
  real(kind=rprec) :: d2KdXiddXje
  ! Local variables
  integer :: kk                                                                         ! Newly added local variable kk
  integer :: ll 
  real(kind=rprec) :: Kdelta
  real(kind=rprec) :: dist                                                              ! Newly added local variable dist
  real(kind=rprec) :: tempnum1, tempnum2                                                ! Newly added local variable tempnum1, tempnum2
  real(kind=rprec) :: term, part                                                        ! Newly added local variable term, part
  real(kind=rprec), allocatable :: xyzAngstrom(:,:) ! Nuclear coordinates in Angstrom
  real(kind=rprec) :: Xiph(1:XvecSize), Ximh(1:XvecSize), Xjpk(1:XvecSize), Xjmk(1:XvecSize)
  real(kind=rprec) :: h, kkk, fxphpk, fxphmk, fxmhpk, fxmhmk
  integer          :: modError

  d2KdXiddXje = 0.0_rprec


    if (option%kernel == 'Gaussian' .and. .not. option%numDerivs) then
      if (dd == ee) then
        Kdelta = 1.0_rprec
      else
        Kdelta = 0.0_rprec
      end if
      d2KdXiddXje = Kfunk(Xsize,Xi,Xj) &
                  * (Kdelta + (Xjd-Xid) &
                            * (Xie-Xje) &
                            / (sigma**2)) &
                  / (sigma**2)
    else
      h = sqrt(epsilon(Xi(dd)))
      kkk = sqrt(epsilon(Xj(ee)))
      if (abs(Xi(dd)) > 0.0_rprec+2.0_rprec*epsilon(Xi(dd))) h = h * Xi(dd)
      if (abs(Xj(ee)) > 0.0_rprec+2.0_rprec*epsilon(Xj(ee))) kkk = kkk * Xj(ee)
      Xiph = Xi
      Xiph(dd) = Xiph(dd) + h
      Ximh(:) = Xi
      Ximh(dd) = Ximh(dd) - h
      Xjpk(ee) = Xjpk(ee) + kkk 
      Xjmk = Xj
      Xjmk(ee) = Xjmk(ee) - kkk
      fxphpk = Kfunk(Xsize,Xiph,Xjpk)
      fxphmk = Kfunk(Xsize,Xiph,Xjmk)
      fxmhpk = Kfunk(Xsize,Ximh,Xjpk)
      fxmhmk = Kfunk(Xsize,Ximh,Xjmk)    
      d2KdXiddXje = (fxphpk - fxphmk - fxmhpk + fxmhmk) / (4.0_rprec * h * kkk)
    end if
   
end function d2KdXiddXje

function d2KdXiddXje_const1(Xsize,Xi,Xj) 
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec                                       
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none 
  ! Arguments
  integer, intent(in) :: Xsize 
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  ! Return value 
  real(kind=rprec) :: d2KdXiddXje_const1 
  ! Local variables

  !d2KdXiddXje_const1 = 0.0_rprec
  if (option%kernel == 'Gaussian') then 
    d2KdXiddXje_const1 = 1.0_rprec / (sigma**2)
  end if 
end function d2KdXiddXje_const1

function d2KdXiddXje_const2(Xsize,Xi,Xj) 
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec                                       
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none 
  ! Arguments
  integer, intent(in) :: Xsize 
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  ! Return value 
  real(kind=rprec) :: d2KdXiddXje_const2
  ! Local variables
  integer :: kk
  real(kind=rprec) :: dist

  d2KdXiddXje_const2 = 0.0_rprec
  if (option%kernel == 'Gaussian') then 
    d2KdXiddXje_const2 = Kfunk(Xsize,Xi,Xj) !/ (sigma**2)
  end if 
end function d2KdXiddXje_const2

function d2KdXiddXje_const3() 
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec                                       
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none 
  ! Arguments 

  ! Return value 
  real(kind=rprec) :: d2KdXiddXje_const3 
  ! Local variable 

  d2KdXiddXje_const3 = 0.0_rprec
  if (option%kernel == 'Gaussian') then 
    d2KdXiddXje_const3 = 1.0_rprec / (sigma**2)
  end if 
end function d2KdXiddXje_const3

function d2KdXiddXje_changing(Xid,Xjd,Xie,Xje)
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec                                       
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none 
  ! Arguments 
  real(kind=rprec) :: Xid, Xjd, Xie, Xje
  ! Return value 
  real(kind=rprec) :: d2KdXiddXje_changing 

  d2KdXiddXje_changing = (Xjd-Xid)*(Xie-Xje)
  
end function d2KdXiddXje_changing

function d2KdMiatdMjbu(ireal,jreal,a,tt,bb,uu)
!==============================================================================
! Calculates partial second-order derivative of the kernel function
! in Cartesian coordinates d2k(x(Mi),x(Mj))/dMi,at dMj,bu
! Implemented only for analytical derivative of
! the Gaussian kernel function and unsorted RE descriptor
!==============================================================================
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer, intent(in) :: ireal,jreal,a,tt,bb,uu
  ! Return value
  real(kind=rprec) :: d2KdMiatdMjbu
  ! Local variables
  integer :: Xsize, PP 
  integer :: Pdd, Pee
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj, dKMiMjdMiat, dKMiMjdMjbu, dKMiMidMiat, dKMjMjdMjbu, d2KMiMjdMiatdMjbu
  real(kind=rprec) :: tmpdKdMiat, tmpdKdMjbu
  integer          :: dd, ee, at1, at2, at3, at4, cc, gg, kk, ll, lb, ub
  real(kind=rprec) :: dist
  real(kind=rprec) :: part
  real(kind=rprec) :: tempnum1, tempnum2
  integer :: modError 
  real(kind=rprec), allocatable :: temp_XYZ_B(:,:,:)
  integer,          allocatable :: permInds(:)
  integer,          allocatable :: inversePermInds(:)
  
  d2KdMiatdMjbu = 0.0_rprec
  if (option%permInvKernel) then 
      ! Initialize 
      KMiMj = 0.0_rprec
      KMiMi = 0.0_rprec
      KMjMj = 0.0_rprec
      dKMiMjdMiat = 0.0_rprec
      dKMiMjdMjbu = 0.0_rprec
      dKMiMidMiat = 0.0_rprec
      dKMjMjdMjbu = 0.0_rprec
      d2KMiMjdMiatdMjbu = 0.0_rprec
      tmpdKdMiat = 0.0_rprec
      tmpdKdMjbu = 0.0_rprec
      Xsize = XvecSize / Nperm

      do PP=1, Nperm
        lb = 1+Xsize*(PP-1)
        ub = Xsize*PP
        KMiMj = KMiMj + Kfunk(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal))
        KMiMi = KMiMi + Kfunk(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal))
        KMjMj = KMjMj + Kfunk(Xsize,X(1:Xsize,jreal),X(lb:ub,jreal))
        dKMiMjdMiat = dKMiMjdMiat +   dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),ireal,jreal, a,tt)

        dKMiMjdMjbu = dKMiMjdMjbu + PdKdMjbu(Xsize,X(1:Xsize,ireal),X(1:Xsize,jreal),X(lb:ub,jreal),ireal,jreal,bb,uu,PP)
        !dKMiMjdMjbu = dKMiMjdMjbu +   dKdMiat_unpermuted(Xsize,X(1:Xsize,jreal),X(lb:ub,ireal),jreal,ireal,bb,uu)

        dKMiMidMiat = dKMiMidMiat + dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal, a,tt) &
                                  + PdKdMjbu(Xsize,X(1:Xsize,ireal),X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal,a,tt,PP)
        !dKMiMidMiat = dKMiMidMiat + 2*dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal, a,tt)

        dKMjMjdMjbu = dKMjMjdMjbu + dKdMiat_unpermuted(Xsize,X(1:Xsize,jreal),X(lb:ub,jreal),jreal,jreal,bb,uu) &
                                  + PdKdMjbu(Xsize,X(1:Xsize,jreal),X(1:Xsize,jreal),X(lb:ub,jreal),jreal,jreal,bb,uu,PP)                              
        !dKMjMjdMjbu = dKMjMjdMjbu + 2*dKdMiat_unpermuted(Xsize,X(1:Xsize,jreal),X(lb:ub,jreal),jreal,jreal,bb,uu)


        d2KMiMjdMiatdMjbu = d2KMiMjdMiatdMjbu + d2KdMiatdMjbu_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),X(lb:ub,ireal),X(1:Xsize,jreal),ireal,jreal,a,tt,bb,uu,PP)
      end do
      d2KdMiatdMjbu = (d2KMiMjdMiatdMjbu &
                    -  dKMiMjdMjbu*dKMiMidMiat/(2*KMiMi) &
                    -  dKMiMjdMiat*dKMjMjdMjbu/(2*KMjMj) &
                    +  dKMiMidMiat*dKMjMjdMjbu*KMiMj/(4*KMiMi*KMjMj)) &
                    / sqrt(KMiMi*KMjMj)
  else if (selectPerm) then 
    d2KdMiatdMjbu = d2KdMiatdMjbu_selectPerm(ireal,jreal,a,tt,bb,uu)
  else
    d2KdMiatdMjbu = d2KdMiatdMjbu_unpermuted(XvecSize,X(:,ireal),X(:,jreal),X(:,ireal),X(:,jreal),ireal,jreal,a,tt,bb,uu,1)
  end if
  if (allocated(inversePermInds)) deallocate(inversePermInds)
end function d2KdMiatdMjbu

function d2KdMiatdMjbu_usepermKvalue(ireal,jreal,a,tt,bb,uu)
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer, intent(in) :: ireal,jreal,a,tt,bb,uu
  ! Return value
  real(kind=rprec) :: d2KdMiatdMjbu_usepermKvalue
  ! Local variables
  integer :: Xsize, PP 
  integer :: Pdd, Pee
  real(kind=rprec) :: KMiMj, KMiMi, KMjMj, dKMiMjdMiat, dKMiMjdMjbu, dKMiMidMiat, dKMjMjdMjbu, d2KMiMjdMiatdMjbu
  real(kind=rprec) :: tmpdKdMiat, tmpdKdMjbu
  integer          :: dd, ee, at1, at2, at3, at4, cc, gg, kk, ll, lb, ub
  real(kind=rprec) :: dist
  real(kind=rprec) :: part
  real(kind=rprec) :: tempnum1, tempnum2
  integer :: modError 
  real(kind=rprec), allocatable :: temp_XYZ_B(:,:,:)
  integer,          allocatable :: permInds(:)
  integer,          allocatable :: inversePermInds(:)

  d2KdMiatdMjbu_usepermKvalue = 0.0_rprec

  if (option%permInvKernel .and. option%kernel=='Gaussian' .and. &
      (option%molDescriptor=='RE' .or. option%molDescriptor == 'ID') .and. .false.) then 
    ! Initialize 
    KMiMj = 0.0_rprec
    KMiMi = 0.0_rprec
    KMjMj = 0.0_rprec
    dKMiMjdMiat = 0.0_rprec
    dKMiMjdMjbu = 0.0_rprec
    dKMiMidMiat = 0.0_rprec
    dKMjMjdMjbu = 0.0_rprec
    d2KMiMjdMiatdMjbu = 0.0_rprec
    Xsize = XvecSize / Nperm

    do PP=1, Nperm
      lb = 1+Xsize*(PP-1)
      ub = Xsize*PP
      KMiMj = KMiMj + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(jreal))
      KMiMi = KMiMi + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(ireal),mod_permKvalue(PP)%oneDintArr(ireal))
      KMjMj = KMjMj + mod_permKvalue(PP)%twoDrArr(mod_permKvalue(PP)%oneDintArr(jreal),mod_permKvalue(PP)%oneDintArr(jreal))

      dKMiMjdMiat = dKMiMjdMiat + PdKdMiat_useKvalue(Xsize,ireal,jreal,a,tt,PP)
!        dKMiMjdMiat = dKMiMjdMiat +   dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),ireal,jreal, a,tt)


      dKMiMjdMjbu = dKMiMjdMjbu + PdKdMjbu_useKvalue(Xsize,ireal,jreal,bb,uu,PP)
!        dKMiMjdMjbu = dKMiMjdMjbu + PdKdMjbu(Xsize,X(1:Xsize,ireal),X(1:Xsize,jreal),X(lb:ub,jreal),ireal,jreal,bb,uu,PP)


      dKMiMidMiat = dKMiMidMiat + PdKdMiat_useKvalue(Xsize,ireal,ireal,a,tt,PP) &
                                + PdKdMjbu_useKvalue(Xsize,ireal,ireal,a,tt,PP)
!        dKMiMidMiat = dKMiMidMiat + dKdMiat_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal, a,tt) &
!                                  + PdKdMjbu(Xsize,X(1:Xsize,ireal),X(1:Xsize,ireal),X(lb:ub,ireal),ireal,ireal,a,tt,PP)



      dKMjMjdMjbu = dKMjMjdMjbu + PdKdMiat_useKvalue(Xsize,jreal,jreal,bb,uu,PP) &
                                + PdKdMjbu_useKvalue(Xsize,jreal,jreal,bb,uu,PP)    
!        dKMjMjdMjbu = dKMjMjdMjbu + dKdMiat_unpermuted(Xsize,X(1:Xsize,jreal),X(lb:ub,jreal),jreal,jreal,bb,uu) &
!                                  + PdKdMjbu(Xsize,X(1:Xsize,jreal),X(1:Xsize,jreal),X(lb:ub,jreal),jreal,jreal,bb,uu,PP)                          
                                

      d2KMiMjdMiatdMjbu = d2KMiMjdMiatdMjbu + Pd2KdMiatdMjbu_useKvalue(Xsize,ireal,jreal,a,tt,bb,uu,PP)
!        d2KMiMjdMiatdMjbu = d2KMiMjdMiatdMjbu + d2KdMiatdMjbu_unpermuted(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),X(lb:ub,ireal),X(1:Xsize,jreal),ireal,jreal,a,tt,bb,uu,PP)
    end do
    d2KdMiatdMjbu_usepermKvalue = (d2KMiMjdMiatdMjbu &
                  -  dKMiMjdMjbu*dKMiMidMiat/(2*KMiMi) &
                  -  dKMiMjdMiat*dKMjMjdMjbu/(2*KMjMj) &
                  +  dKMiMidMiat*dKMjMjdMjbu*KMiMj/(4*KMiMi*KMjMj)) &
                  / sqrt(KMiMi*KMjMj)
  else if (selectPerm) then 
    d2KdMiatdMjbu_usepermKvalue = d2KdMiatdMjbu_selectPerm(ireal,jreal,a,tt,bb,uu)
  else 
    d2KdMiatdMjbu_usepermKvalue = d2KdMiatdMjbu(ireal,jreal,a,tt,bb,uu)
  end if

end function d2KdMiatdMjbu_usepermKvalue
function d2KdMiatdMjbu_unpermuted(Xsize,Xi,PXj,PXi,Xj,ireal,jreal,a,tt,bb,uu,PP,permInds,inversePermInds)
!==============================================================================
! Calculates partial second-order derivative of the kernel function
! in Cartesian coordinates d2k(x(Mi),x(Mj))/dMi,at dMj,bu
! Implemented only for analytical derivative of
! the Gaussian kernel function and unsorted RE descriptor
!==============================================================================
  use constants,     only : Angstrom2Bohr
  use dataset,       only : X, XvecSize, Natoms, NAtomsMax, XYZ_A, Z, Nperm, permutedIatoms
  use mathUtils,     only : Rij, factorial_rprec
  use molDescr,      only : calcX
  use stopper,       only : stopMLatomF
  implicit none
  ! Arguments
  integer,          intent(in) :: Xsize, ireal,jreal,a,tt,bb,uu,PP
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)   ! X arrays
  real(kind=rprec), intent(in) :: PXi(1:Xsize), PXj(1:Xsize) ! permuted X arrays
  integer, intent(in), optional :: permInds(Natoms(ireal))
  integer, intent(in), optional :: inversePermInds(Natoms(jreal))

  ! Return value
  real(kind=rprec) :: d2KdMiatdMjbu_unpermuted
  ! Local variables
  integer          :: dd, ee, at1, at2, at3, at4, cc, gg, kk, ll, Pdd, Pee
  integer :: Pa, Pc, Pddm, permIatoms(1:NAtomsMax), iperm, icount, ii
  real(kind=rprec) :: dist
  real(kind=rprec) :: part
  real(kind=rprec) :: tempnum1, tempnum2, dXdM,dXdMi,dXdMj
  real(kind=rprec) :: in_tempnum1, in_tempnum2, out_tempnum1, out_tempnum2
  integer :: modError 
  real(kind=rprec), allocatable :: temp_XYZ_B(:,:,:)
  
  d2KdMiatdMjbu_unpermuted = 0.0_rprec

  tempnum1 = 0.0_rprec
  tempnum2 = 0.0_rprec
  out_tempnum1 = 0.0_rprec
  out_tempnum2 = 0.0_rprec
  
  
  if (option%molDescriptor == 'RE' .or. option%molDescriptor == 'ID') then  
    if (option%permInvKernel) then 
      do cc=1,Natoms(ireal) 
        if (a /= cc) then 
          dd = mod_ac2dArray(a,cc)
          do gg=1,Natoms(jreal) 
            if (bb /= gg) then 
              ee = mod_ac2dArray(bb,gg)
              
              Pdd = mod_ac2dArray(mod_permInds(a,PP),mod_permInds(cc,PP))
              Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))
              
              d2KdMiatdMjbu_unpermuted = d2KdMiatdMjbu_unpermuted &
                            + d2KdXiddXje(Xsize,Xi,PXj,Xi(dd),PXj(dd),Xi(Pee),Xj(ee),Pdd,ee) &
                            * Xi(dd) * (XYZ_A(tt,cc,ireal) - XYZ_A(tt,a,ireal)) &
                            / Rij(XYZ_A(:,a,ireal),XYZ_A(:,cc,ireal))**2 &
                            * Xj(ee) * (XYZ_A(uu,gg,jreal) - XYZ_A(uu,bb,jreal)) &
                            / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal))**2
         
            end if 
          end do 
        end if 
      end do 

    else 

      do  cc=1,Natoms(ireal) 
        if (a /= cc) then 
          dd = mod_ac2dArray(a,cc)
          dXdMi = Xi(dd) * (XYZ_A(tt,cc,ireal) - XYZ_A(tt,a,ireal)) &
                          / Rij(XYZ_A(:,a,ireal),XYZ_A(:,cc,ireal))**2 
          in_tempnum1 = 0.0_rprec
          in_tempnum2 = 0.0_rprec
          do gg=1,Natoms(jreal) 
            if (bb /= gg) then 
              ee = mod_ac2dArray(bb,gg)
              dXdMj = Xj(ee) * (XYZ_A(uu,gg,jreal) - XYZ_A(uu,bb,jreal)) &
                             / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal))**2 
              in_tempnum1 = in_tempnum1 + dXdMj &
                          * d2KdXiddXje_changing(Xi(dd),Xj(dd),Xi(ee),Xj(ee))
              if (dd == ee) then 
                in_tempnum2 = in_tempnum2 + dXdMj 
              end if 
            end if 
          end do 
          out_tempnum1 = out_tempnum1 + in_tempnum1 * dXdMi 
          out_tempnum2 = out_tempnum2 + in_tempnum2 * dXdMi 
        end if 
      end do 
      out_tempnum1 = out_tempnum1 * d2KdXiddXje_const1(Xsize,Xi,Xj)
      d2KdMiatdMjbu_unpermuted = (out_tempnum1 + out_tempnum2) * d2KdXiddXje_const2(Xsize,Xi,Xj) * d2KdXiddXje_const3()
      
    end if 

  else if (option%molDescriptor == 'CM') then 
    do cc=1,Natoms(ireal) 
      if (a/=cc) then 
        dd = (a-1)*Natoms(ireal)+cc
        do gg=1,Natoms(jreal)
          if(bb/=gg) then 
            ee = (bb-1)*Natoms(jreal)+gg
            d2KdMiatdMjbu_unpermuted = d2KdMiatdMjbu_unpermuted &
                                     + d2KdXiddXje(Xsize,Xi,Xj,Xi(dd),Xj(dd),Xi(ee),Xj(ee),dd,ee) &
                                     * dble(Z(a,ireal)) * dble(Z(cc,ireal)) &
                                     * (XYZ_A(tt,cc,ireal)-XYZ_A(tt,a,ireal)) &
                                     / Rij(XYZ_A(:,a,ireal),XYZ_A(:,cc,ireal)) ** 3 &
                                     * dble(Z(bb,jreal)) * dble(Z(gg,jreal)) &
                                     * (XYZ_A(uu,gg,jreal)-XYZ_A(uu,bb,jreal)) &
                                     / Rij(XYZ_A(:,bb,jreal),XYZ_A(:,gg,jreal)) ** 3
          end if 
        end do 
      end if 
    end do
    d2KdMiatdMjbu_unpermuted = d2KdMiatdMjbu_unpermuted * 4.0_rprec 
  else
    call stopMLatomF('Can only run this type of calculations with the Gaussian/Matern kernel, unsorted/permuted RE, ID, or CM descriptor')
  end if
end function d2KdMiatdMjbu_unpermuted

function Pd2KdMiatdMjbu_useKvalue(Xsize,ireal,jreal,a,tt,bb,uu,PP) 
  use dataset,        only : X, Natoms, XYZ_A 
  use mathUtils,      only : Rij
  implicit none 
  ! Arguments
  integer, intent(in) :: Xsize, ireal, jreal, a, tt, bb, uu, PP 
  ! Return value 
  real(kind=rprec) :: Pd2KdMiatdMjbu_useKvalue
  ! Local variables 
  integer :: cc, gg, dd, ee, Pdd, Pee, lb, ub

  Pd2KdMiatdMjbu_useKvalue = 0.0_rprec
  lb = (PP-1)*Xsize+1
  ub = PP*Xsize

  do cc=1,Natoms(ireal) 
    if (a /= cc) then 
      dd = mod_ac2dArray(a,cc)
      Pdd = mod_ac2dArray(mod_permInds(a,PP),mod_permInds(cc,PP))
      do gg=1,Natoms(jreal) 
        if (bb /= gg) then 
          ee = mod_ac2dArray(bb,gg)
          Pee = mod_ac2dArray(mod_inversePermInds(bb,PP),mod_inversePermInds(gg,PP))

          Pd2KdMiatdMjbu_useKvalue = Pd2KdMiatdMjbu_useKvalue &
                        + d2KdXiddXje(Xsize,X(1:Xsize,ireal),X(lb:ub,jreal),X(dd,ireal),X(lb+dd-1,jreal),X(Pee,ireal),X(ee,jreal),Pdd,ee) &
                        * mod_distanceTimesX%twoDrArr(dd,mod_distanceTimesX%oneDintArr(ireal)) &
                        * mod_distanceTimesX%twoDrArr(ee,mod_distanceTimesX%oneDintArr(jreal)) &
                        * (XYZ_A(tt,cc,ireal) - XYZ_A(tt,a,ireal)) &
                        * (XYZ_A(uu,gg,jreal) - XYZ_A(uu,bb,jreal))

        end if 
      end do 
    end if 
  end do

end function Pd2KdMiatdMjbu_useKvalue

function distance(Xsize,Xi,Xj)
  implicit none
  real(kind=rprec)             :: distance
  integer,          intent(in) :: Xsize
  real(kind=rprec), intent(in) :: Xi(1:Xsize), Xj(1:Xsize) ! X arrays
  ! Local variables
  integer :: kk

  distance = 0.0_rprec

  if (trim(option%kernel) == 'linear') then
    do kk=1,Xsize
      distance = distance + Xi(kk)*Xj(kk)
    end do
  elseif (trim(option%kernel) == 'polynomial') then
    do kk=1,Xsize
      distance = distance + Xi(kk)*Xj(kk)
    end do
  elseif (trim(option%kernel) == 'Gaussian'    .or. &
          trim(option%kernel) == 'exponential' .or. &
          trim(option%kernel) == 'Matern') then
    do kk=1,Xsize
      distance = distance + (Xi(kk) - Xj(kk)) ** 2
    end do
    if (trim(option%kernel) == 'exponential' .or. &
        trim(option%kernel) == 'Matern') then
      distance = sqrt(distance)
    end if
  elseif (trim(option%kernel) == 'Laplacian') then
    do kk=1,Xsize
      distance = distance + abs(Xi(kk) - Xj(kk))
    end do
  end if

end function distance

end module A_KRR_kernel
