module kreg 
  use stopper, only : stopKREG, raiseWarning
  implicit none 
  ! ! Options 
  ! logical :: useVal=.True., useGradXYZ=.False.
  ! logical :: calcVal=.True., calcGradXYZ=.False.
  ! ! Hyperparameters
  ! real(kind=8) :: sigma=1.0, lambdav=0.0001, lambdagradxyz=0.0001

  ! integer :: Natoms 
  ! integer :: Ntrain
  ! integer :: Npredict
  ! integer :: Ksize
  ! integer :: NtrVal, NtrGrXYZ
  ! integer, allocatable :: ac2dArray(:,:)
  ! real(kind=8), allocatable :: XYZ(:,:,:)
  ! real(kind=8), allocatable :: XYZpredict(:,:,:)
  ! real(kind=8), allocatable :: X(:,:)
  ! real(kind=8), allocatable :: Xpredict(:,:)
  ! real(kind=8), allocatable :: Xeq(:), Req(:,:)
  ! real(kind=8), allocatable :: K(:,:)
  ! real(kind=8), allocatable :: alpha(:,:)
  ! real(kind=8), allocatable :: Yref(:)
  ! real(kind=8), allocatable :: YgradXYZref(:,:,:)
  ! real(kind=8), allocatable :: Ytrain(:)
  ! real(kind=8), allocatable :: Yest(:)
  ! real(kind=8), allocatable :: YgradXYZest(:,:,:)
  ! real(kind=8), allocatable :: Kprediction(:,:)
  ! integer, allocatable :: itrgrxyz(:,:)
  ! integer :: Nprops = 1

contains

subroutine get_ac2dArray(Natoms,Xsize,ac2dArray) 
  implicit none 
  ! Arguments
  integer, intent(in) :: Natoms,Xsize
  integer, intent(inout) :: ac2dArray(1:Natoms,1:Natoms)
  ! Local variables
  integer :: ii, jj

  ! Calculate ac2dArray
  do ii=1, Natoms 
    ac2dArray(ii,ii) = 0 
  end do 
  do ii=1, Natoms-1 
    do jj=ii+1,Natoms 
      ac2dArray(ii,jj) = (2*Natoms-ii)*(ii-1)/2 + jj-ii 
      ac2dArray(jj,ii) = ac2dArray(ii,jj)
    end do 
  end do 
end subroutine get_ac2dArray

function distance2(Xsize,Xi,Xj)
  implicit none
  real(kind=8) :: distance2 
  integer, intent(in) :: Xsize 
  real(kind=8), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  ! Local variables 
  integer :: ii 

  distance2 = 0.0
  do ii=1, Xsize
    distance2 = distance2 + (Xi(ii) - Xj(ii)) ** 2
  end do 
end function distance2 

function distance(Xsize,Xi,Xj)
  implicit none
  real(kind=8) :: distance
  integer, intent(in) :: Xsize 
  real(kind=8), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)

  distance = sqrt(distance2(Xsize,Xi,Xj))
end function distance 

function Rij(xyzI, xyzJ)
  implicit none 
  real(kind=8) :: Rij
  ! Arguments
  real(kind=8), intent(in) :: xyzI(1:3)
  real(kind=8), intent(in) :: xyzJ(1:3)
  
  Rij = sqrt((xyzI(1)-xyzJ(1))**2+(xyzI(2)-xyzJ(2))**2+(xyzI(3)-xyzJ(3))**2)
end function Rij 

function kernel(Xsize, Xi, Xj, sigma)
  implicit none
  real(kind=8) :: kernel
  integer, intent(in) :: Xsize
  real(kind=8), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  real(kind=8), intent(in) :: sigma

  kernel = exp(-1.0 * distance2(Xsize,Xi,Xj) / (2.0*sigma**2))
end function kernel 

! function dKdXid(Xsize,Xi,Xj,dd,sigma)
!   implicit none 
!   real(kind=8) :: dKdXid 
!   integer, intent(in) :: Xsize, dd 
!   real(kind=8) :: Xi(1:Xsize), Xj(1:Xsize)
!   real(kind=8), intent(in) :: sigma
!   dKdXid = kernel(Xsize,Xi,Xj) * (Xi(dd) - Xj(dd)) / (sigma**2)
! end function dKdXid 

function dKdMiat(Natoms,Xsize,a,tt,Xi,Xj,XYZi,ac2dArray,sigma)
  implicit none 
  real(kind=8) :: dKdMiat 
  integer, intent(in) :: Natoms, Xsize, a,tt 
  real(kind=8), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  real(kind=8), intent(in) :: XYZi(1:3,1:Natoms)
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: sigma
  ! Local variables 
  integer :: dd, bb

  dKdMiat = 0.0
  do bb=1, Natoms 
    if (a /= bb) then 
      dd = ac2dArray(a,bb)
      dKdMiat = dKdMiat + (Xj(dd)-Xi(dd)) * Xi(dd) * (XYZi(tt,bb)-XYZi(tt,a)) / Rij(XYZi(:,a),XYZi(:,bb))**2
    end if 
  end do 
  dKdMiat = dKdMiat * kernel(Xsize,Xi,Xj,sigma) / sigma**2 
end function dKdMiat

function d2KdMiatdMjbu(Natoms,Xsize,a,tt,bb,uu,Xi,Xj,XYZi,XYZj,ac2dArray,sigma)
  implicit none 
  real(kind=8) :: d2KdMiatdMjbu 
  integer, intent(in) :: Natoms,Xsize,a,tt,bb,uu 
  real(kind=8), intent(in) :: Xi(1:Xsize), Xj(1:Xsize)
  real(kind=8), intent(in) :: XYZi(1:3,1:Natoms),XYZj(1:3,1:Natoms)
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: sigma
  ! Local variables
  integer :: dd, ee, cc, gg, kk, ll
  real(kind=8) :: tempnum1, tempnum2, dXdM,dXdMi,dXdMj 
  real(kind=8) :: in_tempnum1, in_tempnum2, out_tempnum1, out_tempnum2
  tempnum1 = 0.0 
  tempnum2 = 0.0
  out_tempnum1 = 0.0
  out_tempnum2 = 0.0
  do cc=1, Natoms 
    if (a /= cc) then 
      dd = ac2dArray(a,cc)
      dXdMi = Xi(dd) * (XYZi(tt,cc) - XYZi(tt,a)) / Rij(XYZi(:,a),XYZi(:,cc))**2 
      in_tempnum1 = 0.0 
      in_tempnum2 = 0.0 
      do gg=1, Natoms 
        if (bb/=gg) then 
          ee = ac2dArray(bb,gg)
          dXdMj = Xj(ee) * (XYZj(uu,gg) - XYZj(uu,bb)) / Rij(XYZj(:,bb),XYZj(:,gg))**2 
          in_tempnum1 = in_tempnum1 + dXdMj * (Xj(dd)-Xi(dd))*(Xi(ee)-Xj(ee))
          if (dd == ee) then 
            in_tempnum2 = in_tempnum2 + dXdMj 
          end if
        end if 
      end do 
      out_tempnum1 = out_tempnum1 + in_tempnum1 * dXdMi 
      out_tempnum2 = out_tempnum2 + in_tempnum2 * dXdMi 
    end if 
  end do 
  out_tempnum1 = out_tempnum1 / (sigma**2)
  d2KdMiatdMjbu = (out_tempnum1+out_tempnum2) * kernel(Xsize,Xi,Xj,sigma) / (sigma**2)
end function d2KdMiatdMjbu 

subroutine get_Xeq(Natoms,Xsize,ac2dArray,Req,Xeq)
  implicit none 
  ! Arguments
  integer, intent(in) :: Natoms, Xsize
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: Req(1:3,1:Natoms)
  real(kind=8), intent(out) :: Xeq(1:Xsize)
  ! Local variables 
  integer :: ii, jj, icount, Error

  do ii=1, Natoms 
    do jj=ii+1, Natoms 
      Xeq(ac2dArray(ii,jj)) = Rij(Req(:,ii),Req(:,jj))
    end do 
  end do 

end subroutine get_Xeq

subroutine RE_descriptor(Xsize,Natoms,ac2dArray,RE,xyzI,Xeq)
  implicit none 
  integer, intent(in) :: Xsize,Natoms
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(out) :: RE(1:Xsize)
  real(kind=8), intent(in) :: xyzI(1:3,1:Natoms)
  real(kind=8), intent(in) :: Xeq(1:Xsize)
  ! Local variables
  integer :: ii, jj 
  do ii=1, Natoms 
    do jj=ii+1, Natoms 
      RE(ac2dArray(ii,jj)) = Xeq(ac2dArray(ii,jj)) / Rij(xyzI(:,ii),xyzI(:,jj))
    end do 
  end do 
end subroutine RE_descriptor

subroutine calc_RE_descriptors(Xsize,Natoms,Npoints,ac2dArray,XYZ,X,Xeq) 
  implicit none 
  integer, intent(in) :: Xsize, Npoints,Natoms
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: XYZ(1:3,1:Natoms,1:Npoints)
  real(kind=8), intent(out) :: X(1:Xsize,1:Npoints)
  real(kind=8), intent(in) :: Xeq(1:Xsize)
  ! Local variables
  integer :: ii, jj 
  integer :: ipoint 
  integer :: Error
  do ipoint=1, Npoints
    call RE_descriptor(Xsize,Natoms,ac2dArray, X(:,ipoint),XYZ(:,:,ipoint),Xeq)
  end do 
end subroutine calc_RE_descriptors

subroutine get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms,itrgrxyz)
  implicit none  
  integer, intent(in) :: NtrGrXYZ,Ntrain,Natoms 
  integer, intent(out) :: itrgrxyz(1:NtrGrXYZ,3)
  ! Local variables 
  integer :: icount,ip,ii,jj 
  icount = 0
  do ip=1, Ntrain
    do ii=1, Natoms
      do jj=1, 3 
        icount = icount + 1
        itrgrxyz(icount,1) = ip  
        itrgrxyz(icount,2) = ii 
        itrgrxyz(icount,3) = jj
      end do 
    end do 
  end do
  
end subroutine get_itrgrxyz

subroutine calc_kernel_matrix(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,ac2dArray,XYZ,X,K,sigma) 
  implicit none 
  ! Arguments 
  integer, intent(in) :: Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim
  integer, intent(in) :: itrgrxyz(1:itrgrxyzDim,1:3)
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: XYZ(1:3,1:Natoms,1:Ntrain)
  real(kind=8), intent(in) :: X(1:Xsize,1:Ntrain)
  real(kind=8), intent(out) :: K(1:NtrVal+NtrGrXYZ,1:NtrVal+NtrGrXYZ)
  real(kind=8), intent(in) :: sigma

  ! Local variables 
  integer :: ii, jj, icount, ip
  integer :: Error, Ksize

  Ksize = NtrVal+NtrGrXYZ
  K = 0.0

  ! Covariances between values
  !$OMP PARALLEL DO PRIVATE(ii,jj) &
  !$OMP SHARED(K) SCHEDULE(STATIC)
  do ii=1, NtrVal 
    K(ii,ii) = kernel(Xsize,X(:,ii),X(:,ii),sigma)
    do jj=ii+1, NtrVal 
      K(ii,jj) = kernel(Xsize,X(:,ii),X(:,jj),sigma)
      K(jj,ii) = K(ii,jj)
    end do 
  end do 
  !$OMP END PARALLEL DO

  ! Covariances between values and gradients 
  !$OMP PARALLEL DO PRIVATE(ii,jj) &
  !$OMP SHARED(K,itrgrxyz) SCHEDULE(STATIC)
  do ii=1, NtrVal 
    do jj=1, NtrGrXYZ
      K(ii,NtrVal+jj) = dKdMiat(Natoms,Xsize,itrgrxyz(jj,2),itrgrxyz(jj,3),X(:,itrgrxyz(jj,1)),X(:,ii),XYZ(:,:,itrgrxyz(jj,1)),ac2dArray,sigma)
      K(NtrVal+jj,ii) = K(ii,NtrVal+jj)
    end do 
  end do 
  !$OMP END PARALLEL DO

  ! Covariances between gradients
  !$OMP PARALLEL DO PRIVATE(ii,jj) &
  !$OMP SHARED(K,itrgrxyz) SCHEDULE(STATIC) 
  do ii=1, NtrGrXYZ 
    K(NtrVal+ii,NtrVal+ii) = d2KdMiatdMjbu(Natoms,Xsize,itrgrxyz(ii,2),itrgrxyz(ii,3),itrgrxyz(ii,2),itrgrxyz(ii,3),X(:,itrgrxyz(ii,1)),X(:,itrgrxyz(ii,1)),XYZ(:,:,itrgrxyz(ii,1)),XYZ(:,:,itrgrxyz(ii,1)),ac2dArray,sigma)
    do jj=(ii+1), NtrGrXYZ 
      K(NtrVal+ii,NtrVal+jj) = d2KdMiatdMjbu(Natoms,Xsize,itrgrxyz(ii,2),itrgrxyz(ii,3),itrgrxyz(jj,2),itrgrxyz(jj,3),X(:,itrgrxyz(ii,1)),X(:,itrgrxyz(jj,1)),XYZ(:,:,itrgrxyz(ii,1)),XYZ(:,:,itrgrxyz(jj,1)),ac2dArray,sigma)
      K(NtrVal+jj,NtrVal+ii) = K(NtrVal+ii,NtrVal+jj)
    end do 
  end do 
  !$OMP END PARALLEL DO 

end subroutine calc_kernel_matrix

function Yest_KRR(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,Xipredict,XYZ,X,alpha,ac2dArray,sigma)
  implicit none 
  integer, intent(in) :: Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim
  integer, intent(in) :: itrgrxyz(1:itrgrxyzDim,1:3)
  real(kind=8), intent(in) :: Xipredict(1:Xsize),XYZ(1:3,1:Natoms,1:Ntrain),X(1:Xsize,1:Ntrain),alpha(1:NtrVal+NtrGrXYZ,1)
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: sigma
  real(kind=8) :: Yest_KRR 
  ! Local variables 
  integer :: jj
  Yest_KRR = 0.0
  do jj=1, NtrVal 
    Yest_KRR = Yest_KRR + alpha(jj,1)*kernel(Xsize,Xipredict,X(:,jj),sigma)
  end do 
  do jj=1, NtrGrXYZ 
    Yest_KRR = Yest_KRR + alpha(NtrVal+jj,1)*dKdMiat(Natoms,Xsize,itrgrxyz(jj,2),itrgrxyz(jj,3),X(:,itrgrxyz(jj,1)),Xipredict,XYZ(:,:,itrgrxyz(jj,1)),ac2dArray,sigma)
  end do

end function Yest_KRR 

function YgradXYZest_KRR(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,Xipredict,XYZipredict,XYZ,X,alpha,ac2dArray,sigma) 
  implicit none 
  integer, intent(in) :: Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim
  integer, intent(in) :: itrgrxyz(1:itrgrxyzDim,1:3)
  real(kind=8), intent(in) :: Xipredict(1:Xsize),XYZipredict(1:3,1:Natoms),XYZ(1:3,1:Natoms,1:Ntrain),X(1:Xsize,1:Ntrain),alpha(1:NtrVal+NtrGrXYZ,1)
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: sigma
  real(kind=8) :: YgradXYZest_KRR(1:3,1:Natoms)
  ! Local varibales 
  integer :: jj,a,tt 

  YgradXYZest_KRR = 0.0
  do a=1, Natoms 
    do tt=1, 3 
      do jj=1, NtrVal 
        YgradXYZest_KRR(tt,a) = YgradXYZest_KRR(tt,a) + alpha(jj,1)*dKdMiat(Natoms,Xsize,a,tt,Xipredict,X(:,jj),XYZipredict,ac2dArray,sigma)
      end do 
      do jj=1, NtrGrXYZ 
        YgradXYZest_KRR(tt,a) = YgradXYZest_KRR(tt,a) + alpha(NtrVal+jj,1)*d2KdMiatdMjbu(Natoms,Xsize,a,tt,itrgrxyz(jj,2),itrgrxyz(jj,3),Xipredict,X(:,itrgrxyz(jj,1)),XYZipredict,XYZ(:,:,itrgrxyz(jj,1)),ac2dArray,sigma)
      end do 
    end do 
  end do

end function YgradXYZest_KRR

! subroutine calc_estimation()
!   implicit none 
!   ! Local variables
!   integer :: ii,jj,a,tt, Error
!   integer :: Xsize 

!   Xsize = Natoms*(Natoms-1)/2

!   ! if (allocated(Yest)) then 
!   !   deallocate(Yest)
!   ! end if 
!   ! allocate(Yest(1:Npredict), stat=Error)
!   ! if (Error/=0) then 
!   !   call stopKREG('Unable to allocate space for Yest')
!   ! end if 
!   ! Yest = 0.0
!   ! do ii=1, Npredict 
!   !   do jj=1, Ntrain
!   !     Yest(ii) = Yest(ii) + alpha(jj,1)*Kprediction(jj,ii)
!   !   end do 
!   ! end do 

!   if (calcVal) then 
!     if (allocated(Yest)) then 
!       deallocate(Yest) 
!     end if 
!     allocate(Yest(1:Npredict),stat=Error) 
!     if (Error/=0) then 
!       call stopKREG('Unable to allocate space for Yest')
!     end if 
!     Yest = 0.0
!     !$OMP PARALLEL DO PRIVATE(ii) SHARED(Yest) SCHEDULE(STATIC)
!     do ii=1, Npredict 
!       ! do jj=1, NtrVal 
!       !   Yest(ii) = Yest(ii) + alpha(jj,1)*kernel(Xsize,Xpredict(:,ii),X(:,jj))
!       ! end do 
!       ! do jj=1, NtrGrXYZ 
!       !   Yest(ii) = Yest(ii) + alpha(NtrVal+jj,1)*dKdMiat(Xsize,itrgrxyz(jj,1),ii,itrgrxyz(jj,2),itrgrxyz(jj,3))
!       ! end do
!       Yest(ii) = Yest_KRR(ii)
!     end do 
!     !$OMP END PARALLEL DO
!   end if 

!   if (calcGradXYZ) then 
!     if (allocated(YgradXYZest)) then 
!       deallocate(YgradXYZest)
!     end if 
!     allocate(YgradXYZest(1:3,1:Natoms,1:Npredict),stat=Error)
!     if (Error/=0) then 
!       call stopKREG('Unable to allocate space for YgradXYZest')
!     end if 
!     YgradXYZest = 0.0
!     !$OMP PARALLEL DO PRIVATE(ii) SHARED(YgradXYZest) SCHEDULE(STATIC)
!     do ii=1, Npredict 
!       ! do a=1, Natoms 
!       !   do tt=1, 3 
!       !     do jj=1, NtrVal 
!       !       YgradXYZest(tt,a,ii) = YgradXYZest(tt,a,ii) + alpha(jj,1)*dKdMiat(Xsize,ii,jj,a,tt)
!       !     end do 
!       !     do jj=1, NtrGrXYZ 
!       !       YgradXYZest(tt,a,ii) = YgradXYZest(tt,a,ii) + alpha(NtrVal+jj,1)*d2KdMiatdMjbu(Xsize,ii,itrgrxyz(jj,1),a,tt,itrgrxyz(jj,2),itrgrxyz(jj,3))
!       !     end do 
!       !   end do 
!       ! end do
!       YgradXYZest(:,:,ii) = YgradXYZest_KRR(ii,Natoms) 
!     end do 
!     !$OMP END PARALLEL DO
!   end if 


! end subroutine calc_estimation

subroutine train(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,ac2dArray,XYZ,X,Ytrain,K,alpha,sigma,lambdav,lambdagradxyz,calcKernel)
  implicit none 
  ! Arguments
  integer, intent(in) :: Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: XYZ(1:3,1:Natoms,1:Ntrain)
  real(kind=8), intent(in) :: X(1:Xsize,1:Ntrain)
  real(kind=8), intent(in) :: Ytrain(1:NtrVal+NtrGrXYZ)
  real(kind=8), intent(out) :: K(1:NtrVal+NtrGrXYZ,1:NtrVal+NtrGrXYZ)
  real(kind=8), intent(out) :: alpha(1:NtrVal+NtrGrXYZ,1)
  real(kind=8), intent(in) :: sigma, lambdav, lambdagradxyz
  logical, intent(in) :: calcKernel
  ! Local variables
  integer              :: itrgrxyzDim, Error
  integer, allocatable :: itrgrxyz(:,:)

  ! Get itrgrxyz necessary for dealing with derivatives
  if (NtrGrXYZ /= 0) then 
    allocate(itrgrxyz(1:NtrGrXYZ,3),stat=Error)
    if (Error/=0) call stopKREG('Unable to allocate itrgrxyz')
    call get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms,itrgrxyz)
    itrgrxyzDim = NtrGrXYZ 
  else 
    ! If NtrGrXYZ == 0, allocate a trivial itrgrxyz array
    allocate(itrgrxyz(1,3),stat=Error)
    if (Error/=0) call stopKREG('Unable to allocate itrgrxyz')
    itrgrxyz = 0
    itrgrxyzDim = 1
  end if 

  ! Calculate Kernel matrix
  if (calcKernel) then
    call calc_kernel_matrix(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,ac2dArray,XYZ,X,K,sigma) 
  end if
  ! Calculate regression coefficients 
  call calcAlpha(NtrVal,NtrGrXYZ,NtrVal+NtrGrXYZ,K,Ytrain,alpha,lambdav,lambdagradxyz)

end subroutine train 

subroutine predict(Natoms,Xsize,Ntrain,Npredict,NtrVal,NtrGrXYZ,ac2dArray,X,XYZ,Xpredict,XYZpredict,alpha,calcVal,calcGradXYZ,Yest,YgradXYZest,sigma)
  implicit none 
  ! Arguments
  integer, intent(in) :: Natoms,Xsize,Ntrain,Npredict,NtrVal,NtrGrXYZ
  integer, intent(in) :: ac2dArray(1:Natoms,1:Natoms)
  real(kind=8), intent(in) :: X(1:Xsize,1:Ntrain)
  real(kind=8), intent(in) :: XYZ(1:3,1:Natoms,1:Ntrain)
  real(kind=8), intent(in) :: Xpredict(1:Xsize,1:Npredict)
  real(kind=8), intent(in) :: XYZpredict(1:3,1:Natoms,1:Npredict)
  real(kind=8), intent(in) :: alpha(1:NtrVal+NtrGrXYZ,1)
  logical, intent(in) :: calcVal, calcGradXYZ
  real(kind=8), intent(out) :: Yest(1:Npredict)
  real(kind=8), intent(out) :: YgradXYZest(1:3,1:Natoms,1:Npredict)
  real(kind=8), intent(in) :: sigma

  ! Local variables
  integer :: ii,jj,a,tt, Error
  integer              :: itrgrxyzDim
  integer, allocatable :: itrgrxyz(:,:)

  ! Get itrgrxyz necessary for dealing with derivatives
  if (NtrGrXYZ /= 0) then 
    allocate(itrgrxyz(1:NtrGrXYZ,3),stat=Error)
    if (Error/=0) call stopKREG('Unable to allocate itrgrxyz')
    call get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms,itrgrxyz)
    itrgrxyzDim = NtrGrXYZ 
  else 
    ! If NtrGrXYZ == 0, allocate a trivial itrgrxyz array
    allocate(itrgrxyz(1,3),stat=Error)
    if (Error/=0) call stopKREG('Unable to allocate itrgrxyz')
    itrgrxyz = 0
    itrgrxyzDim = 1
  end if 

  Yest = 0.0
  YgradXYZest = 0.0

  if (calcVal) then 
    !$OMP PARALLEL DO PRIVATE(ii) SHARED(Yest) SCHEDULE(STATIC)
    do ii=1, Npredict 
      Yest(ii) = Yest_KRR(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,Xpredict(:,ii),XYZ,X,alpha,ac2dArray,sigma)
    end do 
    !$OMP END PARALLEL DO
  end if 

  if (calcGradXYZ) then 
    !$OMP PARALLEL DO PRIVATE(ii) SHARED(YgradXYZest) SCHEDULE(STATIC)
    do ii=1, Npredict 
      YgradXYZest(:,:,ii) = YgradXYZest_KRR(Natoms,Xsize,Ntrain,NtrVal,NtrGrXYZ,itrgrxyzDim,itrgrxyz,Xpredict(:,ii),XYZpredict(:,:,ii),XYZ,X,alpha,ac2dArray,sigma) 
    end do 
    !$OMP END PARALLEL DO
  end if 


  
end subroutine predict 

subroutine calcAlpha(NtrVal,NtrGrXYZ,Ksize,K,Ytrain,alpha,lambdav,lambdagradxyz)
  use mathUtils, only : solveSysLinEqs
  implicit none 
  ! Arguments
  integer, intent(in) :: NtrVal,NtrGrXYZ,Ksize
  real(kind=8), intent(in) :: K(1:Ksize,1:Ksize)
  real(kind=8), intent(in) :: Ytrain(1:Ksize)
  real(kind=8), intent(out) :: alpha(1:Ksize,1)
  real(kind=8), intent(in) :: lambdav, lambdagradxyz
  ! Local variables 
  integer                       :: jj, Error ! Loop index and error of (de)allocation
  real(kind=8)              :: lGrLoc, lGrXYZloc ! Local lambdas for gradients and XYZ gradients
  real(kind=8), allocatable :: InvMat(:,:)    ! Inversed matrix

  ! Allocate arrays 
  allocate(InvMat(1:Ksize,1:Ksize),stat=Error)
  if (Error/=0) call stopKREG('Unable to allocate inverse matrix')

  InvMat(1:Ksize,1:Ksize) = K(1:Ksize,1:Ksize)
  do jj=1, NtrVal 
    InvMat(jj,jj) = InvMat(jj,jj) + lambdav
  end do 
  do jj=1, NtrGrXYZ 
    InvMat(NtrVal+jj,NtrVal+jj) = InvMat(NtrVal+jj,NtrVal+jj) + lambdagradxyz
  end do 

  call solveSysLinEqs(Ksize,1,InvMat,Ytrain,alpha)


end subroutine calcAlpha

! subroutine cleanUp() 
!   implicit none 
!   useVal = .True. 
!   useGradXYZ = .False. 
!   calcVal = .True. 
!   calcGradXYZ = .False. 
!   sigma = 1.0 
!   lambdav = 0.0
!   lambdagradxyz = 0.0
!   Natoms = 0
!   Ntrain = 0
!   Npredict = 0 
!   Ksize = 0 
!   NtrVal = 0
!   NtrGrXYZ = 0
!   if (allocated(ac2dArray)) deallocate(ac2dArray)
!   if (allocated(XYZ)) deallocate(XYZ)
!   if (allocated(XYZpredict)) deallocate(XYZpredict)
!   if (allocated(X)) deallocate(X)
!   if (allocated(Xpredict)) deallocate(Xpredict)
!   if (allocated(Xeq)) deallocate(Xeq)
!   if (allocated(Req)) deallocate(Req)
!   if (allocated(K)) deallocate(K)
!   if (allocated(alpha)) deallocate(alpha)
!   if (allocated(Yref)) deallocate(Yref)
!   if (allocated(YgradXYZref)) deallocate(YgradXYZref)
!   if (allocated(Ytrain)) deallocate(Ytrain)
!   if (allocated(Yest)) deallocate(Yest)
!   if (allocated(YgradXYZest)) deallocate(YgradXYZest)
!   if (allocated(Kprediction)) deallocate(Kprediction)
!   if (allocated(itrgrxyz)) deallocate(itrgrxyz)
!   Nprops = 1

! end subroutine


end module kreg