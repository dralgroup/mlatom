using Base.Threads
include("stopper.jl")
include("mathUtils.jl")

function get_ac2dArray(Natoms)
    ac2dArray = zeros(Int,(Natoms,Natoms))
    for ii=1:Natoms-1
        for jj=ii+1:Natoms
            dd = (2Natoms-ii)*(ii-1)/2+jj-ii
            ac2dArray[ii,jj] = dd
            ac2dArray[jj,ii] = dd
        end
    end 
    return ac2dArray
end 

function distance2(Xi,Xj)
    distance2 = 0.0
    for ii=1:size(Xi)[1]
        distance2 += (Xi[ii]-Xj[ii])^2
    end 
    
    return distance2
    # sum((Xi-Xj).^2)
end 

function distance(Xi,Xj)
    sqrt(distance2(Xi,Xj))
end 

function Rij(xyzI,xyzJ)
    distance(xyzI,xyzJ)
end 

function kernel(Xi,Xj,sigma)
    # map(exp,-distance2(Xi,Xj)/(2sigma^2))
    exp(-distance2(Xi,Xj)/(2sigma^2))
    # (-distance2(Xi,Xj)/(2sigma^2))
    # Xi[1]/(2sigma^2)
end 

function dKdMiat(Natoms,a,tt,Xi,Xj,XYZi,ac2dArray,sigma)
    dKdMiat = 0.0
    for bb=1:Natoms 
        if a != bb
            dd = ac2dArray[a,bb]
            dKdMiat += (Xj[dd]-Xi[dd]) * Xi[dd] * (XYZi[tt,bb]-XYZi[tt,a]) / Rij(XYZi[:,a],XYZi[:,bb])^2
        end 
    end  
    return dKdMiat * kernel(Xi,Xj,sigma)/sigma^2
end 

function d2KdMiatdMjbu(Natoms,a,tt,bb,uu,Xi,Xj,XYZi,XYZj,ac2dArray,sigma)
    tempnum1 = 0.0
    tempnum2 = 0.0
    out_tempnum1 = 0.0 
    out_tempnum2 = 0.0
    for cc=1:Natoms 
        if a!=cc 
            dd = ac2dArray[a,cc]
            dXdMi = Xi[dd] * (XYZi[tt,cc]-XYZi[tt,a]) / Rij(XYZi[:,a],XYZi[:,cc])^2
            in_tempnum1 = 0.0
            in_tempnum2 = 0.0
            for gg=1:Natoms 
                if bb!=gg 
                    ee = ac2dArray[bb,gg]
                    dXdMj = Xj[ee] * (XYZj[uu,gg]-XYZj[uu,bb]) / Rij(XYZj[:,bb],XYZj[:,gg])^2
                    in_tempnum1 += dXdMj * (Xj[dd]-Xi[dd])*(Xi[ee]-Xj[ee])
                    if dd==ee 
                        in_tempnum2 += dXdMj 
                    end 
                end 
            end 
            out_tempnum1 += in_tempnum1*dXdMi 
            out_tempnum2 += in_tempnum2*dXdMi 
        end 
    end 
    out_tempnum1 /= (sigma^2)
    return (out_tempnum1+out_tempnum2)*kernel(Xi,Xj,sigma)/(sigma^2)
end 

function get_Xeq(Natoms,ac2dArray,Req)
    Xsize = Int64(Natoms*(Natoms-1)/2)
    Xeq = zeros(Xsize)
    for ii=1:Natoms,jj=(ii+1):Natoms 
        Xeq[ac2dArray[ii,jj]] = Rij(Req[:,ii],Req[:,jj])
    end 
    return Xeq 
end

function RE_descriptor(Natoms,ac2dArray,xyzI,Xeq)
    Xsize = Int64(Natoms*(Natoms-1)/2)
    RE = zeros(Xsize)
    for ii=1:Natoms,jj=(ii+1):Natoms 
        RE[view(ac2dArray,ii,jj)] = Xeq[view(ac2dArray,ii,jj)] / Rij(view(xyzI,:,ii),view(xyzI,:,jj))
    end 
    return RE
end 

function calc_RE_descriptors_wrap(Natoms,ac2dArray,XYZ,Xeq)
    return @time calc_RE_descriptors(Natoms,ac2dArray,XYZ,Xeq)
end

function calc_RE_descriptors(Natoms,ac2dArray,XYZ,Xeq)
    Xsize = Int64(Natoms*(Natoms-1)/2)
    Npoints = size(XYZ)[3]
    X = zeros((Xsize,Npoints))
    for ipoint=1:Npoints
        X[:,ipoint] = RE_descriptor(Natoms,ac2dArray,view(XYZ,:,:,ipoint),Xeq)
    end 
    return X 
end

function get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms)
    itrgrxyz  = zeros(Int64,(NtrGrXYZ,3))
    icount = 0 
    for ip=1:Ntrain,ii=1:Natoms,jj=1:3 
        icount += 1
        itrgrxyz[icount,1] = ip 
        itrgrxyz[icount,2] = ii 
        itrgrxyz[icount,3] = jj 
    end 
    return itrgrxyz
end 

function calc_kernel_matrix(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,ac2dArray,XYZ,X,sigma)
    Ksize = NtrVal+NtrGrXYZ
    K = zeros(Float64,(Ksize,Ksize))
    # function distance2(Xi,Xj)
    #     sum((Xi-Xj).^2)
    # end 
    # Covariance between values
    # for ii=1:NtrVal 
    #     K[ii,ii] = map(exp,-sum((view(X,:,ii)-view(X,:,ii)).^2)  /(2sigma^2))#kernel(view(X,:,ii),view(X,:,ii),sigma)
    #     for jj=(ii+1):NtrVal 
    #         value = map(exp,-sum((view(X,:,ii)-view(X,:,jj)).^2) /(2sigma^2))#kernel(view(X,:,ii),view(X,:,jj),sigma)
    #         K[ii,jj] =  value
    #         K[jj,ii] = value
    #     end 
    # end 
    for ii=1:NtrVal 
        K[ii,ii] = kernel(view(X,:,ii),view(X,:,ii),sigma)
        for jj=(ii+1):NtrVal 
            value = kernel(view(X,:,ii),view(X,:,jj),sigma)
            K[ii,jj] = value
            K[jj,ii] = value
        end 
    end 

    # # Covariance between values and gradients 
    # for ii=1:NtrVal, jj=1:NtrGrXYZ 
    #     value = dKdMiat(Natoms,itrgrxyz[jj,2],itrgrxyz[jj,3],X[:,itrgrxyz[jj,1]],X[:,ii],XYZ[:,:,itrgrxyz[jj,1]],ac2dArray,sigma)
    #     K[ii,NtrVal+jj] = value 
    #     K[NtrVal+jj,ii] = value 
    # end 

    # # Covariance between gradients 
    # for ii=1:NtrGrXYZ 
    #     K[NtrVal+ii,NtrVal+ii] = d2KdMiatdMjbu(Natoms,itrgrxyz[ii,2],itrgrxyz[ii,3],itrgrxyz[ii,2],itrgrxyz[ii,3],X[:,itrgrxyz[ii,1]],X[:,itrgrxyz[ii,1]],XYZ[:,:,itrgrxyz[ii,1]],XYZ[:,:,itrgrxyz[ii,1]],ac2dArray,sigma)
    #     for jj=1:NtrGrXYZ 
    #         value = d2KdMiatdMjbu(Natoms,itrgrxyz[ii,2],itrgrxyz[ii,3],itrgrxyz[jj,2],itrgrxyz[jj,3],X[:,itrgrxyz[ii,1]],X[:,itrgrxyz[jj,1]],XYZ[:,:,itrgrxyz[ii,1]],XYZ[:,:,itrgrxyz[jj,1]],ac2dArray,sigma)
    #         K[NtrVal+ii,NtrVal+jj] = value 
    #         K[NtrVal+jj,NtrVal+ii] = value 
    #     end 
    # end 
    return K
end 

function Yest_KRR(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,Xipredict,XYZ,X,alpha,ac2dArray,sigma)
    Yest_KRR = 0.0 
    for jj=1:NtrVal 
        Yest_KRR += alpha[jj]*kernel(Xipredict,view(X,:,jj),sigma)
    end 
    for jj=1:NtrGrXYZ
        Yest_KRR += alpha[NtrVal+jj]*dKdMiat(Natoms,itrgrxyz[jj,2],itrgrxyz[jj,3],X[:,itrgrxyz[jj,1]],Xipredict,XYZ[:,:,itrgrxyz[jj,1]],ac2dArray,sigma)
    end 
    return Yest_KRR
end

function YgradXYZest_KRR(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,Xipredict,XYZipredict,XYZ,X,alpha,ac2dArray,sigma)
    YgradXYZest_KRR = zeros((3,Natoms))
    for a=1:Natoms, tt=1:3
        for jj=1:NtrVal 
            YgradXYZest_KRR[tt,a] += alpha[jj]*dKdMiat[Natoms,a,tt,Xipredict,X[:,jj],XYZipredict,ac2dArray,sigma]
        end 
        for jj=1:NtrGrXYZ 
            YgradXYZest_KRR[tt,a] += alpha[NtrVal+jj]*d2KdMiatdMjbu(Natoms,a,tt,itrgrxyz[jj,2],itrgrxyz[jj,3],Xipredict,X[:itrgrxyz[jj,1]],XYZipredict,XYZ[:,:,itrgrxyz[jj,1]],ac2dArray,sigma)
        end 
    end 
    return YgradXYZest_KRR
end 

function train(Natoms,Ntrain,NtrVal,NtrGrXYZ,ac2dArray,XYZ,X,Ytrain,sigma,lambdav,lambdagradxyz,calcKernel)
    if NtrGrXYZ != 0
        itrgrxyz = get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms)
    else 
        itrgrxyz = zeros(Int64,(1,3))
    end 

    if calcKernel 
        @time K = calc_kernel_matrix(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,ac2dArray,XYZ,X,sigma)
    end 
    alpha = calcAlpha(NtrVal,NtrGrXYZ,K,Ytrain,lambdav,lambdagradxyz)
    return K,alpha
end 

function predict_wrap(Natoms,Ntrain,Npredict,NtrVal,NtrGrXYZ,ac2dArray,X,XYZ,Xpredict,XYZpredict,alpha,calcVal,calcGradXYZ,sigma)
    return @time predict(Natoms,Ntrain,Npredict,NtrVal,NtrGrXYZ,ac2dArray,X,XYZ,Xpredict,XYZpredict,alpha,calcVal,calcGradXYZ,sigma)
end 

function predict(Natoms,Ntrain,Npredict,NtrVal,NtrGrXYZ,ac2dArray,X,XYZ,Xpredict,XYZpredict,alpha,calcVal,calcGradXYZ,sigma)
    if NtrGrXYZ != 0 
        itrgrxyz = get_itrgrxyz(NtrGrXYZ,Ntrain,Natoms)
    else
        itrgrxyz = zeros(Int64,(1,3))
    end 

    Yest = zeros(Npredict)
    YgradXYZest = zeros((3,Natoms,Npredict))

    if calcVal 
        Threads.@threads for ii=1:Npredict 
            Yest[ii] = Yest_KRR(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,view(Xpredict,:,ii),XYZ,X,alpha,ac2dArray,sigma)
        end 
    end 

    if calcGradXYZ 
        for ii=1:Npredict 
            YgradXYZest[:,:,ii] = YgradXYZest_KRR(Natoms,Ntrain,NtrVal,NtrGrXYZ,itrgrxyz,Xpredict[:,ii],XYZpredict[:,:,ii],XYZ,X,alpha,ac2dArray,sigma)
        end 
    end 
    return Yest,YgradXYZest
end 

function calcAlpha(NtrVal,NtrGrXYZ,K,Ytrain,lambdav,lambdagradxyz)
    InvMat = copy(K)
    for jj=1:NtrVal
        InvMat[jj,jj] += lambdav 
    end 
    for jj=1:NtrGrXYZ 
        InvMat[NtrVal+jj,NtrVal+jj] += lambdagradxyz 
    end 
    @time alpha = solveSysLinEqs(InvMat,Ytrain)
    return alpha
end 

