# Calculate the squared Euclidean distance between two vectors
function distance2(Xi,Xj)
    distance2 = 0.0
    for ii=1:size(Xi)[1]
        distance2 += (Xi[ii]-Xj[ii])^2
    end 
    return distance2
end 

# Calculate the Euclidean distance between two vectors
function distance(Xi,Xj)
    sqrt(distance2(Xi,Xj))
end 

# Calcualte interatomic distance
function Rij(xyzI,xyzJ)
    distance(xyzI,xyzJ)
end 

# Gaussian kernel function
function Gaussian_kernel(Xi,Xj,sigma)
    exp(-distance2(Xi,Xj)/(2sigma^2))
end 

# Periodic Gaussian kernel function 
function periodic_Gaussian_kernel(Xi,Xj,sigma,period)
    exp(-2.0*(sin(pi*sqrt(distance2(Xi,Xj)) / period))^2 / sigma^2)
end 

# Decaying periodic Gaussian kernel function
function decaying_periodic_Gaussian_kernel(Xi,Xj,sigma,sigmap,period)
    exp(-distance2(Xi,Xj)/(2sigma^2)-2.0*(sin(pi*sqrt(distance2(Xi,Xj))/period))^2/(sigmap^2))
end 

# Matern kernel function 
function Matern_kernel(Xi,Xj,sigma,nn)
    value = 0.0
    dist = distance(Xi,Xj)
    for kk=0:nn 
        value = value + factorial(nn+kk) / factorial(kk) / factorial(nn-kk) * ((2*dist/sigma)^(nn-kk))
    end 
    value = exp(-1.0*dist/sigma) * factorial(nn) / factorial(2*nn) * value
end 

function Gaussian_dKdXid(dd,Xi,Xj,sigma)
    Gaussian_kernel(Xi,Xj,sigma) * (Xj[dd]-Si[dd]) / sigma^2
end 

function Gaussian_d2KdXiddXje(dd,ee,Xi,Xj,sigma)
    if dd==ee 
        Kdelta = 1.0
    else 
        Kdelta = 0.0 
    end
    Gaussian_kernel(Xi,Xj,sigma) * (Kdelta + (Xj[dd]-Xi[dd])*(Xi[ee]-Xj[ee])/sigma^2)/sigma^2
end 

function Gaussian_dKdMiat(a,tt,Xi,Xj,XYZi,sigma,ac2dArray)
    Natoms = size(XYZi)[2]
    dKdMiat = 0.0
    for bb=1:Natoms 
        if a != bb
            dd = ac2dArray[a,bb]
            dKdMiat += (Xj[dd]-Xi[dd]) * Xi[dd] * (XYZi[tt,bb]-XYZi[tt,a]) / Rij(view(XYZi,:,a),view(XYZi,:,bb))^2
        end 
    end  
    return dKdMiat * Gaussian_kernel(Xi,Xj,sigma)/sigma^2
end 

function Gaussian_d2KdMiatdMjbu(a,tt,bb,uu,Xi,Xj,XYZi,XYZj,sigma,ac2dArray)
    Natoms = size(XYZi)[2]
    tempnum1 = 0.0
    tempnum2 = 0.0
    out_tempnum1 = 0.0 
    out_tempnum2 = 0.0
    for cc=1:Natoms 
        if a!=cc 
            dd = ac2dArray[a,cc]
            dXdMi = Xi[dd] * (XYZi[tt,cc]-XYZi[tt,a]) / Rij(view(XYZi,:,a),view(XYZi,:,cc))^2
            in_tempnum1 = 0.0
            in_tempnum2 = 0.0
            for gg=1:Natoms 
                if bb!=gg 
                    ee = ac2dArray[bb,gg]
                    dXdMj = Xj[ee] * (XYZj[uu,gg]-XYZj[uu,bb]) / Rij(view(XYZj,:,bb),view(XYZj,:,gg))^2
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
    return (out_tempnum1+out_tempnum2)*Gaussian_kernel(Xi,Xj,sigma)/(sigma^2)
end 

