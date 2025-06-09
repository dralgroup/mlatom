using Base.Threads 
include("stopper.jl")
include("mathUtils.jl")
include("kernel_function.jl")

mutable struct krr

    # Options
    kernel::String 
    descriptor::String 

    # Hyperparameters
    lambdav::Number 
    lambdagrad::Number
    lambdagradxyz::Number
    sigma::Number
    sigmap::Number 
    period::Number
    nn::Number

    # Numbers
    Natoms::Number
    Xsize::Number
    NtrVal::Number 
    NtrGr::Number 
    NtrGrXYZ::Number
    Ksize::Number
    prior::Number

    # Arrays
    XYZ 
    X 
    Y 
    Ygrad 
    YgradXYZ 
    K 
    alpha
    ac2dArray
    Ytrain

    # Booleans
    learnX::Bool 
    learnXYZ::Bool 
    learnY::Bool 
    learnYgrad::Bool 
    learnYgradXYZ::Bool
    initialized::Bool

    # Functions
    train::Function 
    predict::Function
    initialize::Function
    get_ac2dArray::Function
    calc_kernel_matrix_x::Function 
    calc_kernel_matrix_xyz::Function 
    calc_alpha::Function
    kernel_function::Function
    dKdXid::Function 
    dKdMiat::Function 
    d2KdXiddXje::Function 
    d2KdMiatdMjbu::Function
    Yest_KRR::Function
    YgradXYZest_KRR::Function
    setter::Function
    getter::Function


    # Initialize and defined functions
    function krr()
        this = new() 

        # Hyperparameters
        this.lambdav = 0.0
        this.lambdagrad = 0.0
        this.lambdagradxyz = 0.0
        this.sigma = 0.0
        this.sigmap = 0.0
        this.period = 0.0
        this.nn = 0

        # Numbers
        this.Natoms = 0
        this.Xsize = 0
        this.NtrVal = 0 
        this.NtrGr = 0 
        this.NtrGrXYZ = 0
        this.Ksize = 0
        this.prior = 0.0


        # Arrays
        this.XYZ = nothing 
        this.X = nothing
        this.Y = nothing 
        this.Ygrad = nothing 
        this.YgradXYZ = nothing 
        this.K = nothing 
        this.alpha = nothing 
        this.ac2dArray = nothing
        this.Ytrain = nothing

        # Booleans
        this.learnX = false
        this.learnXYZ = false
        this.learnY = false 
        this.learnYgrad = false
        this.learnYgradXYZ = false
        this.initialized = false

        # Training
        this.train = function ()
            this.initialize()
            if this.learnX
                this.calc_kernel_matrix_x()
            end 
            if this.learnXYZ 
                stopMLatom("Learning XYZ is not implemented")
                this.calc_kernel_matrix_xyz()
            end 
            this.calc_alpha()
        end 

        # Predicting
        this.predict = function (Xpredict=nothing,XYZpredict=nothing;calcVal=false,calcGradXYZ=false)
            # if !isnothing(XYZpredict)
            Npredict = size(Xpredict)[2]
            # Single output
            if ndims(this.alpha) == 1
                if calcVal
                    Yest = zeros(Npredict)
                    Threads.@threads for ii=1:Npredict 
                        Yest[ii] = this.Yest_KRR(view(Xpredict,:,ii),this.alpha)
                    end
                end 
            # Multiple outputs
            elseif ndims(this.alpha) == 2
                if calcVal 
                    nlearn = size(this.alpha)[2]
                    Yest = zeros((Npredict,nlearn))
                    for jj=1:nlearn 
                        Threads.@threads for ii=1:Npredict
                            Yest[ii,jj] = this.Yest_KRR(view(Xpredict,:,ii),view(this.alpha,:,jj))
                        end 
                    end
                end 
            else 
                stopMLatom("The dimension of alpha is larger than 2")
            end 
            return Yest .+ this.prior
        end

        # Initialize and check inputs
        this.initialize = function()
            if !this.initialized
                if isnothing(this.X) && isnothing(this.XYZ)
                    stopMLatom("X or XYZ not found")
                elseif !isnothing(this.XYZ)
                    # print(size(this.XYZ))
                    this.learnXYZ = true
                    this.Natoms = size(this.XYZ)[2]
                    this.get_ac2dArray(this.Natoms)
                elseif !isnothing(this.X)
                    this.learnX = true
                end 
                if !isnothing(this.Y)
                    this.learnY = true 
                end 
                if !isnothing(this.Ygrad)
                    this.learnYgrad = true 
                end 
                if !isnothing(this.YgradXYZ)
                    this.learnYgradXYZ = true 
                end 
            end 
            # Get Ytrain
            this.Ytrain = zeros(0)
            if this.learnY
                this.NtrVal = size(this.Y)[1]
                this.Ksize += this.NtrVal
                # if ndims(this.prior) == 0
                #     # println(typeof(this.Y))
                #     # println(size(this.Y))
                #     this.Ytrain = cat(this.Ytrain,this.Y.-this.prior,dims=1)
                # else
                #     Ytemp = copy(this.Y)
                #     for ii=1:this.NtrVal
                #         # println(Ytemp[ii])
                #         Ytemp[ii] .-= this.prior
                #     end 
                #     this.Ytrain = vcat(this.Ytrain,Ytemp)
                # end 
                this.Ytrain = cat(this.Ytrain,this.Y.-this.prior,dims=1)
                
            end 
            if this.learnYgrad 
                this.NtrGr = this.Xsize * size(this.Ygrad)[2]
                this.Ksize += this.NtrGr
                stopMLatom("Learning gradients is not implemented")
            end 
            if this.learnYgradXYZ
                this.NtrGrXYZ = 3*this.Natoms*size(this.YgradXYZ)[3]
                this.Ksize += this.NtrGrXYZ
                stopMLatom("Learning XYZ derivatives is not implemented")
            end 

        end 

        # Get ac2dArray
        this.get_ac2dArray = function(Natoms)
            this.ac2dArray = zeros(Int,(Natoms,Natoms))
            for ii=1:Natoms-1,jj=ii+1:Natoms 
                dd = (2Natoms-ii)*(ii-1)/2+jj-ii
                this.ac2dArray[ii,jj] = dd
                this.ac2dArray[jj,ii] = dd
            end 
        end 

        # Calculate kernel matrix when gradients are calculated with respect to x
        this.calc_kernel_matrix_x = function ()
            this.K = zeros((this.Ksize,this.Ksize))
            # .Covariance between values
            Threads.@threads for ii=1:this.NtrVal 
                this.K[ii,ii] = this.kernel_function(view(this.X,:,ii),view(this.X,:,ii))
                for jj=ii+1:this.NtrVal
                    this.K[ii,jj] = this.kernel_function(view(this.X,:,ii),view(this.X,:,jj))
                    this.K[jj,ii] = this.K[ii,jj]
                end 
            end 

        end 

        # Calculate kernel matrix when gradients are calculated with respect to xyz
        this.calc_kernel_matrix_xyz = function () 

        end

        # Calculate regression coefficients
        this.calc_alpha = function ()
            InvMat = copy(this.K)
            for jj=1:this.NtrVal 
                InvMat[jj,jj] += this.lambdav
            end 
            for jj=1:this.NtrGr 
                InvMat[this.NtrVal+jj,this.NtrVal+jj] += this.lambdagrad 
            end 
            for jj=1:this.NtrGrXYZ 
                InvMat[this.NtrVal+jj,this.NtrVal+jj] += this.lambdagradxyz 
            end  
            this.alpha = solveSysLinEqs(InvMat,this.Ytrain)
        end 

        # General kernel function
        this.kernel_function = function (Xi,Xj)
            if this.kernel == "Gaussian"
                return Gaussian_kernel(Xi,Xj,this.sigma)
            elseif this.kernel == "periodic_Gaussian"
                return periodic_Gaussian_kernel(Xi,Xj,this.sigma,this.period)
            elseif this.kernel == "decaying_periodic_Gaussian"
                return decaying_periodic_Gaussian_kernel(Xi,Xj,this.sigma,this.sigmap,this.period)
            elseif this.kernel == "Matern"
                # println(this.nn)
                # exit()
                return Matern_kernel(Xi,Xj,this.sigma,this.nn)
            else
                stopMLatom("Unsupported kernel function: "*this.kernel)
            end 
        end

        # General dKdXid function
        this.dKdXid = function (dd,Xi,Xj)
            if this.kernel == "Gaussian"
                return Gaussian_dKdXid(dd,Xi,Xj,this.sigma)
            else
                stopMLatom("Unsupported kernel function for first derivatives: "*this.kernel)
            end 
        end 

        # General d2KdXiddXje function
        this.d2KdXiddXje = function (dd,ee,Xi,Xj) 
            if this.kernel == "Gaussian"
                return Gaussian_d2KdXiddXje(dd,ee,Xi,Xj,this.sigma)
            else
                stopMLatom("Unsupported kernel function for second derivatives: "*this.kernel)
            end 
        end 

        # General dKdMiat function 
        this.dKdMiat = function (a,tt,Xi,Xj,XYZi) 
            if this.kernel == "Gaussian"
                return Gaussian_dKdMiat(a,tt,Xi,Xj,XYZi,this.sigma,this.ac2dArray)
            else
                stopMLatom("Unsupported kernel function for first derivatives: "*this.kernel)
            end 
        end 

        # General d2KdMiatdMjbu function
        this.d2KdMiatdMjbu = function (a,tt,bb,uu,Xi,Xj,XYZi,XYZj) 
            if this.kernel == "Gaussian"
                return Gaussian_d2KdMiatdMjbu(a,tt,bb,uu,Xi,Xj,XYZi,XYZj,this.sigma,this.ac2dArray)
            else
                stopMLatom("Unsupported kernel function for second derivatives: "*this.kernel)
            end 
        end 

        #
        this.Yest_KRR = function (Xipredict,alpha)
            Yest_KRR = 0.0
            for jj=1:this.NtrVal 
                Yest_KRR += alpha[jj]*this.kernel_function(Xipredict,view(this.X,:,jj)) 
            end 
            # for jj=1:NtrGrXYZ
            #     Yest_KRR += alpha[NtrVal+jj]*dKdMiat(Natoms,itrgrxyz[jj,2],itrgrxyz[jj,3],X[:,itrgrxyz[jj,1]],Xipredict,XYZ[:,:,itrgrxyz[jj,1]],ac2dArray,sigma)
            # end 
            return Yest_KRR 
        end 

        # 
        this.YgradXYZest_KRR = function() 

        end 

        # Set field value in this object
        # Pyjulia cannot directly assign value to a field
        # Probably this is not the best way to do it...
        this.setter = function (arg_name::String,value)
            setfield!(this,eval(Meta.parse(":"*arg_name)),value)
            return 
        end 

        # Get field value in this object
        # This function is just for convenience
        this.getter = function (arg_name::String)
            return getfield(this,eval(Meta.parse(":"*arg_name)))
        end



        return this 
    end 
end 

