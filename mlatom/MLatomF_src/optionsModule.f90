
  !---------------------------------------------------------------------------! 
  ! optionsModule: reads and stores default and user-defined options          ! 
  ! Implementations by: Pavlo O. Dral                                         !
  !                     Yi-Fan Hou (changed default values for                !
  !                                 hyperparameter optimization)              ! 
  !---------------------------------------------------------------------------! 

!==============================================================================
module optionsModule
!==============================================================================
! optionsModule reads and stores default and user-defined options
!==============================================================================
  use precision, only : rprec
  implicit none
  
  ! Available options and their defaults
  type :: optionsType
    logical              :: help = .false.                            ! Suppress help
    ! Input related options
    character(len=256)   :: Nuse          = 'ALL'                     ! Number of entries in dataset to be used
    character(len=256)   :: XYZfile       = ''                        ! Name of file with xyz coordinates
    logical              :: readX         = .false.                   ! Read X values from file
    character(len=256)   :: XfileIn       = ''                        ! Name of file with X values
    character(len=256)   :: Yfile         = ''                        ! Name of file with reference values
    character(len=256)   :: YgradFile     = ''                        ! Name of file with reference gradients   
    character(len=256)   :: YgradXYZfile  = ''                        ! Name of file with reference gradients in Cartesian coordinates
    character(len=256)   :: eqXYZfileIn   = ''                        ! Name of file with equilibrium geometry
    character(len=256)   :: MLmodelFileIn = ''                        ! Name of unformatted file with ML model
    character(len=256)   :: iTrainIn      = ''                        ! Name of file with indices of training points
    character(len=256)   :: iTestIn       = ''                        ! Name of file with indices of test points
    character(len=256)   :: iSubtrainIn   = ''                        ! Name of file with indices of subtraining points
    character(len=256)   :: iValidateIn   = ''                        ! Name of file with indices of validation points
    character(len=256)   :: iCVtestPrefIn = ''                        ! Prefix of file with indices of cross-validation test splits
    character(len=256)   :: iCVoptPrefIn  = ''                        ! Prefix of file with indices of cross-validation splits for hyperparameter optimization
    character(len=256)   :: iCVoptPrefGradIn = ''
    character(len=256)   :: iTrainGradXYZIn    = ''                   ! Name of file with indices of XYZ gradients of training points
    character(len=256)   :: iSubtrainGradXYZIn = ''                   ! Name of file with indices of XYZ gradients of subtraining points
    character(len=256)   :: iValidateGradXYZIn = ''                   ! Name of file with indices of XYZ gradients of validation points

    logical              :: readModPermInd= .false.
    logical              :: usePermInd    = .false.
    character(len=256)   :: permIndIn     = ''
    character(len=256)   :: permlen       = ''

    ! Data set operations
    logical              :: XYZ2X         = .false.                   ! Convert XYZ to X
    logical              :: analyze       = .false.                   ! Analyze data sets
    logical              :: sample        = .false.                   ! Sample
    character(len=256)   :: sampling      = 'random'                  ! Sampling

    ! Task to perform
    logical              :: useMLmodel    = .false.                   ! Use available ML model
      
    logical              :: createMLmodel = .false.                     ! Create ML model
      ! Training ML
      character(len=256)   :: Ntrain        = '0'                       ! Number of entries in dataset to be used for training
      character(len=256)   :: NtrainGradXYZ = '0'                       ! Number of XYZ gradients in dataset to be used for training 

    logical              :: estAccMLmodel = .false.                    ! Estimate accuracy (performance) of ML model
    ! Test ML
      character(len=256)   :: Ntest        = '0'                        ! Number of entries in dataset to be used for testing    
    
    ! Estimate performance of ML model
    logical              :: calcMAE       = .true.                    ! Calculate mean absolute error
    logical              :: calcMSE       = .true.                    ! Calculate mean signed   error
    logical              :: calcRMSE      = .true.                    ! Calculate root-mean-square error
    logical              :: calcMEAN      = .true.                    ! Calculate mean values
    logical              :: calcLPOutlier = .true.                    ! Calculate largest positive outlier
    logical              :: calcLNOutlier = .true.                    ! Calculate largest negative outlier
    logical              :: calcCorrCoeff = .true.                    ! Calculate Pearson correlation coefficient
    logical              :: calcLinReg    = .true.                    ! Calculate linear regression

    ! Cross-validation
    ! For model testing
    logical              :: CVtest        = .false.                   ! Cross-validate ML model to test its accuracy
      integer              :: NcvTestFolds  = 5                         ! Number of folds
    logical              :: LOOtest       = .false.                   ! Leave-one-out cross-validation for testing
    logical              :: CVopt         = .false.                   ! Use cross-validation to select hyperparameters
      integer              :: NcvOptFolds   = 5                         ! Number of folds
    logical              :: LOOopt        = .false.                   ! Leave-one-out cross-validation for hyperparameter optimization

    ! ML algorithm and its variations
    character(len=256)   :: prior         = '0'                       ! Prior
    character(len=3)     :: algorithm     = 'KRR'
      character(len=256) :: KRRtask       = 'learnVal'                ! Task of KRR
      logical            :: learnVal      = .false.
      logical            :: learnGradXYZ  = .false.
      character(len=256) :: lambda        = '0'                       ! Regularization parameter
      character(len=256) :: lambdaGradXYZ = 'lambdaVal'               ! Regularization parameter for XYZ gradients
      character(len=256) :: NlgLambda     = '6'                       ! Number of points on a logarithmic grid for search optimal lambda
      character(len=256) :: lgLambdaL     = '-35'                     ! The lowest  value of log2(lambda) to try
      character(len=256) :: lgLambdaH     = '-6'                      ! The highest value of log2(lambda) to try
      character(len=12)  :: kernel        = 'Gaussian'
        ! Suboptions for Gaussian, Laplacian, exponential, and Matern kernels
        character(len=256) :: sigma       = '100'                       ! Length-scale
        character(len=256) :: NlgSigma     = '6'                       ! Number of points on a logarithmic grid for search optimal sigma
        character(len=256) :: lgSigmaL     = '2'                        ! The lowest  value of log2(sigma) to try
        character(len=256) :: lgSigmaH     = '9'                        ! The highest value of log2(sigma) to try
        logical            :: periodKernel = .false.                    ! Calculations with periodic kernel
        logical            :: decayKernel  = .false.                    ! Calculations with periodic kernel
        character(len=256) :: sigmap       = '100'                      ! Length-scale for the periodic part
        character(len=256) :: period       = '1.0'                      ! period
        ! ... and special suboptions for Matern kernel
        character(len=256) :: nn           = '2'                        ! n in the Matern kernel (nu = n + 1/2)
        ! Suboptions for polynomial kernel (k(x,x') = (x*x'+c)^d)
        character(len=256) :: c            = '100'                      ! Addend
        character(len=256) :: NlgC         = '11'                       ! Number of points on a logarithmic grid for search optimal c
        character(len=256) :: lgCl         = '2'                        ! The lowest  value of log2(c) to try
        character(len=256) :: lgCh         = '9'                        ! The highest value of log2(c) to try
        character(len=256) :: d            = '2'                        ! Exponent
        character(len=256) :: NlgD         = '11'                       ! Number of points on a logarithmic grid for search optimal d
        character(len=256) :: lgDl         = '2'                        ! The lowest  value of log2(d) to try
        character(len=256) :: lgDh         = '9'                        ! The highest value of log2(d) to try
        
      logical            :: hyperOpt      = .false.                   ! Optimize hyperparameters?
      character(len=256)   :: minimizeError = 'RMSE'                    ! What error to minimize during hyperparameters optimization?
      character(len=256) :: lgOptDepth    = '3'                       ! Depth of logarithmic optimization
      character(len=256) :: Nsubtrain     = '0'                       ! Number of entries in dataset to be used for validation
      character(len=256) :: Nvalidate     = '0'                       ! Number of entries in dataset to be used for validation
      character(len=256) :: NsubtrainGradXYZ = '0'                    ! Number of XYZ gradients in dataset to be used for subtraining
      character(len=256) :: NvalidateGradXYZ = '0'                    ! Number of XYZ gradients in dataset to be used for validation

    ! Options related to atomistic simulations
    ! Permutational symmetry information
    logical              :: permInvKernel = .false.                   ! Calculations with permutationally invariant kernel
    character(len=256)   :: Nperm         = ''                        ! Number of permutations
    character(len=256)   :: permInvGroups = ''                        ! Permutationally invariant groups
    character(len=256)   :: permInvNuclei = ''                        ! Permutationally invariant nuclei within the above groups
    logical              :: selectPerm    = .false. 
    ! Molecular descriptor
    character(len=2)     :: molDescriptor = 'RE'                      ! Vector {Req/R}, where R is internuclear distance
    character(len=8)     :: molDescrType  = 'unsorted'                ! Type of molecular descriptor
    logical              :: descriptorDist= .false.

    ! Output options
    logical              :: writeX        = .false.                   ! Write file with X values
    character(len=256)   :: XfileOut      = ''                        ! Name of file with X values
    logical              :: writeXYZsorted= .false.                   ! Write file with sorted XYZ coordinates
    character(len=256)   :: XYZsortedFileOut= ''                      ! Name of file with sorted XYZ coordinates
    character(len=256)   :: YestFile      = ''                        ! Name of file with values estimated by ML model
    character(len=256)   :: YgradEstFile  = ''                        ! Name of file with estimated gradients
    character(len=256)   :: YgradXYZestFile  = ''                        ! Name of file with estimated gradients
    character(len=256)   :: MLmodelFileOut= ''                        ! Name of unformatted file with ML model
    character(len=256)   :: iTrainOut     = ''                        ! Name of file with indices of training points
    character(len=256)   :: iTestOut      = ''                        ! Name of file with indices of test points
    character(len=256)   :: iSubtrainOut  = ''                        ! Name of file with indices of subtraining points
    character(len=256)   :: iValidateOut  = ''                        ! Name of file with indices of validation points
    character(len=256)   :: iCVtestPrefOut= ''                        ! Prefix of file with indices of cross-validation test splits
    character(len=256)   :: iCVoptPrefOut = ''                        ! Prefix of file with indices of cross-validation splits for hyperparameter optimization
    logical              :: benchmark     = .false.
    logical              :: debug         = .false.

    ! Program run options
    character(len=256)   :: matDecomp     = 'Cholesky'                ! Matrix decomposition
    logical              :: invMatrix     = .false.                   ! Invert matrix
    logical              :: refine        = .false.                   ! Refine solution matrix
    logical              :: onTheFly      = .false.                   ! On-the-fly calculations of kernel matrix elements for validation
    logical              :: numDerivs     = .false.                   ! Request calculation of numerical derivatives even when analytical derivatives available
    
    ! Internal options
    logical              :: calcVal       = .false.                   ! Calculate values using ML model
    logical              :: calcGrad      = .false.                   ! Calculate derivatives of ML model
    logical              :: calcGradXYZ   = .false.                   ! Calculate gradients from ML model trained on them
    logical              :: writeEst      = .false.                   ! Save files with estimated values
    
  end type optionsType
  
  type(optionsType), save :: option
  ! Set flags to .true. if respective option was requested to be changed by user
  logical                 :: readXYZ      = .false.
  logical                 :: sigmaFlag    = .false.
  logical                 :: NlgSigmaFlag = .false.
  logical                 :: lgSigmaLflag = .false.
  logical                 :: lgSigmaHflag = .false.
  logical                 :: molDescTypeFlag = .false.
  logical                 :: NcvTestFoldsFlag = .false.
  logical                 :: NcvOptFoldsFlag = .false.
  logical                 :: learnValFlag = .false.

contains

subroutine writeOptions(unit)
  implicit none
  ! Argument
  integer, intent(in) :: unit
  ! Variable
  real(kind=rprec) :: tempDble

  write(unit,'(a)')             ' ___________________________________________________________'
  write(unit,'(a)')             ''
  write(unit,'(a)')             ' Calculations with the following options:'
  
  write(unit,'(a)')             ''
  write(unit,'(a)')             '    Input options'
                                        write(unit,'(a)') '      Nuse:                ' // trim(option%Nuse)
  if (trim(option%XfileIn) /= '')       write(unit,'(a)') '      XfileIn:             ' // trim(option%XfileIn)
  if (trim(option%XYZfile) /= '')       write(unit,'(a)') '      XYZfile:             ' // trim(option%XYZfile)  
  if (trim(option%Yfile) /= '')         write(unit,'(a)') '      Yfile:               ' // trim(option%Yfile)
  if (trim(option%YgradFile) /= '')     write(unit,'(a)') '      YgradFile:           ' // trim(option%YgradFile)
  if (trim(option%YgradXYZfile) /= '')  write(unit,'(a)') '      YgradXYZfile:        ' // trim(option%YgradXYZfile)
  if (trim(option%eqXYZfileIn) /='')    write(unit,'(a)') '      eqXYZfileIn          ' // trim(option%eqXYZfileIn)
  if (option%analyze) then
  if (trim(option%YestFile) /= '')      write(unit,'(a)') '      YestFile:            ' // trim(option%YestFile)
  if (trim(option%YgradEstFile) /= '')  write(unit,'(a)') '      YgradEstFile:        ' // trim(option%YgradEstFile)
  if(trim(option%YgradXYZestFile) /= '')write(unit,'(a)') '      YgradXYZestFile:     ' // trim(option%YgradXYZestFile)
  end if
  if (trim(option%MLmodelFileIn) /= '') write(unit,'(a)') '      MLmodelIn:           ' // trim(option%MLmodelFileIn)
  if (trim(option%iTrainIn) /= '')      write(unit,'(a)') '      iTrainIn:            ' // trim(option%iTrainIn)
  if (trim(option%iTestIn) /= '')       write(unit,'(a)') '      iTestIn:             ' // trim(option%iTestIn)
  if (trim(option%iSubtrainIn) /= '')   write(unit,'(a)') '      iSubtrainIn:         ' // trim(option%iSubtrainIn)
  if (trim(option%iValidateIn) /= '')   write(unit,'(a)') '      iValidateIn:         ' // trim(option%iValidateIn)
  if (trim(option%iCVtestPrefIn) /= '') write(unit,'(a)') '      iCVtestPrefIn:       ' // trim(option%iCVtestPrefIn)
  if (trim(option%iCVoptPrefIn) /= '')  write(unit,'(a)') '      iCVoptPrefIn:        ' // trim(option%iCVoptPrefIn)
    
  write(unit,'(a)')             ''
  write(unit,'(a)')             '    Output options'
  if (option%writeX)                    write(unit,'(a)') '      XfileOut:            ' // trim(option%XfileOut)
  if (option%writeXYZsorted)            write(unit,'(a)') '      XYZsortedFileOut:    ' // trim(option%XYZsortedFileOut)
  if (.not. option%analyze) then
  if (trim(option%YestFile) /= '')      write(unit,'(a)') '      YestFile:            ' // trim(option%YestFile)
  if (trim(option%YgradEstFile) /= '')  write(unit,'(a)') '      YgradEstFile:        ' // trim(option%YgradEstFile)
  if(trim(option%YgradXYZestFile) /= '')write(unit,'(a)') '      YgradXYZestFile:     ' // trim(option%YgradXYZestFile)
  end if
  if (trim(option%MLmodelFileOut) /= '')write(unit,'(a)') '      MLmodelOut:          ' // trim(option%MLmodelFileOut)
  if (trim(option%iTrainOut) /= '')     write(unit,'(a)') '      iTrainOut:           ' // trim(option%iTrainOut)
  if (trim(option%iTestOut) /= '')      write(unit,'(a)') '      iTestOut:            ' // trim(option%iTestOut)
  if (trim(option%iSubtrainOut) /= '')  write(unit,'(a)') '      iSubtrainOut:        ' // trim(option%iSubtrainOut)
  if (trim(option%iValidateOut) /= '')  write(unit,'(a)') '      iValidateOut:        ' // trim(option%iValidateOut)
  if (trim(option%iCVtestPrefOut) /= '')write(unit,'(a)') '      iCVtestPrefOut:      ' // trim(option%iCVtestPrefOut)
  if (trim(option%iCVoptPrefOut) /= '') write(unit,'(a)') '      iCVoptPrefOut:       ' // trim(option%iCVoptPrefOut)
  write(unit,'(a)')             ''
  
  if (option%sample .or. option%createMLmodel .or. option%estAccMLmodel) then
    write(unit,'(a)')             '    Sub-set sizes and splitting options'
    if     (option%LOOtest) then
      write(unit,'(a)')           '      Leave-one-out cross-validation for testing'
    elseif (option%CVtest) then
      write(unit,'(a,i0,a)')      '      ', option%NcvTestFolds, '-fold cross-validation for testing'
    end if
    if     (option%LOOopt) then
      write(unit,'(a)')           '      Leave-one-out cross-validation for hyperparameter optimization'
    elseif (option%hyperOpt .and. option%CVopt) then
      write(unit,'(a,i0,a)')      '      ', option%NcvOptFolds, &
                                  '-fold cross-validation for hyperparameter optimization'
    end if
    if (.not. option%CVtest) then
      if (option%createMLmodel .or. option%estAccMLmodel .or. trim(option%iTrainOut) /= '') then
        read(option%Ntrain,*) tempDble
        if (nint(abs(tempDble)) == 0) then
          if (option%estAccMLmodel) then
            write(unit,'(a)')       '      Training set:        80% of the entire dataset'
          else
            write(unit,'(a)')       '      Training set:        the entire dataset'
          end if
        elseif (tempDble < 1.0_rprec) then
          write(unit,'(a)')         '      Training set:        ' // trim(option%Ntrain) // ' fraction of the dataset'
        else
          write(unit,'(a)')         '      Training set:        ' // trim(option%Ntrain) // ' entries'
        end if
      end if
      if (option%estAccMLmodel .or. trim(option%iTestOut) /= '') then
        read(option%Ntest,*) tempDble
        if (nint(abs(tempDble)) == 0) then
          write(unit,'(a)')         '      Testing with:        all points of the entire dataset except for the training set points'
        elseif (tempDble < 1.0_rprec) then
          write(unit,'(a)')         '      Testing with:        ' // trim(option%Ntest) // ' fraction of the dataset'
        else
          write(unit,'(a)')         '      Testing with:        ' // trim(option%Ntest) // ' entries'
        end if
      end if
    end if
    if ((option%hyperOpt .and. .not. option%CVopt) .or. trim(option%iSubtrainOut) /= '') then
      read(option%Nsubtrain,*) tempDble
      if (nint(abs(tempDble)) == 0) then
        write(unit,'(a)')           '      Sub-training set:    80% of the training set'
      elseif (tempDble < 1.0_rprec) then
        write(unit,'(a)')           '      Sub-training set:    ' // trim(option%Nsubtrain) // ' fraction of the training set'
      else
        write(unit,'(a)')           '      Sub-training set:    ' // trim(option%Nsubtrain) // ' entries'
      end if
    end if
    if ((option%hyperOpt .and. .not. option%CVopt) .or. trim(option%iValidateOut) /= '') then
      read(option%Nvalidate,*) tempDble
      if (nint(abs(tempDble)) == 0) then
        write(unit,'(a)')       '      Validating with:     all points of the training set except for the sub-training set points'
      elseif (tempDble < 1.0_rprec) then
        write(unit,'(a)')       '      Validating with:     ' // trim(option%Nvalidate) // ' fraction of the training set'
      else
        write(unit,'(a)')       '      Validating with:     ' // trim(option%Nvalidate) // ' entries'
      end if
    end if
    write(unit,'(a)')             ''
  end if
    
  if (option%sample) then
    write(unit,'(a)')           '    Data set operations'
    write(unit,'(a)')           '      sample data points from a data set'
    write(unit,'(a)')           '        sampling:          ' // trim(option%sampling)
    write(unit,'(a)')           ''
    write(unit,'(a)')           ' ___________________________________________________________'
    write(unit,'(a)')           ''
    return
  elseif (option%XYZ2X) then
    write(unit,'(a)')           '    Data set operations'
    write(unit,'(a)')           '      convert XYZ coordinates into descriptor X'
    write(unit,'(a)')           ''
    write(unit,'(a)')           '    Molecular descriptor:  ' // trim(option%molDescrType) // ' ' // trim(option%molDescriptor)
    if (trim(option%permInvGroups) /= '') then
      write(unit,'(a)')           '    Permutationally invariant groups: ' // trim(option%permInvGroups)
    end if
    if (trim(option%permInvNuclei) /= '') then
      write(unit,'(a)')           '    Permutationally invariant nuclei: ' // trim(option%permInvNuclei)
    end if
    write(unit,'(a)')           ''
    write(unit,'(a)')           ' ___________________________________________________________'
    write(unit,'(a)')           ''
    return
  elseif (option%analyze) then
    write(unit,'(a)')           '    Data set operations'
    write(unit,'(a)')           '      analyze data sets'
    write(unit,'(a)')           ''
    write(unit,'(a)')           ' ___________________________________________________________'
    write(unit,'(a)')           ''
    return
  else
    write(unit,'(a)')           '    Data set operations'
    write(unit,'(a)')           '      sampling:            ' // trim(option%sampling)
  end if
  
  
  if (option%createMLmodel .or. option%estAccMLmodel .or. option%useMLmodel) then 
    write(unit,'(a)')           ''
    write(unit,'(a)')           '    ML task:'
  end if

  if     (option%useMLmodel) then
    write(unit,'(a)')           '                           use ML model'
  elseif (option%createMLmodel) then
    write(unit,'(a)')           '                           create ML model'
  elseif (option%estAccMLmodel) then
    write(unit,'(a)')           '                           estimate accuracy of ML model'
  end if
  
  ! Start of IF
  if (.not. option%useMLmodel) then

  if (option%createMLmodel .or. option%estAccMLmodel) then
    if (option%algorithm == 'KRR' .and. option%hyperOpt .and. .not. option%CVopt) then
      if (option%KRRtask     == 'learnVal') then
        write(unit,'(a)')       '    KRR task:              learn reference values'
      elseif (option%KRRtask == 'learnGradXYZ') then
        write(unit,'(a)')       '    KRR task:              learn only gradients'
      elseif (option%KRRtask == 'learnValGradXYZ') then
        write(unit,'(a)')       '    KRR task:              learn combined value and gradient information'
      end if
    end if
  end if
  write(unit,'(a)')             ''
  if (.not. option%MLmodelFileIn /= '') then
  if (option%createMLmodel .or. option%estAccMLmodel .and. option%algorithm == 'KRR') then
    write(unit,'(a)')           '    ML algorithm:          ' // trim(option%algorithm)
  end if
  if (option%hyperOpt) then
    write(unit,'(a)')           '      error to minimize:   ' // trim(option%minimizeError)
    write(unit,'(a)')           '      depth of log. opt.:  ' // trim(option%lgOptDepth)
  end if
  if (option%createMLmodel .or. option%estAccMLmodel .and. option%algorithm == 'KRR') then
    write(unit,'(a)')           '      lambda:              ' // trim(option%lambda)
    if (option%lambda == 'OPT') then
       write(unit,'(a)')        '      NlgLambda:           ' // trim(option%NlgLambda)
       write(unit,'(a)')        '      lgLambdaL:           ' // trim(option%lgLambdaL)
       write(unit,'(a)')        '      lgLambdaH:           ' // trim(option%lgLambdaH)
    end if
    if (option%lambdaGradXYZ /= 'lambdaVal') write(unit,'(a)')           '      lambdaGradXYZ:       ' // trim(option%lambdaGradXYZ)
    write(unit,'(a)')           '      Kernel:              ' // trim(option%kernel)
    if ((trim(option%kernel) == 'Gaussian')    .or. &
        (trim(option%kernel) == 'Laplacian')   .or. &
        (trim(option%kernel) == 'exponential') .or. &
        (trim(option%kernel) == 'Matern')) then
      write(unit,'(a)')         '        sigma:             ' // trim(option%sigma)
      if (option%sigma == 'OPT') then
        write(unit,'(a)')       '        NlgSigma:          ' // trim(option%NlgSigma)
        write(unit,'(a)')       '        lgSigmaL:          ' // trim(option%lgSigmaL)
        write(unit,'(a)')       '        lgSigmaH:          ' // trim(option%lgSigmaH)
      end if
      if ((trim(option%kernel) == 'Matern')) then
        write(unit,'(a)')       '        nn:                ' // trim(option%nn)
      end if
    end if
    if (trim(option%kernel) == 'polynomial') then
      write(unit,'(a)')         '        c:                 ' // trim(option%c)
      if (option%c == 'OPT') then
        write(unit,'(a)')       '        NlgC:              ' // trim(option%NlgC)
        write(unit,'(a)')       '        lgCl:              ' // trim(option%lgCl)
        write(unit,'(a)')       '        lgCh:              ' // trim(option%lgCh)
      end if
      write(unit,'(a)')         '        d:                 ' // trim(option%d)
!      if (option%d == 'OPT') then
!        write(unit,'(a)')       '        NlgD:              ' // trim(option%NlgD)
!        write(unit,'(a)')       '        lgDl:              ' // trim(option%lgDl)
!        write(unit,'(a)')       '        lgDh:              ' // trim(option%lgDh)
!      end if
    end if
    write(unit,'(a)')           '      prior:               ' // trim(option%prior)
    write(unit,'(a)')           '    Program run options:'
    if (option%createMLmodel .or. option%estAccMLmodel .and. option%algorithm == 'KRR') then
      write(unit,'(a)')           '      ' // trim(option%matDecomp) // ' decomposition'
      if (option%invMatrix) write(unit,'(a)') '      Invert matrix'
      if (option%refine)    write(unit,'(a)') '      Refine solution matrix'
    end if
    if (option%onTheFly) then
      write(unit,'(a)')           '      on-the-fly           on-the-fly calculation of kernel'
      write(unit,'(a)')           '                           matrix elements for validation'
    else
      write(unit,'(a)')           '                           calculate and store kernel matrix'
      write(unit,'(a)')           '                           elements for validation'
    end if
    if (option%numDerivs) then
      write(unit,'(a)')           '      numDerivs           numerical derivatives'
    end if
    write(unit,'(a)')             ''
  end  if
  
  ! Options related to atomistic simulations
  write(unit,'(a)')               ''
  if (option%readX) then
    write(unit,'(a)')             '    User-defined X-values'
  else
    write(unit,'(a)')             '    Molecular descriptor:  ' // trim(option%molDescrType) // ' ' // trim(option%molDescriptor)
  end if
  if (option%periodKernel) then
    write(unit,'(a)')             '    Calculations with periodic kernel'
    write(unit,'(a)')             '        period:               ' // trim(option%period)
  end if
  if (option%decayKernel) then
    write(unit,'(a)')             '    Calculations with decaying periodic kernel'
    write(unit,'(a)')             '        period:               ' // trim(option%period)
    write(unit,'(a)')             '        sigmap:               ' // trim(option%sigmap)
  end if
  if (option%permInvKernel) then
    write(unit,'(a)')           '    Calculations with permutationally invariant kernel'
  end if
  if (trim(option%Nperm) /= '') then
    write(unit,'(a)')           '    Number of permutations: ' // trim(option%Nperm)
  end if
  if (trim(option%permInvGroups) /= '') then
    write(unit,'(a)')           '    Permutationally invariant groups: ' // trim(option%permInvGroups)
  end if
  if (trim(option%permInvNuclei) /= '') then
    write(unit,'(a)')           '    Permutationally invariant nuclei: ' // trim(option%permInvNuclei)
  end if
  end if
  write(unit,'(a)')             ''
  
  ! Else of IF
  !else

  end if
  ! End of IF
  
  if (option%debug) then
    write(unit,'(a)')           '    Additional output for debugging'
    write(unit,'(a)')           ''
  end if
  if (option%benchmark) then
    write(unit,'(a)')           '    Additional output for benchmarking'
    write(unit,'(a)')           ''
  end if
  write(unit,'(a)')             ' ___________________________________________________________'
  write(unit,'(a)')             ''

end subroutine writeOptions

subroutine help(unit)
  implicit none
  integer, intent(in) :: unit

  write(unit,'(a)') ""
  write(unit,'(a)') " Available options:"
  write(unit,'(a)') ""
  write(unit,'(a)') "    help, -h,"
  write(unit,'(a)') "   -help, --help          print this info and exit"
  write(unit,'(a)') ""
  write(unit,'(a)') " Input-reading options:"
  write(unit,'(a)') "   Nuse=N                 N first entries of input files to be used"
  write(unit,'(a)') "   XYZfile=S              file S with xyz coordinates"
  write(unit,'(a)') "   XfileIn=S              file S with input vectors X"
  write(unit,'(a)') "   Yfile=S                file S with reference values"
  write(unit,'(a)') "   YgradFile=S            file S with reference gradients"
  write(unit,'(a)') "   YgradXYZfile=S         file S with reference gradients in XYZ coordinates"
  write(unit,'(a)') "   eqXYZfileIn=S          file S with equilibrium geometry in XYZ coordinates"
  write(unit,'(a)') "   MLmodelIn=S            file S with ML model"
  write(unit,'(a)') "   iTrainIn=S             file S with indices of training points"
  write(unit,'(a)') "   iTestIn=S              file S with indices of test points"
  write(unit,'(a)') "   iSubtrainIn=S          file S with indices of sub-training points"
  write(unit,'(a)') "   iValidateIn=S          file S with indices of validation points"
  write(unit,'(a)') "   iCVtestPrefIn=S        prefix S of files with indices for CVtest"
  write(unit,'(a)') "   iCVoptPrefIn=S         prefix S of files with indices for CVopt"
  write(unit,'(a)') ""
  write(unit,'(a)') " Data set operations:"
  write(unit,'(a)') "   XYZ2X                  convert XYZ coordinates into molecular descriptor X"
  write(unit,'(a)') "                          and write output file with X vectors"
  write(unit,'(a)') "   analyze                analyze data sets"
  write(unit,'(a)') "   sample                 sample data points from a data set"
  write(unit,'(a)') "   sampling=S             type S of data set sampling into training and other sets"
  write(unit,'(a)') "     random [default]     random sampling"
  write(unit,'(a)') "     none                 simply split the data set into the training, validation and test sets"
  write(unit,'(a)') "                          (in this order) without changing the order of indices"
  write(unit,'(a)') "     user-defined         reads in indices for the training, test, and, if necessary,"
  write(unit,'(a)') "                          for the subtraining and validation sets"
  write(unit,'(a)') "                          from files defined by options iTrainIn, iTestIn, iSubtrainIn, iValidateIn"
  write(unit,'(a)') "                          Ntrain, Ntest, Nsubtrain, and Nvalidate have to be specified as well"
  write(unit,'(a)') "                          reads in indices of cross-validation splits from [prefix]*.dat files"
  write(unit,'(a)') "                          with prefixes specified by options iCVtestPrefIn and iCVoptPrefIn"
  write(unit,'(a)') "     structure-based      structure-based sampling [see JCP 2017, 146, 244108]"
  write(unit,'(a)') "     farthest-point       farthest-point traversal iterative procedure"
  write(unit,'(a)') ""
  write(unit,'(a)') " ML task:"
  write(unit,'(a)') "   useMLmodel             use existing ML model"
  write(unit,'(a)') ""
  write(unit,'(a)') "   createMLmodel          create and save ML model"
  write(unit,'(a)') "     Ntrain=R             entries to be used for training ML model"
  write(unit,'(a)') "       R entries for integer R larger or equal to 1"
  write(unit,'(a)') "       fraction of the entire dataset for R < 1.0"
  write(unit,'(a)') "       If this option is not used then R is set equal to"
  write(unit,'(a)') "       the entire dataset [default]"
  write(unit,'(a)') ""
  write(unit,'(a)') "   estAccMLmodel          estimate accuracy of ML"
  write(unit,'(a)') "     Ntrain=R             entries to be used for training ML model"
  write(unit,'(a)') "       R entries for integer R larger or equal to 1"
  write(unit,'(a)') "       fraction of the entire dataset for R < 1.0"
  write(unit,'(a)') "       If this option is not used then R is set equal to"
  write(unit,'(a)') "       80% of the entire dataset [default]"
  write(unit,'(a)') "     Ntest=R              entries to be used for testing ML model"
  write(unit,'(a)') "       R entries for integer R larger or equal to 1"
  write(unit,'(a)') "       fraction of the entire dataset for R < 1.0"
  write(unit,'(a)') "       all points of the entire dataset except for the training set points [default]"
  write(unit,'(a)') "     CVtest               cross-validate ML model for testing"
  write(unit,'(a)') "     NcvTestFolds=N       sets number of folds to N in cross-validation for testing"
  write(unit,'(a)') "       5 [default]"
  write(unit,'(a)') "       <user-defined>"
  write(unit,'(a)') "     LOOtest              leave-one-out cross-validation for testing"
  write(unit,'(a)') ""
  write(unit,'(a)') "   algorithm=S            ML algorithm S"
  write(unit,'(a)') "     KRR [default]. Suboptions"
  write(unit,'(a)') "       KRRtask=S          task for KRR"
  write(unit,'(a)') "         learnVal         learn reference values (default if only Yfile provided)"
  write(unit,'(a)') "         learnGradXYZ        learn only gradients"
  write(unit,'(a)') "         learnValGradXYZ     learn combined value and gradient information"
  write(unit,'(a)') "           <user-defined>"    
  write(unit,'(a)') "       lambda=R           regularization parameter lambda"
  write(unit,'(a)') "         0.0 [default]"
  write(unit,'(a)') "         <user-defined>"
  write(unit,'(a)') "         opt              requests optimization of lambda"
  write(unit,'(a)') "       NlgLambda=N        N points on a logarithmic grid for lambda optimization"
  write(unit,'(a)') "         6 [default]"
  write(unit,'(a)') "         <user-defined>"
  write(unit,'(a)') "       lgLambdaL=R        Lowest value of log2(lambda) for a logarithmic grid optimization of lambda"
  write(unit,'(a)') "         -16.0 [default]"
  write(unit,'(a)') "         <user-defined>"
  write(unit,'(a)') "       lgLambdaH=R        Highest value of log2(lambda) for a logarithmic grid optimization of lambda"
  write(unit,'(a)') "         -6.0 [default]"
  write(unit,'(a)') "         <user-defined>"
  write(unit,'(a)') "       lambdaGradXYZ=R   regularization parameter lambda for XYZ gradients"
  write(unit,'(a)') "         equal to lambda [default]"
  write(unit,'(a)') "         <user-defined>"
  write(unit,'(a)') "       kernel=S           kernel S"
  
! START of check
!  write(unit,'(a)') "         linear"
!  write(unit,'(a)') "         polynomial. Subsuboptions:"
!  write(unit,'(a)') "            c=S           addend"
!  write(unit,'(a)') "              100.0 [default]"
!  write(unit,'(a)') "              <user-defined value>"
!  write(unit,'(a)') "              opt          requests optimization of c"
!  write(unit,'(a)') "            NlgC=N         N points on a logarithmic grid for c optimization"
!  write(unit,'(a)') "              11 [default]"
!  write(unit,'(a)') "              <user-defined>"
!  write(unit,'(a)') "            lgCl=R        lowest value of log2(c) for a logarithmic grid optimization of c"
!  write(unit,'(a)') "              2.0 [default]"
!  write(unit,'(a)') "              <user-defined>"
!  write(unit,'(a)') "            lgCh=R        highest value of log2(c) for a logarithmic grid optimization of c"
!  write(unit,'(a)') "              9.0 [default]"
!  write(unit,'(a)') "              <user-defined>"
!  write(unit,'(a)') "            d=N           exponent"
!  write(unit,'(a)') "              2 [default]"
!  write(unit,'(a)') "              <user-defined value>"
! End of check

! START of does not work yet
!  write(unit,'(a)') "               opt        requests optimization of d"
!  write(unit,'(a)') "               Note: d is not optimized in cross-validation, specify d='OPT' to optimize it"
!  write(unit,'(a)') "             NlgD=N       N points on a logarithmid grid for d optimization"
!  write(unit,'(a)') "               11 [default]"
!  write(unit,'(a)') "               <user-defined>"
!  write(unit,'(a)') "             lgDl=R       lowest value of log2(d) for a logarithmid grid optimization of d"
!  write(unit,'(a)') "               2.0 [default]"
!  write(unit,'(a)') "               <user-defined>"
!  write(unit,'(a)') "             lgDh=R       highest value of log2(d) for a logarithmid grid optimization of d"
!  write(unit,'(a)') "               9.0 [default]"
!  write(unit,'(a)') "               <user-defined>"
! END of does not work yet

  write(unit,'(a)') "         Gaussian [default]. Subsuboptions:"
  write(unit,'(a)') "            sigma=S       length scale"
  write(unit,'(a)') "              100.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "              opt         requests optimization of sigma"
  write(unit,'(a)') "            NlgSigma=N    N points on a logarithmic grid for sigma optimization"
  write(unit,'(a)') "              11 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaL=R   lowest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              2.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaH=R    highest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              9.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            periodKernel  periodic kernel"
  write(unit,'(a)') "               period=R   period"
  write(unit,'(a)') "                 1.0 [default]"
  write(unit,'(a)') "                 <user-defined>"
  write(unit,'(a)') "            decayKernel   decaying periodic kernel"
  write(unit,'(a)') "              period=R    period"
  write(unit,'(a)') "                 1.0 [default]"
  write(unit,'(a)') "                 <user-defined>"
  write(unit,'(a)') "              sigmap=R    length scale for a periodic part"
  write(unit,'(a)') "                100.0 [default]"
  write(unit,'(a)') "                <user-defined>"
  write(unit,'(a)') "         Laplacian. Subsuboptions:"
  write(unit,'(a)') "            sigma=S       length scale"
  write(unit,'(a)') "              800.0 [default]"
  write(unit,'(a)') "              <user-defined value>"
  write(unit,'(a)') "              opt         requests optimization of sigma"
  write(unit,'(a)') "            NlgSigma=S    S points on a logarithmic grid for sigma optimization"
  write(unit,'(a)') "              11 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaL=R    lowest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              5.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaH=R   highest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              12.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "         exponential. Subsuboptions:"
  write(unit,'(a)') "            sigma=S       length scale"
  write(unit,'(a)') "              800.0 [default]"
  write(unit,'(a)') "              <user-defined value>"
  write(unit,'(a)') "              opt         requests optimization of sigma"
  write(unit,'(a)') "            NlgSigma=S    S points on a logarithmic grid for sigma optimization"
  write(unit,'(a)') "              11 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaL=R    lowest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              5.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaH=R   highest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              12.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "         Matern. Subsuboptions:"
  write(unit,'(a)') "            sigma=S      length-scale"
  write(unit,'(a)') "              100.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "              opt        requests optimization of sigma"
  write(unit,'(a)') "            NlgSigma=N   N points on a logarithmic grid for sigma optimization"
  write(unit,'(a)') "              11 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaL=R   lowest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              2.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            lgSigmaH=R   highest value of log2(sigma) for a logarithmic grid optimization of sigma"
  write(unit,'(a)') "              9.0 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') "            nn=N         n in the Matern kernel (nu = n + 1/2)"
  write(unit,'(a)') "              2 [default]"
  write(unit,'(a)') "              <user-defined>"
  write(unit,'(a)') ""
  write(unit,'(a)') " The following options apply, if hyperparameter optimization is requested:"
  write(unit,'(a)') "   minimizeError=S       type S of error to minimize"
  write(unit,'(a)') "     MAE                 mean absolute error"
  write(unit,'(a)') "     RMSE                root-mean-square error [default]"
  write(unit,'(a)') "   lgOptDepth=N          depth of log. opt. N"
  write(unit,'(a)') "     3 [default]"
  write(unit,'(a)') "     <user-defined>"
  write(unit,'(a)') "   Nsubtrain=R           entries in the sub-training set"
  write(unit,'(a)') "     R entries for integer R larger or equal to 1"
  write(unit,'(a)') "     fraction of the training set for R < 1.0"
  write(unit,'(a)') "     80% of the training set [default]"
  write(unit,'(a)') "   Nvalidate=R           entries to be used for validating ML model trained on the sub-training set"
  write(unit,'(a)') "     R entries of the training set for integer R larger or equal to 1"
  write(unit,'(a)') "     fraction of the training set for R < 1.0"
  write(unit,'(a)') "     all points of the training set except for the sub-training set points [default]"
  write(unit,'(a)') "   CVopt                 cross-validate ML model for optimization of hyperparameters"
  write(unit,'(a)') "   NcvOptFolds=N         sets number of folds to N in cross-validation for optimization of hyperparameters"
  write(unit,'(a)') "     5 [default]"
  write(unit,'(a)') "     <user-defined>"
  write(unit,'(a)') "   LOOopt                leave-one-out cross-validation for hyperparameter optimization"
  write(unit,'(a)') ""
  write(unit,'(a)') "   prior=S               prior S"
  write(unit,'(a)') "     0 [default]"
  write(unit,'(a)') "     mean                average value of Y values in the (sub-)training set"
  write(unit,'(a)') "     <user-defined>      user-defined floating point number"
  write(unit,'(a)') ""
  write(unit,'(a)') " Options related to atomistic simulations"
  write(unit,'(a)') "   permInvKernel         calculations with permutationally invariant kernel"
  write(unit,'(a)') "   Nperm=N               number of permutations N"
  write(unit,'(a)') "   permInvGroups=S       permutationally invariant groups S"
  write(unit,'(a)') "   permInvNuclei=S       permutationally invariant nuclei S"
  write(unit,'(a)') "   molDescriptor=S       molecular descriptor S"
  write(unit,'(a)') "     RE [default]        vector {Req/R}, where R is internuclear distance"
  write(unit,'(a)') "     CM                  Coulomb matrix"
  write(unit,'(a)') "     ID                  inverse internuclear distances"
  write(unit,'(a)') "   molDescrType=S        type S of molecular descriptor"
  write(unit,'(a)') "     sorted              sort by:"
  write(unit,'(a)') "                         norms of CM matrix (for molDescrType=CM [default])"
  write(unit,'(a)') "                         nuclear repulsions (for molDescrType=RE)"
  write(unit,'(a)') "     unsorted            option is [default] for molDescrType=RE"
  write(unit,'(a)') "     permuted            generate molecular descriptor for each permutation of atoms"
  write(unit,'(a)') "   selectperm            calculations with selected permutations"
  write(unit,'(a)') "                         select permutations by minimizing dRMSD"
  write(unit,'(a)') ""
  write(unit,'(a)') " Program output options:" 
  write(unit,'(a)') "   XfileOut=S             file S with X values"
  write(unit,'(a)') "   XYZsortedFileOut=S     file S with sorted XYZ coordinates"
  write(unit,'(a)') "                          only works for molDescriptor=RE molDescrType=sorted"    
  write(unit,'(a)') "   MLmodelOut=S           file S with ML model" 
  write(unit,'(a)') "   YestFile=S             file S with estimated Y values"
  write(unit,'(a)') "   YgradEstFile=S         file S with estimated gradients"
  write(unit,'(a)') "   YgradXYZestFile=S      file S with estimated XYZ gradients"
  write(unit,'(a)') "   iTrainOut=S            file S with indices of training points"
  write(unit,'(a)') "   iTestOut=S             file S with indices of test points"
  write(unit,'(a)') "   iSubtrainOut=S         file S with indices of sub-training points"
  write(unit,'(a)') "   iValidateOut=S         file S with indices of validation points"
  write(unit,'(a)') "   iCVtestPrefOut=S       prefix S of files with indices for CVtest"
  write(unit,'(a)') "   iCVoptPrefOut=S        prefix S of files with indices for CVopt"
  write(unit,'(a)') "   benchmark              additional output for benchmarking"
  write(unit,'(a)') "   debug                  additional output for debugging"
  write(unit,'(a)') ""
  write(unit,'(a)') " Program run options:"
  write(unit,'(a)') "   matDecomp=S            type of matrix decomposition"
  write(unit,'(a)') "     Cholesky             default"
  write(unit,'(a)') "     LU                   "
  write(unit,'(a)') "     Bunch-Kaufman        "
  write(unit,'(a)') "   invMatrix              invert matrix"
  write(unit,'(a)') "   refine                 refine solution matrix"
  write(unit,'(a)') "   on-the-fly             on-the-fly calculation of kernel"
  write(unit,'(a)') "                          matrix elements for validation,"
  write(unit,'(a)') "                          by default it is false and those"
  write(unit,'(a)') "                          elements are stored"
  write(unit,'(a)') "   numDerivs              numerical derivatives"
  write(unit,'(a)') ""
  
end subroutine help

subroutine getOptions()
  implicit none
  integer            :: nOptions        ! Number of options
  integer            :: i               ! Loop index
  character(len=15000):: userOption      ! User-defined options as provided and with all upper-case letters
  character(len=256) :: userOptions(50) ! User-defined options as provided
  character(len=256) :: errorMessage                

  errorMessage = ''
  userOptions  = ''

  ! Get number of options
  nOptions = iargc()
  if (nOptions > 50) then
    call exitHelp('Maximum number of options (50) exceeded')
  end if

  ! Parse options
  do i=1,nOptions
    call getarg(i,userOption)
    userOptions(i) = userOption
    call parseOption(userOption, errorMessage)
    if (errorMessage /= '' .or. option%help) then
      call exitHelp(errorMessage)
    end if
  enddo

  ! Check options
  call checkOptions(errorMessage)
  if (errorMessage /= '') then
    call exitHelp(errorMessage)
  end if

end subroutine getOptions

subroutine exitHelp(errorMessage)
  use MLatomFInfo, only : header
  use stopper,     only : stopMLatomF
  implicit none
  character(len=*), intent(in) :: errorMessage              

  call header()
  call help(6)
  if (errorMessage == 'HELP') then
    call stopMLatomF('')
  else
    call stopMLatomF(errorMessage)
  end if

end subroutine exitHelp

subroutine parseOption(userOption, errorMessage)
  use strings, only : strUpper
  implicit none
  ! Arguments
  character(len=*),   intent(in)  :: userOption
  character(len=256), intent(out) :: errorMessage
  ! Variables
  character(len=80) :: temp

  errorMessage = ''
  temp = ''

  ! Help
  if ((strUpper(userOption) == '-H') .or. (strUpper(userOption) == 'HELP') &
       .or. (strUpper(userOption) == '-HELP') .or. (strUpper(userOption) == '--HELP')) then
    option%help = .true.

  ! Input read options
  elseif (strUpper(userOption(1:5))=='NUSE=') then
    option%Nuse = userOption(6:)
  elseif (strUpper(userOption(1:8))=='XYZFILE=') then
    readXYZ = .true.
    option%XYZfile = userOption(9:)
  elseif (strUpper(userOption(1:6))=='YFILE=') then
    option%Yfile = userOption(7:)
  elseif (strUpper(userOption(1:10))=='YGRADFILE=') then
    option%YgradFile = userOption(11:)
  elseif (strUpper(userOption(1:13))=='YGRADXYZFILE=') then
    option%YgradXYZfile = userOption(14:)
  elseif (strUpper(userOption(1:12))=='EQXYZFILEIN=') then 
    option%eqXYZfileIn = userOption(13:)
  elseif (strUpper(userOption(1:8))=='XFILEIN=') then
    option%readX = .true.
    option%XfileIn = userOption(9:)
  elseif (strUpper(userOption(1:10))=='MLMODELIN=') then
    option%MLmodelFileIn = userOption(11:)
  elseif (strUpper(userOption(1:9))=='ITRAININ=') then
    option%iTrainIn = userOption(10:)
  elseif (strUpper(userOption(1:8))=='ITESTIN=') then
    option%iTestIn = userOption(9:)
  elseif (strUpper(userOption(1:12))=='ISUBTRAININ=') then
    option%iSubtrainIn = userOption(13:)
  elseif (strUpper(userOption(1:12))=='IVALIDATEIN=') then
    option%iValidateIn = userOption(13:)
  elseif (strUpper(userOption(1:14))=='ICVTESTPREFIN=') then
    option%iCVtestPrefIn = userOption(15:)
  elseif (strUpper(userOption(1:13))=='ICVOPTPREFIN=') then
    option%iCVoptPrefIn = userOption(14:)
  elseif (strUpper(userOption(1:17))=='ICVOPTPREFGRADIN=') then 
    option%iCVoptPrefGradIn = userOption(18:)
  elseif (strUpper(userOption(1:16))=='ITRAINGRADXYZIN=') then 
    option%iTrainGradXYZIn = userOption(17:)
  elseif (strUpper(userOption(1:19))=='ISUBTRAINGRADXYZIN=') then 
    option%iSubtrainGradXYZIn = userOption(20:)
  elseif (strUpper(userOption(1:19))=='IVALIDATEGRADXYZIN=') then 
    option%iValidateGradXYZIn = userOption(20:)

  elseif (strUpper(userOption(1:10))=='PERMINDIN=') then 
    option%permIndIn = userOption(11:)
    option%usePermInd = .true.
  elseif (strUpper(userOption(1:8))=='PERMLEN=') then 
    option%permlen = userOption(9:)
    
  ! Data set operations
  elseif (strUpper(userOption(1:5))=='XYZ2X') then
    option%XYZ2X = .true.
    option%writeX = .true.
  elseif (strUpper(userOption(1:7))=='ANALYZE') then
    option%analyze = .true.
  elseif (strUpper(userOption(1:6))=='SAMPLE') then
    option%sample = .true.
  elseif (strUpper(userOption(1:9))=='SAMPLING=') then
    option%sampling = strUpper(userOption(10:))
    if (option%sampling == strUpper('none')) then
      option%sampling = 'none'
    elseif (option%sampling == strUpper('random')) then
      option%sampling = 'random'
    elseif (option%sampling == strUpper('structure-based')) then
      option%sampling = 'structure-based'
    elseif (option%sampling == strUpper('farthest-point')) then
      option%sampling = 'farthest-point'
    elseif (option%sampling == strUpper('user-defined')) then
      option%sampling = 'user-defined'
    else
      errorMessage = 'Unsupported type of sampling : '//option%sampling
      return
    end if

  ! ML Task
  elseif (strUpper(userOption(1:10))=='USEMLMODEL') then
    option%useMLmodel    = .true.
    option%createMLmodel = .false.
    option%estAccMLmodel = .false.

  elseif (strUpper(userOption(1:13))=='CREATEMLMODEL') then
    option%createMLmodel = .true.
    option%useMLmodel    = .false.
    option%estAccMLmodel = .false.
  elseif (strUpper(userOption(1:7))=='NTRAIN=') then
    option%Ntrain = strUpper(userOption(8:))
  elseif (strUpper(userOption(1:14))=='NTRAINGRADXYZ=') then 
    option%NtrainGradXYZ = strUpper(userOption(15:))

  elseif (strUpper(userOption(1:13))=='ESTACCMLMODEL') then
    option%estAccMLmodel = .true.    
  
  ! Estimate accuracy of the ML model
  elseif (strUpper(userOption(1:6))=='CVTEST') then
    option%CVtest = .true.
  elseif (strUpper(userOption(1:13))=='NCVTESTFOLDS=') then
    read(userOption(14:),*) option%NcvTestFolds
    NcvTestFoldsFlag = .true.
  elseif (strUpper(userOption(1:7))=='LOOTEST') then
    option%CVtest = .true.
    option%LOOtest = .true.
    
  elseif (strUpper(userOption(1:6))=='NTEST=') then
    option%Ntest = strUpper(userOption(7:))

  ! ML Algorithm
  elseif (strUpper(userOption(1:10))=='ALGORITHM=') then
    option%algorithm = strUpper(userOption(11:))
    if (option%algorithm == strUpper('KRR')) then
      option%algorithm = 'KRR'
    else
      errorMessage = 'Unsupported ML algorithm : '//option%algorithm
      return
    end if
    
  elseif (strUpper(userOption(1:8))=='KRRTASK=') then
    option%KRRtask = strUpper(userOption(9:))
    if (option%KRRtask == strUpper('LEARNVALGRADXYZ')) then
      option%KRRtask = 'learnValGradXYZ'
    elseif (option%KRRtask == strUpper('LEARNVAL')) then
      option%KRRtask = 'learnVal'
      learnValFlag = .true.
    elseif (option%KRRtask == strUpper('LEARNGRADXYZ')) then
      option%KRRtask = 'learnGradXYZ'
    else
      errorMessage = 'Unsupported KRR task : '//option%KRRtask
      return
    end if

  elseif (strUpper(userOption(1:7))=='LAMBDA=') then
    option%lambda    = strUpper(userOption(8:))
  elseif (strUpper(userOption(1:10))=='NLGLAMBDA=') then
    option%NlgLambda = strUpper(userOption(11:))
  elseif (strUpper(userOption(1:10))=='LGLAMBDAL=') then
    option%lgLambdaL = strUpper(userOption(11:))
  elseif (strUpper(userOption(1:10))=='LGLAMBDAH=') then
    option%lgLambdaH = strUpper(userOption(11:))
  elseif (strUpper(userOption(1:14))=='LAMBDAGRADXYZ=') then
    option%lambdaGradXYZ= strUpper(userOption(15:))
      
  elseif (strUpper(userOption(1:7))=='KERNEL=') then
    option%kernel = strUpper(userOption(8:))
    if (option%kernel == strUpper('linear')) then
      option%kernel = 'linear'
    elseif (option%kernel == strUpper('polynomial')) then
      option%kernel = 'polynomial'
    elseif (option%kernel == strUpper('Gaussian')) then
      option%kernel = 'Gaussian'
    elseif (option%kernel == strUpper('Laplacian')) then
      option%kernel = 'Laplacian'
    elseif (option%kernel == strUpper('exponential')) then
      option%kernel = 'exponential'
    elseif (option%kernel == strUpper('Matern')) then
      option%kernel = 'Matern'
    else
      errorMessage = 'Unsupported kernel : '//option%kernel
      return
    end if

  elseif (strUpper(userOption(1:6))=='SIGMA=') then
    option%sigma = strUpper(userOption(7:))
    sigmaFlag    = .true.
  elseif (strUpper(userOption(1:9))=='NLGSIGMA=') then
    option%NlgSigma = strUpper(userOption(10:))
    NlgSigmaFlag    = .true.
  elseif (strUpper(userOption(1:9))=='LGSIGMAL=') then
    option%lgSigmaL = strUpper(userOption(10:))
    lgSigmaLflag    = .true.
  elseif (strUpper(userOption(1:9))=='LGSIGMAH=') then
    option%lgSigmaH = strUpper(userOption(10:))
    lgSigmaHflag    = .true.
  elseif (strUpper(userOption(1:12))=='PERIODKERNEL') then
    option%periodKernel = .true.
  elseif (strUpper(userOption(1:7))=='PERIOD=') then
    option%period = strUpper(userOption(8:))
  elseif (strUpper(userOption(1:11))=='DECAYKERNEL') then
    option%decayKernel = .true.
  elseif (strUpper(userOption(1:7))=='SIGMAP=') then
    option%sigmap = strUpper(userOption(8:))
    
  elseif (strUpper(userOption(1:2))=='C=') then
    option%c = strUpper(userOption(3:))
  elseif (strUpper(userOption(1:5))=='NLGC=') then
    option%NlgC = strUpper(userOption(6:))
  elseif (strUpper(userOption(1:5))=='LGCL=') then
    option%lgCl = strUpper(userOption(6:))
  elseif (strUpper(userOption(1:5))=='LGCH=') then
    option%lgCh = strUpper(userOption(6:))
    
  elseif (strUpper(userOption(1:2))=='D=') then
    option%d = strUpper(userOption(3:))
  elseif (strUpper(userOption(1:5))=='NLGD=') then
    option%NlgD = strUpper(userOption(6:))
  elseif (strUpper(userOption(1:5))=='LGDL=') then
    option%lgDl = strUpper(userOption(6:))
  elseif (strUpper(userOption(1:5))=='LGDH=') then
    option%lgDh = strUpper(userOption(6:))

  elseif (strUpper(userOption(1:3))=='NN=') then
    option%nn = strUpper(userOption(4:))
    
  elseif (strUpper(userOption(1:14))=='MINIMIZEERROR=') then
    option%minimizeError    = strUpper(userOption(15:))
  elseif (strUpper(userOption(1:11))=='LGOPTDEPTH=') then
    option%lgOptDepth    = strUpper(userOption(12:))
  elseif (strUpper(userOption(1:10))=='NSUBTRAIN=') then
    option%Nsubtrain = strUpper(userOption(11:))
  elseif (strUpper(userOption(1:10))=='NVALIDATE=') then
    option%Nvalidate = strUpper(userOption(11:))
  elseif (strUpper(userOption(1:17))=='NSUBTRAINGRADXYZ=') then 
    option%NsubtrainGradXYZ = strUpper(userOption(18:))
  elseif (strUpper(userOption(1:17))=='NVALIDATEGRADXYZ=') then 
    option%NvalidateGradXYZ = strUpper(userOption(18:))
    
  ! Options relevant, If hyperparameter optimization is requested
  elseif (strUpper(userOption(1:5))=='CVOPT') then
    option%CVopt = .true.
  elseif (strUpper(userOption(1:12))=='NCVOPTFOLDS=') then
    read(userOption(13:),*) option%NcvOptFolds
    NcvOptFoldsFlag = .true.
  elseif (strUpper(userOption(1:6))=='LOOOPT') then
    option%CVopt = .true.
    option%LOOopt = .true.
    
  ! Prior
  elseif (strUpper(userOption(1:6))=='PRIOR=') then
    option%prior = strUpper(userOption(7:))
    if     (option%prior == strUpper('0')) then
      option%prior = '0'
    elseif (option%prior == strUpper('mean')) then
      option%prior = 'mean'
    end if
    
  ! Options related to atomistic simulations
  ! Permutational symmetry information
  elseif (strUpper(userOption(1:13))=='PERMINVKERNEL') then
    option%permInvKernel = .true.
  elseif (strUpper(userOption(1:6))=='NPERM=') then
    option%Nperm = strUpper(userOption(7:))
  elseif (strUpper(userOption(1:14))=='PERMINVGROUPS=') then
    option%permInvGroups = strUpper(userOption(15:))
  elseif (strUpper(userOption(1:14))=='PERMINVNUCLEI=') then
    option%permInvNuclei = strUpper(userOption(15:))
    elseif (strUpper(userOption(1:10))=='SELECTPERM') then 
      option%selectPerm = .true.
  ! Molecular descriptor
  elseif (strUpper(userOption(1:14))=='MOLDESCRIPTOR=') then
    option%molDescriptor = strUpper(userOption(15:))
    if (option%molDescriptor == strUpper('CM')) then
      option%molDescriptor = 'CM'
    elseif (option%molDescriptor == strUpper('RE')) then
      option%molDescriptor = 'RE'
    elseif (option%molDescriptor == strUpper('ID')) then
      option%molDescriptor = 'ID'
    else
      errorMessage = 'Unsupported molecular descriptor : '//option%molDescriptor
      return
    end if
  elseif (strUpper(userOption(1:13))=='MOLDESCRTYPE=') then
    option%molDescrType = strUpper(userOption(14:))
    molDescTypeFlag     = .true.
    if (trim(option%molDescrType) == strUpper('sorted')) then
      option%molDescrType = 'sorted'
    elseif (trim(option%molDescrType) == strUpper('unsorted')) then
      option%molDescrType = 'unsorted'
    elseif (trim(option%molDescrType) == strUpper('permuted')) then
      option%molDescrType = 'permuted'
    else
      errorMessage = 'Unsupported type of molecular descriptor : '//option%molDescrType
      return
    end if
  elseif (strUpper(userOption(1:14)) == 'DESCRIPTORDIST') then 
    option%descriptorDist = .true.
  ! Output options
  elseif (strUpper(userOption(1:9))=='XFILEOUT=') then
    option%writeX = .true.
    option%XfileOut = userOption(10:)
  elseif (strUpper(userOption(1:17))=='XYZSORTEDFILEOUT=') then
    option%writeXYZsorted = .true.
    option%XYZsortedFileOut = userOption(18:)
  elseif (strUpper(userOption(1:11))=='MLMODELOUT=') then
    option%MLmodelFileOut = userOption(12:)
  elseif (strUpper(userOption(1:9))=='YESTFILE=') then
    option%YestFile = userOption(10:)
    option%writeEst = .true.
  elseif (strUpper(userOption(1:13))=='YGRADESTFILE=') then
    option%YgradEstFile = userOption(14:)
    option%writeEst = .true.
  elseif (strUpper(userOption(1:16))=='YGRADXYZESTFILE=') then
    option%YgradXYZestFile = userOption(17:)
    option%writeEst = .true.
  elseif (strUpper(userOption(1:10))=='ITRAINOUT=') then
    option%iTrainOut = userOption(11:)
  elseif (strUpper(userOption(1:9))=='ITESTOUT=') then
    option%iTestOut = userOption(10:)
  elseif (strUpper(userOption(1:13))=='ISUBTRAINOUT=') then
    option%iSubtrainOut = userOption(14:)
  elseif (strUpper(userOption(1:13))=='IVALIDATEOUT=') then
    option%iValidateOut = userOption(14:)
  elseif (strUpper(userOption(1:15))=='ICVTESTPREFOUT=') then
    option%iCVtestPrefOut = userOption(16:)
  elseif (strUpper(userOption(1:14))=='ICVOPTPREFOUT=') then
    option%iCVoptPrefOut = userOption(15:)    
  elseif (strUpper(userOption(1:9))=='BENCHMARK') then
    option%benchmark = .true.
  elseif (strUpper(userOption(1:5))=='DEBUG') then
    option%debug = .true.

  ! Program run options
  elseif (strUpper(userOption(1:10))=='MATDECOMP=') then
    option%matDecomp = strUpper(userOption(11:))
    if (option%matDecomp == strUpper('Cholesky')) then
      option%matDecomp = 'Cholesky'
    elseif (option%matDecomp == strUpper('LU')) then
      option%matDecomp = 'LU'
    elseif (option%matDecomp == strUpper('Bunch-Kaufman')) then
      option%matDecomp = 'Bunch-Kaufman'
    else
      errorMessage = 'Unsupported matrix decomposition : '//option%matDecomp
      return
    end if
  elseif (strUpper(userOption(1:9))=='INVMATRIX') then
    option%invMatrix = .true.
  elseif (strUpper(userOption(1:6))=='REFINE') then
    option%refine = .true.
  elseif (strUpper(userOption(1:10))=='ON-THE-FLY') then
    option%onTheFly = .true.
  elseif (strUpper(userOption(1:9))=='NUMDERIVS') then
    option%numDerivs = .true.

  ! Stop
  else
    errorMessage = 'Unrecognized option ' // userOption
    return
  end if

end subroutine parseOption

subroutine checkOptions(errorMessage)
  implicit none
  character(len=256), intent(out) :: errorMessage
  
  errorMessage = ''

  if (readXYZ .and. option%readX) then
    errorMessage = 'Choose input file with either XYZfile or XfileIn option, but not both'
  elseif (.not. readXYZ .and. .not. option%readX .and. .not. option%analyze) then
    errorMessage = 'Choose input file with either XYZfile or XfileIn option'
  elseif (readXYZ) then
    if (.not. checkFile(option%XYZfile)) errorMessage = 'Provide correct file name with XYZfile option'
  elseif (option%readX) then
    if (.not. checkFile(option%XfileIn)) errorMessage = 'Provide correct file name with XfileIn option'
  end if
  
  if (option%XYZ2X .and. .not. checkFile(option%XYZfile)) then
    errorMessage = 'Provide correct file name with XYZfile option'
  end if
  
  if (option%createMLmodel .or. option%estAccMLmodel) then
    if (option%KRRtask == 'learnVal' .and. trim(option%Yfile) == '' &
        .and. trim(option%YgradXYZfile) /= '') option%KRRtask = 'learnGradXYZ'
    if (option%KRRtask == 'learnVal' .and. trim(option%Yfile) == '' &
        .and. trim(option%YgradFile) /= '') option%KRRtask = 'learnGrad'
    if (option%KRRtask == 'learnVal' .and. trim(option%Yfile) /= '' &
       .and. trim(option%YgradXYZfile) /= '' .and. .not. learnValFlag) &
       option%KRRtask = 'learnValGradXYZ'
    if (option%KRRtask == 'learnVal' .and. trim(option%Yfile) == '') then
      errorMessage = 'KRRtask=learnVal must be used with Yfile option'
    elseif (option%KRRtask == 'learnValGradXYZ' .and. trim(option%Yfile) == '' .and. trim(option%YgradXYZfile) == '') then
      errorMessage = 'KRRtask=learnValGradXYZ must be used with Yfile and YgradXYZfile options'
    elseif (option%KRRtask == 'learnGradXYZ' .and. trim(option%YgradXYZfile) == '') then
      errorMessage = 'KRRtask=learnGradXYZ must be used with YgradXYZfile option'
    end if
  end if
    
  if (option%useMLmodel .and. option%createMLmodel) then
    errorMessage = 'Using and creating ML model cannot be done in one run'
  elseif (option%useMLmodel .and. option%estAccMLmodel) then
    errorMessage = 'Using and estimating accuracy of ML model cannot be done in one run'
  elseif (option%createMLmodel .and. option%estAccMLmodel) then
    errorMessage = 'Creating and estimating accuracy of ML model cannot be done in one run'
  elseif ((option%useMLmodel .or. option%createMLmodel .or. option%estAccMLmodel) .and. &
          option%XYZ2X) then
    errorMessage = 'XYZ2X cannot be used together with useMLmodel, createMLmodel, or estAccMLmodel'
  elseif ((option%useMLmodel .or. option%createMLmodel .or. option%estAccMLmodel) .and. &
          option%sample) then
    errorMessage = 'sample cannot be used together with useMLmodel, createMLmodel, or estAccMLmodel'
  elseif ((option%useMLmodel .or. option%createMLmodel .or. option%estAccMLmodel) .and. &
          option%analyze) then
    errorMessage = 'analyze cannot be used together with useMLmodel, createMLmodel, or estAccMLmodel'
  elseif (option%sample .and. option%analyze) then
    errorMessage = 'analyze cannot be used together with sample'
  end if

  if (.not. option%useMLmodel    .and. .not. option%createMLmodel .and. &
      .not. option%estAccMLmodel .and. .not. option%XYZ2X         .and. &
      .not. option%sample        .and. .not. option%analyze) then
    errorMessage = 'At least one task for MLatomF should be set'
  end if
  
  if ((option%createMLmodel .or. option%estAccMLmodel) .and. &
    .not. (checkFile(option%Yfile) .or. checkFile(option%YgradFile) .or. checkFile(option%YgradXYZfile))) then
      errorMessage = 'Provide correct file name with Yfile or YgradFile or YgradXYZfile option'
  end if
  
  if (.not. (option%MLmodelFileIn /= '' .or. option%YgradXYZfile /= '') .and. option%YgradXYZestFile /= '') then
    errorMessage = 'Gradients can be calculated only with MLmodelIn or YgradXYZfile option'
  end if
  
  if (option%useMLmodel .and. .not. checkFile(option%MLmodelFileIn)) then
    errorMessage = 'Provide correct file name with MLmodelIn option'
  end if
  
  if (trim(option%sampling) == 'user-defined') then
    if     (trim(option%iTrainIn)/='') then
      if (.not. checkFile(option%iTrainIn)) errorMessage = 'Provide correct file name with iTrainIn option'
    elseif (trim(option%iTestIn)/='') then
      if (.not. checkFile(option%iTestIn)) errorMessage = 'Provide correct file name with iTestIn option'
    elseif (trim(option%iSubtrainIn)/='' .and. .not. option%CVtest) then
      if (.not. checkFile(option%iSubtrainIn)) errorMessage = 'Provide correct file name with iSubtrainIn option'
    elseif (trim(option%iValidateIn)/='' .and. .not. option%CVtest) then
      if (.not. checkFile(option%iValidateIn)) errorMessage = 'Provide correct file name with iValidateIn option'
    end if
    if (option%CVopt .and. ((trim(option%iTrainIn)/='') .and. trim(option%iCVoptPrefIn)=='')) then
      errorMessage = 'CVopt cannot be used with sampling=user-defined and iTrainIn'
    end if
  end if
  
  if (option%writeX) then
    if (trim(option%XfileOut) == '') then
      errorMessage = 'Provide correct file name with XfileOut option'
    elseif (checkFile(option%XfileOut)) then
      errorMessage = 'File ' // trim(option%XfileOut) // ' already exists, delete or rename it'
    end if
  end if
  
  if (option%writeXYZsorted) then
    if (trim(option%XYZsortedFileOut) == '') then
      errorMessage = 'Provide correct file name with XYZsortedFileOut option'
    elseif (checkFile(option%XYZsortedFileOut)) then
      errorMessage = 'File ' // trim(option%XYZsortedFileOut) // ' already exists, delete or rename it'
    end if
  end if
  
  if (trim(option%YestFile) /= '' .and. .not. option%analyze) then
    if (checkFile(option%YestFile)) errorMessage = 'File ' // trim(option%YestFile) // ' already exists, delete or rename it'
  end if
  
  if (trim(option%YgradEstFile) /= '' .and. .not. option%analyze) then
    if (checkFile(option%YgradEstFile)) errorMessage = 'File ' // &
      trim(option%YgradEstFile) // ' already exists, delete or rename it'
  end if
  
  if (trim(option%YgradXYZestFile) /= '' .and. .not. option%analyze) then
    if (checkFile(option%YgradXYZestFile)) errorMessage = 'File ' // &
      trim(option%YgradXYZestFile) // ' already exists, delete or rename it'
  end if
  
  if (trim(option%iTrainOut) /= '') then
    if (checkFile(option%iTrainOut)) then
      errorMessage = 'File ' // trim(option%iTrainOut) // ' already exists, delete or rename it'
    end if
  end if
  if (trim(option%iTestOut) /= '') then
    if (checkFile(option%iTestOut)) then
      errorMessage = 'File ' // trim(option%iTestOut) // ' already exists, delete or rename it'
    end if
  end if
  if (trim(option%iSubtrainOut) /= '') then
    if (checkFile(option%iSubtrainOut)) then
      errorMessage = 'File ' // trim(option%iSubtrainOut) // ' already exists, delete or rename it'
    end if
  end if
  if (trim(option%iValidateOut) /= '') then
    if (checkFile(option%iValidateOut)) then
      errorMessage = 'File ' // trim(option%iValidateOut) // ' already exists, delete or rename it'
    end if
  end if
  ! Implement checks for cross-validation files, otherwise they are re-written by default
  !character(len=256)   :: iCVtestPrefOut= ''                        ! Prefix of file with indices of cross-validation test splits
  !character(len=256)   :: iCVoptPrefOut = ''                        ! Prefix of file with indices of cross-validation splits for hyperparameter optimization
  
  if (option%createMLmodel) then
    if (trim(option%MLmodelFileOut) == '') then
      errorMessage = 'Provide correct file name with MLmodelOut option'
    elseif (checkFile(option%MLmodelFileOut)) then
      errorMessage = 'File ' // trim(option%MLmodelFileOut) // ' already exists, delete or rename it'
    end if
  end if

  if (option%CVtest .and. .not. (option%estAccMLmodel .or. option%sample)) then
      errorMessage = 'Cross-validation for estimating generalization error has to be run with estAccMLmodel or sample options'
  end if

  if (option%LOOtest .and. NcvTestFoldsFlag) then
      errorMessage = 'NcvTestFolds cannot be requested with leave-one-out cross-validation'
  end if

  if (option%LOOopt .and. NcvOptFoldsFlag) then
      errorMessage = 'NcvOptFolds cannot be requested with leave-one-out cross-validation'
  end if
  
  if (option%CVtest .or. option%CVopt) then
    if (trim(option%sampling) == 'structure-based' .or. trim(option%sampling) == 'farthest-point') then
      errorMessage = 'Cross-validation cannot be used with the structure-based or farthest-point sampling'
    end if
  end if

  if ((option%lambda == 'OPT') .or. &
      (option%sigma  == 'OPT') .or. &
      (option%c      == 'OPT')) then ! .or. &
!      (option%d      == 'OPT')) then
    option%hyperOpt = .true.
  end if
  
  ! Change default options, if Laplacian kernel was requested, but options were not user-defined
  if (trim(option%kernel) == 'Laplacian' .or. trim(option%kernel) == 'exponential') then
    if (.not. sigmaFlag)    option%sigma    = '800' ! Length-scale
    if (.not. lgSigmaLflag) option%lgSigmaL = '5'   ! The lowest  value of log2(sigma) to try
    if (.not. lgSigmaHflag) option%lgSigmaH = '12'  ! The highest value of log2(sigma) to try
  end if

  if ( .not. ( &
       (option%minimizeError(1:3) == 'MAE')       .or. &
       (option%minimizeError(1:4) == 'RMSE')       .or. &
       (option%minimizeError(1:2) == 'R2') ) ) then
    errorMessage = 'Unsupported minimizeError type'
  end if
  
  ! Change default option for RE molecular descriptor
  if (trim(option%molDescriptor) == 'RE' .or. trim(option%molDescriptor) == 'ID') then
    if (.not. molDescTypeFlag) option%molDescrType = 'unsorted'
  end if
  
  if (option%permInvKernel) then
    if (.not. (trim(option%Nperm) /= '' .and. option%readX) &
	   .and. .not. ((trim(option%permInvGroups) /= '' .or. trim(option%permInvNuclei) /= '' .or. trim(option%permIndIn) /= '') &
	                .and. trim(option%molDescrType) == 'permuted')) then
	  errorMessage = 'permInvKernel should be requested with Nperm and XfileIn or permInvNuclei and molDescrType=permuted'
    end if
  end if
  
  if ((trim(option%molDescrType) == 'permuted') .and. &
  .not. (trim(option%permInvGroups) /= '' .or. trim(option%permInvNuclei) /= '' .or. trim(option%permIndIn) /= ''))  then
    errorMessage = 'molDescrType=permuted should be used with option permInvNuclei'
  end if

  if (option%selectPerm .and. trim(option%molDescrType) /= 'permuted') then 
    errorMessage = 'selectPerm should be used with option molDescrType=permuted'
  end if  

end subroutine checkOptions

function checkFile(filename)
  implicit none
  ! Argument
  character(len=*),intent(in) :: filename
  ! Return value
  logical                     :: checkFile

  ! Check if the file exists
  inquire(file=trim(filename), exist=checkFile)
end function

end module optionsModule
!==============================================================================
