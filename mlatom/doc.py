#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! doc: handling help of MLatom                                              ! 
  ! Implementations and documentation by:                                     !
  ! Fuchun Ge, Pavlo O. Dral, Bao-Xin Xue                                     ! 
  !---------------------------------------------------------------------------! 
'''
import time, sys

class Doc():
    @classmethod
    def printDoc(cls,argdict):
        mlatom_alias=['mlatomf','mlatom','kreg']
        gap_alias=[ 'gap', 'gap_fit', 'gapfit']
        sgdml_alias=['sgdml']
        deepmd_alias=['dp','dpmd', 'deepmd','deepmd-kit']
        physnet_alias=['physnet']
        mace_alias=['mace']
        ani_alias=['torchani','ani']
        overview=True
        for key, value in argdict.items():
            for k, v in cls.help_text.items():
                if key.lower()==k.lower() and value is True:  
                    print(v)
                    overview=False
            if key.lower() in ['mlmodeltype','mlprog'] and value:
                if value == True: value='S'
                if value.lower() in mlatom_alias:
                    from .interface_MLatomF import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in ani_alias:
                    from .interfaces.torchani_interface import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in gap_alias:
                    from .interfaces.gap_interface import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in sgdml_alias:
                    from .interfaces.sgdml_interface import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in deepmd_alias:
                    from .interfaces.dpmd_interface import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in physnet_alias:
                    from .interfaces.physnet_interface import printHelp
                    printHelp()
                    overview=False
                elif value.lower() in mace_alias:
                    from .interfaces.mace_interface import printHelp
                    printHelp()
                    overview=False
                else:
                    print(cls.Default_MLprog_type)
                    overview=False
        if overview:
            print(cls.help_text['overview'])

        print(' %s ' % ('='*78))
        print(time.strftime(" MLatom terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print(' %s ' % ('='*78))
        sys.stdout.flush()
        
    molDescriptorDoc = '''
  Optional arguments specifying descriptor:
    molDescriptor=S            molecular descriptor S
      RE [default]             vector {Req/R}, where R is internuclear distance
      CM                       Coulomb matrix
    molDescrType=S             type S of molecular descriptor
      sorted                   default for molDescrType=CM
                               sort by:
                                 norms of CM matrix (for molDescrType=CM)
                                 nuclear repulsions (for molDescrType=RE)
      unsorted                 default for molDescrType=RE
      permuted                 molecular descriptor with all atom permutations
  
  If molDescrType=sorted requested, additional output can be requested:
    XYZsortedFileOut=S         file S with sorted XYZ coordinates, only works
                               for molDescriptor=RE molDescrType=sorted
  
  If molDescrType=permuted requested, at least one of the arguments needed:
    permInvGroups=S            permutationally invariant groups S
                               e.g. for water dimer permInvGroups=1,2,3-4,5,6
                               permute water molecules (atoms 1,2,3 and 5,6,7)
    permInvNuclei=S            permutationally invariant nuclei S
                               e.g.permInvNuclei=2-3.5-6
                               will permute atoms 2,3 and 6,7
'''

    Default_MLprog_type='''
  Use of interfaces to ML programs
  
  Arguments:
    MLprog=S                   ML program S
      or
    MLmodelType=S              ML model type S

  Supported ML model types and default programs:

  +-------------+----------------+ 
  | MLmodelType | default MLprog | 
  +-------------+----------------+ 
  | KREG        | MLatomF        | 
  +-------------+----------------+ 
  | sGDML       | sGDML          | 
  +-------------+----------- ----+ 
  | GAP-SOAP    | GAP            | 
  +-------------+----------------+ 
  | PhysNet     | PhysNet        | 
  +-------------+----------------+ 
  | MACE        | MACE           | 
  +-------------+----------------+ 
  | DeepPot-SE  | DeePMD-kit     | 
  +-------------+----------------+ 
  | ANI         | TorchANI       | 
  +-------------+----------------+ 
  
  Supported interfaces with default and tested ML model types:
  
  +------------+----------------------+
  | MLprog     | MLmodelType          |
  +------------+----------------------+
  | MLatomF    | KREG [default]       |
  |            | see                  |
  |            | MLatom.py KRR help   |
  +------------+----------------------+
  | sGDML      | sGDML [default]      |
  |            | GDML                 |
  +------------+----------------------+
  | GAP        | GAP-SOAP             |
  +------------+----------------------+
  | PhysNet    | PhysNet              |
  +------------+----------------------+ 
  | MACE       | MACE                 | 
  +------------+----------------------+
  | DeePMD-kit | DeepPot-SE [default] |
  |            | DPMD                 |
  +------------+----------------------+
  | TorchANI   | ANI [default]        |
  +------------+----------------------+
  
  See interface's help for more details, e.g.
    MLatom.py MLprog=TorchANI help
'''

    help_text={
    'overview':'''
  Usage:
    MLatom.py [options]

  Getting help:
    MLatom.py help             print this help and exit
    MLatom.py [option] help    print help for [option] and exit
                               e.g. MLatom.py useMLmodel help

  Options:
    ML tasks:
      geomopt                  perform geometry optimizations
      freq                     perform frequency calculations 
      TS                       perform transition state searches
      IRC                      perform reaction path searches
      useMLmodel               use existing ML model    
      createMLmodel            create and save ML model 
      estAccMLmodel            estimate accuracy of ML model
      deltaLearn               use delta-learning
      selfCorrect              use self-correcting ML
      learningCurve            generate learning curve
      crossSection             simulate absorption spectrum
      MLTPA                    simulate two-photon absorption
    
    General purpose methods:
      AIQM1                    perform AIQM1       calculations
      ANI-1ccx                 perform ANI-1ccx    calculations
      ANI-1x                   perform ANI-1x      calculations
      ANI-2x                   perform ANI-2x      calculations
      ANI-1x-D4                perform ANI-1x-D4   calculations
      ANI-2x-D4                perform ANI-2x-D4   calculations
      ANI-1xnr                 perform ANI-1xnr    calculations
      AIMNet2@B973c            perform AIMNet2@B97-3c calculations
      AIMNet2@wB97M-D3         perform AIMNet2@wB97M-D3 calculations
      ODM2
      ODM2*
      GFN2-xTB
      CCSD(T)*/CBS             only single-point calculations

    Data set tasks:
      XYZ2X                    convert XYZ coordinates into descriptor X
      analyze                  analyze data sets
      sample                   sample data points from a data set
      slice                    slice data set
      sampleFromSlices         sample from each slice
      mergeSlices              merge indices from slices

    Molecular dynamics:
      MD                       3D molecular dynamics

    Quantum dynamics:
      MLQD                     ML-accelerated quantum dynamics

    Multithreading:
      nthreads=N               set number of threads (N)
 
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat MLmodelOut=CH3Cl.unf
''',
  'AIQM1':'''
  Run AIQM1 calculations
  Similarly, AIQM1@DFT and AIQM1@DFT* can be requested
  
  Usage: MLatom.py AIQM1 [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates

    Other options (at least one of these arguments is required):
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
      geomopt                  perform geometry optimizations
      freq                     perform frequency calculations
      TS                       perform transition state searches
      IRC                      perform reaction path searches
      QMprog=S                 program to calculate QM part of AIQM1
                               (MNDO is default, Sparrow is optional and used when MNDO is not found)
      mndokeywords=S           file S with MNDO keywords, separating by a blank line for each molecules
                               (it is required to set iop=-22 immdp=-1
                               Often, e.g., for geomopt, jop=-2 igeom=1 nsav15=3 are also needed)
      
  For the geomopt and freq options, see
    MLatom.py freq help
      
  Example:
    MLatom.py AIQM1 XYZfile=opt.xyz YestFile=en.dat
''',
  'geomopt':'''
  Perform geometry optimization
  
  Usage: 
         MLatom.py geomopt usemlmodel mlmodelin=... mlmodeltype=... XYZfile=... [arguments]
         or 
         MLatom.py geomopt AIQM1 XYZfile=... [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates
    
    Optional input for choosing interfaced program:
      optProg=scipy            use scipy package [default]
      optProg=gaussian         use Gaussian program
      optProg=ASE              use ASE
      optProg=geometric        use geomeTRIC
      optxyz=S                 save optimized geometries in file S [default: optgeoms.xyz]
    The following options only used for ASE program:
        ase.fmax=R                    threshold of maximum force (in eV/A)
                                      [default values: 0.02]
        ase.steps=N                   maximum steps
                                      [default values: 200]
        ase.optimizer=S               optimizer
           LBFGS [default]
           BFGS
      
  Example:
    MLatom.py AIQM1 geomopt XYZfile=opt.xyz optprog=ASE
''',
  'freq':'''
  Perform geometry optimization followed by frequencies
  
  Usage: 
         MLatom.py freq usemlmodel mlmodelin=... mlmodeltype=... XYZfile=... [arguments]
         or 
         MLatom.py freq [AIQM1 or another general-purpose method] XYZfile=... [arguments]

  Arguments:
    Input file options:
      XYZfile=S                file S with XYZ coordinates
    
    Optional input for choosing interfaced program:
      freqProg=gaussian         use Gaussian program [primary default]
      freqProg=PySCF            use PySCF program    [secondary default]
      freqProg=ASE              use ASE
      when do frequence analysis with ASE, the following options are also required:
      ase.linear=N,...,N            0 for nonlinear molecule, 1 for linear molecule
                                    [default vaules: 0]
      ase.symmetrynumber=N,...,N    rotational symmetry number for each molecule
                                    [default vaules: 1]
      
  Example:
    MLatom.py AIQM1 freq XYZfile=opt.xyz freqProg=ASE
''',
  'useMLmodel':'''
  Use existing ML model
  
  Usage: MLatom.py useMLmodel [arguments]

  Arguments:
    Input file options:
      MLmodelIn=S              file S with ML model
      MLprog=S                 only required for third-party programs. See
                               MLatom.py MLprog help
      
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X

    Output file options (at least one of these arguments is required):
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
      
  Example:
    MLatom.py useMLmodel MLmodelIn=CH3Cl.unf XYZfile=CH3Cl.xyz YestFile=MLen.dat  
''',
  'createMLmodel':'''
  Create and save ML model
  
  Usage: MLatom.py createMLmodel [arguments]

  Required arguments:
    Input files:
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X
      
      Yfile=S                  file S with reference values
         and/or
      YgradXYZfile=S           file S with reference XYZ gradients

    Output files:
      MLmodelOut=S             file S with ML model
    
  Optional output files:
    XfileOut=S                 file S with X values
    XYZsortedFileOut=S         file S with sorted XYZ coordinates
                               only works for
                               molDescrType=RE molDescrType=sorted
    YestFile=S                 file S with estimated Y values
    YgradEstFile=S             file S with estimated gradients
    YgradXYZestFile=S          file S with estimated XYZ gradients
    
  KREG model with unsorted RE descriptor is trained by default
  Default hyperparameters sigma=100.0 and lambda=0.0 are used
    
  To train ML model with third-party program, see:
    MLatom.py MLprog help      to list available interfaces and 
                               default ML models
           and
    MLatom.py MLprog=S help    for specific help of interface S
    
  To train another KRR model with MLatomF, see:
    MLatom.py KRR help

  Any hyperparameters can be optimized with the hyperopt package, see:
    MLatom.py hyperopt help
      
  To optimize hyperparameters, the training set should be split into the
  sub-training and validation sets, see:
    MLatom.py sample help
  Additional optional arguments:
    sampling=user-defined      reads in indices for the sub-training and
                               validation sets from files defined by arguments
      iSubtrainIn=S            file S with indices of sub-training points
      iValidateIn=S            file S with indices of validation points
      iCVoptPrefIn=S           prefix S of files with indices for CVopt
    
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat MLmodelOut=CH3Cl.unf
''',
  "estAccMLmodel":'''
  Estimate accuracy of ML model
  
  Usage: MLatom.py estAccMLmodel [arguments]

  estAccMLmodel task estimates accuracy of created ML models, thus it can
  take the same arguments as createMLmodel task, see:
    MLatom.py createMLmodel help
  with exception that MLmodelOut=S argument is optional
  
  To estimate accuracy, the data set should be split into the
  training and validation sets, see:
    MLatom.py sample help
  Additional optional arguments:
    sampling=user-defined      reads in indices for the training and test sets
                               from files defined by arguments
      iTrainIn=S               file S with indices of training points
      iTestIn=S                file S with indices of test points
      iCVtestPrefIn=S          prefix S of files with indices for CVtest
    MLmodelIn=S                file S with ML model
    
  Example:
    MLatom.py estAccMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat
''',
  'deltaLearn':'''
  Use delta-learning
  
  Usage: MLatom.py deltaLearn [arguments]

  Arguments:
      Yb=S                     file with data obtained with baseline method
      Yt=S                     file with data obtained with target   method
      YestT=S                  file with ML estimations of  target   method
      YestFile=S               file with ML corrections to  baseline method

  delta-learning should be used with one of the following tasks, see:
    MLatom.py useMLmodel    help
    MLatom.py createMLmodel help
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py estAccMLmodel deltaLearn XfileIn=x.dat Yb=UHF.dat \\
    Yt=FCI.dat YestT=D-ML.dat YestFile=corr_ML.dat
''',
  'selfCorrect':'''
  Use self-correcting ML
  
  Usage: MLatom.py selfCorrect [arguments]

  Currently works only with four layers and MLatomF
  Self-correction should be used with one of the following tasks, see:
    MLatom.py useMLmodel    help
    MLatom.py createMLmodel help
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py estAccMLmodel selfcorrect XYZfile=xyz.dat Yfile=y.dat 
''',
  "learningCurve":'''
  Generate learning curve
  
  Usage: MLatom.py learningCurve [arguments]

  Required arguments:
      lcNtrains=N,N,N,...,N    training set sizes
  Optional arguments:
      lcNrepeats=N,N,N,...,N   numbers of repeats for each Ntrain
                 or
                =N             number  of repeats for all  Ntrains
                               [3 repeats default]

  Output files in directory learningCurve:
    results.json               JSON database file with all results
    lcy.csv                    CSV  database file with results for values
    lcygradxyz.csv             CSV  database file with results for XYZ gradients
    lctimetrain.csv            CSV  database file with training   timings
    lctimepredict.csv          CSV  database file with prediction timings
    
  learningCurve task also requires arguments used in estAccMLmodel task, see:
    MLatom.py estAccMLmodel help

  Example:
    MLatom.py learningCurve XYZfile=xyz.dat Yfile=en.dat \\ 
    lcNtrains=100,1000,10000 lcNrepeats=32
''',
  'crossSection':'''
  Simulate absorption cross-section using ML-NEA (nuclear ensemble approach)
      
  Usage:
    MLatom.py crossSection [arguments]

  Optional arguments:
    nExcitations=N             number of excited states [3 by default]
    nQMpoints=N                number of QM calculations
                               [determined iteratively by default]
    plotQCNEA                  plot QC-NEA cross section
    deltaQCNEA=float           set broadening parameter of QC-NEA cross section
    plotQCSPC                  plot single point convolution cross section
  
    Advanced arguments:
      nMaxPoints=N             maximum number of QC calculations
                               in the iterative procedure [10000 by default]
      nNEpoints=N              number of nuclear ensemble points
                               [50000 by default]

  Environment settings:
    $NX                        Newton-X environment
    In addition, Gaussian program environment should be setup
  
  Required input and data set files
    gaussian_optfreq.com       input file for optimization and frequency
                               calculations with Gaussian program
    gaussian_ef.com            template input file for excited-state QC
                               calculations with Gaussian program
        or
    eq.xyz                     file with optimized geometry
    nea_geoms.xyz              file with all geometries in nuclear ensemble
    
  Optional input data files:
    E[i].dat                   files with excitation energies  for excitation i
    f[i].dat                   files with oscillator strengths for excitation i
    cross-section_ref.dat      reference cross section spectrum
  
  Output files in directory cross-section:
    cross-section_ml-nea.dat   ML-NEA cross section
    cross-section_qc-nea.dat   QC-NEA cross section
    cross-section_spc.dat      single point convolution cross section
    plot.png                   cross section plots
''',
  'MLTPA':'''
  Simulate two-photon absorption cross-section with ML (ML-TPA approach)
      
  Usage:
    MLatom.py MLTPA [arguments]

  Input arguments:
    Input file options:
    SMILESfile=S               file S with SMILES
    auxfile=S                  optional file S with 
                               the information of wavelength and Et30 in the format of
                               'wavelength_lowbound,wavelength_upbound,Et30']
                               (wavelength in nm.)
                               One line per SMILES string.
                               Default value of Et30 will be 33.9 (toluene) and
                               the whole spectra between 600-1100 nm will be output.

  After the calculations finish, the predicted TPA cross section values are saved
  in a folder named 'tpa+{absolute time}'.
  The folder will contain files tpa[sequential molecular number].txt with predicted
  TPA cross section values.

  Example:
    MLatom.py MLTPA SMILESfile=Smiles.csv auxfile=_aux.txt
''',
  'XYZ2X':('''
  Convert XYZ coordinates into descriptor X
  
  Usage:
    MLatom.py XYZ2X [arguments]

  Required arguments:
    MLmodelIn=S                file S with ML model
    XYZfile=S                  file S with XYZ coordinates
    XfileOut=S                 file S with X values
''' +
    molDescriptorDoc
    +
'''
  Example:
    MLatom.py XYZ2X XYZfile=CH3Cl.xyz XfileOut=CH3Cl.x
'''),
  'analyze':'''
  Analyze data sets
  
  Usage: MLatom.py analyze [arguments]

  Arguments:
    For reference data (at least one of these arguments is required):
      Yfile=S                  file S with values
      Ygrad=S                  file S with gradients
      YgradXYZfile=S           file S with gradients in XYZ coordinates

    For estimated data:
      YestFile=S               file S with estimated Y values
      YgradEstFile=S           file S with estimated gradients
      YgradXYZestFile=S        file S with estimated XYZ gradients
    
  Example: 
    MLatom.py analyze Yfile=en.dat YestFile=enest.dat
''',
  'sample':'''
  Sample data points from a data set
  
  Usage:
    MLatom.py sample [arguments]

  Required arguments:
    Data set file
      XYZfile=S                file S with XYZ coordinates
          or
      XfileIn=S                file S with input vectors X
    
    Splitting arguments (at least one is required):
      Splitting the data set into the sub-sets:
        iTrainOut=S            file S with indices of training points
        iTestOut=S             file S with indices of test points
        iSubtrainOut=S         file S with indices of sub-training points
        iValidateOut=S         file S with indices of validation points
        
      Cross-validation for testing:
        CVtest                 CV task with optional arguments:
          NcvTestFolds=N       sets number of folds to N [5 by default]
          LOOtest              leave-one-out cross-validation
          iCVtestPrefOut=S     prefix S of files with indices for CVtest
      Cross-validation for hyperparameter optimization:
        CVopt                  CV task with optional arguments:
          NcvOptFolds=N        sets number of folds to N [5 by default]
          LOOopt               leave-one-out cross-validation
          iCVoptPrefOut=S      prefix S of files with indices for CVopt

  Optional arguments:
    sampling=S                 type S of data set sampling into splits
      random [default]         random sampling
      none                     simply split unshuffled data set into
                               the training and test sets (in this order)
                               (and sub-training and validation sets)
      structure-based          structure-based sampling
      farthest-point           farthest-point traversal iterative procedure
    Nuse=N                     N first entries of the data set file to be used
    Ntrain=R                   number of the training points
                               [0.8, i.e. 80% of the data set, by default]
    Ntest=R                    number of the test points
                               [remainder of data set, by default]
    Nsubtrain=R                number of the sub-training points
                               [0.8,    80% of the training set, by default]
    Nvalidate=R                number of the validation points
                               [remainder of the training set, by default]
  Note: Number of indices R can either be positive integer or a fraction of 1.
        R entries for integer R >= 1
        fraction of the entire data set for 0<R<1
  
  Example:
    MLatom.py sample sampling=structure-based XYZfile=CH3Cl.xyz Ntrain=1000 \\
    Ntest=10000 iTrainOut=itrain.dat iTestOut=itest.dat
''',
  'slice':'''
  Slice data set
  
  Usage:
    MLatom.py slice [arguments]
  
  Required arguments:
    XfileIn=S                  file S with input vectors X
    eqXfileIn=S                file S with input vector for equilibrium geometry
  
  Optional arguments:
    Nslices=N                  number of slices [3 by default]
  
  Example:
    MLatom.py slice Nslices=3 XfileIn=x_sorted.dat eqXfileIn=eq.x

''',
  'sampleFromSlices':'''
  Sample from each slice
  
  Usage:
    MLatom.py sampleFromSlices [arguments]
  
  Required argument:
    Ntrain=N                   total integer number N of training points
                               from all slices
  
  Optional argument:
    Nslices=N                  number of slices [3 by default]

  Example:
    MLatom.py sampleFromSlices Nslices=3 sampling=structure-based Ntrain=4480
''',
  'mergeSlices':'''
  Merge indices from slices
  
  Usage:
    MLatom.py mergeSlices [arguments]
  
  Required argument:
    Ntrain=N                   total integer number N of training points
                               from all slices
  
  Optional argument:
    Nslices=N                  number of slices [3 by default]

  Example:
    MLatom.py mlatom mergeSlices Nslices=3 Ntrain=4480
''',
  "KRR":'''
  Kernel ridge regression (KRR) calculations with MLatomF

  Optional arguments:  
    kernel=S                   kernel function type S
      Gaussian [default]
        periodKernel           periodic kernel
        decayKernel            decaying periodic kernel
      Laplacian
      exponential
      Matern
    permInvKernel              permutationally invariant kernel, sub-option:
      Nperm=N                  number of permutations N (with XfileIn argument)
      molDescrType=permuted    (with XYZfile argument, see below)
    lambda=R                   regularization hyperparameter R [0 by default]
      opt                      requests optimization of lambda on a log grid
        NlgLambda=N            N points on a logarithmic grid
                               [6 by default]
        lgLambdaL=R            lowest  value of log2(lambda) [-16.0 by default]
        lgLambdaH=R            highest value of log2(lambda) [ -6.0 by default]
    sigma=S                    length scale
                               [default values: 100 (Gaussian  & Matern)
                                                800 (Laplacian & exponential)]
      opt                      requests optimization of sigma on a log grid
        NlgSigma=N             N points on a logarithmic grid
                               [11 by default]
        lgSigmaL=R             lowest  value of log2(lambda)
                               [default values:  2.0 (Gaussian  & Matern)
                                                 5.0 (Laplacian & exponential)]
        lgSigmaH=R             highest value of log2(lambda)
                               [default values:  9.0 (Gaussian  & Matern)
                                                12.0 (Laplacian & exponential)]
    period                     period in periodic and decayed periodic kernel
                               [1.0 by default]
    sigmap=R                   length scale for a periodic part
                               in decayed periodic kernel
                               [100.0 by default]
    matDecomp=S                type of matrix decomposition
      Cholesky [default]
      LU                   
      Bunch-Kaufman   
    invMatrix                  invert matrix
    refine                     refine solution matrix
    on-the-fly                 on-the-fly calculation of kernel
                               matrix elements for validation,
                               by default it is false and those
                               elements are stored
    benchmark                  additional output for benchmarking
    debug                      additional output for debugging
    
  Additional arguments for hyperparameter optimization on a log grid:
    minimizeError=S            type S of error to minimize
      RMSE                     root-mean-square error [default]
      MAE                      mean absolute error
   lgOptDepth=N                depth of log grid optimization N [3 by default]
''' +
    molDescriptorDoc +
'''
  Example:
    MLatom.py createMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat \\
              MLmodelOut=CH3Cl.unf sigma=opt kernel=Matern
''',
  "hyperopt":'''
  Hyperparameter optimization with the hyperopt package
  For now, only Tree-Structured Parzen Estimator algorithm is supported
  
  Usage: substitute numeric value(s) with hyperopt.xx()
  
  Arguments for hyperopt.xx():
    hyperopt.uniform(lb,ub)    linear search space form
    hyperopt.loguniform(lb,ub) logarithmic search space, base 2
  lb is lower bound, ub is upper bound

  Arguments:
      hyperopt.max_evals=N     max number of search attempts [8 by default]

  Example:
    MLatom.py estAccMLmodel XYZfile=CH3Cl.xyz Yfile=en.dat \\
              sigma=hyperopt.loguniform(4,20)

 Cite hyperopt:
  Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model 
  Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision 
  Architectures. TProc. of the 30th International Conference on Machine 
  Learning (ICML 2013), June 2013, pp. I-115 to I-23.
''',
  'MD':'''
  Perform molecular dynamics

  !!! Warning !!! 
  In MLatom, energy unit is Hartree and distance unit is Angstrom. Make sure that units in your model are consistent!

  Usage:
        MLatom.py MD usemlmodel mlmodelin=... mlmodeltype=... [arguments]
        or
        MLatom.py MD AIQM1 [arguments]

  Arguments:
    Input options:
      dt=R                       time step in fs [0.1 by default]
      trun=R                     maximum runtime in fs [1ps by default]
      initXYZ=S                  initial geometry (should be in Angstrom)
      initVXYZ=S                 initial velocity (should be in Angstrom/fs)
      initConditions=S           algorithm of generating initial conditions
        user-defined [default]     user provide initial condition using 
                                   initXYZ & initVXYZ option 
        random                     generate random velocities; user still 
                                   needs to provide initXYZ; must be used 
                                   together with initTemperature option
      initTemperature=R          initial temperature in Kelvin, necessary when generating 
                                 random initial velocity
                                 [300 by default]
      initVXYZout=S              output file of initial velocity
      initXYZout=S               output file of initial geometry
      Thermostat=S               MD thermostat
        NVE [default]              NVE (microcononical) ensemble
        Andersen                   Andersen NVT thermostat
        Nose-Hoover                Nose-Hoover chain NVT ensemble
      Temperature=R              environment temperature (only valid when using NVT ensemble)
                                 [300 by default]
      Gamma=R                    collision frequency in fs^-1 [0.2 by default] 
                                 *for Andersen thermostat only*
      NHClength=N                Nose-Hoover chain length [3 by default]
                                 *for Nose-Hoover thermostat only*
      Nc=N                       Multiple time step [3 by default]
                                 *for Nose-Hoover thermostat only*
      Nys=N                      number of Yoshida-Suzuki steps used in NHC [7 by default]
                                 only 1,3,5,7 are available
                                 *for Nose-Hoover thermostat only* 
      NHCfreq=R                  Nose-Hoover chain frequency in fs^-1 [0.0625 by default]
                                 
    Output options:
      trajH5MDout=S              trajectory saved in H5MD file format 
                                 [traj.h5 by default]
      trajTextout=S              trajectory saved in plain text file format
                                 [traj by default]
  Example:
    MLatom.py MD AIQM1 initXYZ=H2.xyz initVXYZ=H2.vxyz
''',
  'MLQD':'''
  Perform ML-accelerated quantum dynamics via the interface
  to the MLQD package (https://github.com/Arif-PhyChem/MLQD).
  
  Usage:
        MLatom.py MLQD [arguments]
 
  Arguments:
   These arguments can also be found at http://mlatom.com/manual/#mlqd.

   Input Options:
        QDmodel=[createQDmodel or useQDmodel] (not optional)	        Default option is 'useQDmodel'. It requests MLQD to create or use a QD model.
        QDmodelIn=[user-provided model file]  (not optional)            If QDmodel=useQDmodel, then it passes a name of file with the trained model.
        QDmodelType=[KRR or AIQD or OSTL]	                            Default option is OSTL. It asks MLQD what type of QD model to use.
        systemType=[SB, FMO or any other type] (not optional)	        There is no default option. It tells MLQD what type of system we are studying.
        hyperParam=[True or False]	                                    Default is False. It is case sensitive and it asks MLQD to optimize the hyper parameters of the model.
        patience=[integer non-negative number]	                        Default value is 10	and it defines the patience for early stopping in CNN training [OSTL and AIQD methods].
        OptEpochs=[integer non-negative number]	                        Default value is 100. It defines the number of epochs for optimization of CNN model [OSTL and AIQD methods].
        TrEpochs=[integer non-negative number]	                        Default value is 100. It defines the number of epochs for training of CNN model [OSTL and AIQD methods].
        max_evals=[integer non-negative number]	                        Default value is 100 and it defines the number of maximum evaluations in hyperopt optimization of CNN model [OSTL and AIQD methods].
        XfileIn=[name of X file]	                                    Default is x_data if QDmodel=createQDmodel and prepInput=True. In the case of QDmodel=createQDmodel, it is optional and provides a name for X file. 
                                                                        It saves the X file with this name if prepInput=True, and it provides the Xfile if prepInput=False. However if QDmodel=useQDmodel and QDmodelType=KRR , 
                                                                        then it is not optional as you need to pass a shot-time trajectory as a input.
        YfileIn=[name of Y file]	                                    Default is y_data if QDmodel = createQDmodel and prepInput=True. In the case of QDmodel = createQDmodel, it is optional and it provides a name for Y file. 
                                                                        It saves the Y file with this name if prepInput=True , and it provides the Y file if prepInput=False.
        dataPath=[absolute or relative path of training data]		    In the case of QDmodel=createQDmodel, and prepInput=True, we need to pass datapath, so MLQD can prepare the X and Y files. It should be noted that, 
                                                                        data should be in the same format as our in our database QDDSET-1 especially when QDmodelType=OSTL or AIQD.
        krrSigma=[float, value of sigma hyperparameter]                 Specific to KRR. It defines value for Gaussian kernal in KRR. Default value is 4.0.
        krrLamb=[float, value of lambda hyperparameter]                 Specific to KRR. It defines value for Gaussian kernal in KRR. Default value is 0.00000001
        n_states=[number of states or sites, integer]	                Default is 2 for SB and 7 for FMO. It defines the number of states (SB) or sites (FMO).
        initState=[number of initial site]	                            Default value is 1 (Initial exictation is on site-1). It represents initial site in FMO complex and it is required when we propagate dynamics with OSTL or AIQD method
        time=[propagation time]	                                        Default is 20\Delta for SB and 50ps for FMO	complex.
        time_step=[time step of propagation]	                        Default is 0.05\Delta for SB and 0.005ps for FMO complex.
        energyDiff=[energy difference]	                                Default value is 1.0\Delta. In the case of SB, it defines the energy difference between the states and it is required when QDmodelType=OSTL or AIQD.
        Delta=[tunneling matrix element]	                            Default value is 1.0. It is adopted as a energy unit in SB and defines the tunneling matrix element. It is required only when QDmodelType = OSTL or AIQD.
        gamma=[characteristic frequency]	                            Default value is 10\Delta for SB and 500cm^-1 for FMO. It defines the characteristic frequency and required only when QDmodelType=OSTL or AIQD.
        lamb=[system-bath coupling strength]	                        Default value is 1.0\Delta for SB and 520cm^-1 for FMO. It defines system-bath coupling strength and required only when QDmodelType=OSTL or AIQD.
        temp=[temperature]	                                            Default value is 1.0\Delta for SB and 510cm^-1 for FMO.	It defines temperature (K) in the case FMO complex and inverse temperature in the case of SB. 
                                                                        We need to define it when QDmodelType=OSTL or AIQD.
        energyNorm=[normalizer]	                                        Default value is 1.0\Delta. It is a Normalizer for the energy difference between the states (SB). 
        energyNorm=[normalizer]	                                        Default value is 1.0. It serves as a Normalizer for the tunneling matrix element in the case of SB.
        gammaNorm=[normalizer]	                                        Default value is 10\Delta (in the case of SB) and 500cm^-1 (in the case of FMO). It serves as a Normalizer for characteristic frequency.
        lambNorm=[normalizer]	                                        Default value is 1.0\Delta (in the case of SB) and 520cm^-1 (in the case of FMO). It serves as a Normalizer for system-bath coupling strength.
        tempNorm=[normalizer]	                                        Default value is 1.0\Delta (in the case of SB) and 510cm^-1 (in the case of FMO). It serves as a Normalizer for temperature (FMO) or inverse temperature (SB).
        numLogf=[number of logistic functions]	                        Default value is 1. It defines the number of logistic functions in AIQD which is used to normalize the dimension of time.
        LogCa=[coefficient]	                                            Default value is 1.0. It defines coefficient “a” in the logistic function.
        LogCb=[coefficient]	                                            Default value is 15.0. It defines coefficient “b” in the logistic function.
        LogCc=[coefficient]	                                            Default value is -1.0. It defines coefficient “c” in the logistic function.
        LogCd=[coefficient]	                                            Default value is 1.0.  It defines coefficient “d” in the logistic function.
        dataCol=[column number]	                                        Default value is 1 and it is required when QDmodelType=KRR. It is used to grab the corresponding column from the reference trajectory for plotting. 
                                                                        you need to mention which column to grab.
        dtype=[real or imag]	                                        Default is real. It serves when QDmodelType=KRR and your data is complex. As KRR deals only with real data, we need to mention which part of the complex data 
                                                                        MLQD needs to grab; real or imaginary.
        xlength=[number of time steps in the short seed trajectory]	    Default value is 81 which defines the length of the input short trajectory for KRR. It is the number of time steps in the data you passed with dataCol.
        refTraj		                                                    MLQD has the option to plot the predicted dynamics against the reference trajectory. It is optional. If reference trajectory is provided, MLQD will go for plotting otherwise not.
        xlim=[xaxis limit]	                                            Default option is equal to the propagation time. However, user can define xaxis limit for plotting.
        pltNstates=[number of states to be plotted]	                    Default option is to plot all states. However, User can define how many states should be plotted.
        prepInput=[True or False]	                                    Default is False and both options are case sensitive. It asks MLQD to prepare input files X and Y from the data given that the data is given in the same format as our in our database QDDSET-1.

    Output Options:
        QDmodelOut=[user-defined name of created model] (optional)      You can pass it if QDmodel=createQDmodel and MLQD will save the trained model with this name. However, it's optional, thus if you don’t pass it, MLQD will choose a random name.
        QDtrajOut=[file name for the output trajectory]                 You can pass it if QDmodel=useQDmodel and MLQD will save the predicted dynamics with the provided name. However, it's optional, thus if you don’t pass it, MLQD will choose a random name.
  

    Examples:
        For tutorials, visit http://mlatom.com/tutorial/mlqd/ 
''',

  'MD2vibr':'''
  Infrared spectrum simulation

  Arguments:
    trajH5MDin=S                 file with trajectory in H5MD format
                                 should contain dipole moments
    trajVXYZin=S                 file containing velocities 
    trajdpin=S                   file containing dipole moments 
                                 !!!Note!!! trajH5MDin, trajVXYZin, trajdpin 
                                 cannot be used at the same time
    threshold=R                  print peaks with abosorption larger than R
                                 0.0 < R <= 1.0 [0.1 by default]
    start_time=R                 unit: fs [0.0 by default]
    end_time=R                   unit: fs [maximum time by default]
                                 use trajectory from start_time to end_time 
                                 [use the whole trajectory by default]
    autocorrelationDepth=N       autocorrelation depth; unit: fs [1024 by default]
    zeropadding=N                zero padding; unit: fs [1024 by default]
    title=S                      title of the plot [no title by default]
    output=S
      ir                         output infrared spectrum
      ps                         output power spectrum
                                 !!!Note!!! when this option is not specified, mlatom 
                                 will output infrared spectrum if there are dipole moments 
                                 in H5MD file, otherwise it will output power spectrum

  Examples:
    MLatom.py MD2vibr trajH5MDin=traj.h5
''',

'qmprog':'''
    Specify which QM program to use for calculation 

    Programs available:
    gaussian, pyscf, mndo, sparrow, orca
        
    Examples:
      MLatom.py method=b3lyp/6-31G* qmprog=gaussian xyzfile=sp.xyz yestfile=enest.dat
''',

  'method':'''
    Specify the method to use for calculation

    Methods available:

    QM methods
    ML methods
      ANI1x, ANI-1ccx, ANI-1x, ANI-2x, ANI-1x-D4, ANI-2x-D4
    AIQM methods
      AIQM1, AIQM1@DFT

    Examples:
      MLatom.py method=b3lyp/6-31G* qmprog=gaussian xyzfile=sp.xyz yestfile=enest.dat
      MLatom.py method=AIQM1 qmprog=sparrow xyzfile=sp.xyz yestfile=enest.dat
''',

  'charges':'''
    Specify charge(s) for molecule(s). By default 0.
    If many molecules are given, please use comma to separate the charges, i.e. charge_1, charge_2, ...

    Examples:
      MLatom.py method=b3lyp/6-31G* qmprog=gaussian \\
      xyzfile=sp.xyz yestfile=enest.dat charges=1,1,-1
''',

  'multiplicities':'''
    Specify multiplicitie(s) for molecule(s). By default 1.
    If many molecules are given, please use comma to separate the multiplicities, i.e. multiplicity_1, multiplicity_2, ...
  
    Examples:
      MLatom.py method=b3lyp/6-31G* qmprog=gaussian \\
      xyzfile=sp.xyz yestfile=enest.dat multiplicities=3,3,1
''',

  'QMprogramKeywords':'''
    Specify keyword file to read for QM programs

    keywords-supported programs:

      xTB  : xTB command line keywords, e.g. `-c 0` to define charge for molecule. 
             More details see https://xtb-docs.readthedocs.io/en/latest/commandline.html
      mndo : use keywords in mndo

    Examples:
      MLatom.py method=GFN2-xTB xyzfile=sp.xyz yestfile=enest.dat QMprogramKeywords=xtb_kw
      contents of xtb_kw: `-c 1 -u 2`
'''
    }
