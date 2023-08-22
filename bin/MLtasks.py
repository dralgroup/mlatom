#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! MLtasks: Machine learning tasks of MLatom                                 ! 
  ! Implementations by: Pavlo O. Dral and Fuchun Ge                           ! 
  !---------------------------------------------------------------------------! 
'''

import os, sys, time, shutil, re, copy, json
import numpy as np
from io         import StringIO
from contextlib import redirect_stdout
try:
    from .args_class import ArgsBase
    from . import constants, data, interface_MLatomF, models, simulations, stopper, utils
    from . import NX_MLatom_interface # DEVELOPMENT VERSION
except:
    from args_class import ArgsBase
    import constants, data, interface_MLatomF, models, simulations, stopper, utils
    import NX_MLatom_interface # DEVELOPMENT VERSION

class MLtasksCls(object):
    dataSampled = False
    dataSplited = False
    dataPrepared = False
    subdatasets = []
    default_MLprog={
                'kreg': 'mlatomf',
                'id': 'mlatomf',
                'kid': 'mlatomf',
                'ukid': 'mlatomf',
                'akid': 'mlatomf',
                'pkid': 'mlatomf',
                'krr-cm': 'mlatomf',
                'gap-soap': 'gap',
                'sgdml': 'sgdml',
                'gdml':'sgdml',
                'dpmd': 'deepmd-kit',
                'deeppot-se': 'deepmd-kit',
                'physnet': 'physnet',
                'ani': 'torchani',
                'ani-aev':'torchani',
                'ani1x': 'torchani',
                'ani1ccx': 'torchani',
                'ani2x': 'torchani',
                'ani-tl':'torchani',
                'renn':'renn',
                'nequip':'nequip',
                 'mlqd': 'mlqd',   
            }
    mlatom_alias=['mlatomf','mlatom']
    gap_alias=[ 'gap', 'gap_fit', 'gapfit'] 
    sgdml_alias=['sgdml']
    deepmd_alias=['dp','deepmd','deepmd-kit']
    physnet_alias=['physnet']
    ani_alias=['torchani','ani']
    renn_alias=['renn']
    nequip_alias=['nequip']
    KMs = mlatom_alias+gap_alias+sgdml_alias
    model_needs_subvalid = deepmd_alias+ani_alias+physnet_alias+sgdml_alias+renn_alias
    index_prefix = ''
    def __init__(self, argsMLtasks = sys.argv[1:]):
        global args
        args = Args()
        args.parse(argsMLtasks)

        # Check whether we are dealing with a method MLatom recognizes
        from requests.structures import CaseInsensitiveDict
        method_aliases = CaseInsensitiveDict({'AIQM1DFT': 'AIQM1@DFT', 'AIQM1DFTstar': 'AIQM1@DFT*',
                                              'ani1ccx': 'ANI-1ccx', 'ani1x': 'ANI-1x', 'ani2x': 'ANI-2x',
                                              'ani1xd4': 'ANI-1x-D4', 'ani2xd4': 'ANI-2x-D4',
                                              'ODM2star': 'ODM2*', 'gfn2xtb': 'GFN2-xTB'})
        method = None
        for arg in args.args2pass:
            argname = arg
            if arg in method_aliases.keys():
                argname = method_aliases[arg]
            if models.methods.is_known_method(argname):
                kwargs = {}
                if args.mndokeywords != '': args.QMprogramKeywords = args.mndokeywords
                if args.QMprogramKeywords != '':
                    if 'aiqm'.casefold() in argname.casefold():
                        kwargs['qm_program_kwargs'] = {}
                        kwargs['qm_program_kwargs']['read_keywords_from_file'] = args.QMprogramKeywords
                        if args.mndokeywords != '': kwargs['qm_program_kwargs']['save_files_in_current_directory'] = True
                    else:
                        kwargs['read_keywords_from_file'] = args.QMprogramKeywords
                        if args.mndokeywords != '': kwargs['save_files_in_current_directory'] = True
                if args.qmprog != '':
                    if 'aiqm'.casefold() in argname.casefold():
                        kwargs['qm_program'] = args.qmprog
                    else:
                        kwargs['program'] = args.qmprog
                method = models.methods(method=argname, **kwargs)
                break

        # Read molecular charges and multiplicities
        def get_mol_database_from_xyz():
            db = data.molecular_database()
            if args.data['XYZfile'] == '': stopper.stopMLatom(f'XYZfile argument not found in input')
            if not os.path.exists(args.XYZfile): stopper.stopMLatom(f'File {args.XYZfile} was not found')
            db.read_from_xyz_file(filename = args.XYZfile)
            if args.charges != '':
                charges = [int(xx) for xx in str(args.charges).split(',')]
                for imol in range(len(db.molecules)):
                    db.molecules[imol].charge = charges[imol]
            if args.multiplicities != '':
                multiplicities = [int(xx) for xx in str(args.multiplicities).split(',')]
                for imol in range(len(db.molecules)):
                    db.molecules[imol].multiplicity = multiplicities[imol]
            return db

        if args.deltaLearn:
            self.deltaLearn()
        elif args.selfCorrect:
            self.selfCorrect()
        elif args.learningCurve:
            self.learningCurve()
        elif args.crossSection:
            import ML_NEA 
            CrossSection_args=copy.deepcopy(args.args2pass)
            deadlist=[]
            for arg in CrossSection_args:
                flagmatch = re.search('(^nthreads)|(^hyperopt)|(^setname=)|(^learningcurve$)|(^lcntrains)|(^lcnrepeats)|(^mlmodeltype)|(^mlmodel2type)|(^mlmodel2in)|(^opttrajxyz)|(^opttrajh5)|(^mlprog)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^nlayers=)|(^selfcorrect)|(^initbetas)|(^minimizeError)|(^mlmodelsin)|(^activeLearning)|(^ase.fmax=)|(^ase.steps=)|(^ase.optimizer=)|(^ase.linear=)|(^ase.symmetrynumber=)|(^qmprog=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch:
                    deadlist.append(arg)
            for i in deadlist: CrossSection_args.remove(i)
            # print(CrossSection_args)
            ML_NEA.parse_api(CrossSection_args)
        elif args.callNXinterface:        # DEVELOPMENT VERSION
            CrossSection_args=copy.deepcopy(args.args2pass)
            deadlist=[]
            for arg in CrossSection_args:
                flagmatch = re.search('(^nthreads)|(^callnxinterface)|(^hyperopt)|(^setname=)|(^learningcurve$)|(^lcntrains)|(^lcnrepeats)|(^mlmodeltype)|(^mlmodel2type)|(^opttrajxyz)|(^opttrajh5)|(^mlmodel2in)|(^mlprog)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^nlayers=)|(^selfcorrect)|(^initbetas)|(^minimizeError)|(^mlmodelsin)|(^activeLearning)|(^ase.fmax=)|(^ase.steps=)|(^ase.optimizer=)|(^ase.linear=)|(^ase.symmetrynumber=)|(^qmprog=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch:
                    deadlist.append(arg)
            for i in deadlist: CrossSection_args.remove(i)
            # print(CrossSection_args)
            #NX_MLatom_interface.NX_MLatom_interfaceCls(argsNX = args.args2pass) # DEVELOPMENT VERSION
            NX_MLatom_interface.NX_MLatom_interfaceCls(argsNX = CrossSection_args) # DEVELOPMENT VERSION
        elif args.geomopt or args.freq or args.ts or args.irc:
            if method != None:
                print('')
                db = get_mol_database_from_xyz()
                if args.geomopt or args.ts:
                    fname = "optgeoms.xyz"
                    if args.data['optxyz'] != '': fname = args.optxyz
                    if os.path.exists(fname): stopper.stopMLatom(f'File {fname} already exists; please delete or rename it')
                    db_opt = data.molecular_database()
                    kwargs = {}
                    if args.data['optprog'] != '': kwargs['program'] = args.optprog
                    if args.ts: kwargs['ts'] = True
                    if args.ase.fmax != '': kwargs['convergence_criterion_for_forces'] = float(args.ase.fmax)
                    if args.ase.steps != '': kwargs['maximum_number_of_steps'] = int(args.ase.steps)
                    if args.ase.optimizer != '': kwargs['optimization_algorithm'] = args.ase.optimizer
                    for imol in range(len(db.molecules)):
                        mol = db.molecules[imol] ; mol.number = imol+1
                        geomopt = simulations.optimize_geometry(method=method,
                                                        initial_molecule=mol,
                                                        **kwargs)
                        db_opt.molecules.append(geomopt.optimized_molecule)
                        print(' %s ' % ('='*78))
                        print(' Optimization of molecule %d' % (imol+1))
                        print(' %s \n' % ('='*78))
                        print(f'   {"Iteration":^10s}    {"Energy (Hartree)":^25s}')
                        for step in geomopt.optimization_trajectory.steps:
                            print('   %10d    %25.13f' % (step.step+1, step.molecule.energy))
                        if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar:
                            print('\n\n Final properties of molecule %d\n' % (imol+1))
                            print_aiqm1_results(aiqm1=args.AIQM1, molecule=geomopt.optimized_molecule)
                            print('\n')
                        elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                            print('\n\n Final properties of molecule %d\n' % (imol+1))
                            print_animethod_results(methodname=method.method, molecule=geomopt.optimized_molecule)
                            print('\n')
                        else:
                            print('\n Final energy of molecule %6d: %25.13f Hartree\n\n' % (imol+1, geomopt.optimized_molecule.energy))
                    db_opt.write_file_with_xyz_coordinates(filename=fname)
                elif args.freq:
                    kwargs = {}
                    if args.data['optprog'] != '': kwargs['program'] = args.optprog
                    for imol in range(len(db.molecules)):
                        mol = db.molecules[imol] ; mol.number = imol+1
                        if args.ase.linear != '':
                            linearlist = args.ase.linear.split(',')
                            if linearlist[imol] == '1': mol.shape = 'linear'
                            else: mol.shape='nonlinear'
                        if args.ase.symmetrynumber != '':
                            symmetrynumbers = args.ase.symmetrynumber.split(',')
                            mol.symmetry_number = int(symmetrynumbers[imol])
                        geomopt = simulations.thermochemistry(method=method,
                                                        molecule=mol,
                                                        **kwargs)
                        print(' %s ' % ('='*78))
                        print(' %s Vibration analysis for molecule %6d' % (' '*20, imol+1))
                        print(' %s ' % ('='*78))
                        print(' Multiplicity: %s' % mol.multiplicity)
                        if 'symmetry_number' in mol.__dict__: print(' Rotational symmetry number: %s' % mol.symmetry_number)
                        if 'shape' in mol.__dict__: print(f' This is a {mol.shape.casefold()} molecule')
                        
                        print('   Mode     Frequencies     Reduced masses     Force Constants')
                        print('              (cm^-1)            (AMU)           (mDyne/A)')
                        for i in range(len(mol.frequencies)):
                            print('%6d %15.4f %15.4f %18.4f' % (i+1, mol.frequencies[i], mol.reduced_masses[i], mol.force_constants[i]))
                            
                        print(' %s ' % ('='*78))
                        print(' %s Thermochemistry for molecule %6d' % (' '*20, imol+1))
                        print(' %s ' % ('='*78))
                        fmt = ' %-41s: %15.5f Hartree'
                        if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar:
                            print_aiqm1_results(aiqm1=args.AIQM1, molecule=mol)
                            print('')
                        elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                            print_animethod_results(methodname=method.method, molecule=mol)
                            print('')
                        print(fmt % ('ZPE-exclusive internal energy at      0 K', mol.energy))
                        print(fmt % ('Zero-point vibrational energy', mol.ZPE))
                        print(fmt % ('Internal energy                  at   0 K', mol.U0))
                        print(fmt % ('Enthalpy                         at 298 K', mol.H))
                        print(fmt % ('Gibbs free energy                at 298 K', mol.G))
                        # To-do: add entropy
                        if 'DeltaHf298' in mol.__dict__:
                            print('')
                            fmt = ' %-41s: %15.5f Hartree %15.5f kcal/mol'
                            print(fmt % ('Atomization enthalpy             at   0 K', mol.atomization_energy_0K, mol.atomization_energy_0K * constants.Hartree2kcalpermol))
                            print(fmt % ('ZPE-exclusive atomization energy at   0 K', mol.ZPE_exclusive_atomization_energy_0K, mol.ZPE_exclusive_atomization_energy_0K * constants.Hartree2kcalpermol))
                            print(fmt % ('Heat of formation                at 298 K', mol.DeltaHf298, mol.DeltaHf298 * constants.Hartree2kcalpermol))
                            # To-do: make it work for ANI-1ccx
                            if args.AIQM1:
                                if mol.aiqm1_nn.energy_standard_deviation > 0.41*constants.kcalpermol2Hartree:
                                    print(' * Warning * Heat of formation have high uncertainty!')
                            if args.ani1ccx:
                                if mol.ani1ccx.energy_standard_deviation > 1.68*constants.kcalpermol2Hartree:
                                    print(' * Warning * Heat of formation have high uncertainty!')
                        print('')
                elif args.irc:
                    for imol in range(len(db.molecules)):
                        mol = db.molecules[imol] ; mol.number = imol+1
                        geomopt = simulations.irc(method=method,
                                                  ts_molecule=mol)
                print('')
            else: # DEPRICATED - kept for legacy reasons
                import geomopt
                geomopt.geomoptCls(args.args2pass)
        
        elif args.CCSDTstarCBS: # DEPRICATED - kept for legacy reasons
            import ccsdtstarcbs
            ccsdtstarcbs.calc(args.XYZfile, args.YestFile)
        
        elif args.mlqd:
            MLQDdir = os.environ['MLQD']
            sys.path.append(os.path.dirname(MLQDdir))
            MLQD = __import__(os.path.basename(MLQDdir))
            from MLQD.evolution import quant_dyn
            mlqd_args = {}
            for argg in argsMLtasks:
                if argg.lower() != 'mlqd':
                    splits = argg.split('=', maxsplit=1)
                    if len(splits) == 1:
                        mlqd_args[splits[0]] = 'True'
                    else:
                        yy = splits[1]
                        if yy[0] in '0123456789':
                            if '.' in yy:
                                yy = float(yy)
                            else:
                                yy = int(yy)
                        mlqd_args[splits[0]] = yy
            quant_dyn(**mlqd_args)
        
        elif args.MD: # DEPRICATED - kept for legacy reasons
            import ThreeDMD
            ThreeDMD.MultiThreeDcls.dynamics(args.args2pass)
        
        elif args.IRSS:
            import IRSS
            IRSS.IRSScls.simulate(args.args2pass)
        
        elif args.Gaussian:
            import interface_gaussian 
            interface_gaussian.GaussianCls.calculate(args.args2pass)
        
        elif args.MLTPA:
            import MLTPA
            mltpa=MLTPA.MLTPA(args.args2pass)
            mltpa.predict()
        
        elif args.XYZ2SMI:
            import xyz2smi
            mltpa=xyz2smi.xyz2smi(args.args2pass)
            mltpa.genxyz2smi()
        
        elif args.RMSD or args.AlignXYZ:
            import RMSD
            RMSD.RMSDcls(args.args2pass)
        
        elif args.createMLmodel or args.useMLmodel or args.estAccMLmodel:   
            self.chooseMLop(args.args2pass)
        
        else: # Single-point calculations with a method MLatom recognizes
            gradcalc = False ; hesscalc = False
            if args.YgradXYZestFile != '': gradcalc = True
            if args.hessianestfile != '': hesscalc = True
            
            db = get_mol_database_from_xyz()

            method.predict(molecular_database = db,
                        calculate_energy_gradients = gradcalc,
                        calculate_hessian = hesscalc)
            
            print('')
            if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar or args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                nmol = 0
                for mol in db.molecules:
                    nmol += 1
                    print('\n\n Properties of molecule %d\n' % nmol)
                    if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar:
                        print_aiqm1_results(aiqm1=args.AIQM1, molecule=mol)
                    elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                        print_animethod_results(methodname=method.method, molecule=mol)
            else:
                for imol in range(len(db.molecules)): print(' Energy of molecule %6d: %25.13f Hartree' % (imol+1, db.molecules[imol].energy))
            print('')
            sys.stdout.flush()
            db.write_file_with_properties(filename = args.YestFile, property_to_write = 'energy')
            if gradcalc: db.write_file_with_xyz_derivative_properties(filename = args.YgradXYZestFile, xyz_derivative_property_to_write = 'energy_gradients')
            if hesscalc: db.write_file_with_hessian(filename = args.hessianestfile, hessian_property_to_write = 'hessian')

    @classmethod
    def deltaLearn(cls):
        locargs = args.args2pass
        if args.Yb != '': yb = [float(line) for line in open(args.Yb, 'r')]
        if args.YgradB != '': ygradb = [[float(xx) for xx in line.split()] for line in open(args.YgradB, 'r')]
        if args.YgradXYZb != '': ygradxyzb = utils.readXYZgrads(args.YgradXYZb)
        if args.createMLmodel or args.estAccMLmodel:
            # Delta of Y values
            if args.Yb != '':
                ydatname = '%s-%s.dat' % (utils.fnamewoExt(args.Yt), utils.fnamewoExt(args.Yb))
                locargs = utils.addReplaceArg('Yfile', 'Yfile=%s' % ydatname, locargs)
                yt = [float(line) for line in open(args.Yt, 'r')]
                with open(ydatname, 'w') as fcorr:
                    for ii in range(len(yb)):
                        fcorr.writelines('%25.13f\n' % (yt[ii] - yb[ii]))
            # Delta of gradients
            if args.YgradB != '':
                ygradfname = '%s-%s.dat' % (utils.fnamewoExt(args.YgradT), utils.fnamewoExt(args.YgradB))
                locargs = utils.addReplaceArg('YgradFile', 'YgradFile=%s' % ygradfname, locargs)
                ygradt = [[float(xx) for xx in line.split()] for line in open(args.YgradT, 'r')]
                with open(ygradfname, 'w') as fcorr:
                    for ii in range(len(ygradb)):
                        strtmp = ''
                        for jj in range(len(ygradb[ii])):
                            strtmp += '%25.13f   ' % (ygradt[ii][jj] - ygradb[ii][jj])
                        fcorr.writelines('%s\n' % strtmp)
            # Delta of XYZ gradients
            # 'YgradXYZfile', 'YgradXYZestFile', 'YgradXYZb', 'YgradXYZt', 'YgradXYZestT',
            if args.YgradXYZb != '':
                ygradxyzfname = '%s-%s.dat' % (utils.fnamewoExt(args.YgradXYZt), utils.fnamewoExt(args.YgradXYZb))
                locargs = utils.addReplaceArg('YgradXYZfile', 'YgradXYZfile=%s' % ygradxyzfname, locargs)
                ygradxyzt = utils.readXYZgrads(args.YgradXYZt)
                with open(ygradxyzfname, 'w') as fcorr:
                    for imol in range(len(ygradxyzb)):
                        fcorr.writelines('%d\n\n' % len(ygradxyzb[imol]))
                        for iatom in range(len(ygradxyzb[imol])):
                            strtmp = ''
                            for idim in range(3):
                                strtmp += '%25.13f   ' % (ygradxyzt[imol][iatom][idim] - ygradxyzb[imol][iatom][idim])
                            fcorr.writelines('%s\n' % strtmp)
                        
            cls.chooseMLop(locargs)
        elif args.useMLmodel:
            cls.chooseMLop(locargs)
        
        if utils.argexist('YestFile=', locargs):
            corr = [float(line) for line in open(args.YestFile, 'r')]
            with open(args.YestT, 'w') as fyestt:
                for ii in range(len(yb)):
                    fyestt.writelines('%25.13f\n' % (yb[ii] + corr[ii]))
        if utils.argexist('YgradEstFile=', locargs):
            corr = [[float(xx) for xx in line.split()] for line in open(args.YgradEstFile, 'r')]
            with open(args.YgradEstT, 'w') as fyestt:
                for ii in range(len(ygradb)):
                    strtmp = ''
                    for jj in range(len(ygradb[ii])):
                        strtmp += '%25.13f   ' % (ygradb[ii][jj] + corr[ii][jj])
                    fyestt.writelines('%s\n' % strtmp)
        if utils.argexist('YgradXYZestFile=', locargs):
            corr = utils.readXYZgrads(args.YgradXYZestFile)
            with open(args.YgradXYZestT, 'w') as fyestt:
                for imol in range(len(ygradxyzb)):
                    fyestt.writelines('%d\n\n' % len(ygradxyzb[imol]))
                    for iatom in range(len(ygradxyzb[imol])):
                        strtmp = ''
                        for idim in range(3):
                            strtmp += '%25.13f   ' % (ygradxyzb[imol][iatom][idim] + corr[imol][iatom][idim])
                        fyestt.writelines('%s\n' % strtmp)

    @classmethod
    def selfCorrect(cls):
        locargs = args.args2pass
        yfilename = ''
        if args.createMLmodel or args.estAccMLmodel:
            if args.createMLmodel:
                MLtaskPos = [arg.lower() for arg in locargs].index('createmlmodel')
            else:
                MLtaskPos = [arg.lower() for arg in locargs].index('estaccmlmodel')
            locargslower = [arg.lower() for arg in locargs]
            if ('sampling=structure-based' in locargslower or
                'sampling=farthest-point'  in locargslower or
                'sampling=random'          in locargslower):
                print('\n Running sampling\n')
                sys.stdout.flush()
                interface_MLatomF.ifMLatomCls.run(['sample',
                                 'iTrainOut=itrain.dat',
                                 'iTestOut=itest.dat',
                                 'iSubtrainOut=isubtrain.dat',
                                 'iValidateOut=ivalidate.dat']
                                + locargs[:MLtaskPos] + locargs[MLtaskPos+1:])
            for arg in locargs:
                if 'yfile=' in arg.lower():
                    locargs.remove(arg)
                    yfilename = arg[len('yfile='):]
        for nlayer in range(1,args.nlayers+1):
            print('\n Starting calculations for layer %d\n' % nlayer)
            sys.stdout.flush()
            if args.createMLmodel or args.estAccMLmodel:
                if nlayer == 1:
                    ydatname = yfilename
                else:
                    ydatname = 'deltaRef-%s_layer%d.dat' % (utils.fnamewoExt(yfilename), (nlayer - 1))
                    yrefs    = [float(line) for line in open(yfilename, 'r')]
                    ylayerm1 = [float(line) for line in open('ylayer%d.dat' % (nlayer - 1), 'r')]
                    with open(ydatname, 'w') as fydat:
                        for ii in range(len(ylayerm1)):
                            fydat.writelines('%25.13f\n' % (yrefs[ii] - ylayerm1[ii]))
                for arg in locargs:
                    if ('sampling=structure-based' == arg.lower() or
                        'sampling=farthest-point'  == arg.lower() or
                        'sampling=random'          == arg.lower()):
                        locargs.remove(arg)
                        locargs += ['sampling=user-defined',
                                    'iTrainIn=itrain.dat',
                                    'iTestIn=itest.dat',
                                    'iSubtrainIn=isubtrain.dat',
                                    'iValidateIn=ivalidate.dat']
                if args.createMLmodel:
                    locargs += ['MLmodelOut=mlmodlayer%d.unf' % nlayer]
                locargs = utils.addReplaceArg('Yfile', 'Yfile=%s' % ydatname, locargs)
                cls.chooseMLop(['YestFile=yest%d.dat' % nlayer] + locargs)
            elif args.useMLmodel:
                if nlayer > 1:
                    ylayerm1 = [float(line) for line in open('ylayer%d.dat' % (nlayer - 1), 'r')]
                cls.chooseMLop(['MLmodelIn=mlmodlayer%d.unf' % nlayer, 'YestFile=yest%d.dat' % nlayer] + locargs)
            
            if nlayer == 1:
                shutil.move('yest1.dat', 'ylayer1.dat')
            else:
                yestlayer = [float(line) for line in open('yest%d.dat' % nlayer, 'r')]
                with open('ylayer%d.dat' % nlayer, 'w') as fylayer:
                    for ii in range(len(yestlayer)):
                        fylayer.writelines('%25.13f\n' % (ylayerm1[ii] + yestlayer[ii]))

    @classmethod
    def learningCurve(cls):
        locargs=args.args2pass
        try:
            args.lcNtrains = [int(i) for i in str(args.lcNtrains).split(',')]
        except: 
            stopper.stopMLatom('Please provide lcNtrains')
        if args.lcNrepeats:
            if type(args.lcNrepeats) ==  str:
                args.lcNrepeats = [int(i) for i in args.lcNrepeats.split(',')]
            else: 
                args.lcNrepeats = [args.lcNrepeats]*len(args.lcNtrains)
        else:
            args.lcNrepeats = [3]*len(args.lcNtrains)

        surfix = '_'
        surfix+=args.mlmodeltype
        if args.xyzfile: surfix+='_en'
        if args.ygradxyzfile and not args.MLmodelIn: surfix+='_grad'

        dirname = 'learningCurve'
        if os.path.isdir(dirname): pass
        else: os.mkdir(dirname)
        os.chdir(dirname)
        print('\n entered %s\n' % dirname)
        cls.index_prefix = '../'
            
        if os.path.isdir(args.MLprog+surfix): pass
        else: os.mkdir(args.MLprog+surfix)
        if os.path.isfile(args.MLprog+surfix+'/results.json'): 
            with open(args.MLprog+surfix+'/results.json','r') as f:
                results = json.load(f)
            print(results,'\nprevious results ')
        else: results = {}

        for i, ntrain in enumerate(args.lcNtrains):
            if not os.path.isdir('Ntrain_'+str(ntrain)): os.mkdir('Ntrain_'+str(ntrain))
            
            # enter Ntrain dir
            os.chdir('Ntrain_'+str(ntrain))
            print('\n\n testing for Ntrain = %d\n ==============================================================================' % ntrain)
            sys.stdout.flush()
            if args.MLmodelIn:
                with open('../'+args.MLprog+surfix+'/results.json','r') as f:
                    old_results = json.load(f)
            for j in range(args.lcNrepeats[i]):
                if not os.path.isdir(str(j)): os.mkdir(str(j))

                # enter repeat dir
                os.chdir(str(j))

                cls.dataSampled = True
                for i in ['Train','Subtrain','Validate','Test']:
                    if not os.path.isfile('i'+i.lower()+'.dat'):
                        cls.dataSampled = False

                cls.dataSplited = True
                for i in ['Train','Subtrain','Validate','Test']:
                    if not os.path.isfile('xyz.dat_'+i.lower()):##
                        cls.dataSplited = False
                
                if os.path.isdir(args.MLprog+surfix): pass
                else: os.mkdir(args.MLprog+surfix)
                # enter model dir
                os.chdir(args.MLprog+surfix)
                print('\n\n repeat %d for Ntrain_%d\n ------------------------------------------------------------------------------' % (j,ntrain))
                
                sys.stdout.flush()
                if os.path.isfile('result.json') and not args.MLmodelIn:
                    with open('result.json') as f:
                        result = json.load(f)
                    print('this repeat is already done')
                
                elif os.path.isfile('.running'):
                    print(' bypass this repeat since it\'s running already...')
                    os.chdir('../..')
                    continue

                else:
                    try:
                        os.system('touch .running')
                        # set args for learning curve
                        lcargs = utils.addReplaceArg('Ntrain','Ntrain='+str(ntrain),locargs)
                        lcargs = utils.addReplaceArg('estAccMLmodel','estAccMLmodel',lcargs)
                        args.Ntrain = ntrain
                        if args.XfileIn:
                            lcargs = utils.addReplaceArg('XfileIn','XfileIn='+os.path.relpath(args.absXfileIn),lcargs)
                        else:
                            lcargs = utils.addReplaceArg('XYZfile','XYZfile='+os.path.relpath(args.absXYZfile),lcargs)
                        if args.Yfile: lcargs = utils.addReplaceArg('Yfile','Yfile='+os.path.relpath(args.absYfile),lcargs)
                        if args.YgradXYZfile: lcargs = utils.addReplaceArg('YgradXYZfile','YgradXYZfile='+os.path.relpath(args.absYgradXYZfile),lcargs)
                        if args.MLprog.lower() in cls.mlatom_alias:
                            os.system('cp ../../../../eq.xyz .> /dev/null 2>&1')
                            lcargs = utils.addReplaceArg('sampling','sampling=user-defined',lcargs)
                            lcargs = utils.addReplaceArg('itrainin','itrainin=../itrain.dat',lcargs)
                            lcargs = utils.addReplaceArg('itestin','itestin=../itest.dat',lcargs)
                            lcargs = utils.addReplaceArg('isubtrainin','isubtrainin=../isubtrain.dat',lcargs)
                            lcargs = utils.addReplaceArg('ivalidatein','ivalidatein=../ivalidate.dat',lcargs)
                            lcargs = utils.addReplaceArg('benchmark','benchmark',lcargs)

                        # estAccMLmodel
                        cls.dataPrepared = False
                        cls.prepareData(lcargs)
                        result = cls.estAccMLmodel(lcargs, shutup=True)
                    finally:
                        os.system('rm .running')

                if os.path.isfile('../../../'+args.MLprog+surfix+'/results.json'): 
                    with open('../../../'+args.MLprog+surfix+'/results.json','r') as f:
                        results = json.load(f)
                if str(ntrain) not in results.keys(): results[str(ntrain)] = {}
                if args.MLmodelIn:
                    result['t_train'] = old_results[str(ntrain)]['t_train'][j]
                if j == 0: 
                    for k, v in result.items():
                        results[str(ntrain)][k] = [v]
                else:
                    for k, v in result.items():
                        results[str(ntrain)][k].append(v)
                with open('result.json','w') as f:
                    json.dump(result, f, sort_keys=False, indent=4)
                with open('../../../'+args.MLprog+surfix+'/results.json','w') as f:
                    json.dump(results, f, sort_keys=False, indent=4)
                
                meanResults = {i:{j: np.mean(results[i][j]) for j in results[i]} for i in results}
                stdevs = {i:{j: np.std(results[i][j],ddof=1) for j in results[i]} for i in results}

                if args.Yfile:
                    with open('../../../'+args.MLprog+surfix+'/lcy.csv', 'w') as f:
                        f.write('Ntrain, meanRMSE, SD, Nrepeats, RMSEs\n')
                        for nTrain in results.keys():
                            f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,meanResults[nTrain]['eRMSE'],stdevs[nTrain]['eRMSE'],len(results[nTrain]['eRMSE']),','.join([str(i) for i in results[nTrain]['eRMSE']])))
                if args.YgradXYZfile:
                    with open('../../../'+args.MLprog+surfix+'/lcygradxyz.csv', 'w') as f:
                        f.write('Ntrain, meanRMSE, SD, Nrepeats, RMSEs\n')
                        for nTrain in results.keys():
                            try:
                                f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,meanResults[nTrain]['fRMSE'],stdevs[nTrain]['fRMSE'],len(results[nTrain]['fRMSE']),','.join([str(i) for i in results[nTrain]['fRMSE']])))
                            except:
                                pass
                with open('../../../'+args.MLprog+surfix+'/lctimetrain.csv','w') as ft, open ('../../../'+args.MLprog+surfix+'/lctimepredict.csv','w') as fp:
                    ft.write('Ntrain, meanTime, SD, Nrepeats, times\n')
                    fp.write('Ntrain, meanTime, SD, Nrepeats, times\n')
                    for nTrain in results.keys():
                        ft.write('%s, %f, %f, %d, "%s"\n' % (nTrain,meanResults[nTrain]['t_train'],stdevs[nTrain]['t_train'],len(results[nTrain]['t_train']),','.join(["%.2f"%i for i in results[nTrain]['t_train']])))
                        fp.write('%s, %f, %f, %d, "%s"\n' % (nTrain,meanResults[nTrain]['t_pred'],stdevs[nTrain]['t_pred'],len(results[nTrain]['t_pred']),','.join(["%.2f"%i for i in results[nTrain]['t_pred']])))

                os.chdir('../..')
            os.chdir('..')
        os.chdir('..')
    
    
        

    @classmethod
    def chooseMLop(cls, locargs):
        if   args.useMLmodel:
            MLtasksCls.useMLmodel(locargs)
        elif args.createMLmodel:
            MLtasksCls.createMLmodel(locargs)
        elif args.estAccMLmodel:
            MLtasksCls.estAccMLmodel(locargs)
        
    @classmethod
    def createMLmodel(cls, locargs, shutup=False):
        if args.MLprog.lower() in cls.mlatom_alias and not args.optlines and not args.optBetas:
            locargs=cls.getBetas(locargs)
            interface_MLatomF.ifMLatomCls.run(locargs)
        else:
            cls.prepareData(locargs)

            # training
            locargs=cls.getBetas(locargs)
            if args.optBetas:
                from scipy.optimize import least_squares
                optparam=[]
                if args.betas:
                    optparam=[float(i) for i in args.betas.split(',')]                    
                zeroidx=[i for i in range(len(optparam)) if optparam[i]==0]
                optparam=[i for i in optparam if i!=0]
                nbetas=len(optparam)
                optLambda=False
                optSigma=False
                try:
                    if args.data['lambda'] =='opt':
                        optparam.append(-10)
                        optLambda=True
                except:
                    pass
                try:
                    if args.data['sigma'] =='opt':
                        optparam.append(6.643856189774724)
                        optSigma=True
                except:
                    pass

                niter=0
                def getLoss(param):
                    nonlocal locargs, nbetas, optLambda, optSigma, niter, zeroidx
                    betas=['%.6g'% abs(i) for i in param[:nbetas]]
                    for i in zeroidx:
                        betas.insert(i, '0')
                    niter += 1
                    print(' iteration # %d\n ------------------------------------------------------------------------------\n betas=%s' % (niter,','.join(betas)))
                    # return [-2.0]*100
                    argsopt=utils.addReplaceArg('betas','betas=%s'%','.join(betas),locargs)
                    if optLambda:
                        argsopt=utils.addReplaceArg('lambda','lambda=%s'% 2**param[nbetas],argsopt)
                        print(' lambda=%s'% 2**param[nbetas])
                        if optSigma:
                            argsopt=utils.addReplaceArg('sigma','sigma=%s'% 2**param[nbetas+1],argsopt)
                            print(' sigma=%s'% 2**param[nbetas+1])
                    elif optSigma:
                        argsopt=utils.addReplaceArg('sigma','sigma=%s'% 2**param[nbetas],argsopt)
                        print(' sigma=%s'% 2**param[nbetas])
                    if args.optlines:
                        cls.optraining(argsopt,shutup=shutup)
                    else:
                        _, rmsedic, _ =cls.training(argsopt)
                    if args.CVopt:
                        with open(cls.index_prefix+'y.dat_validate_cvopt') as f:
                            ref=[]
                            for line in f:
                                ref.append(float(line))
                            ref=np.array(ref)
                        with open('yest.dat_validate_cvopt') as f:
                            est=[]
                            for line in f:
                                est.append(float(line))
                            est=np.array(est)
                        res=est-ref
                        _, _, results, _ = interface_MLatomF.ifMLatomCls.run(argsopt)
                        print(' test RMSE: %f' % results['eRMSE'])
                    else:
                        with open(args.absYfile) as f:
                            ref=[]
                            for line in f:
                                ref.append(float(line))
                            ref=np.array(ref)
                        with open('yest.dat_validate_optBetas') as f:
                            est=[]
                            for line in f:
                                est.append(float(line))
                            est=np.array(est)
                        res=est-ref
                        with open('ivalidate.dat_optBetas') as f:
                            ivalidate=[]
                            for line in f:
                                ivalidate.append(int(line)-1)
                        with open('itrain.dat') as f:
                            itrain=[]
                            for line in f:
                                itrain.append(int(line)-1)
                        print(' training RMSE: %f' % np.sqrt(np.mean(res[itrain]**2)))  
                        with open('itest.dat') as f:
                            itest=[]
                            for line in f:
                                itest.append(int(line)-1)
                        print(' test RMSE: %f' % np.sqrt(np.mean(res[itest]**2)))
                        res=res[ivalidate]

                    print('Residues in validation:\n',res,'\n validation RMSE: %f'% np.sqrt(np.mean(res**2)))
                    return res
                starttime = time.time()
                out=least_squares(getLoss,np.array(optparam),method='lm',diff_step=0.05,verbose=1,max_nfev=512)
                endtime = time.time()
                t_train = endtime - starttime
                print(' optimization finished. \n', out.x)
                betas = [str(np.abs(i)) for i in out.x[:nbetas]]
                for i in zeroidx:
                    betas.insert(i, '0')
                locargs=utils.addReplaceArg('betas','betas=%s'%','.join(betas),locargs)
                if optLambda:
                    locargs=utils.addReplaceArg('lambda','lambda=%s'%  2**out.x[nbetas],locargs)
                    if optSigma:
                        locargs=utils.addReplaceArg('sigma','sigma=%s'% 2**out.x[nbetas+1],locargs)
                elif optSigma:
                    locargs=utils.addReplaceArg('sigma','sigma=%s'% 2**out.x[nbetas],locargs)

            if args.MLprog.lower() in cls.mlatom_alias and not args.optlines:
                pass
            else:
                if args.optlines:
                    t_train, locargs = cls.optraining(locargs,shutup=shutup)
                    
                else:
                    t_train, _, _ = cls.training(locargs)

                if args.MLprog.lower() not in cls.mlatom_alias:
                    cls.estimate(locargs, 'train', shutup=shutup)
            
            
            locargs=utils.addReplaceArg('CVopt','',locargs)

            return [t_train, locargs]

    @classmethod
    def estAccMLmodel(cls, locargs, shutup=False):
        if args.MLprog.lower() in cls.mlatom_alias and not args.optlines and not args.optBetas:
            locargs=cls.getBetas(locargs)
            t_train, t_pred, results, _ = interface_MLatomF.ifMLatomCls.run(locargs)
        elif args.CVtest:
            t_train, t_pred, results = cls.CVtest(locargs,shutup=shutup)
        else:
            cls.prepareData(locargs)

            if args.mlmodelin:
                t_train = None
                print('Use existing model: '+args.mlmodelin)
            else:
                t_train, locargs = cls.createMLmodel(locargs, shutup=shutup)

            if args.optBetas:                
                t_last, t_pred, results, _ = interface_MLatomF.ifMLatomCls.run(locargs)
                t_train+=t_last
            else:
                t_pred, results, _ = cls.estimate(locargs, 'test', shutup=shutup)

        results['t_train'] = t_train
        results['t_pred'] = t_pred
        return results

    @classmethod
    def CVtest(cls, locargs, shutup=False):
        os.system('rm yest_CVtest.dat gradest_CVtest.dat > /dev/null 2>&1')
        def getCVidx(name):
            allidx=np.arange(args.Ntotal)+1
            testidx=np.loadtxt(name+'.dat').astype(int)
            availableidx=np.setdiff1d(allidx,testidx)
            np.random.shuffle(availableidx)
            np.savetxt(name+'.dat_train',availableidx,fmt='%d')
            
            Ntrain=len(availableidx)
            if args.Nsubtrain:
                if type(args.Nsubtrain)==int:
                    Nsubtrain=args.Nsubtrain
                if type(args.Nsubtrain)==float:
                    Nsubtrain=int(Ntrain*args.Nsubtrain)
                if not args.Nvalidate:
                    Nvalidate=Ntrain-Nsubtrain
            if args.Nvalidate:
                if type(args.Nvalidate)==int:
                    Nvalidate=args.Nvalidate
                if type(args.Nvalidate)==float:
                    Nvalidate=int(Ntrain*args.Nvalidate)
                if not args.Nsubtrain:
                    Nsubtrain=Ntrain-Nvalidate
            if not args.Nsubtrain and not args.Nvalidate:
                Nsubtrain=int(Ntrain*0.8)
                Nvalidate=Ntrain-Nsubtrain
            np.savetxt(name+'.dat_subtrain',availableidx[:Nsubtrain],fmt='%d')
            np.savetxt(name+'.dat_validate',availableidx[-Nvalidate:],fmt='%d')
      
        if not args.iCVtestPrefIn: cls.sample([])
        args.CVtest=False
        t_train, t_pred=0,0
        for cvid in range(args.NcvTestFolds):
            print(' CVtest #%d'% (cvid+1))
            print(' ___________________________________________________________')
            getCVidx(args.iCVtestPrefIn+str(cvid+1))
            cvargs=utils.addReplaceArg('sampling','sampling=user-defined',locargs)
            cvargs=utils.addReplaceArg('itestin','itestin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat',cvargs)
            cvargs=utils.addReplaceArg('itrainin','itrainin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_train',cvargs)
            cvargs=utils.addReplaceArg('isubtrainin','isubtrainin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_subtrain',cvargs)
            cvargs=utils.addReplaceArg('ivalidatein','ivalidatein=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_validate',cvargs)
            args.iTestIn=args.iCVtestPrefIn+str(cvid+1)+'.dat'
            args.iTrainIn=args.iCVtestPrefIn+str(cvid+1)+'.dat_train'
            args.iSubtrainIn=args.iCVtestPrefIn+str(cvid+1)+'.dat_subtrain'
            args.iValidateIn=args.iCVtestPrefIn+str(cvid+1)+'.dat_validate'
            results=cls.estAccMLmodel(cvargs,shutup=True)
            t_train+=results['t_train']
            t_pred+=results['t_pred']
            cls.dataSampled=False
            cls.dataSplited=False
            cls.dataPrepared=False
            if args.Yfile: 
                os.system('cat yest.dat_test >> yest.dat_CVtest')
                os.system('cat y.dat_test >> y.dat_CVtest')
            if args.YgradXYZfile: 
                os.system('cat gradest.dat_test >> gradest.dat_CVtest')
                os.system('cat grad.dat_test >> grad.dat_CVtest')
            os.system('rm -rf %s latest.pt runs > /dev/null 2>&1' % args.MLmodelOut )
        errordict, analresults=cls.analyze('CVtest', shutup=shutup)
        if args.Yestfile: os.system(f'cp yest.dat_CVtest {args.Yestfile}')
        if args.YgradXYZestfile: os.system(f'cp gradest.dat_CVtest {args.YgradXYZestfile}')
        print('\n combined CVtest results:')
        print(' ___________________________________________________________')
        print(analresults)
        return t_train,t_pred,errordict
        
    @classmethod
    def useMLmodel(cls, locargs, shutup=False):
        starttime = time.time()
        errordict = {}
        if args.MLprog.lower() in cls.mlatom_alias:
            _, wallclock, errordict, _ = interface_MLatomF.ifMLatomCls.run(locargs,shutup=shutup)
        else:
            cls.prepareData(locargs)
            if args.MLprog.lower() in cls.deepmd_alias:   
                from interfaces.DeePMDkit import interface_DeePMDkit
                wallclock = interface_DeePMDkit.DeePMDCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.gap_alias:
                from interfaces.GAP       import interface_GAP
                wallclock = interface_GAP.GAPCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.ani_alias:
                from interfaces.TorchANI  import interface_TorchANI
                wallclock = interface_TorchANI.ANICls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.physnet_alias:
                from interfaces.PhysNet   import interface_PhysNet
                wallclock = interface_PhysNet.PhysNetCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.sgdml_alias:
                from interfaces.sGDML     import interface_sGDML
                wallclock = interface_sGDML.sGDMLCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.nequip_alias:
                from interfaces.Nequip    import interface_Nequip
                wallclock = interface_Nequip.NequipCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.renn_alias:
                import RENN
                wallclock = RENN.RENNCls.useMLmodel(locargs, cls.subdatasets)
        endtime = time.time()
        wallclock = endtime - starttime
        if args.benchmark:
            print('\tPrediction time: \t\t\t%f s\n' % wallclock)
        sys.stdout.flush()
        return [wallclock, errordict, '']
            
    @classmethod
    def training(cls, locargs, setname='',cvoptid=''):
        if cvoptid: cvopt= '_'+args.iCVoptPrefIn+str(cvoptid)
        else: cvopt = ''
        if args.MLprog.lower() in cls.mlatom_alias:
            mlatom_arg=utils.addReplaceArg('estaccmlmodel','createmlmodel',locargs)
            mlatom_arg=utils.addReplaceArg('MLmodelOut','MLmodelOut=%s'%args.MLmodelOut,mlatom_arg)
            if args.XfileIn:
                mlatom_arg=utils.addReplaceArg('XfileIn','XfileIn=%s'% os.path.relpath(args.absXfileIn) ,mlatom_arg)
            else:
                mlatom_arg=utils.addReplaceArg('xyzfile','xyzfile=%s'% os.path.relpath(args.absXYZfile) ,mlatom_arg)
            if args.Yfile:
                mlatom_arg=utils.addReplaceArg('yfile','yfile=%s'% os.path.relpath(args.absYfile),mlatom_arg)
            if args.YgradXYZfile: 
                mlatom_arg=utils.addReplaceArg('ygradxyzfile','ygradxyzfile=%s'% os.path.relpath(args.absYgradXYZfile),mlatom_arg)
            if args.optBetas and not args.CVopt:
                setname = 'subtrain'
                os.system('rm yest.dat_validate_optBetas ivalidate.dat_optBetas > /dev/null 2>&1')
                mlatom_arg=utils.addReplaceArg('YestFile','YestFile=yest.dat_validate_optBetas',mlatom_arg)
                os.system('cp %s ivalidate.dat_optBetas'% (cls.index_prefix+args.iValidateIn))

            idx_dict={
                'train': args.itrainin,
                'subtrain': args.isubtrainin,
                'validate': args.ivalidatein,
                'test': args.itestin,
                '': args.itrainin
            }
            if cvoptid:
                idx_dict['subtrain'] = 'isubtrain.dat_cvopt'+str(cvoptid)
                # idx_dict['subtrain'] = args.itrainin
                idx_dict['validate'] = args.iCVoptPrefIn+str(cvoptid)+'.dat'

            with open(cls.index_prefix+idx_dict[setname],'r')as f:
                ntrain=0
                for line in f:
                    ntrain+=1
            mlatom_arg=utils.addReplaceArg('Ntrain','Ntrain=%s'% ntrain,mlatom_arg)
            mlatom_arg=utils.addReplaceArg('Ntest','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('Nvalidate','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('Nsubtrain','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('sampling','sampling=user-defined',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('itrainin','itrainin='+cls.index_prefix+idx_dict[setname],mlatom_arg)
            mlatom_arg=utils.addReplaceArg('itestin','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('ivalidatein','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('isubtrainin','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('benchmark','benchmark',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('CVopt','',mlatom_arg)
            mlatom_arg=utils.addReplaceArg('iCVoptPrefIn','',mlatom_arg)
            os.system('rm '+ args.mlmodelout+' > /dev/null 2>&1 ')
            print(' Training using MLatomF\n ............................................\n')
            wallclock, _, rmsedic, _ = interface_MLatomF.ifMLatomCls.run(mlatom_arg,shutup=False)
            print(' \n .............................................\n \n\n\tTraining Time: \t\t\t\t%f s\n' % wallclock)
        else:
            print(' Training started\n .............................................')
            starttime = time.time()

            trargs = utils.addReplaceArg('setname', 'setname='+setname+cvopt, locargs)
            if setname:
                if args.XfileIn:
                    trargs = utils.addReplaceArg('XfileIn', 'XfileIn='+cls.index_prefix+'x.dat_'+setname+cvopt, trargs)
                else:
                    trargs = utils.addReplaceArg('XYZFile', 'XYZFile='+cls.index_prefix+'xyz.dat_'+setname+cvopt, trargs)
                if args.Yfile or True:    trargs = utils.addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname+cvopt, trargs)
                if args.YgradXYZfile or True: trargs = utils.addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname+cvopt, trargs)  
            if args.MLprog.lower() in cls.deepmd_alias:
                from interfaces.DeePMDkit import interface_DeePMDkit
                wallclock = interface_DeePMDkit.DeePMDCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.gap_alias:
                from interfaces.GAP       import interface_GAP
                wallclock = interface_GAP.GAPCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.ani_alias:
                from interfaces.TorchANI  import interface_TorchANI
                wallclock = interface_TorchANI.ANICls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.physnet_alias:
                from interfaces.PhysNet   import interface_PhysNet
                wallclock = interface_PhysNet.PhysNetCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.sgdml_alias:
                from interfaces.sGDML     import interface_sGDML
                wallclock = interface_sGDML.sGDMLCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.nequip_alias:
                from interfaces.Nequip    import interface_Nequip
                wallclock = interface_Nequip.NequipCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.renn_alias:
                import RENN
                wallclock = RENN.RENNCls.createMLmodel(trargs, cls.subdatasets)
            endtime = time.time()
            wallclock = endtime - starttime
            print(' .............................................\n Training finished\n\n\tTraining Time: \t\t\t\t%f s\n' % wallclock)
            sys.stdout.flush()
            rmsedic=None
        return [wallclock, rmsedic, '']



    @classmethod
    def optraining(cls, locargs, shutup=False, final=True):
        if os.path.isdir('.tmp'): os.system('rm -rf .tmp > /dev/null 2>&1')
        os.mkdir('.tmp')
        #prepare args for optimization
        optargs = utils.addReplaceArg('setname', 'setname=subtrain', locargs)
        optlines=args.optlines
        optplaces = re.findall('hyperopt\..+?\(.+?\)',optlines) # find all places of hyperparas

        # prepare spaces for each hyperparas
        optspace = {}
        for i, place in enumerate(optplaces):
            optlines = optlines.replace(place, 'hypara'+str(i), 1) # mark position of hyperpara
            
            spacetype = place.split('(')[0].split('.')[-1] # get type of space
            lb = float(place.split('(')[1][:-1].split(',')[0]) # lower boundary
            hb = float(place.split('(')[1][:-1].split(',')[1]) # higher boundary
            # set space for each type
            if spacetype == 'loguniform':
                optspace['hypara'+str(i)] = hyperopt.hp.loguniform('hypara'+str(i), np.log(2)*lb, np.log(2)*hb)
            elif spacetype == 'uniform':
                optspace['hypara'+str(i)] = hyperopt.hp.uniform('hypara'+str(i), lb, hb)
            elif spacetype == 'quniform':
                q = int(place.split('(')[1][:-1].split(',')[-1])
                optspace['hypara'+str(i)] = hyperopt.hp.quniform('hypara'+str(i), int(lb), int(hb), q)
            # see https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions

        # replace hyperparas with marked position slots
        for line in optlines.split():
            optargs =  utils.addReplaceArg(line.split('=')[0], line, optargs) 
        
        bestsofar = np.inf # store best loss
        bestreport = 'nothing happended yet' # store analyze report of best run so far

        if args.MLprog.lower()=='mlatomf' and args.minimizeError.lower()=='median':
            for i in ['Validate']:
                if args.XfileIn:
                    exec("cls.splitDataset(1, cls.index_prefix+args.i"+i+"In, args.absXfileIn, cls.index_prefix+'x.dat_'+i.lower())")
                else:
                    exec("cls.splitDataset(0, cls.index_prefix+args.i"+i+"In, args.absXYZfile, cls.index_prefix+'xyz.dat_'+i.lower())")  
                if args.Yfile:
                    exec("cls.splitDataset(1, cls.index_prefix+args.i"+i+"In, args.absYfile, cls.index_prefix+'y.dat_'+i.lower())")
                if args.YgradXYZfile:
                    exec("cls.splitDataset(0, cls.index_prefix+args.i"+i+"In, args.absYgradXYZfile, cls.index_prefix+'grad.dat_'+i.lower())")
        ##############################################################################################
        def getloss(opt): # function that to plug in hyperopt.fmin later
            nonlocal optargs, bestsofar, bestreport, optlines
            optlines_tmp=copy.deepcopy(optlines)
            # get values of hyperparas
            for k, v in opt.items(): 
                optlines_tmp = optlines_tmp.replace(k, str(v), 1)
            # repalce slots with actual values
            for line in optlines_tmp.split():
                optargs =  utils.addReplaceArg(line.split('=')[0], line, optargs) 

            print(' ------------------------------------------------------------------------------\n with hyperparameter(s):\n\n%s\n' % optlines_tmp)
            sys.stdout.flush()
            # train with subtrain set, estimate & analyze with validate set
            if args.MLprog.lower() == 'MLatomF'.lower():
                optargs=utils.addReplaceArg('yestfile','',optargs)
                optargs=utils.addReplaceArg('ygradXYZestfile','',optargs)
            
            if args.CVopt:
                os.system('rm yest.dat_validate_cvopt yest.dat_validate_%s* gradest.dat_validate_* > /dev/null 2>&1'% args.iCVoptPrefIn)
                for i in range(args.NcvOptFolds):
                    if args.MLprog.lower() in cls.mlatom_alias :
                        cls.training(optargs,'subtrain',i+1)
                        _, errordict, report = cls.estimate(optargs, 'validate', i+1,shutup=True)
                    else:
                        optargs=utils.addReplaceArg('MLmodelOut','MLmodelOut='+args.MLmodelOut+'_'+args.iCVoptPrefIn+str(i+1),optargs)
                        cls.training(optargs,'subtrain', i+1)
                        cls.estimate(optargs, 'validate',i+1, shutup=True)
                    if args.Yfile:
                        os.system('cat yest.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> yest.dat_validate_cvopt')
                    if args.YgradXYZfile:
                        os.system('cat gradest.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> gradest.dat_validate_cvopt')
                # errordict, report = cls.analyze('validate_cvopt', shutup=shutup)
                errordict=utils.getError(yfile='y.dat_validate_cvopt',yestfile='yest.dat_validate_cvopt',ygradxyzfile=('grad.dat_validate_cvopt' if args.YgradXYZfile else None),ygradxyzestfile=('gradest.dat_validate_cvopt' if args.YgradXYZfile else None),errorType=args.minimizeError)
                rmses = list(errordict.values())
                optargs=utils.addReplaceArg('MLmodelOut','MLmodelOut='+args.MLmodelOut,optargs)
                # os.system('rm yest.dat_validate_* gradest.dat_validate_* > /dev/null 2>&1')
            else:   
                if args.minimizeError.lower()=='median':
                    optargs=utils.addReplaceArg('minimizeError','minimizeError=RMSE',optargs)
                cls.training(optargs,'subtrain')
                _, errordict_validate, report = cls.estimate(optargs, 'validate', shutup=True)
                if args.minimizeError.lower()=='median':
                    surfix='_validate'
                    if args.MLprog.lower()=='mlatomf':
                        useargs=['useMLmodel',f'MLmodelIn={args.MLmodelOut}',f'xyzfile=xyz.dat_validate',f'yestfile=yest.dat_validate','ygradxyzestfile=gradest.dat_validate']
                        interface_MLatomF.ifMLatomCls.run(useargs,shutup=True)
                    errordict=utils.getError(yfile='y.dat'+surfix,yestfile='yest.dat'+surfix,ygradxyzfile=('grad.dat'+surfix if args.YgradXYZfile else None),ygradxyzestfile=('gradest.da'+surfix if args.YgradXYZfile else None),errorType=args.minimizeError)
                # print(errordict_validate)
                # _, errordict_subtrain, report = cls.estimate(optargs, 'subtrain', shutup=True)
                valid_rmses=np.array(list(errordict_validate.values()))
                # train_rmses=np.array(list(errordict_subtrain.values()))
                # rmses = valid_rmses+1*(valid_rmses-train_rmses)
                rmses=valid_rmses
            if len(rmses)>1:
                if args.hyperopt.losstype.lower() == 'geomean': loss = np.sqrt(np.prod(rmses))
                elif args.hyperopt.losstype.lower() == 'weighted': 
                    loss = args.hyperopt.w_y*rmses[0] + args.hyperopt.w_ygrad*rmses[1]
                else: 
                    print('invalid loss type, using geomean instead...')
                    loss = np.prod(rmses)
            else: loss = rmses[0]

            if args.MLprog.lower() == 'MLatomF'.lower():
                optargs=utils.addReplaceArg('yestfile','YestFile='+args.YestFile,optargs)
                optargs=utils.addReplaceArg('ygradXYZestfile','YgradXYZestFile='+args.YgradXYZestFile,optargs)
            # check if this run performs best
            if loss < bestsofar:
                bestsofar = loss
                # bestreport = report
                os.system('for file in `ls %s* `;do mv $file .tmp/${file}; done > /dev/null 2>&1' % args.mlmodelout)
                os.system('for file in `ls *est.dat_validate*`;do mv $file .tmp/${file}; done > /dev/null 2>&1')# move mlmodel to tmp
            print('\tLoss:\t\t\t\t\t%s\n\n' % loss)
            return loss
        ##############################################################################################
        starttime = time.time()
        print('\n optimizing hyperparameter(s)...\n')
        # optimization process for different algorithm...
        initial_guesses=[]
        if args.hyperopt.points_to_evaluate:
            point={}
            point_v=""
            for c in args.hyperopt.points_to_evaluate:
                if c == '[':
                    point={}
                    point_v=""
                elif c == ']':
                    assert len(optspace)==len(point_v.split(","))
                    for i in range(len(optspace)):
                        point['hypara'+str(i)]=json.loads(point_v.split(",")[i])
                    initial_guesses.append(point)
                else: point_v+=c

            # print(initial_guesses)
        if args.hyperopt.algorithm.lower() == 'tpe':
            bestpara=hyperopt.fmin(fn=getloss, space=optspace, algo=hyperopt.tpe.suggest, max_evals=args.hyperopt.max_evals, points_to_evaluate=initial_guesses, show_progressbar=False)
        # elif args.hyperopt.algorithm.lower() == 'random':
        #     bestpara=hyperopt.fmin(fn=getloss, space=optspace, algo=hyperopt.random.suggest, max_evals=args.hyperopt.max_evals)
        else: print('under construction...')

        endtime = time.time()
        wallclock = endtime - starttime

        # get and print optimized hyperparas
        for k, v in bestpara.items():
            optlines = optlines.replace(k, str(v), 1)
        print('\n ==============================================================================\n optimizing finished\n\n Optimal hyperparameter(s):\n\n%s\n ==============================================================================\n' % optlines)
        sys.stdout.flush()
        # move best model back
        os.system('for file in `ls .tmp`; do mv .tmp/$file .; done > /dev/null 2>&1')

        for line in optlines.split():
            optargs =  utils.addReplaceArg(line.split('=')[0], line, optargs) 
        # if KM then train train set with optargs 
        if args.MLprog.lower() in cls.KMs and not args.optBetas:            
            optargs=utils.addReplaceArg('CVopt','',optargs)
            print("Training whole training set with optimized hyperparameter(s)...")
            optargs = utils.addReplaceArg('setname', 'setname=train', optargs)
            cls.training(optargs,'train')
        # get optargs with optimized hyperparas
        with open('hyperopt.inp','w') as f:
            for i in optargs:
                f.write(i+'\n')

        return wallclock, optargs
        
    @classmethod
    def estimate(cls, locargs, setname, cvoptid='', shutup=False):
        if args.MLprog.lower() in cls.mlatom_alias and args.optlines:

            setdict={
                'train':['train','train','train'],
                'subtrain':['subtrain','subtrain','subtrain'],
                'validate':['subtrain','validate','subtrain'],
                'test':['train','test','train']
            }

            idx_dict={
                'train': args.itrainin,
                'subtrain': args.isubtrainin,
                'validate': args.ivalidatein,
                'test': args.itestin
            }
            if cvoptid:
                idx_dict['subtrain'] = 'isubtrain.dat_cvopt'+str(cvoptid)
                # idx_dict['subtrain'] = args.itrainin
                idx_dict['validate'] = args.iCVoptPrefIn+str(cvoptid)+'.dat'

            with open(cls.index_prefix+idx_dict[setdict[setname][0]],'r')as f:
                ntrain=0
                for line in f:
                    ntrain+=1
                # if setname=='train': ntrain=0
                # MLatomF still check this number...
            with open(cls.index_prefix+idx_dict[setname],'r')as f:
                ntest=0
                for line in f:
                    ntest+=1
            
            if cvoptid:
                useargs=utils.addReplaceArg('estAccMLmodel','',locargs)
                useargs=utils.addReplaceArg('createmlmodel','useMLmodel',useargs)
                useargs=utils.addReplaceArg('itrainin','',useargs)
                useargs=utils.addReplaceArg('itestin','',useargs)
                useargs=utils.addReplaceArg('isubtrainin','',useargs)
                useargs=utils.addReplaceArg('ivalidatein','',useargs)
                if args.XfileIn:
                    useargs=utils.addReplaceArg('xfilein',f'xfilein={cls.index_prefix}x.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                else:
                    useargs=utils.addReplaceArg('xyzfile',f'xyzfile={cls.index_prefix}xyz.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                useargs=utils.addReplaceArg('mlmodelin','mlmodelin=%s'%args.mlmodelout,useargs)
                if args.Yfile:
                    useargs=utils.addReplaceArg('yfile',f'yfile={cls.index_prefix}y.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                    useargs=utils.addReplaceArg('yestfile','yestfile=yest.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                if args.YgradXYZfile:
                    useargs=utils.addReplaceArg('ygradxyzfile',f'ygradxyzfile={cls.index_prefix}grad.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                    useargs=utils.addReplaceArg('ygradxyzestfile','ygradxyzestfile=gradest.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                setname+='_'+args.iCVoptPrefIn+str(cvoptid)
            else:
                useargs=utils.addReplaceArg('estAccMLmodel','',locargs)
                useargs=utils.addReplaceArg('createmlmodel','estAccMLmodel',useargs)
                useargs=utils.addReplaceArg('Ntrain','Ntrain=%s'% ntrain ,useargs)
                useargs=utils.addReplaceArg('Ntest','Ntest=%s'%ntest,useargs)
                useargs=utils.addReplaceArg('sampling','sampling=user-defined',useargs)
                useargs=utils.addReplaceArg('itrainin','itrainin='+cls.index_prefix+idx_dict[setdict[setname][2]],useargs)
                useargs=utils.addReplaceArg('itestin','itestin='+cls.index_prefix+idx_dict[setname],useargs)
                useargs=utils.addReplaceArg('mlmodelin','mlmodelin=%s'%args.mlmodelout,useargs)
                useargs=utils.addReplaceArg('yestfile','',useargs)
                useargs=utils.addReplaceArg('CVopt','',useargs)
                useargs=utils.addReplaceArg('mlmodelout','',useargs)
                useargs=utils.addReplaceArg('iCVoptPrefIn','',useargs)
                useargs=utils.addReplaceArg('ygradXYZestfile','',useargs)
        else:
            if cvoptid: setname+='_'+args.iCVoptPrefIn+str(cvoptid)
            useargs = utils.addReplaceArg('setname', 'setname='+setname, locargs)
            if setname:
                if args.XfileIn:
                    useargs = utils.addReplaceArg('XfileIn', 'XfileIn='+cls.index_prefix+'x.dat_'+setname, useargs)
                else:
                    useargs = utils.addReplaceArg('XYZFile', 'XYZFile='+cls.index_prefix+'xyz.dat_'+setname, useargs)
                if args.Yfile:    useargs = utils.addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname, useargs)
                if args.YgradXYZfile: useargs = utils.addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname, useargs)

        print(' %s:\n\t' % setname)
        t_pred, errordict, report= cls.useMLmodel(useargs, shutup=shutup)
        if args.MLprog.lower() != 'MLatomF'.lower() or cvoptid:
            errordict, report = cls.analyze(setname, shutup=shutup)
        if not shutup: print(report)
        else: print('\t'+'\n\t'.join([k+':\t\t\t\t\t'+str(v) for k, v in errordict.items()])+'\n')
        return [t_pred, errordict, report]

    @classmethod
    def analyze(cls, setname, shutup=False):
        analyzeargs = ['analyze']
        if setname:
            if args.Yfile:
                analyzeargs = utils.addReplaceArg('Yfile', 'Yfile='+cls.index_prefix+'y.dat_'+setname, analyzeargs)
                analyzeargs = utils.addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname, analyzeargs)
            if args.YgradXYZfile:
                analyzeargs = utils.addReplaceArg('YgradXYZfile', 'YgradXYZfile='+cls.index_prefix+'grad.dat_'+setname, analyzeargs)
                analyzeargs = utils.addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname, analyzeargs)

        analout = StringIO() # variable with disgusting name
        with  redirect_stdout(analout):
            interface_MLatomF.ifMLatomCls.run(analyzeargs)
        analresults = analout.getvalue()
        rmses=[float(x.split()[2]) for x in re.findall('RMSE =.+',analresults)]
        errordict = {}
        if args.Yfile:
            errordict['eRMSE'] = rmses[0]
        if args.YgradXYZfile:
            errordict['fRMSE'] = rmses[-1]

        return [errordict, analresults]

    @classmethod
    def prepareData(cls, locargs):
        if cls.dataPrepared:
            return
        if args.MLprog.lower() in cls.mlatom_alias and not args.CVopt:
            cls.dataSplited = True
            # cls.dataConverted = True
        # store subtest needed
        cls.subdatasets = ['Train']
        if args.estAccMLmodel or args.learningCurve: cls.subdatasets.append('Test')
        if args.MLprog in cls.model_needs_subvalid or (args.optlines and not args.CVopt) or args.optBetas: 
            cls.subdatasets.append('Subtrain')
            cls.subdatasets.append('Validate')
        if args.useMLmodel: cls.subdatasets = []
        if args.learningCurve: cls.subdatasets = ['Train','Subtrain','Validate','Test']
        if args.CVopt: 
            try:
                cls.subdatasets.remove('Subtrain')
                cls.subdatasets.remove('Validate')
            except:
                pass
        # check if sampling is needed
        
        ldic={
            'Train':os.path.isfile(args.iTrainIn),
            'Subtrain':os.path.isfile(args.iSubtrainIn),
            'Validate':os.path.isfile(args.iValidateIn),
            'Test':os.path.isfile(args.iTestIn), 
            'nosample': False
        }
        if cls.subdatasets: exec('nosample='+' and '.join([i for i in cls.subdatasets]),globals(),ldic)
        nosample = ldic['nosample']
        if args.CVopt:
            try: 
                if not os.path.isfile(cls.index_prefix+args.iCVoptPrefIn+'1.dat'):
                    nosample = False
                    args.parse_input_content(['iCVoptPrefIn=%s' % args.iCVoptPrefOut])
            except:
                pass
        if not nosample and cls.subdatasets and not cls.dataSampled :
            cls.sample(cls.subdatasets)
            # modify args.iXxxIn
            for i in cls.subdatasets:
                exec('args.i'+i+'In="i'+i.lower()+'.dat"')
            cls.dataSampled = True
        if args.CVopt:
            for i in range(args.NcvOptFolds):
                os.system(f'rm {cls.index_prefix}isubtrain.dat_cvopt{i+1} > /dev/null 2>&1')
                for j in range(args.NcvOptFolds):
                    if j != i:
                        os.system(f'cat {cls.index_prefix+args.iCVoptPrefIn}{j+1}.dat >> {cls.index_prefix}isubtrain.dat_cvopt{i+1}')
        # split datasets
        if not cls.dataSplited:
            if args.learningCurve:
                for i in cls.subdatasets:
                    exec('args.i'+i+'In="i'+i.lower()+'.dat"')

            if cls.subdatasets: print('\n Spliting data...\n')
            sys.stdout.flush()
            for i in cls.subdatasets:
                if args.XfileIn:
                    exec("cls.splitDataset(1, cls.index_prefix+args.i"+i+"In, args.absXfileIn, cls.index_prefix+'x.dat_'+i.lower())")
                else:
                    exec("cls.splitDataset(0, cls.index_prefix+args.i"+i+"In, args.absXYZfile, cls.index_prefix+'xyz.dat_'+i.lower())")  
                if args.Yfile:
                    exec("cls.splitDataset(1, cls.index_prefix+args.i"+i+"In, args.absYfile, cls.index_prefix+'y.dat_'+i.lower())")
                if args.YgradXYZfile:
                    exec("cls.splitDataset(0, cls.index_prefix+args.i"+i+"In, args.absYgradXYZfile, cls.index_prefix+'grad.dat_'+i.lower())")
            if args.CVopt:
                os.system('rm x.dat_validate_cvopt xyz.dat_validate_cvopt y.dat_validate_cvopt grad.dat_validate_cvopt > /dev/null 2>&1')
                
                for i in range(args.NcvOptFolds):
                    cls.subdatasets.append('subtrain_'+args.iCVoptPrefIn+str(i+1))
                    cls.subdatasets.append('validate_'+args.iCVoptPrefIn+str(i+1))
                    if args.xfilein:
                        cls.splitDataset(1, cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat', args.absXfileIn,cls.index_prefix+'x.dat_validate_'+args.iCVoptPrefIn+str(i+1))
                        cls.splitDataset(1, cls.index_prefix+args.itrainIn, args.absXfileIn,cls.index_prefix+'x.dat_subtrain_'+args.iCVoptPrefIn+str(i+1),cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat')
                        os.system('cat '+cls.index_prefix+'x.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> '+cls.index_prefix+'x.dat_validate_cvopt')
                    else:
                        cls.splitDataset(0, cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat', args.absXYZfile,cls.index_prefix+'xyz.dat_validate_'+args.iCVoptPrefIn+str(i+1))
                        cls.splitDataset(0, cls.index_prefix+args.itrainIn, args.absXYZfile,cls.index_prefix+'xyz.dat_subtrain_'+args.iCVoptPrefIn+str(i+1),cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat')
                        os.system('cat '+cls.index_prefix+'xyz.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> '+cls.index_prefix+'xyz.dat_validate_cvopt')
                    if args.Yfile:
                        cls.splitDataset(1, cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat', args.absYfile,cls.index_prefix+'y.dat_validate_'+args.iCVoptPrefIn+str(i+1))
                        cls.splitDataset(1, cls.index_prefix+args.itrainIn, args.absYfile,cls.index_prefix+'y.dat_subtrain_'+args.iCVoptPrefIn+str(i+1),cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat')
                        os.system('cat '+cls.index_prefix+'y.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> '+cls.index_prefix+'y.dat_validate_cvopt')
                    if args.YgradXYZfile:
                        cls.splitDataset(0,cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat', args.absYgradXYZfile,cls.index_prefix+'grad.dat_validate_'+args.iCVoptPrefIn+str(i+1))
                        cls.splitDataset(0,cls.index_prefix+args.itrainIn, args.absYgradXYZfile,cls.index_prefix+'grad.dat_subtrain_'+args.iCVoptPrefIn+str(i+1),cls.index_prefix+args.iCVoptPrefIn+str(i+1)+'.dat')
                        os.system('cat '+cls.index_prefix+'grad.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> '+cls.index_prefix+'grad.dat_validate_cvopt')
            cls.dataSplited = True
        cls.dataPrepared = True

    @classmethod
    def splitDataset(cls, issizeone, indices, filein, fileout, inverse=''):
        # if os.path.isfile(fileout):
        #     if input(f'{fileout} exists, use it?    y/n\n').lower() in ['y','yes','yeah','ya','da']:
        #         return
        with open(indices,'r') as f:
            idx = [int(line) for line in f]
        idxn=[]
        if inverse:
            with open(inverse,'r') as f:
                idxn = [int(line) for line in f]

        with open(filein,'r') as fi, open(fileout,'w') as fo:
            if issizeone:
                for i, line in enumerate(fi):
                    if i+1 in idx and i+1 not in idxn:
                        fo.writelines(line)
            else:
                i = 0
                for line in fi:
                    i = i+1
                    limit=int(line)
                    fi.readline()
                    if i in idx and i not in idxn: fo.write(line+'\n')
                    for j in range(limit):
                            if i in idx and i not in idxn: fo.write(fi.readline())
                            else: fi.readline()
    @classmethod
    def sample(cls, subdatasets):
        print('\n Running sampling...\n')
        sys.stdout.flush()
        smplargs   = ['sample']
        smplargs   = utils.addReplaceArg('molDescriptor','molDescriptor=CM',smplargs)
        pwd=os.getcwd()
        if args.XfileIn:
            smplargs   = utils.addReplaceArg('XfileIn', 'XfileIn=%s' % os.path.relpath(args.absXfileIn), smplargs)
        else:
            smplargs   = utils.addReplaceArg('XYZfile', 'XYZfile=%s' % os.path.relpath(args.absXYZfile), smplargs)
        if args.sampling: smplargs   = utils.addReplaceArg('sampling', 'sampling=%s' % args.sampling, smplargs)
        ldic={'smplargs':smplargs}
        for i in subdatasets:
            exec('if args.N'+i.lower()+': smplargs=utils.addReplaceArg("N'+i.lower()+'","N'+i.lower()+'=%s" % args.N'+i.lower()+', smplargs)',globals(),ldic)
        # for i in ['Train','Test','Subtrain','Validate']:
            exec('smplargs = utils.addReplaceArg("i'+i.lower()+'Out","i'+i.lower()+'Out='+cls.index_prefix+'i'+i.lower()+'.dat",smplargs)',globals(),ldic)
        smplargs=ldic['smplargs']
        if args.CVopt:
            if not args.learningCurve:
                if args.iTrainIn:
                    smplargs = utils.addReplaceArg('itrainout','iTrainIn=%s'%args.iTrainIn,smplargs)
                if args.iTestIn:
                    smplargs = utils.addReplaceArg('itestout','iTestIn=%s'%args.iTestIn,smplargs)
            smplargs = utils.addReplaceArg('CVopt','CVopt',smplargs)
            smplargs = utils.addReplaceArg('iCVoptPrefOut','iCVoptPrefOut=%s'%(cls.index_prefix+args.iCVoptPrefOut),smplargs)
            smplargs = utils.addReplaceArg('NcvOptFolds','NcvOptFolds=%d'%args.NcvOptFolds,smplargs)
            args.parse_input_content(['iCVoptPrefIn=%s'%args.iCVoptPrefOut])
        if args.CVtest:
            smplargs = utils.addReplaceArg('CVtest','CVtest',smplargs)
            smplargs = utils.addReplaceArg('iCVtestPrefOut','iCVtestPrefOut=%s'%(cls.index_prefix+args.iCVtestPrefOut),smplargs)
            smplargs = utils.addReplaceArg('NcvTestFolds','NcvTestFolds=%d'%args.NcvTestFolds,smplargs)
            args.parse_input_content(['iCVtestPrefIn=%s'%args.iCVtestPrefOut])
        interface_MLatomF.ifMLatomCls.run(smplargs, shutup=True)
        for i in ['Train','Test','Subtrain','Validate']:
            exec('if not args.i'+i+'in: args.i'+i+'in="i'+i.lower()+'.dat"',globals(),ldic)

    @classmethod
    def getBetas(cls,locargs):
        if args.kernel.lower() in ['agaussian'] and not args.customBetas:
            if args.XfileIn:
                with open(args.absXfileIn) as f:
                    nbetas =  len(f.readline().split())
            if args.XYZfile:
                with open(args.absXYZfile) as f:
                    natom = int(f.readline())
                    nbetas = natom*(natom-1)//2
            betas=[]
            if args.initBetas.lower()[:2]=='rr':
                coefarg=['estaccmlmodel', "sampling=user-defined",'kernel=linear', 'lambda=opt', 'prior=mean', 'minimizeError=R2']
                if args.XfileIn:
                    coefarg=utils.addReplaceArg('XfileIn','XfileIn=%s'%args.absXfileIn,coefarg)
                if args.XYZfile:
                    coefarg=utils.addReplaceArg('XYZfile','XYZfile=%s'%args.absXYZfile,coefarg)
                if args.Yfile:
                    coefarg=utils.addReplaceArg('Yfile','Yfile=%s'%args.absYfile,coefarg)

                if args.learningCurve:
                    args.iTrainIn='itrain.dat'

                coefarg=utils.addReplaceArg('iTrainIn','iTrainIn=%s'%cls.index_prefix+args.iTrainIn,coefarg)
                coefarg=utils.addReplaceArg('iSubtrainIn','iSubtrainIn=%s'%cls.index_prefix+args.iSubtrainIn,coefarg)
                coefarg=utils.addReplaceArg('iValidateIn','iValidateIn=%s'%cls.index_prefix+args.iValidateIn,coefarg)
                coefarg=utils.addReplaceArg('iTestIn','iTestIn=%s'%cls.index_prefix+args.iTestIn,coefarg)
                coefarg=utils.addReplaceArg('Ntrain','Ntrain=%s'%args.Ntrain,coefarg)
                coefarg=utils.addReplaceArg('Nsubtrain','Nsubtrain=%s'%args.Nsubtrain,coefarg)
                coefarg=utils.addReplaceArg('Nvalidate','Nvalidate=%s'%args.Nvalidate,coefarg)
                coefarg=utils.addReplaceArg('Ntest','Ntest=%s'%args.Ntest,coefarg)
                _, _, _, coefmsg = interface_MLatomF.ifMLatomCls.run(coefarg)
                coefmsg=coefmsg.split('\n')

                if len(args.initBetas.lower())>2:
                    order=float(args.initBetas.lower()[2:])
                else:
                    order=1
                for i in range(coefmsg.index(' Multiple linear regression coefficients:'),coefmsg.index(' Multiple linear regression coefficients:')+nbetas):
                    betas.append(float(coefmsg[i+1])**order)
            elif args.initBetas.lower()=='ones':
                betas = [1]*nbetas
            elif args.initBetas.lower()=='re':
                eqgeo=[]
                with open('eq.xyz') as f:
                    f.readline()
                    f.readline()
                    for i in range(natom):
                        eqgeo.append(f.readline().split()[-3:])
                eqgeo=np.array(eqgeo).astype(float)
                for i in range(natom-1):
                    for j in range(i+1,natom):
                        betas.append(np.linalg.norm(eqgeo[i]-eqgeo[j]))
            locargs=utils.addReplaceArg('betas','betas=%s'%','.join(["%.6g"% i for i in betas]),locargs)
            args.betas=','.join([str(i)for i in betas])
        if args.reduceBetas:
            betas=[float(i) for i in args.betas.split(',')]
            nbetas = len(betas)
            print(" Number of betas used will be reduced...")
            _, _, errordict, _ = interface_MLatomF.ifMLatomCls.run(locargs)
            loss=np.mean(list(errordict.values()))
            print(" With all betas used, loss: %f" % loss)
            for i in range(nbetas):
                bestloss=np.inf
                bestidx=None
                for j in range(nbetas):
                    if betas[j] != 0:
                        locargs=utils.addReplaceArg('betas','betas=%s'%','.join([str(betas[i]) if i!=j else '0' for i in range(nbetas)]),locargs)
                        _, _, errordict, _ = interface_MLatomF.ifMLatomCls.run(locargs)
                        if np.mean(list(errordict.values())) < bestloss:
                            bestloss = np.mean(list(errordict.values()))
                            bestidx = j
                if bestloss < loss * 1.05:
                    betas[bestidx]=0
                else:
                    break
            locargs=utils.addReplaceArg('betas','betas=%s'%','.join([str(i)for i in betas]),locargs)
            args.betas=','.join([str(i)for i in betas])
        return locargs

def print_aiqm1_results(aiqm1=True, molecule=None):
    fmt = ' %-41s: %15.8f Hartree'
    print(fmt % ('Standard deviation of NN contribution', molecule.aiqm1_nn.energy_standard_deviation), end='')
    print(' %15.5f kcal/mol' % (molecule.aiqm1_nn.energy_standard_deviation * constants.Hartree2kcalpermol))
    print(fmt % ('NN contribution', molecule.aiqm1_nn.energy))
    print(fmt % ('Sum of atomic self energies', molecule.aiqm1_atomic_energy_shift.energy))
    print(fmt % ('ODM2* contribution', molecule.odm2star.energy), file=sys.stdout)
    if aiqm1: print(fmt % ('D4 contribution', molecule.d4_wb97x.energy))
    print(fmt % ('Total energy', molecule.energy))
    
def print_animethod_results(methodname=None, molecule=None):
    if 'D4'.casefold() in methodname.casefold():
        modelname = methodname.lower().replace('-','')
        modelname = f'{modelname}_nn'
    else:
        modelname = methodname.lower().replace('-','')
    fmt = ' %-41s: %15.8f Hartree'
    print(fmt % ('Standard deviation of NNs', molecule.__dict__[modelname].energy_standard_deviation), end='')
    print(' %15.5f kcal/mol' % (molecule.__dict__[modelname].energy_standard_deviation * constants.Hartree2kcalpermol))
    
    if 'D4'.casefold() in methodname.casefold():
        print(fmt % ('NN energy', molecule.__dict__[modelname].energy))
        print(fmt % ('D4 correction', molecule.d4_wb97x.energy))
        print(fmt % ('Total energy', molecule.energy))
    else:
        print(fmt % ('Energy', molecule.energy))

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.task_list = [
            'XYZ2X', 'analyze', 'sample', 'slicedata', 'sampleFromSlices', 'mergeSlices',
            'useMLmodel', 'createMLmodel', 'estAccMLmodel', 'learningCurve', 
            'crossSection', # Interfaces
            'deltaLearn', 'selfCorrect',
            'MD','IRSS','Gaussian',
            'reactionPath','mlqd',
            'callNXinterface', # DEVELOPMENT VERSION
            'AIQM1', 'AIQM1DFT', 'AIQM1DFTstar',
            'ODM2', 'ODM2star', 'CCSDTstarCBS', 'gfn2xtb',
            'geomopt', 'freq', 'ts', 'irc', 'molecularDynamics', 'CVopt','CVtest',
            'ani1x', 'ani2x', 'ani1ccx', 'ani1xd4', 'ani2xd4',
            'benchmark',
            'optBetas',
            'MLTPA',
            'XYZ2SMI',
            'RMSD', 'AlignXYZ',
        ]
        self.add_default_dict_args(self.task_list, bool)
        self.add_default_dict_args([
            'MLprog'
            ],
            'MLatomF'
        )
        self.add_default_dict_args([
            'XYZfile', 'XfileIn',
            'Yfile', 'YestFile', 'Yb', 'Yt', 'YestT',
            'YgradFile', 'YgradEstFile', 'YgradB', 'YgradT', 'YgradEstT',
            'YgradXYZfile', 'YgradXYZestFile', 'YgradXYZb', 'YgradXYZt', 'YgradXYZestT',
            'optxyz', 'optprog',
            'hessianestfile', # rename
            'absXfileIn', 'absXYZfile', 'absYfile', 'absYgradXYZfile',
            'trajsList',
            'Nuse', 'Ntrain', 'Nsubtrain', 'Nvalidate', 'Ntest', 'iTrainIn', 'iTestIn', 'iSubtrainIn', 'iValidateIn', 'sampling', 'MLmodelIn', 'MLmodelOut',
            'molDescriptor', 'kernel', 'qmprog',
            'mndokeywords', 'QMprogramKeywords', 'charges', 'multiplicities',
            'iCVtestPrefIn',
            'iCVoptPrefIn',
            'betas',
            'customBetas',
            'reduceBetas'
            ],
            ""
        )       
        self.add_dict_args({'MLmodelType': None, 'nthreads': None})
        self.set_keyword_alias('crossSection', ['ML-NEA', 'ML_NEA', 'crossSection', 'cross-section', 'cross_section','MLNEA'])
        self.set_keyword_alias('AIQM1DFTstar', ['AIQM1@DFT*'])
        self.set_keyword_alias('AIQM1DFT', ['AIQM1@DFT'])
        self.set_keyword_alias('ODM2star', ['ODM2*'])
        self.set_keyword_alias('CCSDTstarCBS', ['CCSD(T)*/CBS'])
        self.set_keyword_alias('ani1ccx', ['ANI-1ccx'])
        self.set_keyword_alias('ani1x', ['ANI-1x'])
        self.set_keyword_alias('ani2x', ['ANI-2x'])
        self.set_keyword_alias('ani1xd4', ['ANI-1x-D4'])
        self.set_keyword_alias('ani2xd4', ['ANI-2x-D4'])
        self.set_keyword_alias('gfn2xtb', ['GFN2-xTB'])
        self.args2pass = []
        self.parse_input_content([
            'minimizeError=RMSE',
            'initBetas=RR',
            'hyperopt.max_evals=8',
            'hyperopt.algorithm=tpe',
            'hyperopt.losstype=geomean',
            'hyperopt.w_y=1',
            'hyperopt.w_ygrad=1',
            'hyperopt.points_to_evaluate=0',
            'ase.fmax=0.02',
            'ase.steps=200',
            'ase.optimizer=LBFGS',
            'ase.linear=',
            'ase.symmetrynumber='
            ])
        self.optlines=''

    def parse(self, argsraw):
        if len(argsraw) == 0:
            Doc.printDoc({})
            stopper.stopMLatom('At least one option should be provided')
        elif len(argsraw) == 1:
            if os.path.exists(argsraw[0]):
                self.parse_input_file(argsraw[0])
            else:
                self.parse_input_content(argsraw[0])
            self.args2pass = self.args_string_list(['', None])
        elif len(argsraw) >= 2:
            self.parse_input_content(argsraw)
            self.args2pass = self.args_string_list(['', None])
        self.argProcess()

    def argProcess(self):
        
        if self.ani1x or self.ani2x or self.ani1ccx or self.ani1xd4 or self.ani2xd4:
            self.MLmodelType = 'ani'
            self.useMLmodel = True
        if self.XYZfile:
            with open(self.XYZfile) as f:
                natom=int(f.readline())
                nline=1
                for line in f:
                    nline+=1
            self.parse_input_content(['Ntotal=%d' % int(nline/(natom+2)),'Natom=%d'% natom])
        if self.CVopt:
            try:
                self.parse_input_content(['NcvOptFolds=%s' % self.NcvOptFolds])
            except:
                self.parse_input_content(['NcvOptFolds=5'])
            try:
                self.parse_input_content(['iCVoptPrefOut=%s' % self.iCVoptPrefOut])
            except:
                self.parse_input_content(['iCVoptPrefOut=icvopt'])
        if self.CVtest:
            try:
                self.parse_input_content(['NcvTestFolds=%s' % self.NcvtestFolds])
            except:
                self.parse_input_content(['NcvTestFolds=5'])
            try:
                self.parse_input_content(['iCVtestPrefOut=%s' % self.iCVtestPrefOut])
            except:
                self.parse_input_content(['iCVtestPrefOut=icvtest'])
        self.add_keyword_value_args(
                ['absXfileIn','absXYZfile','absYfile','absYgradXYZfile'],
            [os.path.abspath(self.XfileIn),os.path.abspath(self.XYZfile),os.path.abspath(self.Yfile),os.path.abspath(self.YgradXYZfile)]
            )
        for arg in self.args2pass:
            if bool(re.search('hyperopt\..+?\(.+?\)',arg)):
                import hyperopt 
                globals()['hyperopt']=hyperopt
                self.optlines = self.optlines+arg+'\n'

        if self.MLmodelType:
            try: self.MLprog=MLtasksCls.default_MLprog[self.MLmodelType.lower()]
            except: stopper.stopMLatom('Unkown MLmodelType')
        else:
            self.MLmodelType = 'KREG'
        if self.selfCorrect:
            self.parse_input_content([
            'nlayers=4'])
        if self.selfCorrect and (utils.argexist('iTrainOut=', self.args2pass) or utils.argexist('iTestOut=', self.args2pass) or utils.argexist('iSubtrainOut=', self.args2pass) or utils.argexist('iValidateOut=', self.args2pass)):
            stopper.stopMLatom('Indices of subsets cannot be saved for self-correction')
        
        if self.mlmodeltype=='krr-cm':
            self.parse_input_content([
            'molDescriptor=CM'
            ])
        if self.MLprog.lower() in MLtasksCls.mlatom_alias and not self.MLmodelOut:
            self.MLmodelOut='MLatom.unf'
        
        if self.betas:
            self.customBetas=self.betas
            
        if self.minimizeError.lower()=='rmse': self.args2pass=utils.addReplaceArg('minimizeError','',self.args2pass)

if __name__ == '__main__': 
    print(__doc__)
    MLtasks()
