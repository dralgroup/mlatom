#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! MLtasks: Machine learning tasks of MLatom                                 ! 
  ! Implementations by: Pavlo O. Dral and Fuchun Ge                           ! 
  !---------------------------------------------------------------------------! 
'''

import os, sys, time, shutil, json
import numpy as np
from .args_class import mlatom_args
from . import constants, data, interface_MLatomF, models, stopper, utils, stats
    
class CLItasks(object):
    MLatomF_tasks = [
        'useMLmodel',
        'createMLmodel',
        'estAccMLmodel',
        'XYZ2X',
        'analize',
        'sample',
    ]
    def __init__(self, args):
        self.args = args
    def run(self):
        if self.args.MLprog.lower() == 'mlatomf' and self.args.task in self.MLatomF_tasks and not self.args.method and not self.args.hyperparameter_optimization['hyperparameters']:
            return run_with_mlatomF(self.args)
        else: 
            return globals()[self.args.task](self.args)

def run_with_mlatomF(args):
    if args.nthreads:
        os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
        os.environ["MKL_NUM_THREADS"] = str(args.nthreads)
    if args.deltaLearn:
        pre_delta_learning(args)
    results = interface_MLatomF.ifMLatomCls.run(args.args2pass)
    if args.deltaLearn:
        post_delta_learning(args)
    return results
    
# Functions for tasks bleow. Name with corresponding task name in args_class.mLatom_args._task_list
def useMLmodel(args):
    if args.deltaLearn:
        pre_delta_learning(args)
    model = loading_model(args)
    molecular_database = loading_data(args.XYZfile, args.Yfile, args.YgradXYZfile, charges=args.charges, multiplicities=args.multiplicities)
    t_predict = predicting(model=model, molecular_database=molecular_database, value=bool(args.YestFile), gradient=bool(args.YgradXYZestFile), hessian=bool(args.hessianEstFile), ismethod=args.method)
    saving_predictions(molecular_database, YestFile=args.YestFile, YgradXYZestFile=args.YgradXYZestFile, hessianEstFile=args.hessianEstFile, method=args.method)
    if args.method:
        print('')
        if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar or args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
            nmol = 0
            for mol in molecular_database:
                nmol += 1
                print('\n\n Properties of molecule %d\n' % nmol)
                if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar:
                    printing_aiqm1_results(aiqm1=args.AIQM1, molecule=mol)
                elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                    printing_animethod_results(methodname=model.method, molecule=mol)
        else:
            for imol in range(len(molecular_database.molecules)): print(' Energy of molecule %6d: %25.13f Hartree' % (imol+1, molecular_database.molecules[imol].energy))
        print('')
    if args.deltaLearn:
        post_delta_learning(args)
    return t_predict

def createMLmodel(args):
    if args.deltaLearn:
        pre_delta_learning(args)
    molecular_database = loading_data(args.XYZfile, args.Yfile, args.YgradXYZfile, charges=args.charges, multiplicities=args.multiplicities)
    i_train, i_subtrain, i_validate, i_test, i_cvtest, i_cvopt = sampling(args=args)
    model = loading_model(args)
    result = training(
        model, molecular_database, 
        bool(args.Yfile), bool(args.YgradXYZfile), 
        i_train, i_subtrain, i_validate, i_cvopt,
        args.hyperparameter_optimization
    )
    if args.deltaLearn:
        post_delta_learning(args)
    return result

def estAccMLmodel(args):
    if args.deltaLearn:
        pre_delta_learning(args)
    molecular_database = loading_data(args.XYZfile, args.Yfile, args.YgradXYZfile)
    i_train, i_subtrain, i_validate, i_test, i_cvtest, i_cvopt = sampling(args=args)
    model = loading_model(args)
    if args.CVtest:
        t_train = 0
        t_pred  = 0
        if i_cvopt is None:
            i_cvopt = [None for _ in range(args.NcvTestFolds)]            
        for j in range(args.NcvTestFolds):
            i_cv_train = np.concatenate([i_cvtest[i] for i in range(args.NcvTestFolds) if i != j])
            i_cv_test  = i_cvtest[j]
            t_train += training(
                model, molecular_database, 
                bool(args.Yfile), bool(args.YgradXYZfile), 
                i_cv_train, i_subtrain, i_validate, i_cvopt[j],
                args.hyperparameter_optimization
            )['t_train']
            t_pred += predicting(
                model=model, molecular_database=molecular_database[i_cv_test], 
                value=bool(args.Yfile), gradient=bool(args.YgradXYZfile)
            )
        i_cv_all_test = np.concatenate(i_cvtest)
        result =  analyzing(
            molecular_database[i_cv_all_test], 
            'y', 'estimated_y' if args.Yfile else '',
            'xyz_derivatives', 'estimated_xyz_derivatives_y' if args.YgradXYZfile else '',
            'combined cross-validation test '
        )
    else:
        t_train = training(
            model, molecular_database, 
            bool(args.Yfile), bool(args.YgradXYZfile), 
            i_train, i_subtrain, i_validate, i_cvopt,
            args.hyperparameter_optimization
        )['t_train']
        t_pred = predicting(
            model=model, molecular_database=molecular_database[i_test], 
            value=bool(args.Yfile), gradient=bool(args.YgradXYZfile)
        )
        result =  analyzing(
            molecular_database[i_test], 
            'y', 'estimated_y' if args.Yfile else '',
            'xyz_derivatives', 'estimated_xyz_derivatives_y' if args.YgradXYZfile else '',
            'test '
        )
    result['t_train'] = t_train
    result['t_pred'] = t_pred
    saving_predictions(molecular_database, YestFile=args.YestFile, YgradXYZestFile=args.YgradXYZestFile, hessianEstFile=args.hessianEstFile, method=args.method)
    if args.deltaLearn:
        post_delta_learning(args)
    return result

def selfCorrect(args):
    yfilename = ''
    if args.createMLmodel or args.estAccMLmodel:
        sample_args = args.args2pass
        if args.createMLmodel:
            sample_args.remove('createMLmodel')
        if args.estAccMLmodel:
            sample_args.remove('estAccMLmodel')
        sample_args_lower = [arg.lower() for arg in sample_args]
        if ('sampling=structure-based' in sample_args_lower or
            'sampling=farthest-point'  in sample_args_lower or
            'sampling=random'          in sample_args_lower):
            if args.iTrainOut or args.iTestOut or args.iSubtrainOut or args.iValidateOut:
                stopper.stopMLatom('Indices of subsets cannot be saved for self-correction')
            print('\n Running sampling\n')
            sys.stdout.flush()
            interface_MLatomF.ifMLatomCls.run(['sample',
                                'iTrainOut=itrain.dat',
                                'iTestOut=itest.dat',
                                'iSubtrainOut=isubtrain.dat',
                                'iValidateOut=ivalidate.dat']
                            + sample_args)
        yfilename = args.Yfile
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
            args.sampling = 'user-defined'
            args.iTrainIn = 'itrain.dat'
            args.iTestIn = 'itest.dat'
            args.iSubtrainIn = 'isubtrain.dat'
            args.iValidateIn = 'ivalidate.dat'
            args.Yfile = ydatname
            args.YestFile = 'yest%d.dat' % nlayer
            if args.createMLmodel:
                args.MLmodelOut = 'mlmodlayer%d.unf' % nlayer
                args.task = 'createMLmodel'
            if args.estAccMLmodel:
                args.task = 'estAccMLmodel'
        elif args.useMLmodel:
            if nlayer > 1:
                ylayerm1 = [float(line) for line in open('ylayer%d.dat' % (nlayer - 1), 'r')]
            args.MLmodelIn = 'mlmodlayer%d.unf' % nlayer
            args.YestFile = 'yest%d.dat' % nlayer
            args.task = 'useMLmodel'
        CLItasks(args).run()
        
        if nlayer == 1:
            shutil.move('yest1.dat', 'ylayer1.dat')
        else:
            yestlayer = [float(line) for line in open('yest%d.dat' % nlayer, 'r')]
            with open('ylayer%d.dat' % nlayer, 'w') as fylayer:
                for ii in range(len(yestlayer)):
                    fylayer.writelines('%25.13f\n' % (ylayerm1[ii] + yestlayer[ii]))

def learningCurve(args):
    molecular_database = loading_data(args.XYZfile, args.Yfile, args.YgradXYZfile, charges=args.charges, multiplicities=args.multiplicities)
    if args.XfileIn:
        args.XfileIn = os.path.abspath(args.Xfilein)
    else:
        args.XYZfile = os.path.abspath(args.XYZfile)
    if args.Yfile: 
        args.Yfile = os.path.abspath(args.Yfile)
    if args.YgradXYZfile: 
        args.YgradXYZfile = os.path.abspath(args.YgradXYZfile)
    args.task = 'estAccMLmodel'
    args.iTrainIn = '../itrain.dat'
    args.iSubtrainIn = '../isubtrain.dat'
    args.iValidateIn = '../ivalidate.dat'
    args.iTestIn = '../itest.dat'
    args.iCVtestPrefIn = '../icvtest'
    args.iCVoptPrefIn = '../icvopt'
    Ntrains = [int(i) for i in str(args.lcNtrains).split(',')]
    Nrepeats = [int(i) for i in str(args.lcNrepeats).split(',')]
    if len(Nrepeats) == 1:
        Nrepeats = Nrepeats * len(Ntrains)
    surfix = '_'
    surfix+=args.mlmodeltype
    if args.xyzfile: surfix+='_en'
    if args.ygradxyzfile and not args.MLmodelIn: surfix+='_grad'
    dirname = args.lcDir
    if not os.path.isdir(dirname): os.mkdir(dirname)
    os.chdir(dirname)
    lcdir = os.getcwd()
    modeldirname = args.MLprog+surfix
    if not os.path.isdir(modeldirname): os.mkdir(modeldirname)
    if os.path.isfile(modeldirname+'/results.json'): 
        with open(modeldirname+'/results.json','r') as f:
            results = json.load(f)
        print(results,'\nprevious results ')
    else: results = {}
    for i, ntrain in enumerate(Ntrains):
        args.Ntrain = ntrain
        if not os.path.isdir('Ntrain_'+str(ntrain)): os.mkdir('Ntrain_'+str(ntrain))
        # enter Ntrain dir
        os.chdir('Ntrain_'+str(ntrain))
        print('\n\n testing for Ntrain = %d\n ==============================================================================' % ntrain)
        sys.stdout.flush()
        if args.MLmodelIn:
            with open('../'+args.MLprog+surfix+'/results.json','r') as f:
                old_results = json.load(f)
        for j in range(Nrepeats[i]):
            if not os.path.isdir(str(j)): os.mkdir(str(j))
            # enter repeat dir
            os.chdir(str(j))
            dataSampled = True
            for i in ['Train','Subtrain','Validate','Test']:
                if not os.path.isfile('i'+i.lower()+'.dat'):
                    dataSampled = False
            if not dataSampled:
                sampling(args=args,
                    iTrainOut='itrain.dat', iSubtrainOut='isubtrain.dat', iValidateOut='ivalidate.dat', iTestOut='itest.dat', iCVtestPrefOut='icvtest', iCVoptPrefOut='icvopt',
                )
            if not os.path.isdir(modeldirname): os.mkdir(modeldirname)
            # enter model dir
            os.chdir(modeldirname)
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
                    # args.Ntrain = ntrain
                    # estAccMLmodel
                    result = CLItasks(args).run()
                except:
                    print(" FAILED!")
                    continue
                finally:
                    os.system('rm .running')
            resultsfile=os.path.join(lcdir,modeldirname,'results.json')
            if os.path.isfile(resultsfile): 
                with open(resultsfile,'r') as f:
                    results = json.load(f)
            if str(ntrain) not in results.keys(): results[str(ntrain)] = {}
            if str(j) not in results[str(ntrain)].keys(): results[str(ntrain)][str(j)] = {}
            if args.MLmodelIn:
                result['t_train'] = old_results[str(ntrain)][str(j)]['t_train']
            results[str(ntrain)][str(j)] = result
            with open('result.json','w') as f:
                json.dump(result, f, sort_keys=False, indent=4)
            with open(resultsfile,'w') as f:
                json.dump(results, f, sort_keys=False, indent=4)
            
            mean_y_rmse = {i: np.mean([results[i][j]["values"]["rmse"] for j in results[i]]) for i in results}

            if args.Yfile:
                with open(os.path.join(lcdir,modeldirname,'lcy.csv'), 'w') as f:
                    f.write('Ntrain, meanRMSE, SD, Nrepeats, RMSEs\n')
                    mean_y_rmse = {i: np.mean([results[i][j]["values"]["rmse"] for j in results[i]]) for i in results}
                    mean_y_rmse_sd = {i: np.std([results[i][j]["values"]["rmse"] for j in results[i]], ddof=1) for i in results}
                    for nTrain in results.keys():
                        f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,mean_y_rmse[nTrain],mean_y_rmse_sd[nTrain],len(results[nTrain]),','.join([str(results[nTrain][i]["values"]["rmse"] ) for i in results[nTrain]])))
            if args.YgradXYZfile:
                with open(os.path.join(lcdir,modeldirname,'lcygradxyz.csv'), 'w') as f:
                    f.write('Ntrain, meanRMSE, SD, Nrepeats, RMSEs\n')
                    mean_y_rmse = {i: np.mean([results[i][j]["gradients"]["rmse"] for j in results[i]]) for i in results}
                    mean_y_rmse_sd = {i: np.std([results[i][j]["gradients"]["rmse"] for j in results[i]], ddof=1) for i in results}
                    for nTrain in results.keys():
                        try:
                            f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,mean_y_rmse[nTrain],mean_y_rmse_sd[nTrain],len(results[nTrain]),','.join([str(results[nTrain][i]["gradients"]["rmse"] ) for i in results[nTrain]])))
                        except:
                            pass
            with open(os.path.join(lcdir,modeldirname, 'lctimetrain.csv'),'w') as f:
                f.write('Ntrain, meanTime, SD, Nrepeats, times\n')
                mean_y_rmse = {i: np.mean([results[i][j]["t_train"] for j in results[i]]) for i in results}
                mean_y_rmse_sd = {i: np.std([results[i][j]["t_train"] for j in results[i]], ddof=1) for i in results}
                for nTrain in results.keys():
                    f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,mean_y_rmse[nTrain],mean_y_rmse_sd[nTrain],len(results[nTrain]),','.join([str(results[nTrain][i]["t_train"]) for i in results[nTrain]])))
            with open (os.path.join(lcdir,modeldirname, 'lctimepredict.csv'),'w') as f:
                f.write('Ntrain, meanTime, SD, Nrepeats, times\n')
                mean_y_rmse = {i: np.mean([results[i][j]["t_pred"] for j in results[i]]) for i in results}
                mean_y_rmse_sd = {i: np.std([results[i][j]["t_pred"] for j in results[i]], ddof=1) for i in results}
                for nTrain in results.keys():
                    f.write('%s, %f, %f, %d, "%s"\n' % (nTrain,mean_y_rmse[nTrain],mean_y_rmse_sd[nTrain],len(results[nTrain]),','.join([str(results[nTrain][i]["t_pred"]) for i in results[nTrain]])))
            os.chdir('../..')
        os.chdir('..')
    os.chdir('..')
    return results
  
def XYZ2X(args):
    interface_MLatomF.ifMLatomCls.run(args.argsraw)

def XYZ2SMI(args):
    molDB = data.molecular_database.from_xyz_file(args.XYZfile)
    molDB.write_file_with_smiles(args.SMIout)

def SMI2XYZ(args):
    molDB = data.molecular_database.from_smiles_file(args.SMIin)
    molDB.write_file_with_xyz_coordinates(args.XYZout)

def analyze(args):
    return interface_MLatomF.ifMLatomCls.run(args.args2pass)

def sample(args):
    interface_MLatomF.ifMLatomCls.run(args.args2pass)

def sampleFromSlices(args):
    from . import sliceData
    sliceData.sliceDataCls(argsSD = args.args2pass)

def mergeSlices(args):
    from . import sliceData
    sliceData.sliceDataCls(argsSD = args.args2pass)

def slice(args):
    from . import sliceData
    sliceData.sliceDataCls(argsSD = args.args2pass)

def geomopt(args):
    from . import simulations
    molDB = loading_data(XYZfile=args.XYZfile, charges=args.charges, multiplicities=args.multiplicities)
    model = loading_model(args)
    fname = args.optXYZ
    # if os.path.exists(fname): stopper.stopMLatom(f'File {fname} already exists; please delete or rename it')
    if os.path.exists(fname): 
        os.remove(fname)
        print(f' * Warning * File {fname} already exists; MLatom will overwrite {fname}')
    db_opt = data.molecular_database()
    kwargs = {}
    if args.optProg:        kwargs['program'] = args.optprog
    if args.ts:             kwargs['ts'] = True
    if args.ase.fmax:       kwargs['convergence_criterion_for_forces'] = float(args.ase.fmax)
    if args.ase.steps:      kwargs['maximum_number_of_steps'] = int(args.ase.steps)
    if args.ase.optimizer:  kwargs['optimization_algorithm'] = args.ase.optimizer
    if len(molDB.molecules)<=10: 
        kwargs['print_properties'] = 'all'
    else:
        kwargs['print_properties'] = 'min'
    if args.printall:       kwargs['print_properties'] = 'all'
    if args.printmin:       kwargs['print_properties'] = None
    if args.dumpopttrajs:
        kwargs['dump_trajectory_interval'] = 1
    else:
        kwargs['dump_trajectory_interval'] = None
    kwargs['format'] = 'json'
    for imol, mol in enumerate(molDB):
        mol.number = imol+1
        print(' %s ' % ('='*78))
        print(' Optimization of molecule %d' % (imol+1))
        print(' %s \n' % ('='*78))
        kwargs['filename'] = f'opttraj{imol+1}.json'
        geomopt = simulations.optimize_geometry(model=model,
                                        initial_molecule=mol,
                                        **kwargs)
        db_opt.molecules.append(geomopt.optimized_molecule)
        print(f'\n\n   {"Iteration":^10s}    {"Energy (Hartree)":^25s}')
        for step in geomopt.optimization_trajectory.steps:
            print('   %10d    %25.13f' % (step.step+1, step.molecule.energy))
        if args.AIQM1 or args.AIQM1DFT or args.AIQM1DFTstar:
            print('\n\n Final properties of molecule %d\n' % (imol+1))
            printing_aiqm1_results(aiqm1=args.AIQM1, molecule=geomopt.optimized_molecule)
            print('\n')
        elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
            print('\n\n Final properties of molecule %d\n' % (imol+1))
            printing_animethod_results(methodname=model.method, molecule=geomopt.optimized_molecule)
            print('\n')
        else:
            print('\n Final energy of molecule %6d: %25.13f Hartree\n\n' % (imol+1, geomopt.optimized_molecule.energy))
    db_opt.write_file_with_xyz_coordinates(filename=fname)

def ts(args):
    geomopt(args)

def freq(args):
    from . import simulations
    molDB = loading_data(args.XYZfile, charges=args.charges, multiplicities=args.multiplicities)
    model = loading_model(args)
    kwargs = {}
    if args.freqProg:       kwargs['program'] = args.freqProg
    elif args.optProg:      kwargs['program'] = args.optProg
    for imol, mol in enumerate(molDB):
        mol.number = imol+1
        if args.ase.linear:
            if len(molDB) == 1:
                linearlist = [args.ase.linear]
            else:
                linearlist = args.ase.linear.split(',')
            if linearlist[imol] == '1': mol.shape = 'linear'
            else: mol.shape='nonlinear'
        else:
            if mol.is_it_linear():
                mol.shape = 'linear'
            else:
                mol.shape = 'nonlinear'
        if args.ase.symmetrynumber:
            if len(molDB) == 1:
                symmetrynumbers = [args.ase.symmetrynumber]
            else:
                symmetrynumbers = args.ase.symmetrynumber.split(',')
            mol.symmetry_number = int(symmetrynumbers[imol])
        geomopt = simulations.thermochemistry(model=model,
                                        molecule=mol,
                                        **kwargs)
        mol.dump(f'freq{mol.number}.json',format='json')
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
            printing_aiqm1_results(aiqm1=args.AIQM1, molecule=mol)
            print('')
        elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
            printing_animethod_results(methodname=model.method, molecule=mol)
            print('')
        if hasattr(mol, 'energy'):  print(fmt % ('ZPE-exclusive internal energy at      0 K', mol.energy))
        if hasattr(mol, 'ZPE'):     print(fmt % ('Zero-point vibrational energy', mol.ZPE))
        if hasattr(mol, 'U0'):      print(fmt % ('Internal energy                  at   0 K', mol.U0))
        if hasattr(mol, 'H'):       print(fmt % ('Enthalpy                         at 298 K', mol.H))
        if hasattr(mol, 'G'):       print(fmt % ('Gibbs free energy                at 298 K', mol.G))
        # To-do: add entropy
        if 'DeltaHf298' in mol.__dict__:
            print('')
            fmt = ' %-41s: %15.5f Hartree %15.5f kcal/mol'
            if 'atomization_energy_0K' in mol.__dict__:               print(fmt % ('Atomization enthalpy             at   0 K', mol.atomization_energy_0K, mol.atomization_energy_0K * constants.Hartree2kcalpermol))
            if 'ZPE_exclusive_atomization_energy_0K' in mol.__dict__: print(fmt % ('ZPE-exclusive atomization energy at   0 K', mol.ZPE_exclusive_atomization_energy_0K, mol.ZPE_exclusive_atomization_energy_0K * constants.Hartree2kcalpermol))
            print(fmt % ('Heat of formation                at 298 K', mol.DeltaHf298, mol.DeltaHf298 * constants.Hartree2kcalpermol))
            # To-do: make it work for ANI-1ccx
            if args.AIQM1:
                if mol.aiqm1_nn.energy_standard_deviation > 0.41*constants.kcalpermol2Hartree:
                    print(' * Warning * Heat of formation have high uncertainty!')
            if args.ani1ccx:
                if mol.ani1ccx.energy_standard_deviation > 1.68*constants.kcalpermol2Hartree:
                    print(' * Warning * Heat of formation have high uncertainty!')
        print('')

def irc(args):
    from . import simulations
    molDB = loading_data(args.XYZfile, charges=args.charges, multiplicities=args.multiplicities)
    model = loading_model(args)
    for imol, mol in enumerate(molDB):
        mol.number = imol+1
        geomopt = simulations.irc(model=model,
                                    ts_molecule=mol)

def MD(args):
    from . import md_cmd 
    model = loading_model(args)
    md_cmd.MD_CMD.dynamics(args.args2pass, model)

def MD2vibr(args):
    from . import md2vibr_cmd
    md2vibr_cmd.md2vibr.simulate(args.args2pass)

def crossSection(args):
    from . import ML_NEA 
    ML_NEA.parse_api(args.argsraw)

def MLQD(args):
    from mlqd.evolution import quant_dyn
    mlqd_args = {}
    for argg in args.args2pass:
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

def MLTPA(args):
    from .MLTPA import MLTPA as mltpa
    mltpa(args.args2pass).predict()

def al(args):
    from .al import active_learning
    molDB = loading_data(args.XYZfile, charges=args.charges, multiplicities=args.multiplicities)
    args.nthreads = 1
    model = loading_model(args)
    active_learning(
        molecule=molDB[0],
        reference_method=model,
    )

# Reusable functions below. Name with -ing form
def loading_data(XYZfile, Yfile=None, YgradXYZfile=None, charges=None, multiplicities=None):
    assert XYZfile, 'please provide data file(s) needed.'
    if not os.path.exists(XYZfile):
        stopper.stopMLatom(f'xyz file {XYZfile} is not found!')
    molecular_database = data.molecular_database.from_xyz_file(XYZfile)
    if Yfile: 
        molecular_database.add_scalar_properties_from_file(Yfile)
    if YgradXYZfile:
        molecular_database.add_xyz_derivative_properties_from_file(YgradXYZfile)
    if charges:
        molecular_database.charges = [int(xx) for xx in str(charges).split(',')]
    if multiplicities:
        molecular_database.multiplicities = [int(xx) for xx in str(multiplicities).split(',')]
    return molecular_database

def loading_method(args):  
    kwargs = {'method': args.method}
    if args.mndokeywords: args.QMprogramKeywords = args.mndokeywords
    if args.QMprogramKeywords:
        if 'AIQM' in kwargs['method']:
            kwargs['qm_program_kwargs'] = {'read_keywords_from_file': args.QMprogramKeywords}
            if args.mndokeywords or 'sparrow' in args.qmprog.lower():
                kwargs['qm_program_kwargs']['save_files_in_current_directory'] = True
        else:
            kwargs['read_keywords_from_file'] = args.QMprogramKeywords
            if args.mndokeywords or 'sparrow' in args.qmprog.lower():
                kwargs['save_files_in_current_directory'] = True
    if args.qmprog:
        if 'AIQM' in kwargs['method']:
            kwargs['qm_program'] = args.qmprog
        else:
            kwargs['program'] = args.qmprog                    
    method = models.methods(**kwargs)
    method.set_num_threads(args.nthreads)
    return method

def loading_model(args):
    if args.method:
        return loading_method(args)
    model_file = args.MLmodelIn if args.MLmodelIn else args.MLmodelOut
    if args.MLprog.lower() in ['torchani', 'ani']:
        model = models.ani(model_file=model_file)
    elif args.MLprog.lower() in ['dp', 'dpmd', 'deepmd', 'deepmd-kit']:
        model = models.dpmd(model_file=model_file)
    elif args.MLprog.lower() in ['physnet']:
        model = models.physnet(model_file=model_file)
    elif args.MLprog.lower() in ['mace']:
        model = models.mace(model_file=model_file)
    elif args.MLprog.lower() in ['gap', 'gap-soap']:
        model = models.gap(model_file=model_file)
    elif args.MLprog.lower() in ['sgdml', 'gdml']:
        model = models.sgdml(model_file=model_file)
    elif args.MLprog.lower() in ['mlatomf', '']:
        model = models.kreg(model_file=model_file, ml_program='MLatomF')
    elif args.MLprog.lower() in ['kreg_api', 'kregapi']:
        model = models.kreg(model_file=model_file, ml_program='KREG_API')
    else:
        stopper.stopMLatom("unknown MLmodelType/MLprog")
    model.parse_args(args)
    model.set_num_threads(args.nthreads)
    return model

def sampling(args=None, XYZfile=None, XfileIn=None, sampling=None, Nuse=None, 
             Ntrain=None, Nsubtrain=None, Nvalidate=None, Ntest=None, 
             iTrainIn=None, iSubtrainIn=None, iValidateIn=None, iTestIn=None, 
             iTrainOut=None, iSubtrainOut=None, iValidateOut=None, iTestOut=None,
             CVtest=False, NcvTestFolds=None, iCVtestPrefIn=None, iCVtestPrefOut=None,
             CVopt=False, NcvOptFolds=None, iCVoptPrefIn=None, iCVoptPrefOut=None, sample_test=False):
    i_train    = None
    i_subtrain = None
    i_validate = None
    i_test     = None
    i_cvtest   = None
    i_cvopt    = None

    if args:
        if args.estAccMLmodel or args.learningCurve:
            sample_test = True
        args = args.copy('sample', ["XYZfile", "XfileIn", "sampling", "CVtest", "CVopt", "Nuse", "Ntrain", "Nsubtrain", "Nvalidate", "Ntest", "NcvTestFolds", "NcvOptFolds", "iTrainIn", "iSubtrainIn", "iValidateIn", "iTestIn", "iCVtestPrefIn", "iCVoptPrefIn", "iTrainOut", "iSubtrainOut", "iValidateOut", "iTestOut", "iCVtestPrefOut", "iCVoptPrefOut"])
    else:
        args = mlatom_args()
        args.parse(['sample'])

    if XYZfile:        args.XYZfile        = XYZfile
    if XfileIn:        args.XfileIn        = XfileIn
    if sampling:       args.sampling       = sampling
    if CVtest:         args.CVtest         = CVtest
    if CVopt:          args.CVopt          = CVopt
    if Nuse:           args.Nuse           = Nuse
    if Ntrain:         args.Ntrain         = Ntrain
    if Nsubtrain:      args.Nsubtrain      = Nsubtrain
    if Nvalidate:      args.Nvalidate      = Nvalidate
    if Ntest:          args.Ntest          = Ntest
    if NcvTestFolds:   args.NcvTestFolds   = NcvTestFolds
    if NcvOptFolds:    args.NcvOptFolds    = NcvOptFolds 
    if iTrainIn:       args.iTrainIn       = iTrainIn
    if iSubtrainIn:    args.iSubtrainIn    = iSubtrainIn
    if iValidateIn:    args.iValidateIn    = iValidateIn
    if iTestIn:        args.iTestIn        = iTestIn
    if iCVtestPrefIn:  args.iCVtestPrefIn  = iCVtestPrefIn
    if iCVoptPrefIn:   args.iCVoptPrefIn   = iCVoptPrefIn
    if not iTrainOut:      iTrainOut      = args.iTrainOut
    if not iSubtrainOut:   iSubtrainOut   = args.iSubtrainOut
    if not iValidateOut:   iValidateOut   = args.iValidateOut
    if not iTestOut:       iTestOut       = args.iTestOut
    if not iCVtestPrefOut: iCVtestPrefOut = args.iCVtestPrefOut
    if not iCVoptPrefOut:  iCVoptPrefOut  = args.iCVoptPrefOut

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        if args.sampling.lower() != 'user-defined':
            args.iTrainIn      = ''
            args.iSubtrainIn   = ''
            args.iValidateIn   = ''
            args.iTestIn       = ''
            args.iCVtestPrefIn = ''
            args.iCVoptPrefIn  = ''
            args.iTrainOut      = tmpdirname + '/itrain.dat'
            args.iSubtrainOut   = tmpdirname + '/isubtrain.dat'
            args.iValidateOut   = tmpdirname + '/ivalidate.dat'
            if sample_test:
                args.iTestOut   = tmpdirname + '/itest.dat'
            args.iCVtestPrefOut = tmpdirname + '/icvtest'
            args.iCVoptPrefOut  = tmpdirname + '/icvopt' if not args.CVtest else 'icvopt'
            sample(args)
            if os.path.exists(tmpdirname + '/itrain.dat'):    args.iTrainIn      = tmpdirname + '/itrain.dat'
            if os.path.exists(tmpdirname + '/isubtrain.dat'): args.iSubtrainIn   = tmpdirname + '/isubtrain.dat'
            if os.path.exists(tmpdirname + '/ivalidate.dat'): args.iValidateIn   = tmpdirname + '/ivalidate.dat'
            if os.path.exists(tmpdirname + '/itest.dat'):     args.iTestIn       = tmpdirname + '/itest.dat'
            if os.path.exists(tmpdirname + '/icvtest1.dat'):  args.iCVtestPrefIn = tmpdirname + '/icvtest'
            if os.path.exists(tmpdirname + '/icvopt1.dat'):   args.iCVoptPrefIn  = tmpdirname + '/icvopt'
            if os.path.exists(tmpdirname + '/icvtest1icvopt1.dat'):   args.iCVoptPrefIn  = 'icvopt'
            if iTrainOut:      os.system(f'cp {tmpdirname}/itrain.dat {iTrainOut}')
            if iSubtrainOut:   os.system(f'cp {tmpdirname}/isubtrain.dat {iSubtrainOut}')
            if iValidateOut:   os.system(f'cp {tmpdirname}/ivalidate.dat {iValidateOut}')
            if iTestOut:       os.system(f'cp {tmpdirname}/itest.dat {iTestOut}')
            if args.CVtest and iCVtestPrefOut: 
                if args.CVopt and iCVoptPrefOut:
                    os.system(f"find {tmpdirname} -name '*icvopt*' -exec bash -c ' mv $0 ${{0/\"icvopt\"/\"{iCVoptPrefOut}\"}}' {{}} \;")
                    args.iCVoptPrefIn = iCVoptPrefOut
                os.system(f"find {tmpdirname} -name 'icvtest*' -exec bash -c ' cp $0 ${{0/\"{tmpdirname}/icvtest\"/\"{iCVtestPrefOut}\"}}' {{}} \;")
            elif args.CVopt and iCVoptPrefOut: 
                os.system(f"find {tmpdirname} -name 'icvopt*' -exec bash -c ' cp $0 ${{0/\"{tmpdirname}/icvopt\"/\"{iCVoptPrefOut}\"}}' {{}} \;")
    
        if args.iTrainIn:
            i_train = np.loadtxt(args.iTrainIn).astype(int) - 1 
        if args.iSubtrainIn:
            i_subtrain = np.loadtxt(args.iSubtrainIn).astype(int) - 1 
        if args.iValidateIn:
            i_validate = np.loadtxt(args.iValidateIn).astype(int) - 1
        if args.iTestIn:
            i_test = np.loadtxt(args.iTestIn).astype(int) - 1
        if args.iCVtestPrefIn:
            i_cvtest = []
            for i in range(args.NcvTestFolds):
                i_cvtest.append(np.loadtxt(f'{args.iCVtestPrefIn}{i+1}.dat').astype(int) - 1)
        if args.iCVoptPrefIn:
            i_cvopt = []
            if args.iCVtestPrefIn:
                for j in range(args.NcvTestFolds):
                    i_cvtest_cvopt = []
                    for i in range(args.NcvOptFolds):
                        i_cvtest_cvopt.append(np.loadtxt(f'{args.iCVtestPrefIn}{j+1}{args.iCVoptPrefIn}{i+1}.dat').astype(int) - 1)
                    i_cvopt.append(i_cvtest_cvopt)
            else:
                for i in range(args.NcvOptFolds):
                    i_cvopt.append(np.loadtxt(f'{args.iCVoptPrefIn}{i+1}.dat').astype(int) - 1)

    return i_train, i_subtrain, i_validate, i_test, i_cvtest, i_cvopt

def predicting(model, molecular_database, value=None, gradient=None, hessian=None, ismethod=False):
    time_start =  time.time()
    if ismethod:
        model.predict(
            molecular_database=molecular_database, 
            calculate_energy=value,
            calculate_energy_gradients=gradient,
            calculate_hessian=hessian
        )
    else:
        model.predict(
            molecular_database=molecular_database, 
            property_to_predict='estimated_y' if value else None,
            xyz_derivative_property_to_predict='estimated_xyz_derivatives_y' if gradient else None,
            hessian_to_predict='estimated_hessian_y' if hessian else None
        )
    time_stop = time.time()
    return time_stop - time_start

def saving_predictions(molecular_database, YestFile=None, YgradXYZestFile=None, hessianEstFile=None, method=None):
    if YestFile:
        molecular_database.write_file_with_properties(YestFile, 'energy' if method else 'estimated_y')
    if YgradXYZestFile:
        molecular_database.write_file_with_xyz_derivative_properties(YgradXYZestFile, 'energy_gradients' if method else 'estimated_xyz_derivatives_y')
    if hessianEstFile:
        molecular_database.write_file_with_hessian(hessianEstFile, 'hessian' if method else 'estimated_hessian_y')

def training(model, molecular_database, value=None, gradient=None, i_train=None, i_subtrain=None, i_validate=None, i_cvopt=None, hyperparameter_optimization=None):
    time_start =  time.time()
    if hyperparameter_optimization and hyperparameter_optimization['hyperparameters']:
        model.optimize_hyperparameters(
            subtraining_molecular_database=molecular_database[i_subtrain],
            validation_molecular_database=molecular_database[i_validate],
            cv_splits_molecular_databases=[molecular_database[i_split] for i_split in i_cvopt] if i_cvopt else None,
            training_kwargs={
                'property_to_learn': 'y' if value else None, 
                'xyz_derivative_property_to_learn': 'xyz_derivatives' if gradient else None
            },
            prediction_kwargs={
                'property_to_predict': 'estimated_y' if value else None,
                'xyz_derivative_property_to_predict': 'estimated_xyz_derivatives_y' if gradient else None
            }, 
            **hyperparameter_optimization,
        )
    # if not (hyperparameter_optimization and hyperparameter_optimization['optimization_algorithm']) or model.meta_data['genre'] == 'kernel_method':
    model.train(
        molecular_database=molecular_database[i_train], 
        property_to_learn='y' if value else None, 
        xyz_derivative_property_to_learn='xyz_derivatives' if gradient else None
    )
    time_stop = time.time()
    predicting(model=model, molecular_database=molecular_database[i_train], value=value, gradient=gradient)
    result = analyzing(molecular_database[i_train], 
        'y', 'estimated_y' if value else '',
        'xyz_derivatives', 'estimated_xyz_derivatives_y' if gradient else '',
        'training '
    )
    result['t_train'] = time_stop - time_start
    return result

def analyzing(molecular_database, ref_value='', est_value='', ref_grad='', est_grad='', set_name=''):
    result={}
    length = len(molecular_database)
    if est_value:
        ref = molecular_database.get_properties(ref_value)
        est = molecular_database.get_properties(est_value)
        mae = stats.mae(est, ref)
        mse = stats.mse(est, ref)
        rmse = stats.rmse(est, ref)
        mean_y = stats.mean(ref)
        mean_yest = stats.mean(est)
        corr_coef = stats.correlation_coefficient(ref, est)
        a, b, r_squared, SE_a, SE_b = stats.linear_regression(ref, est)
        pos_off, pos_off_est, pos_off_ref, pos_off_idx = stats.largest_positive_outlier(est, ref)
        neg_off, neg_off_est, neg_off_ref, neg_off_idx = stats.largest_negative_outlier(est, ref)
        result["values"] = {
            "length": length,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mean_y": mean_y,
            "mean_yest": mean_yest,
            "corr_coef": corr_coef,
            "a": a,
            "b": b,
            "r_squared": r_squared,
            "pos_off": pos_off,
            "pos_off_est": pos_off_est,
            "pos_off_ref": pos_off_ref,
            "pos_off_idx": pos_off_idx,
            "neg_off": neg_off,
            "neg_off_est": neg_off_est,
            "neg_off_ref": neg_off_ref,
            "neg_off_idx": neg_off_idx,
        }
        print(f''' Analysis for values
 Statistical analysis for {length} entries in the {set_name}set
   MAE                      ={'%26.13f' % mae}
   MSE                      ={'%26.13f' % mse}
   RMSE                     ={'%26.13f' % rmse}
   mean(Y)                  ={'%26.13f' % mean_y}
   mean(Yest)               ={'%26.13f' % mean_yest}
   correlation coefficient  ={'%26.13f' % corr_coef}
   linear regression of {{y, y_est}} by f(a,b) = a + b * y
     R^2                    ={'%26.13f' % r_squared}
     a                      ={'%26.13f' % a}
     b                      ={'%26.13f' % b}
     SE_a                   ={'%26.13f' % SE_a}
     SE_b                   ={'%26.13f' % SE_b}
   largest positive outlier
     error                  ={'%26.13f' % pos_off}
     index                  ={'%26.13f' % pos_off_idx}
     estimated value        ={'%26.13f' % pos_off_est}
     reference value        ={'%26.13f' % pos_off_ref}
   largest negative outlier
     error                  ={'%26.13f' % neg_off}
     index                  ={'%26.13f' % neg_off_idx}
     estimated value        ={'%26.13f' % neg_off_est}
     reference value        ={'%26.13f' % neg_off_ref}''')
    if est_grad:
        ref = molecular_database.get_xyz_vectorial_properties(ref_grad).flatten()
        est = molecular_database.get_xyz_vectorial_properties(est_grad).flatten()
        mae = stats.mae(est, ref)
        mse = stats.mse(est, ref)
        rmse = stats.rmse(est, ref)
        mean_y = stats.mean(ref)
        mean_yest = stats.mean(est)
        corr_coef = stats.correlation_coefficient(ref, est)
        a, b, r_squared, SE_a, SE_b = stats.linear_regression(ref, est)
        pos_off, pos_off_est, pos_off_ref, pos_off_idx = stats.largest_positive_outlier(est, ref)
        neg_off, neg_off_est, neg_off_ref, neg_off_idx = stats.largest_negative_outlier(est, ref)
        result["gradients"] = {
            "length": length,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mean_y": mean_y,
            "mean_yest": mean_yest,
            "corr_coef": corr_coef,
            "a": a,
            "b": b,
            "r_squared": r_squared,
            "pos_off": pos_off,
            "pos_off_est": pos_off_est,
            "pos_off_ref": pos_off_ref,
            "pos_off_idx": pos_off_idx,
            "neg_off": neg_off,
            "neg_off_est": neg_off_est,
            "neg_off_ref": neg_off_ref,
            "neg_off_idx": neg_off_idx,
        }
        print(f''' Analysis for gradients in XYZ coordinates
 Statistical analysis for {length} entries in the {set_name}set
   MAE                      ={'%26.13f' % mae}
   MSE                      ={'%26.13f' % mse}
   RMSE                     ={'%26.13f' % rmse}
   mean(Y)                  ={'%26.13f' % mean_y}
   mean(Yest)               ={'%26.13f' % mean_yest}
   correlation coefficient  ={'%26.13f' % corr_coef}
   linear regression of {{y, y_est}} by f(a,b) = a + b * y
     R^2                    ={'%26.13f' % r_squared}
     a                      ={'%26.13f' % a}
     b                      ={'%26.13f' % b}
     SE_a                   ={'%26.13f' % SE_a}
     SE_b                   ={'%26.13f' % SE_b}
   largest positive outlier
     error                  ={'%26.13f' % pos_off}
     estimated value        ={'%26.13f' % pos_off_est}
     reference value        ={'%26.13f' % pos_off_ref}
   largest negative outlier
     error                  ={'%26.13f' % neg_off}
     estimated value        ={'%26.13f' % neg_off_est}
     reference value        ={'%26.13f' % neg_off_ref}\n
    ''')
    return result

def pre_delta_learning(args):
    if args.Yb != '': yb = [float(line) for line in open(args.Yb, 'r')]
    if args.YgradB != '': ygradb = [[float(xx) for xx in line.split()] for line in open(args.YgradB, 'r')]
    if args.YgradXYZb != '': ygradxyzb = utils.readXYZgrads(args.YgradXYZb)
    if args.createMLmodel or args.estAccMLmodel:
        # Delta of Y values
        if args.Yb != '':
            ydatname = '%s-%s.dat' % (utils.fnamewoExt(args.Yt), utils.fnamewoExt(args.Yb))
            args.Yfile = ydatname
            yt = [float(line) for line in open(args.Yt, 'r')]
            with open(ydatname, 'w') as fcorr:
                for ii in range(len(yb)):
                    fcorr.writelines('%25.13f\n' % (yt[ii] - yb[ii]))
        # Delta of gradients
        if args.YgradB != '':
            ygradfname = '%s-%s.dat' % (utils.fnamewoExt(args.YgradT), utils.fnamewoExt(args.YgradB))
            args.YgradFile = ygradfname
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
            args.YgradXYZfile = ygradxyzfname
            ygradxyzt = utils.readXYZgrads(args.YgradXYZt)
            with open(ygradxyzfname, 'w') as fcorr:
                for imol in range(len(ygradxyzb)):
                    fcorr.writelines('%d\n\n' % len(ygradxyzb[imol]))
                    for iatom in range(len(ygradxyzb[imol])):
                        strtmp = ''
                        for idim in range(3):
                            strtmp += '%25.13f   ' % (ygradxyzt[imol][iatom][idim] - ygradxyzb[imol][iatom][idim])
                        fcorr.writelines('%s\n' % strtmp)

def post_delta_learning(args):
    if args.Yb != '': yb = [float(line) for line in open(args.Yb, 'r')]
    if args.YgradB != '': ygradb = [[float(xx) for xx in line.split()] for line in open(args.YgradB, 'r')]
    if args.YgradXYZb != '': ygradxyzb = utils.readXYZgrads(args.YgradXYZb)
    if utils.argexist('YestFile=', args.args2pass):
        corr = [float(line) for line in open(args.YestFile, 'r')]
        with open(args.YestT, 'w') as fyestt:
            for ii in range(len(yb)):
                fyestt.writelines('%25.13f\n' % (yb[ii] + corr[ii]))
    if utils.argexist('YgradEstFile=', args.args2pass):
        corr = [[float(xx) for xx in line.split()] for line in open(args.YgradEstFile, 'r')]
        with open(args.YgradEstT, 'w') as fyestt:
            for ii in range(len(ygradb)):
                strtmp = ''
                for jj in range(len(ygradb[ii])):
                    strtmp += '%25.13f   ' % (ygradb[ii][jj] + corr[ii][jj])
                fyestt.writelines('%s\n' % strtmp)
    if utils.argexist('YgradXYZestFile=', args.args2pass):
        corr = utils.readXYZgrads(args.YgradXYZestFile)
        with open(args.YgradXYZestT, 'w') as fyestt:
            for imol in range(len(ygradxyzb)):
                fyestt.writelines('%d\n\n' % len(ygradxyzb[imol]))
                for iatom in range(len(ygradxyzb[imol])):
                    strtmp = ''
                    for idim in range(3):
                        strtmp += '%25.13f   ' % (ygradxyzb[imol][iatom][idim] + corr[imol][iatom][idim])
                    fyestt.writelines('%s\n' % strtmp)

def printing_aiqm1_results(aiqm1=True, molecule=None):
    fmt = ' %-41s: %15.8f Hartree'
    print(fmt % ('Standard deviation of NN contribution', molecule.aiqm1_nn.energy_standard_deviation), end='')
    print(' %15.5f kcal/mol' % (molecule.aiqm1_nn.energy_standard_deviation * constants.Hartree2kcalpermol))
    print(fmt % ('NN contribution', molecule.aiqm1_nn.energy))
    print(fmt % ('Sum of atomic self energies', molecule.aiqm1_atomic_energy_shift.energy))
    print(fmt % ('ODM2* contribution', molecule.odm2star.energy), file=sys.stdout)
    if aiqm1: print(fmt % ('D4 contribution', molecule.d4_wb97x.energy))
    print(fmt % ('Total energy', molecule.energy))
    
def printing_animethod_results(methodname=None, molecule=None):
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
