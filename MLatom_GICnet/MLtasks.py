#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! MLtasks: Machine learning tasks of MLatom                                 ! 
  ! Implementations by: Pavlo O. Dral and Fuchun Ge                           ! 
  !---------------------------------------------------------------------------! 
'''

import os, sys, time, shutil, re, copy, json
from cgi import test
from typing import Tuple
from numpy.core.defchararray import add
from io                   import StringIO
from contextlib           import redirect_stdout
from math                 import inf, log
import os, sys, subprocess, time, shutil, re, copy, json
import numpy as np
from io         import StringIO
from contextlib import redirect_stdout
try:
    from .args_class import ArgsBase
    from . import stopper
    from . import interface_MLatomF
    #from . import ML_NEA
    #from . import geomopt
    #from .interfaces.sGDML     import interface_sGDML
    #from .interfaces.PhysNet   import interface_PhysNet
    #from .interfaces.GAP       import interface_GAP
    #from .interfaces.DeePMDkit import interface_DeePMDkit
    #from .interfaces.TorchANI  import interface_TorchANI
    #from .ML_NEA import CrossSection_calc
except:
    from args_class import ArgsBase
    import stopper
    import interface_MLatomF
    import ML_NEA
    import geomopt
    from ML_NEA import CrossSection_calc
    from interfaces.sGDML     import interface_sGDML
    from interfaces.PhysNet   import interface_PhysNet
    from interfaces.GAP       import interface_GAP
    from interfaces.DeePMDkit import interface_DeePMDkit
    from interfaces.TorchANI  import interface_TorchANI

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
                'ani-tl':'torchani'
                    
            }
    mlatom_alias=['mlatomf','mlatom']
    gap_alias=[ 'gap', 'gap_fit', 'gapfit']
    sgdml_alias=['sgdml']
    deepmd_alias=['dp','deepmd','deepmd-kit']
    physnet_alias=['physnet']
    ani_alias=['torchani','ani']
    KMs = mlatom_alias+gap_alias+sgdml_alias
    model_needs_subvalid = deepmd_alias+ani_alias+physnet_alias+sgdml_alias
    index_prefix = ''
    def __init__(self, argsMLtasks = sys.argv[1:]):
        global args
        args = Args()
        args.parse(argsMLtasks)
        
        if args.deltaLearn:
            self.deltaLearn()
        elif args.selfCorrect:
            self.selfCorrect()
        elif args.learningCurve:
            self.learningCurve()
        elif args.crossSection:
            CrossSection_args=copy.deepcopy(args.args2pass)
            deadlist=[]
            for arg in CrossSection_args:
                flagmatch = re.search('(^nthreads)|(^hyperopt)|(^setname=)|(^learningcurve$)|(^lcntrains)|(^lcnrepeats)|(^mlmodeltype)|(^mlprog)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^nlayers=)|(^selfcorrect)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch:
                    deadlist.append(arg)
            for i in deadlist: CrossSection_args.remove(i)
            # print(CrossSection_args)
            ML_NEA.parse_api(CrossSection_args)
        elif args.geomopt or args.freq or args.ts or args.irc:
            geomopt.geomoptCls(args.args2pass)
        elif args.AIQM1DFTstar or args.AIQM1DFT or args.AIQM1:
            import AIQM1
            AIQM1.AIQM1Cls(args.args2pass).forward()
        elif args.create4Dmodel or args.use4Dmodel or args.reactionPath or args.estAcc4Dmodel:
            import FourDAtwoItf as FourDAtwoI
            if args.create4Dmodel:
                FourDAtwoI.FourDcls(args.args2pass).create4Dmodel()
            elif args.use4Dmodel:
                FourDAtwoI.FourDcls(args.args2pass).use4Dmodel()
            elif args.reactionPath:
                FourDAtwoI.FourDcls(args.args2pass).reactionPath()
            elif args.estAcc4Dmodel:
                FourDAtwoI.FourDcls(args.args2pass).estAcc4Dmodel()
        elif args.MLTPA:
            import MLTPA
            mltpa=MLTPA.MLTPA(args.args2pass)
            mltpa.predict()
        else:
            self.chooseMLop(args.args2pass)
    
    @classmethod
    def deltaLearn(cls):
        locargs = args.args2pass
        if args.Yb != '': yb = [float(line) for line in open(args.Yb, 'r')]
        if args.YgradB != '': ygradb = [[float(xx) for xx in line.split()] for line in open(args.YgradB, 'r')]
        if args.YgradXYZb != '': ygradxyzb = readXYZgrads(args.YgradXYZb)
        if args.createMLmodel or args.estAccMLmodel:
            # Delta of Y values
            if args.Yb != '':
                ydatname = '%s-%s.dat' % (fnamewoExt(args.Yt), fnamewoExt(args.Yb))
                locargs = addReplaceArg('Yfile', 'Yfile=%s' % ydatname, locargs)
                yt = [float(line) for line in open(args.Yt, 'r')]
                with open(ydatname, 'w') as fcorr:
                    for ii in range(len(yb)):
                        fcorr.writelines('%25.13f\n' % (yt[ii] - yb[ii]))
            # Delta of gradients
            if args.YgradB != '':
                ygradfname = '%s-%s.dat' % (fnamewoExt(args.YgradT), fnamewoExt(args.YgradB))
                locargs = addReplaceArg('YgradFile', 'YgradFile=%s' % ygradfname, locargs)
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
                ygradxyzfname = '%s-%s.dat' % (fnamewoExt(args.YgradXYZt), fnamewoExt(args.YgradXYZb))
                locargs = addReplaceArg('YgradXYZfile', 'YgradXYZfile=%s' % ygradxyzfname, locargs)
                ygradxyzt = readXYZgrads(args.YgradXYZt)
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
        
        if argexist('YestFile=', locargs):
            corr = [float(line) for line in open(args.YestFile, 'r')]
            with open(args.YestT, 'w') as fyestt:
                for ii in range(len(yb)):
                    fyestt.writelines('%25.13f\n' % (yb[ii] + corr[ii]))
        if argexist('YgradEstFile=', locargs):
            corr = [[float(xx) for xx in line.split()] for line in open(args.YgradEstFile, 'r')]
            with open(args.YgradEstT, 'w') as fyestt:
                for ii in range(len(ygradb)):
                    strtmp = ''
                    for jj in range(len(ygradb[ii])):
                        strtmp += '%25.13f   ' % (ygradb[ii][jj] + corr[ii][jj])
                    fyestt.writelines('%s\n' % strtmp)
        if argexist('YgradXYZestFile=', locargs):
            corr = readXYZgrads(args.YgradXYZestFile)
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
                    ydatname = 'deltaRef-%s_layer%d.dat' % (fnamewoExt(yfilename), (nlayer - 1))
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
                locargs = addReplaceArg('Yfile', 'Yfile=%s' % ydatname, locargs)
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
                    print('this repeat is already done')

                else:                    
                    # set args for learning curve
                    lcargs = addReplaceArg('Ntrain','Ntrain='+str(ntrain),locargs)
                    lcargs = addReplaceArg('estAccMLmodel','estAccMLmodel',lcargs)
                    args.Ntrain = ntrain
                    if args.XfileIn:
                        lcargs = addReplaceArg('XfileIn','XfileIn='+os.path.relpath(args.absXfileIn),lcargs)
                    else:
                        lcargs = addReplaceArg('XYZfile','XYZfile='+os.path.relpath(args.absXYZfile),lcargs)
                    if args.Yfile: lcargs = addReplaceArg('Yfile','Yfile='+os.path.relpath(args.absYfile),lcargs)
                    if args.YgradXYZfile: lcargs = addReplaceArg('YgradXYZfile','YgradXYZfile='+os.path.relpath(args.absYgradXYZfile),lcargs)
                    if args.MLprog.lower() in cls.mlatom_alias:
                        os.system('cp ../../../../eq.xyz .> /dev/null 2>&1')
                        lcargs = addReplaceArg('sampling','sampling=user-defined',lcargs)
                        lcargs = addReplaceArg('itrainin','itrainin=../itrain.dat',lcargs)
                        lcargs = addReplaceArg('itestin','itestin=../itest.dat',lcargs)
                        lcargs = addReplaceArg('isubtrainin','isubtrainin=../isubtrain.dat',lcargs)
                        lcargs = addReplaceArg('ivalidatein','ivalidatein=../ivalidate.dat',lcargs)
                        lcargs = addReplaceArg('benchmark','benchmark',lcargs)

                    # estAccMLmodel
                    cls.dataPrepared = False
                    cls.prepareData(lcargs)
                    result = cls.estAccMLmodel(lcargs, shutup=True)
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
                    with open('../../../'+args.MLprog+surfix+'/results.json','w') as f:
                        json.dump(results, f, sort_keys=False, indent=4)
                    with open('result.json','w') as f:
                        json.dump(result, f, sort_keys=False, indent=4)
                    
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
        if args.MLprog.lower() in cls.mlatom_alias and not args.optlines:
            interface_MLatomF.ifMLatomCls.run(locargs)
        else:
            cls.prepareData(locargs)

            # training
            if args.MLprog.lower() in cls.mlatom_alias and not args.optlines:
                pass
            else:
                if args.optlines:
                    t_train, locargs = cls.optraining(locargs,shutup=shutup)
                    
                else:
                    t_train, _, _ = cls.training(locargs)

                if args.MLprog.lower() not in cls.mlatom_alias:
                    cls.estimate(locargs, 'train', shutup=shutup)
            
            
            locargs=addReplaceArg('CVopt','',locargs)

            return [t_train, locargs]

    @classmethod
    def estAccMLmodel(cls, locargs, shutup=False):
        if args.MLprog.lower() in cls.mlatom_alias and not args.optlines:
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
            cvargs=addReplaceArg('sampling','sampling=user-defined',locargs)
            cvargs=addReplaceArg('itestin','itestin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat',cvargs)
            cvargs=addReplaceArg('itrainin','itrainin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_train',cvargs)
            cvargs=addReplaceArg('isubtrainin','isubtrainin=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_subtrain',cvargs)
            cvargs=addReplaceArg('ivalidatein','ivalidatein=%s'%args.iCVtestPrefIn+str(cvid+1)+'.dat_validate',cvargs)
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
        rmsedict, analresults=cls.analyze('CVtest', shutup=shutup)
        if args.Yestfile: os.system(f'cp yest.dat_CVtest {args.Yestfile}')
        if args.YgradXYZestfile: os.system(f'cp gradest.dat_CVtest {args.YgradXYZestfile}')
        print('\n combined CVtest results:')
        print(' ___________________________________________________________')
        print(analresults)
        return t_train,t_pred,rmsedict
        
    @classmethod
    def useMLmodel(cls, locargs, shutup=False):
        starttime = time.time()
        rmsedict = {}
        if args.MLprog.lower() in cls.mlatom_alias:
            _, wallclock, rmsedict, _ = interface_MLatomF.ifMLatomCls.run(locargs,shutup=shutup)
        else:
            cls.prepareData(locargs)
            if args.MLprog.lower() in cls.deepmd_alias:   
                wallclock = interface_DeePMDkit.DeePMDCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.gap_alias:
                wallclock = interface_GAP.GAPCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.ani_alias:
                wallclock = interface_TorchANI.ANICls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.physnet_alias:
                wallclock = interface_PhysNet.PhysNetCls.useMLmodel(locargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.sgdml_alias:
                wallclock = interface_sGDML.sGDMLCls.useMLmodel(locargs, cls.subdatasets)
        endtime = time.time()
        wallclock = endtime - starttime
        if args.benchmark:
            print('\tPrediction time: \t\t\t%f s\n' % wallclock)
        sys.stdout.flush()
        return [wallclock, rmsedict, '']
            
    @classmethod
    def training(cls, locargs, setname='',cvoptid=''):
        if cvoptid: cvopt= '_'+args.iCVoptPrefIn+str(cvoptid)
        else: cvopt = ''
        if args.MLprog.lower() in cls.mlatom_alias:
            mlatom_arg=addReplaceArg('estaccmlmodel','createmlmodel',locargs)
            mlatom_arg=addReplaceArg('MLmodelOut','MLmodelOut=%s'%args.MLmodelOut,mlatom_arg)
            if args.XfileIn:
                mlatom_arg=addReplaceArg('XfileIn','XfileIn=%s'% os.path.relpath(args.absXfileIn) ,mlatom_arg)
            else:
                mlatom_arg=addReplaceArg('xyzfile','xyzfile=%s'% os.path.relpath(args.absXYZfile) ,mlatom_arg)
            if args.Yfile:
                mlatom_arg=addReplaceArg('yfile','yfile=%s'% os.path.relpath(args.absYfile),mlatom_arg)
            if args.YgradXYZfile: 
                mlatom_arg=addReplaceArg('ygradxyzfile','ygradxyzfile=%s'% os.path.relpath(args.absYgradXYZfile),mlatom_arg)

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
            mlatom_arg=addReplaceArg('Ntrain','Ntrain=%s'% ntrain,mlatom_arg)
            mlatom_arg=addReplaceArg('Ntest','',mlatom_arg)
            mlatom_arg=addReplaceArg('Nvalidate','',mlatom_arg)
            mlatom_arg=addReplaceArg('Nsubtrain','',mlatom_arg)
            mlatom_arg=addReplaceArg('sampling','sampling=user-defined',mlatom_arg)
            mlatom_arg=addReplaceArg('itrainin','itrainin='+cls.index_prefix+idx_dict[setname],mlatom_arg)
            mlatom_arg=addReplaceArg('itestin','',mlatom_arg)
            mlatom_arg=addReplaceArg('ivalidatein','',mlatom_arg)
            mlatom_arg=addReplaceArg('isubtrainin','',mlatom_arg)
            mlatom_arg=addReplaceArg('benchmark','benchmark',mlatom_arg)
            mlatom_arg=addReplaceArg('CVopt','',mlatom_arg)
            mlatom_arg=addReplaceArg('iCVoptPrefIn','',mlatom_arg)
            os.system('rm '+ args.mlmodelout+' > /dev/null 2>&1 ')
            print(' Training using MLatomF\n ............................................\n')
            wallclock, _, rmsedic, _ = interface_MLatomF.ifMLatomCls.run(mlatom_arg,shutup=False)
            print(' \n .............................................\n \n\n\tTraining Time: \t\t\t\t%f s\n' % wallclock)
        else:
            print(' Training started\n .............................................')
            starttime = time.time()

            trargs = addReplaceArg('setname', 'setname='+setname+cvopt, locargs)
            if setname:
                if args.XfileIn:
                    trargs = addReplaceArg('XfileIn', 'XfileIn='+cls.index_prefix+'x.dat_'+setname+cvopt, trargs)
                else:
                    trargs = addReplaceArg('XYZFile', 'XYZFile='+cls.index_prefix+'xyz.dat_'+setname+cvopt, trargs)
                if args.Yfile or True:    trargs = addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname+cvopt, trargs)
                if args.YgradXYZfile or True: trargs = addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname+cvopt, trargs)  
            if args.MLprog.lower() in cls.deepmd_alias:
                wallclock = interface_DeePMDkit.DeePMDCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.gap_alias:
                wallclock = interface_GAP.GAPCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.ani_alias:
                wallclock = interface_TorchANI.ANICls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.physnet_alias:
                wallclock = interface_PhysNet.PhysNetCls.createMLmodel(trargs, cls.subdatasets)
            elif args.MLprog.lower() in cls.sgdml_alias:
                wallclock = interface_sGDML.sGDMLCls.createMLmodel(trargs, cls.subdatasets)
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
        optargs = addReplaceArg('setname', 'setname=subtrain', locargs)
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
                optspace['hypara'+str(i)] = hyperopt.hp.loguniform('hypara'+str(i), log(2)*lb, log(2)*hb)
            elif spacetype == 'uniform':
                optspace['hypara'+str(i)] = hyperopt.hp.uniform('hypara'+str(i), lb, hb)
            elif spacetype == 'quniform':
                q = int(place.split('(')[1][:-1].split(',')[-1])
                optspace['hypara'+str(i)] = hyperopt.hp.quniform('hypara'+str(i), int(lb), int(hb), q)
            # see https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions

        # replace hyperparas with marked position slots
        for line in optlines.split():
            optargs =  addReplaceArg(line.split('=')[0], line, optargs) 
        
        bestsofar = inf # store best loss
        bestreport = 'nothing happended yet' # store analyze report of best run so far

        ##############################################################################################
        def getloss(opt): # function that to plug in hyperopt.fmin later
            nonlocal optargs, bestsofar, bestreport, optlines
            optlines_tmp=copy.deepcopy(optlines)
            # get values of hyperparas
            for k, v in opt.items(): 
                optlines_tmp = optlines_tmp.replace(k, str(v), 1)
            # repalce slots with actual values
            for line in optlines_tmp.split():
                optargs =  addReplaceArg(line.split('=')[0], line, optargs) 

            print(' ------------------------------------------------------------------------------\n with hyperparameter(s):\n\n%s\n' % optlines_tmp)
            sys.stdout.flush()
            # train with subtrain set, estimate & analyze with validate set
            if args.MLprog.lower() == 'MLatomF'.lower():
                optargs=addReplaceArg('yestfile','',optargs)
                optargs=addReplaceArg('ygradXYZestfile','',optargs)
            
            if args.CVopt:
                os.system('rm yest.dat_validate_cvopt yest.dat_validate_%s* gradest.dat_validate_* > /dev/null 2>&1'% args.iCVoptPrefIn)
                for i in range(args.NcvOptFolds):
                    if args.MLprog.lower() in cls.mlatom_alias :
                        cls.training(optargs,'subtrain',i+1)
                        _, rmsedict, report = cls.estimate(optargs, 'validate', i+1,shutup=True)
                    else:
                        optargs=addReplaceArg('MLmodelOut','MLmodelOut='+args.MLmodelOut+'_'+args.iCVoptPrefIn+str(i+1),optargs)
                        cls.training(optargs,'subtrain', i+1)
                        cls.estimate(optargs, 'validate',i+1, shutup=True)
                    if args.Yfile:
                        os.system('cat yest.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> yest.dat_validate_cvopt')
                    if args.YgradXYZfile:
                        os.system('cat gradest.dat_validate_'+args.iCVoptPrefIn+str(i+1)+' >> gradest.dat_validate_cvopt')
                rmsedict, report = cls.analyze('validate_cvopt', shutup=shutup)
                rmses = list(rmsedict.values())
                optargs=addReplaceArg('MLmodelOut','MLmodelOut='+args.MLmodelOut,optargs)
                # os.system('rm yest.dat_validate_* gradest.dat_validate_* > /dev/null 2>&1')
            else:   
                cls.training(optargs,'subtrain')
                _, rmsedict_validate, report = cls.estimate(optargs, 'validate', shutup=True)
                # _, rmsedict_subtrain, report = cls.estimate(optargs, 'subtrain', shutup=True)
                valid_rmses=np.array(list(rmsedict_validate.values()))
                # train_rmses=np.array(list(rmsedict_subtrain.values()))
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
                optargs=addReplaceArg('yestfile','YestFile='+args.YestFile,optargs)
                optargs=addReplaceArg('ygradXYZestfile','YgradXYZestFile='+args.YgradXYZestFile,optargs)
            # check if this run performs best
            if loss < bestsofar:
                bestsofar = loss
                bestreport = report
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
            optargs =  addReplaceArg(line.split('=')[0], line, optargs) 
        # if KM then train train set with optargs 
        if args.MLprog.lower() in cls.KMs:            
            optargs=addReplaceArg('CVopt','',optargs)
            print("Training whole training set with optimized hyperparameter(s)...")
            optargs = addReplaceArg('setname', 'setname=train', optargs)
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
                useargs=addReplaceArg('estAccMLmodel','',locargs)
                useargs=addReplaceArg('createmlmodel','useMLmodel',useargs)
                useargs=addReplaceArg('itrainin','',useargs)
                useargs=addReplaceArg('itestin','',useargs)
                useargs=addReplaceArg('isubtrainin','',useargs)
                useargs=addReplaceArg('ivalidatein','',useargs)
                if args.XfileIn:
                    useargs=addReplaceArg('xfilein',f'xfilein={cls.index_prefix}x.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                else:
                    useargs=addReplaceArg('xyzfile',f'xyzfile={cls.index_prefix}xyz.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                useargs=addReplaceArg('mlmodelin','mlmodelin=%s'%args.mlmodelout,useargs)
                if args.Yfile:
                    useargs=addReplaceArg('yfile',f'yfile={cls.index_prefix}y.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                    useargs=addReplaceArg('yestfile','yestfile=yest.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                if args.YgradXYZfile:
                    useargs=addReplaceArg('ygradxyzfile',f'ygradxyzfile={cls.index_prefix}grad.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                    useargs=addReplaceArg('ygradxyzestfile','ygradxyzestfile=gradest.dat_validate_'+args.iCVoptPrefIn+str(cvoptid),useargs)
                setname+='_'+args.iCVoptPrefIn+str(cvoptid)
            else:
                useargs=addReplaceArg('estAccMLmodel','',locargs)
                useargs=addReplaceArg('createmlmodel','estAccMLmodel',useargs)
                useargs=addReplaceArg('Ntrain','Ntrain=%s'% ntrain ,useargs)
                useargs=addReplaceArg('Ntest','Ntest=%s'%ntest,useargs)
                useargs=addReplaceArg('sampling','sampling=user-defined',useargs)
                useargs=addReplaceArg('itrainin','itrainin='+cls.index_prefix+idx_dict[setdict[setname][2]],useargs)
                useargs=addReplaceArg('itestin','itestin='+cls.index_prefix+idx_dict[setname],useargs)
                useargs=addReplaceArg('mlmodelin','mlmodelin=%s'%args.mlmodelout,useargs)
                useargs=addReplaceArg('yestfile','',useargs)
                useargs=addReplaceArg('CVopt','',useargs)
                useargs=addReplaceArg('mlmodelout','',useargs)
                useargs=addReplaceArg('iCVoptPrefIn','',useargs)
                useargs=addReplaceArg('ygradXYZestfile','',useargs)
        else:
            if cvoptid: setname+='_'+args.iCVoptPrefIn+str(cvoptid)
            useargs = addReplaceArg('setname', 'setname='+setname, locargs)
            if setname:
                if args.XfileIn:
                    useargs = addReplaceArg('XfileIn', 'XfileIn='+cls.index_prefix+'x.dat_'+setname, useargs)
                else:
                    useargs = addReplaceArg('XYZFile', 'XYZFile='+cls.index_prefix+'xyz.dat_'+setname, useargs)
                if args.Yfile:    useargs = addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname, useargs)
                if args.YgradXYZfile: useargs = addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname, useargs)

        print(' %s:\n\t' % setname)
        t_pred, rmsedict, report= cls.useMLmodel(useargs, shutup=shutup)
        if args.MLprog.lower() != 'MLatomF'.lower() or cvoptid:
            rmsedict, report = cls.analyze(setname, shutup=shutup)
        if not shutup: print(report)
        else: print('\t'+'\n\t'.join([k+':\t\t\t\t\t'+str(v) for k, v in rmsedict.items()])+'\n')
        return [t_pred, rmsedict, report]

    @classmethod
    def analyze(cls, setname, shutup=False):
        analyzeargs = ['analyze']
        if setname:
            if args.Yfile:
                analyzeargs = addReplaceArg('Yfile', 'Yfile='+cls.index_prefix+'y.dat_'+setname, analyzeargs)
                analyzeargs = addReplaceArg('YestFile', 'YestFile=yest.dat_'+setname, analyzeargs)
            if args.YgradXYZfile:
                analyzeargs = addReplaceArg('YgradXYZfile', 'YgradXYZfile='+cls.index_prefix+'grad.dat_'+setname, analyzeargs)
                analyzeargs = addReplaceArg('YgradXYZestFile', 'YgradXYZestFile=gradest.dat_'+setname, analyzeargs)

        analout = StringIO() # variable with disgusting name
        with  redirect_stdout(analout):
            interface_MLatomF.ifMLatomCls.run(analyzeargs)
        analresults = analout.getvalue()
        rmses=[float(x.split()[2]) for x in re.findall('RMSE =.+',analresults)]
        rmsedict = {}
        if args.Yfile:
            rmsedict['eRMSE'] = rmses[0]
        if args.YgradXYZfile:
            rmsedict['fRMSE'] = rmses[-1]
        return [rmsedict, analresults]

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
        if args.MLprog in cls.model_needs_subvalid or (args.optlines and not args.CVopt): 
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
        smplargs   = addReplaceArg('molDescriptor','molDescriptor=CM',smplargs)
        pwd=os.getcwd()
        if args.XfileIn:
            smplargs   = addReplaceArg('XfileIn', 'XfileIn=%s' % os.path.relpath(args.absXfileIn), smplargs)
        else:
            smplargs   = addReplaceArg('XYZfile', 'XYZfile=%s' % os.path.relpath(args.absXYZfile), smplargs)
        if args.sampling: smplargs   = addReplaceArg('sampling', 'sampling=%s' % args.sampling, smplargs)
        ldic={'smplargs':smplargs}
        for i in subdatasets:
            exec('if args.N'+i.lower()+': smplargs=addReplaceArg("N'+i.lower()+'","N'+i.lower()+'=%s" % args.N'+i.lower()+', smplargs)',globals(),ldic)
        for i in ['Train','Test','Subtrain','Validate']:
            exec('smplargs = addReplaceArg("i'+i.lower()+'Out","i'+i.lower()+'Out='+cls.index_prefix+'i'+i.lower()+'.dat",smplargs)',globals(),ldic)
        smplargs=ldic['smplargs']
        if args.CVopt:
            if not args.learningCurve:
                if args.iTrainIn:
                    smplargs = addReplaceArg('itrainout','iTrainIn=%s'%args.iTrainIn,smplargs)
                if args.iTestIn:
                    smplargs = addReplaceArg('itestout','iTestIn=%s'%args.iTestIn,smplargs)
            smplargs = addReplaceArg('CVopt','CVopt',smplargs)
            smplargs = addReplaceArg('iCVoptPrefOut','iCVoptPrefOut=%s'%(cls.index_prefix+args.iCVoptPrefOut),smplargs)
            smplargs = addReplaceArg('NcvOptFolds','NcvOptFolds=%d'%args.NcvOptFolds,smplargs)
            args.parse_input_content(['iCVoptPrefIn=%s'%args.iCVoptPrefOut])
        if args.CVtest:
            smplargs = addReplaceArg('CVtest','CVtest',smplargs)
            smplargs = addReplaceArg('iCVtestPrefOut','iCVtestPrefOut=%s'%(cls.index_prefix+args.iCVtestPrefOut),smplargs)
            smplargs = addReplaceArg('NcvTestFolds','NcvTestFolds=%d'%args.NcvTestFolds,smplargs)
            args.parse_input_content(['iCVtestPrefIn=%s'%args.iCVtestPrefOut])
        interface_MLatomF.ifMLatomCls.run(smplargs, shutup=True)
        for i in ['Train','Test','Subtrain','Validate']:
            exec('if not args.i'+i+'in: args.i'+i+'in="i'+i.lower()+'.dat"',globals(),ldic)
         
def fnamewoExt(fullfname):
    fname = os.path.basename(fullfname)
    fname = os.path.splitext(fname)[0]
    return fname

def argexist(argname, largs):
    for iarg in range(len(largs)):
        arg = largs[iarg]
        if argname.lower() in arg.lower():
            return True
    else:
        return False

def addReplaceArg(argname, newarg, originalargs):
    finalargs = copy.deepcopy(originalargs)
    for iarg in range(len(finalargs)):
        arg = finalargs[iarg]
        if argname.lower() == arg.split('=')[0].lower():
            if newarg:
                finalargs[iarg] = newarg
            else:
                del finalargs[iarg]
            break            
    else:
        finalargs.append(newarg)
    return finalargs

def readXYZgrads(fname):
    ygradxyz = []
    with open(fname, 'r') as ff:
        Nlines = 0
        Natoms = 0
        ltmp = []
        for line in ff:
            Nlines += 1
            if Nlines == 1:
                Natoms = int(line)
                ltmp = []
            elif Nlines > 2:
                ltmp.append([float(xx) for xx in line.split()])
                if Nlines == 2 + Natoms:
                    Nlines = 0
                    ygradxyz.append(ltmp)
    return ygradxyz

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.task_list = [
            'XYZ2X', 'analyze', 'sample', 'slicedata', 'sampleFromSlices', 'mergeSlices',
            'useMLmodel', 'createMLmodel', 'estAccMLmodel', 'learningCurve', 
            'crossSection', # Interfaces
            'deltaLearn', 'selfCorrect',
            'AIQM1DFTstar', 'AIQM1DFT', 'AIQM1',
            'geomopt', 'freq', 'ts', 'irc', 'CVopt','CVtest',
            'create4Dmodel','use4Dmodel','reactionPath','estAcc4Dmodel',
            'MLTPA',
            'benchmark'
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
            'absXfileIn', 'absXYZfile', 'absYfile', 'absYgradXYZfile',
            'trajsList',
            'Nuse', 'Ntrain', 'Nsubtrain', 'Nvalidate', 'Ntest', 'iTrainIn', 'iTestIn', 'iSubtrainIn', 'iValidateIn', 'sampling', 'MLmodelIn', 'MLmodelOut',
            'molDescriptor', 'kernel',
            'iCVtestPrefIn',
            'iCVoptPrefIn'
            ],
            ""
        )       
        self.add_dict_args({'MLmodelType': None, 'nthreads': None})
        self.set_keyword_alias('crossSection', ['ML-NEA', 'ML_NEA', 'crossSection', 'cross-section', 'cross_section','MLNEA'])
        self.set_keyword_alias('AIQM1DFTstar', ['AIQM1@DFT*'])
        self.set_keyword_alias('AIQM1DFT', ['AIQM1@DFT'])
        self.set_keyword_alias('usemlmodel mlmodeltype=ani1ccx', ['ANI-1ccx'])
        self.set_keyword_alias('usemlmodel mlmodeltype=ani1x', ['ANI-1x'])
        self.set_keyword_alias('usemlmodel mlmodeltype=ani2x', ['ANI-2x'])
        self.args2pass = []
        self.parse_input_content([
            'hyperopt.max_evals=8',
            'hyperopt.algorithm=tpe',
            'hyperopt.losstype=geomean',
            'hyperopt.w_y=1',
            'hyperopt.w_ygrad=1',
            'hyperopt.points_to_evaluate=0'
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
        if self.selfCorrect and (argexist('iTrainOut=', self.args2pass) or argexist('iTestOut=', self.args2pass) or argexist('iSubtrainOut=', self.args2pass) or argexist('iValidateOut=', self.args2pass)):
            stopper.stopMLatom('Indices of subsets cannot be saved for self-correction')
        
        if self.mlmodeltype=='krr-cm':
            self.parse_input_content([
            'molDescriptor=CM'
            ])
        if self.MLprog.lower() in MLtasksCls.mlatom_alias and not self.MLmodelOut:
            self.MLmodelOut='MLatom.unf'

if __name__ == '__main__': 
    print(__doc__)
    MLtasks()
