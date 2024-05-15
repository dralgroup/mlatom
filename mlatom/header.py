import json
import sys,os
from . import models

def printHeader(args):
    with open(os.path.dirname(__file__)+'/ref.json','r', encoding='utf-8') as f:
        refs = json.load(f)
    refItems = {}
    # ML tasks
    if args.MD: refItems['MD']  = refs['MD']
    if args.mlqd: refItems['MLQD program']  = refs['MLQDprog']
    if args.deltaLearn:  refItems['delta-learning']  = refs['JCTC2015_DeltaLearning']
    if args.molDescriptor.upper() == 'CM': refItems['Coulomb matrix'] = refs['CM']
    if args.selfCorrect: refItems['cross-section'] = refs['JCP2017_MLCH3Cl']
    if (args.MLprog.lower() == 'mlatomf' or args.MLprog == '') and (args.MLmodelType.lower() == 'kreg' or not args.MLmodelType) and (args.createMLmodel or args.estAccMLmodel or args.XYZ2X):
        if args.XYZfile != '' and (args.molDescriptor.upper() == 'RE' or not args.molDescriptor):
            if args.kernel.lower() == 'gaussian' or args.kernel == '' and not args.XYZ2X:
                refItems['KREG model'] = refs['JCP2017_MLCH3Cl']
                if args.YgradXYZfile:
                     refItems['KREG model'] += '\n\n' + refs['KREGgrads']
            else:
                refItems['RE descriptor'] = refs['JCP2017_MLCH3Cl']
    if (args.MLprog.lower() == 'sgdml' or args.MLmodelType.lower() == 'sgdml' or args.MLmodelType.lower() == 'gdml'):
        refItems['sGDML program'] = refs['sGDMLprog']
        if (args.MLmodelType.lower() == 'sgdml' or args.MLmodelType.lower() == ''):
            refItems['sGDML model'] = refs['sGDMLmodel']
        if (args.MLmodelType.lower() == 'gdml'):
            refItems['GDML model'] = refs['sGDMLmodel']
    if (args.MLprog.lower() in ['torchani','ani'] or args.MLmodelType.lower() == 'ani' or args.MLmodelType.lower() == 'ani-aev'):
        refItems['TorchANI program'] = refs['TorchANI']
        if (args.MLmodelType.lower() == 'ani' or args.MLmodelType.lower() == 'ani-aev' or args.MLmodelType.lower() == ''):
            refItems['ANI model'] = refs['ANI']
    if (args.MLprog.lower() in ['dp','deepmd','deepmd-kit'] or args.MLmodelType.lower() == 'dpmd' or args.MLmodelType.lower() == 'deeppot-se'):
        refItems['DeePMD-kit program'] = refs['DeePMD-kit']
        if (args.MLmodelType.lower() == 'deeppot-se' or args.MLmodelType.lower() == ''):
            refItems['DeepPot-SE model'] = refs['DeepPot-SE']
        if (args.MLmodelType.lower() == 'dpmd'):
            refItems['DPMD model'] = refs['DPMD']
    if (args.MLprog.lower() == 'physnet' or args.MLmodelType.lower() == 'physnet'):
        refItems['PhysNet model & program'] = refs['PhysNet']
    if (args.MLprog.lower() == 'mace' or args.MLmodelType.lower() == 'mace'):
        refItems['MACE model & program'] = refs['MACE']
    if (args.MLprog.lower() in [ 'gap', 'gap_fit', 'gapfit'] or args.MLmodelType.lower() == 'gap-soap'):
        refItems['GAP model'] = refs['GAP']
        if (args.MLmodelType.lower() == 'gap-soap' or args.MLmodelType.lower() == ''):
            refItems['SOAP descriptor'] = refs['SOAP']
    
    if 'qmprog=pyscf' in ' '.join(args.args2pass).lower():
        refItems['PySCF']  = refs['PySCF']
    elif 'qmprog=gaussian' in ' '.join(args.args2pass).lower():
        refItems['Gaussian program']  = refs['Gaussian']
    elif 'qmprog=sparrow' in ' '.join(args.args2pass).lower():
            refItems['Sparrow program']  = refs['Sparrow']
    elif 'qmprog=mndo' in ' '.join(args.args2pass).lower():
            refItems['MNDO program']  = refs['MNDOprog']
    elif 'qmprog=xtb' in ' '.join(args.args2pass).lower():
        refItems['xtb program']  = refs['xtb']
    elif 'qmprog=orca' in ' '.join(args.args2pass).lower():
        refItems['ORCA program']  = refs['ORCA']
            
    if args.crossSection:  
        refItems['ML-NEA']  = refs['ML-NEA']
        refItems['NEA']  = refs['NEA']
        refItems['Wigner distribution']  = refs['Wigner']
        refItems['NEWTON-X']  = refs['NEWTON-X']

    if args.MLTPA:
        refItems['Machine learning-two photon absorption']  = refs['MLTPA']
        refItems['RDKit']  = refs['RDKit']  

    if 'hyperopt' in ' '.join(args.args2pass).lower():
        refItems['hyperopt program'] = refs['hyperopt']
        refItems['Tree-Structured Parzen Estimator algorithm'] = refs['TPE']
        
    if args.method in models.methods.methods_map['aiqm1']:
        refItems['AIQM1']  = refs['AIQM1']
        refItems['ODM2']  = refs['ODM2']
        if 'qmprog=sparrow' in ' '.join(args.args2pass).lower() and 'sparrowbin' in os.environ:
            refItems['Sparrow program']  = refs['Sparrow']
        elif 'mndobin' in os.environ:
            refItems['MNDO program']  = refs['MNDOprog']
        elif 'sparrowbin' in os.environ:
            refItems['Sparrow program']  = refs['Sparrow']
        refItems['D4']  = refs['D4']
        refItems['D4 program']  = refs['D4prog']
        refItems['ANI model'] = refs['ANI']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.freq:
            refItems['Uncertainty quantification of AIQM1 heats of formation'] = refs['HoF-ANI1ccx']
    
    if args.method in models.methods.methods_map['mndo'] + models.methods.methods_map['sparrow']:
        if args.ODM2:
            refItems['ODM2']  = refs['ODM2']
        elif args.ODM2star:
            refItems['ODM2']  = refs['ODM2']
            refItems['ODM2*']  = refs['AIQM1']
        if 'qmprog=sparrow' in ' '.join(args.args2pass).lower() and 'sparrowbin' in os.environ:
            refItems['Sparrow program']  = refs['Sparrow']
        elif 'mndobin' in os.environ:
            refItems['MNDO program']  = refs['MNDOprog']
        elif 'sparrowbin' in os.environ:
            refItems['Sparrow program']  = refs['Sparrow']

    if args.CCSDTstarCBS:
        refItems['CCSD(T)*/CBS']  = refs['ANI-1ccx']
        refItems['ORCA program']  = refs['ORCA']
        
    if args.gfn2xtb:
        refItems['GFN2-xTB']  = refs['GFN2-xTB']
        refItems['xtb program']  = refs['xtb']
        
    if args.ani1ccx:
        refItems['ANI-1ccx'] = refs['ANI-1ccx']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.freq:
            refItems['ANI-1ccx enthalpies of formation'] = refs['HoF-ANI1ccx']
            
    if args.ani1x or args.ani1xd4:
        refItems['ANI-1x'] = refs['ANI-1x']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.ani1xd4:
            refItems['D4']  = refs['D4']
            refItems['D4 program']  = refs['D4prog']
        
    if args.ani2x or args.ani2xd4:
        refItems['ANI-2x'] = refs['ANI-2x']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.ani2xd4:
            refItems['D4']  = refs['D4']
            refItems['D4 program']  = refs['D4prog']

    if args.method:
        if 'DM21' in args.method.upper():
            refItems['DM21'] = refs['DM21']
    
    if args.ani1xnr:
        refItems['ANI-1xnr'] = refs['ANI-1xnr']
    if args.aimnet2atb973c:
        refItems['AIMNet2@B973c'] = refs['AIMNet2']
    if args.aimnet2atwb97md3:
        refItems['AIMNet2@wb97M-D3'] = refs['AIMNet2']

    
    if args.geomopt:
        if 'optprog=ase' in ' '.join(args.args2pass).lower():
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
        elif 'optprog=geometric' in ' '.join(args.args2pass).lower():
            refItems['geomeTRIC program']  = refs['geometric']
        elif 'optprog=gaussian' in ' '.join(args.args2pass).lower():
            refItems['Gaussian program']  = refs['Gaussian']
        elif args.freq and 'optprog=pyscf' in ' '.join(args.args2pass).lower():
            refItems['PySCF program']  = refs['PySCF']
        elif "GAUSS_EXEDIR" in os.environ:
            refItems['Gaussian program']  = refs['Gaussian']
        else:
            try:
                import ase
                refItems['Atomic simulation environment (ASE)']  = refs['ASE']
            except:
                try: 
                    import scipy
                    refItems['SciPy']  = refs['SciPy']
                except:
                    pass
                    
    if args.freq:
        if not args.freqProg:
            args.freqProg = args.optProg            
        if 'freqprog=ase' in ' '.join(args.args2pass).lower():
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
        elif 'freqprog=gaussian' in ' '.join(args.args2pass).lower():
            refItems['Gaussian program']  = refs['Gaussian']
        elif args.freq and 'freqprog=pyscf' in ' '.join(args.args2pass).lower():
            refItems['PySCF program']  = refs['PySCF']
        elif "GAUSS_EXEDIR" in os.environ:
            refItems['Gaussian program']  = refs['Gaussian']
        else:
            try:
                import pyscf
                refItems['PySCF program']  = refs['PySCF']
            except:
                try: 
                    import ase
                    refItems['Atomic simulation environment (ASE)']  = refs['ASE']
                except:
                    pass
                    
    if args.ts:
        if 'optprog=ase' in ' '.join(args.args2pass).lower():
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
        if 'optprog=geometric' in ' '.join(args.args2pass).lower():
            refItems['geomeTRIC program']  = refs['geometric']
        elif 'optprog=gaussian' in ' '.join(args.args2pass).lower():
            refItems['Gaussian program']  = refs['Gaussian']
        elif "GAUSS_EXEDIR" in os.environ:
            refItems['Gaussian program']  = refs['Gaussian']
        else:
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
                
    if args.irc:
        if 'optprog=ase' in ' '.join(args.args2pass).lower():
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
        elif 'optprog=gaussian' in ' '.join(args.args2pass).lower():
            refItems['Gaussian program']  = refs['Gaussian']
        else:
            refItems['Gaussian program']  = refs['Gaussian']
    
    if 'periodkernel' in ' '.join(args.args2pass).lower():
        refItems['Periodic kernel'] = refs['4D'] + '\n\n' + refs['sklpaper']

    if 'decaykernel' in ' '.join(args.args2pass).lower():
        refItems['Decaying periodic kernel'] = refs['MLQDSBbench'] + '\n\n' + refs['sklpaper'] + '\n\n' + refs['RasmussenGP']

    # Data set tasks
    if args.sampleFromSlices or args.mergeSlices or args.slice: refItems['slicing'] = refs['JCP2017_MLCH3Cl']
    if args.sampling == 'structure-based': refItems['structure-based sampling'] = refs['JCP2017_MLCH3Cl']

    if len(refItems.keys()) > 0:
        print('\n %s \n' % ('*'*78))
        print(' You are going to use feature(s) listed below. \n Please cite corresponding work(s) in your paper:')
        for key in refItems.keys():
            print('\n %s:'%key)
            refstrs = refItems[key].split('\n')
            for refstr in refstrs:
                print('   %s' % refstr)
        print('\n %s ' % ('*'*78))
