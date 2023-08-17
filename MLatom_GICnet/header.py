import json
import sys,os

def printHeader(args):
    with open(os.path.dirname(__file__)+'/ref.json','r') as f:
        refs = json.load(f)
    refItems = {}

    # ML tasks
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
    if (args.MLprog.lower() in [ 'gap', 'gap_fit', 'gapfit'] or args.MLmodelType.lower() == 'gap-soap'):
        refItems['GAP model'] = refs['GAP']
        if (args.MLmodelType.lower() == 'gap-soap' or args.MLmodelType.lower() == ''):
            refItems['SOAP descriptor'] = refs['SOAP']
    
    if args.crossSection:  
        refItems['ML-NEA']  = refs['ML-NEA']
        refItems['NEA']  = refs['NEA']
        refItems['Wigner distribution']  = refs['Wigner']
        refItems['NEWTON-X']  = refs['NEWTON-X']

    if args.MLTPA:
        refItems['Machine learning-two photon absorption']  = refs['MLTPA']

    if 'hyperopt' in ' '.join(args.args2pass).lower():
        refItems['hyperopt program'] = refs['hyperopt']
        refItems['Tree-Structured Parzen Estimator algorithm'] = refs['TPE']
        
    if 'aiqm1' in ' '.join(args.args2pass).lower():
        refItems['AIQM1']  = refs['AIQM1']
        refItems['ODM2']  = refs['ODM2']
        refItems['MNDO program']  = refs['MNDOprog']
        refItems['D4']  = refs['D4']
        refItems['D4 program']  = refs['D4prog']
        refItems['ANI model'] = refs['ANI']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.freq:
            refItems['Uncertainty quantification of AIQM1 heats of formation'] = refs['HoF-ANI1ccx']
        
    if 'ani1ccx' in ' '.join(args.args2pass).lower():
        refItems['ANI-1ccx'] = refs['ANI-1ccx']
        refItems['TorchANI program'] = refs['TorchANI']
        if args.freq:
            refItems['ANI-1ccx enthalpies of formation'] = refs['HoF-ANI1ccx']
            
    if 'ani-1x' in ' '.join(args.args2pass).lower():
        refItems['ANI-1x'] = refs['ANI-1x']
        refItems['TorchANI program'] = refs['TorchANI']
        
    if 'ani-2x' in ' '.join(args.args2pass).lower():
        refItems['ANI-2x'] = refs['ANI-2x']
        refItems['TorchANI program'] = refs['TorchANI']
    
    if (args.geomopt or args.freq):# or args.ts or args.irc):
        if 'optprog=ase' in ' '.join(args.args2pass).lower():
            refItems['Atomic simulation environment (ASE)']  = refs['ASE']
        elif 'optprog=gaussian' in ' '.join(args.args2pass).lower():
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
