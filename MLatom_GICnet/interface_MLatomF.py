#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! MLatomF_interface: Interface between MLatomF and MLatom.py                ! 
  ! Implementations by: Pavlo O. Dral and Fuchun Ge                           ! 
  !---------------------------------------------------------------------------! 
'''

import sys, os, subprocess, re
mlatomdir=os.path.dirname(__file__)
mlatomfbin="%s/MLatomF" % mlatomdir

class ifMLatomCls(object):           
    @classmethod
    def run(cls, argsMLatomF, shutup=False, cwdpath='.'):
        t_train=0
        t_descr=0
        t_hyperopt=0
        t_finaltrain=0
        t_pred=0
        t_wc=0
        Ntrain=None
        Ntest=None
        rmsedict={}
        yflag=0
        gflag=0
        pflag=0
        deadlist=[]
        output=''
        for arg in argsMLatomF:
            flagmatch = re.search('(^nthreads)|(^hyperopt)|(^setname=)|(^learningcurve$)|(^molecularDynamics)|(^lcntrains)|(^lcnrepeats)|(^mlmodeltype)|(^mlmodel2type)|(^mlmodel2in)|(^opttrajxyz)|(^opttrajh5)|(^mlprog)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^ygradb=)|(^ygradt=)|(^ygradestt=)|(^ygradxyzb=)|(^ygradxyzt=)|(^ygradxyzestt=)|(^nlayers=)|(^selfcorrect)|(^optBetas)|(^dummy)|(^customBetas)|(^reduceBetas)|(^initBetas)|(^$)|(^geomopt)|(^freq)|(^ts)|(^irc)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if flagmatch:
                deadlist.append(arg)
            if re.search('^mlmodelin=',arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE):
                pflag=1
        for i in deadlist: argsMLatomF.remove(i)
        print('> '+mlatomfbin+' '+' '.join(argsMLatomF))
        proc = subprocess.Popen([mlatomfbin] + argsMLatomF, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwdpath, universal_newlines=True)
        for readable in proc.stdout:
            output+=readable
            try:
                # readable = line.decode('ascii')
                if 'Descriptor generation time' in readable:
                    t_descr = float(readable.split()[-2])
                if 'Hyperparameter optimization time:' in readable:
                    t_hyperopt = float(readable.split()[-2])
                if 'Training time' in readable:
                    t_finaltrain = float(readable.split()[-2])
                elif 'Training time' in readable:
                    t_train += float(readable.split()[2])
                elif 'Validating time' in readable:
                    t_train += float(readable.split()[2])
                elif 'Test time' in readable:
                    t_pred = float(readable.split()[-2])
                if 'Wall-clock time:' in readable and not 'min' in readable:
                    t_wc = float(readable.split()[-2])  
                if 'Statistical analysis for' in readable and 'entries in the training set' in readable and Ntrain == None:
                    Ntrain = float(readable.split()[3])  
                if 'Statistical analysis for' in readable and 'entries in the test set' in readable and Ntest == None:
                    Ntest = float(readable.split()[3])  
                elif 'Prediction time' in readable:
                    t_pred += float(readable.split()[2])
                elif 'Analysis for values' in readable: yflag=1
                elif 'Analysis for gradients in XYZ coordinates' in readable: gflag=1
                elif 'RMSE ='in readable:
                    if yflag:
                        rmsedict['eRMSE'] = float(readable.split()[2])
                        yflag = 0
                    if gflag:
                        rmsedict['fRMSE'] = float(readable.split()[2])
                        gflag = 0
                # if not shutup or '<!>' in readable: print(readable.replace('\n',''))
                print(readable.replace('\n',''))
                sys.stdout.flush()
            except:
                pass
            
        proc.stdout.close()
        if Ntrain != None:
            t_train = t_hyperopt + t_finaltrain + t_descr
        if pflag or (Ntrain == None and Ntest == None):
            t_pred = t_wc
        if Ntrain != None and Ntest != None:
            t_train -= t_descr / (Ntrain + Ntest) * Ntest
            t_pred  += t_descr / (Ntrain + Ntest) * Ntest
            
        return [t_train, t_pred, rmsedict, output]

def printHelp():
    proc = subprocess.Popen([mlatomfbin] + ['help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in iter(proc.stdout.readline, b''):
        readable = line.decode('ascii')
        print(readable.rstrip())

if __name__ == '__main__':
    print(__doc__)
    ifMLatomCls.run(sys.argv[1:])
