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
        result = {}
        t_train=0
        t_descr=0
        t_hyperopt=0
        t_finaltrain=0
        t_pred=0
        t_wc=0
        Ntrain=None
        Ntest=None
        yflag=0
        gflag=0
        uflag=0
        pflag=0
        nflag=0
        deadlist=[]
        output=''
        for arg in argsMLatomF:
            flagmatch = re.search('(^nthreads)|(^hyperopt)|(^setname=)|(^learningcurve$)|(^molecularDynamics)|(^lcntrains)|(^lcnrepeats)|(^mlmodeltype)|(^mlmodel2type)|(^mlmodel2in)|(^opttrajxyz)|(^opttrajh5)|(^mlprog)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^ygradb=)|(^ygradt=)|(^ygradestt=)|(^ygradxyzb=)|(^ygradxyzt=)|(^ygradxyzestt=)|(^nlayers=)|(^selfcorrect)|(^optBetas)|(^dummy)|(^customBetas)|(^reduceBetas)|(^initBetas)|(^$)|(^geomopt)|(^freq)|(^ts)|(^irc)|(^ase.fmax=)|(^ase.steps=)|(^ase.optimizer=)|(^ase.linear=)|(^ase.symmetrynumber=)|(^qmprog=)|(^optprog=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if flagmatch:
                deadlist.append(arg)
            if re.search('^mlmodelin=',arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE):
                uflag=1
        for i in deadlist: argsMLatomF.remove(i)
        if not shutup: print('> '+mlatomfbin+' '+' '.join(argsMLatomF))
        proc = subprocess.Popen([mlatomfbin] + argsMLatomF, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwdpath, universal_newlines=True)
        # proc.wait()
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
                elif 'Analysis for values' in readable: 
                    yflag = 1
                    gflag = 0
                    result["values"] = {}
                elif 'Analysis for gradients in XYZ coordinates' in readable:
                    gflag = 1
                    yflag = 0
                    result["gradients"] = {}
                elif 'largest positive outlier' in readable:
                    pflag = 1
                    nflog = 0
                elif 'largest negative outlier' in readable:
                    nflag = 1
                    pflag = 0
                elif 'MAE ='in readable:
                    if yflag:
                        result['values']['mae'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['mae'] = float(readable.split()[2])
                elif ' MSE ='in readable:
                    if yflag:
                        result['values']['mse'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['mse'] = float(readable.split()[2])
                elif 'RMSE ='in readable:
                    if yflag:
                        result['values']['rmse'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['rmse'] = float(readable.split()[2])
                elif 'mean(Y) ='in readable:
                    if yflag:
                        result['values']['mean_y'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['mean_y'] = float(readable.split()[2])
                elif 'mean(Yest) = 'in readable:
                    if yflag:
                        result['values']['mean_yest'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['mean_yest'] = float(readable.split()[2])
                elif 'correlation coefficient ='in readable:
                    if yflag:
                        result['values']['corr_coef'] = float(readable.split()[3])
                    if gflag:
                        result['gradients']['corr_coef'] = float(readable.split()[3])
                elif 'a ='in readable:
                    if yflag:
                        result['values']['a'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['a'] = float(readable.split()[2])
                elif 'b ='in readable:
                    if yflag:
                        result['values']['b'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['b'] = float(readable.split()[2])
                elif 'R^2 ='in readable:
                    if yflag:
                        result['values']['r_squared'] = float(readable.split()[2])
                    if gflag:
                        result['gradients']['r_squared'] = float(readable.split()[2])
                elif 'error ='in readable:
                    if yflag:
                        if pflag:
                            result['values']['pos_off'] = float(readable.split()[2])
                        if nflag:
                            result['values']['neg_off'] = float(readable.split()[2])
                    if gflag:
                        if pflag:
                            result['gradients']['pos_off'] = float(readable.split()[2])
                        if nflag:
                            result['gradients']['neg_off'] = float(readable.split()[2])
                elif 'estimated value ='in readable:
                    if yflag:
                        if pflag:
                            result['values']['pos_off_est'] = float(readable.split()[3])
                        if nflag:
                            result['values']['neg_off_est'] = float(readable.split()[3])
                    if gflag:
                        if pflag:
                            result['gradients']['pos_off_est'] = float(readable.split()[3])
                        if nflag:
                            result['gradients']['neg_off_est'] = float(readable.split()[3])
                elif 'reference value ='in readable:
                    if yflag:
                        if pflag:
                            result['values']['pos_off_ref'] = float(readable.split()[3])
                        if nflag:
                            result['values']['neg_off_ref'] = float(readable.split()[3])
                    if gflag:
                        if pflag:
                            result['gradients']['pos_off_ref'] = float(readable.split()[3])
                        if nflag:
                            result['gradients']['neg_off_ref'] = float(readable.split()[3])
                elif 'index ='in readable:
                    if yflag:
                        if pflag:
                            result['values']['pos_off_idx'] = float(readable.split()[2])
                        if nflag:
                            result['values']['neg_off'] = float(readable.split()[2])
                    if gflag:
                        if pflag:
                            result['gradients']['pos_off_idx'] = float(readable.split()[2])
                        if nflag:
                            result['gradients']['neg_off_idx'] = float(readable.split()[2])
                if not shutup: print(readable.replace('\n',''))
                #print(readable.replace('\n',''))
                sys.stdout.flush()
            except:
                pass
            
        proc.stdout.close()
        if Ntrain != None:
            t_train = t_hyperopt + t_finaltrain + t_descr
        if uflag or (Ntrain == None and Ntest == None):
            t_pred = t_wc
        if Ntrain != None and Ntest != None:
            t_train -= t_descr / (Ntrain + Ntest) * Ntest
            t_pred  += t_descr / (Ntrain + Ntest) * Ntest
        
        result["t_train"] = t_train
        result["t_pred"] = t_pred
        return result

def printHelp():
    proc = subprocess.Popen([mlatomfbin] + ['help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    for line in iter(proc.stdout.readline, b''):
        readable = line.decode('ascii')
        print(readable.rstrip())

if __name__ == '__main__':
    print(__doc__)
    ifMLatomCls.run(sys.argv[1:])
