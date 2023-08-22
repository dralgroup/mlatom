import scipy
import numpy as np
from numpy import linalg as LA
import subprocess
from scipy.optimize import dual_annealing
import re
import sys,os
import hyperopt
import time
from math import log


#best=2.0716697215712
best=3.6107862658531
sigma=3449.08985798006916
lamb=0.00000163304955
n_pass=0
n_total=2
time_start=time.time()
while n_pass < n_total:
    n_pass+=1
    with open('betas.dat') as f:
        betas=[]
        for line in f:
            betas.append(float(line))
    for i in range(36):
        old_beta=betas[i]
        #betas[i]='hyperopt.uniform(0,400)'
        betas[i]='hyperopt.uniform(0,141106)'
        optbetas=[str(i) for i in betas]
        print('betas=%s'%','.join(optbetas))
        sys.stdout.flush()
        os.system("sed -e 's/SLOT/%s/' -e 's/LOTS/%s/' -e 's/OLDS/%s/g' -e 's/OLDL/%s/g' en.tmp > en.inp" % (','.join(optbetas),str(old_beta),str(sigma),str(lamb)))
        result= subprocess.check_output("$mlatom en.inp |tee output", shell=True)
        loss=min([float(i) for i in re.findall('(?<=Loss:\t\t\t\t\t).+(?=\n)',result.decode('UTF-8'))])
        rmse=[float(i) for i in re.findall('(?<=RMSE =).+(?=\n)',result.decode('UTF-8'))]
        print('Validation Loss: %f, Test RMSE: %f' % (loss,rmse[-1]))
        sys.stdout.flush()
        result=subprocess.check_output("cat hyperopt.inp | grep betas",shell=True).decode("UTF-8")
        sigma=float(subprocess.check_output("cat hyperopt.inp | grep sigma",shell=True).decode("UTF-8")[6:])
        lamb=float(subprocess.check_output("cat hyperopt.inp | grep lambda",shell=True).decode("UTF-8")[7:])
        new_beta=float(result[7:].split(',')[i])
        ##### CHEATING #####
        if rmse[-1]<best:
            betas[i]=new_beta
            best=rmse[-1]
        else: betas[i]=old_beta
        ##### NO CHEATING #####
        # betas[i]=new_beta

    print(betas)
    os.system("sed -e 's/SLOT/%s/' -e 's/LOTS/1/' -e 's/hyperopt.loguniform(-1,10)/%s/' -e 's/hyperopt.loguniform(-45,-4)/%s/' en.tmp > en.inp" % (','.join([str(i) for i in betas]),str(sigma),str(lamb)))
    rmse= subprocess.check_output("$mlatom en.inp |tee output| grep RMSE| tail -1| awk '{print $3}'", shell=True).decode('UTF-8')
    # print('Test RMSE: %s'% rmse)

    with open('betas.dat','w') as f:
        for i in betas:
            f.write('%f\n'%i)
    time_stop=time.time()
    wallclock=-time_start+time_stop
    print("#PASS %d, Test RMSE: %s, Wallclock: %s s"%(n_pass,rmse,wallclock))

