#!/usr/bin/python
'''

  !---------------------------------------------------------------------------!
  !                                                                           !
  !     MLatom: a Package for Atomistic Simulations with Machine Learning     !
  !                          Development version                              !
  !                           http://mlatom.com/                              !
  !                                                                           !
  !                  Copyright (c) 2013-2020 Pavlo O. Dral                    !
  !                           http://dr-dral.com/                             !
  !                                                                           !
  ! All rights reserved. No part of MLatom may be used, published or          !
  ! redistributed without written permission by Pavlo Dral.                   !
  !                                                                           !
  ! The above copyright notice and this permission notice shall be included   !
  ! in all copies or substantial portions of the Software.                    !
  !                                                                           !
  ! The software is provided "as is", without warranty of any kind, express   !
  ! or implied, including but not limited to the warranties of                !
  ! merchantability, fitness for a particular purpose and noninfringement. In !
  ! no event shall the authors or copyright holders be liable for any claim,  !
  ! damages or other liability, whether in an action of contract, tort or     !
  ! otherwise, arising from, out of or in connection with the software or the !
  ! use or other dealings in the software.                                    !
  !                                                                           !
  ! Cite as: Pavlo O. Dral, J. Comput. Chem. 2019, 40, 2339-2347              !
  !                                                                           !
  !          Pavlo O. Dral, MLatom: a Package for Atomistic Simulations with  !
  !          Machine Learning, development version,                           !
  !          Xiamen University, Xiamen, China, 2013-2020.                     !
  !                                                                           !  
  !---------------------------------------------------------------------------!

'''

import os, sys, subprocess, time, shutil, re, math, random
from itertools import combinations
import numpy as np
try:
    from . import stopper
except:
    import stopper
import time
mlatomdir=os.path.dirname(__file__)
#mlatomfbin="%s/MLatomF_20171208" % mlatomdir
mlatomfbin="%s/MLatomF" % mlatomdir
mlatompy="%s/MLatom.py" % mlatomdir
#au2ang = 0.52917720859 # a.u. to Angstrom. Use the same conversion coefficient as in NEWTON-X (NX-2-B13)

def copy(orig, to):
    while True:
        if os.path.exists(orig):
            os.system('cp %s %s' % (orig, to))
            break
        else:
            time.sleep(1)

class stateDataCls():
    def __init__(self):
        self.energy       = None
        self.gradients    = []

class MDstepCls(object):
    def __init__(self):
        self.step         = None
        self.state        = None
        self.atCoords     = {'el': [], 'coords': []}
        self.states       = [] # List with instancies of stateDataCls
        self.nadcouplings = []
        self.nadvectors   = []
        self.velocities   = []

class MDtrajCls(object):
    def __init__(self):
        self.mdsteps       = [] # List with instancies of MDstepCls
        self.hoppingNsteps = [] # Number of steps where hopping occured
        
class MDdatabaseCls(object):
    def __init__(self):
        self.mdtrajs = [] # List with instancies of MDtrajCls
    def combine(self):
        self.combindices = []
        for itraj in range(len(self.mdtrajs)):
            for istep in range(len(self.mdtrajs[itraj].mdsteps)):
                if self.mdtrajs[itraj].mdsteps[istep].states[0].energy == None:
                    continue
                self.combindices.append({'itraj': itraj, 'istep': istep})

class NAC():
    def __init__(self, nac_file, vector_num):
        with open(nac_file) as f:
            data = f.read().splitlines()
        self.data = []
        self.v_num = vector_num
        for line in data:
            x, y, z  = list(map(float, line.split())) 
            self.data.append([x, y, z])
    
    def correct(self, prev):
        line_num = len(self.data)
        v_line_num = line_num // self.v_num
        for i in range(self.v_num):
            curr_v = self.data[i * v_line_num: (i + 1) * v_line_num]
            prev_v = prev.data[i * v_line_num: (i + 1) * v_line_num]
            if not self.multiply(curr_v, prev_v):
                for j in range(i * v_line_num, (i + 1) * v_line_num):
                    self.data[j] = [-x for x in self.data[j]]
    
    @staticmethod
    def multiply(one, other):
        tmp = 0.0
        for a, b in zip(one, other):
            x1, y1, z1 = a
            x2, y2, z2 = b
            tmp += x1*x2 + y1*y2 + z1*z2
        if tmp >= 0:
            return True
        else:
            return False
            
    def print_new(self, out_name):
        print(('nad_vector line number: %d' % len(self.data)))
        with open(out_name, 'w') as f:
            for line in self.data:
                x, y, z = line
                f.write('%21.8f%21.8f%21.8f\n' % (x, y, z))

def sub_file(a_file, b_file, out_file):
    with open(a_file) as f:
        a_data = f.read().splitlines()
        a_data = map(float, a_data)
    with open(b_file) as f:
        b_data = f.read().splitlines()
        b_data = map(float, b_data)
    out = open(out_file, 'w')
    for a, b in zip(a_data, b_data):
        out.write('%50.12f\n' % (a - b))
    out.close()
    
def sub_grad(a_file, b_file, out_file):
    with open(a_file) as fa, open(b_file) as fb, open(out_file, 'w') as fout:
        a_data = fa.readline().strip() ; next(fa)
        fout.writelines('%s\n\n' % a_data)
        next(fb); next(fb)
        for line in fa:
            a_data = [float(xx) for xx in line.split()]
            b_data = [float(xx) for xx in fb.readline().split()]
            fout.write('%50.12f%50.12f%50.12f\n' % (a_data[0] - b_data[0], a_data[1] - b_data[1], a_data[2] - b_data[2]))

def add_grad(a_file, b_file, out_file):
    with open(a_file) as fa, open(b_file) as fb, open(out_file, 'w') as fout:
        for line in fa:
            a_data = [float(xx) for xx in line.split()]
            b_data = [float(xx) for xx in fb.readline().split()]
            fout.write('%20.5f%20.5f%20.5f\n' % (a_data[0] + b_data[0], a_data[1] + b_data[1], a_data[2] + b_data[2]))

def conv_unit_for_grad(grad_inp_file, grad_out_file):
    with open(grad_inp_file) as f:
        data = f.read().splitlines()
    f_out = open(grad_out_file, 'w')
    for line in data:
        x, y, z = line.split()
        x, y, z = float(x), float(y), float(z)
        f_out.write('%20.5f%20.5f%20.5f\n' % ((x*0.529167/(23.061*27.21)), (y*0.529167/(23.061*27.21)), (z*0.529167/(23.061*27.21))))
    f_out.close()    
    
def conv_unit_for_nac(nac_inp_file, nac_out_file):
    with open(nac_inp_file) as f:
        data = f.read().splitlines()
    f_out = open(nac_out_file, 'w')
    for line in data:
        x, y, z = line.split()
        x, y, z = float(x), float(y), float(z)
        f_out.write('%20.6f%20.6f%20.6f\n' % ((x*0.529167), (y*0.529167), (z*0.529167)))
    f_out.close() 

def average_matrix(matrixfile_list, matrix_out_file):
    matrix_list = []
    for mf in matrixfile_list:
        m = np.loadtxt(mf, dtype=np.float64)
        matrix_list.append(m)
    all_matrix = np.array(matrix_list)
    out_matrix = all_matrix.mean(axis=0)
    np.savetxt(matrix_out_file, out_matrix, fmt='%20.12f', delimiter=' ')

def average_num(numfile_list, num_out_file):
    num_list = []
    for nf in numfile_list:
        with open(nf, 'r') as fnf:
            n = float(fnf.readlines()[0])
        num_list.append(n)
    sum = 0.0
    for num in num_list:
        sum += num
    out_num = float(sum/len(num_list))
    f_out = open(num_out_file, 'w')
    f_out.write('%20.12f\n' % out_num)
    f_out.close()

def std_num(numfile_list, num_out_file):
    num_list = []
    for nf in numfile_list:
        n = np.loadtxt(nf, dtype=np.float64)
        num_list.append(n)
    final_n = num_list[0]
    for i in range(1, len(num_list)):
        final_n = np.c_[final_n, num_list[i]]
    out_num = np.std(final_n, axis = 1)
    np.savetxt(num_out_file, out_num, fmt='%20.12f', delimiter=' ')

class NX_MLatom_interfaceCls(object):
    
    def __init__(self, argsNX = sys.argv[1:]):
        starttime = time.time()
        
        print(' %s ' % ('='*78))
        print(time.strftime(" NX_MLatom_interface started on %d.%m.%Y at %H:%M:%S", time.localtime()))
        args.parse(argsNX)
        print('        with the following options:')
        argsstr = '        '
        for arg in argsNX:
            argsstr += arg + ' '
        print(argsstr.rstrip())
        print(' %s ' % ('='*78))
        sys.stdout.flush()
        
        # Perform requested task
        if args.collectDataForNX:
            self.collectDataForNX(args.dirname)
        if args.createMLforNX:
            self.createMLforNX()
        if args.useMLforNX:
            self.useMLforNX()
        
        endtime = time.time()
        wallclock = endtime - starttime
        print(' %s ' % ('='*78))
        print(' Wall-clock time: %.2f s (%.2f min, %.2f hours)\n' % (wallclock, wallclock / 60.0, wallclock / 3600.0))
        print(time.strftime(" NX_MLatom_interface terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print(' %s ' % ('='*78))
        sys.stdout.flush()
        
    @classmethod
    def collectDataForNX(cls, dirname):
        if os.path.exists('MLatomData'):
            shutil.rmtree('MLatomData')
        os.mkdir('MLatomData')
        mddb = MDdatabaseCls() # MD database
        traj = 1
        while os.path.exists('TRAJECTORIES/TRAJ%d' % traj):# and traj <= 15:
            dirname = 'TRAJECTORIES/TRAJ%d' % traj
        #if os.path.exists('%s' % dirname):
            print(' %s exists. Processing...' % dirname)
            mddb.mdtrajs.append(MDtrajCls())
            # Get geometries
            if os.path.exists('%s/RESULTS/dyn.out' % dirname):
                with open('%s/RESULTS/dyn.out' % dirname, 'r') as fdyn:
                    readGeom = False
                    readVelocity = False
                    for line in fdyn:
                        if 'Molecular dynamics on state' in line:                            
                            mddb.mdtrajs[-1].mdsteps.append(MDstepCls())
                            mddb.mdtrajs[-1].mdsteps[-1].step = int(line.split()[1])
                            mddb.mdtrajs[-1].mdsteps[-1].state = int(line.split()[6])
                            mddb.mdtrajs[-1].mdsteps[-1].states = [stateDataCls() for ii in range(args.nstat)]
                            if mddb.mdtrajs[-1].mdsteps[-1].step > 0:
                                #print mddb.mdtrajs[-1].mdsteps[-1].state, mddb.mdtrajs[-1].mdsteps[-2].state
                                if mddb.mdtrajs[-1].mdsteps[-1].state != mddb.mdtrajs[-1].mdsteps[-2].state:
                                    print('Hopping, step: ', mddb.mdtrajs[-1].mdsteps[-2].step)
                                    mddb.mdtrajs[-1].hoppingNsteps.append(mddb.mdtrajs[-1].mdsteps[-2].step) # Hopping actually was accepted on the previous step
                        if readGeom:
                            if len(line.strip()) == 0:
                                readGeom = False
                            else:
                                xx = line.split()
                                mddb.mdtrajs[-1].mdsteps[-1].atCoords['el'].append(xx[0].upper())
                                mddb.mdtrajs[-1].mdsteps[-1].atCoords['coords'].append([yy for yy in [xx[2], xx[3], xx[4]]])
                        if readVelocity:
                            if len(line.strip()) == 0:
                                readVelocity = False
                            else:
                                xx = line.split()
                                mddb.mdtrajs[-1].mdsteps[-1].velocities.append([yy for yy in [xx[0], xx[1], xx[2]]])
                        if ' geometry:' in line:
                            readGeom = True
                        if ' velocity:' in line:
                            readVelocity = True
            else:
                print(' dyn.out is not found. Data collection for %s aborted' % dirname)
                traj += 1
                continue
                #return
            
            # Get energies
            if os.path.exists('%s/RESULTS/en.dat' % dirname):
                with open('%s/RESULTS/en.dat' % dirname, 'r') as fen:
                    Nstep = -1
                    for line in fen:
                        Nstep += 1
                        xx = line.split()
                        for nstate in range(1,args.nstat+1):
                            mddb.mdtrajs[-1].mdsteps[Nstep].states[nstate-1].energy = xx[nstate]
            else:
                print(' en.dat is not found. Data collection for %s aborted' % dirname)
                traj += 1
                continue
                #return
            
            # Get gradients
            if os.path.exists('%s/RESULTS/nx.log' % dirname):
                with open('%s/RESULTS/nx.log' % dirname, 'r') as fdyn:
                    Nstep    = -1
                    readGrad = False
                    gradients = []
                    nstate = None
                    hoppingFirstRead = False
                    for line in fdyn:
                        if readGrad:
                            if len(line.split()) != 3:
                                readGrad = False
                                if len(gradients) > args.natoms: # We are dealing with grad.all
                                    for nstate in range(1,args.nstat+1):
                                        mddb.mdtrajs[-1].mdsteps[Nstep].states[nstate-1].gradients = gradients[args.natoms*(nstate-1):args.natoms*(nstate-1) + args.natoms]
                                else: # We are dealing with grad (for current state)
                                    mddb.mdtrajs[-1].mdsteps[Nstep].states[nstate-1].gradients = gradients
                                gradients = []
                                if Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == False:
                                    hoppingFirstRead = True
                                    Nstep -= 1 # To read gradients after the hopping
                                elif Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == True:
                                    hoppingFirstRead = False
                            else:
                                xx = line.split()
                                gradients.append([yy for yy in [xx[0], xx[1], xx[2]]])
                        if 'Gradient (a.u.):' in line or 'Gradients (a.u.):' in line or 'Gradient vectors (a.u.) =' in line:
                            Nstep += 1
                            nstate = mddb.mdtrajs[-1].mdsteps[Nstep].state
                            if Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == True:
                                nstate = mddb.mdtrajs[-1].mdsteps[Nstep+1].state
                            readGrad = True
            else:
                print(' nx.log is not found. Data collection for %s aborted' % dirname)
                traj += 1
                continue
                #return
            
            # Get nonadiabatic couplings
            if os.path.exists('%s/RESULTS/nx.log' % dirname):
                with open('%s/RESULTS/nx.log' % dirname, 'r') as fdyn:
                    Nstep    = -1
                    readNAD = False
                    for line in fdyn:
                        if readNAD:
                            if len(line.split()) != 1:
                                readNAD = False
                            else:
                                mddb.mdtrajs[-1].mdsteps[Nstep].nadcouplings.append(line.strip())
                        if 'Nonadiabatic coupling terms v.h (a.u.):' in line:
                            Nstep += 1
                            readNAD = True
            else:
                print(' nx.log is not found. Data collection for %s aborted' % dirname)
                traj += 1
                continue
                #return
            
            # Get nonadiabatic coupling vectors
            if os.path.exists('%s/RESULTS/nx.log' % dirname):
                with open('%s/RESULTS/nx.log' % dirname, 'r') as fdyn:
                    Nstep    = -1
                    readNAD = False
                    readNACME = False
                    hoppingFirstRead = False  
                    for line in fdyn:
                        if readNAD:
                            if len(line.split()) != 3:
                                readNAD = False
                                if Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == False:
                                    hoppingFirstRead = True
                                    Nstep -= 1 # To read NACs after the hopping
                                elif Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == True:
                                    hoppingFirstRead = False
                            else:
                                mddb.mdtrajs[-1].mdsteps[Nstep].nadvectors.append(line.strip().split())
                        if readNACME:
                            if len(line.split()) != 5:
                                readNACME = False
                                if Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == False:
                                    hoppingFirstRead = True
                                    Nstep -= 1 # To read NACs after the hopping
                                elif Nstep in mddb.mdtrajs[-1].hoppingNsteps and hoppingFirstRead == True:
                                    hoppingFirstRead = False
                            else:
                                mddb.mdtrajs[-1].mdsteps[Nstep].nadvectors.append(line.strip().split())
                        if 'Nonadiabatic coupling vectors (a.u.):' in line:
                            Nstep += 1
                            readNAD = True
                        if 'NACME vectors (a.u.) =' in line:
                            Nstep += 1
                            readNACME = True
            
            print(' %d data points are collected from %s' % (len(mddb.mdtrajs[-1].mdsteps), dirname))
            traj += 1                   
        #else:
        #    print ' %s does not exist. Stopping...' % dirname
        #    return
        
        mddb.combine()
        #cls.genInitCond(random.sample([ii+1 for ii in range(len(mddb.combindices))], 1000), mddb)
        #return
        cls.printRefData('.', [ii+1 for ii in range(len(mddb.combindices))], mddb)
        print(' Data collected in the directory MLatomData')
        
    @classmethod
    def printRefData(cls, dirname, lindices, mddb):
        if dirname != '.':
            if os.path.exists('MLatomData/%s' % dirname):
                shutil.rmtree('MLatomData/%s' % dirname)
            os.mkdir('MLatomData/%s' % dirname)


        # Print grad.dat file with gradients (Hartree/Angstrom)
        nstates = list(range(1,args.nstat+1))
        for nstate in nstates:
            with open('MLatomData/%s/grad%d_train.dat' % (dirname, nstate), 'w') as fgrad: # ''', open('MLatomData/%s/dVdR%d_train.dat' % (dirname, nstate), 'w') as fdVdR ''':
                for ii in lindices:
                    itraj = mddb.combindices[ii-1]['itraj']
                    istep = mddb.combindices[ii-1]['istep']
                    gradient = mddb.mdtrajs[itraj].mdsteps[istep].states[nstate-1].gradients
                    if len(gradient) > 0:
                        fgrad.writelines('%d\n\n' % args.natoms)
                        for iatom in range(1,args.natoms+1):
                            gradat = gradient[iatom-1]
                            fgrad.writelines('%s  %s  %s\n' % (gradat[0], gradat[1], gradat[2]))
                            # fdVdR.writelines('%s ' % (gradat[0]))
                        # fdVdR.writelines('\n')

        itr = mddb.combindices[lindices[0]]['itraj']
        ist = mddb.combindices[lindices[0]]['istep']
        if len(mddb.mdtrajs[itr].mdsteps[ist].states[0].gradients) == 0 or len(mddb.mdtrajs[itr].mdsteps[ist].states[1].gradients == 0):
            for nstate in nstates:
                # with open('MLatomData/%s/xyz%d_train.dat' % (dirname, nstate), 'w') as fxyz, open('MLatomData/%s/en%d_%d_train.dat' % (dirname, nstate, nstate), 'w') as fen:
                with open('MLatomData/%s/xyz%d_train.dat' % (dirname, nstate), 'w') as fxyz, open('MLatomData/%s/en%d_train.dat' % (dirname, nstate), 'w') as fen:
                    for ii in lindices:
                        itraj = mddb.combindices[ii-1]['itraj']
                        istep = mddb.combindices[ii-1]['istep']
                        if len(mddb.mdtrajs[itraj].mdsteps[istep].states[nstate-1].gradients) > 0:
                            ens = mddb.mdtrajs[itraj].mdsteps[istep].states[nstate-1].energy
                            fen.writelines('%s\n' % ens)
                            geom = mddb.mdtrajs[itraj].mdsteps[istep].atCoords
                            fxyz.writelines('%d\n\n' % args.natoms)
                            for iatom in range(args.natoms):
                                fxyz.writelines(' %4s  %s  %s  %s\n' % (geom['el'][iatom], geom['coords'][iatom][0], geom['coords'][iatom][1], geom['coords'][iatom][2]))

                            
        
    @classmethod
    def createMLforNX(cls):
        os.environ["OMP_NUM_THREADS"] = '10'
        os.environ["MKL_NUM_THREADS"] = '10'
        os.environ["OMP_PROC_BIND"] = 'true'
        dimdict = {1: 'x', 2: 'y', 3: 'z'}
        Ntrain='Ntrain=10000'
        Nsubtrain = 'Nsubtrain=0.8'

        if True:  # train energy model to get Prior value
            for istate in range(1, args.nstat + 1):
                print(' %s ' % ('='*78))
                print('training ML energy model for the state %d' % istate)
                sys.stdout.flush()
                nname = 'en%d' % istate
                inpfile = '''estAccMLmodel
XYZfile=xyz%d_train.dat
Yfile=%s_train.dat
# Ntrain=800
prior=opt
molDescriptor=RE molDescrType=unsorted
sigma=opt lgSigmaL=-5
lambda=opt lgLambdaL=-35
itrainout=itrain%d.dat
itestout=itest%d.dat
isubtrainout=isubtrain%d.dat
ivalidateout=ivalidate%d.dat
                ''' % (istate, nname, istate, istate, istate, istate)
                with open('en_%d.inp' % istate, 'w') as ff:
                    ff.writelines(inpfile)
                os.system(sys.path[0] + '/MLatom.py en_%d.inp &> en_%d.log' % (istate, istate))

        if True:  # get the Prior value
            prior_value = []
            command = "grep Prior en_%d.log | sed -n 3p | awk '{print $4}' > mlatom_tmp"
            for istate in range(1,args.nstat+1):
                os.system(command % istate)
                with open('mlatom_tmp', 'r') as ff:
                    tmp = ff.read().strip()
                prior_value.append(float(tmp))
            os.system('rm mlatom_tmp')

        if True:
            for istate in range(1,args.nstat+1):
                print(' %s ' % ('='*78))
                print(' ML model of gradients for state %d is being created...' % istate)
                sys.stdout.flush()
                nname = 'grad%d' % istate
                inpfile='''# createMLmodel
estAccMLmodel
XYZfile=xyz%d_train.dat
Yfile=en%d_train.dat
YgradXYZfile=%s_train.dat
MLmodelOut=mlmod_engrad%d.unf
molDescriptor=RE molDescrType=unsorted
sigma=opt lgSigmaL=-5
lambda=opt lgLambdaL=-35
sampling=user-defined
itrainin=itrain%d.dat
itestin=itest%d.dat
isubtrainin=isubtrain%d.dat
ivalidatein=ivalidate%d.dat
prior=%.10f
matDecomp=Bunch-Kaufman
''' % (istate, istate, nname, istate, istate, istate, istate, istate,
        prior_value[istate - 1])      
                with open('create_%s.inp' % nname, 'w') as ff:
                    ff.writelines(inpfile)
                os.system(sys.path[0] + '/MLatom.py create_%s.inp  &> create_%s.log' % (nname, nname))
                

        if False:
        #with open('grad.nonzeros', 'w') as fflag:
            for istate in range(1,args.nstat+1):
                for iatom in range(1,args.natoms+1):
                    for idim in range(1,4):
                        with open('grad%d_%d_%d_train.dat' % (istate, iatom, idim), 'r') as fgrad:
                            #for line in fgrad:
                            #    if abs(float(line)) > 1e-10:
                                    #fflag.writelines('1\n')
                                    print(' %s ' % ('='*78))
                                    print(' ML model of the energy gradient of state %d for atom %d along coordinate %s is being created...' % (istate, iatom, dimdict[idim]))
                                    sys.stdout.flush()
                                    nname = 'grad%d_%d_%d' % (istate, iatom, idim)
                                    inpfile='''createMLmodel XYZfile=xyz%d_train.dat
Yfile=%s_train.dat
MLmodelOut=mlmod_%s.unf
Nsubtrain=8000
Nvalidate=2000
molDescriptor=RE molDescrType=unsorted
sigma=opt lgSigmaL=-5
lambda=opt lgLambdaL=-35
prior=opt
''' % (istate, nname, nname)      
                                    with open('create_%s.inp' % nname, 'w') as ff:
                                        ff.writelines(inpfile)
                                    os.system('cp mlatom.bsub mlatom_%s.bsub' % nname)
                                    os.system('sed -i "s/create/create_%s/g" mlatom_%s.bsub' % (nname, nname))
                                    os.system('bsub < mlatom_%s.bsub' % nname)
        
        os.environ["OMP_NUM_THREADS"] = '1'
        os.environ["MKL_NUM_THREADS"] = '1'
        os.environ["OMP_PROC_BIND"] = 'true'
        
    @classmethod
    def useMLforNX(cls):
        try:
            from . import data
        except:
            import data
        import os
        import sys
        sys.path.append(os.getcwd())
        import mlatom_predictions
        mol = data.molecule()
        mol.read_from_xyz_file(filename='xyz.dat')
        mlatom_predictions.model().predict(molecule=mol, nstat=args.nstat, nstatdyn=args.nstatdyn)
        #print(mol.__dict__)
        return
        # flag to choose NAMD or GSMD
        namd = True
        gsmd = False
        bomd = False
        # flag to choose COLUMBUS or KRR or MNDO or AIQM1 or ensemble or ANI-1ccx
        engrad_COLUMBUS = False
        nac_COLUMBUS = False
        engrad_KRR = True
        nac_KRR = True
        mndo = False
        engrad_AIQM1 = False
        nac_AIQM1 = False
        ensemble_KREG_sGDML_ANI = False
        ensemble_2ANI = False
        ensemble_2KREG = False
        engrad_ANI1ccx = False
        # flag to choose some functions
        use_delta = True
        model0 = False
        save_COLUen = False

        # run COLUMBUS or MNDO or AIQM1 calculations according to flags
        cls.columbus_calc(engrad_COLUMBUS, nac_COLUMBUS, save_COLUen, namd, gsmd, bomd)
        cls.mndo_calc(mndo)
        cls.aiqm1_calc(engrad_AIQM1, nac_AIQM1, namd, gsmd)
        
        
        # get current surface
        if os.path.isfile("curr_surf"):
            with open('curr_surf', 'r') as fcs:
                curr_surf = int(fcs.readlines()[0])
        else: 
            curr_surf = args.nstatdyn
        
        # KRR engrad
        if engrad_KRR:
            if namd:
                os.system('rm -f grad.all grad epot entemp')
                for istate in range(1,args.nstat+1):
                    os.system('rm -f yest_en%d.dat'   % istate)
                    os.system('rm -f yest_grad%d.dat' % istate)
                    subprocess.call([mlatomfbin, 'useMLmodel', 'eqXYZfileIn=eq.xyz', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod_engrad%d.unf' % istate,
                                    'YestFile=yest_en%d.dat' % istate, 'YgradXYZestFile=yest_grad%d.dat' % istate])
                    
                    en = np.loadtxt('yest_en%d.dat' % istate, ndmin=1)
                    np.savetxt('epot.tmp', en, fmt='%20.12f')
                    os.system('cat epot.tmp >> epot')
                    # os.system('cat yest_en%d.dat > entemp' % istate)
                    # os.system("awk '{printf(\"%20.12f\",0.0+$1)}' entemp >> epot")
                    # os.system('echo "" >> epot')
                    os.system('tail -n +3 yest_grad%d.dat >> grad.all' % istate)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
            if gsmd:
                os.system('rm -f grad.all grad epot')
                os.system('rm -f yest_en.dat yest_grad.dat grad.dat')
                with open('xyz.dat') as f:
                    data = f.read().splitlines()
                f_geom = open('xyz.dat', 'w')
                for i in range(len(data)):
                    if 0 <= i <= 1:
                        text = data[i]
                        f_geom.write('%s\n' % (text))
                    else:
                        ele, x, y, z = data[i].split()
                        x, y, z = float(x), float(y), float(z)
                        f_geom.write('%s%15.8f%15.8f%15.8f\n' % (ele, x*0.52917721092, y*0.52917721092, z*0.52917721092))
                f_geom.close()
                subprocess.call([mlatomfbin, 'useMLmodel', 'XYZfile=xyz.dat', 'MLmodelIn=h2.unf',
                                'YestFile=yest_en.dat', 'YgradXYZestFile=yest_grad.dat'])
                os.system('cp yest_en.dat epot')
                os.system("sed -i '1,2d' yest_grad.dat")
                with open('yest_grad.dat') as fg:
                    data = fg.read().splitlines()
                fg_f = open('grad.dat', 'w')
                for line in data:
                    x, y, z = line.split()
                    x, y, z = float(x), float(y), float(z)
                    fg_f.write('%25.13f%25.13f%25.13f\n' % (x*0.52917721092, y*0.52917721092, z*0.52917721092))
                fg_f.close()
                os.system('cp grad.dat grad')
                os.system('cp grad.dat grad.all')

        # KRR NACs 
        if nac_KRR:
            # generate graddiff_descriptor
            state_comb = list(combinations(range(1, args.nstat+1), 2))
            delta_engrad_list = [f'{b}-{a}' for a, b in state_comb]
            delta_en_list = [f'{b}-{a}' for a, b in state_comb]
            # delta_engrad_list = ['2-1','3-1','3-2']
            # delta_en_list = ['2-1','3-1','3-2']
            if use_delta and engrad_COLUMBUS:
                for d_engrad, d_en in zip(delta_engrad_list, delta_en_list):
                    a, b = d_en.split('-')
                    os.system("sed '%dq;d' epot > yest_en%s.dat" % (int(a), int(a)))
                    os.system("sed '%dq;d' epot > yest_en%s.dat" % (int(b), int(b)))
                    a, b = 'yest_en%s.dat' % a, 'yest_en%s.dat' % b
                    sub_file(a, b, 'yest_en%s.dat' % d_en)
                    en = np.loadtxt('yest_en%s.dat' % d_en, ndmin=1)
                    np.savetxt('en%s.dat' % d_en, en, fmt='%20.12f')
                    # os.system('mv yest_en%s.dat deltaentemp' % d_en)
                    # os.system("awk '{printf(\"%20.12f\",$1)}' deltaentemp > deltaentemp2")
                    # os.system('mv deltaentemp2 en%s.dat' % d_en)
                    c, d = d_engrad.split('-')
                    os.system('echo %d > yest_grad%s.dat' % (args.natoms, c) )
                    os.system('echo   >> yest_grad%s.dat' % c)
                    os.system('echo %d > yest_grad%s.dat' % (args.natoms, d) )
                    os.system('echo   >> yest_grad%s.dat' % d)
                    
                    cc = int(c)
                    os.system("sed -n '%d,%dp' grad.all > yest_grad%s.dat" % ((cc-1)*args.natoms + 1, cc*args.natoms ,c))
                    
                    dd = int(d)
                    os.system("sed -n '%d,%dp' grad.all > yest_grad%s.dat" % ((dd-1)*args.natoms + 1, dd*args.natoms ,d))

                    c, d = 'yest_grad%s.dat' % c, 'yest_grad%s.dat' % d
                    sub_grad(c, d, 'yest_grad%s.dat' % d_engrad)
                    os.system('cp yest_grad%s.dat grad%s_column.dat' % (d_engrad,d_engrad))
                    os.system('sed -n "3,%dp" grad%s_column.dat| tr "\n" " "  > grad%s.dat' % (args.natoms+2 , d_engrad, d_engrad))
            elif use_delta and not engrad_COLUMBUS:
                for d_engrad, d_en in zip(delta_engrad_list, delta_en_list):
                    a, b = d_en.split('-')
                    a, b = 'yest_en%s.dat' % a, 'yest_en%s.dat' % b
                    sub_file(a, b, 'yest_en%s.dat' % d_en)
                    en = np.loadtxt('yest_en%s.dat' % d_en, ndmin=1)
                    np.savetxt('en%s.dat' % d_en, en, fmt='%20.12f')
                    # os.system('mv yest_en%s.dat deltaentemp' % d_en)
                    # os.system("awk '{printf(\"%20.12f\",$1)}' deltaentemp > deltaentemp2")
                    # os.system('mv deltaentemp2 en%s.dat' % d_en)
                    c, d = d_engrad.split('-')
                    c, d = 'yest_grad%s.dat' % c, 'yest_grad%s.dat' % d
                    sub_grad(c, d, 'yest_grad%s.dat' % d_engrad)
                    os.system('cp yest_grad%s.dat grad%s_column.dat' % (d_engrad,d_engrad))
                    os.system('sed -n "3,%dp" grad%s_column.dat| tr "\n" " "  > grad%s.dat' % (args.natoms+2 , d_engrad, d_engrad))
            else:
                for d_engrad, d_en in zip(delta_engrad_list, delta_en_list):
                    os.system('rm -f yest_en%s.dat' % d_engrad)
                    os.system('rm -f en%s.dat' % d_en)
                    os.system('rm -f yest_grad%s.dat' % d_engrad)
                    os.system('rm -f grad%s_column.dat' % d_engrad)
                    os.system('rm -f grad%s.dat' % d_engrad)
                    subprocess.call([mlatomfbin, 'useMLmodel', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod_engrad%s.unf' % d_engrad,
                                    'YestFile=yest_en%s.dat' % d_engrad,'YgradXYZestFile=yest_grad%s.dat' % d_engrad])
                    en = np.loadtxt('yest_en%s.dat' % d_en, ndmin=1)
                    np.savetxt('en%s.dat' % d_en, en, fmt='%20.12f')
                    # os.system('mv yest_en%s.dat deltaentemp' % d_engrad)
                    # os.system("awk '{printf(\"%20.12f\",$1)}' deltaentemp > deltaentemp2")
                    # os.system('mv deltaentemp2 en%s.dat' % d_en)
                    os.system('cp yest_grad%s.dat grad%s_column.dat' % (d_engrad,d_engrad))
                    os.system('sed -n "3,%dp" grad%s_column.dat| tr "\n" " "  > grad%s.dat' % (args.natoms+2,d_engrad,d_engrad))
            # generate nac1-2/1-3/2-3.dat
            state_comb = list(combinations(range(1, args.nstat+1), 2))
            nac_list = [f'{b}-{a}' for a, b in state_comb]
            # nac_list = ['2-1','3-1','3-2']
            os.system('rm -f nad_vectors')
            for inac in range(len(nac_list)):
                os.system('rm -f yest_nac%s_dE.dat' % nac_list[inac])
                os.system('rm -f nac%s_dE.dat' % nac_list[inac])
                os.system('rm -f en%s_%d.dat' % (delta_en_list[inac], args.natoms))
                os.system('rm -f nac%s.dat' % nac_list[inac])
                subprocess.call([mlatomfbin, 'useMLmodel', 'XfileIn=grad%s.dat' % nac_list[inac], 'MLmodelIn=mlmod_nac%s_dE.unf' % nac_list[inac], 'YgradXYZestFile=yest_nac%s_dE.dat' % nac_list[inac]])
                #subprocess.call([mlatomfbin, 'useMLmodel', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod_nac%s_dE.unf' % nac_list[inac], 'YgradXYZestFile=yest_nac%s_dE.dat' % nac_list[inac]])
                os.system('tail -n +3 yest_nac%s_dE.dat >  nac%s_dE.dat' % (nac_list[inac],nac_list[inac]))

                # os.system('awk \'{print $0"\\n"$0"\\n"$0"\\n"$0"\\n"$0"\\n"$0}\' en%s.dat > en%s_6.dat' % (delta_en_list[inac],delta_en_list[inac]))
                nacde = np.loadtxt('nac%s_dE.dat' % nac_list[inac], ndmin=1)
                en = np.loadtxt('en%s.dat' % delta_en_list[inac], ndmin=1)
                for idx, en_value in enumerate(en):
                    nacde[idx] = nacde[idx] / en_value
                np.savetxt('nac%s.dat' % nac_list[inac], nacde, fmt='%20.12f')
                # Debugging: set all NACs to zero
                os.remove('nac%s.dat' % nac_list[inac])
                for j in range(1, args.natoms + 1):
                    os.system('echo 0.0 0.0 0.0 >> nac%s.dat' % nac_list[inac])
                # os.system('awk \'BEGIN{num=%d} {for (i=0; i<num; i++){print $0} }\' en%s.dat > en%s_%d.dat' % (args.natoms, delta_en_list[inac], delta_en_list[inac], args.natoms) )
                # os.system('paste en%s_%d.dat nac%s_dE.dat > en_nac_dE_temp.dat' % (delta_en_list[inac], args.natoms, nac_list[inac]))
                # os.system(' awk \'{printf("%50.12f%50.12f%50.12f\\n", $2/$1, $3/$1, $4/$1)}\'  en_nac_dE_temp.dat > nac_temp.dat')
                # os.system('cp nac_temp.dat  nac%s.dat' % nac_list[inac])
                # os.system('rm *temp*')
                os.system('cat  nac%s.dat >> nad_vectors' % nac_list[inac])

        # ensemble_KREG_sGDML_ANI engrad
        if ensemble_KREG_sGDML_ANI:
            if namd:
                results_dir = '../RESULTS'
                # Delete output files of last iteration
                os.system('rm -f grad.all grad epot epot.tmp std_en.all std_en')
                for istate in range(1,args.nstat+1):
                    for mlmod in range(1, 3+1):
                        os.system('rm -f mlmod%d_yest_en%d.dat'   % (mlmod, istate))
                        os.system('rm -f mlmod%d_yest_grad%d.dat'   % (mlmod, istate))
                        os.system('rm -f mlmod%d_yest_grad%d_m.dat'   % (mlmod, istate))
                    os.system('rm -f yest_en%d.dat'   % istate)
                    os.system('rm -f yest_grad%d.dat'   % istate)
                    os.system('rm -f std_en%d.dat' % istate)
                    # Predict en and grad
                    subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=KREG', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod1_engrad%d.unf' % istate,
                                'YestFile=mlmod1_yest_en%d.dat' % istate, 'YgradXYZestFile=mlmod1_yest_grad%d.dat' % istate])
                    subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=sGDML', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod2_engrad%d.unf.npz' % istate,
                                'YestFile=mlmod2_yest_en%d.dat' % istate, 'YgradXYZestFile=mlmod2_yest_grad%d.dat' % istate])
                    subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=ANI', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod3_engrad%d.unf' % istate,
                                'YestFile=mlmod3_yest_en%d.dat' % istate, 'YgradXYZestFile=mlmod3_yest_grad%d.dat' % istate])
                    # Save mlmod1/2/3 en grad in RESULTS
                    for mlmod in range(1, 3+1):
                        os.system('cat mlmod%d_yest_en%d.dat >> %s/mlmod%d_yest_en%d.dat' % (mlmod, istate, results_dir, mlmod, istate))
                        os.system('cat mlmod%d_yest_grad%d.dat >> %s/mlmod%d_yest_grad%d.dat' % (mlmod, istate, results_dir, mlmod, istate))
                    # Change mlmod1/2/3_yest_grad.dat format
                        os.system('tail -n +3 mlmod%d_yest_grad%d.dat >> mlmod%d_yest_grad%d_m.dat' % (mlmod, istate, mlmod, istate))
                    # Obtain averaged en and grad
                    average_num(["mlmod1_yest_en%d.dat" % istate, "mlmod2_yest_en%d.dat" % istate, "mlmod3_yest_en%d.dat" % istate], "yest_en%d.dat" % istate)
                    average_matrix(["mlmod1_yest_grad%d_m.dat" % istate, "mlmod2_yest_grad%d_m.dat" % istate, "mlmod3_yest_grad%d_m.dat" % istate], "yest_grad%d.dat" % istate)
                    # Save averaged en grad in RESULTS
                    #os.system('cat yest_en%d.dat >> %s/mean_en%d.dat' % (istate, results_dir, istate))
                    #with open('%s/mean_grad%d.dat' % (results_dir, istate), 'w') as fmg:
                    #    fmg.write(str(args.natoms)+"\n")
                    #os.system('cat yest_grad%d.dat >> %s/mean_grad%d.dat' % (istate, results_dir, istate))
                    # Obtain std_en, std_en.all
                    std_num(["mlmod1_yest_en%d.dat" % istate, "mlmod2_yest_en%d.dat" % istate, "mlmod3_yest_en%d.dat" % istate], "std_en%d.dat" % istate)
                    os.system('cat std_en%d.dat >> std_en.all' % istate)
                    # Save std_en in RESULTS
                    os.system('cat std_en%d.dat >> %s/std_en%d.dat' % (istate, results_dir, istate))
                # Create epot, grad.all and grad
                    en = np.loadtxt('yest_en%d.dat' % istate, ndmin=1)
                    np.savetxt('epot.tmp', en, fmt='%20.12f')
                    os.system('cat epot.tmp >> epot')
                    os.system('cat yest_grad%d.dat >> grad.all' % istate)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
                # Create std_en, std_en.list
                os.system("sed -n '%dp' std_en.all > std_en" % (curr_surf))
                os.system('cat std_en >> std_en.list')
                # Check threshold
                with open('std_en', 'r') as fsd:
                    std_en_value = float(fsd.readlines()[0])
                std_en_threshold = 0.006
                if std_en_value > std_en_threshold:
                    os.system('echo "std_en_value > std_en_threshold" >> std_en')
            if bomd:
                results_dir = '../RESULTS'
                # Delete output files of last iteration
                os.system('rm -f grad.all grad epot epot.tmp std_en')
                for mlmod in range(1, 3+1):
                    os.system('rm -f mlmod%d_yest_en%d.dat'   % (mlmod, args.nstatdyn))
                    os.system('rm -f mlmod%d_yest_grad%d.dat'   % (mlmod, args.nstatdyn))
                    os.system('rm -f mlmod%d_yest_grad%d_m.dat'   % (mlmod, args.nstatdyn))
                os.system('rm -f yest_en%d.dat'   % args.nstatdyn)
                os.system('rm -f yest_grad%d.dat'   % args.nstatdyn)
                os.system('rm -f std_en%d.dat' % args.nstatdyn)
                # Predict en and grad
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=KREG', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod1_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod1_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod1_yest_grad%d.dat' % args.nstatdyn])
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=sGDML', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod2_engrad%d.unf.npz' % args.nstatdyn,
                                'YestFile=mlmod2_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod2_yest_grad%d.dat' % args.nstatdyn])
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=ANI', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod3_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod3_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod3_yest_grad%d.dat' % args.nstatdyn])
                # Save mlmod1/2/3 en grad in RESULTS
                for mlmod in range(1, 3+1):
                    os.system('cat mlmod%d_yest_en%d.dat >> %s/mlmod%d_yest_en%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                    os.system('cat mlmod%d_yest_grad%d.dat >> %s/mlmod%d_yest_grad%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                # Change mlmod1/2/3_yest_grad.dat format
                    os.system('tail -n +3 mlmod%d_yest_grad%d.dat >> mlmod%d_yest_grad%d_m.dat' % (mlmod, args.nstatdyn, mlmod, args.nstatdyn))
                # Obtain averaged en and grad
                average_num(["mlmod1_yest_en%d.dat" % args.nstatdyn, "mlmod2_yest_en%d.dat" % args.nstatdyn, "mlmod3_yest_en%d.dat" % args.nstatdyn], "yest_en%d.dat" % args.nstatdyn)
                average_matrix(["mlmod1_yest_grad%d_m.dat" % args.nstatdyn, "mlmod2_yest_grad%d_m.dat" % args.nstatdyn, "mlmod3_yest_grad%d_m.dat" % args.nstatdyn], "yest_grad%d.dat" % args.nstatdyn)
                # Obtain std_en
                std_num(["mlmod1_yest_en%d.dat" % args.nstatdyn, "mlmod2_yest_en%d.dat" % args.nstatdyn, "mlmod3_yest_en%d.dat" % args.nstatdyn], "std_en%d.dat" % args.nstatdyn)
                # Save std_en in RESULTS
                os.system('cat std_en%d.dat >> %s/std_en%d.dat' % (args.nstatdyn, results_dir, args.nstatdyn))
                # Create epot, grad.all and grad
                en = np.loadtxt('yest_en%d.dat' % args.nstatdyn, ndmin=1)
                np.savetxt('epot.tmp', en, fmt='%20.12f')
                for i in range(1, args.nstatdyn):
                    os.system('echo 0.0 >> epot')
                    for j in range(1, args.natoms + 1):
                        os.system('echo 0.0 0.0 0.0 >> grad.all')
                os.system('cat epot.tmp >> epot')
                os.system('cat yest_grad%d.dat >> grad.all' % args.nstatdyn)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
                # Create std_en, std_en.list
                os.system("cat std_en%d.dat > std_en" % (args.nstatdyn))
                os.system('cat std_en >> std_en.list')
                # Check threshold
                with open('std_en', 'r') as fsd:
                    std_en_value = float(fsd.readlines()[0])
                std_en_threshold = 0.006
                if std_en_value > std_en_threshold:
                    os.system('echo "std_en_value > std_en_threshold" >> std_en')

        # ensemble_2ANI engrad
        if ensemble_2ANI:
            if bomd:
                results_dir = '../RESULTS'
                # Delete output files of last iteration
                os.system('rm -f grad.all grad epot epot.tmp std_en')
                for mlmod in range(1, 2+1):
                    os.system('rm -f mlmod%d_yest_en%d.dat'   % (mlmod, args.nstatdyn))
                    os.system('rm -f mlmod%d_yest_grad%d.dat'   % (mlmod, args.nstatdyn))
                    os.system('rm -f mlmod%d_yest_grad%d_m.dat'   % (mlmod, args.nstatdyn))
                os.system('rm -f yest_en%d.dat'   % args.nstatdyn)
                os.system('rm -f yest_grad%d.dat'   % args.nstatdyn)
                os.system('rm -f std_en%d.dat' % args.nstatdyn)
                # Predict en and grad
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=ANI', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod1_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod1_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod1_yest_grad%d.dat' % args.nstatdyn])
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=ANI', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod2_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod2_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod2_yest_grad%d.dat' % args.nstatdyn])
                # Save mlmod1/2 en grad in RESULTS
                for mlmod in range(1, 2+1):
                    os.system('cat mlmod%d_yest_en%d.dat >> %s/mlmod%d_yest_en%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                    os.system('cat mlmod%d_yest_grad%d.dat >> %s/mlmod%d_yest_grad%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                # Change mlmod1/2_yest_grad.dat format
                    os.system('tail -n +3 mlmod%d_yest_grad%d.dat >> mlmod%d_yest_grad%d_m.dat' % (mlmod, args.nstatdyn, mlmod, args.nstatdyn))
                # Obtain averaged en and grad
                average_num(["mlmod1_yest_en%d.dat" % args.nstatdyn, "mlmod2_yest_en%d.dat" % args.nstatdyn], "yest_en%d.dat" % args.nstatdyn)
                average_matrix(["mlmod1_yest_grad%d_m.dat" % args.nstatdyn, "mlmod2_yest_grad%d_m.dat" % args.nstatdyn], "yest_grad%d.dat" % args.nstatdyn)
                # Obtain std_en
                std_num(["mlmod1_yest_en%d.dat" % args.nstatdyn, "mlmod2_yest_en%d.dat" % args.nstatdyn], "std_en%d.dat" % args.nstatdyn)
                # Save std_en in RESULTS
                os.system('cat std_en%d.dat >> %s/std_en%d.dat' % (args.nstatdyn, results_dir, args.nstatdyn))
                # Create epot, grad.all and grad
                en = np.loadtxt('yest_en%d.dat' % args.nstatdyn, ndmin=1)
                np.savetxt('epot.tmp', en, fmt='%20.12f')
                for i in range(1, args.nstatdyn):
                    os.system('echo 0.0 >> epot')
                    for j in range(1, args.natoms + 1):
                        os.system('echo 0.0 0.0 0.0 >> grad.all')
                os.system('cat epot.tmp >> epot')
                os.system('cat yest_grad%d.dat >> grad.all' % args.nstatdyn)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
                # Create std_en, std_en.list
                os.system("cat std_en%d.dat > std_en" % (args.nstatdyn))
                os.system('cat std_en >> std_en.list')
                # Check threshold
                with open('std_en', 'r') as fsd:
                    std_en_value = float(fsd.readlines()[0])
                std_en_threshold = 0.006
                if std_en_value > std_en_threshold:
                    os.system('echo "std_en_value > std_en_threshold" >> std_en')

        # ensemble_2KREG engrad
        if ensemble_2KREG:
            if bomd:
                results_dir = '../RESULTS'
                # Delete output files of last iteration
                os.system('rm -f grad.all grad epot epot.tmp std_en')
                for mlmod in range(1, 2+1):
                    os.system('rm -f mlmod%d_yest_en%d.dat'   % (mlmod, args.nstatdyn))
                for mlmod in range(2, 2+1):
                    os.system('rm -f mlmod%d_yest_grad%d.dat'   % (mlmod, args.nstatdyn))
                    os.system('rm -f mlmod%d_yest_grad%d_m.dat'   % (mlmod, args.nstatdyn))
                os.system('rm -f yest_en%d.dat'   % args.nstatdyn)
                os.system('rm -f yest_grad%d.dat'   % args.nstatdyn)
                os.system('rm -f std_en%d.dat' % args.nstatdyn)
                # Predict en and grad
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=KREG', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod1_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod1_yest_en%d.dat' % args.nstatdyn])
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=KREG', 'XYZfile=xyz.dat', 'MLmodelIn=mlmod2_engrad%d.unf' % args.nstatdyn,
                                'YestFile=mlmod2_yest_en%d.dat' % args.nstatdyn, 'YgradXYZestFile=mlmod2_yest_grad%d.dat' % args.nstatdyn])
                # Save mlmod1/2 en grad in RESULTS
                for mlmod in range(1, 2+1):
                    os.system('cat mlmod%d_yest_en%d.dat >> %s/mlmod%d_yest_en%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                for mlmod in range(2, 2+1):
                    os.system('cat mlmod%d_yest_grad%d.dat >> %s/mlmod%d_yest_grad%d.dat' % (mlmod, args.nstatdyn, results_dir, mlmod, args.nstatdyn))
                # Change mlmod2_yest_grad.dat format
                    os.system('tail -n +3 mlmod%d_yest_grad%d.dat >> mlmod%d_yest_grad%d_m.dat' % (mlmod, args.nstatdyn, mlmod, args.nstatdyn))
                # Obtain std_en
                std_num(["mlmod1_yest_en%d.dat" % args.nstatdyn, "mlmod2_yest_en%d.dat" % args.nstatdyn], "std_en%d.dat" % args.nstatdyn)
                # Save std_en in RESULTS
                os.system('cat std_en%d.dat >> %s/std_en%d.dat' % (args.nstatdyn, results_dir, args.nstatdyn))
                # Create epot, grad.all and grad
                en = np.loadtxt('mlmod2_yest_en%d.dat' % args.nstatdyn, ndmin=1)
                np.savetxt('epot.tmp', en, fmt='%20.12f')
                for i in range(1, args.nstatdyn):
                    os.system('echo 0.0 >> epot')
                    for j in range(1, args.natoms + 1):
                        os.system('echo 0.0 0.0 0.0 >> grad.all')
                os.system('cat epot.tmp >> epot')
                os.system('cat mlmod2_yest_grad%d_m.dat >> grad.all' % args.nstatdyn)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
                # Create std_en, std_en.list
                os.system("cat std_en%d.dat > std_en" % (args.nstatdyn))
                os.system('cat std_en >> std_en.list')
                # Check threshold
                with open('std_en', 'r') as fsd:
                    std_en_value = float(fsd.readlines()[0])
                std_en_threshold = 0.006
                if std_en_value > std_en_threshold:
                    os.system('echo "std_en_value > std_en_threshold" >> std_en')

        # ANI-1ccx engrad
        if engrad_ANI1ccx:
            if gsmd:
                os.system('rm -f grad.all grad epot')
                os.system('rm -f yest_en.dat yest_grad.dat grad.dat')
                with open('xyz.dat') as f:
                    data = f.read().splitlines()
                f_geom = open('xyz.dat', 'w')
                for i in range(len(data)):
                    if 0 <= i <= 1:
                        text = data[i]
                        f_geom.write('%s\n' % (text))
                    else:
                        ele, x, y, z = data[i].split()
                        x, y, z = float(x), float(y), float(z)
                        f_geom.write('%s%15.8f%15.8f%15.8f\n' % (ele, x*0.52917721092, y*0.52917721092, z*0.52917721092))
                f_geom.close()
                subprocess.call([mlatompy, 'useMLmodel', 'MLmodelType=ANI1ccx', 'XYZfile=xyz.dat',
                                'YestFile=yest_en.dat', 'YgradXYZestFile=yest_grad.dat'])
                os.system('cp yest_en.dat epot')
                os.system("sed -i '1,2d' yest_grad.dat")
                with open('yest_grad.dat') as fg:
                    data = fg.read().splitlines()
                fg_f = open('grad.dat', 'w')
                for line in data:
                    x, y, z = line.split()
                    x, y, z = float(x), float(y), float(z)
                    fg_f.write('%25.13f%25.13f%25.13f\n' % (x*0.52917721092, y*0.52917721092, z*0.52917721092))
                fg_f.close()
                os.system('cp grad.dat grad')
                os.system('cp grad.dat grad.all')


        # model0
        if model0:
            energies = []
            with open('epot', 'r') as ff:
                for line in ff:
                    energies.append(float(line))
            Egap = 0.03
            if args.nstatdyn == 1:
                if abs(energies[1] - energies[0]) <= Egap:
                    cls.columbus_calc(True, True)
            elif args.nstatdyn == args.nstat:
                if abs(energies[args.nstatdyn-1] - energies[args.nstatdyn-2]) <= Egap:
                    cls.columbus_calc(True, True)
            else:
                if ((abs(energies[args.nstatdyn-1] - energies[args.nstatdyn-2]) <= Egap) or
                    (abs(energies[args.nstatdyn-1] - energies[args.nstatdyn]) <= Egap)):
                    cls.columbus_calc(True, True)

        # postprocessing for NACs
        if namd:
            # NAC phase correction
            phase_nac_prev = 'old_nac'
            nad_vector = 'nad_vectors'
            if os.path.exists(phase_nac_prev):
                v_num = args.nstat * (args.nstat-1) // 2
                curr_nac = NAC(nad_vector, v_num)
                prev_nac = NAC(phase_nac_prev, v_num)
                curr_nac.correct(prev_nac)
                curr_nac.print_new(nad_vector)
            os.system('cp %s %s' % (nad_vector, phase_nac_prev))
            # prevent hoppings back
            # pre_hop_back = False
            # if pre_hop_back:
            #     if curr_surf == 1:
            #         os.system('sed -i \'1,18s/.*/0.0 0.0 0.0/g\' nad_vectors')
            #     elif curr_surf == 2:
            #         os.system('sed -i \'13,18s/.*/0.0 0.0 0.0/g\' nad_vectors')
            # prevent 3->1 1->3 hoppings back
            pre_31hop_back = False
            if pre_31hop_back:
                state_comb = list(combinations(range(1, args.nstat+1), 2))
                for idx,(a, b) in enumerate(state_comb):
                    if abs(a-b) != 1:
                        os.system("sed -i '%d,%ds/.*/0.0 0.0 0.0/g' nad_vectors" % (idx*args.natoms+1, (idx+1)*args.natoms))
        
    def columbus_calc(engrad_COLUMBUS, nac_COLUMBUS, save_COLUen, namd, gsmd, bomd):
        # get current surface
        if os.path.isfile("curr_surf"):
            with open('curr_surf', 'r') as fcs:
                curr_surf = int(fcs.readlines()[0])
        else: 
            curr_surf = args.nstatdyn
        # columbus calculation        
        if engrad_COLUMBUS or nac_COLUMBUS:
            print('From MLatom to NX: calling QC program')
            mocoef = 'mocoef'
            mocoef_mc = 'mocoef_mc.sp'
            rundir = 'columbus'
            # create rundir folder
            if os.path.exists(rundir):
                os.system('rm -rf %s' % rundir)
            if namd:
                os.system('cp -r JOB_NAD/columbus %s' % rundir)
            if gsmd or bomd:
                os.system('cp -r JOB_AD/columbus %s' % rundir)
            # copy mocoef to rundir
            if os.path.exists(mocoef):
                os.system('cp %s %s' % (mocoef, rundir))
            # copy geom to rundir
            os.system('cp geom %s' % rundir)
            # enter rundir
            os.chdir(rundir)
            # change geom format
            with open('geom') as f:
                data = f.read().splitlines()
            f_geom = open('geom', 'w')
            for line in data:
                ele, num, x, y, z, abundance = line.split()
                x, y, z = float(x), float(y), float(z)
                f_geom.write('%2s%8s%14.8f%14.8f%14.8f%14s\n' % (ele, num, x, y, z, abundance))
            f_geom.close()
            # run columbus
            os.system('$COLUMBUS/runc -m 1700 &> runls')
            # copy new mocoef to TEMP
            os.system('cp MOCOEFS/%s ../mocoef' % mocoef_mc)
            # copy results to TEMP
            os.chdir('..')
            state_comb = list(combinations(range(1, args.nstat+1), 2))
            nac_del = ' '.join([f'nac{b}-{a}.dat' for a, b in state_comb]) 
            grad_del = ' '.join([ f'grad{ii}.dat' for ii in range(1, args.nstat+1) ])
            os.system('rm -f energy ' + nac_del + ' ' + grad_del)
            # os.system('rm -f energy grad1.dat grad2.dat grad3.dat nac2-1.dat nac3-1.dat nac3-2.dat')
            os.chdir(rundir)
            copy('LISTINGS/energy', '..')
            for s in range(1, args.nstat+1):
                copy('GRADIENTS/cartgrd.drt1.state%d.sp' % s, '../grad%d.dat' % s)
            # for s, e in [(2, 1), (3, 1), (3, 2)]:
            for e, s in state_comb:
                copy('GRADIENTS/cartgrd.nad.drt1.state%d.drt1.state%d.sp' % (s, e), '../nac%d-%d.dat' % (s, e))
            os.chdir('..')
            # COLUMBUS engrad
            if engrad_COLUMBUS:
                os.system('rm -f epot grad.all grad')
                os.system("sed -n 2,%dp energy | awk '{print $2}' > epot" % (args.nstat+1))
                os.system(": > grad.all")
                # for ii in [1, 2, 3]:
                for ii in range(1, args.nstat+1):
                    os.system("sed s/D/E/g grad%d.dat >> grad.all" % ii)
                os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
            if save_COLUen:
                results_dir = '../RESULTS/'
                os.system('rm -f epot_COLU_tmp epot_COLU')
                if namd:
                    os.system("sed -n 2,%dp energy | awk '{print $2}' > epot_COLU_tmp" % (args.nstat+1))
                    os.system("sed -n '%dp' epot_COLU_tmp >> epot_COLU_tmp" % curr_surf)
                    # if curr_surf == 1:
                    #     os.system('sed -n \'1p\' epot_COLU_tmp >> epot_COLU_tmp')
                    # elif curr_surf == 2:
                    #     os.system('sed -n \'2p\' epot_COLU_tmp >> epot_COLU_tmp')
                    # else:
                    #     os.system('sed -n \'3p\' epot_COLU_tmp >> epot_COLU_tmp')  
                    os.system("awk '{printf\"%20.9f\",$1}' epot_COLU_tmp > epot_COLU")
                    os.system('cat epot_COLU >> %s/en_COLU.dat' % results_dir)
                    os.system('echo >> %s/en_COLU.dat' % results_dir)
                if gsmd or bomd:
                    os.system("sed -n %dp energy | awk '{print $2}' > epot_COLU_tmp" % (args.nstatdyn+1))
                    os.system("awk '{printf\"%20.9f\",$1}' epot_COLU_tmp > epot_COLU")
                    os.system('cat epot_COLU >> %s/en_COLU.dat' % results_dir)
                    os.system('echo >> %s/en_COLU.dat' % results_dir)
            # COLUMBUS NACs    
            if nac_COLUMBUS:
                if namd:
                    os.system('rm -f nad_vectors')
                    # for nac in ['2-1','3-1','3-2']:
                    for nac in [ f'{b}-{a}' for a, b in combinations(range(1, args.nstat+1), 2) ]:
                        os.system('cat nac%s.dat >> nad_vectors' % nac)

    def mndo_calc(mndo):
        # get current surface
        if os.path.isfile("curr_surf"):
            with open('curr_surf', 'r') as fcs:
                curr_surf = int(fcs.readlines()[0])
        else: 
            curr_surf = args.nstatdyn
            # curr_surf = 3
        # mndo namd      
        if mndo:
            print('From MLatom to NX: calling MNDO NAMD program')
            mocoef = 'fort.12'
            rundir = 'mndo'
            # create rundir folder
            if os.path.exists(rundir):
                os.system('rm -rf %s' % rundir)
            os.system('cp -r JOB_NAD/mndo %s' % rundir)
            # copy mocoef to rundir
            if os.path.exists(mocoef):
                os.system('cp %s %s' % (mocoef, rundir))
            # copy geom to rundir
            os.system('cp geom %s' % rundir)
            # enter rundir
            os.chdir(rundir)
            # change geom format
            with open('geom') as f:
                data = f.read().splitlines()
            f_geom = open('geom', 'w')
            for line in data:
                ele, num, x, y, z, abundance = line.split()
                x, y, z = float(x), float(y), float(z)
                f_geom.write('%s%15.8f%2.0f%15.8f%2.0f%15.8f%2.0f\n' % (num, x*0.52917721092, 1, y*0.52917721092, 1, z*0.52917721092, 1))
            f_geom.close()
            # change input
            os.system('sed -i \'8r geom\' mndo.inp')
            # run mndo
            os.system('$MNDO < mndo.inp > mndo.out')
            # copy new mocoef to TEMP
            os.system('cp fort.12 ..')
            # collect results 
            os.system(': > energy.mndo')
            for ii in range(1, args.nstat+1):
                os.system('grep \"State  %d\" mndo.out | awk -FE= \'{print $2}\'| awk \'{print $1}\' >> energy.mndo' % ii)
            # os.system('grep \"State  1\" mndo.out | awk -FE= \'{print $2}\'| awk \'{print $1}\' > energy.mndo')
            # os.system('grep \"State  2\" mndo.out | awk -FE= \'{print $2}\'| awk \'{print $1}\' >> energy.mndo')
            # os.system('grep \"State  3\" mndo.out | awk -FE= \'{print $2}\'| awk \'{print $1}\' >> energy.mndo')
            os.system(" sed -n '/CI CALCULATION FOR STATE:  1/, /CI CALCULATION FOR STATE:  2/ p' mndo.out | grep -A 9 'GRADIENTS (KCAL' |\
                        sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad1.dat.mndo")
            os.system(" sed -n '/CI CALCULATION FOR STATE:  2/, /CI CALCULATION FOR STATE:  3/ p' mndo.out | grep -A 9 'GRADIENTS (KCAL' |\
                        sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad2.dat.mndo")
            os.system(" sed -n '/CI CALCULATION FOR STATE:  3/, /CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  2   1/ p' mndo.out | \
                        grep -A 9 'GRADIENTS (KCAL' |\
                        sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad3.dat.mndo")     
            os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  2   1/,/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   1/ p' mndo.out | \
                        grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac2-1.dat.mndo")
            os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   1/,/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   2/ p' mndo.out | \
                        grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac3-1.dat.mndo")
            os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   2/,/SUMMARY OF INTERSTATE COUPLING CALCULATIONS/ p' mndo.out | \
                        grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac3-2.dat.mndo")       
            # convert unit
            with open('energy.mndo') as f:
                data = f.read().splitlines()
            f_en = open('energy', 'w')
            for i in data:
                i = float(i)
                f_en.write('%20.6f\n' % (i/27.21))
            f_en.close()
            for ii in range(1, args.nstat):
                conv_unit_for_grad("grad%d.dat.mndo", "grad%d.dat" % (ii, ii))
            
            for nac in [ f'{b}-{a}' for a, b in combinations(range(1, args.nstat+1), 2) ]:
                conv_unit_for_nac("nac%s.dat.mndo", "nac%s.dat" % (nac, nac))
                # conv_unit_for_nac("nac2-1.dat.mndo", "nac2-1.dat")
                # conv_unit_for_nac("nac3-1.dat.mndo", "nac3-1.dat")
                # conv_unit_for_nac("nac3-2.dat.mndo", "nac3-2.dat")
            # copy results to TEMP
            os.chdir('..')
            state_comb = list(combinations(range(1, args.nstat+1), 2))
            nac_del = ' '.join([f'nac{b}-{a}.dat' for a, b in state_comb]) 
            grad_del = ' '.join([ f'grad{ii}.dat' for ii in range(1, args.nstat+1) ])
            os.system('rm -f energy ' + nac_del + ' ' + grad_del)
            # os.system('rm -f energy grad1.dat grad2.dat grad3.dat nac2-1.dat nac3-1.dat nac3-2.dat')
            os.chdir(rundir)
            copy('energy', '..')
            for i in range(1, args.nstat+1):
                copy('grad%d.dat' % i, '..')
            for i, j in state_comb:
                copy('nac%d-%d.dat' % (j, i), '..')
            # delete files
            nac_del = ' '.join([f'nac{b}-{a}.dat.mndo' for a, b in state_comb]) 
            grad_del = ' '.join([ f'grad{ii}.dat.mndo' for ii in range(1, args.nstat+1) ])
            os.system('rm -f energy.mndo ' + nac_del + ' ' + grad_del)
            # os.system('rm -f energy.mndo grad1.dat.mndo grad2.dat.mndo grad3.dat.mndo nac2-1.dat.mndo nac3-1.dat.mndo nac3-2.dat.mndo')
            nac_del = ' '.join([f'nac{b}-{a}.dat' for a, b in state_comb]) 
            grad_del = ' '.join([ f'grad{ii}.dat' for ii in range(1, args.nstat+1) ])
            os.system('rm -f energy ' + nac_del + ' ' + grad_del)
            # os.system('rm -f energy grad1.dat grad2.dat grad3.dat nac2-1.dat nac3-1.dat nac3-2.dat')
            # mndo engrad
            os.chdir('..')
            os.system('rm -f epot grad.all grad')
            os.system('cp energy  epot')
            os.system(': > grad.all')
            for ii in range(1, args.nstat+1):
            # for ii in [1, 2, 3]:
                os.system('cat grad%d.dat >> grad.all' % ii)
            os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
            # mndo NACs    
            os.system('rm -f nad_vectors')
            # for nac in ['2-1','3-1','3-2']:
            for nac in [ f'{b}-{a}' for a, b in combinations(range(1, args.nstat+1), 2) ]:
                os.system('cat nac%s.dat >> nad_vectors' % nac)

    def aiqm1_calc(engrad_AIQM1, nac_AIQM1, namd, gsmd):
        # get current surface
        if os.path.isfile("curr_surf"):
            with open('curr_surf', 'r') as fcs:
                curr_surf = int(fcs.readlines()[0])
        else: 
            curr_surf = 3
        # AIQM1 MD     
        if engrad_AIQM1 or nac_AIQM1:
            print('From MLatom to NX: calling AIQM1 program')
            mocoef = 'fort.12'
            rundir = 'mndo'
            # create rundir folder
            if os.path.exists(rundir):
                os.system('rm -rf %s' % rundir)
            if namd:
                os.system('cp -r JOB_NAD/mndo %s' % rundir)
            if gsmd:
                os.system('cp -r JOB_AD/mndo %s' % rundir)
            # copy mocoef to rundir
            if namd:
                if os.path.exists(mocoef):
                    os.system('cp %s %s' % (mocoef, rundir))
                    os.chdir(rundir)
                    os.system('cp %s aiqm1' % (mocoef))
                    os.chdir('..')
            if gsmd:
                if os.path.exists(mocoef):
                    os.system('cp %s %s' % (mocoef, rundir))
            # copy geom to rundir
            os.system('cp geom %s' % rundir)
            # enter rundir
            os.chdir(rundir)
            # change geom format
            with open('geom') as f:
                data = f.read().splitlines()
            f_geom = open('geom', 'w')
            for line in data:
                ele, num, x, y, z, abundance = line.split()
                x, y, z = float(x), float(y), float(z)
                f_geom.write('%s%15.8f%2.0f%15.8f%2.0f%15.8f%2.0f\n' % (num, x*0.52917721092, 1, y*0.52917721092, 1, z*0.52917721092, 1))
            f_geom.close()
            if gsmd:
                # change input
                os.system('sed -i \'6r geom\' aiqm1.inp')
                # run mndo
                os.system('$MNDO < aiqm1.inp > aiqm1.out')
                # copy new mocoef to TEMP
                os.system('cp fort.12 ..')
                # collect results 
                os.system('grep \"Final energy (kcal/mol)\" aiqm1.out| awk \'{print $4}\' > energy.aiqm1')
                os.system(" grep -A 9 'GRADIENTS (KCAL' aiqm1.out| sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad.dat.aiqm1")
                # convert unit
                with open('energy.aiqm1') as f:
                    data = f.read().splitlines()
                f_en = open('energy.dat', 'w')
                for i in data:
                    i = float(i)
                    f_en.write('%20.6f\n' % (i/(23.061*27.21)))
                f_en.close()
                conv_unit_for_grad("grad.dat.aiqm1", "grad.dat")
                # copy results to TEMP
                os.chdir('..')
                os.system('rm -f epot grad.all grad')
                os.chdir(rundir)
                copy('energy.dat', '../epot')
                copy('grad.dat', '../grad.all')
                copy('grad.dat', '../grad')
                # delete files
                os.system('rm -f energy.aiqm1 grad.dat.aiqm1')
                os.system('rm -f energy.dat grad.dat')
                os.chdir('..')

            if namd:
                # change input
                os.system('sed -i \'8r geom\' mndo.inp')
                os.system('cp geom aiqm1')
                os.chdir('aiqm1')
                os.system('sed -i \'6r geom\' aiqm1.inp')
                os.chdir('..')
                # run mndo
                os.system('$MNDO_ORIGIN < mndo.inp > mndo.out')
                os.chdir('aiqm1')
                os.system('$MNDO < aiqm1.inp > aiqm1.out')
                os.chdir('..')
                # copy new mocoef to TEMP
                os.system('cp fort.12 ..')
                # collect results 
                for ii in range(1, args.nstat+1):
                    os.system('grep \"State  %d\" mndo.out | awk -FE= \'{print $2}\'| awk \'{print $1}\' >> energy.mndo' % ii)
                os.system(" sed -n '/CI CALCULATION FOR STATE:  1/, /CI CALCULATION FOR STATE:  2/ p' mndo.out | grep -A 9 'GRADIENTS (KCAL' |\
                            sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad1.dat.mndo")
                os.system(" sed -n '/CI CALCULATION FOR STATE:  2/, /CI CALCULATION FOR STATE:  3/ p' mndo.out | grep -A 9 'GRADIENTS (KCAL' |\
                            sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad2.dat.mndo")
                os.system(" sed -n '/CI CALCULATION FOR STATE:  3/, /CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  2   1/ p' mndo.out | \
                            grep -A 9 'GRADIENTS (KCAL' |\
                            sed 1,4d| awk '{printf(\"%20.5f%20.5f%20.5f\\n\", $6,$7,$8)}' > grad3.dat.mndo")     
                os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  2   1/,/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   1/ p' mndo.out | \
                            grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac2-1.dat.mndo")
                os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   1/,/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   2/ p' mndo.out | \
                            grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac3-1.dat.mndo")
                os.system(" sed -n '/CI CALCULATION FOR INTERSTATE COUPLING OF STATES:  3   2/,/SUMMARY OF INTERSTATE COUPLING CALCULATIONS/ p' mndo.out | \
                            grep -A 7 'COMPLETE EXPRESSION'|sed 1,2d | awk '{printf(\"%20.6f%20.6f%20.6f\\n\", $2,$3,$4)}' > nac3-2.dat.mndo")       
                os.chdir('aiqm1')
                os.system("sed -n '2p' energy|sed 's/D/E/g'|awk '{printf\"%20.6f\",$1}' > energy.corr")
                os.system("cat grad |sed 1d |sed 's/D/E/g' | awk '{printf\"%20.5f%20.5f%20.5f\\n\",$1*0.529167,$2*0.529167,$3*0.529167}' > grad.dat.corr")
                os.system('cp *.corr ..')
                os.chdir('..')
                # convert unit
                with open('energy.mndo') as f:
                    data = f.read().splitlines()
                f_en = open('energy.conv', 'w')
                for i in data:
                    i = float(i)
                    f_en.write('%20.6f\n' % (i/27.21))
                f_en.close()

                for ii in range(1, args.nstat):
                    conv_unit_for_grad("grad%d.dat.mndo", "grad%d.dat.conv" % (ii, ii))
                for nac in [ f'{b}-{a}' for a, b in combinations(range(1, args.nstat+1), 2) ]:
                    conv_unit_for_nac("nac%s.dat.mndo", "nac%s.dat" % (nac, nac))
                # calculate aiqm1 results
                with open('energy.conv') as f:
                    data_conv = f.read().splitlines()
                with open('energy.corr') as f:
                    data_corr = f.read().splitlines()
                f_en = open('energy.dat', 'w')
                en_corr = float(data_corr[0])
                for i in data_conv:
                    i = float(i)
                    f_en.write('%20.6f\n' % (i+en_corr))
                f_en.close()
                for ii in range(1, args.nstat+1):
                    add_grad('grad%d.dat.conv' % ii, 'grad.dat.corr', 'grad%d.dat' % ii)
                # copy results to TEMP
                os.chdir('..')
                state_comb = list(combinations(range(1, args.nstat+1), 2))
                nac_del = ' '.join([f'nac{b}-{a}.dat' for a, b in state_comb]) 
                grad_del = ' '.join([ f'grad{ii}.dat' for ii in range(1, args.nstat+1) ])
                os.system('rm -f energy ' + nac_del + ' ' + grad_del)
                # os.system('rm -f energy.dat grad1.dat grad2.dat grad3.dat nac2-1.dat nac3-1.dat nac3-2.dat')
                os.chdir(rundir)
                copy('energy.dat', '..')
                for i in range(1, args.nstat+1):
                    copy('grad%d.dat' % i, '..')
                for i, j in state_comb:
                    copy('nac%d-%d.dat' % (j, i), '..')
                # delete files
                nac_del = ' '.join([f'nac{b}-{a}.dat.mndo' for a, b in state_comb]) 
                grad_del = ' '.join([ f'grad{ii}.dat.mndo' for ii in range(1, args.nstat+1) ])
                os.system('rm -f energy.mndo ' + nac_del + ' ' + grad_del)
                # os.system('rm -f energy.mndo grad1.dat.mndo grad2.dat.mndo grad3.dat.mndo nac2-1.dat.mndo nac3-1.dat.mndo nac3-2.dat.mndo')
                grad_del = ' '.join([ f'grad{ii}.dat.conv' for ii in range(1, args.nstat+1) ])
                os.system('rm -f energy.conv ' + grad_del)
                os.system('rm -f energy.corr grad.dat.corr')
                nac_del = ' '.join([f'nac{b}-{a}.dat' for a, b in state_comb]) 
                grad_del = ' '.join([ f'grad{ii}.dat' for ii in range(1, args.nstat+1) ])
                os.system('rm -f energy.dat ' + nac_del + ' ' + grad_del)
                os.chdir('aiqm1')
                os.system('rm -f energy.corr grad.dat.corr')
                os.chdir('..')
                if engrad_AIQM1:
                    # aiqm1 engrad
                    os.chdir('..')
                    os.system('rm -f epot grad.all grad')
                    os.system('cp energy.dat  epot')
                    os.system(': > grad.all')
                    for ii in range(1, args.nstat+1):
                        os.system('cat grad%d.dat >> grad.all' % ii)
                    os.system("sed -n '%d,%dp' grad.all > grad" % ((curr_surf-1)*args.natoms + 1, curr_surf*args.natoms))
                if nac_AIQM1: 
                    # aiqm1 NACs    
                    os.system('rm -f nad_vectors')
                    for nac in [ f'{b}-{a}' for a, b in combinations(range(1, args.nstat+1), 2) ]:
                    # for nac in ['2-1','3-1','3-2']:
                        os.system('cat nac%s.dat >> nad_vectors' % nac)



class args(object):
    # Default values:
    dirname          = ''
    collectDataForNX = False
    createMLforNX    = False
    useMLforNX       = False
    natoms: int      = None
    nstat: int       = None
    nstatdyn: int    = None
    
    @classmethod
    def parse(cls, argsraw):
        if len(argsraw) == 0:
            printHelp()
            stopper.stopMLatom('At least one option should be provided')
        for arg in argsraw:
            if  (arg.lower() == 'help'
              or arg.lower() == '-help'
              or arg.lower() == '-h'
              or arg.lower() == '--help'):
                printHelp()
                stopper.stopMLatom('')
            elif arg.lower()                     == 'collectDataForNX'.lower():
                cls.collectDataForNX              = True
            elif arg.lower()                     == 'createMLforNX'.lower():
                cls.createMLforNX                 = True
            elif arg.lower()                     == 'useMLforNX'.lower():
                cls.useMLforNX                    = True
            elif arg.lower()[0:len('natoms=')]   == 'natoms='.lower():
                cls.natoms                        = int(arg[len('natoms='):])
            elif arg.lower()[0:len('nstat=')]    == 'nstat='.lower():
                cls.nstat                         = int(arg[len('nstat='):])
            elif arg.lower()[0:len('nstatdyn=')] == 'nstatdyn='.lower():
                cls.nstatdyn                      = int(arg[len('nstatdyn='):])
            elif arg.lower()[0:len('dirname=')]  == 'dirname='.lower():
                cls.dirname                       = arg[len('dirname='):]
            else:
                printHelp()
                stopper.stopMLatom('Option "%s" is not recognized' % arg)
        cls.checkArgs()
            
    @classmethod
    def checkArgs(cls):
        Ntasks = (cls.collectDataForNX + cls.createMLforNX + cls.useMLforNX)
        if Ntasks == 0:
            printHelp()
            stopper.stopMLatom('At least one task should be requested')
        #if cls.collectDataForNX and cls.dirname == '':
        #    printHelp()
        #    stopper.stopMLatom('dirname could not be empty')
            
def printHelp():
    helpText = '''
  !---------------------------------------------------------------------------! 
  !                                                                           ! 
  !       NX_MLatom_interface: Interface between NEWTON-X and MLatom          ! 
  !                                                                           ! 
  !---------------------------------------------------------------------------!
  
  Usage:
    NX_MLatom_interface.py [options]
    
  Options:
      help            print this help and exit
    
    Tasks for NX_MLatom_interface. At least one task should be requested.
      collectDataForNX extracts all necessary information from NEWTON-X outputs to prepare dataset for ML
        dirname=S      name of the directory with a trajectory
      createMLforNX    trains all the necessary machine learning models for NEWTON-X. Suboptions:
      useMLforNX       creates all the necessary files for NEWTON-X. Suboptions:
    Other required arguments:
      natoms=N       number of atoms
      nstat=N        number of states
      nstatdyn=N     initial state
'''
    print(helpText)

if __name__ == '__main__':
    NX_MLatom_interfaceCls()
    
