#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! geomopt: Geometry optimization                                            ! 
  ! Implementations by: Peikun Zheng                                          ! 
  !---------------------------------------------------------------------------! 
'''
import os, sys, re, copy
import numpy as np
import scipy
import MLtasks, stopper
from thermo import hofcalc
from thermo import get_gau_thermo
from AIQM1 import read_mndokw

mlatomdir=os.path.dirname(__file__)
pythonbin = sys.executable

bohr2a = 0.52917721092
idx2element = """ X
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()

class geomoptCls(object):
    def __init__(self, argsopt = sys.argv[1:]):
        
        args.parse(argsopt)

        self.prog = args.optprog.lower()
        self.task = args.opttask.lower()
        self.xyzfile = args.xyzfile
        self.optxyz = args.optxyz
        self.hofmethod = args.hofmethod
        self.fmax = args.fmax
        self.steps = args.steps
        self.optimizer = args.optimizer
        self.linear = args.linear
        self.symmetrynumber = args.symmetrynumber
        self.mndokw = args.mndokw
        self.charges = args.charges
        self.spins = args.spins
        self.aiqm1 = args.aiqm1
        self.ani = args.ani
        self.do_geomopt()

    def do_geomopt(self):
        if   self.prog == 'gaussian':
            Gaussianbin, version = self.check_gaussian()
            print(f' optprog: Gaussian {version}\n')
            self.gau_geomopt(Gaussianbin)
        elif self.prog == 'ase':
            print(f' optprog: ASE\n')
            self.atoms = io.read(self.xyzfile, index=':', format='xyz')
            if self.task == 'opt':
                self.ase_opt(self.fmax, self.steps, self.optimizer)
            elif self.task == 'freq':
                self.ase_freq()
        elif self.prog == 'scipy': # fcge added this line
            print(f' optprog: SciPy\n') # fcge added this line
            if self.task == 'opt': # fcge added this line
                self.scipy_opt() # fcge added this line
            else: # fcge added this line
                stopper.stopMLatom('nothing else than geomopt can be done with optprog=SciPy') # fcge added this line
        # remove temporary files
        os.system('rm -f enest.dat gradest.dat hessest.dat')
        os.system('rm -f mndokw_split* std')
        #os.system('rm -f taskargs xyz_temp.dat')

    def scipy_opt(self):
        stepFactor=100
        with open(self.xyzfile) as fxyz:
            for line in fxyz:
                natom=int(line)
                fxyz.readline()
                elem=[]; coord=[]
                for i in range(natom):
                    sp, *xyz = fxyz.readline().split()
                    elem.append(sp)
                    coord.append(xyz)
                def objfun(coord):
                    write_xyz_tmp(0,0,elem,coord.reshape(-1,3)*stepFactor)
                    os.system('cat xyz_temp.dat')
                    os.system('rm -f enest.dat gradest.dat hessest.dat')
                    args2pass = np.loadtxt('taskargs', dtype=str)
                    # print(args2pass)
                    MLtasks.MLtasksCls(args2pass)
                    # os.system('$mlatom_dev taskargs > mlatom.log')
                    E = np.loadtxt('enest.dat')
                    fx = np.loadtxt('gradest.dat', skiprows=2)
                    # print(E)
                    # return E
                    return (E,fx.flatten())
                res=minimize(objfun,np.array(coord).astype(float).flatten()/stepFactor,jac=True)
                # res=minimize(objfun,np.array(coord).astype(float).flatten()/stepFactor)
                write_xyz_tmp(0,0,elem,res.x.reshape(-1,3)*stepFactor)
                os.system('cat xyz_temp.dat >> ' + self.optxyz)
        
    def gau_geomopt(self, Gaussianbin):
        nmol = 0
        charge = 0; spin = 1
        element2number = {'H':1, 'C':6, 'N':7, 'O':8}
        with open(self.xyzfile, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i].strip().isdigit():
                    element = []; coordinate = []
                    nat = int(lines[i])
                    for j in range(i+2, i+2+nat):
                        element.append(lines[j].strip().split()[0])
                        coordinate.append(lines[j].strip().split()[1:])
                    if element[0].isdigit():
                        number = element
                    else:    
                        number = np.array([element2number[i.upper()] for i in element])
                    number = np.array(number).astype('int')
                    coordinate = np.array(coordinate).astype('float')
                    if self.mndokw is not None:
                        cmd = "sed -i 's/mndokeywords.*/" + f"mndokeywords=mndokw_split{nmol+1}/ig' taskargs"
                        os.system(cmd)
                        charge = self.charges[nmol]; spin = self.spins[nmol]
                    nmol = nmol + 1
                    write_gau_inp(nmol, number, coordinate, charge, spin, self.task)
                    run_gau(nmol, Gaussianbin)
                    if self.aiqm1:
                        cmd = f"sed -n '/Standard dev/, /Total energy/p' mol_{nmol}.log | tail -n 6"  
                        result = os.popen(cmd).read()
                        print(result)
                    elif self.ani:
                        cmd = f"sed -n '/Standard dev/, /Total energy/p' mol_{nmol}.log | tail -n 2"  
                        result = os.popen(cmd).read()
                        print(result)

                    if self.task == 'opt':
                        os.system('cat xyz_temp.dat >> ' + self.optxyz)
                    if self.task == 'freq' and self.hofmethod:
                        energy, ZPE, H298 = get_gau_thermo(nmol, self.hofmethod)
                        hofcalc(self.hofmethod, energy, ZPE, H298, number)

    def ase_opt(self, fmax, steps, optimizer='LBFGS'):
        print(' Forces convergence criterion: %s eV/A' % fmax)
        print(' Maximum number of step: %s' % steps)
        print(' optimization algorithm: %s\n' % optimizer)
        for i in range(len(self.atoms)):
            traj = f'mol_{i+1}_geo.traj'
            calculator = MLatomCalculator(i, self.mndokw)
            self.atoms[i].set_calculator(calculator)

            print(' =============== Begin minimizing ===============')
            #print(' \t', end=' ')
            if optimizer == 'LBFGS':
                opt = optimize.LBFGS(self.atoms[i], trajectory=traj)
            elif optimizer == 'BFGS':
                opt = optimize.BFGS(self.atoms[i], trajectory=traj)
            opt.run(fmax=fmax, steps=steps)
            print('Final energy: %15.6f eV (%20.12f a.u.)\n'
                  % (self.atoms[i].get_potential_energy(), self.atoms[i].get_potential_energy() / ase.units.Hartree))
            
            io.write(self.optxyz, self.atoms[i], format='extxyz', plain=True, append=True)
    
    def ase_freq(self):
        nmol = len(self.atoms)
        linear = np.full(nmol, 0)
        sn = np.full(nmol, 1)
        if self.linear:
            linear = self.linear
        if self.symmetrynumber:
            sn = self.symmetrynumber
        for i in range(nmol):
            numbers = self.atoms[i].get_atomic_numbers()
            mult = 1
            io.write('xyz_temp.dat', self.atoms[i], format='extxyz', plain=True)
            args2pass = np.loadtxt('taskargs', dtype=str)
            if self.mndokw is not None:
                cmd = "sed -i 's/mndokeywords.*/" + f"mndokeywords=mndokw_split{nmol+1}/ig' taskargs"
                os.system(cmd)
            if re.search('aiqm1', ''.join(args2pass), flags=re.IGNORECASE):
                import AIQM1
                _, mults = AIQM1.AIQM1Cls(args2pass).forward()
                mult = mults[0]
            else:   
                args2pass = np.append(args2pass, 'freq')
                MLtasks.MLtasksCls.useMLmodel(args2pass)
            energy, ZPE, H298 = thermocalc(self.atoms[i], linear[i], sn[i], mult)
            if self.hofmethod:
                hofcalc(self.hofmethod, energy, ZPE, H298, numbers)


    @staticmethod
    def check_gaussian():
        status = os.popen('echo $GAUSS_EXEDIR').read().strip()
        if len(status) != 0:
            Gaussianroot = status.split('bsd')[0]
            if 'g16' in Gaussianroot:
                version = 'g16'
            elif 'g09' in Gaussianroot:
                version = 'g09'
            Gaussianbin = Gaussianroot + version
        else :
            stopper.stopMLatom('Can not find Gaussian software in the environment variable, $GAUSS_EXEDIR variable not exists')
        version = version.replace('g', '')
        return Gaussianbin, version


def write_gau_inp(imol, element, coord, charge, spin, task='opt'):
    # opt  -> cmd = #p opt(nomicro) external='python ...'
    # freq -> cmd = #p freq external='python ...'
    # ts   -> cmd = #p opt(ts,calcfc,noeigen,nomicro) external='python ...'
    # irc  -> cmd = #p irc(calcfc,maxpoints=20) external='python ...'
    if   task == 'opt':
        cmd = "#p opt(nomicro) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'freq':
        cmd = "#p freq external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'ts':
        cmd = "#p opt(ts,calcfc,noeigen,nomicro) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'irc':
        cmd = "#p irc(calcfc,maxpoints=20) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    
    gau_inp = f'mol_{imol}.com'
    with open(gau_inp, 'w') as f:
        f.write('%nproc=1\n')
        f.write(cmd)
        f.write('\nTitle Card Required\n')
        f.write(f'\n{charge} {spin}\n')
        for i in range(len(element)):
            f.write('%3s %12.8f %12.8f %12.8f\n' % (element[i], coord[i][0], coord[i][1], coord[i][2]))
        f.write('\n')    

def run_gau(imol, Gaussianbin):
    gau_inp = f'mol_{imol}.com'
    os.system(Gaussianbin + ' ' + gau_inp)

def gau_external(EIn_file, EOu_file):
    # write new coordinate into 'xyz_temp.dat'
    derivs, charge, spin, element, coord = read_gau_EIn(EIn_file)
    write_xyz_tmp(charge, spin, element, coord)
    # calculate energy, grad, hessian in new coordinate
    os.system('rm -f enest.dat gradest.dat hessest.dat')
    args2pass = np.loadtxt('taskargs', dtype=str)
    MLtasks.MLtasksCls(args2pass)
    E = np.loadtxt('enest.dat')
    fx = np.loadtxt('gradest.dat', skiprows=2)
    if derivs != 2:
        write_gau_EOu(EOu_file, derivs, E, fx)
    else:
        ffx = np.loadtxt('hessest.dat', skiprows=2).reshape(-1, len(element)*3)
        write_gau_EOu(EOu_file, derivs, E, fx, ffx)
            
def read_gau_EIn(EIn_file):
    with open(EIn_file, 'r') as f:
        lines = f.readlines()
        line0 = lines[0].strip().split()
        nat = int(line0[0]); derivs = int(line0[1])
        charge = int(line0[2]); spin = int(line0[3])
        
        idx = []; coord = []
        for i in range(1, nat+1):
            idx.append(int(lines[i].strip().split()[0]))
            coord.append(lines[i].strip().split()[1:-1])
        #element = [idx2element[i] for i in idx]
        element = np.array(idx)
        coord = np.array(coord).astype('float') * bohr2a
    
    return derivs, charge, spin, element, coord

def write_xyz_tmp(charge, spin, element, coord):
    with open('xyz_temp.dat', 'w') as f:
        f.write(str(len(element)) + '\n\n')
        for i in range(len(element)):
            f.write('%s %12.8f %12.8f %12.8f\n' % (element[i], coord[i][0], coord[i][1], coord[i][2]))

def write_gau_EOu(EOu_file, derivs, E, fx, ffx=None):
    with open(EOu_file, 'w') as f:
        # energy, dipole-moment (xyz)   E, Dip(I), I=1,3
        writer = ff.FortranRecordWriter('(4D20.12)')
        output = writer.write([E, 0.0, 0.0, 0.0])
        f.write(output)
        f.write('\n')
        
        writer = ff.FortranRecordWriter('(3D20.12)')
        # gradient on atom (xyz)        FX(J,I), J=1,3; I=1,NAtoms
        output = writer.write(fx.flatten()*bohr2a)
        f.write(output)
        f.write('\n')
        if derivs == 2:
            # polarizability                Polar(I), I=1,6
            polor = np.zeros(6)
            output = writer.write(polor)
            f.write(output)
            f.write('\n')
            # dipole derivatives            DDip(I), I=1,9*NAtoms
            ddip = np.zeros(9*len(fx))
            output = writer.write(ddip)
            f.write(output)
            f.write('\n')
            # force constants               FFX(I), I=1,(3*NAtoms*(3*NAtoms+1))/2
            ffx_lower = ffx[np.tril_indices(len(ffx))] * bohr2a**2
            output = writer.write(ffx_lower)
            f.write(output)

class args(object):
    # Default values:
    optprog             = 'Gaussian'
    opttask             = 'opt'    
    xyzfile             = ''
    optxyz              = 'optgeoms.xyz'
    hofmethod           = None
    mndokw              = None
    charges             = None
    spins               = None
    aiqm1               = False
    ani                 = False
    # ASE options
    fmax = 0.02
    steps = 200
    optimizer = 'LBFGS'
    linear = None
    symmetrynumber = None
    #yestfile = 'enest.dat'
    #ygradxyzestfile = 'gradest.dat'

    @classmethod
    def parse(cls, argsraw):
        argslower = [i.lower() for i in argsraw]
        if 'geomopt' in argslower:
            cls.opttask = 'opt'
        elif 'freq' in argslower:
            cls.opttask = 'freq'
        elif 'ts' in argslower:
            cls.opttask = 'ts'
        elif 'irc' in argslower:
            cls.opttask = 'irc'
        deadlist1 = []; deadlist2 = []; deadlist3 = []
        for arg in argsraw:
            flagmatch1 = re.search('(^hyperopt)|(^setname=)|(^learningcurve$)|(^lcntrains)|(^lcnrepeats)|(^deltalearn)|(^yb=)|(^yt=)|(^yestt=)|(^nlayers=)|(^selfcorrect)|(^geomopt)|(^freq)|(^ts)|(^irc)|(^mlprog)|(^initBetas)', arg, flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            flagmatch2 = re.search('(^usemlmodel)|(^mlmodeltype=)|(^xyzfile=)', arg, flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            flagmatch3 = re.search('(^nthreads)|(^usemlmodel)|(^mlmodelin=)|(^mlmodeltype=)', arg, flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if flagmatch1: deadlist1.append(arg)
            if flagmatch2: deadlist2.append(arg)
            if flagmatch3: deadlist3.append(arg)
        # deadlist1 do not need for geomopt and freq
        for i in deadlist1: argsraw.remove(i)
        args2pass = copy.deepcopy(argsraw)
        if re.search('ani1ccx', ''.join(args2pass), flags=re.IGNORECASE):
            cls.hofmethod = 'ani1ccx'
        if 'ani' in ''.join(args2pass).lower():
            cls.ani = True
        # deadlist3 do not need for aiqm1
        for i in deadlist3: argsraw.remove(i)
        if 'aiqm1' in ''.join(args2pass).lower():
            cls.aiqm1 = True
            if 'aiqm1dft' in ''.join(args2pass).lower():
                cls.hofmethod = 'aiqm1dft'
            elif 'aiqm1dftstar' in ''.join(args2pass).lower():
                cls.hofmethod = 'aiqm1dftstart'
            else:
                cls.hofmethod = 'aiqm1'
            for i in deadlist3: args2pass.remove(i)
        # args in deadlist2 are needed for other mlmodel
        elif len(deadlist2) < 3:
            printHelp()
            stopper.stopMLatom('')

        for arg in argsraw:

            if  (arg.lower() == 'help'
              or arg.lower() == '-help'
              or arg.lower() == '-h'
              or arg.lower() == '--help'):
                printHelp()
                stopper.stopMLatom('')
            
            elif arg.lower()[0:len('optprog=')]          == 'optprog=':  
                cls.optprog = arg[len('optprog='):]
            elif arg.lower()[0:len('xyzfile=')]          == 'xyzfile=':  
                cls.xyzfile = arg[len('xyzfile='):]
            elif arg.lower()[0:len('optxyz=')]          == 'optxyz=':  
                cls.optxyz = arg[len('optxyz='):]
            elif arg.lower()[0:len('ase.fmax')]          == 'ase.fmax':
                cls.fmax = float(arg[len('ase.fmax='):])
            elif arg.lower()[0:len('ase.steps')]         == 'ase.steps':
                cls.steps = int(arg[len('ase.steps='):])
            elif arg.lower()[0:len('ase.optimizer')]     == 'ase.optimizer':
                cls.optimizer = arg[len('ase.optimizer='):]
            elif arg.lower()[0:len('ase.linear='):]      == 'ase.linear=':
                cls.linear = arg[len('ase.linear='):].split(',')
                cls.linear = [int(i) for i in cls.linear]
            elif arg.lower()[0:len('ase.symmetrynumber='):]  == 'ase.symmetrynumber=':
                cls.symmetrynumber = arg[len('ase.symmetrynumber='):].split(',')
                cls.symmetrynumber = [int(i) for i in cls.symmetrynumber]
            elif arg.lower()[0:len('mndokeywords='):]    == 'mndokeywords=':
                cls.mndokw = arg[len('mndokeywords='):]
                _, charges, spins = read_mndokw(cls.mndokw, True)
                cls.charges = charges
                cls.spins = spins
            #else:
            #    printHelp()
            #    stopper.stopMLatom('Option "%s" is not recognized' % arg)
        deadlist = []
        # not store geomopt args in taskargs
        for arg in args2pass:
            flagmatch = re.search('(^opt)|(^ase)|^(xyzfile)', arg, flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if flagmatch: deadlist.append(arg)
        for i in deadlist: args2pass.remove(i)

        cls.checkArgs()
        #args2pass.append('useMLmodel')
        args2pass.append('xyzfile=xyz_temp.dat')
        args2pass.append('yestfile=enest.dat')
        args2pass.append('ygradxyzestfile=gradest.dat')
        if cls.opttask in ['opt', 'ts', 'irc']:
            if os.path.isfile(cls.optxyz):
                stopper.stopMLatom(f'File {cls.optxyz} already exists, delete or rename it')
        if cls.opttask in ['freq', 'ts', 'irc']:
            args2pass.append('hessianestfile=hessest.dat')
        np.savetxt('taskargs', args2pass, fmt='%s')

    @classmethod
    def checkArgs(cls):
        # if cls.optprog.lower() != 'gaussian' and cls.optprog.lower() != 'ase':
        if cls.optprog.lower() != 'gaussian' and cls.optprog.lower() != 'ase' and cls.optprog.lower() != 'scipy': # fcge modified this line
            printHelp()
            stopper.stopMLatom('unrecognized geometry optimization program')
        if cls.optprog.lower() == 'ase' and cls.opttask.lower() not in ['opt', 'freq']:
            printHelp()
            stopper.stopMLatom('ASE program can only use with opttask=opt, freq')
        if cls.optprog.lower() == 'scipy' and cls.opttask.lower() not in ['opt']: # fcge added this line
            printHelp() # fcge added this line
            stopper.stopMLatom('optimization with SciPy only works with opttask=opt') # fcge added this line
        if cls.optprog.lower() == 'gaussian':
            try:
                import fortranformat as ff
            except:
                stopper.stopMLatom('Please install Python module fortranformat')
            globals()['ff'] = ff
        if cls.optprog.lower() == 'ase':
            try:
                import ase
                from ase import io
                from ase import optimize
                import calculator
                from calculator import MLatomCalculator
                from thermo import thermocalc
            except:
                stopper.stopMLatom('Please install ASE')
            globals()['ase'] = ase
            globals()['io'] = io
            globals()['optimize'] = optimize
            globals()['MLatomCalculator'] = MLatomCalculator
            globals()['thermocalc'] = thermocalc
        if cls.optprog.lower() == 'scipy': # fcge added this line
            try: from scipy.optimize import minimize # fcge added this line
            except: stopper.stopMLatom('Please install SciPy') # fcge added this line
            globals()['minimize'] = minimize # fcge added this line
            
def printHelp():
    helpText = '''
  !---------------------------------------------------------------------------! 
  !                         Geometry optimization options                     ! 
  !---------------------------------------------------------------------------!
  
  Usage:
  MLatom.py [geomopt|freq] usemlmodel mlmodelin=... mlmodeltype=... xyzfile=... [options]
  or 
  MLatom.py [geomopt|freq] aiqm1 xyzfile=... [options]

  geomopt [default]          Optimization for energy minimum
  freq                       Frequence analysis for input geometry
  Options:
      optprog=S                     Geometry optimization program S
         Gaussian [default]
         ASE
      optxyz=S               save optimized geometries in file S [default: optgeoms.xyz]

      The following options only used for ASE program:
        ase.fmax=R                    threshold of maximum force (in eV/A)
                                      [default values: 0.02]
        ase.steps=N                   maximum steps
                                      [default values: 200]
        ase.optimizer=S               optimizer
           LBFGS [default]
           BFGS
        when do frequence analysis, the following options are also required:
        ase.linear=N,...,N            0 for nonlinear molecule, 1 for linear molecule
                                      [default vaules: 0]
        ase.symmetrynumber=N,...,N    rotational symmetry number for each molecule
                                      [default vaules: 1]
'''
    print(helpText)


if __name__ == '__main__': 
    try:
        import fortranformat as ff
    except:
        stopper.stopMLatom('Please install Python module fortranformat')
    globals()['ff'] = ff
    
    _, EIn_file, EOu_file, _, _, _ = sys.argv[1:]      
    gau_external(EIn_file, EOu_file)

