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
from AIQM1_cmd import read_mndokw
from AIQM1_cmd import check_element
from utils import *
from args_class import ArgsBase 

mlatomdir=os.path.dirname(__file__)
pythonbin = sys.executable

bohr2a = 0.52917721092

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.add_default_dict_args([
            'geomopt', 'freq', 'ts', 'irc', 'useMLmodel',
            'AIQM1DFTstar', 'AIQM1DFT', 'AIQM1', 'ODM2', 'ODM2star',
            'ani1x', 'ani2x', 'ani1ccx', 'ani1xd4', 'ani2xd4',
            'gfn2xtb'
            ],
            bool
        )
        self.add_default_dict_args([
            'opttrajxyz', 'opttrajh5', 'hofmethod', 'MLmodelIn', 'MLmodelType',
            'qmprog'
            ],
            None
        )        
        self.add_dict_args({
            'xyzfile': None,
            'optprog': None,
            'opttask': None,
            'optxyz': "optgeoms.xyz",
            'yestfile': None,
            'ygradxyzestfile': None,
            'hessianestfile': None,
            'mndokeywords': None,
            'gaussiankeywords': None,
            'charges': None,
            'spins': None,
            'addDelta': False,
            'mlmodel2in': None,
            'mlmodel2type': "KREG",
            # ASE options
            #'ase.fmax': 0.02,
            #'ase.steps': 200,
            #'ase.optimizer': "LBFGS",
            #'ase.linear': None,
            #'ase.symmetrynumber': None

            
        })
        self.set_ignore_keyword_list([
            'initBetas=RR',
            'hyperopt.max_evals=8',
            'hyperopt.algorithm=tpe',
            'hyperopt.losstype=geomean',
            'hyperopt.w_y=1',
            'hyperopt.w_ygrad=1',
            'hyperopt.points_to_evaluate=0'
             ])
        self.parse_input_content([
            'ase.fmax=0.02',
            'ase.steps=200',
            'ase.optimizer=LBFGS',
            'ase.linear=',
            'ase.symmetrynumber='
            ])

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.argProcess()

    def argProcess(self):
        if self.geomopt:
            self.opttask = 'opt' 
        elif self.freq:
            self.opttask = 'freq' 
        elif self.ts:
            self.opttask = 'ts' 
        elif self.irc:
            self.opttask = 'irc' 

        if not self.optprog: # fcge added this
            if 'GAUSS_EXEDIR' in os.environ.keys():
                self.optprog = 'gaussian'
            else:
                try:
                    import ase
                    self.optprog = 'ase'
                except:
                    try:
                        from scipy.optimize import minimize
                        self.optprog = 'scipy'
                    except:
                        printHelp()
                        stopper.stopMLatom('please set $GAUSS_EXEDIR or install ase or install scipy ')
        self.optprog = self.optprog.lower()               
        if self.optprog not in ['gaussian', 'ase', 'scipy']:
            printHelp()
            stopper.stopMLatom('unrecognized geometry optimization program')
        if self.optprog == 'ase' and self.opttask not in ['opt', 'freq']:
            printHelp()
            stopper.stopMLatom('ASE program can only use for geometry optimization and frequence calculation')
        if self.optprog == 'scipy' and self.opttask not in ['opt']: # fcge added this line
            printHelp() # fcge added this line
            stopper.stopMLatom('Scipy can only use for geometry optimization') 
        if self.optprog == 'gaussian':
            try:
                import fortranformat as ff
            except:
                stopper.stopMLatom('Please install python module fortranformat')
            globals()['ff'] = ff
        if self.optprog == 'ase':
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
        if self.optprog == 'scipy': # fcge added this line
            try: from scipy.optimize import minimize # fcge added this line
            except: stopper.stopMLatom('Please install SciPy') # fcge added this line
            globals()['minimize'] = minimize # fcge added this line

        if self.opttask in ['opt', 'ts', 'irc']:
            if os.path.isfile(self.optxyz):
                stopper.stopMLatom(f'File {self.optxyz} already exists, delete or rename it')
            #if os.path.isfile(self.opttrajxyz):
            #    stopper.stopMLatom(f'File {self.opttrajxyz} already exists, delete or rename it')
            #if os.path.isfile(self.opttrajh5):
            #    stopper.stopMLatom(f'File {self.opttrajh5} already exists, delete or rename it')
        
        if self.ase.linear:
            self.ase.linear = [int(i) for i in self.ase.linear.split(',')]
        if self.ase.symmetrynumber:
            self.ase.symmetrynumber = [int(i) for i in self.ase.symmetrynumber.split(',')]
        
        if self.gaussiankeywords:
            self.gaussiankeywords = self.gaussiankeywords.strip("'").strip("\"")
        
        args2pass = []
        if self.aiqm1: 
            self.hofmethod = 'aiqm1'
            args2pass.append('AIQM1')
            args2pass.append(f'qmprog={self.qmprog}')
            if self.mndokeywords:
                _, charges, spins = read_mndokw(self.mndokeywords, True)
                self.charges = charges
                self.spins = spins
                args2pass.append(f'mndokeywords={self.mndokeywords}')
        elif self.aiqm1dftstar:
            self.hofmethod = 'aiqm1dftstar'
            args2pass.append('AIQM1@DFT*')
            args2pass.append(f'qmprog={self.qmprog}')
        elif self.aiqm1dft: 
            self.hofmethod = 'aiqm1dft'
            args2pass.append('AIQM1@DFT')
            args2pass.append(f'qmprog={self.qmprog}')
        elif self.ODM2:
            args2pass.append('ODM2')
            args2pass.append(f'qmprog={self.qmprog}')
            if self.mndokeywords:
                _, charges, spins = read_mndokw(self.mndokeywords, True)
                self.charges = charges
                self.spins = spins
                args2pass.append(f'mndokeywords={self.mndokeywords}')
        elif self.ODM2star:
            args2pass.append('ODM2*')
            args2pass.append(f'qmprog={self.qmprog}')
            if self.mndokeywords:
                _, charges, spins = read_mndokw(self.mndokeywords, True)
                self.charges = charges
                self.spins = spins
                args2pass.append(f'mndokeywords={self.mndokeywords}')
        elif self.ani1ccx:
            self.hofmethod = 'ani1ccx'
            args2pass.append('ANI-1ccx')
        elif self.ani1x:
            args2pass.append('ANI-1x')
        elif self.ani2x:
            args2pass.append('ANI-2x')
        elif self.ani1xd4:
            args2pass.append('ANI-1x-D4')
        elif self.ani2xd4:
            args2pass.append('ANI-2x-D4')
        elif self.gfn2xtb:
            args2pass.append('GFN2-xTB')
        elif self.usemlmodel:
            args2pass.append('useMLmodel')
            if not self.MLmodelIn or not self.MLmodelType:
                printHelp()
                stopper.stopMLatom('')
            else:
                args2pass.append(f'MLmodelIn={self.MLmodelIn}')
                args2pass.append(f'MLmodelType={self.MLmodelType}')
        else:
            printHelp()
            stopper.stopMLatom('')
        args2pass.append('xyzfile=xyz_temp.dat')
        args2pass.append('yestfile=enest.dat')
        args2pass.append('ygradxyzestfile=gradest.dat')
        
        if self.opttask in ['freq', 'ts', 'irc']:
            args2pass.append('hessianestfile=hessest.dat')
        np.savetxt('taskargs', args2pass, fmt='%s')

class geomoptCls(Args):
    #def __init__(self, argsopt = sys.argv[1:]):
    def __init__(self, argsGeomopt):
        super().__init__()
        
        #args = Args() 
        #args.parse(argsGeomopt)
        self.parse(argsGeomopt)

        #args.parse(argsopt)
        #self.prog = args.optprog.lower()
        #self.task = args.opttask.lower()
        #self.xyzfile = args.xyzfile
        #self.optxyz = args.optxyz
        #self.hofmethod = args.hofmethod
        #self.fmax = args.fmax
        #self.steps = args.steps
        #self.optimizer = args.optimizer
        #self.linear = args.linear
        #self.symmetrynumber = args.symmetrynumber
        #self.mndokw = args.mndokw
        #self.charges = args.charges
        #self.spins = args.spins
        #self.aiqm1 = args.aiqm1
        #self.ani = args.ani
        #self.addDelta = args.addDelta
        #self.mlmodel2in = args.mlmodel2in
        #self.mlmodel2type = args.mlmodel2type
        #self.opttrajxyz = args.opttrajxyz
        #self.opttrajh5 = args.opttrajh5
        #self.gaukw = args.gaukw
        self.do_geomopt()

    def do_geomopt(self):   
        if   self.optprog == 'gaussian':
            Gaussianbin, version = self.check_gaussian()
            print(f' optprog: Gaussian {version}\n')
            self.gau_geomopt(Gaussianbin)
        elif self.optprog == 'ase':
            print(f' optprog: ASE\n')
            self.atoms = io.read(self.xyzfile, index=':', format='xyz')
            if self.opttask == 'opt':
                self.ase_opt(self.ase.fmax, self.ase.steps, self.ase.optimizer)
            elif self.opttask == 'freq':
                self.ase_freq()
        elif self.optprog == 'scipy': # fcge added this line
            print(f' optprog: SciPy\n') # fcge added this line
            if self.opttask == 'opt': # fcge added this line
                self.scipy_opt() # fcge added this line
            else: # fcge added this line
                stopper.stopMLatom('nothing else than geomopt can be done with optprog=SciPy') # fcge added this line
        # remove temporary files
        os.system('rm -f enest.dat gradest.dat hessest.dat')
        os.system('rm -f *_temp.traj')
        os.system('rm -f mndokw_split* std')
        #os.system('rm -f taskargs xyz_temp.dat')

    def scipy_opt(self):
        stepFactor=100
        with open(self.xyzfile) as fxyz:
            mol_id=0
            for line in fxyz:
                mol_id+=1
                natom=int(line)
                fxyz.readline()
                elem=[]; coord=[]
                for i in range(natom):
                    sp, *xyz = fxyz.readline().split()
                    elem.append(sp)
                    coord.append(xyz)
                coord=np.array(coord).astype(float)
                if self.opttrajh5:
                    h5=optTrajH5(f'{self.opttrajh5}_{mol_id}.h5',natom,sp)
                def objfun(coord):
                    write_xyz_tmp(0,0,elem,coord.reshape(-1,3)*stepFactor)
                    if self.opttrajxyz:
                        os.system(f'cat xyz_temp.dat >> {self.opttrajxyz}_{mol_id}.xyz')
                    # os.system('cat xyz_temp.dat')
                    os.system('rm -f enest.dat gradest.dat hessest.dat')
                    args2pass = np.loadtxt('taskargs', dtype=str)
                    MLtasks.MLtasksCls(args2pass)
                    E = np.loadtxt('enest.dat')
                    fx = np.loadtxt('gradest.dat', skiprows=2)
                    if self.opttrajh5:
                        h5.stepForward(xyz=coord.reshape(-1,3)*stepFactor,grad=fx,ep=E)
                    return (E,fx.flatten())
                    
                def objfun2(coord):
                    args2pass = np.loadtxt('taskargs', dtype=str)
                    if os.path.isfile('enest.dat') and os.path.isfile('gradest.dat'):
                        pass
                    else:
                        write_xyz_tmp(0,0,elem,coord.reshape(-1,3)*stepFactor)
                        os.system('rm -f enest.dat gradest.dat hessest.dat')
                        MLtasks.MLtasksCls(args2pass)
                    E = np.loadtxt('enest.dat')

                    os.system('rm -f enest2.dat gradest2.dat')
                    args22pass=addReplaceArg('mlmodelin',f'mlmodelin={self.mlmodel2in}',list(args2pass))
                    args22pass=addReplaceArg('mlmodeltype',f'mlmodeltype={self.mlmodel2type}',args22pass)
                    args22pass=addReplaceArg('yestfile',f'yestfile=enest2.dat',args22pass)
                    args22pass=addReplaceArg('ygradxyzestfile',f'ygradxyzestfile=gradest2.dat',args22pass)
                    MLtasks.MLtasksCls(args22pass)
                    E2 = np.loadtxt('enest2.dat')
                    return E-E2
                def objjac2(coord):
                    E = np.loadtxt('enest.dat')
                    E2 = np.loadtxt('enest2.dat')                                                
                    print("MECP energies:",E,E2,E-E2)

                    fx = np.loadtxt('gradest.dat', skiprows=2)
                    fx2 = np.loadtxt('gradest2.dat', skiprows=2)
                    return (fx-fx2).flatten()

                res=minimize(objfun,coord.flatten()/stepFactor,jac=True, constraints={'type':'eq','fun':objfun2,'jac':objjac2} if self.mlmodel2in else None)
                print(f' optimization success: {res.success}')
                write_xyz_tmp(0,0,elem,res.x.reshape(-1,3)*stepFactor)
                os.system('cat xyz_temp.dat >> ' + self.optxyz)
                if self.opttrajh5: h5.close()
        
    def gau_geomopt(self, Gaussianbin):
        nmol = 0
        charge = 0; spin = 1
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
                        element = np.array([number2element[int(i)] for i in number])
                    else:    
                        element = np.array(element)
                        number = np.array([element2number[i.upper()] for i in element])
                    number = np.array(number).astype('int')
                    coordinate = np.array(coordinate).astype('float')
                    if self.aiqm1: check_element([element])
                    if self.mndokeywords is not None:
                        cmd = "sed -i 's/mndokeywords.*/" + f"mndokeywords=mndokw_split{nmol+1}/ig' taskargs"
                        os.system(cmd)
                        charge = self.charges[nmol]; spin = self.spins[nmol]
                    nmol = nmol + 1
                    write_gau_inp(nmol, number, coordinate, charge, spin, self.opttask, self.gaussiankeywords)
                    run_gau(nmol, Gaussianbin)
                    if self.opttrajxyz or self.opttrajh5:
                        savetraj(self.opttrajxyz, self.opttrajh5, nmol, nat, element)

                    if self.aiqm1:
                        cmd = f"sed -n '/Standard dev/, /Total energy/p' mol_{nmol}.log | tail -n 6"  
                        result = os.popen(cmd).read()
                        print(result)
                    elif self.ODM2 or self.ODM2star:
                        cmd = f"sed -n '/Total energy/p' mol_{nmol}.log | tail -n 1"  
                        result = os.popen(cmd).read()
                        print(result)
                    elif self.ani1ccx or self.ani1x or self.ani2x:
                        cmd = f"sed -n '/Standard dev/, /Total energy/p' mol_{nmol}.log | tail -n 2"  
                        result = os.popen(cmd).read()
                        print(result)
                    elif self.ani1xd4 or self.ani2xd4:
                        cmd = f"sed -n '/Standard dev/, /Total energy/p' mol_{nmol}.log | tail -n 4"  
                        result = os.popen(cmd).read()
                        print(result)
                    if self.addDelta:
                        cmd = f'''
                              grep 'Delta energy correction' mol_{nmol}.log | tail -n 1 && 
                              grep 'Corrected energy' mol_{nmol}.log | tail -n 1           
                              '''
                        result = os.popen(cmd).read()
                        print(result)

                    if self.opttask in ['opt','ts']:
                        os.system('cat xyz_temp.dat >> ' + self.optxyz)
                    if self.opttask == 'freq' and self.hofmethod:
                        energy, ZPE, H298 = get_gau_thermo(nmol, self.hofmethod)
                        hofcalc(self.hofmethod, energy, ZPE, H298, number)

    def ase_opt(self, fmax, steps, optimizer='LBFGS'):
        print(' Forces convergence criterion: %s eV/A' % fmax)
        print(' Maximum number of steps: %s' % steps)
        print(' optimization algorithm: %s\n' % optimizer)
        for i in range(len(self.atoms)):
            element = self.atoms[i].get_chemical_symbols()
            nat = len(element)
            if self.aiqm1: check_element([np.array(element)])

            traj = f'mol_{i+1}_ase.traj'
            calculator = MLatomCalculator(i, self.mndokeywords)
            self.atoms[i].set_calculator(calculator)

            #print(' \t', end=' ')
            if optimizer == 'LBFGS':
                opt = optimize.LBFGS(self.atoms[i], trajectory=traj)
            elif optimizer == 'BFGS':
                opt = optimize.BFGS(self.atoms[i], trajectory=traj)
            print(' =============== Begin minimizing ===============')
            opt.run(fmax=fmax, steps=steps)
            print(' Final energy: %15.6f eV (%20.12f Hartree)\n'
                  % (self.atoms[i].get_potential_energy(), self.atoms[i].get_potential_energy() / ase.units.Hartree))
            if opt.nsteps == steps:
                print(' * Warning * Maximum number of steps reached, optimization has not been finished and terminated!')
            io.write(self.optxyz, self.atoms[i], format='extxyz', plain=True, append=True)
            if self.opttrajxyz or self.opttrajh5:
                savetraj(self.opttrajxyz, self.opttrajh5, i+1, nat, element)
            print(' %s \n' % ('-'*78))
    
    def ase_freq(self):
        nmol = len(self.atoms)
        linear = np.full(nmol, 0)
        sn = np.full(nmol, 1)
        if self.ase.linear:
            linear = self.ase.linear
        if self.ase.symmetrynumber:
            sn = self.ase.symmetrynumber
        for i in range(nmol):
            numbers = self.atoms[i].get_atomic_numbers()
            mult = 1
            io.write('xyz_temp.dat', self.atoms[i], format='extxyz', plain=True)
            args2pass = np.loadtxt('taskargs', dtype=str)
            if self.mndokeywords is not None:
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


def write_gau_inp(imol, element, coord, charge, spin, task='opt', gaukw=None):
    # opt  -> cmd = #p opt(nomicro) external='python ...'
    # freq -> cmd = #p freq external='python ...'
    # ts   -> cmd = #p opt(ts,calcfc,noeigen,nomicro) external='python ...'
    # irc  -> cmd = #p irc(calcfc,maxpoints=20) external='python ...'
    if   task == 'opt':
        cmd = "#p opt(nomicro) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'freq':
        cmd = "#p freq geom=nocrowd external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'ts':
        cmd = "#p opt(ts,calcfc,noeigen,nomicro) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)
    elif task == 'irc':
        cmd = "#p irc(calcfc) external='%s %s/geomopt.py'\n" % (pythonbin, mlatomdir)

    if gaukw:
        cmd = "#p %s external='%s %s/geomopt.py'\n" % (gaukw, pythonbin, mlatomdir)
    
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
    derivs, charge, spin, number, coord = read_gau_EIn(EIn_file)
    write_xyz_tmp(charge, spin, number, coord)
    nat = len(number)
    element = np.array([number2element[int(i)] for i in number])
    # calculate energy, grad, hessian in new coordinate
    os.system('rm -f enest.dat gradest.dat hessest.dat')
    args2pass = np.loadtxt('taskargs', dtype=str)
    MLtasks.MLtasksCls(args2pass)
    E = np.loadtxt('enest.dat')
    fx = np.loadtxt('gradest.dat', skiprows=2)
    
    os.system('cat xyz_temp.dat >> xyz_temp.traj')
    os.system('cat enest.dat >> en_temp.traj')
    os.system('cat gradest.dat >> grad_temp.traj')

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
        
        number = []; coord = []
        for i in range(1, nat+1):
            number.append(int(lines[i].strip().split()[0]))
            coord.append(lines[i].strip().split()[1:-1])
        number = np.array(number)
        coord = np.array(coord).astype('float') * bohr2a
    
    return derivs, charge, spin, number, coord

def write_xyz_tmp(charge, spin, number, coord):
    with open('xyz_temp.dat', 'w') as f:
        f.write(str(len(number)) + '\n\n')
        for i in range(len(number)):
            f.write('%s %12.8f %12.8f %12.8f\n' % (number[i], coord[i][0], coord[i][1], coord[i][2]))

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

def get_xyz(xyzfile):
    coordinates = []; numbers = []; elements = []
    with open(xyzfile, 'r') as f:
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
                    element = np.array([number2element[int(i)] for i in number])
                else:    
                    number = np.array([element2number[i.upper()] for i in element])
                number = np.array(number).astype('int')
                coordinate = np.array(coordinate).astype('float')
                elements.append(element)
                numbers.append(number)
                coordinates.append(coordinate)
    return numbers, elements, coordinates

class optTrajH5(object):
    def __init__(self,h5file,natom,sp=None) -> None:
        from pyh5md import File, element
        self.natom=natom
        self.h5=File(h5file,'w',creator='MLatom')
        self.part=self.h5.particles_group('all')
        self.part.create_box(dimension=3, boundary=['none','none','none'])
        self.xyz=element(self.part,'position',shape=(natom,3),dtype=np.float64, time=None,store='time')
        self.grad=element(self.part,'gradients',shape=(natom,3),dtype=np.float64, time=None,store='time')
        # element(self.part, 'species', data=np.array(z), store='fixed')
        if sp is not None: element(self.part, 'names', data=np.array(sp,dtype='|S2'), store='fixed')
        #if sp: element(self.part, 'names', data=np.array(sp,dtype='|S2'), store='fixed')
        self.h5.observables = self.h5.require_group('observables')
        self.ep=element(self.h5.observables,'potential_energy', shape=(), dtype=np.float64, time=None,store='time')
        self.step=0
    def close(self):
        self.h5.close()        
    def stepForward(self,xyz=None,grad=None,ep=None):
        if xyz is not None:     self.xyz.append(xyz,self.step)
        if grad is not None:    self.grad.append(grad,self.step)
        if ep is not None:      self.ep.append(ep,self.step)
        self.step+=1    

def savetraj(opttrajxyz, opttrajh5, mol_id, nat, element):
    if opttrajxyz:
        os.system(f'cat xyz_temp.traj >> {opttrajxyz}_{mol_id}.xyz')
    if opttrajh5:
        h5 = optTrajH5(f'{opttrajh5}_{mol_id}.h5', nat, element)
        E_traj = np.loadtxt('en_temp.traj').reshape(-1)
        _, _, xyz_traj = get_xyz('xyz_temp.traj')
        grad_traj = readXYZgrads('grad_temp.traj')
        for j in range(len(E_traj)):
            h5.stepForward(xyz=xyz_traj[j], grad=grad_traj[j], ep=E_traj[j])
        h5.close()
    os.system('rm -f xyz_temp.traj en_temp.traj grad_temp.traj')

            
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
         scipy [default]
         Gaussian
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

