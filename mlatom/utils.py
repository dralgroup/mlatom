'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! utils.py: Module with some utils                                          ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''
import os, copy
import numpy as np

def getError(yfile=None,yestfile=None,ygradxyzfile=None,ygradxyzestfile=None,ygradfile=None,ygradestfile=None,errorType="RMSE"):
    def calError(yr,yp):
        if errorType.lower()=='rmse':
            return np.sqrt(np.mean(np.square(yp-yr)))
        if errorType.lower()=='mae':
            return np.mean(np.abs(yp-yr))
        if errorType.lower()=='median':
            return np.median((np.mean(np.abs(yp-yr),axis=1)))

    def loadXYZ(fname):
        xyz=[]
        with open(fname) as f:
            for line in f:
                xyz_=[]
                natom=int(line)
                f.readline()
                for i in range(natom):
                    _xyz_=f.readline().split()[-3:]
                    xyz_.append(_xyz_)
                xyz.append(np.array(xyz_).astype(float))
        return np.array(xyz)

    errdict={"type":errorType}
    if yfile and yestfile:
        yr=np.loadtxt(yfile)[:,np.newaxis]
        yp=np.loadtxt(yestfile)[:,np.newaxis]
        errdict['y']=calError(yr,yp)
    if ygradfile and ygradestfile:
        yr=np.loadtxt(ygradfile)
        yp=np.loadtxt(ygradestfile)
        errdict['ygrad']=calError(yr,yp)
    if ygradxyzfile and ygradxyzestfile:
        yr=loadXYZ(ygradxyzfile)
        yp=loadXYZ(ygradxyzestfile)
        errdict['ygradxyz']=calError(yr,yp)
    return errdict
     
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

def saveXYZ(fname, element, coord, number=None, charge=0, spin=1):
    with open(fname, 'w') as f:
        f.write(str(len(element)) + '\n\n')
        for i in range(len(element)):
            f.write('%s %12.8f %12.8f %12.8f\n' % (element[i], coord[i][0], coord[i][1], coord[i][2]))


periodic_table = """ X
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
number2element = {k: v for k, v in enumerate(periodic_table)}
element2number = {v: k for k, v in number2element.items()}

def loadXYZ(fname,dtype=np.array,getsp=True):
    xyz=[]
    sp=[]
    with open(fname) as f:
        for line in f:
            xyz_=[]
            sp_=[]
            natom=int(line)
            f.readline()
            for _ in range(natom):
                if getsp: 
                    _sp_,*_xyz_=f.readline().split()
                    sp_.append(_sp_)
                else: _xyz_=f.readline().split()
                xyz_.append(_xyz_)
            xyz.append(np.array(xyz_).astype(float))
            sp.append(np.array(sp_))
    return dtype(xyz),dtype(sp)

def saveXYZs(fname,xyzs,sp=None,mode='w',msgs=None):
    xyzs=xyzs.reshape(-1,sp.shape[0],3)
    with open(fname,mode)as f:
        for xyz in xyzs:
            f.write("%d\n"%sp.shape[0])
            if msgs: f.write(msgs.pop(0)+"\n")
            else: f.write('\n')
            for j in range(sp.shape[0]):
                if sp is not None: f.write(sp[j]+' ')
                f.write('%20.8f %20.8f %20.8f\n'%tuple(xyz[j]))

    
