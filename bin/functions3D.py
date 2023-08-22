import numpy as np
import os 
import interface_MLatomF
import re
from stopper import stopMLatom
from interfaces.sGDML     import interface_sGDML
from interfaces.PhysNet   import interface_PhysNet
from interfaces.GAP       import interface_GAP
from interfaces.DeePMDkit import interface_DeePMDkit
from interfaces.TorchANI  import interface_TorchANI

# Read xyz and vxyz files
def readXYZ(filename):
    rawdata = np.genfromtxt(filename,dtype=str,skip_header=2)
    Natoms = len(rawdata)
    Aname_ = [rawdata[i][0] for i in range(Natoms)]
    Aname = []
    for each in Aname_:
        try:
            idx = eval(each)
            if type(idx) == int:
                Aname.append(idx2element[idx])
        except:
            Aname.append(each)
    xyz = [[eval(rawdata[i][j]) for j in [-3,-2,-1]] for i in range(Natoms)]

    return Natoms, Aname, xyz

# Read xyz files 
def readXYZs(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        Anames = []
        Coords = []
        ii = 0
        while ii < len(lines):
            Natoms = eval(lines[ii].strip())
            a = []
            c = []
            for iatom in range(Natoms):
                raw = lines[ii+2+iatom].strip().split()
                a.append(raw[0])
                c.append([eval(each) for each in raw[1:]])
            Anames.append(a)
            Coords.append(c)
            ii += Natoms+2
    return Anames, Coords
          
# Read Gaussian input file for molecule information (element symbols & coordinates)
def readCom(input_file):
    file_name = input_file.split('/')[-1][:-4]
    rawdata = np.genfromtxt(input_file,usecols=[0,1,2,3],dtype=str,skip_header=6)
    #print(rawdata)
    atom = [rawdata[iatom][0] for iatom in range(len(rawdata))]
    coord = [[eval(rawdata[iatom][i]) for i in range(1,4)] for iatom in range(len(rawdata))]
    
    return file_name, atom, coord

# Write Gaussian input file with specified information (keywords & coordinates)
# nproc 
def writeCom(nproc,keywords,output_file,ichg,imul,atom,coord,link0=False):

    param_line = '#'+keywords
    title_line = output_file
    charge_line = str(ichg)+' '+str(imul)
    with open(output_file+'.com','w') as fcom:
        fcom.write('%'+'nproc=%d\n'%nproc)
        if link0:
            fcom.write(r'%chk='+output_file+'.chk\n')
        fcom.write(param_line+'\n')
        fcom.write('\n')
        fcom.write(title_line+'\n')
        fcom.write('\n')
        fcom.write(charge_line+'\n')
        for iatom in range(len(atom)):
            fcom.write(' %s\t%f\t%f\t%f\n'%(atom[iatom],coord[iatom][0],coord[iatom][1],coord[iatom][2]))
        fcom.write('\n\n')

# Read Gaussian output file (with optimization job) for equilibrium geometry
def readOptXYZ(input_file,Natom):
    with open(input_file,'r') as fgauss:
        lines=fgauss.readlines()
        for line in lines:
            if 'Standard orientation:' in line:
                index = lines.index(line)+5
                with open('temp','w') as tempf:
                    for i in range(Natom):
                        tempf.write(lines[index+i])
                coord = np.genfromtxt('temp',usecols=[3,4,5],dtype=float)
                os.system('rm temp')
    return coord

# Read Gaussian output file for energies & forces & charges & dipoles
def readGaussOut(input_file):
    with open(input_file,'r') as fgauss:
        lines = fgauss.readlines()
        for iline in range(len(lines)):
            # Check Error
            if 'Error termination' in lines[iline]:
                stopMLatom('Gaussian calculation failed')
            #if 'Converged?' in lines[iline]:
            #    for i in range(4):
            #        if lines[iline+1+i].strip().split()[-1] == 'NO':
            #            stopMLatom('Gaussian calculation failed to converge')
            # Read Natoms 
            if 'Input orientation:' in lines[iline]:
                n_index = iline+4
                ii = 0 
                while True:
                    raw = lines[n_index+ii+1].strip().split()
                    if raw[0].isdigit():
                        ii += 1
                    else:
                        break
                Natom = ii

            # Read forces (Hartree/Bohr)
            if 'Forces (Hartree' in lines[iline]:
                f_index = iline+3
                with open('temp','w') as temp:
                    for i in range(Natom):
                        temp.write(lines[f_index+i])
                forces = np.genfromtxt('temp',usecols=[2,3,4],dtype=float)
                forces = forces / 0.529177249 # Unit: Hartrees/Bohr
                #print(forces)
            # Read charges
            if 'Mulliken charges:' in lines[iline]:
                c_index = iline+2
                with open('temp','w') as temp:
                    for i in range(Natom):
                        temp.write(lines[c_index+i])
                charges = np.genfromtxt('temp',usecols=[2],dtype=float)
                #print(charges)
            # Read dipole moment
            if 'Dipole moment' in lines[iline]:
                dp_index = iline+1
                raw = lines[dp_index].strip().split() 
                dipole = [eval(raw[1]),eval(raw[3]),eval(raw[5]),eval(raw[7])]
                dipole = np.array(dipole)
            # Read single point energy & dipole moment
            if '1\\1\\' in lines[iline]:
                archive_index = iline
                tmpline = lines[iline]
                archive = ''
                while(tmpline!='\n'):
                    archive += tmpline[1:-1]
                    archive_index += 1
                    tmpline = lines[archive_index]
                archive_split = archive.split('\\')
                for ii in range(len(archive_split)):
                    # Single point energy 
                    #if 'HF=' in archive_split[ii]:
                    #    energy = eval(archive_split[ii][3:])
                    #    #print(energy)
                    if 'RMSD=' in archive_split[ii]:
                        energy = eval(archive_split[ii-1].split('=')[-1])
                    #if 'Dipole=' in each:
                    #    dipole = each[7:].split(',')
                    #    dipole = [eval(dipole[i]) for i in range(3)]
                        #print(dipole)
    return energy, forces, charges, dipole

# Get its mass when given element symbol
def ele2mass(ele):
    ele2mass_dic = {'H':1.007825037,
                  'He':4.00260325,
                  'Li':7.0160045,
                  'Be':9.0121825,
                  'B':11.0093053,
                  'C':12.0,
                  'N':14.003074008,
                  'O':15.99491464,
                  'F':18.99840325,
                  'Ne':19.9924391,
                  'Na':22.9897697,
                  'Mg':23.9850450,
                  'Al':26.9815413,
                  'Si':27.9769284,
                  'P':30.9737634,
                  'S':31.9720718,
                  'Cl':34.968852729,
                  'Ar':39.9623831,
                  'K':38.9637079,
                  'Ca':39.9625907,
                  'Sc':44.9559136,
                  'Ti':47.9479467,
                  'V':50.9439625,
                  'Cr':51.9405097,
                  'Mn':54.9380463,
                  'Fe':55.9349393,
                  'Co':58.9331978,
                  'Ni':57.9353471,
                  'Cu':62.9295992,
                  'Zn':63.9291454,
                  'Ga':68.9255809,
                  'Ge':73.9211788,
                  'As':74.9215955,
                  'Se':79.9165205,
                  'Br':78.9183361,
                  'Kr':83.80,
                  'I':126.9045}
    return ele2mass_dic.get(ele)

def idx2mass(idx):
    ele = idx2element[idx]
    return ele2mass(ele)

idx2element = """ X
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
ele2idx = {}
for i in range(len(idx2element)):
    ele2idx[idx2element[i]] = i 




# Write MLatom-type xyz coordinates (xyz.dat)
def writeXYZ(file_name,atom,coord):
    with open(file_name,'w') as xyzf:
        xyzf.write('%d\n\n'%(len(atom)))
        for iatom in range(len(atom)):
            xyzf.write('%s\t%f\t%f\t%f\n'%(atom[iatom],coord[iatom][0],coord[iatom][1],coord[iatom][2]))


def writeXYZ_3DMD(wdir,atom,coord):
    with open(wdir+'xyz_3DMD.dat','w') as xyzf:
        xyzf.write('%d\n\n'%(len(atom)))
        for iatom in range(len(atom)):
            xyzf.write('%s\t%f\t%f\t%f\n'%(atom[iatom],coord[iatom][0],coord[iatom][1],coord[iatom][2]))

# Generate random velocity when given temperature 
# Unit:
#   Temperature: K (Kelvin)
#   Velocity   : Angstrom / fs
# Total energy = (3 / 2) * kb * T * #atoms 
def randVelocity(temp,atom,coord,noang=False):
    np.random.seed()
    randnum = np.random.randn(len(atom),3)
    
    mass_ = np.array([ele2mass(each) for each in atom])
    mass = np.array(mass_).reshape(len(atom),1)
    
    kb = 1.380649E-23  # Unit: J/K
    hartree = 27.21070 * 1.602176565E-19 # Unit: J/Hartree
    kb = kb / hartree # Unit: Hartree/K
    total_energy = 3.0/2.0 * kb * temp * len(atom)
    
    rand_velocity = randnum / np.sqrt(mass*1822)

    # Eliminate total angular momentum
    if noang:
        # omega = getAngV(coord,rand_velocity,mass_)
        # coord_ = coord - getCoM(coord,mass_)
        # rand_velocity = rand_velocity - np.cross(omega,coord)
        rand_velocity = getridofang(coord,rand_velocity,mass_)

    # Eliminate total translational momentum
    total_mass = sum(mass)[0]
    v_cm = sum(rand_velocity*mass)/total_mass
    rand_velocity = rand_velocity - v_cm
    
    rand_energy = np.sum((rand_velocity**2) * (mass*1822)) / 2.0  
    ratio = rand_energy / total_energy 
    velocity = rand_velocity / np.sqrt(ratio) # Unit: a.u.
    
    bohr = 0.5291772083 # 1 Bohr = 0.5291772083 Angstrom
    time = 2.4189E-2 # Time Unit: 1 a.u. = 2.4189E-17 s = 2.4189E-2 fs
    velocity = velocity * bohr / time # Unit: Angstrom / fs

    return velocity

def getridofang(coord,vel,mass):
    omega = getAngV(coord,vel,mass)
    coord_ = coord - getCoM(coord,mass)
    vel = vel - np.cross(omega,coord)

    return vel 

def useMLmodel(argsMLmodel,prog,notes=''):
    # Remove predictions of last step
    if os.path.exists('enest_3DMD.dat'): os.remove('enest_3DMD.dat')
    if os.path.exists('gradest_3DMD.dat'): os.remove('gradest_3DMD.dat')

    subdatasets=[]
    other_properties = []

    # Run MLatomF
    if prog == 'mlatomf':
        interface_MLatomF.ifMLatomCls.run(argsMLmodel,shutup=True)
    elif prog == 'sgdml':
        interface_sGDML.sGDMLCls.useMLmodel(argsMLmodel,subdatasets)
    elif prog == 'deepmd-kit':
        interface_DeePMDkit.DeePMDCls.useMLmodel(argsMLmodel,subdatasets)
    elif prog == 'gap':
        interface_GAP.GAPCls.useMLmodel(argsMLmodel,subdatasets)
    elif prog == 'torchani':
        interface_TorchANI.ANICls.useMLmodel(argsMLmodel,subdatasets)
    elif prog == 'physnet':
        interface_PhysNet.PhysNetCls.useMLmodel(argsMLmodel,subdatasets)
    
    elif prog == 'aiqm1':
        import AIQM1_cmd
        charges, mults, e_std = AIQM1_cmd.AIQM1Cls(argsMLmodel).forward()
        #print(charges,mults)

    elif prog == 'gaussian':
        import interface_gaussian
        if not os.path.exists(str(notes).zfill(6)):
            os.mkdir(str(notes).zfill(6))
        os.chdir(str(notes).zfill(6))
        os.system('mv ../xyz_3DMD.dat .')
        interface_gaussian.GaussianCls.calculate(argsMLmodel)

    # Read energy and forces 
    if not prog=='gaussian':
        with open('enest_3DMD.dat','r') as enf:
            energy = eval(enf.readline()[:-1])
        gradients = np.genfromtxt('gradest_3DMD.dat',dtype=float,skip_header=2)
        forces = - gradients 

        # Read dipole moments (for AIQM1 only)
        if prog == 'aiqm1' and 'AIQM1' in argsMLmodel:
            try:
                if os.path.exists('mndo0.out'):
                    os.system('mv mndo0.out mndo.out')
                with open('mndo.out','r') as mndof:
                    lines = mndof.readlines() 
                    for iline in range(len(lines)):
                        if 'DIPOLE' in lines[iline]:
                            sum_line = lines[iline+4].rstrip()
                            sum_line = sum_line.split(' ')
                            sum_line = [each for each in sum_line if each != '' ][1:]
                            dipole_moment = [eval(each) for each in sum_line]
                    other_properties.append(dipole_moment)
            except:
                other_properties.append([])
    else:
        energy, forces, charges, dipole = readGaussOut('000000.log')
        os.chdir('../')
        other_properties.append(dipole)

    if prog == 'aiqm1':
        return energy, forces, e_std, other_properties
    else:
        return energy, forces, other_properties

# Calculate instantaneous temperature
def calc_temp(DOF,ekin,dist_unit):
    hartree2kcal = 627.509474
    #kb = 1.3806505E-23 # J K^-1
    R_const = 8.3142 # Gas constant J K^-1 mol^-1
    cal2J = 4.18585182085

    inst_temp = ekin*2 / DOF
    inst_temp = inst_temp  * 1822.888515 * (0.024188432 / dist_unit)**2 * hartree2kcal # kcal mol^-1
    inst_temp = inst_temp * cal2J * 1000 / R_const

    return inst_temp

# Calculate total kinetic energy


# Nose-Hoover chain (NHC) half integration step 
# NVT ensemble
def Nose_Hoover_chain(KE,velocity,dt,xi,vxi,Nc,YSlist,Q,Natoms,T,dist_unit,DOF):
    # https://doi.org/10.1080/00268979600100761
    R_const = 8.3142 # Gas constant J K^-1 mol^-1
    cal2J = 4.18585182085
    hartree2kcal = 627.509474

    # Energy per degree of freedom
    avg_en = T * R_const / cal2J / 1000.0  # Unit: kcal/mol
    # Kinetic energy
    abc = 1822.888515 * (0.024188432 / dist_unit)**2 * hartree2kcal
    KE = KE * 1822.888515 * (0.024188432 / dist_unit)**2 * hartree2kcal # Unit: kcal/mol

    #print(avg_en,KE)
    #print(2*KE-3*Natoms*avg_en)

    M = len(Q)

    scale = 1.0
    for inc in range(Nc):
        for inys in range(len(YSlist)):
            dt_nc = YSlist[inys]*dt / float(inc+1)
            dt_2 = dt_nc / 2.0
            dt_4 = dt_nc / 4.0
            dt_8 = dt_nc / 8.0

            GM = (Q[M-2]*vxi[M-2]*vxi[M-2]*abc-avg_en)/Q[M-1]/abc
            vxi[M-1] = vxi[M-1] + GM*dt_4

            for i in range(M-2):
                ii = M-i-2
                vxi[ii] = vxi[ii] * np.exp(-vxi[ii+1]*dt_8) 
                Gii = (Q[ii-1]*vxi[ii-1]*vxi[ii-1]*abc-avg_en)/Q[ii]/abc
                vxi[ii] = vxi[ii] + Gii * dt_4 
                vxi[ii] = vxi[ii] * np.exp(-vxi[ii+1]*dt_8) 

            vxi[0] = vxi[0] * np.exp(-vxi[1]*dt_8)
            G1 = (2*KE-DOF*avg_en)/Q[0]/abc
            vxi[0] = vxi[0] + G1*dt_4 
            vxi[0] = vxi[0] * np.exp(-vxi[1]*dt_8)


            for i in range(M):
                xi[i] = xi[i] + vxi[i]*dt_2
            
            scale = scale * np.exp(-vxi[0]*dt_2) # Scalar factor
             
            KE = KE*scale**2
                
            vxi[0] = vxi[0] * np.exp(-vxi[1]*dt_8)
            G1 = (2*KE-DOF*avg_en)/Q[0]/abc
            vxi[0] = vxi[0] + G1*dt_4
            vxi[0] = vxi[0] * np.exp(-vxi[1]*dt_8)
            for i in range(M-2):
                ii = i + 1
                vxi[ii] = vxi[ii] * np.exp(-vxi[ii+1]*dt_8) 
                Gii = (Q[ii-1]*vxi[ii-1]*vxi[ii-1]*abc-avg_en)/Q[ii]/abc
                vxi[ii] = vxi[ii] + Gii * dt_4 
                vxi[ii] = vxi[ii] * np.exp(-vxi[ii+1]*dt_8)


            GM = (Q[M-2]*vxi[M-2]*vxi[M-2]*abc-avg_en)/Q[M-1]/abc
            vxi[M-1] = vxi[M-1] + GM*dt_4

    velocity = velocity * scale

    return KE,velocity,xi,vxi

def getCoM(xyz,m=None):
    if m is None:
        m=np.ones(xyz.shape[-2])
    return np.sum(xyz*m[:,np.newaxis],axis=-2)/np.sum(m)

def getAngM(xyz,v,m,center=None):
    if center is None:
        centered=xyz-getCoM(xyz,m)
    else:
        centered=xyz-center
    L=np.sum(m[:,np.newaxis]*np.cross(centered,v),axis=0)
    return L

def getMomentOfInertiaTensor(xyz,m,center=None):
    if center is None:
        center=getCoM(xyz,m)
    centered=xyz-center
    I=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for k in range(len(m)):
                I[i,j]+=m[k]*(np.sum(centered[k]**2)-centered[k,i]*centered[k,j]) if i==j else m[k]*(-centered[k,i]*centered[k,j])
    return I

def getAngV(xyz,v,m,center=None):
    L=getAngM(xyz,v,m,center)
    I=getMomentOfInertiaTensor(xyz,m,center)
    omega=np.linalg.inv(I).dot(L)
    return omega

class ANI_for_MD():
    def __init__(self,argsMLatom):
        import torch
        import torchani
        self.args = interface_TorchANI.Args()
        self.args.parse(argsMLatom)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.element2number = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'S':16, 'CL':17}
        #if self.args.mlmodeltype.lower() == 'ani1x':
        if self.args.ani1x or self.args.ani1xd4:
            self.model = torchani.models.ANI1x(periodic_table_index=True).to(self.device).double()
        #elif self.args.mlmodeltype.lower() == 'ani1ccx':
        elif self.args.ani1ccx:
            self.model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device).double()
        #elif self.args.mlmodeltype.lower() == 'ani2x':
        elif self.args.ani2x or self.args.ani2xd4:
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device).double()

        self.addD4 = False
        if self.args.ani1xd4 or self.args.ani2xd4:
            self.addD4 = True
            from interface_dftd4 import DFTD4
            self.dftd4 = DFTD4()

    def ani_predict(self,coordinate,number,element):
        # Copied from interfaces.TorchANI.TorchANI_predict.ani_predict
        import torch
        from torchani.units import hartree2kcalmol
        species = torch.tensor(number).to(self.device).unsqueeze(0)
        coordinate_ = torch.tensor(coordinate).to(self.device).requires_grad_(True).unsqueeze(0)
        energies = []; grads = []
        fmt = ' %-40s: %15.8f Hartree'
        for mod in self.model:
            energy = mod((species,coordinate_)).energies
            grad = torch.autograd.grad(energy.sum(),coordinate_,create_graph=True,retain_graph=True)[0]
            energies.append(energy)
            grads.append(grad)

        energies = torch.stack([e for e in energies], dim=0)
        e_std = energies.std(dim=0, unbiased=False)
        print(fmt % ('Standard deviation of NN contribution',e_std),end='')
        e_std = hartree2kcalmol(e_std).cpu().detach().numpy()
        print('%15.5f kcal/mol' % e_std)
        energy = energies.mean(0)
        grad = torch.stack([g for g in grads], dim=0).mean(0)
        
        if self.addD4:
            self.dftd4.calculate(element, coordinate, gradientCalc=True, hessianCalc=False)
            d4_results = self.dftd4.get_results()
            d4_energy = torch.tensor(d4_results['energy'])
            print(fmt % ('NN contribution', energy))
            print(fmt % ('D4 contribution', d4_energy))
            energy = energy + d4_energy
            d4_grad = torch.tensor(d4_results['gradient']).reshape(-1, 3)
            grad = grad + d4_grad
        
        print(fmt % ('Total energy', energy))
        energy = energy.cpu().detach().numpy()[0]
        force = -grad.cpu().detach().numpy()[0]
        #print(force)
        #print(type(force))
        #print(energy,force,e_std)
        return energy, force, e_std


