from glob import glob
from args_class import ArgsBase 
from functions3D import *
from pyh5md import File, element
import time
import re, sys, os
from stopper import stopMLatom
from multiprocessing import Pool 
import traceback
import MDAnalysis as mda


class Args(ArgsBase):
    default_MLprog={
                'kreg': 'mlatomf',
                'id': 'mlatomf',
                'kid': 'mlatomf',
                'ukid': 'mlatomf',
                'akid': 'mlatomf',
                'pkid': 'mlatomf',
                'krr-cm': 'mlatomf',
                'mlatomf':'mlatomf',
                'gap-soap': 'gap',
                'gapsoap':'gap',
                'gap':'gap',
                'sgdml': 'sgdml',
                'gdml':'sgdml',
                'dpmd': 'deepmd-kit',
                'deepmd':'deepmd-kit',
                'deepmd-kit':'deepmd-kit',
                'deeppot-se': 'deepmd-kit',
                'physnet': 'physnet',
                'ani': 'torchani',
                'ani-aev':'torchani',
                'ani1x': 'torchani',
                'ani1ccx': 'torchani',
                'ani2x': 'torchani',
                'ani1xd4':'torchani',
                'ani2xd4':'torchani',
                'ani-tl':'torchani',
                'torchani':'torchani',
                'aiqm1':'aiqm1',
                'aiqm1@dft':'aiqm1dft',
                'aiqm1@dft*':'aiqm1dftstar',
                'gaussian':'gaussian',
                'mixed': 'mixed'
                
            }

    def __init__(self):
        super().__init__()
        self.args2pass = []
        self.add_default_dict_args([
            'ani1x', 'ani2x', 'ani1ccx','ani1xd4','ani2xd4'
            ],
            bool
        )

        self.add_dict_args({
            'MLprog':'mlatomf',
            'nthreads':1,
            'mlmodeltype':''
        })

        self.add_dict_args({
            'nrepeats':1,                  # *Option for development only*
            'dt':0.1,                      # Time step; Unit: fs
            'trun':1000,                   # Length of trajectory; Unit: fs
            'initXYZ':'',                  # File containing initial geometry
            'initVXYZ':'',                 # File containing initial velocity
            'initcond':'',                 # How to generate initial condition
            'nmfile':'',                   # Gaussian ouput file containing normal modes
            'optXYZfile':'',               # File containing XYZ file with optimized geometry
            'trajH5MDout':'traj.h5',       # Output file: H5MD format
            'trajTime':'traj.t',           # Output file containing time
            'trajXYZout':'traj.xyz',       # Output file containing geometries
            'trajVXYZout':'traj.vxyz',     # Output file containing velocities
            'trajEpot':'traj.epot',        # Output file containing potential energies
            'trajEkin':'traj.ekin',        # Output file containing kinetic energies
            'trajEtot':'traj.etot',        # Output file containing total energies 
            'trajEgradXYZ':'traj.grad',    # Output file containing energy gradients
            'trajDipoles':'traj.dp',       # Output file containing dipole moments
            'trajTemperature':'traj.temp', # Output file containing instantaneous temperatures
            'trajLog':'traj.log',          # *Option for development only*
            'MLenergyUnits':'',            # Energy unit in ML model
            'MLdistanceUnits':'',          # Distance unit in ML model
            'thermostat':'',               # Thermostat
            'gamma':0.2,                   # Option for Anderson thermostat
            'initTemp':300,                # Initial temperature 
            'temp':300,                    # Thermostat temperature
            'initVXYZout':'',              # Output file containing initial velocity
            'initXYZout':'',               # Output file containing initial geometry
            'thermostaton':0,              # Turn on the thermostat after ** fs
            'thermostatoff':-1,            # Turn off the thermostat after ** fs
            'NHClength':3,                 # Nose-Hoover chain length
            'Nc':3,                        # Multiple time step
            'Nys':7,                       # Number of Yoshida Suzuki steps used in NHC (1,3,5,7)
            'NHCfreq':16,
            'noang':0,
            'DOF':-6,
            'linear':0,


        })

        self.add_dict_args({
            'cheapmodel':'',                 
            'tbemodel':'', 
            'sd_thre':'', 
            'bond_len_lowthre':'',
            'bond_len_highthre':'',
            'bond':'',

        })

        #self.set_keyword_alias('AIQM1 MLprog=aiqm1',['AIQM1'])
        #self.set_keyword_alias('AIQM1DFTstar MLprog=aiqm1', ['AIQM1@DFT*'])
        #self.set_keyword_alias('AIQM1DFT MLprog=aiqm1', ['AIQM1@DFT'])
        #self.set_keyword_alias('mlmodeltype=ani1ccx', ['ANI-1ccx'])
        #self.set_keyword_alias('mlmodeltype=ani1x', ['ANI-1x'])
        #self.set_keyword_alias('mlmodeltype=ani2x', ['ANI-2x'])


    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['', None])
        #self.argProcess()

class MultiThreeDcls(object):
    def __init__(self,args3D):
        args = Args()
        args.parse(args3D)

    @classmethod 
    def dynamics(cls,args3D):
        args = Args()
        args.parse(args3D)

        if (args.nrepeats == 1):
            ThreeDcls.dynamics(args.args2pass)
        elif (args.nrepeats < 1):
            stopMLatom('Number of repeats should be larger than 0')
        else:
            global thread_task
            def thread_task(num,args2pass):
                try:
                    print('Job id:%d, Traj:%d <start>'%(os.getpid(),num))
                    if not os.path.exists('TRAJ'+str(num).zfill(5)):
                        os.mkdir('TRAJ'+str(num).zfill(5))
                    os.chdir('TRAJ'+str(num).zfill(5))
                    ThreeDcls.dynamics(args2pass)
                    os.chdir('../')
                    print('Job id:%d, Traj:%d <end>'%(os.getpid(),num))
                except Exception as e:
                    print('Exception:'+str(e))
                    traceback.print_exc()
                    raise Exception(str(e))

            nthreads = args.nthreads
            MDargs = args.args2pass 
            deadlist = []
            for arg in MDargs:
                flagmatch1 = re.search('(^nthreads=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                flagmatch2 = re.search('(^initxyz=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch1:
                    deadlist.append(arg)
                if flagmatch2:
                    deadlist.append(arg)
                    tempxyz = arg
            for i in deadlist: MDargs.remove(i)
            MDargs.append('nthreads=1')
            tempxyz = tempxyz.split('=')
            MDargs.append(tempxyz[0]+'=../'+tempxyz[1])

            '''
            # Multiprocessing (low efficiency)
            thread_pool = Pool(nthreads)
            for itraj in range(args.nrepeats):
                print(itraj)
                thread_pool.apply_async(thread_task,args=(itraj,MDargs,))
            thread_pool.close()
            thread_pool.join()
            '''
            print('Run %d MD trajectories'%(args.nrepeats))
            for itraj in range(args.nrepeats):
                thread_task(itraj,MDargs)

    
class ThreeDcls(object):
    def __init__(self,args3D):
        args = Args()
        args.parse(args3D)

    @classmethod
    def dynamics(cls,args3D):
        args = Args()
        args.parse(args3D)
        
        try:
            args.MLprog = args.default_MLprog[args.MLprog.lower()]
        except: 
            stopMLatom('Unknown MLprog: %s'%(args.MLprog))
        
        if args.MLprog.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
            args.MLenergyUnits = 'Hartree'
            args.MLdistanceUnits = 'Angstrom'
            deadlist = []
            argsMLmodel = args.args2pass
            for arg in argsMLmodel:
                flagmatch = re.search('(^nthreads=)|(^mlmodeltype=)|(^MLprog)|(^dt=)|(^trun=)|(^initXYZ=)|(^initVXYZ=)|(^trajH5MDout=)|(^trajTime=)|(^trajXYZout=)|(^trajVXYZout=)|(^trajEpot=)|(^trajEkin=)|(^trajEtot)|(^trajEgradXYZ=)|(^trajDipoles=)|(^MLenergyUnits=)|(^MLdistanceUnits=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch:
                    deadlist.append(arg)
            for i in deadlist: argsMLmodel.remove(i)
            argsMLmodel.append('XYZfile=xyz_3DMD.dat')
            argsMLmodel.append('YestFile=enest_3DMD.dat')
            argsMLmodel.append('YgradXYZestFile=gradest_3DMD.dat')
            if args.MLprog.lower == 'aiqm1dft':
                argsMLmodel.append('AIQM1DFT')
            elif args.MLprog.lower == 'aiqm1dftstar':
                argsMLmodel.append('AIQM1DFTstar')
            else: 
                argsMLmodel.append('AIQM1')
            args.MLprog = 'aiqm1'

        #elif args.MLmodelType and args.MLmodelType in ['ani1ccx','ani1x','ani2x']:
        elif args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
            args.MLprog = 'torchani'
            args.MLenergyUnits = 'Hartree'
            args.MLdistanceUnits = 'Angstrom'
            #argsMLmodel = ['useMLmodel','XYZfile=%s'%(args.initXYZ),'YestFile=enest_3DMD.dat','YgradXYZestFile=gradest_3DMD.dat']
            if args.initXYZ != '':
                tempfile = args.initXYZ 
            elif args.optXYZfile != '':
                tempfile = args.optXYZfile
            argsMLmodel = ['XYZfile=%s'%(tempfile),'YestFile=enest_3DMD.dat','YgradXYZestFile=gradest_3DMD.dat']
            #argsMLmodel.append('MLmodelType=%s'%(args.MLmodelType))
            argsMLmodel.append('nthreads=%s'%(args.nthreads))
            if args.ani1ccx: argsMLmodel.append('ani1ccx')
            if args.ani1x: argsMLmodel.append('ani1x')
            if args.ani2x: argsMLmodel.append('ani2x')
            if args.ani1xd4: argsMLmodel.append('ani1xd4')
            if args.ani2xd4: argsMLmodel.append('ani2xd4')
            ANImodel_for_MD = ANI_for_MD(argsMLmodel)

        elif args.MLprog.lower() == 'mixed':
            args.MLenergyUnits = 'Hartree'
            args.MLdistanceUnits = 'Angstrom'
            tbe_flag=False
            if args.cheapmodel.lower() == 'ani1ccx':
                args.MLprog_main = 'torchani'
                #argsMLmodel = ['useMLmodel','XYZfile=%s'%(args.initXYZ),'YestFile=enest_3DMD.dat','YgradXYZestFile=gradest_3DMD.dat']
                if args.initXYZ != '':
                    tempfile = args.initXYZ 
                elif args.optXYZfile != '':
                    tempfile = args.optXYZfile
                argsMLmodel_main = ['XYZfile=%s'%(tempfile),'YestFile=enest_3DMD.dat','YgradXYZestFile=gradest_3DMD.dat']
                #argsMLmodel.append('MLmodelType=%s'%(args.MLmodelType))
                argsMLmodel_main.append('nthreads=%s'%(args.nthreads))
                argsMLmodel_main.append('ani1ccx')
                ANImodel_for_MD = ANI_for_MD(argsMLmodel_main)
            elif args.cheapmodel.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
                if args.cheapmodel.lower() == 'aiqm1dft':
                    args.MLprog_main = 'aiqm1dft'
                elif args.cheapmodel.lower() == 'aiqm1dftstar':
                    args.MLprog_main = 'aiqm1dftstar'
                else: 
                    args.MLprog_main = 'aiqm1'
                # deadlist = []
                # argsMLmodel_main = args.args2pass
                # for arg in argsMLmodel_main:
                #     flagmatch = re.search('(^nthreads=)|(^mlmodeltype=)|(^MLprog)|(^dt=)|(^trun=)|(^initXYZ=)|(^initVXYZ=)|(^trajH5MDout=)|(^trajTime=)|(^trajXYZout=)|(^trajVXYZout=)|(^trajEpot=)|(^trajEkin=)|(^trajEtot)|(^trajEgradXYZ=)|(^trajDipoles=)|(^MLenergyUnits=)|(^MLdistanceUnits=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                #     if flagmatch:
                #         deadlist.append(arg)
                # for i in deadlist: argsMLmodel_main.remove(i)
                argsMLmodel_main = []
                for arg in args.args2pass:
                    flagmatch = re.search('(^qmprog=)|(^nthreads=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                    if flagmatch:
                        argsMLmodel_main.append(arg)
                argsMLmodel_main.append('XYZfile=xyz_3DMD.dat')
                argsMLmodel_main.append('YestFile=enest_3DMD.dat')
                argsMLmodel_main.append('YgradXYZestFile=gradest_3DMD.dat')
                if args.MLprog_main.lower() == 'aiqm1dft':
                    argsMLmodel_main.append('aiqm1dft')
                elif args.MLprog_main.lower() == 'aiqm1dftstar':
                    argsMLmodel_main.append('aiqm1dftstar')
                else: 
                    argsMLmodel_main.append('aiqm1')
            if args.tbemodel.lower() == 'gaussian':
                args.MLprog_alter = 'gaussian'
                argsMLmodel_alter = []
                for arg in args.args2pass:
                    flagmatch = re.search('(^gaukw=)|(^spin=)|(^charge=)|(^nthreads=)', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                    if flagmatch:
                        argsMLmodel_alter.append(arg)
                argsMLmodel_alter.append('XYZfile=xyz_3DMD.dat')

        elif args.MLprog == 'gaussian':
            args.MLenergyUnits = 'Hartree'
            args.MLdistanceUnits = 'Angstrom'
            argsMLmodel = args.args2pass
            argsMLmodel.append('XYZfile=xyz_3DMD.dat')
        else:
            argsMLmodel = ['useMLmodel','XYZfile=xyz_3DMD.dat','YestFile=enest_3DMD.dat','YgradXYZestFile=gradest_3DMD.dat']
            argsMLmodel.append('MLmodelIn='+args.MLmodelIn)

        if args.MLenergyUnits == '' or args.MLenergyUnits.lower() == 'kcal/mol':
            MLenergyUnits = 627.509474
        elif args.MLenergyUnits.lower() == 'hartree':
            MLenergyUnits = 1.0
        else:
            stopMLatom('Unknown MLenergyUnits: %s'%(args.MLenergyUnits))

        if args.MLdistanceUnits == '' or args.MLdistanceUnits.lower() == 'angstrom':
            MLdistanceUnits = 0.529177249 
        elif args.MLdistanceUnits.lower() == 'bohr':
            MLdistanceUnits = 1.0
        else: 
            stopMLatom('Unknown MLdistanceUnits: %s'%(args.MLenergyUnits))

        # Deal with initial conditions
        if args.initcond == '' or args.initcond == 'user-defined':
            # Check file 
            if args.initXYZ == '':
                stopMLatom('Please provide initial XYZ file')
            if args.initVXYZ == '':
                stopMLatom('Please provide initial VXYZ file')
            if not os.path.exists(args.initXYZ):
                stopMLatom('User-defined initial XYZ file %s does not exist'%(args.initXYZ))
            if not os.path.exists(args.initVXYZ):
                stopMLatom('User-defined initial VXYZ file %s does not exist'%(args.initVXYZ))
            # Get initial geometry
            Natoms, atom, init_xyz = readXYZ(args.initXYZ)

            # Get initial velocity (generate random velocity if not provided)
            _Natoms,_atom, init_vxyz = readXYZ(args.initVXYZ)
        elif args.initcond == 'random':
            # Check file
            if args.initXYZ != '':
                print(' Use user-defined initial XYZ file: %s'%(args.initXYZ))
            else:
                stopMLatom('MLatom can now only generate random velocities, please provide initial XYZ file')
            if not os.path.exists(args.initXYZ):
                stopMLatom('User-defined initial XYZ file %s does not exist'%(args.initXYZ))
            if args.initVXYZ != '':
                stopMLatom('Initial VXYZ file is provided, please use "Initcond=user-defined" option instead')
            # Get initial geometry
            Natoms, atom, init_xyz = readXYZ(args.initXYZ)
            # Get initial velocity (generate random velocity)
            if args.noang == 0:
                init_vxyz = randVelocity(args.initTemp,atom,init_xyz)
            else:
                init_vxyz = randVelocity(args.initTemp,atom,init_xyz,noang=True)
            _atom = np.copy(atom)
            if args.initVXYZout != '':
                writeXYZ(args.initVXYZout,atom,init_vxyz)
        else:
            stopMLatom('Unknown initial codition option: %s'%(args.initcond))

        # Degrees of freedom
        if args.linear!=0 and args.linear!=1:
            stopMLatom('The value of option "Linear" can only be 0 or 1')
        if args.initcond == '' or args.initcond == 'user-defined':
            DOF = max(1,3*Natoms + args.DOF)
        elif args.initcond == 'random':
            DOF = 3*Natoms-3        # Linear momentum will not be included
            if args.noang != 0:
                DOF = max(1,DOF-3+args.linear)  # Not include angular momentum  

        print(' Degrees of freedom: %d'%(DOF))



        # Args of thermostat
        Nose_Hoover = False

        if args.thermostat.lower() == 'nose-hoover':
            print("Length of Nose-Hoover chain: %d"%args.NHClength)
            avg_en = args.temp * 8.3142 / 4.18585182085 / 1000.0 / 627.509474  # Unit: Hartree
            nose_freq = 1.0 / (args.NHCfreq) # Unit: fs^-1
            Nose_Hoover = True
            NHC_xi = [0.0]*args.NHClength   # Unit: Angstrom
            NHC_vxi = [0.0]*args.NHClength  # Unit: fs^-1

            # Choices of Q (effective mass)
            # Q1 = NkT/w^2; Qi = kT/w^2
            # w is the frequency at which the particle thermostats fluctuate
            # https://doi.org/10.1063/1.463940
            # https://doi.org/10.1080/00268979600100761
            Qi = avg_en/1822.888515*0.529177249**2*(100.0/2.4188432)**2 # Unit: a.m.u. * A^2/(fs^2)
            Qi = Qi / nose_freq**2 # Unit: a.m.u. * A^2
            NHC_Q = [DOF*Qi]+[Qi]*(args.NHClength-1)
            #NHC_Q = [0.1,0.1]
            print("Nose mass: ")
            for i in range(len(NHC_Q)):
                print("    Q%d: %f"%(i+1,NHC_Q[i]))

            if args.Nys == 1:
                YSlist = [1.0]
            elif args.Nys == 3:
                YSlist = [0.828981543588751, -0.657963087177502, 0.828981543588751]
                #a = 1.0/(2-2**(1.0/3))
                #YSlist = [a,1-2*a,a]
            elif args.Nys == 5:
                YSlist = [0.2967324292201065, 0.2967324292201065, -0.186929716880426, 0.2967324292201065,
                    0.2967324292201065]
            elif args.Nys == 7:
                YSlist = [0.784513610477560, 0.235573213359357, -1.17767998417887, 1.31518632068391,
                    -1.17767998417887, 0.235573213359357, 0.784513610477560]
            else:
                stopMLatom('The number of Yoshida-Suzuki steps has to be 1,3,5 or 7')
            

        if args.thermostatoff == -1:
            args.thermostatoff = args.trun

        trun = args.trun
        dt = args.dt

        nsteps = int(trun / dt)

              

        # Get atom indices
        atomIdx = [ele2idx[i] for i in atom]
        #for i in range(Natoms):
        #    if len(_atom[i]) ==1: _atom[i]+ ' '

        # MD simulation starts from here
        coord = init_xyz        # Unit: MLdistanceUnits
        velocity = init_vxyz    # Unit: MLdistanceUnits/fs

        atom_mass = [ele2mass(each) for each in atom]
        atom_mass = np.array(atom_mass).reshape(len(atom),1)
        coord = np.array(coord)
        velocity = np.array(velocity)


        with File(args.trajH5MDout,'w',creator='MLatom') as h5f, open(args.trajTime,'w') as timef, open(args.trajXYZout,'w') as xyzf, open(args.trajVXYZout,'w') as vf, open(args.trajEpot,'w') as epotf, open(args.trajEkin,'w') as ekinf, open(args.trajEtot,'w') as etotf, open(args.trajEgradXYZ,'w') as gradf, open(args.trajDipoles,'w') as dpf, open(args.trajTemperature,'w') as tempf, open(args.trajLog,'w') as logf:
            part = h5f.particles_group('all')
            part.create_box(dimension=3,boundary=['none','none','none'])
            h5xyz = element(part,'position',store='time',shape=(Natoms,3),dtype=np.float64, time=True)
            h5v = element(part,'velocities',store='time',shape=(Natoms,3),dtype=np.float64,time=True)
            h5a = element(part,'accelations',store='time',shape=(Natoms,3),dtype=np.float64,time=True)
            h5grad = element(part,'gradients',store='time',shape=(Natoms,3),dtype=np.float64,time=True)
            element(part,'species',data=np.array(atomIdx),store='fixed')
            #element(part,'names',data=np.array(_atom),store='fixed')
            element(part,'mass',data=np.array(atom_mass).reshape(Natoms),store='fixed')
            h5f.observables = h5f.require_group('observables')
            h5ek = element(h5f.observables,'kinetic_energy',shape=(),dtype=np.float64,store='time',time=True)
            h5ep = element(h5f.observables,'potential_energy',shape=(),dtype=np.float64,store='time',time=True)
            h5et = element(h5f.observables,'total_energy',shape=(),dtype=np.float64,store='time',time=True)
            h5dp = element(h5f.observables,'dipole_moment',shape=(1,4),dtype=np.float64,store='time',time=True)

            for istep in range(0,nsteps+1):
                starttime=time.time()
                if istep==0:
                    #print('xi: ',NHC_xi,'\nvxi: ',NHC_vxi)
                    #if args.MLmodelType and args.MLmodelType in ['ani1ccx','ani1x','ani2x']:
                    if args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                        energy,forces,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)

                    elif args.MLprog.lower() == 'mixed':
                        # obtain bond length
                        if ((args.bond_len_lowthre != '') or (args.bond_len_highthre != '')) and (args.bond != ''):
                            atom1_id, atom2_id = args.bond.split('-')
                            os.system('cp %s %s.xyz' % (args.initXYZ, args.initXYZ))
                            geom_traj = mda.Universe('%s.xyz' % (args.initXYZ))
                            geom_atomgroup = geom_traj.atoms[int(atom1_id)-1] + geom_traj.atoms[int(atom2_id)-1]
                            geom_bond_len = round(geom_atomgroup.bond.value(),4)
                            print(" Bond Length between %s is %.4f" %(args.bond, geom_bond_len))
                        # change tbe_flag; for t=0 there is no SD value for last step
                        if ('geom_bond_len' in vars().keys()) and (args.bond_len_lowthre != ''):
                            if (geom_bond_len < args.bond_len_lowthre):
                                tbe_flag = True
                        if ('geom_bond_len' in vars().keys()) and (args.bond_len_highthre != ''):
                            if (geom_bond_len > args.bond_len_highthre):
                                tbe_flag = True
                        # if (geom_bond_len < args.bond_len_lowthre) or (geom_bond_len > args.bond_len_highthre):
                        #     tbe_flag = True
                            print(" Judgement Finished")
                        if tbe_flag:
                            print(" Start Using TBE Model")
                            # run tbe e+f
                            if args.tbemodel.lower() == 'gaussian':
                                writeXYZ_3DMD('',atom,coord)
                                energy, forces, other_properties = useMLmodel(argsMLmodel_alter,args.MLprog_alter,notes=istep) 
                                mixed_whichone = 'gaussian'
                            # calculate SD
                            if args.cheapmodel.lower() == 'ani1ccx':
                                writeXYZ_3DMD('',atom,coord)
                                energy_ani,forces_ani,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)
                            elif args.cheapmodel.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
                                writeXYZ_3DMD('',atom,coord)
                                energy_aiqm1, forces_aiqm1, e_std, other_properties_aiqm1 = useMLmodel(argsMLmodel_main,args.MLprog_main,notes='')
                            tbe_flag = False
                        elif not tbe_flag:
                            print(" Start Using Cheap Model")
                            # e+f and SD
                            if args.cheapmodel.lower() == 'ani1ccx':
                                writeXYZ_3DMD('',atom,coord)
                                energy,forces,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)
                                mixed_whichone = 'ani1ccx'
                            elif args.cheapmodel.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
                                writeXYZ_3DMD('',atom,coord)
                                energy, forces, e_std, other_properties = useMLmodel(argsMLmodel_main,args.MLprog_main,notes='')
                                mixed_whichone = args.cheapmodel.lower()
                        print(" TBE_flag is", tbe_flag)

                    elif args.MLprog=='gaussian':
                        writeXYZ_3DMD('',atom,coord)
                        energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes=istep)
                    else:
                        writeXYZ_3DMD('',atom,coord)
                        energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes='')
                    acceleration = forces / atom_mass / 1822.888515 * (MLdistanceUnits**2) * (100.0/2.4188432)**2 / MLenergyUnits
                    kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0
                else:
                    # Velocity Verlet algorithm (VVA)
                    if Nose_Hoover:
                        kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0 # Raw kinetic energy
                        #KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain2(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,NHC_Q,Natoms,args.temp,MLdistanceUnits)
                        KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,args.Nc,YSlist,NHC_Q,Natoms,args.temp,MLdistanceUnits,DOF)

                        #print('xi: ',NHC_xi,'\nvxi: ',NHC_vxi)

                        coord = coord + velocity*dt + acceleration*dt**2*0.5
                        #if args.MLmodelType and args.MLmodelType in ['ani1ccx','ani1x','ani2x']:
                        if args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                            energy,forces,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)
                        elif args.MLprog=='gaussian':
                            writeXYZ_3DMD('',atom,coord)
                            energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes=istep)
                        else:
                            writeXYZ_3DMD('',atom,coord)
                            energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes='')
                        velocity = velocity + acceleration * dt * 0.5
                        acceleration = forces / atom_mass / 1822.888515 * (MLdistanceUnits**2) * (100.0/2.4188432)**2 / MLenergyUnits
                        
                        velocity = velocity + acceleration*dt*0.5

                        # Raw kinetic energy (Unit: relative_mass MLdistanceUnits^2 / fs^2)
                        kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0
                        #KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain2(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,NHC_Q,Natoms,args.temp,MLdistanceUnits)
                        KE,velocity,NHC_xi,NHC_vxi = Nose_Hoover_chain(kin_en_raw,velocity,dt,NHC_xi,NHC_vxi,args.Nc,YSlist,NHC_Q,Natoms,args.temp,MLdistanceUnits,DOF)
                        #print('xi: ',NHC_xi,'\nvxi: ',NHC_vxi)

                    else:
                        coord = coord + velocity*dt + 0.5*acceleration*dt**2 
                        velocity = velocity + 0.5*acceleration*dt 
                        #if args.MLmodelType and args.MLmodelType in ['ani1ccx','ani1x','ani2x']:
                        if args.ani1ccx or args.ani1x or args.ani2x or args.ani1xd4 or args.ani2xd4:
                            energy,forces,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)

                        elif args.MLprog.lower() == 'mixed':
                            writeXYZ_3DMD('',atom,coord)
                            print(" Standard Deviation of the Last Step is %.5f" % e_std)
                            # obtain bond length
                            if ((args.bond_len_lowthre != '') or (args.bond_len_highthre != '')) and (args.bond != ''):
                                atom1_id, atom2_id = args.bond.split('-')
                                os.system('cp xyz_3DMD.dat xyz_3DMD.xyz')
                                geom_traj = mda.Universe('xyz_3DMD.xyz')
                                geom_atomgroup = geom_traj.atoms[int(atom1_id)-1] + geom_traj.atoms[int(atom2_id)-1]
                                geom_bond_len = round(geom_atomgroup.bond.value(),4)
                                print(" Bond Length between %s is %.4f" %(args.bond, geom_bond_len))
                            # change tbe_flag
                            if ('geom_bond_len' in vars().keys()) and (args.bond_len_lowthre != ''):
                                if (geom_bond_len < args.bond_len_lowthre):
                                    tbe_flag = True
                            if ('geom_bond_len' in vars().keys()) and (args.bond_len_highthre != ''):
                                if (geom_bond_len > args.bond_len_highthre):
                                    tbe_flag = True
                            if ('e_std' in vars().keys()) and (args.sd_thre != ''):
                                if (e_std > args.sd_thre):
                                    tbe_flag = True
                            # if (geom_bond_len < args.bond_len_lowthre) or (geom_bond_len > args.bond_len_highthre) or (e_std > args.sd_thre):
                            #     tbe_flag = True
                                print(" Judgement Finished")
                            if tbe_flag:
                                print(" Start Using TBE Model")
                                # run tbe e+f
                                if args.tbemodel.lower() == 'gaussian':
                                    writeXYZ_3DMD('',atom,coord)
                                    energy, forces, other_properties = useMLmodel(argsMLmodel_alter,args.MLprog_alter,notes=istep) 
                                    mixed_whichone = 'gaussian'
                                # calculate SD
                                if args.cheapmodel.lower() == 'ani1ccx':
                                    writeXYZ_3DMD('',atom,coord)
                                    energy_ani,forces_ani,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)
                                elif args.cheapmodel.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
                                    writeXYZ_3DMD('',atom,coord)
                                    energy_aiqm1, forces_aiqm1, e_std, other_properties_aiqm1 = useMLmodel(argsMLmodel_main,args.MLprog_main,notes='')
                                tbe_flag = False
                            elif not tbe_flag:
                                print(" Start Using Cheap Model")
                                # e+f and SD
                                if args.cheapmodel.lower() == 'ani1ccx':
                                    writeXYZ_3DMD('',atom,coord)
                                    energy,forces,e_std = ANImodel_for_MD.ani_predict(coord,atomIdx,atom)
                                    mixed_whichone = 'ani1ccx'
                                elif args.cheapmodel.lower() in ['aiqm1','aiqm1dft','aiqm1dftstar']:
                                    writeXYZ_3DMD('',atom,coord)
                                    energy, forces, e_std, other_properties = useMLmodel(argsMLmodel_main,args.MLprog_main,notes='')
                                    mixed_whichone = args.cheapmodel.lower()
                            print(" TBE_flag is", tbe_flag)
                            

                        elif args.MLprog=='gaussian':
                            writeXYZ_3DMD('',atom,coord)
                            energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes=istep)
                        else:
                            writeXYZ_3DMD('',atom,coord)
                            energy, forces, other_properties = useMLmodel(argsMLmodel,args.MLprog,notes='')
                        acceleration = forces / atom_mass / 1822.888515 * (MLdistanceUnits**2) * (100.0/2.4188432)**2 / MLenergyUnits
                        
                        velocity = velocity + 0.5*acceleration*dt


                # Raw kinetic energy (Unit: relative_mass MLdistanceUnits^2 / fs^2)
                kin_en_raw = np.sum(velocity**2 * atom_mass) / 2.0
                # Instantaneous temperature (Unit: K)
                inst_temp = calc_temp(DOF,kin_en_raw,MLdistanceUnits)
                # Kinetic energy (Unit: MLenergyUnits)
                kin_en = kin_en_raw * 1822.888515 * (0.024188432 / MLdistanceUnits)**2 * MLenergyUnits

                # Save this step
                # Time
                tt = istep*dt
                timef.write('%f\n'%(tt))
                timef.flush()
                # XYZ
                xyzf.write('%d\n\n'%(Natoms))
                for iatom in range(Natoms):
                    xyzf.write('%s\t%.7f\t%.7f\t%.7f\n'%(atom[iatom],coord[iatom][0],coord[iatom][1],coord[iatom][2]))
                xyzf.flush()
                # VXYZ
                vf.write('%d\n\n'%(Natoms))
                for iatom in range(Natoms):
                    vf.write('\t%.7f\t%.7f\t%.7f\n'%(velocity[iatom][0],velocity[iatom][1],velocity[iatom][2]))
                vf.flush()
                # Epot 
                epotf.write('%f\n'%(energy))
                epotf.flush()
                # Ekin 
                ekinf.write('%f\n'%(kin_en))
                ekinf.flush()
                # Etot
                etotf.write('%f\n'%(energy+kin_en))
                etotf.flush()
                # EgradXYZ
                gradf.write('%d\n\n'%(Natoms))
                for iatom in range(Natoms):
                    gradf.write('\t%.7f\t%.7f\t%.7f\n'%(-forces[iatom][0],-forces[iatom][1],-forces[iatom][2]))
                gradf.flush()
                # Dipoles 
                if args.MLprog.lower() == 'aiqm1' or args.MLprog == 'gaussian':
                    if other_properties[0] != []:
                        dpf.write('%.7f\t%.7f\t%.7f\t%.7f\n'%(other_properties[0][0],other_properties[0][1],other_properties[0][2],other_properties[0][3]))
                        dpf.flush()
                # Temperature
                tempf.write('%f\n'%(inst_temp))
                tempf.flush()
                # H5MD format
                h5xyz.append(coord,istep,tt)
                h5v.append(velocity,istep,tt)
                h5a.append(acceleration,istep,tt)
                h5grad.append(-forces,istep,tt) 
                h5ek.append(kin_en,istep,tt) 
                h5ep.append(energy,istep,tt) 
                h5et.append(energy+kin_en,istep,tt)
                if args.MLprog.lower() == 'aiqm1':
                    if other_properties[0] != []:
                        h5dp.append(other_properties[0],istep,tt)

                
                sys.stdout.flush()

                endtime = time.time() - starttime
                print('Step %d finished in %.2f s (%.2f min, %.2f hours)\n'%(istep,endtime,endtime/60.0,endtime/3600.0))
                print('================================================================')
