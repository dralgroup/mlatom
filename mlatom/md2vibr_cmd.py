'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! md2vibr_cmd: Module for converting MD to vibrational spectra              ! 
  ! Implementations by: Yi-Fan Hou                                            ! 
  !---------------------------------------------------------------------------! 
'''
from . import data 

import os
import numpy as np
import matplotlib.pyplot as plt 
import numpy.fft as nf
from .args_class import ArgsBase
from pyh5md import File, element
from .stopper import stopMLatom



import statsmodels.tsa.api as smt
import statsmodels.tsa

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        
        self.args2pass = []
        self.add_dict_args({
            'trajH5MDin':'',
            'trajvxyzin':'',
            'trajdpin':'',
            'dt': 0,
            'threshold':0.1,
            'start_time':0,
            'end_time':-1,
            'step':1,
            'miu':0.0,
            'sigma':0,
            'debug':6,
            'nrepeats':1,
            'autocorrelationDepth':1024,
            'zeropadding':1024,
            'title':'',
            'output':'',
            'normalize_intensity': 1,
        })

    def parse(self, argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['', None])
        #self.argProcess()

class md2vibr(object):
    def __init__(self,argsMD2vibr):
        args = Args()
        args.parse(argsMD2vibr)

    @classmethod 
    def simulate(cls,argsMD2vibr):
        args = Args() 
        args.parse(argsMD2vibr)

        def Gaussian_kernel(x,miu,sigma):
            return np.exp(-(x-miu)**2/sigma**2)
        
        def Hann(k,N):
            return 0.5*(1-np.cos(2*np.pi*k/N))

        def rmavg(p):
            avg = np.mean(p)
            return np.array(p)/avg

        def ft_ir(fft_array,dt):
            comp_arr = nf.fft(fft_array)

            freqs = nf.fftfreq(comp_arr.size,dt)
            pows = np.array(abs(comp_arr))

            #max_pow = max(pows)
            #pows = 1.0-(pows / max_pow)

            pows = pows[freqs>0]
            freqs = freqs[freqs>0]*1.0e15/2.9978e10

            freqs_shown = freqs[freqs<4000]
            pows_shown = pows[freqs<4000]

            # Corrected frequencies
            coeff = 2.9978E-5 # Unit: cm/fs
            deltat = dt
            freqs_shown = [np.sqrt(2-2*np.cos(2*np.pi*deltat*each*coeff))/(2*np.pi*deltat*coeff) for each in freqs_shown]

            return freqs_shown,pows_shown

        def ft_ps(fft_array,dt):
            comp_arr = nf.fft(fft_array)

            freqs = nf.fftfreq(comp_arr.size,dt)
            pows = np.array(abs(comp_arr))

            #max_pow = max(pows)
            #pows = 1.0-(pows / max_pow)

            pows = pows[freqs>0]
            freqs = freqs[freqs>0]*1.0e15/2.9978e10

            #pows = pows[freqs>200]
            #freqs = freqs[freqs>200]

            freqs_shown = freqs[freqs<4000]
            pows_shown = pows[freqs<4000]
            return freqs_shown,pows_shown

        
        def normalize(p):
            max_pow = max(p)
            p = (p/max_pow)
            return p

        def readDipole(h5mdin):
            if h5mdin=='':
                stopMLatom('Please provide H5MD file')
            elif not os.path.exists(h5mdin):
                stopMLatom('H5MD file not found: %s does not exist'%(h5mdin))

            # with File(h5mdin,'r') as h5f:
            #     part = h5f.particles_group('all')
            #     h5f.observables = h5f.require_group('observables')
            #     h5dp = element(h5f.observables,'dipole_moment').value[:]
            #     h5time = element(h5f.observables,'dipole_moment').time[:]
            #     dt = h5time[1]-h5time[0]
            #     nsteps = len(h5time)
            #     dipoles = [h5dp[i][0][3] for i in range(nsteps)]
            traj = data.molecular_trajectory() 
            traj.load(filename=h5mdin,format='h5md')
            dipoles = [each.molecule.dipole_moment[-1] for each in traj.steps]
            dt = traj.steps[1].time - traj.steps[0].time
            nsteps = len(traj.steps)

            return dipoles,dt,nsteps

        def readVelocity(h5mdin):
            if h5mdin=='':
                stopMLatom('Please provide H5MD file')
            elif not os.path.exists(h5mdin):
                stopMLatom('H5MD file not found: %s does not exist'%(h5mdin))

            # with File(h5mdin,'r') as h5f:
            #     part = h5f.particles_group('all')
            #     h5v = element(part,'velocities').value[:]
            #     h5time = element(part,'velocities').time[:]
            #     dt = h5time[1]-h5time[0]
            #     nsteps = len(h5time)
            #     velocities = [h5v[i] for i in range(nsteps)]

            traj = data.molecular_trajectory() 
            traj.load(filename=h5mdin,format='h5md')
            velocities = [each.molecule.get_xyz_vectorial_properties('xyz_velocities') for each in traj.steps]
            dt = traj.steps[1].time - traj.steps[0].time
            nsteps = len(traj.steps)
            
            
            return velocities, dt, nsteps

        def readVXYZs(filename):
            with open(filename,'r') as f:
                lines = f.readlines()
                Anames = [] 
                Coords = []
                ii = 0
                while ii < len(lines):
                    Natoms = eval(lines[ii].strip())
                    #a = [] 
                    c = []
                    for iatom in range(Natoms):
                        raw = lines[ii+2+iatom].strip().split()
                        #a.append(raw[0])
                        c.append([eval(each) for each in raw[-3:]])
                    #Anames.append(a)
                    Coords.append(c)
                    ii += Natoms+2
            return Anames, Coords

        def readDPfile(filename):
            raw_data = np.genfromtxt(filename,dtype=float)
            dipoles = [each[-1] for each in raw_data]
            return dipoles

        
        
        def calcVACF(velocities):
            raw_data = [velocities[i] for i in time_ind]
            vacf = []
            Ntotal = len(raw_data)
            vacf = np.zeros(int(args.autocorrelationDepth/dt))

            for iatom in range(len(raw_data[0])):
                for icoord in range(3):
                    data_ = [raw_data[i][iatom][icoord] for i in range(Ntotal)]
                    #tcf = smt.stattools.acf(data_,nlags=len(time_ind)-1,unbiased=True)
                    #tcf = tcf[len(time_ind)-args.autocorrelationDepth:]
                    #tcf = smt.stattools.acf(data_,nlags=int(args.autocorrelationDepth/dt)-1,unbiased=True)
                    #print(data_[:100])
                    if np.mean(np.array(data_)) > 1e-8:

                        tcf = statsmodels.tsa.stattools.acf(data_,nlags=int(args.autocorrelationDepth/dt)-1,fft=True)
                    else:
                        tcf = np.zeros(int(args.autocorrelationDepth/dt))


                    vacf += tcf

            return vacf

        def powerSpectrum(raw_data):
            tcf = raw_data
            for i in range(len(tcf)):
                tcf[i] = tcf[i] * Hann(i,len(tcf))
            tcf = [tcf[i] for i in range(len(tcf))]
            # zeropadding
            tcf += [0]*int(args.zeropadding/dt)
            fft_array = np.array(tcf)
            fft_array = rmavg(fft_array)
            freqs_shown,pows_shown = ft_ps(fft_array,dt*step)
            return freqs_shown, pows_shown

        def ir(dipoles):
            if args.debug==1:
                tcf = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                for i in range(len(tcf)):
                    tcf[i] = tcf[i] * Gaussian_kernel(dt*i,args.miu,args.sigma)
                fft_array = tcf
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
            elif args.debug==0:
                dipole_prime = [(dipoles[i+1]-dipoles[i])/dt for i in time_ind]
                dipole_prime = np.array(dipole_prime)
                dipole_prime = smt.stattools.acf(dipole_prime,nlags=len(time_ind)-1,unbiased=True)
                for i in range(len(dipole_prime)):
                    dipole_prime[i] = dipole_prime[i] * Gaussian_kernel(dt*i,args.miu,args.sigma)
                fft_array = dipole_prime
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
            elif args.debug==2:
                tcf = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                fft_array = tcf
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
            elif args.debug==3:
                tcf = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                fft_array = tcf
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
            elif args.debug==4:
                tcf = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                for i in range(len(tcf)):
                    tcf[i] = tcf[i] * Gaussian_kernel(dt*i,args.miu,args.sigma)
                fft_array = tcf
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
            elif args.debug == 5:
                #autocorrelation depth
                tcf_ = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                tcf = tcf_[len(time_ind)-args.autocorrelationDepth:]
                for i in range(len(tcf)):
                    tcf[i] = tcf[i] * Hann(i,len(tcf))
                print(tcf[-10:])
                tcf = [tcf[i] for i in range(len(tcf))]
                for ii in range(len(tcf)):
                    tcf.append(0)
                fft_array = np.array(tcf)
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
                pows_shown = new_pows
            elif args.debug == 6:
                #autocorrelation depth
                #tcf_ = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=len(time_ind)-1,unbiased=True)
                #tcf = tcf_[len(time_ind)-args.autocorrelationDepth:]
                #tcf = smt.stattools.acf([dipoles[i] for i in time_ind],nlags=int(args.autocorrelationDepth/dt)-1,unbiased=True)
                tcf = statsmodels.tsa.stattools.acf([dipoles[i] for i in time_ind],nlags=int(args.autocorrelationDepth/dt)-1,fft=True)
                #print('new tcf')
                for i in range(len(tcf)):
                    tcf[i] = tcf[i] * Hann(i,len(tcf))
                
                tcf = [tcf[i] for i in range(len(tcf))]
                # zeropadding
                tcf += [0]*int(args.zeropadding/dt)
                
                fft_array = np.array(tcf)
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
                pows_shown = new_pows
            elif args.debug ==7:
                tcf = [dipoles[i] for i in time_ind]
                #print(tcf)
                for i in range(len(tcf)):
                    tcf[i] = tcf[i] * Hann(i,len(tcf))
                #for ii in range(len(tcf)):
                #    tcf.append(0)
                fft_array = np.array(tcf)
                freqs_shown,pows_shown = ft_ir(fft_array,dt*step)
                #pows_shown = pows_shown**2
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
            elif args.debug==99:
                Nensemble = 20
                abc = len(time_ind)//Nensemble
                for i in range(Nensemble):
                    tcf = smt.stattools.acf([dipoles[i] for i in time_ind[i*abc:(i+1)*abc]],nlags=abc-1,unbiased=True)
                    freqs,pows = ft_ir(tcf,dt*step)
                    if i==0:
                        freqs_shown = np.copy(freqs) 
                        pows_shown = np.copy(pows)
                    else:
                        pows_shown = pows_shown + np.array(pows)
                new_pows = []
                c = 2.9978e10 #cm/s
                h = 6.6260693E-34
                kb = 1.3806505E-23
                T = 300
                for i in range(len(freqs_shown)):
                    v = c*freqs_shown[i]
                    new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
            return freqs_shown, pows_shown

        #nrepeats = args.nrepeats
        #if nrepeats == 1:
        #    dipoles,dt,nsteps = readDipole(args.trajH5MDin)
        #elif nrepeats < 1:
        #    stopMLatom('Number of repeats should be larger than 0')
        #else:
        #    dipoles = []
        #    for irepeat in range(nrepeats):
        #        h5path = 'TRAJ'+str(irepeat).zfill(5)+'/traj.h5'
        #        dipoles_,dt_,nsteps_ = readDipole(h5path)
        #        if irepeat == 0:
        #            dt = dt_
        #            nsteps = nsteps_ 
        #        else:
        #            if dt != dt_ or nsteps != nsteps_:
        #                stopMLatom('%s should have the same dt or trun as others'%h5path)
        #        dipoles.append(dipoles_)

        
        

        #if nrepeats==1:
        #    freqs_shown,pows_shown = ir(dipoles)
        #    pows_shown = normalize(pows_shown)
        #else:
        #    for i in range(nrepeats):
        #        freqs,pows = ir(dipoles[i])
        #        if i==0:
        #            freqs_shown = np.copy(freqs)
        #            pows_shown = np.copy(pows)
        #        else:
        #            pows_shown = pows_shown + np.array(pows)
        #    pows_shown = normalize(pows_shown)

        # Check options
        h5md_flag = False 
        dp_flag = False 
        vxyz_flag = False 
        countlist = []
        if args.trajH5MDin != '':
            h5md_flag = True
            countlist.append('trajH5MDin')
        if args.trajvxyzin != '':
            vxyz_flag = True 
            countlist.append('trajVXYZin')
        if args.trajdpin != '':
            dp_flag = True 
            countlist.append('trajdpin')
        if len(countlist) > 1:
            errormsg = ' and '.join(countlist)
            stopMLatom(errormsg+' cannot be used at the same time')
        
        if args.normalize_intensity == 0:
            normalize_intensity = False 
        else:
            normalize_intensity = True
         

        IR_flag = False
        PS_flag = False
        if h5md_flag:
            if (args.output == ''):
                try:
                    raw_data,dt,nsteps = readDipole(args.trajH5MDin)
                    IR_flag = True
                    print('Dipole moments found in H5MD file, generate infrared spectrum')
                except:
                    raw_data,dt,nsteps = readVelocity(args.trajH5MDin)
                    PS_flag = True 
                    print('Dipole moments not found in H5MD file, generate power spectrum instead')
            elif (args.output.lower() == 'ir'):
                # try:
                raw_data,dt,nsteps = readDipole(args.trajH5MDin)
                IR_flag = True
                print('Generate infrared spectrum as defined by the user')
                # except:
                #     stopMLatom('Dipole moments not found in H5MD file')
            elif (args.output.lower() == 'ps'):
                try: 
                    raw_data,dt,nsteps = readVelocity(args.trajH5MDin)
                    PS_flag = True 
                    print('Generate power spectrum as defined by the user')
                except:
                    print('Error in reading H5MD file')
            else:
                stopMLatom('Unknown output option: %s'%(args.output))
        if vxyz_flag: 
            PS_flag = True
            if args.dt <= 0:
                stopMLatom('Please use correct dt')
            else:
                dt = args.dt
            if (args.output == '' or args.output == 'ps'):
                a,raw_data = readVXYZs(args.trajvxyzin)
                nsteps = len(raw_data)
            elif (args.output == 'ir'):
                stopMLatom('IR spectrum needs dipole moments')
            else:
                stopMLatom('Unknown output option: %s'%(args.output))

        if dp_flag:
            IR_flag = True
            if args.dt <= 0:
                stopMLatom('Please use correct dt')
            else:
                dt = args.dt 
            if (args.output == '' or args.output == 'ir'):
                raw_data = readDPfile(args.trajdpin)
                nsteps = len(raw_data)
            elif (args.output == 'ps'):
                stopMLatom('Power spectrum needs velocities')
            else:
                stopMLatom('Unknown output option: %s'%(args.output))

        # debug
        if IR_flag and PS_flag:
            stopMLatom('Contradiction found in the code, please contact the developer')

        step = int(args.step)
        start_time = args.start_time
        if args.end_time == -1:
            end_time = dt*(nsteps-1)
        else:
            end_time = args.end_time 
        if start_time < 0:
            stopMLatom('lower boundary should be larger than 0')
        if start_time > end_time:
            stopMLatom('upper boundary should be larger than lower boundary')
        
        if args.sigma==0:
            args.sigma = end_time-start_time

        print('Using trajectory from %.3f fs to %.3f fs\n'%(start_time,end_time))
        time_ind = range(int(start_time/dt),int(end_time/dt),step)

        if IR_flag:
            freqs_shown,pows_shown = ir(raw_data)
        elif PS_flag:
            raw_data = calcVACF(raw_data)
            freqs_shown,pows_shown = powerSpectrum(raw_data)

        # Normalize intensity
        if normalize_intensity:
            pows_shown = normalize(pows_shown)
        
                





        fig,ax = plt.subplots(1,1)
        ax.plot(freqs_shown,pows_shown)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.title(args.title)
        plt.xlabel('$cm^{-1}$')
        if normalize_intensity:
            plt.ylabel('Normalized intensity')
        else:
            plt.ylabel('Intensity')

        if IR_flag:
            plt.savefig('ir.png',bbox_inches='tight',dpi=300)

            print('IR spectrum saved in ir.png\n')

            np.save('ir.npy',np.array([freqs_shown,pows_shown]))
            print('IR spectrum saved in ir.npy\n')
        elif PS_flag:
            plt.savefig('ps.png',bbox_inches='tight',dpi=300)

            print('Power spectrum saved in ps.png\n')

            np.save('ps.npy',np.array([freqs_shown,pows_shown]))
            print('Power spectrum saved in ps.npy\n')

        pows_shown = normalize(pows_shown)
        peak_int = [] 
        peak_pos = []
        for i in range(1,len(freqs_shown)-1):
            if pows_shown[i] > args.threshold:
                if pows_shown[i-1] < pows_shown[i] and pows_shown[i] > pows_shown[i+1]:
                    peak_int.append(pows_shown[i])
                    peak_pos.append(freqs_shown[i])
        

        peak_inf = zip(peak_pos,peak_int)

        print('Prominent Peaks:\n')
        print('\tPeak\tFrequencies\tAbsorption')
        print('\t    \t (cm^-1)\n')
        
        ipeak=0
        for each in peak_inf:
            ipeak += 1
            print('\t%d\t%.2f\t\t%.2f'%(ipeak,each[0],each[1]))

