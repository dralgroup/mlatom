'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! md2vibr: Module for converting MD to vibrational spectra                  ! 
  ! Implementations by: Yi-Fan Hou                                            ! 
  !---------------------------------------------------------------------------! 
'''
from . import stopper 

import numpy as np 
import numpy.fft as nf 



class vibrational_spectrum():
    def __init__(self,molecular_database,dt):
        self.molecular_database = molecular_database 
        self.dt = dt
        self.nsteps = len(molecular_database.molecules)

    def plot_infrared_spectrum(self,filename,autocorrelation_depth=1024,zero_padding=1024,lb=None,ub=None,normalize=True,title='',return_spectrum=False,format='modulus'):
        import matplotlib.pyplot as plt 

        self.autocorrelation_depth = autocorrelation_depth
        self.zero_padding = zero_padding
        if lb==None:
            lb = 0
        if ub == None:
            ub = (self.nsteps-1)*self.dt
        dipole_moments = [mol.dipole_moment for mol in self.molecular_database.molecules[int(lb/self.dt):int(ub/self.dt)+1]]
        
        if format.casefold() == 'modulus'.casefold():
            data = np.array([each[-1] for each in dipole_moments])
            freqs_shown, pows_shown = self.infrared_spectrum(data)
        elif format.casefold() == 'vector'.casefold():
            dipole_x = np.array([each[0] for each in dipole_moments])
            dipole_y = np.array([each[1] for each in dipole_moments])
            dipole_z = np.array([each[2] for each in dipole_moments])
            freqs_shown, pows_shown_x = self.infrared_spectrum(dipole_x)
            _, pows_shown_y = self.infrared_spectrum(dipole_y)
            _, pows_shown_z = self.infrared_spectrum(dipole_z)
            pows_shown = np.array(pows_shown_x) + np.array(pows_shown_y) + np.array(pows_shown_z)
        else:
            stopper.stopMLatom(f"Unrecognized format: {format}")

        if normalize:
            pows_shown = self.normalize(pows_shown)
        fig,ax = plt.subplots(1,1)
        ax.plot(freqs_shown,pows_shown)
        ax.invert_xaxis() 
        ax.invert_yaxis() 
        plt.title(title)
        plt.xlabel('$Frequency/cm^{-1}$')
        plt.ylabel('Intensity/arbitrary unit')
        plt.savefig(filename,bbox_inches='tight',dpi=300)
        fig.clear()

        if return_spectrum:
            return freqs_shown, pows_shown

    def plot_power_spectrum(self,filename,autocorrelation_depth=1024,zero_padding=1024,lb=None,ub=None,normalize=True,title='',return_spectrum=False):
        import matplotlib.pyplot as plt 

        self.autocorrelation_depth = autocorrelation_depth 
        self.zero_padding = zero_padding
        if lb==None:
            lb = 0
        if ub == None:
            ub = (self.nsteps-1)*self.dt
        velocities = self.molecular_database.get_xyz_vectorial_properties('xyz_velocities')[int(lb/self.dt):int(ub/self.dt)+1]
        data = self.calc_VACF(velocities)
        freqs_shown, pows_shown = self.power_spectrum(data)
        if normalize:
            pows_shown = self.normalize(pows_shown)
        fig,ax = plt.subplots() 
        ax.plot(freqs_shown,pows_shown)
        ax.invert_xaxis() 
        ax.invert_yaxis() 
        plt.title(title)
        plt.xlabel('$Frequency/cm^{-1}$')
        plt.ylabel('Intensity/arbitrary unit')
        plt.savefig(filename,bbox_inches='tight',dpi=300)
        fig.clear()

        if return_spectrum:
            return freqs_shown, pows_shown


    def Hann(self,k,N):
        return 0.5*(1-np.cos(2*np.pi*k/N))
    
    def rmavg(self,p):
        avg = np.mean(p)
        return np.array(p)/avg
    
    def ft_ir(self,fft_array,dt):
        comp_arr = nf.fft(fft_array)

        freqs = nf.fftfreq(comp_arr.size,dt)
        pows = np.array(abs(comp_arr))

        pows = pows[freqs>0]
        freqs = freqs[freqs>0]*1.0e15/2.9978e10

        freqs_shown = freqs[freqs<4000]
        pows_shown = pows[freqs<4000]

        # Corrected frequencies
        coeff = 2.9978E-5 # Unit: cm/fs
        deltat = dt
        freqs_shown = [np.sqrt(2-2*np.cos(2*np.pi*deltat*each*coeff))/(2*np.pi*deltat*coeff) for each in freqs_shown]

        return freqs_shown,pows_shown
    
    def ft_ps(self,fft_array,dt):
        comp_arr = nf.fft(fft_array)

        freqs = nf.fftfreq(comp_arr.size,dt)
        pows = np.array(abs(comp_arr))


        pows = pows[freqs>0]
        freqs = freqs[freqs>0]*1.0e15/2.9978e10

        freqs_shown = freqs[freqs<4000]
        pows_shown = pows[freqs<4000]
        return freqs_shown,pows_shown
    
    def normalize(self,p):
        max_pow = max(p)
        p = (p/max_pow)
        return p

    def calc_VACF(self,velocities):
        import statsmodels.tsa.api as smt
        import statsmodels.tsa 

        data = velocities
        vacf = []
        Ntotal = len(data)
        vacf = np.zeros(int(self.autocorrelation_depth/self.dt))

        for iatom in range(len(data[0])):
            for icoord in range(3):
                data_ = [data[i][iatom][icoord] for i in range(Ntotal)]
                if np.mean(np.array(data_)) > 1e-8:
                    tcf = statsmodels.tsa.stattools.acf(data_,nlags=int(self.autocorrelation_depth/self.dt)-1,fft=True)
                else:
                    tcf = np.zeros(int(self.autocorrelation_depth/self.dt))
                vacf += tcf
        return vacf
    
    def power_spectrum(self,data):
        tcf = data
        for i in range(len(tcf)):
            tcf[i] = tcf[i] * self.Hann(i,len(tcf))
        tcf = [tcf[i] for i in range(len(tcf))]
        # zeropadding
        tcf += [0]*int(self.zero_padding/self.dt)
        fft_array = np.array(tcf)
        fft_array = self.rmavg(fft_array)
        freqs_shown,pows_shown = self.ft_ps(fft_array,self.dt)
        return freqs_shown, pows_shown
    
    def infrared_spectrum(self,dipoles):
        import statsmodels.tsa.api as smt 
        import statsmodels.tsa
        tcf = statsmodels.tsa.stattools.acf(dipoles,nlags=int(self.autocorrelation_depth/self.dt)-1,fft=True)
        for i in range(len(tcf)):
            tcf[i] = tcf[i] * self.Hann(i,len(tcf))
        
        tcf = [tcf[i] for i in range(len(tcf))]
        # zeropadding
        tcf += [0]*int(self.zero_padding/self.dt)
        
        fft_array = np.array(tcf)
        freqs_shown,pows_shown = self.ft_ir(fft_array,self.dt)
        new_pows = []
        c = 2.9978e10 #cm/s
        h = 6.6260693E-34
        kb = 1.3806505E-23
        T = 300
        for i in range(len(freqs_shown)):
            v = c*freqs_shown[i]
            new_pows.append(v*np.tanh(h*v/(2*kb*T))*pows_shown[i])
        pows_shown = new_pows
        return freqs_shown, pows_shown