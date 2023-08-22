#!/usr/bin/env python3

import numpy as np 
try:
    from . import constants, stopper
except:
    import constants, stopper

class Thermostat():
    def __init__(self):
        pass

    def save_thermostat(self):
        pass


class Nose_Hoover_thermostat(Thermostat):
    def __init__(self,**kwargs):
        # 'NHClength':3,                 # Nose-Hoover chain length
        # 'Nc':3,                        # Multiple time step
        # 'Nys':7,                       # Number of Yoshida Suzuki steps used in NHC (1,3,5,7)
        # 'NHCfreq':16,
        if 'nose_hoover_chain_length' in kwargs:
            self.NHC_length = kwargs['nose_hoover_chain_length']
        else:
            self.NHC_length = 3
        if 'multiple_time_steps' in kwargs:
            self.Nc = kwargs['multiple_time_step']
        else:
            self.Nc = 3 
        if 'number_of_yoshida_suzuki_steps' in kwargs:
            self.Nys  = kwargs['number_of_yoshida_suzuki_steps']
        else:
            self.Nys = 7 
        if 'nose_hoover_chain_frequency' in kwargs:
            self.NHC_frequency = kwargs['nose_hoover_chain_frequency']
        else:
            self.NHC_frequency = 16.0
        if 'temperature' in kwargs:
            self.temperature = kwargs['temperature']
        else:
            self.temperature = 300.0
        if 'molecule' in kwargs:
            self.molecule = kwargs['molecule']
            self.Natoms = len(self.molecule.atoms)
        if 'degrees_of_freedom' in kwargs:
            self.degrees_of_freedom = kwargs['degrees_of_freedom']
            if self.degrees_of_freedom <= 0:
                self.degrees_of_freedom = 3*self.Natoms + self.degrees_of_freedom
        else:
            self.degrees_of_freedom = max(1,3*self.Natoms-6)
        if 'NHC_xi' in kwargs:
            self.NHC_xi = kwargs['NHC_xi']
            if len(self.NHC_xi) != self.NHC_length:
                stopper.stopMLatom('size of NHC_xi is different from NHC_length')
        else:
            self.NHC_xi = [0.0]*self.NHC_length   # Unit: Angstrom
        if 'NHC_vxi' in kwargs:
            self.NHC_vxi = kwargs['NHC_vxi']
            if len(self.NHC_vxi) != self.NHC_length:
                stopper.stopMLatom('size of NHC_vxi is different from NHC_length')
        else:
            self.NHC_vxi = [0.0]*self.NHC_length  # Unit: fs^-1

        avg_en = self.temperature * constants.Gas_constant / constants.Calorie2Joule / 1000.0 / constants.Hartree2kcalpermol  # Unit: Hartree
        nose_freq = 1.0 / (self.NHC_frequency) # Unit: fs^-1

        # Choices of Q (effective mass)
        # Q1 = NkT/w^2; Qi = kT/w^2
        # w is the frequency at which the particle thermostats fluctuate
        # https://doi.org/10.1063/1.463940
        # https://doi.org/10.1080/00268979600100761
        Qi = avg_en/1822.888515*0.529177249**2*(100.0/2.4188432)**2 # Unit: a.m.u. * A^2/(fs^2)
        
        if 'NHC_w' in kwargs:
            self.NHC_Q = [Qi/kwargs['NHC_w']**2]*self.NHC_length
        else:
            Qi = Qi / nose_freq**2 # Unit: a.m.u. * A^2
            self.NHC_Q = [3*self.Natoms*Qi]+[Qi]*(self.NHC_length-1)

        # print("Nose mass: ")
        # for i in range(len(self.NHC_Q)):
        #     print("    Q%d: %f"%(i+1,self.NHC_Q[i]))

        if self.Nys == 1:
            self.YS_list = [1.0]
        elif self.Nys == 3:
            self.YS_list = [0.828981543588751, -0.657963087177502, 0.828981543588751]
            #a = 1.0/(2-2**(1.0/3))
            #YSlist = [a,1-2*a,a]
        elif self.Nys == 5:
            self.YS_list = [0.2967324292201065, 0.2967324292201065, -0.186929716880426, 0.2967324292201065,
                0.2967324292201065]
        elif self.Nys == 7:
            self.YS_list = [0.784513610477560, 0.235573213359357, -1.17767998417887, 1.31518632068391,
                -1.17767998417887, 0.235573213359357, 0.784513610477560]
        else:
            # raise exception
            pass
        

    def update_velocities_first_half_step(self,**kwargs):
        return self.update_velocities_half_step(**kwargs)

    def update_velocities_second_half_step(self,**kwargs):
        return self.update_velocities_half_step(**kwargs)

    def update_velocities_half_step(self,**kwargs):
        if 'molecule' in kwargs:
            molecule = kwargs['molecule']
        if 'time_step' in kwargs:
            time_step = kwargs['time_step']
        # https://doi.org/10.1080/00268979600100761
        R_const = 8.3142 # Gas constant J K^-1 mol^-1
        cal2J = 4.18585182085
        hartree2kcal = 627.509474

        # Velocities of molecule
        velocities = np.array([each.xyz_velocities for each in molecule.atoms])
        mass_ = np.array([each.nuclear_mass for each in molecule.atoms])
        mass = mass_.reshape(self.Natoms,1)

        # Energy per degree of freedom
        avg_en = self.temperature * R_const / cal2J / 1000.0  # Unit: kcal/mol
        # Kinetic energy
        abc = 1822.888515 * (0.024188432 / constants.Bohr2Angstrom)**2 * hartree2kcal
        KE = np.sum(velocities**2 * mass) / 2.0
        KE = KE * 1822.888515 * (0.024188432 / constants.Bohr2Angstrom)**2 * hartree2kcal # Unit: kcal/mol
        M = len(self.NHC_Q)

        scale = 1.0
        for inc in range(self.Nc):
            for inys in range(len(self.YS_list)):
                dt_nc = self.YS_list[inys]*time_step / float(inc+1)
                dt_2 = dt_nc / 2.0
                dt_4 = dt_nc / 4.0
                dt_8 = dt_nc / 8.0

                GM = (self.NHC_Q[M-2]*self.NHC_vxi[M-2]*self.NHC_vxi[M-2]*abc-avg_en)/self.NHC_Q[M-1]/abc
                self.NHC_vxi[M-1] = self.NHC_vxi[M-1] + GM*dt_4

                for i in range(M-2):
                    ii = M-i-2
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] * np.exp(-self.NHC_vxi[ii+1]*dt_8) 
                    Gii = (self.NHC_Q[ii-1]*self.NHC_vxi[ii-1]*self.NHC_vxi[ii-1]*abc-avg_en)/self.NHC_Q[ii]/abc
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] + Gii * dt_4 
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] * np.exp(-self.NHC_vxi[ii+1]*dt_8) 

                self.NHC_vxi[0] = self.NHC_vxi[0] * np.exp(-self.NHC_vxi[1]*dt_8)
                G1 = (2*KE-self.degrees_of_freedom*avg_en)/self.NHC_Q[0]/abc
                self.NHC_vxi[0] = self.NHC_vxi[0] + G1*dt_4 
                self.NHC_vxi[0] = self.NHC_vxi[0] * np.exp(-self.NHC_vxi[1]*dt_8)


                for i in range(M):
                    self.NHC_xi[i] = self.NHC_xi[i] + self.NHC_vxi[i]*dt_2
                
                scale = scale * np.exp(-self.NHC_vxi[0]*dt_2) # Scalar factor
                
                KE = KE*scale**2
                    
                self.NHC_vxi[0] = self.NHC_vxi[0] * np.exp(-self.NHC_vxi[1]*dt_8)
                G1 = (2*KE-self.degrees_of_freedom*avg_en)/self.NHC_Q[0]/abc
                self.NHC_vxi[0] = self.NHC_vxi[0] + G1*dt_4
                self.NHC_vxi[0] = self.NHC_vxi[0] * np.exp(-self.NHC_vxi[1]*dt_8)
                for i in range(M-2):
                    ii = i + 1
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] * np.exp(-self.NHC_vxi[ii+1]*dt_8) 
                    Gii = (self.NHC_Q[ii-1]*self.NHC_vxi[ii-1]*self.NHC_vxi[ii-1]*abc-avg_en)/self.NHC_Q[ii]/abc
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] + Gii * dt_4 
                    self.NHC_vxi[ii] = self.NHC_vxi[ii] * np.exp(-self.NHC_vxi[ii+1]*dt_8)


                GM = (self.NHC_Q[M-2]*self.NHC_vxi[M-2]*self.NHC_vxi[M-2]*abc-avg_en)/self.NHC_Q[M-1]/abc
                self.NHC_vxi[M-1] = self.NHC_vxi[M-1] + GM*dt_4
        velocities = velocities * scale
        for iatom in range(self.Natoms):
            molecule.atoms[iatom].xyz_velocities = velocities[iatom]
        return molecule
    