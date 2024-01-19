#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! ccsdtstarcbs: Single-point calculations with CCSD(T)*/CBS                 ! 
  ! Implementations by: Peikun Zheng, Pavlo O. Dral                           ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import os, math

fnames = ['mp2_tz' , 'mp2_qz', 'dlpno_normal_dz', 'dlpno_normal_tz', 'dlpno_tight_dz']

def read_xyz(xyzfile):
    element2number = {'H':1, 'C':6, 'N':7, 'O':8}
    number2element = {v: k for k, v in element2number.items()}
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


def gen_orca_inp(nmol, element, coord, charge=0, spin=1):
    coordstr = ''
    for i in range(len(element)):
        coordstr+= '%3s %12.8f %12.8f %12.8f\n' % (element[i], coord[i][0], coord[i][1], coord[i][2])
    inps = ['', '', '', '', '']
    inps[0] = '''! RIMP2 RIJK cc-pVTZ cc-pVTZ/JK cc-pVTZ/C
! tightscf noautostart scfconvforced miniprint nopop
'''
    inps[1] = '''! RIMP2 RIJK cc-pVQZ cc-pVQZ/JK cc-pVQZ/C
! tightscf noautostart scfconvforced miniprint nopop
'''
    inps[2] = '''! DLPNO-CCSD(T) normalPNO RIJK cc-pVDZ cc-pVDZ/C cc-pvTZ/JK
! tightscf noautostart scfconvforced miniprint nopop
'''
    inps[3] = '''! DLPNO-CCSD(T) normalPNO RIJK cc-pVTZ cc-pVTZ/C cc-pVTZ/JK
! tightscf noautostart scfconvforced miniprint nopop
'''
    inps[4] ='''! DLPNO-CCSD(T) tightPNO RIJK cc-pVDZ cc-pVDZ/C cc-pVTZ/JK
! tightscf noautostart scfconvforced miniprint nopop
'''
    for ii in range(5):
        inps[ii] += '%maxcore 4000\n' + '* xyz 0 1\n' + coordstr + '*\n'
        fname = 'temp_%d_%s.inp' % (nmol, fnames[ii])
        with open(fname, 'w') as f:
            f.write('%s' % inps[ii])

def run_orca(nmol):
    global fnames
    orcabin = '$orcabin'
    for ii in range(5):
        fname = 'temp_%d_%s' % (nmol, fnames[ii])
        os.system('%s %s.inp > %s.out 2> /dev/null' % (orcabin, fname, fname))
        #os.system('sleep 5s')

def calc_energy(nmol):
        alpha = 5.46; beta = 2.51
 
        imol = nmol
        out1 = 'temp_' + str(imol) + '_mp2_tz.out' 
        out2 = 'temp_' + str(imol) + '_mp2_qz.out' 
        out3 = 'temp_' + str(imol) + '_dlpno_normal_dz.out' 
        out4 = 'temp_' + str(imol) + '_dlpno_normal_tz.out' 
        out5 = 'temp_' + str(imol) + '_dlpno_tight_dz.out' 
        
        hf_tz = os.popen("grep 'Total Energy' " + out1 + " | awk '{printf $4}'").readlines()
        hf_qz = os.popen("grep 'Total Energy' " + out2 + " | awk '{printf $4}'").readlines()
        hf_tz = float(hf_tz[0])
        hf_qz = float(hf_qz[0])
        mp2_tz = os.popen("grep 'RI-MP2 CORRELATION ENERGY' " + out1 + " | awk '{printf $4}'").readlines()
        mp2_qz = os.popen("grep 'RI-MP2 CORRELATION ENERGY' " + out2 + " | awk '{printf $4}'").readlines()
        mp2_tz = float(mp2_tz[0]); mp2_qz = float(mp2_qz[0])
        
        npno_dz = os.popen("grep 'Final correlation energy' " + out3 + " | awk '{printf $NF}'").readlines()
        npno_tz = os.popen("grep 'Final correlation energy' " + out4 + " | awk '{printf $NF}'").readlines()
        tpno_dz = os.popen("grep 'Final correlation energy' " + out5 + " | awk '{printf $NF}'").readlines()
        npno_dz = float(npno_dz[0]); npno_tz = float(npno_tz[0]); tpno_dz = float(tpno_dz[0]);
        
        E_hf_xtrap = (math.exp(-alpha * 4**0.5) * hf_tz - math.exp(-alpha * 3**0.5) * hf_qz) \
                / (math.exp(-alpha * 4**0.5) - math.exp(-alpha * 3**0.5))
        
        E_mp2_xtrap = (4**beta * mp2_qz - 3**beta * mp2_tz) / (4**beta - 3**beta)
        
        energy = E_hf_xtrap + E_mp2_xtrap - mp2_tz + npno_tz + tpno_dz - npno_dz
        
        return energy

def calc(xyzfile, yestfile):
    numbers, elements, coordinates = read_xyz(xyzfile)
    nmols = len(numbers)
    
    for i in range(nmols):
        element = elements[i]
        coord = coordinates[i]
        gen_orca_inp(i+1, element, coord)
    
    for i in range(nmols):
        run_orca(i+1)
    
    with open(yestfile, 'w') as fyest:
        for i in range(nmols):
            energy = calc_energy(i+1)
            fyest.writelines('%.8f\n' % energy)

if __name__ == '__main__': 
    calc('sp.xyz', 'enest.dat')


