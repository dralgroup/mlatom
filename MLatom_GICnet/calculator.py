#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! calculator: module for ASE calculations                                   ! 
  ! Implementations by: Peikung Zheng                                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import re
import os
import ase
from ase import io
from ase.calculators.calculator import Calculator, all_changes 
import MLtasks

class MLatomCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    def __init__(self, nmol, mndokw=None):
        super(MLatomCalculator, self).__init__()
        self.nmol = nmol
        self.mndokw = mndokw

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super(MLatomCalculator, self).calculate(atoms, properties, system_changes)
        
        #print(' \t', end=' ')
        element = self.atoms.get_chemical_symbols()
        coord = self.atoms.get_positions()
        io.write('xyz_temp.dat', self.atoms, format='extxyz', plain=True)

        args2pass = np.loadtxt('taskargs', dtype=str)
        if self.mndokw is not None:
            cmd = "sed -i 's/mndokeywords.*/" + f"mndokeywords=mndokw_split{self.nmol+1}/ig' taskargs"
            os.system(cmd)

        os.system('rm -f enest.dat gradest.dat hessest.dat')
        if re.search('aiqm1', ''.join(args2pass), flags=re.IGNORECASE):
            import AIQM1
            AIQM1.AIQM1Cls(args2pass).forward()
        else:   
            args2pass = np.append(args2pass, 'geomopt')
            args2pass = args2pass.tolist()
            MLtasks.MLtasksCls.useMLmodel(args2pass)

        energy = np.loadtxt('enest.dat')
        forces = -np.loadtxt('gradest.dat', skiprows=2)
        energy *= ase.units.Hartree
        forces *= ase.units.Hartree

        self.results['energy'] = energy

        if 'forces' in properties:
            self.results['forces'] = forces

