#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_PySCF: interface to the PySCF program                           ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''

import os
import numpy as np

import pyscf
from pyscf import gto, scf
from pyscf.dft.libxc import *
from pyscf import hessian
from pyscf.hessian import thermo
from pyscf.hessian.thermo import *
from pyscf.hessian.thermo import _get_rotor_type, _get_TR
import tempfile

from .. import constants, data, simulations, stopper, models
from ..decorators import doc_inherit

class OMP_pyscf(models.model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
            pyscf.lib.misc.num_threads(n=self.nthreads)

class pyscf_methods(OMP_pyscf):
    '''
    PySCF interface

    Arguments:
        method (str): Method to use
        nthreads (int): Set the number of OMP threads

    .. note::

        Methods supported:

        Energy: HF, MP2, DFT, CISD, FCI, CCSD/CCSD(T), TD-DFT/TD-HF

        Gradients: HF, MP2, DFT, CISD, CCSD, RCCSD(T), TD-DFT/TD-HF

        Hessian: HF, DFT
        
    '''

    def __init__(self, method='B3LYP/6-31g', **kwargs):
        
        self.method = method.split('/')[0]
        self.basis = method.split('/')[1]

        if 'nthreads' in kwargs.keys():
            self.nthreads = kwargs['nthreads']
        else:
            self.nthreads = 1

    def predict_for_molecule(self, molecule=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        pyscf_mol = gto.Mole()
        pyscf_mol.atom = [
            [a.element_symbol, tuple(a.xyz_coordinates)]
        for a in molecule.atoms]
        pyscf_mol.basis = self.basis
        pyscf_mol.charge = molecule.charge
        pyscf_mol.spin = molecule.multiplicity-1
        pyscf_mol.verbose = 0
        pyscf_mol.unit = 'Ang'
        pyscf_mol.build()

        # DM21
        if 'DM21' in self.method.upper():
            self.predict_for_molecule_DM21(molecule=molecule, pyscf_mol=pyscf_mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, **kwargs)
            return

        # HF
        if 'HF' == self.method.upper():
            pyscf_method = scf.HF(pyscf_mol)
        
        # MP2
        elif 'MP2' == self.method.upper():
            from pyscf import mp
            pyscf_method = mp.MP2(pyscf_mol)
        
        # CISD
        elif 'CISD' == self.method.upper():
            from pyscf import ci
            pyscf_method_hf = scf.HF(pyscf_mol)
            pyscf_method = ci.CISD(pyscf_method_hf.run())
        
        # Full CI
        elif 'FCI' == self.method.upper():
            from pyscf import fci
            pyscf_method_hf = scf.HF(pyscf_mol)
            pyscf_method = fci.FCI(pyscf_method_hf.run())

        # CCSD and CCSD(T)
        elif 'CCSD' in self.method.upper():
            from pyscf import cc
            pyscf_method_hf = scf.HF(pyscf_mol)
            pyscf_method = cc.CCSD(pyscf_method_hf.run())

        # TDSCF/TDDFT
        elif 'TD ' in self.method.upper():
            f = self.method.split()[1]
            if 'HF' == f.upper():
                pyscf_method_hf = scf.HF(pyscf_mol).run()
                pyscf_method = pyscf_method_hf.TDHF()
            else:
                from pyscf import tddft, dft
                try:
                    parse_xc(f)
                except:
                    errmsg = 'Method not supported in pyscf interface'
                    raise ValueError(errmsg)
                pyscf_method_dft = dft.KS(pyscf_mol)
                pyscf_method_dft.xc = f.upper()
                pyscf_method = tddft.TDDFT(pyscf_method_dft.run())

        # DFT
        else:
            try:
                parse_xc(self.method.upper())
            except:
                errmsg = 'Method not supported in pyscf interface'
                raise ValueError(errmsg)
            from pyscf import dft
            pyscf_method = dft.KS(pyscf_mol)
            pyscf_method.xc = self.method.upper()
            
        # GW

        # CASCI/CASSCF

        pyscf_method.kernel()

        if calculate_energy:
            molecule.energy = pyscf_method.e_tot

            if 'CCSD(T)' == self.method.upper():
                molecule.energy = pyscf_method.e_tot + pyscf_method.ccsd_t()

        if calculate_energy_gradients:

            # FCI not supported 
            if 'FCI' == self.method.upper():
                errmsg = 'Gradients in pyscf do not support FCI '
                raise ValueError(errmsg)  
            
            # NOTE: PySCF use Bohr as unit by default for gradients calculation
            molecule_gradients = pyscf_method.nuc_grad_method().kernel()
            molecule_gradients = molecule_gradients/constants.Bohr2Angstrom

            if 'CCSD(T)' == self.method.upper():
                from pyscf.grad import ccsd_t as ccsd_t_grad
                # gradients of uccsdt are not supported
                if isinstance(pyscf_method, cc.uccsd.UCCSD):
                    errmsg = 'Gradients for UCCSD(T) not supported in pyscf'
                    raise ValueError(errmsg)
                molecule_gradients = ccsd_t_grad.Gradients(pyscf_method).kernel()
            for ii in range(len(molecule.atoms)):
                molecule.atoms[ii].energy_gradients = molecule_gradients[ii]            
        if calculate_hessian:
            
            # HF, DFT only
            from pyscf import hessian
            if isinstance(pyscf_method, scf.hf.RHF):      
                #hess = hessian.RHF(pyscf_method).kernel()
                pass

            elif isinstance(pyscf_method, scf.uhf.UHF):
                #hess = hessian.UHF(pyscf_method).kernel()
                pass

            elif isinstance(pyscf_method, dft.rks.RKS):
                #hess = hessian.RKS(pyscf_method).kernel()
                pass
            
            elif isinstance(pyscf_method, dft.uks.UKS):
                #hess = hessian.UKS(pyscf_method).kernel()
                pass
            
            else:
                errmsg = 'Hessian in pyscf only support HF and DFT'
                raise ValueError(errmsg)

            # NOTE: PySCF use Bohr as unit by default for hessian calculation
            hess = pyscf_method.Hessian().kernel()
            natom = len(molecule.atoms)
            h = np.zeros((3*natom, 3*natom))
            for ii in range(natom):
                for jj in range(natom):
                    h[ii*3:(ii+1)*3, jj*3:(jj+1)*3] = hess[ii][jj]
            molecule.hessian = h / constants.Bohr2Angstrom**2

    def predict_for_molecule_DM21(self, molecule=None, pyscf_mol=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        # reference: https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/density_functional_approximation_dm21
        # METHODS AVAILABE:
        # DM21 - trained on molecules dataset, and fractional charge, and fractional spin constraints.
        # DM21m - trained on molecules dataset.
        # DM21mc - trained on molecules dataset, and fractional charge constraints.
        # DM21mu - trained on molecules dataset, and electron gas constraints.

        from pyscf import dft
        try:
            import density_functional_approximation_dm21 as dm21
        except:
            errmsg = 'Please install required packages for DM21. For more details, please refer to https://github.com/google-deepmind/deepmind-research/tree/master/density_functional_approximation_dm21.'
            raise ModuleNotFoundError(errmsg)
        
        if pyscf_mol.spin == 0:
            pyscf_method = dft.RKS(pyscf_mol)
        else:
            pyscf_method = dft.UKS(pyscf_mol)

        pyscf_method.xc = 'B3LYP'
        pyscf_method.run()
        dm0 = pyscf_method.make_rdm1()
        pyscf_method._numint = dm21.NeuralNumInt(dm21.Functional.__dict__[self.method])
        # relax convergence tolerances to increase success to convergence
        pyscf_method.conv_tol = 1E-6
        pyscf_method.conv_tol_grad = 1E-3
        try:
            pyscf_method.kernel(dm0=dm0)
        except:
            errmsg = 'DM21 cannot converge properly'
            raise ValueError(errmsg)
                
        if calculate_energy:
            molecule.energy = pyscf_method.e_tot

        if calculate_energy_gradients:           
            errmsg = 'DM21 by pyscf does not support gradients calculation'
            raise ValueError(errmsg)

        if calculate_hessian:
            errmsg = 'DM21 by pyscf does not support hessian calculation'
            raise ValueError(errmsg)

    @doc_inherit
    def predict(self, molecule=None, molecular_database=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, **kwargs)


def thermo_calculation(molecule):
    # construct pyscf molecule object
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = [[a.element_symbol, tuple(a.xyz_coordinates)] for a in molecule.atoms]
    #pyscf_mol.basis = method.split('/')[1]
    pyscf_mol.charge = molecule.charge
    pyscf_mol.spin = molecule.multiplicity-1
    pyscf_mol.verbose = 0
    pyscf_mol.build()
    pyscf_mol.energy = molecule.energy
    #method = method.split('/')[0]

    # reconstruct hessian
    natom = len(molecule.atoms)
    h = molecule.hessian
    hess = np.zeros((natom, natom, 3, 3))
    for ii in range(natom):
        for jj in range(natom):
            hess[ii][jj] = h[ii*3:(ii+1)*3, jj*3:(jj+1)*3]
    hess = hess / constants.Angstrom2Bohr**2
    # harmonic analysis
    
    thermo_results = harmonic_analysis(pyscf_mol, hess, imaginary_freq=True)
    freq_wn = thermo_results['freq_wavenumber']
    if numpy.iscomplexobj(freq_wn):
        freq_wn = freq_wn.real - abs(freq_wn.imag)
    molecule.frequencies = freq_wn
    molecule.force_constants = thermo_results['force_const_dyne'].real
    molecule.reduced_masses = thermo_results['reduced_mass'].real
    # normal modes
    mode = thermo_results['norm_mode'].real
    for iatom in range(len(molecule.atoms)):
        molecule.atoms[iatom].normal_modes = []
        for ii in range(mode.shape[0]):
            xyzs = mode[ii, iatom]
            molecule.atoms[iatom].normal_modes.append(xyzs)
        molecule.atoms[iatom].normal_modes = np.array(molecule.atoms[iatom].normal_modes)
    thermo_results = thermo_modified_from_pyscf(pyscf_mol, thermo_results['freq_au'], 298.15, 101325)
    molecule.ZPE = thermo_results['ZPE'][0]
    #molecule.DeltaE2U = 
    #molecule.DeltaE2H = 
    #molecule.DeltaE2G =
    molecule.U0 = thermo_results['E_0K'][0]
    molecule.H0 = molecule.U0
    molecule.U = thermo_results['E_tot'][0]
    molecule.H = thermo_results['H_tot'][0]
    molecule.G = thermo_results['G_tot'][0]
    molecule.S = thermo_results['S_tot'][0]

    return True 

def thermo_modified_from_pyscf(molecule, freq, temperature=298.15, pressure=101325):
# https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

    from functools import reduce
    import numpy
    from pyscf import lib
    from pyscf.data import nist

    mol = molecule
    atom_coords = mol.atom_coords()
    mass = mol.atom_mass_list(isotope_avg=True)
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center

    kB = nist.BOLTZMANN
    h = nist.PLANCK
    # c = nist.LIGHT_SPEED_SI
    # beta = 1. / (kB * temperature)
    R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)

    results = {}
    results['temperature'] = (temperature, 'K')
    results['pressure'] = (pressure, 'Pa')

    E0 = mol.energy
    results['E0'] = (E0, 'Eh')

    # Electronic part
    results['S_elec' ] = (R_Eh * numpy.log(mol.multiplicity), 'Eh/K')
    results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
    results['E_elec' ] = results['H_elec' ] = (E0, 'Eh')

    # Translational part. See also https://cccbdb.nist.gov/thermo.asp for the
    # partition function q_trans
    mass_tot = mass.sum() * nist.ATOMIC_MASS
    q_trans = ((2.0 * numpy.pi * mass_tot * kB * temperature / h**2)**1.5
               * kB * temperature / pressure)
    results['S_trans' ] = (R_Eh * (2.5 + numpy.log(q_trans)), 'Eh/K')
    results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
    results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
    results['E_trans' ] = (1.5 * R_Eh * temperature, 'Eh')
    results['H_trans' ] = (2.5 * R_Eh * temperature, 'Eh')

    # Rotational part
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    results['rot_const'] = (rot_const, 'GHz')
    rotor_type = _get_rotor_type(rot_const)

    sym_number = rotational_symmetry_number(mol)
    results['sym_number'] = (sym_number, '')

    # partition function q_rot (https://cccbdb.nist.gov/thermo.asp)
    if rotor_type == 'ATOM':
        results['S_rot' ] = (0, 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (0, 'Eh')
    elif rotor_type == 'LINEAR':
        B = rot_const[1] * 1e9
        q_rot = kB * temperature / (sym_number * h * B)
        results['S_rot' ] = (R_Eh * (1 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (R_Eh * temperature, 'Eh')
    else:
        ABC = rot_const * 1e9
        q_rot = ((kB*temperature/h)**1.5 * numpy.pi**.5
                 / (sym_number * numpy.prod(ABC)**.5))
        results['S_rot' ] = (R_Eh * (1.5 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (1.5 * R_Eh * temperature, 'Eh')

    # Vibrational part.
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    idx = freq.real > 0
    vib_temperature = freq.real[idx] * au2hz * h / kB
    # reduced_temperature
    rt = vib_temperature / max(1e-14, temperature)
    e = numpy.exp(-rt)

    ZPE = R_Eh * .5 * vib_temperature.sum()
    results['ZPE'] = (ZPE, 'Eh')

    results['S_vib' ] = (R_Eh * (rt*e/(1-e) - numpy.log(1-e)).sum(), 'Eh/K')
    results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (e * rt**2/(1-e)**2).sum(), 'Eh/K')
    results['E_vib' ] = results['H_vib' ] = \
            (ZPE + R_Eh * temperature * (rt * e / (1-e)).sum(), 'Eh')

    results['G_elec' ] = (results['H_elec' ][0] - temperature * results['S_elec' ][0], 'Eh')
    results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
    results['G_rot'  ] = (results['H_rot'  ][0] - temperature * results['S_rot'  ][0], 'Eh')
    results['G_vib'  ] = (results['H_vib'  ][0] - temperature * results['S_vib'  ][0], 'Eh')

    def _sum(f):
        keys = ('elec', 'trans', 'rot', 'vib')
        return sum(results.get(f+'_'+key, (0,))[0] for key in keys)
    results['S_tot' ] = (_sum('S' ), 'Eh/K')
    results['Cv_tot'] = (_sum('Cv'), 'Eh/K')
    results['Cp_tot'] = (_sum('Cp'), 'Eh/K')
    results['E_0K' ]  = (E0 + ZPE, 'Eh')
    results['E_tot' ] = (_sum('E'), 'Eh')
    results['H_tot' ] = (_sum('H'), 'Eh')
    results['G_tot' ] = (_sum('G'), 'Eh')

    return results


if __name__ == '__main__':
    pass
                

