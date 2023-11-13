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
import tempfile

from .. import constants, data, simulations, stopper, models
from ..utils import doc_inherit

class pyscf_methods(models.model):
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
        pyscf_mol.build()
        pyscf.lib.misc.num_threads(n=self.nthreads)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # save temp file to temp dir
            os.environ['PYSCF_TMPDIR'] = tmpdirname
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
                        stopper.stopMLatom(errmsg)
                    pyscf_method_dft = dft.KS(pyscf_mol)
                    pyscf_method_dft.xc = f.upper()
                    pyscf_method = tddft.TDDFT(pyscf_method_dft.run())

            # DFT
            else:
                try:
                    parse_xc(self.method.upper())
                except:
                    errmsg = 'Method not supported in pyscf interface'
                    stopper.stopMLatom(errmsg)
                from pyscf import dft
                pyscf_method = dft.KS(pyscf_mol)
                pyscf_method.xc = self.method.upper()
                
            # GW

            # CASCI/CASSCF

            pyscf_method.run()

            if calculate_energy:
                molecule.energy = pyscf_method.e_tot

                if 'CCSD(T)' == self.method.upper():
                    molecule.energy = pyscf_method.e_tot + pyscf_method.ccsd_t()

            if calculate_energy_gradients:

                # FCI not supported 
                if 'FCI' == self.method.upper():
                    errmsg = 'Gradients in pyscf do not support FCI '
                    stopper.stopMLatom(errmsg)  
                
                molecule_scanner = pyscf_method.nuc_grad_method().as_scanner()
                _, molecule_gradients = molecule_scanner(pyscf_mol)
                if 'CCSD(T)' == self.method.upper():
                    from pyscf.grad import ccsd_t as ccsd_t_grad
                    # gradients of uccsdt are not supported
                    if isinstance(pyscf_method, cc.uccsd.UCCSD):
                        errmsg = 'Gradients for UCCSD(T) not supported in pyscf'
                        stopper.stopMLatom(errmsg)
                    molecule_gradients = ccsd_t_grad.Gradients(pyscf_method).kernel()
                for ii in range(len(molecule.atoms)):
                    molecule.atoms[ii].energy_gradients = molecule_gradients[ii]            
            if calculate_hessian:
                
                # HF, DFT only
                from pyscf import hessian
                if isinstance(pyscf_method, scf.hf.RHF):      
                    hess = hessian.RHF(pyscf_method).kernel()

                elif isinstance(pyscf_method, scf.uhf.UHF):
                    hess = hessian.UHF(pyscf_method).kernel()

                elif isinstance(pyscf_method, dft.rks.RKS):
                    hess = hessian.RKS(pyscf_method).kernel()
                
                elif isinstance(pyscf_method, dft.uks.UKS):
                    hess = hessian.UKS(pyscf_method).kernel()
                
                else:
                    errmsg = 'Hessian in pyscf only support HF and DFT'
                    stopper.stopMLatom(errmsg)

                ndim = len(molecule.atoms)*3
                molecule.hessian = hess.reshape(ndim, ndim)
    
    @doc_inherit
    def predict(self, molecule=None, molecular_database=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, **kwargs)
    
    def thermo_calculation(self, molecule):

        pyscf_mol = gto.Mole()
        pyscf_mol.atom = [
            [a.element_symbol, tuple(a.xyz_coordinates)]
        for a in molecule.atoms]
        pyscf_mol.basis = self.basis
        pyscf_mol.charge = molecule.charge
        pyscf_mol.spin = molecule.multiplicity-1
        pyscf_mol.verbose = 0
        pyscf_mol.build()

        if 'HF' == self.method.upper():
            pyscf_method = scf.HF(pyscf_mol)
        else:
            try:
                parse_xc(self.method.upper())
            except:
                errmsg = 'Hessian in pyscf only support (U)HF, (U)KS'
                stopper.stopMLatom(errmsg)
            from pyscf import dft
            pyscf_method = dft.KS(pyscf_mol)
            pyscf_method.xc = self.method.upper()

        try:
            pyscf_method.run()
        except:
            return False
        
        molecule.energy = pyscf_method.e_tot

        # if isinstance(pyscf_method, scf.hf.RHF):      
        #     hess = hessian.RHF(pyscf_method).kernel()

        # elif isinstance(pyscf_method, scf.uhf.UHF):
        #     hess = hessian.UHF(pyscf_method).kernel()

        # elif isinstance(pyscf_method, dft.rks.RKS):
        #     hess = hessian.RKS(pyscf_method).kernel()
        
        # elif isinstance(pyscf_method, dft.uks.UKS):
        #     hess = hessian.UKS(pyscf_method).kernel()
        hess = pyscf_method.Hessian().kernel()

        thermo_results = harmonic_analysis(pyscf_mol, hess)
        freq_wn = thermo_results['freq_wavenumber']
        idx = freq_wn.real > 0
        molecule.frequencies = freq_wn.real[idx]
        molecule.force_constants = thermo_results['force_const_dyne'].real[idx]
        molecule.reduced_masses = thermo_results['reduced_mass'].real[idx]
        # normal modes
        mode = thermo_results['norm_mode'].real[idx]
        for iatom in range(len(molecule.atoms)):
            molecule.atoms[iatom].normal_modes = []
            for ii in range(mode.shape[0]):
                xyzs = mode[ii, iatom]
                molecule.atoms[iatom].normal_modes.append(xyzs)
            molecule.atoms[iatom].normal_modes = np.array(molecule.atoms[iatom].normal_modes)
        thermo_results = thermo(pyscf_method, thermo_results['freq_au'], 298.15, 101325)
        molecule.ZPE = thermo_results['ZPE'][0]
        #molecule.DeltaE2U = 
        #molecule.DeltaE2H = 
        #molecule.DeltaE2G =
        molecule.U0 = thermo_results['E0'][0]
        molecule.H0 = molecule.U0
        molecule.U = thermo_results['E_tot'][0]
        molecule.H = thermo_results['H_tot'][0]
        molecule.G = thermo_results['G_tot'][0]
        molecule.S = thermo_results['S_tot'][0]

        return True 
if __name__ == '__main__':
    pass
                

