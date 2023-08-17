#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! thermo: thermochemical calculations                                       ! 
  ! Implementations by: Peikung Zheng                                         ! 
  !---------------------------------------------------------------------------! 
'''
import numpy as np
import math
import os
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def vibrational_analysis(masses, hessian, mode_type='MDU'):
    # this function is modified from TorchANI
    """Computing the vibrational wavenumbers from hessian.

    Note that normal modes in many popular software packages such as
    Gaussian and ORCA are output as mass deweighted normalized (MDN).
    Normal modes in ASE are output as mass deweighted unnormalized (MDU).
    Some packages such as Psi4 let ychoose different normalizations.
    Force constants and reduced masses are calculated as in Gaussian.

    mode_type should be one of:
    - MWN (mass weighted normalized)
    - MDU (mass deweighted unnormalized)
    - MDN (mass deweighted normalized)

    MDU modes are not orthogonal, and not normalized,
    MDN modes are not orthogonal, and normalized.
    MWN modes are orthonormal, but they correspond
    to mass weighted cartesian coordinates (x' = sqrt(m)x).
    """
    mhessian2fconst = 4.359744650780506
    unit_converter = 17091.7006789297
    # Solving the eigenvalue problem: Hq = w^2 * T q
    # where H is the Hessian matrix, q is the normal coordinates,
    # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
    # We solve this eigenvalue problem through Lowdin diagnolization:
    # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
    # Letting q' = T^(1/2) q, we then have
    # T^(-1/2) H T^(-1/2) q' = w^2 * q'
    inv_sqrt_mass = np.repeat(np.sqrt(1 / masses), 3, axis=1) # shape (molecule, 3 * atoms)
    mass_scaled_hessian = hessian * np.expand_dims(inv_sqrt_mass, axis=1) * np.expand_dims(inv_sqrt_mass, axis=2)
    if mass_scaled_hessian.shape[0] != 1:
        raise ValueError('The input should contain only one molecule')
    mass_scaled_hessian = np.squeeze(mass_scaled_hessian, axis=0)
    eigenvalues, eigenvectors = np.linalg.eig(mass_scaled_hessian)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    angular_frequencies = np.sqrt(eigenvalues)
    frequencies = angular_frequencies / (2 * math.pi)
    # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 
    wavenumbers = unit_converter * frequencies

    # Note that the normal modes are the COLUMNS of the eigenvectors matrix
    mw_normalized = eigenvectors.T
    md_unnormalized = mw_normalized * inv_sqrt_mass
    norm_factors = 1 / np.linalg.norm(md_unnormalized, axis=1)  # units are sqrt(AMU)
    md_normalized = md_unnormalized * np.expand_dims(norm_factors, axis=1)

    rmasses = norm_factors**2  # units are AMU
    # converting from Ha/(AMU*A^2) to mDyne/(A*AMU) 
    fconstants = mhessian2fconst * eigenvalues * rmasses  # units are mDyne/A

    if mode_type == 'MDN':
        modes = (md_normalized).reshape(frequencies.size, -1, 3)
    elif mode_type == 'MDU':
        modes = (md_unnormalized).reshape(frequencies.size, -1, 3)
    elif mode_type == 'MWN':
        modes = (mw_normalized).reshape(frequencies.size, -1, 3)

    return wavenumbers, modes, fconstants, rmasses

def thermocalc(mol, linear, sn, mult):
    import ase
    from ase.thermochemistry import IdealGasThermo
    import ase.units as units
    
    print(' %s ' % ('='*78))
    print(' %s Vibration analysis' % (' '*30))
    print(' %s ' % ('='*78))
    print(' Multiplicity: %s' % mult)
    print(' Rotational symmetry number: %s' % sn)
    if linear:
        geometry = 'linear'
    else:
        geometry = 'nonlinear'
    spin = (mult - 1) / 2
    print(f' This is a {geometry} molecule')
    numbers = mol.get_atomic_numbers()
    nat = len(numbers)
    energy = np.loadtxt('enest.dat') * units.Hartree
    hessian = np.loadtxt('hessest.dat', skiprows=2).reshape(-1, nat*3)
    masses = np.expand_dims(mol.get_masses(), axis=0)
    freq, modes, fconstants, rmasses = vibrational_analysis(masses, hessian, mode_type='MDU')
    
    print('   Mode     Frequencies     Reduced masses     Force Constants')
    print('              (cm^-1)            (AMU)           (mDyne/A)')
    for i in range(len(freq)):
        print('%6d %15.4f %15.4f %18.4f' % (i+1, freq[i], rmasses[i], fconstants[i]))
        
    print(' %s ' % ('='*78))
    print(' %s Thermochemistry' % (' '*30))
    print(' %s ' % ('='*78))
    cm2ev = 100.0 * units._c * units._hplanck / units._e
    ev2kcal = units.mol / units.kcal
    vib_energies = freq * cm2ev
    thermo = IdealGasThermo(vib_energies=vib_energies,
                            potentialenergy=energy,
                            atoms=mol,
                            geometry=geometry,
                            symmetrynumber=sn, spin=spin)
    G = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.) * ev2kcal
    H298 = thermo.get_enthalpy(temperature=298.15, verbose=False) * ev2kcal
    H0 = thermo.get_enthalpy(temperature=0.0, verbose=False) * ev2kcal
    ZPE = thermo.get_ZPE_correction() * ev2kcal
    print(' %-40s: %15.5f kcal/mol' % ('Zero-point vibrational energy', ZPE))
    
    return energy, ZPE, H298

def hofcalc(hofmethod, energy, ZPE, H298, numbers):
    # energy in eV, ZPE, H298 in kcal/mol
    au2kcal = 627.5094738898777
    ev2kcal = 23.060548012069493
    au2ev = 27.211386024367243 
    if   hofmethod == 'aiqm1':
        E_atom = {1:-0.50088038, 6:-37.79221710, 7:-54.53360298, 8:-75.00986203}
    elif hofmethod == 'aiqm1dft' or hofmethod == 'aiqm1dftstart':
        E_atom = {1:-0.50139362, 6:-37.84623117, 7:-54.59175573, 8:-75.07674376}
    elif hofmethod == 'ani1ccx':
        E_atom = {1:-0.50088088, 6:-37.79199048, 7:-54.53379230, 8:-75.00968205}
    H0_atom = [52.102, 170.89, 113.00, 59.559]
    deltaH_atom = [1.4811, 1.4811, 1.4811, 1.4811]
    count = Counter(numbers)
    ncount = []
    for i in E_atom.keys():
        ncount.append(count[i])
    ncount = np.array(ncount)
    Ea = np.array(list(E_atom.values()))
    
    H0 = energy * ev2kcal + ZPE
    D0 = np.dot(Ea, ncount) * au2kcal - H0
    deltaH298 = np.dot(H0_atom, ncount) - D0 + (H298 - H0) - np.dot(deltaH_atom, ncount)
    e_std = np.loadtxt('std')
    fmt = ' %-40s: %15.5f kcal/mol'
    print(fmt % ('Atomization enthalpy at 0 K', D0))
    print(fmt % ('ZPE exclusive atomization energy at 0 K', D0 + ZPE))
    print(fmt % ('Heat of formation at 298.15 K', deltaH298))

    if   hofmethod == 'aiqm1':
        if e_std > 0.41:
            print(' * Warning * Heat of formation have high uncertainty!')
    elif hofmethod == 'ani1ccx':
        if e_std > 1.68:
            print(' * Warning * Heat of formation have high uncertainty!')
    print('\n')



def get_gau_thermo(imol, hofmethod):
    au2kcal = 627.5094738898777
    au2ev = 27.211386024367243 
    gau_log = f'mol_{imol}.log'
    energy = os.popen("grep 'Recovered energy=' " + gau_log + " | awk '{printf $3}'").readlines()[0].strip()
    energy = float(energy)
    #print(' Total energy: %15.8f Eh' % energy)
    energy = energy * au2ev
    ZPE = os.popen("grep 'Zero-point correction=' " + gau_log + " | awk '{printf $(NF-1)}'").readlines()[0].strip()
    ZPE = float(ZPE) * au2kcal
    H298 = os.popen("grep 'Sum of electronic and thermal Enthalpies=' " + gau_log + " | awk '{printf $NF}'").readlines()[0].strip()
    H298 = float(H298) * au2kcal
   
    return energy, ZPE, H298

