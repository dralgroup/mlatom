#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_PySCF: interface to the PySCF program                           ! 
  ! Implementations by: Yuxinxin Chen                                         !
  !---------------------------------------------------------------------------! 
'''

# TODO: check convergence criterion for gradients and energy

import os
import numpy as np

from multiprocessing import cpu_count
import sys

from .. import constants, stopper
from ..model_cls import model
from ..decorators import doc_inherit


def _attach_mo_data(molecule, pyscf_method):
    '''Attach mol.mo_energies + mol.mo_occupations from a pyscf method.

    Shape and values mirror pyscf_method.mo_energy / .mo_occ verbatim:
      RHF / RKS:   ndarray shape (N,);     mo_occ ∈ {0.0, 2.0}
      UHF / UKS:   ndarray shape (2, N);   mo_occ ∈ {0.0, 1.0}
                                             — index 0 = alpha, 1 = beta
    Units: mo_energies in Hartree (consistent with mol.energy).

    Stored as np.ndarray on the molecule. ``mlatom.data`` serializes
    ndarray → JSON list at dump time and reconstructs ndarray on load
    (``class_instance_to_dict`` + ``dict_to_molecule_class_instance``);
    consumers see ndarrays in both directions, no list-vs-array drift.

    Best-effort: silently skip if the pyscf object doesn't expose the
    attrs (e.g. some post-HF wrappers). Never raise.
    '''
    try:
        mo_e = getattr(pyscf_method, "mo_energy", None)
        mo_o = getattr(pyscf_method, "mo_occ", None)
        # Post-HF wrappers (MP2, CCSD, …) may not expose mo_energy directly.
        # Fall through to the underlying SCF object.
        if mo_e is None or mo_o is None:
            underlying = getattr(pyscf_method, "_scf", None)
            if underlying is not None:
                if mo_e is None:
                    mo_e = getattr(underlying, "mo_energy", None)
                if mo_o is None:
                    mo_o = getattr(underlying, "mo_occ", None)
        if mo_e is not None and mo_o is not None:
            molecule.mo_energies = np.asarray(mo_e, dtype=float)
            molecule.mo_occupations = np.asarray(mo_o, dtype=float)
    except Exception:
        # Never let MO surfacing kill the calc.
        pass


def _dump_minimal_chkfile(molecule, pyscf_method):
    '''Persist the minimal PySCF wavefunction artifact for orbital cubes (D3b).

    Writes a MINIMAL pyscf chkfile (``mo_coeff`` + ``mo_energy`` + ``mo_occ``
    + Mole pickle only — no DIIS history, no density-matrix history, no 2e
    integrals) to ``<cwd>/pyscf_mo.chk``. Bounded by ``mo_coeff`` size:
    ~50 KB for water/6-31G*, ~600 KB for caffeine, ~10 MB for 50-atom
    def2-TZVP. The full default chkfile (which can reach 50-200 MB on
    large systems) is NOT written.

    This is the source-of-truth for the post-D3b ``cube_from_calc`` MCP
    tool's PySCF branch: read ``pyscf_mo.chk`` → reconstruct Mole +
    ``mo_coeff`` via ``pyscf.lib.chkfile.load_mol`` + ``load`` →
    ``pyscf.tools.cubegen.orbital(mol, outfile, mo_coeff[:, idx])``.

    Path convention: written to the calc's CWD (typically the engine's
    ``scr/`` dir). For OPT, this is
    overwritten on every step's SCF; the converged-geometry SCF wins
    because it fires last. For SP/FREQ at a single geometry it's the
    only write. The bundle-promotion layer in
    the calling layer is responsible for moving
    ``pyscf_mo.chk`` to ``<bundle>/scr/<stem>.pyscf.chk`` post-success
    if the bundle layout policy requires; this helper only ensures the
    artifact exists on disk in the calc's working directory.

    AO ordering convention: PySCF's native ordering (spherical or
    cartesian depending on ``mol.cart``). The chkfile carries the Mole
    pickle so the reader rebuilds the Mole with the exact same basis +
    ordering — no cross-engine convention drift.

    Best-effort: silently skip if the pyscf object doesn't have a
    usable ``.mol`` (e.g. post-HF wrappers around an SCF object) or
    if chkfile.dump raises. Never let MO persistence kill the calc.
    '''
    try:
        import os
        from pyscf import lib as _pyscflib
        # Reach the underlying SCF object if this is a post-HF wrapper.
        scf_obj = pyscf_method
        if getattr(scf_obj, "mo_coeff", None) is None:
            scf_obj = getattr(pyscf_method, "_scf", None) or pyscf_method
        mo_coeff = getattr(scf_obj, "mo_coeff", None)
        mo_energy = getattr(scf_obj, "mo_energy", None)
        mo_occ = getattr(scf_obj, "mo_occ", None)
        pyscf_mol = getattr(scf_obj, "mol", None)
        if mo_coeff is None or pyscf_mol is None:
            return
        # Numerical D3b convention: write to a stable filename in cwd
        # so a caller can find it at a deterministic path.
        chk_path = os.path.join(os.getcwd(), "pyscf_mo.chk")
        # _pyscflib.chkfile API: dump_mol writes the Mole; dump writes
        # arbitrary scalars/arrays under named keys. Mirrors what
        # mf.chkfile would persist for SCF, but only the keys we need.
        _pyscflib.chkfile.dump_mol(pyscf_mol, chk_path)
        _pyscflib.chkfile.dump(chk_path, "scf/mo_coeff", mo_coeff)
        if mo_energy is not None:
            _pyscflib.chkfile.dump(chk_path, "scf/mo_energy", mo_energy)
        if mo_occ is not None:
            _pyscflib.chkfile.dump(chk_path, "scf/mo_occ", mo_occ)
    except Exception:
        # Never let chkfile persistence kill the calc.
        pass

def _split_unrestricted(method):
    '''Split a leading unrestricted "U" off a method name.

    The "U" prefix is the standard notation for a spin-unrestricted reference,
    e.g. UB3LYP, UPBE0, UHF. Returns (method_without_U, unrestricted), keeping
    any leading TD-/TDA- excited-state prefix in place: "UB3LYP" -> ("B3LYP",
    True), "TD-UB3LYP" -> ("TD-B3LYP", True), "B3LYP" -> ("B3LYP", False).
    '''
    import re
    m = re.match(r'(?i)(td-|tda-)?u(.+)', method)
    if m:
        return (m.group(1) or '') + m.group(2), True
    return method, False

class OMP_pyscf(model):
    def set_num_threads(self, nthreads=0):
        super().set_num_threads(nthreads)
        if self.nthreads:
            import pyscf
            # os.environ["OMP_NUM_THREADS"] = str(self.nthreads)
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
    supported_methods=["HF", 'MP2', "FCI", "CISD", "CCSD", "CCSD(T)", "DM21", "DM21m", "DM21mu", "DM21mc"]
    
    def __init__(self, method='B3LYP/6-31g', nthreads=None, density_fitting=False):
        self.init_kwargs = {'method': method, 'nthreads': nthreads, 'density_fitting': density_fitting}
        
        self.method = method.split('/')[0]
        self.method, self.unrestricted = _split_unrestricted(self.method)
        if not 'DM21' in self.method.upper():
            if 'PYSCF_PATH' in os.environ:
                sys.path.insert(0,os.environ['PYSCF_PATH'])
       
        self.basis = method.split('/')[1]

        if nthreads is None:
            self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads
        self.density_fitting = density_fitting
            
    @classmethod
    def is_method_supported(cls, method):
        # methods can be used without `method` keywords:
        # DFT, HF, MP2, FCI, CISD, CCSD, CCSD(T) / [basis set]
        if not 'DM21' in method.upper():
            if 'PYSCF_PATH' in os.environ:
                sys.path.insert(0,os.environ['PYSCF_PATH'])
        method = method.split('/')[0]
        method, _ = _split_unrestricted(method)
        if method.casefold() in [m.casefold() for m in cls.supported_methods]:
            return True
        try:
            from pyscf.dft.libxc import parse_xc
            if 'TDA'.casefold() in method.casefold():
                import re
                method = re.sub('tda-', '', method, flags=re.IGNORECASE)
            elif 'TD'.casefold() in method.casefold():
                import re
                method = re.sub('td-', '', method, flags=re.IGNORECASE)
            parse_xc(method)
            return True
        except:
            return False
            
    def predict_for_molecule(self, molecule=None, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=False, calculate_dipole_derivatives=False, nstates=1, current_state=0, **kwargs):
        from pyscf import gto, scf
        from pyscf.dft.libxc import parse_xc
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
        # scf.HF / dft.KS already return the unrestricted variant when the
        # spin is non-zero; self.unrestricted (set by a leading "U" on the
        # method) forces it even for a closed-shell spin.
        if 'HF' == self.method.upper():
            pyscf_method = (scf.UHF if self.unrestricted else scf.HF)(pyscf_mol)
        
        # MP2
        elif 'MP2' == self.method.upper():
            from pyscf import mp
            pyscf_method = mp.MP2(pyscf_mol)
        
        # CISD
        elif 'CISD' == self.method.upper():
            from pyscf import ci
            pyscf_method_hf = (scf.UHF if self.unrestricted else scf.HF)(pyscf_mol)
            pyscf_method = ci.CISD(pyscf_method_hf.run())
        
        # Full CI
        elif 'FCI' == self.method.upper():
            from pyscf import fci
            pyscf_method_hf = (scf.UHF if self.unrestricted else scf.HF)(pyscf_mol)
            pyscf_method = fci.FCI(pyscf_method_hf.run())

        # CCSD and CCSD(T)
        elif 'CCSD' in self.method.upper():
            from pyscf import cc
            pyscf_method_hf = (scf.UHF if self.unrestricted else scf.HF)(pyscf_mol)
            pyscf_method = cc.CCSD(pyscf_method_hf.run())

        elif 'TDA' in self.method.upper():
            from pyscf import tddft,dft
            import re
            try:
                method = re.sub('tda-', '', self.method, flags=re.IGNORECASE)
                method = method.upper()
                parse_xc(method)
            except:
                errmsg = 'Method not supported in pyscf interface'
                raise ValueError(errmsg)
            pyscf_method = (dft.UKS if self.unrestricted else dft.KS)(pyscf_mol)
            pyscf_method.xc = method
            pyscf_method.run()
            pyscf_method_tda = tddft.TDA(pyscf_method.run()).run(nstates=nstates)
            if not molecule.electronic_states:
                 molecule.electronic_states.extend([molecule.copy() for _ in range(nstates)])
            # Per-state gradients honesty: seed every state with NaN gradients so that any state
            # whose gradient is not freshly computed below stays NaN rather than inheriting the
            # molecule's stale (carried) atom-level energy_gradients.
            for _es in molecule.electronic_states:
                _es.add_xyz_derivative_property(np.full((len(_es.atoms), 3), np.nan), 'energy', 'energy_gradients')
            molecule.electronic_states[0].energy = pyscf_method.e_tot
            for ii in range(1,nstates):
               molecule.electronic_states[ii].energy = molecule.electronic_states[0].energy + pyscf_method_tda.e[ii-1]

            if calculate_energy_gradients:
                _state_gradients = []
                _state_gradients.append(pyscf_method.nuc_grad_method().kernel()/constants.Bohr2Angstrom)
                molecule.electronic_states[0].energy_gradients = _state_gradients[0]
                tdg_tda = pyscf_method_tda.nuc_grad_method()
                for ii in range(1,nstates):
                   _state_gradients.append(tdg_tda.kernel(state=ii)/constants.Bohr2Angstrom) 
                   molecule.electronic_states[ii].energy_gradients = _state_gradients[ii]
                molecule.energy = molecule.electronic_states[current_state].energy
                molecule.energy_gradients = molecule.electronic_states[current_state].energy_gradients
            molecule.oscillator_strengths = pyscf_method_tda.oscillator_strength()[:-1]

        elif 'TD' in self.method.upper():
            from pyscf import tddft,dft
            import re
            #from  mlatom.interfaces.TDDFT_Davidson import  eigen_solver, mod_davidson_solver
            try:
                method = re.sub('td-', '', self.method, flags=re.IGNORECASE)
                method = method.upper()
                parse_xc(method)
            except:
                errmsg = 'Method not supported in pyscf interface'
                raise ValueError(errmsg)
            pyscf_method = (dft.UKS if self.unrestricted else dft.KS)(pyscf_mol)
            pyscf_method.xc = method
            pyscf_method_tddft = tddft.TDDFT(pyscf_method.run()).run(nstates=nstates)
            if not molecule.electronic_states:
                 molecule.electronic_states.extend([molecule.copy() for _ in range(nstates)])
            # Per-state gradients honesty: seed every state with NaN gradients so that any state
            # whose gradient is not freshly computed below stays NaN rather than inheriting the
            # molecule's stale (carried) atom-level energy_gradients.
            for _es in molecule.electronic_states:
                _es.add_xyz_derivative_property(np.full((len(_es.atoms), 3), np.nan), 'energy', 'energy_gradients')
            molecule.electronic_states[0].energy = pyscf_method.e_tot
            for ii in range(1,nstates):
               molecule.electronic_states[ii].energy = molecule.electronic_states[0].energy + pyscf_method_tddft.e[ii-1]

            if calculate_energy_gradients:
                _state_gradients = []
                _state_gradients.append(pyscf_method.nuc_grad_method().kernel()/constants.Bohr2Angstrom)
                molecule.electronic_states[0].energy_gradients = _state_gradients[0]
                tdg_tddft = pyscf_method_tddft.nuc_grad_method()
                for ii in range(1,nstates):
                   _state_gradients.append(tdg_tddft.kernel(state=ii)/constants.Bohr2Angstrom)
                   molecule.electronic_states[ii].energy_gradients = _state_gradients[ii]
                molecule.energy = molecule.electronic_states[current_state].energy
                molecule.energy_gradients = molecule.electronic_states[current_state].energy_gradients
            molecule.oscillator_strengths = pyscf_method_tddft.oscillator_strength()[:-1]

        # DFT
        else:
            try:
                parse_xc(self.method.upper())
            except:
                errmsg = 'Method not supported in pyscf interface'
                raise ValueError(errmsg)
            from pyscf import dft
            pyscf_method = (dft.UKS if self.unrestricted else dft.KS)(pyscf_mol)
            pyscf_method.xc = self.method.upper()
            
        # GW

        # CASCI/CASSCF

        # check if the method support density fitting
        if self.method.upper() in ['MP2', "FCI", "CISD", "CCSD", "CCSD(T)"] and self.density_fitting:
            self.density_fitting = False
            print(f'Density fitting is not supported for {self.method} in pyscf and will be turned off automatically')

        if calculate_energy:
            if not calculate_energy_gradients and not calculate_hessian and self.density_fitting:
                # density_fit() returns a NEW mean-field object and leaves the original
                # un-run, so the energy has to be read from the object that ran. It used
                # to be rebuilt as sum(scf_summary.values()) on the original instead,
                # which held only as long as that dict contained exactly the additive
                # energy components. PySCF 2.14 also files 'e2' in there - already
                # counted as coul + exc - and 'gap', which is a HOMO-LUMO gap and not an
                # energy at all, so the sum came back as -0.73 Hartree for CH4 where the
                # answer is -40.51, with nothing raised. Reading e_tot off the object
                # that ran needs no such assumption. Density fitting now also goes
                # through the same convergence check as every other method, rather than
                # storing whatever the run left behind.
                pyscf_method = pyscf_method.density_fit()
                pyscf_method.run()
            else:
                pyscf_method.kernel()

            converged = self.check_convergence(pyscf_method)
            if converged:
                molecule.energy = pyscf_method.e_tot
                if 'CCSD(T)' == self.method.upper():
                    molecule.energy = pyscf_method.e_tot + pyscf_method.ccsd_t()
                _attach_mo_data(molecule, pyscf_method)
                _dump_minimal_chkfile(molecule, pyscf_method)
            else:
                print("PySCF doesn't converge and energy will not be stored in molecule")

        if calculate_energy_gradients and self.method.upper() not in ['TDA','TDDFT']:
            # FCI not supported 
            if 'FCI' == self.method.upper():
                errmsg = 'Gradients in pyscf do not support FCI '
                raise ValueError(errmsg)  
            
            # NOTE: PySCF use Bohr as unit by default for gradients calculation
            molecule_gradients = pyscf_method.nuc_grad_method().kernel()
            
            converged = self.check_convergence(pyscf_method)
            if converged:
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


            else:
                print("PySCF doesn't converge and gradients will not be stored in molecule")            
        if calculate_hessian:
            method_type = ''

            # HF, DFT only
            from pyscf import hessian
            from pyscf import dft
            # rks inherits rhf, check rks first
            if isinstance(pyscf_method, dft.rks.RKS):
                method_type = 'rks'
                #hess = hessian.RKS(pyscf_method).kernel()
                pass
            
            elif isinstance(pyscf_method, dft.uks.UKS):
                method_type = 'uks'
                #hess = hessian.UKS(pyscf_method).kernel()
                pass

            elif isinstance(pyscf_method, scf.hf.RHF):      
                method_type = 'rhf'
                #hess = hessian.RHF(pyscf_method).kernel()
                pass

            elif isinstance(pyscf_method, scf.uhf.UHF):
                method_type = 'uhf'
                #hess = hessian.UHF(pyscf_method).kernel()
                pass

            else:
                errmsg = 'Hessian in pyscf only support HF and DFT'
                raise ValueError(errmsg)

            # NOTE: PySCF use Bohr as unit by default for hessian calculation
            hess = pyscf_method.Hessian().kernel()

            converged = self.check_convergence(pyscf_method)
            if converged:
                natom = len(molecule.atoms)
                h = np.zeros((3*natom, 3*natom))
                for ii in range(natom):
                    for jj in range(natom):
                        h[ii*3:(ii+1)*3, jj*3:(jj+1)*3] = hess[ii][jj]
                molecule.hessian = h / constants.Bohr2Angstrom**2

                if calculate_dipole_derivatives:
                    calc_dipole_derivatives(pyscf_method,molecule,method_type)
            else:
                print("PySCF doesn't converge and hessian will not be stored in molecule")

    def check_convergence(self,pyscf_method):
        if 'converged' in pyscf_method.__dict__:
            return pyscf_method.converged
        else:
            # For MP2
            return pyscf_method._scf.converged


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
            _attach_mo_data(molecule, pyscf_method)
            _dump_minimal_chkfile(molecule, pyscf_method)

        if calculate_energy_gradients:
            errmsg = 'DM21 by pyscf does not support gradients calculation'
            raise ValueError(errmsg)

        if calculate_hessian:
            errmsg = 'DM21 by pyscf does not support hessian calculation'
            raise ValueError(errmsg)

    @doc_inherit
    def predict(self, molecule=None, molecular_database=None, calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, nstates=1, current_state=1, **kwargs):
        '''
            **kwargs: ``# needs to be documented``.
        '''
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB.molecules:
            self.predict_for_molecule(molecule=mol, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, \
                    calculate_hessian=calculate_hessian, nstates=nstates, current_state=current_state, **kwargs)


def thermo_calculation(molecule):
    # construct pyscf molecule object
    from pyscf.hessian.thermo import harmonic_analysis
    from pyscf import gto
    import numpy
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
    # Pyscf provides mass deweighted unnormalized normal modes. Normalize them here.
    for imode in range(mode.shape[0]):
        # mode /= np.sqrt(np.sum(mode[imode]**2))
        mode[imode] /= np.sqrt(np.sum(mode[imode]**2))
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

    import numpy
    from pyscf.data import nist
    from pyscf.hessian.thermo import _get_rotor_type, rotation_const, rotational_symmetry_number

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

# There is no guarantee that this function gives correct dipole derivatives,
# as the package itself is not fully tested
def calc_dipole_derivatives(pyscf_method,mol,method_type=''):
    import warnings
    try:
        # Ignore tons of warnings in pyscf.prop package
        warnings.filterwarnings("ignore")
        from pyscf.prop.infrared import rhf,rks,uhf,uks
        # Raise a much simpler warning
        warnings.filterwarnings("default")
        stopper.warningMLatom('Trying to calculate dipole derivatives using pyscf.prop package. It is not fully tested.')
    except:
        stopper.warningMLatom('Essential package for dipole derivatives not found, please refer to https://github.com/pyscf/properties/tree/master.')
        return 
    # print(method_type)
    if method_type == 'rhf':
        mf_ir = rhf.Infrared(pyscf_method)
    elif method_type == 'rks':
        mf_ir = rks.Infrared(pyscf_method)
    elif method_type == 'uhf':
        mf_ir = uhf.Infrared(pyscf_method)
    elif method_type == 'uks':
        mf_ir = uks.Infrared(pyscf_method)
    else:
        return
    dipole_derivatives = np.array(mf_ir.kernel_dipderiv()) / constants.Debye
    # print(dipole_derivatives)
    mol.dipole_derivatives = dipole_derivatives.reshape(9*len(mol))


if __name__ == '__main__':
    pass
                

