#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------!
  ! universal_mlips: what MLatom's universal models are made of               !
  !---------------------------------------------------------------------------!
'''
'''
What each of MLatom's universal models is made of.

A universal model is a fixed baseline, plus a trained network, plus a dispersion
term - and which of those it has, and which one, is a fact about the model. Those
facts were previously spread over three places in two shapes: a dict of native
dispersion terms in :mod:`mlatom.dispersion`, an ``if method.startswith('AIQM1')``
chain for the baseline in the delta-learning module, and AIQM1's atomic-energy
shift in a third. One row per model, here, so a new family is described in one
place rather than found in three.

Nothing in this module computes anything; it is the table and the lookups on it.
'''

# D4 damping for wB97X, given as explicit parameters rather than asked for by
# name - [s6, s8, a1, a2], the argument dftd4 takes as --param.
#
# dftd4 4.0.0 (2025-11-11) renamed `wb97x` to `wb97x-2008` and gave the freed
# name to a different parameter set, so `-f wb97x` silently changed meaning: s8
# went -0.07519516 -> 0.5093, a1 0.45094893 -> 0.0662, a2 6.78425255 -> 5.4487.
# Which functional the new set belongs to is deliberately not stated here: its
# entry cites one paper as the functional reference and another as the damping
# fit, and the neighbouring `wb97x-rev` entry cites the same functional paper,
# so the metadata does not identify it unambiguously. That ambiguity is itself
# the argument for naming parameters instead of functionals.
#
# The models here were built against the 2008 functional, because ANI-1x and
# ANI-2x are trained on wB97X/6-31G* data and this term was chosen to match, so
# on dftd4 >= 4.0.0 they were getting a damping fitted for something else.
#
# It does not wash out. Measured on one geometry across dftd4 3.6.0 and 4.2.0,
# AIQM2 differed by 3.5 kcal/mol, and the methane-dimer well deepened from -0.65
# to -1.16 kcal/mol with its minimum moved inward - the C6 asymptote is untouched
# but the damping and C8 are refitted, so anything that changes a contact
# distance does not cancel. Overbinding of that size is what a user reporting
# "AIQM2 is bad for noncovalent interactions" would see, and only on an
# environment built after 2025-11-11.
#
# Naming the parameters is accepted by both versions and gives bit-identical
# results on each, which asking by either name cannot: `wb97x-2008` does not
# exist in 3.x. Full precision is from dftd4's own assets/parameters.toml; the
# 4-decimal values it prints are rounded and reproduce only to ~4e-9 Eh.
# UAIQM already pinned its D3 terms this way, so this is the existing convention.
D4_WB97X = [1.0, -0.07519516, 0.45094893, 6.78425255]

_D4_WB97X_TERM = {'method': 'd4', 'functional': 'wb97x',
                  'damping_function_params': D4_WB97X}

# B97-3c's D3(BJ) damping, read from s-dftd3's own parameters.toml
# ([parameter.b973c]: a1=0.37, s8=1.50, a2=4.10), in this code's order
# (s6, s8, a1, a2).  Pinned for the same reason as the D4 term above: a
# functional name is resolved by whichever version of the program is installed,
# and those names have been reassigned between major versions.  Verified to
# leave AIQM3 unchanged - s-dftd3 gives -8.4724025443131E-04 Eh on a water
# molecule both by name and from these numbers.
D3BJ_B973C = [1.0, 1.50, 0.37, 4.10]

_D3BJ_B973C_TERM = {'method': 'd3bj', 'functional': 'b973c',
                    'damping_function_params': D3BJ_B973C}

composition = {
    #                 baseline      native dispersion term                  atomic shift
    'aiqm1':      {'baseline': 'ODM2*',     'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': True},
    'aiqm1@dft':  {'baseline': 'ODM2*',     'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': True},
    'aiqm1@dft*': {'baseline': 'ODM2*',     'dispersion': None,                                     'atomic_shift': True},
    'aiqm2':      {'baseline': 'GFN2-xTB*', 'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': False},
    'aiqm2@dft':  {'baseline': 'GFN2-xTB*', 'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': False},
    'aiqm2@dft*': {'baseline': 'GFN2-xTB*', 'dispersion': None,                                     'atomic_shift': False},
    'aiqm3':      {'baseline': 'GFN2-xTB*', 'dispersion': dict(_D3BJ_B973C_TERM),                   'atomic_shift': False},
    'aiqm3@dft':  {'baseline': 'GFN2-xTB*', 'dispersion': dict(_D3BJ_B973C_TERM),                   'atomic_shift': False},
    'aiqm3@dft*': {'baseline': 'GFN2-xTB*', 'dispersion': None,                                     'atomic_shift': False},
    'omni-p1':    {'baseline': None,        'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': False},
    'omnip1':     {'baseline': None,        'dispersion': dict(_D4_WB97X_TERM),                     'atomic_shift': False},
    'omni-p2x':   {'baseline': None,        'dispersion': None,                                     'atomic_shift': False},
    'omnip2x':    {'baseline': None,        'dispersion': None,                                     'atomic_shift': False},
}


def entry(method):
    '''
    The row for ``method``, or ``None`` for a model this table does not describe
    (a plain ANI, say, which is a network and nothing else).
    '''
    if not method:
        return None
    return composition.get(str(method).casefold(), None)


def baseline_method(method):
    '''
    The fixed method a universal model sits on top of - ``'ODM2*'`` for the AIQM1
    family, ``'GFN2-xTB*'`` for AIQM2 and AIQM3 - or ``None`` for a pure network.
    '''
    row = entry(method)
    return row['baseline'] if row else None


def native_dispersion(method):
    '''
    The dispersion term a model carries as part of itself, or ``None``.

    ``None`` means two different things to a caller and they are worth keeping
    apart: this model has no dispersion of its own, versus MLatom does not know
    this model. Use :func:`entry` when the difference matters.
    '''
    row = entry(method)
    return dict(row['dispersion']) if row and row['dispersion'] else None


def has_atomic_shift(method):
    '''Whether the model carries a fixed empirical atomic-energy shift (AIQM1).'''
    row = entry(method)
    return bool(row['atomic_shift']) if row else False


def atomic_shift_energy(method, molecule):
    '''
    The per-molecule atomic-energy shift, evaluated on a copy.

    On the molecule itself it would overwrite ``molecule.energy`` - the very
    reference label a caller is subtracting from.
    '''
    from .aiqm1 import atomic_energy_shift
    scratch = molecule.copy()
    atomic_energy_shift(method=method).predict(molecule=scratch, calculate_energy=True)
    return scratch.energy


def atomic_shift_gradients(molecule):
    '''The atomic-energy shift is a constant per composition, so its gradients are zero.'''
    import numpy as np
    return np.zeros((len(molecule.atoms), 3))
