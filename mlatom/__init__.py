#!/usr/bin/env python3
from . import MLatom, args_class, MLtasks, header, data, models, plot, simulations, stats, xyz, namd, constants, spectra, model_cls
from .MLatom import run
from .ciopt import ci_opt
from .simulations import optimize_geometry, irc, freq, thermochemistry, md, generate_initial_conditions, filter_by_excitation_energy_window, vibrational_spectrum, md_parallel
from .data import atom, molecule, molecular_database
from .models import kreg, ani, msani
from .models import methods, aiqm1, aiqm2, aiqm3, dens, ani_methods, aimnet2_methods, gaussian_methods, pyscf_methods, orca_methods, turbomole_methods, mndo_methods, sparrow_methods, xtb_methods, dftbplus_methods, columbus_methods, dftd3_methods, dftd4_methods
from .models import MDtrajNet
from . import al_utils
from .al import al
from .gap_md import gap_md, gap_model
# add-ons
from .models import uaiqm, omnip1, omnip2x, vecmsani

try:
    from ._version import __version__
except Exception:  # _version_static.py missing (should not happen)
    __version__ = '0.0.0+unknown'


# Throttled, non-blocking "a newer version is available" notice. Covers both the
# CLI and `import mlatom`; shown on stderr at most once a day, with the PyPI
# check running in the background (skipped when offline). Opt out by setting
# MLATOM_NO_VERSION_CHECK=1.
def _check_for_updates():
    try:
        import sys
        # `mlatom -v` / --version runs its own explicit PyPI check; don't double up
        if any(a in ('-v', '--version') for a in sys.argv):
            return
        from . import _update_check
        notice = _update_check.update_notice()
        if notice:
            print(notice, file=sys.stderr)
    except Exception:
        pass

_check_for_updates()
