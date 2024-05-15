'''
Interfaces to third-party software.
'''

def aiqm1(**kwargs):
    from ..aiqm1 import aiqm1 as interface
    return interface(**kwargs)

def ani(**kwargs):
    from .torchani_interface import ani_methods as interface
    return interface(**kwargs)

def aimnet2(**kwargs):
    from .torchani_interface import aimnet2_methods as interface 
    return interface(**kwargs)

def mndo(**kwargs):
    from .mndo_interface import mndo_methods as interface
    return interface(**kwargs)

def sparrow(**kwargs):
    from .sparrow_interface import sparrow_methods as interface
    return interface(**kwargs)

def xtb(**kwargs):
    from .xtb_interface import xtb_methods as interface
    return interface(**kwargs)

def dftd4(**kwargs):
    from .dftd4_interface import dftd4_methods as interface
    return interface(**kwargs)

def ccsdtstarcbs(**kwargs):
    from ..composite_methods import ccsdtstarcbs_legacy as interface
    return interface(**kwargs)

def gaussian(**kwargs):
    from .gaussian_interface import gaussian_methods as interface
    return interface(**kwargs)

def columbus(**kwargs):
    from .columbus_interface import columbus_methods as interface
    return interface(**kwargs)

def turbomole(**kwargs):
    from .turbomole_interface import turbomole_methods as interface
    return interface(**kwargs)

def pyscf(**kwargs):
    from .pyscf_interface import pyscf_methods as interface
    return interface(**kwargs)

def orca(**kwargs):
    from .orca_interface import orca_methods as interface
    return interface(**kwargs)