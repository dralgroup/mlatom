'''
Interfaces to third-party software.
'''

def aiqm1():
    from ..aiqm1 import aiqm1 as interface
    return interface

def aiqm2():
    from ..aiqm2 import aiqm2 as interface 
    return interface

def dens():
    from ..dens import dens as interface 
    return interface

def torchani():
    from .torchani_interface import torchani_methods as interface
    return interface

def aimnet2():
    from .torchani_interface import aimnet2_methods as interface 
    return interface

def mndo():
    from .mndo_interface import mndo_methods as interface
    return interface

def sparrow():
    from .sparrow_interface import sparrow_methods as interface
    return interface

def xtb():
    from .xtb_interface import xtb_methods as interface
    return interface

def dftd4():
    from .dftd4_interface import dftd4_methods as interface
    return interface

def dftd3():
    from .dftd3_interface import dftd3_methods as interface
    return interface

def ccsdtstarcbs():
    from ..composite_methods import ccsdtstarcbs_legacy as interface
    return interface

def gaussian():
    from .gaussian_interface import gaussian_methods as interface
    return interface

def columbus():
    from .columbus_interface import columbus_methods as interface
    return interface

def turbomole():
    from .turbomole_interface import turbomole_methods as interface
    return interface

def pyscf():
    from .pyscf_interface import pyscf_methods as interface
    return interface

def orca():
    from .orca_interface import orca_methods as interface
    return interface