#!/usr/bin/env python3

try:
    from .MLatom import *
    from . import MLatom
    __doc__ = MLatom.__doc__
    from . import data, models, plot, simulations, stats, xyz
    from .simulations import optimize_geometry, irc, freq, thermochemistry, md, generate_initial_conditions
except:
    import MLatom
    __doc__ = MLatom.__doc__
    import data, models, plot, simulations, stats, xyz
    from simulations import optimize_geometry, irc, freq, thermochemistry, md, generate_initial_conditions

if __name__ == '__main__':
    MLatom.run()
