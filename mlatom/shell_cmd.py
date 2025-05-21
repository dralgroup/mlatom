#!/usr/bin/env python
import sys
import os
import time
#from MLatom import run

def mlatom_cmd_run(sleep=True):
    import importlib.util
    # ~POD, 2025.04.16
    # the complicated import below is required to load the same instance of mlatom,
    # which called this external script in the first place.
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path2init = os.path.join(dir_path, '__init__.py')
    dirname = os.path.basename(dir_path)
    spec = importlib.util.spec_from_file_location(dirname, path2init)
    mlatom4gaussian = importlib.util.module_from_spec(spec)
    sys.modules[dirname] = mlatom4gaussian
    spec.loader.exec_module(mlatom4gaussian)
    mlatom4gaussian.run()
    if sleep:
        time.sleep(1)

def MLatomF():
    path = os.path.abspath(os.path.dirname(__file__))
    os.system(f'{path}/MLatomF {" ".join(sys.argv[1:])}')
    time.sleep(1)

if __name__ == '__main__':
    #run()
    mlatom_cmd_run(sleep=False)
