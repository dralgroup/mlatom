import sys
import os
import time

def mlatom():
    path = os.path.abspath(os.path.dirname(__file__))
    py = sys.executable
    os.system(f'{py} {path}/MLatom.py {" ".join(sys.argv[1:])}')
    time.sleep(1)

def MLatomF():
    path = os.path.abspath(os.path.dirname(__file__))
    os.system(f'{path}/MLatomF {" ".join(sys.argv[1:])}')
    time.sleep(1)
