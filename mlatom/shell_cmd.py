#!/usr/bin/env python
import sys
import os
import time
from mlatom.MLatom import run

def mlatom():
    run()
    time.sleep(1)

def MLatomF():
    path = os.path.abspath(os.path.dirname(__file__))
    os.system(f'{path}/MLatomF {" ".join(sys.argv[1:])}')
    time.sleep(1)

if __name__ == '__main__':
    run()
