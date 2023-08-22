import numpy as np
import os
import json
import stopper
from utils import saveXYZ

bohr2a = 0.52917721067

class DFTD4:
    def __init__(self, method='wb97x'):
        self.check_program()
        self.method = method
        self.results = {}

    def calculate(self, element, coordinate, charge=0, fxyz=None, gradientCalc=None, hessianCalc=None):
        self.natom = len(element)
        if not fxyz:
            fxyz = 'coordinate'
            saveXYZ(fxyz, element, coordinate)
        command = f"{self.progbin} {fxyz} -s --orca --json -f {self.method} -c {charge}"
        if hessianCalc:
            command = command + " --grad --hess "
        elif gradientCalc:
            command = command + " --grad "
        command = command + " > /dev/null 2>&1 "
        os.system(command)

        self.parse_results(gradientCalc, hessianCalc)

    def parse_results(self, gradientCalc, hessianCalc):
        with open('dftd4.json', 'r') as f:
            data = json.load(f)
        self.results['energy'] = np.array(data['energy'])
        if gradientCalc:
            self.results['gradient'] = np.array(data['gradient']).reshape(-1, 3) / bohr2a
        if hessianCalc:
            self.results['hessian'] = np.array(data['hessian']).reshape(-1, self.natom*3) / (bohr2a * bohr2a)
    
    def get_results(self):
        return self.results
    
    def check_program(self):
        env = os.environ.get('dftd4bin')
        if env:
           self.progbin = env
        else:
           stopper.stopMLatom('Can not find dftd4 software, please set the environment variable: export dftd4bin=...')

