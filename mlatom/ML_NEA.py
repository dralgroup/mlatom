#!/usr/bin/env python3
'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! ML_NEA: ML-nuclear ensemble UV/vis spectra                                ! 
  ! Implementations by: Bao-Xin Xue                                           ! 
  !---------------------------------------------------------------------------! 
'''

import random
import time
from typing import Callable, Iterable, List, Optional, Tuple, Union, Any
import inspect
import os
import sys
import math
from subprocess import getstatusoutput, Popen, PIPE
import multiprocessing as mp
from contextlib import redirect_stdout
from ctypes import c_double, CDLL, c_long 
from functools import wraps, reduce
from itertools import product
import json
import re
from io import StringIO
import argparse
try:
    import matplotlib.pyplot as plt
except:
    pass
''' version: 1.5   script author: Bao-Xin Xue   Xiamen University
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                    │
│   ┌────────────────────────────────┐          ┌────────────────────────────────┐   │
│   │0. equilibrium and velocity calc│          │  1. user defined multiple xyz  │   │
│   └────────────────────────────────┘          └────────────────────────────────┘   │
│                    │                                           │                   │
│                    │                                           │                   │
│            Newton-X MD calc                                    │                   │
│                    │                                           │                   │
│                    ▼                                           ▼                   │
│   ┌────────────────────────────────┐          ┌────────────────────────────────┐   │
│   │  gaussian calc N conformation  │          │  gaussian calc N conformation  │   │
│   └────────────────────────────────┘          └────────────────────────────────┘   │
│                    │                                           │                   │
│                    │                                           │                   │
│                    │                                           │                   │
│                    │                                           │                   │
│                    │                                           │                   │
│                    │    ┌────────────────────────────────┐     │                   │
│                    └───▶│    2. gaussian output file     │◀────┘                   │
│                         └────────────────────────────────┘                         │
│                                          │                                         │
│                                          │                                         │
│                                          ▼                                         │
│                         ┌────────────────────────────────┐                         │
│                         │    3. extract E and f data     │                         │
│                         └────────────────────────────────┘                         │
│                                          │                                         │
│                                          │                                         │
│                                          ▼                                         │
│                         ┌────────────────────────────────┐                         │
│                         │    calculate cross section     │                         │
│                         └────────────────────────────────┘                         │
│                                          │                                         │
│                                          │                                         │
│                                          ▼                                         │
│                         ┌────────────────────────────────┐                         │
│                         │     draw absorption figure     │                         │
│                         └────────────────────────────────┘                         │
└────────────────────────────────────────────────────────────────────────────────────┘
'''
# environment
# todo: define the path of MLatom and starting working directory 

# Typing
_Str_or_Float = Union[str, float]
_Int_or_Float = Union[int, float]

# python script path
py_script_path = ''

class Error(Exception):
    def __init__(self):
        # todo: to define various error
        pass

    def raise_Error(self, err_pmt: str, err_code: int=1):
        # todo: to raise customed error
        # todo: to invoke stopper
        pass

    def stopper(self, stop_location: str='', err_code: int=1):
        print(f'program exit at {stop_location}')
        sys.exit(err_code)
        pass


def err_print(content: str):  # need to improve
    print('--- ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR ---')
    print(content)
    print('--- ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR  ERROR ---')
    sys.stdout.flush()

def warning_print(content: str):
    print(f'!!!!! {content} !!!!!')
    sys.stdout.flush()

# todo: complete the typing of logger decorator
def logger(log_content: str, logfile: str='out.log') -> Callable[[Any], Any]:
    def logging_decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(func)
        def wrapped_function(*args: Any, **kwargs: Any) -> Any:
            log_string = '\n\n' + '=' * 90 + '\n'
            log_string += f'{log_content} ( {func.__name__} ) started at {time.ctime()} {time.tzname[0]}'
            print(log_string)
            sys.stdout.flush()

            start_time = time.time()
            result: Any = func(*args, **kwargs)
            end_time = time.time()

            spend_time = end_time - start_time
            log_string = f'{log_content} ( {func.__name__} ) finished at {time.ctime()} {time.tzname[0]} |||| total spent {spend_time:.2f} sec'
            log_string += '\n' + '=' * 90 + '\n\n'
            print(log_string)
            sys.stdout.flush()

            return result
        return wrapped_function
    return logging_decorator


class LinuxInterface():
    def __init__(self):
        pass

    # todo: to finish the bash brdige with sub_process
    @classmethod
    def exec(cls, cmd: str) -> Tuple[int, str]:
        # todo: replace `os.system` with subprocess
        return_code, output = getstatusoutput(cmd)
        return return_code, output

    @classmethod
    def cd(cls, path: str) -> int:
        try:
            os.chdir(path)
        except:
            pass
            err_code = 1
        else:
            err_code = 0
        return err_code

    @classmethod
    def rm(cls, path: str) -> int:
        err_code, _ = cls.exec(f'rm -rf {path}')
        # todo: solve with the exit code is 0
        if err_code != 0:
            pass
        return err_code

    @classmethod
    def mv(cls, from_obj: str, to: str) -> int:
        err_code, _ = cls.exec(f'mv {from_obj} {to}')
        # todo: solve with the exit code is 0
        if err_code != 0:
            pass
        return err_code

    @classmethod
    def mkdir(cls, path: str) -> int:
        err_code, _ = cls.exec(f'mkdir {path}')
        # todo: solve with the exit code is 0
        if err_code != 0:
            pass
        return err_code
    
    @classmethod
    def cp(cls, from_obj: str, to: str) -> int:
        # todo 1: need to consider the situation that copying a folder
        err_code, _ = cls.exec(f'cp {from_obj} {to}')
        # todo: solve with the exit code is not 0
        if err_code != 0:
            pass
        return err_code

    @classmethod
    def ln(cls, orig: str, link: str) -> int:
        err_code, _ = cls.exec(f'ln -s {orig} {link}')
        # todo: solve with the exit code is not 0
        if err_code != 0:
            pass
        return err_code
    
    @classmethod
    def new_dir(cls, path: str) -> int:
        if cls.exist_path(path):
            cls.rm(path)
        err_code = cls.mkdir(path)
        # todo: solve with the exit code is not 0
        if err_code != 0:
            pass
        return err_code
        
    @classmethod
    def exist_path(cls, path: str) -> bool:
        if os.path.exists(path):
            return True
        return False
    
    @classmethod
    def check_path_or_exit(cls, path: str):
        if not cls.exist_path(path):
            print(f'!!!!!"{path}" not exists, program exit!!!!!')
            exit()
    
    def write_file(self, content: str, file_name: str) -> bool:
        try:
            with open(file_name, 'w') as f:
                f.write(content)
            return True
        # todo 3: defined error on exception
        except:
            # todo 4: raise file write error
            return False
    @staticmethod
    def print_warning(content: str): 
        print('*' * 80)
        print(content)
        print('*' * 80)
        sys.stdout.flush()
    

class XYZ():
    def __init__(self, x: _Str_or_Float, y: _Str_or_Float, z: _Str_or_Float):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
    
    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        if key == 2:
            return self.z
        else:
            return 0.0

    def __sub__(self, other: 'XYZ') -> 'XYZ':
        return self.__class__(other.x - self.x, other.y - self.y, other.z - self.z) 
        # reutrn ( (self.x - other.x)**2 +
        #          (self.y - other.y)**2 +
        #          (self.z - other.z)**2 )**0.5

    def __mul__(self, other: 'XYZ') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def mod(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2)**0.5


class Atom(XYZ):
    def __init__(self, x:_Str_or_Float, y: _Str_or_Float, z: _Str_or_Float, label: str='', au_unit: bool=False):
        self.label = label
        # keep all the unit is angstrom
        if au_unit:
            self.x = self.au2ang(float(x))
            self.y = self.au2ang(float(y))
            self.z = self.au2ang(float(z))
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        # todo 3: to transfer element index into element label 

    @classmethod
    def ang2au(cls, value: float) -> float:
        return 1.8897261339 * value

    @classmethod
    def au2ang(cls, value: float) -> float:
        return 0.5291772083 * value

    def get_au_xyz(self) -> Tuple[float, float, float]:
        return self.ang2au(self.x), self.ang2au(self.y), self.ang2au(self.z)

    def get_ang_xyz(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z
    
    @classmethod
    def get_atom_instance(cls, line_str: str, au_unit: bool=False) -> 'Atom':
        label, x, y, z = line_str.split()
        # todo 2: to judge whether label is a element number, and then transfer it
        return Atom(x, y, z, label)

    def get_atom_line_expr(self):
        return f'{self.label}\t{self.x}\t{self.y}\t{self.z}\n'


class Molecule():
    def __init__(self):
        self.atoms: List['Atom'] = []

    def read_xyz(self, file_path: str, au_unit: bool=False):
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        try:
            atom_num: int = int(data[0].strip())
        except:
            # todo: to raise atom number error Exception
            atom_num = 0
            pass

        # todo: to detect whether 2nd line is empty
        for line in data[2:]:
            try:
                label, x, y, z = line.split()
                self.atoms.append(Atom(x, y, z, label))
            except:
                # todo: raise data number not meet the required format
                pass
        
        if atom_num != self.atom_num:
            # todo 2: raise xyz atom number error problem
            pass
    
    @staticmethod
    def read_element_label(input_file: str, idx: int=1) -> List[str]:
        idx -= 1
        with open(input_file) as f:
            data = f.readlines()
        return [line.split()[idx] for line in data]

    @classmethod
    def read_multi_xyz(cls, file_path: str, au_unit: bool=False) -> List['Molecule']:
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        try:
            atom_num = int(data[0].strip())
        except:
            # todo: to raise atom error Exception
            atom_num = 0
            pass
        molecules: List[Molecule] = []
        inverval = atom_num + 2
        for atom_idx in range(len(data) // inverval):
            mole = Molecule()
            for line in data[2 + atom_idx * inverval: (atom_idx + 1) * inverval]:
                atom = Atom.get_atom_instance(line)
                mole.atoms.append(atom)
            molecules.append(mole)
        return molecules
                
    def gaussian_opt_input(self, functionals: str='CAM-B3LYP', basis: str='def2tzvp', 
                           cores: int=16, memory: int=16,
                           charge: int=0, multiplicity: int=1) -> str:
        inp  = f'%nproc={cores}\n'
        inp += f'%mem={memory}gb\n'
        inp += f'# {functionals}/{basis} opt freq\n\n'
        inp += 'MLatom atuo molecule opt and freq\n\n'
        inp += f'{charge} {multiplicity}\n'
        for atom in self.atoms:
            inp += f'{atom.label}\t{atom.x}\t{atom.y}\t{atom.z}\n'
        inp += '\n'
        return inp

    def gaussian_calc_inp(self, functionals: str='CAM-B3LYP', basis: str='def2tzvp',
                           cores: int=16, memory: int=16, nStates: int=10,
                           charge: int=0, multiplicity: int=1) -> str:
        inp  = f'%nproc={cores}\n'
        inp += f'%mem={memory}gb\n'
        inp += f'# {functionals}/{basis} TD(nStates={nStates})\n\n'
        inp += 'MLatom atuo molecule TD calculation\n\n'
        inp += f'{charge} {multiplicity}\n'
        for atom in self.atoms:
            inp += f'{atom.label}\t{atom.x}\t{atom.y}\t{atom.z}\n'
        inp += '\n'
        return inp

    @property
    def atom_num(self):
        return len(self.atoms)
    
    def print_xyz(self, out_file: str):
        with open(out_file, 'w') as f:
            f.write(f'{self.atom_num}\n\n')
            for atom in self.atoms:
                f.write(f'{atom.label}\t{atom.x}\t{atom.y}\t{atom.z}\n')
    
    @staticmethod
    def multi_print_xyz(multi_xyz: List['Molecule'], out_file_path: str, print_num: int=0):
        with open(out_file_path, 'w') as f:
            if print_num:
                for i in range(print_num):
                    mole = multi_xyz[i]
                    f.write(f'{mole.atom_num}\n\n')
                    for atom in mole.atoms:
                        f.write(f'{atom.label}\t{atom.x}\t{atom.y}\t{atom.z}\n')
            else:
                for mole in multi_xyz:
                    f.write(f'{mole.atom_num}\n\n')
                    for atom in mole.atoms:
                        f.write(f'{atom.label}\t{atom.x}\t{atom.y}\t{atom.z}\n')
            

class Plot():
    def __init__(self) -> None:
        plt.switch_backend('agg')
        self.fig, self.ax = plt.subplots()
        self.data: List[List[float]] = []
        self.xlabel: str = ''
        self.ylabel: str = ''
        self.title: str = ''
        self.grid: bool = False
        
    def set_info(self, xlabel: str='', ylabel: str='', title: str='', grid: bool=False):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.grid = grid

    def plot_file(self, file: str, out_img: str='fig.png', x_column: int=1, y_column: int=2):
        if not EnvCheck.check_matplotlib():
            return
        self.read_data(file)
        x = self.column_data(x_column)
        y = self.column_data(y_column)
        self.ax.plot(x, y)
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        if self.grid:
            self.ax.grid()
        self.fig.savefig(out_img)
    
    def plot_multiple_file(self, files: List[str], labels: List[str], colors: Optional[List[str]]=None, x_column: int=1, y_column: int=2):
        if not EnvCheck.check_matplotlib():
            return
        plt.rcParams['legend.fontsize'] = 'x-large'
        plt.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.16, bottom=0.15, top=0.95, right=0.95)
        # plt.legend(frameon=False)
        # plt.rcParams['xtick.labelsize'] = 'x-large'
        # plt.rcParams['ytick.labelsize'] = 'x-large'
        # plt.rcParams['axes.labelsize'] = 'x-large'
        # plt.rcParams['axes.titlesize'] = 'x-large'
        for idx, file in enumerate(files):
            self.read_data(file)
            x = self.column_data(x_column)
            y = self.column_data(y_column)
            if idx == 0:
                if colors:
                    self.ax.plot(x, y, colors[idx], label=labels[idx], linewidth=3)
                else:    
                    self.ax.plot(x, y, label=labels[idx], linewidth=3)
            else:
                if colors:
                    self.ax.plot(x, y, colors[idx], label=labels[idx])
                else:    
                    self.ax.plot(x, y, label=labels[idx])
        self.ax.legend(frameon=False)
        # self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        self.ax.set_xlabel(self.xlabel, fontsize=18)
        self.ax.set_ylabel(self.ylabel, fontsize=18)
        # if self.grid:
        #     self.ax.grid()
        # self.fig.savefig(out_img_name)
    
    def plot_spc_bar(self, x_data: List[float], y_data: List[float], width: float=0.025, color: str='g') -> None:
        self.ax.bar(x_data, y_data, width=width, color=color)
        
    def save_fig(self, fig_name: str):
        if self.grid:
            self.ax.grid()
        self.fig.savefig(fig_name)

    def column_data(self, i: int) -> List[float]:
        result = []
        for line in self.data:
            result.append(line[i - 1])
        return result

    def read_data(self, file: str, drop_1st_line: bool=True, sep: str=''):
        with open(file) as f:
            data = f.read().splitlines()
        data_list = []
        if drop_1st_line:
            data = data[1:]
        for line in data:
            if sep.strip():
                data_list.append(self.trans_to_float(line.split(sep)))
            else:
                data_list.append(self.trans_to_float(line.split()))
        self.data = data_list
    
    @staticmethod
    def trans_to_float(inp: List[str]) -> List[float]:
        return [ float(x) for x in inp ]
                

class GaussCalc(LinuxInterface):
    def __init__(self, program_path: str, work_path: str):
        self.program_path = program_path
        self.work_path = work_path
        self.temp_path = os.path.join(work_path, 'TEMP')
        self.gauss_path = os.path.join(work_path, 'GAUSS')
        self.all_com_path = os.path.join(self.gauss_path, 'all_com')
        self.all_log_path = os.path.join(self.gauss_path, 'all_log')
        self.gauss_data_path = os.path.join(self.gauss_path, 'data')
        self.path_check()
        # todo 5: check gaussian environment
        self.calc_flag = False
        # extract xyz file needed data
        self.element = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus',' Uuo']
        self.incre = 5
        self.sep = '-' * 69
        self.colums = [1, 3, 4, 5]
        self.header = ['Standard orient', 'Z-matrix orient', 'Input orient']
        self.inp_pre_xyz: str = ''
        self.inp_after_xyz: str = ''
        self.gauss: str = EnvCheck.check_gaussian_version()

    def path_check(self):
        # check the TEMP directory is new
        self.cd(self.work_path)
        if self.exist_path(self.temp_path):
            self.__class__.rm(self.temp_path)
        self.__class__.mkdir(self.temp_path)
        # check the GAUSS directory is new
        if self.exist_path(self.gauss_path):
            self.__class__.rm(self.gauss_path)
        self.__class__.mkdir(self.gauss_path)
        self.__class__.mkdir(self.all_com_path)
        self.__class__.mkdir(self.all_log_path)
        self.__class__.mkdir(self.gauss_data_path)

    @logger('optimize and calc molecule freq')
    def gaussain_opt(self, molecule: Molecule, functionals: str='CAM-B3LYP', basis: str='def2tzvp', 
                     cores: int=16, memory: int=16,
                     charge: int=0, multiplicity: int=1):
        # we assume that the current working directory is at $root/TEMP
        self.__class__.cd(self.temp_path)
        inp = molecule.gaussian_opt_input(functionals, basis, cores, memory, charge, multiplicity)
        with open(os.path.join(self.temp_path, 'opt.com'), 'w') as f:
            f.write(inp)
        self.gaussian_opt_calc()

    def gaussian_opt_calc(self, file_name: str='opt.com'):
        if not file_name.endswith('.com'):
            err_print('gauss input file name not ends with .com')
            exit()
        self.cp(file_name, self.temp_path)
        cwd = os.getcwd()
        self.cd(self.temp_path)
        if self.gauss == '':
            cls.print_warning('Gaussian 09 or Gaussian 16 not exists!!!!! If you use old version of Gaussian ,please update it!!!!!')
            exit()
        err_code, out = self.__class__.exec(f'{self.gauss}  {file_name}')
        if err_code != 0 or out != '':
            err_print(f'Gaussian calculation failed!!!\nstatus code: {err_code} (0 stands for normal exit, none 0 stands for error)\noutput: {out}')
        gauss_name_pre = file_name[:-3]
        self.mv(f'{gauss_name_pre}*', self.gauss_path)
        self.cd(cwd)

    def gaussain_TD_with_para(self, idx: int, molecule: Molecule, functionals: str='CAM-B3LYP', basis: str='def2svp', 
                     cores: int=16, memory: int=16, nStates: int=10,
                     charge: int=0, multiplicity: int=1):
        # we assume that the current working directory is at $root/TEMP
        inp = molecule.gaussian_calc_inp(functionals, basis, cores, memory, nStates, charge, multiplicity)
        self.gaussain_TD_calc(idx, inp)

    def gaussian_TD_calc_with_template(self, idx: int, molecule: Molecule):
        inp = self.inp_pre_xyz  # inp_pre_xyz has included an '\n'
        for atom in molecule.atoms:
            inp += atom.get_atom_line_expr()
        inp += self.inp_after_xyz + '\n'
        self.gaussain_TD_calc(idx, inp)
        
    def gaussain_TD_calc(self, idx: int, inp: str):
        orig_path = os.getcwd()
        self.cd(self.temp_path)
        if self.gauss == '':
            cls.print_warning('Gaussian 09 or Gaussian 16 not exists!!!!! If you use old version of Gaussian ,please update it!!!!!')
            exit()
        with open(os.path.join(self.temp_path, f'{idx}.com'), 'w') as f:
            f.write(inp)
        err_code, out = self.__class__.exec(f'{self.gauss} {idx}.com')
        if err_code != 0 or out != '':
            err_print(f'Gaussian calculation failed!!!\nstatus code: {err_code} (0 stands for normal exit, none 0 stands for error)\noutput: {out}')
        self.__class__.mv(f'{idx}.com', self.all_com_path)
        self.__class__.mv(f'{idx}.log', self.all_log_path)
        self.cd(orig_path)

    @staticmethod
    def check_opt_freq_inp(path: str) -> bool:
        with open(path) as f:
            data = f.read().splitlines()
        for line in data:
            parts = re.split('[ /]', line.strip())
            try:
                if parts[0][0] == '#':
                    for each in parts:
                        if each.lower() == 'freq':
                            return True
            except:
                pass
        return False

    @staticmethod
    def check_TD_inp(path: str) -> bool:
        with open(path) as f:
            data = f.read().splitlines()
        for line in data:
            parts = re.split('[ /]', line.strip())
            try:
                if parts[0][0] == '#':
                    for each in parts:
                        if each.lower() == 'td':
                            return True
            except:
                pass
        return False
    
    def parse_gauss_inp_file(self, path: str) -> bool:
        def check_charge_multi(line: str):
            splited = line.strip().split()
            if len(splited) == 2:
                try:
                    charge, multiplicity = int(splited[0]), int(splited[1])
                    return True
                except:
                    pass
            return False
        
        def check_geom(line: str):
            splited = line.strip().split()
            element = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus',' Uuo']
            ele_little = list(map(lambda x: x.lower(), element))
            if len(splited) == 4:
                try:
                    label, x, y, z = splited
                    x = float(x); y = float(y); z=float(z)
                    if label.lower() in ele_little:
                        return True
                    label_num = int(label)
                    if label_num < 100:
                        return True
                except:
                    pass
            return False

        with open(path) as f:
            data = f.read().splitlines()
        flag_charge_multi = False
        # flag_xyz_finished = False
        pre_xyz = ''
        # after_xyz = ''
        idx_not_geom = 0
        for idx, line in enumerate(data):
            if flag_charge_multi and not check_geom(line):
                # flag_xyz_finished = True
                idx_not_geom = idx
                # after_xyz += line + '\n'
                break
            if check_charge_multi(line) and check_geom(data[idx + 1]):
                flag_charge_multi = True
                pre_xyz += line + '\n'
                continue
            # if flag_xyz_finished:
            #     after_xyz += line + '\n'
            # if not flag_charge_multi and not flag_xyz_finished:
            if not flag_charge_multi:
                pre_xyz += line + '\n'
        if flag_charge_multi:
            self.inp_pre_xyz = pre_xyz
            self.inp_after_xyz = '\n'.join(data[idx_not_geom:])
        return flag_charge_multi
    
    def extract_xyz(self, log_file: str) -> 'Molecule':
        idx = 0
        start = 0
        end = 0
        with open(log_file) as f:
            geom = []
            for line in f:
                idx += 1
                # if line.strip() in header:
                if reduce(lambda x, y: x or y, map(lambda x: x in line.strip(), self.header)):
                    start = idx + self.incre
                    end = start
                    geom = []
                if start <= idx <= end:
                    if line.strip() == self.sep:
                        end =  idx - 1
                    else:
                        end += 1
                        splited = line.strip().split()
                        geom.append([ splited[x] for x in self.colums ])
        molecule = Molecule()
        if len(geom) == 0:
            # todo 3: raise error about extracted xyz file atom number is wrong
            pass
        else:
            for atom in geom:
                # todo 1: to check whether atom[0] is not a int type var
                # todo 2: to check whether atom[1, 2, 3] is not a float type var
                label = self.element[int(atom[0])]
                x, y, z = list(map(float, atom[1:]))
                molecule.atoms.append(Atom(x, y, z, label))
        return molecule
    
    def process_gaussian_log(self, folder: str, n_state: int=10, pattern: str=r'[0-9]+?\.log', regex: bool=False):
        orig_folder = os.getcwd()
        self.cd(folder)
        if pattern == "":
            empty = True
        else:
            empty = False
        if empty:
            grep_str = "''"
        else:
            if regex:
                # use grep -E
                grep_str = f"-E '{pattern}'"
                pass
            else:
                grep_str = f"'{pattern}'"
                # using common grep command
                pass
        cmd = '''rm %s/E*.dat %s/f*.dat                     # gauss_data_path
                 for file in `ls | grep %s | sort -n`; do   # grep_str
                 for i in `seq 1 %d`; do                    # number of excited state
                     grep 'Excited State' $file | grep ' '$i':' | awk '{print $5}' >> %s/'E'$i.dat
                     grep 'Excited State' $file | grep ' '$i':' | awk '{print $9}' | sed s/f=//g >> %s/'f'$i.dat
                 done
                 done''' % (self.gauss_data_path, self.gauss_data_path, grep_str, n_state, self.gauss_data_path, self.gauss_data_path)
        err_code, _ = self.__class__.exec(cmd)   # todo 3: raise error on fail
        # todo 3: to check whether all the E*.inp & f*.inp has the same number
        if err_code != 0:
            err_print(f'post gaussian data process failed with error code {err_code}')
            exit()
        self.cd(orig_folder)
    
    def process_top_n_E_f(self, folder: str, max_state: int, line_num: int):
        orig_folder = os.getcwd()
        self.cd(folder)
        cmd = '''rm %s/E*.dat %s/f*.dat                        # gauss_data_path
                 for file in `ls | sort -n | head -n %d`; do   # get top n line
                 for i in `seq 1 %d`; do                       # number of excited state
                     grep 'Excited State' $file | grep ' '$i':' | awk '{print $5}' >> %s/'E'$i.dat
                     grep 'Excited State' $file | grep ' '$i':' | awk '{print $9}' | sed s/f=//g >> %s/'f'$i.dat
                 done
                 done''' % (self.gauss_data_path, self.gauss_data_path,
                            line_num, max_state,
                            self.gauss_data_path, self.gauss_data_path)
        err_code, _ = self.exec(cmd)
        if err_code != 0:  # todo 3: raise an error
            err_print(f'pre-calculated gaussian log file process failed with error code {err_code}')
            exit()
        self.cd(orig_folder)

    def process_one_gauss_log(self, log_file: str) -> Tuple[List[float], List[float]]:
        cmd = '''grep -E 'Excited State +?[0-9]+' %s | awk '{print $5}' > %s/SPC_E.TEMP''' % (log_file, self.temp_path)
        err_code, _ = self.exec(cmd)
        E_file = os.path.join(self.temp_path, 'SPC_E.TEMP')
        if err_code != 0:
            err_print(f'extract E value from one gaussian log failed with error code: {err_code}')

        cmd = '''grep -E 'Excited State +?[0-9]+' %s | awk '{print $9}' | sed s/f=//g > %s/SPC_f.TEMP''' % (log_file, self.temp_path)
        err_code, _ = self.exec(cmd)
        f_file = os.path.join(self.temp_path, 'SPC_f.TEMP')
        if err_code != 0:
            err_print(f'extract f value from one gaussian log failed with error code: {err_code}')

        with open(E_file) as f:
            E_list = list(map(float, f.read().splitlines()))
        with open(f_file) as f:
            f_list = list(map(float, f.read().splitlines()))
        self.rm(E_file)
        self.rm(f_file)
        return E_list, f_list
        

class NewtonXCalc(LinuxInterface):
    au = 0.5291772083
    def __init__(self, work_path: str) -> None:
        self.work_path = work_path
        self.temp_path = os.path.join(work_path, 'TEMP')
        self.NX_path = os.path.join(self.work_path, 'NX')
        self.path_check()
        self.calc_flag = False
        self.multi_xyz: List[Molecule] = []
        # todo 4: check NX environment
        pass

    def path_check(self) -> None:
        if self.exist_path(self.temp_path):
            self.__class__.rm(self.temp_path)
        self.__class__.mkdir(self.temp_path)
        if self.exist_path(self.NX_path):
            self.__class__.rm(self.NX_path)
        self.__class__.mkdir(self.NX_path)

    @logger('start NX calculation to get nuclear ensemble')
    def invoke_NX(self, geom_path: str, freq_path: str, atom_num: int, NEA_point: int=500) -> None:
        self.cd(self.NX_path)
        self.change_geom(geom_path)
        self.run_dynamic(freq_path, atom_num, NEA_point)
        eq_mole = Molecule()
        eq_mole.read_xyz(geom_path)
        self.multi_xyz.append(eq_mole)
        self.restore_multi_xyz(atom_num)

    def change_geom(self, geom_path: str) -> None:
        # check whether trnasfer success (delete /dev/null and then check the output log)
        self.exec(f'$NX/xyz2nx < {geom_path} &> /dev/null')
        
    def run_dynamic_old(self, freq_file: str, atom_num: int, NEA_point: int) -> None:
        self.mv(freq_file, os.path.join(self.NX_path, 'freq.out') )
        nx_inp = '&dat\n nact = 2\n numat = %d\n npoints = %d\n file_geom = geom\n iprog = 4\n' % (atom_num, NEA_point) + \
                 ' file_nmodes = freq.out\n anh_f = 1\n temp = 0.0\n ics_flg = n\n chk_e = 0\n' + \
                 ' file_out = ini_qv\n file_vib = qvector\n' + \
                 ' nis = 1\n nfs = 2\n kvert = 1\n de = 100\n prog = 6.5\n iseed = 0\n lvprt = 1\n /' 
        with open('initqp_input', 'w') as f:
            f.write(nx_inp)
        err_code, out = self.exec(f'$NX/initcond.pl > NX-MD.log 2>&1')
        if err_code != 0 or out != '':
            err_print(f'Newton-X calculation failed!!!\nstatus code: {err_code} (0 stands for normal exit, none 0 stands for error)\noutput: {out}')
        # todo 3: remove NX temp folder and other uncessary folder
    
    def run_dynamic(self, freq_file: str, atom_num: int, NEA_point: int) -> None:
        inp_name = 'initqp_input'
        self.mv(freq_file, os.path.join(self.NX_path, 'freq.out') )
        nx_inp = '&dat\n nact = 2\n numat = %d\n npoints = %d\n file_geom = "geom"\n iprog = 4\n' % (atom_num, NEA_point) + \
                 ' file_nmodes = "freq.out"\n anh_f = 1\n temp = 0.0\n ics_flg = "n"\n chk_e = 0\n' + \
                 ' file_out = "ini_qv"\n file_vib = "qvector"\n' + \
                 ' nis = 1\n nfs = 2\n kvert = 1\n de = 100\n prog = 6.5\n iseed = 1234\n lvprt = 1\n/\n'
        with open(inp_name, 'w') as f:
            f.write(nx_inp)
        os.system('$NX/mk_qvector < initqp_input')
        os.system('$NX/weight     < initqp_input > weight.log')
        os.system('$NX/initqp     < initqp_input > initqp.log')
    
    def restore_multi_xyz(self, atom_num: int):
        with open('ini_qv') as f:
            data = f.readlines()
        elem = Molecule.read_element_label('geom')
        cnt = 0
        mole = Molecule()
        for line in data[1:]:
            tmp = line.split()
            if len(tmp) == 2:
                # if cnt:
                self.multi_xyz.append(mole)
                mole = Molecule()
                cnt = 0
            elif len(tmp) == 6:
                x, y, z = self._process_ini_qv_line(tmp)
                atom = Atom(x, y, z, elem[cnt])
                cnt += 1
                mole.atoms.append(atom)
            else:
                err_print('processing ini_qv failed! exit!')
                exit()
        self.multi_xyz.append(mole)
    
    def _process_ini_qv_line(self, splited_line: List[str]) -> List[float]:
        xyz = map(lambda x: float(x.replace('D', 'e')), splited_line[0: 3])
        return list(map(lambda x: self.au * x, xyz))
    
    def restore_multi_xyz_old(self, atom_num: int):
        # working directory: In self.NX_path
        awk_inp = '''atom_num=%d \n''' % (atom_num) + \
                 r'''sed -n /Geometry/,/Velocity\ in\ NX/p final_output | sed  /Velocity/d | awk '{printf ("%s\t", $1)}''' + \
                 r''' {printf ("%.8f\t", $3*0.5291772083)} {printf ("%.8f\t", $4*0.5291772083)} {printf ("%.8f\n", $5*0.5291772083)}' ''' + \
                 r''' | sed -r s/Geometry.+/$atom_num\\n/g > xyz.dat'''
        err_code, out = self.exec(awk_inp)
        # todo 4: even it execute fail, it will also returns 0, so we should judge it throught the output form (err_code, output)
        if err_code != 0 or out != '':
            err_print(f'awk process for multiple xyz file (xyz.dat) failed!!!\nstatus code: {err_code} (0 stands for normal exit, none 0 stands for error)\noutput: {out}')
        # self.multi_xyz = Molecule.read_multi_xyz(os.path.join(self.work_path, 'xyz.dat'))
        self.multi_xyz = Molecule.read_multi_xyz('xyz.dat')
    
    def clean_file(self):
        pass

    def NX_interface(self) -> int:
        err_code = 0
        return err_code
        pass


class LineDate():
    def __init__(self, string: str) -> None:
        self.string: str = string

    def __iadd__(self, other: str) -> 'LineDate':
        return LineDate(self.string + other + '\n')

    @property
    def string_one_line(self) -> str:
        return self.string.replace('\n', ' ')


class ML_calc(LinuxInterface):

    def __init__(self, program_path:str, work_path: str) -> None:
        self.program_path: str = program_path
        self.work_path: str = work_path
        self.task_type: str = ''
        self.inp: LineDate = LineDate('')
        self.exist_python_var: bool = EnvCheck.check_python3()
        self.ML_output: str = ''
    
    def check_file_(self):
        # todo 5: to check whether itrain isubtrain ivalidate itest and *.unf *.log or other MLatom is exists
        pass

    def var_to_str(self, obj: Any) -> str:
        namespace = globals()
        return [ n for n in namespace if namespace[n] is obj ] [0]

    def add_inp(self, inp_vars: List[Any], forbidden: List[Any], key: str=''):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        all = set()
        for _, val in callers_local_vars:
            if val in forbidden:
                continue
            if val in all:
                continue
            if val not in inp_vars:
                continue
            else:
                all.add(val)
                candidates = [var_name for var_name, var_val in callers_local_vars if var_val is val]
                for var_name_out in candidates:
                    if key:
                        self.inp += f'{key}={var_name_out}'
                    else:
                        self.inp += f'{var_name_out}={val}'
                    # print(f'variable name: {var_name_out}, variable value: {val}')

    def only_one_True(self, *obj: bool) -> bool:
        cnt = 0
        for each in obj:
            if each:
                cnt += 1
        if cnt == 1:
            return True
        else:
            return False

    def set_task_type(self, estAccMLmodel: bool=False, createMLmodel: bool=False, useMLmodel: bool=False):
        # todo 2: to check whether it exists 
        check = self.only_one_True(estAccMLmodel, createMLmodel, useMLmodel)
        if not check:
            # todo 4: to raise an error: 
            pass
        if estAccMLmodel:
            self.inp += 'estAccMLmodel'
        elif createMLmodel:
            self.inp += 'createMLmodel'
        elif useMLmodel:
            self.inp += 'useMLmodel'
        else:
            # todo 3: raise multiple task error
            pass
            
    def set_index_num(self, Ntrain: _Int_or_Float=0, NSubtrain: _Int_or_Float=0, Nvalidate: _Int_or_Float=0, Ntest: _Int_or_Float=0, Nuse: int=0):
        self.add_inp([Ntrain, NSubtrain, Nvalidate, Ntest, Nuse], [0])
       #  for each in [Ntrain, NSubtrain, Nvalidate, Ntest]:
       #      if each:
       #          self.inp += f'{self.var_to_str(each)}={each}'
       #  if Nuse != 0:
       #      self.inp += f'{self.var_to_str(Nuse)}={Nuse}'

    def set_index_file(self, iTrainin: str='', iSubtrainin: str='', iValidatein: str='', iTestin: str=''):
        # todo 5: sampling=user-defined
        self.add_inp([iTrainin, iSubtrainin, iValidatein, iTestin], [''])
        # for each in [iTrainin, iSubtrainin, iValidatein, iTestin]:
        #     if each != '':
        #         self.inp += f'{self.var_to_str(each)}={each}'

    def set_data_set_inp_file(self, XfileIn: str='', Yfile: str='', XYZfile: str='', Yb: str='', Yt: str=''):
        if XfileIn != '' and XYZfile != '':
            # todo 4: raise a error: xfilein and xyzfile can not appear at the same time
            pass
        self.add_inp([XfileIn, Yfile, XYZfile, Yb, Yt], [''])
        # for each in [XfileIn, Yfile, XYZfile, Yb, Yt]:
        #     if each != '':
        #         self.inp += f'{self.var_to_str(each)}={each}'

    def set_kernel(self, kernel: str='', Gaussian: bool=False, Laplacian: bool=False, exponential: bool=False, Matern: bool=False):
        if not self.only_one_True(Gaussian, Laplacian, exponential, Matern):
            # todo 3: deal with the multiple True kernel
            pass
        self.add_inp([Gaussian, Laplacian, exponential, Matern], [False], key='kernel')
        if Matern and kernel in ['1', '2']:
            self.add_other('nn', kernel)
        if not (Gaussian or Laplacian or exponential or Matern) and kernel != '':
            self.add_other('kernel', kernel)
        # seted = False
        # for each in [Gaussian, Laplacian, exponential, Matern]:
        #     if each :
        #         self.inp += f'kernel={self.var_to_str(each)}'
        #         if Matern:
        #             if kernel in ['1', '2']:
        #                 self.inp += f'nn={kernel}'
        #         seted = True
        #         break
        # if not seted:
        #     self.inp += f'kernel={kernel}'
            
    def set_CV_pref_inp(self, iCVtestPrefIn: str='', iCVoptPrefIn: str=''):
        self.add_inp([iCVtestPrefIn, iCVoptPrefIn], [''])
        # for each in [iCVtestPrefIn, iCVoptPrefIn]: 
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'

    def set_model_inp(self, MLmodelIn: str=''):
        self.add_inp([MLmodelIn], [''])
        # for each in [MLmodelIn]:
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'

    def set_hyper_parameter(self, Lambda: _Str_or_Float=0.0, NlgLambda: int=0, lgLambdaL: int=0, lgLambdaH: int=0,
                            Sigma: _Str_or_Float=0.0, NlgSigma: int=0, lgSigmaL: int=0, lgSigmaH: int=0, lgOptDepth: int=0):
        # todo 3: if sigma or lambda not equal to opt, but type is str, to raise an error
        if (type(Lambda) == str and Lambda != 'opt') or (type(Sigma) == str and Sigma != 'opt'):
            # todo 3: raise an error about input an string to hyper parameter
            pass
        self.add_inp([Lambda, Sigma, NlgLambda, lgLambdaL, lgLambdaH, NlgSigma, lgSigmaL, lgSigmaH, lgOptDepth], [0, 0.0, ''])
        # for each in [Lambda, Sigma, NlgLambda, lgLambdaL, lgLambdaH, NlgSigma, lgSigmaL, lgSigmaH]: 
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'

    def set_index_out(self, iTrainOut: str='', iSubtrainOut: str='', iValidateOut: str='', iTestOut: str=''):
        self.add_inp([iTrainOut, iSubtrainOut, iValidateOut, iTestOut], [''])
        # for each in [iTrainOut, iSubtrainOut, iValidateOut, iTestOut]:
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'
        
    def set_est_out(self, YestFile: str='', YgradEstFile: str='', YestT: str=''):
        self.add_inp([YestFile, YgradEstFile, YestT], [''])
        # for each in [YestFile, YgradEstFile, YestT]:
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'
    
    def set_model_out(self, MLmodelOut: str=''):
        self.add_inp([MLmodelOut], [''])
        # for each in [MLmodelOut]:
        #     if each:
        #         self.inp += f'{self.var_to_str(each)}={each}'
    
    def set_sampling(self, user_defined: bool=True):
        if user_defined:
            self.inp += 'sampling=user-defined'

    def mix_single_file_data(self, est_file: str, orig_file: str) -> str:
        with open(est_file) as f:
            est = f.read().splitlines()
        with open(orig_file) as f:
            orig = f.read().splitlines()
        orig_num = len(orig)
        mixed_data = []
        for idx, e in enumerate(est):
            if idx > orig_num - 1:
                tmp = e if float(e) > 0.0 else '0.0' 
            else:  # use orig value
                tmp = orig[idx]
            mixed_data.append(tmp)
        return '\n'.join(mixed_data)
    
    def mix_multiple_file(self, max_state: int, est_path: str, orig_path: str, out_path: str) -> None:
        for i in range(1, max_state + 1):
            E_string = self.mix_single_file_data(os.path.join(est_path, f'E{i}.dat'), os.path.join(orig_path, f'E{i}.dat'))
            f_string = self.mix_single_file_data(os.path.join(est_path, f'f{i}.dat'), os.path.join(orig_path, f'f{i}.dat'))
            with open(os.path.join(out_path, f'E{i}.dat'), 'w') as f:
                f.write(E_string)
            with open(os.path.join(out_path, f'f{i}.dat'), 'w') as f:
                f.write(f_string)
        
    
    def run_ML_shell_layer(self, work_path: str=''):
        path: str = ''
        if work_path:
            path = work_path
        else:
            path = self.work_path
        if not self.exist_path(path):
            self.mkdir(path)
        self.cd(path)
        prg = os.path.join(self.program_path, 'MLatom.py')
        # todo 1: add the exception of writing error
        with open('ml.inp', 'w') as f:
            f.write(self.inp.string)
        if self.exist_python_var:
            err_code, _ = self.exec(f'$python3 {prg} ml.inp &> ml.log')
        else:
            err_code, _ = self.exec(f'{prg} ml.inp &> ml.log')
        if err_code != 0:
            err_print('ML traiing failed with exit code %d' % err_code)
            exit()
        self.inp: LineDate = LineDate('')
        # todo 3: post data process

    def run_ML(self, work_path: str=''):
        path = work_path if work_path else self.work_path
        if not self.exist_path(path):
            self.mkdir(path)
        self.cd(path)
        with open('ml.inp', 'w') as f:
            f.write(self.inp.string)
        s = StringIO()
        with redirect_stdout(s):
            prg = os.path.join(self.program_path, 'MLatomF')
            para = self.inp.string_one_line.split()
            with Popen([prg] + para, stdout=PIPE, stderr=PIPE) as proc:
                print(proc.stdout.read().decode('utf-8'))
                err = proc.stderr.read().decode('utf-8')
        self.ML_output = s.getvalue()
        with open('log', 'w') as f:
            f.write(self.ML_output)
        if err:
            warning_print(f'MLatom.run_ML returns a non-empty error: {err}, PAY ATTENTION to it.')
        self.inp: LineDate = LineDate('')

    def add_other(self, para: str, value: Any):
        self.inp += f'{para}={value}'

    def collect_validate_RMSE_shell_layer(self) -> float:
        cmd = "grep RMSE ml.log | sed -n 4p | awk '{print $3}' > RMSE.tmp"
        self.exec(cmd)
        with open('RMSE.tmp') as f:
            data = f.read().strip()
        self.rm('RMSE.tmp')
        if data == '':
            err_print('collect RMSE value failed unexpectly, exit!')
            exit()
        return float(data)
    # todo 3: transfer xyz file to x.dat file

    def collect_validate_RMSE(self) -> float:
        find_result = re.findall('RMSE = *(.+)', self.ML_output)
        self.ML_output = ''
        try:
            return float(find_result[1])
        except:
            err_print('encountered error when extract RMSE value at MLatom output')
            exit()


class CrossSection_calc(LinuxInterface):
    spec_type = 1
    init_state = 1
    final_state = [2, 3, 4]
    prob_kind = 'F'
    screen = 0
    os_condon = -1
    norm = 'local'
    seed = 0
    function = 'gauss'
    # delta = 0.01
    temp = 0
    nref = 1
    eps = 0.002
    kappa = 3
    run_IS = 0

    BK        = 0.3166813639E-5     # Boltzmann constant Hartree/kelvin
    bk_ev     = 8.617343E-5         # Boltzmann constant eV/kelvin
    proton    = 1822.888515         # (a.m.u.)/(electron mass)
    timeunit  = 24.188843265E-3     # femtoseconds
    au2ev     = 27.21138386         # au to eV
    pi        = 3.141592653589793   # pi
    au2ang    = 0.52917720859       # au to angstrom
    deg2rad   = 1.745329252E-2      # degree to radian (pi/180)
    au2cm     = 219474.625          # hartree to cm-1
    cm2au     = 4.55633539E-6       # cm-1 to hartree
    h_planck  = 1.5198298508E-16    # h hartree.s
    h_evs     = 4.13566733E-15      # h eV.s
    light     = 299792458           # speed of light m/s
    lightau   = 137.035999139       # speed of light au
    e_charge  = -1.602176487E-19    # electron charge C
    e_mass    = 9.10938215E-31      # electron mass kg
    eps0      = 8.854187817e-12     # vacuum permitivity C^2*s^2*kg^-1*m^-3
    # cross section const
    hplanck = h_evs
    ev2nm = 1E9 * hplanck * light
    gamma = 1 # temperature = 0  ==>  gamma = 1
    nref = 1 # ratio
    cs_coeff_pre = pi * e_charge**2 / (2 * e_mass * light * eps0 * nref)  # m^2/s  unit
    cs_coeff = cs_coeff_pre * hplanck / (2 * pi) * 1E20                   # Angstrom^2*eV


    def __init__(self, work_path: str):
        self.data_path: str = ''
        self.max_state: int = 0
        self.work_path: str = work_path
        self.cs_path: str = os.path.join(os.path.join(self.work_path, 'cross-section'))
        # todo 3: check whether this folder exists, and exit
        if self.exist_path(self.cs_path):
            self.rm(self.cs_path)
        self.mkdir(self.cs_path)
        self.E_data_list: List[List[float]] = []
        self.f_data_list: List[List[float]] = []
        self.delta: float = 0.01
        self._coeff_calc(self.delta)   # todo 2: can redefine the delta value in the inherited class
        self._output_name: str = ''
        self.x_min: float = 0.0
        self.x_max: float = 0.0

    def _coeff_calc(self, delta: float):
        # gauss function const
        self.coeff: float = 1 / (delta * math.sqrt(self.pi / 2))
        self.devided_num: float = - delta**2 / 2

    def _calc_cs(self, data_path: str, max_state: int, delta: float, output_name: str):
        self.data_path = data_path
        self.max_state = max_state
        self.delta = delta
        self._output_name = output_name
        self._coeff_calc(delta)
        self._read_all_data(data_path)
        self._calc_with_data()

    def _calc_at_de(self, de: float) -> Tuple[float, float, float]:
        spect = 0.0
        whole_num_point = len(self.E_data_list[0])
        for state_index, E_data in enumerate(self.E_data_list):
            f_data = self.f_data_list[state_index]
            for target_index, E in enumerate(E_data):
                f = f_data[target_index]
                spect += E * f * self._gauss(de, E)
        cross_section = self.cs_coeff * spect * self.gamma / (whole_num_point * de)
        wavelength = self.ev2nm / de
        return de, wavelength, cross_section

    def _calc_SPC(self, E_list: List[float], f_list: List[float], delta: float, output_name: str):
        self.delta = delta
        self._coeff_calc(delta)
        self.E_data_list = [[x] for x in E_list]
        self.f_data_list = [[x] for x in f_list]
        self._output_name = output_name
        self._calc_with_data()
        
    def _calc_with_data(self):
        #####   data pre-process     #####
        E_max = -10000
        E_min = 10000
        # whole_num_point = 0
        for E_data in self.E_data_list:
            tmp_max = max(E_data)
            tmp_min = min(E_data)
            E_max = tmp_max if tmp_max > E_max else E_max
            E_min = tmp_min if tmp_min < E_min else E_min
        # whole_num_point = len(self.E_data_list[0])  # todo: check the E and f is has the same number 
        range_min = E_min - self.kappa * self.delta 
        range_max = E_max + self.kappa * self.delta
        self.x_min = range_min
        self.x_max = range_max
        # de = round(range_min, 4)
        de: float = range_min
        
        ##### calc cross section #####
        file_handle = open(os.path.join(self.cs_path, self._output_name), 'w')
        file_handle.write('DE/eV    lambda/nm    sigma/A^2\n')
        de_list = []
        while de < range_max:
            de_list.append(de)
            de += self.eps
        
        whole_num_point = len(self.E_data_list[0])
        c_E_list = ((c_double * whole_num_point) * self.max_state)()
        c_f_list = ((c_double * whole_num_point) * self.max_state)()
        for state_index, E_data in enumerate(self.E_data_list):
            f_data = self.f_data_list[state_index]
            for target_index, E in enumerate(E_data):
                f = f_data[target_index]
                c_E_list[state_index][target_index] = E
                c_f_list[state_index][target_index] = f
        de_num = len(de_list)
        c_cs = (c_double * de_num)()
        c_de_list = (c_double* de_num)()
        wavelength = []
        for idx, de in enumerate(de_list):
            c_de_list[idx] = de
            wavelength.append(self.ev2nm / de)
        cs = CDLL(os.path.join(py_script_path, 'cs.so'))
        c_delta = c_double(self.delta)
        c_coeff = c_double(self.coeff)
        c_cs_coeff = c_double(self.cs_coeff)
        c_de_num = c_long(de_num)
        # print(f'coeff: {self.coeff}, divided: {self.devided_num}')
        cs.cs_calc(c_E_list, c_f_list, 
                   self.max_state, whole_num_point,
                   c_delta, c_coeff, c_de_num, c_cs_coeff,
                   c_cs, c_de_list)
        cross_section = []
        for cs_single in c_cs:
            cross_section.append(cs_single)
        for idx, de in enumerate(de_list):
            w, c = wavelength[idx], cross_section[idx]
            file_handle.write('%.4f   %.4f     %.8f\n' % (de, w, c))
        
        
        # pool = mp.get_context('spawn').Pool()
        # result_list = pool.map(self._calc_at_de, de_list)
        # pool.close()
        # pool.join()
        # for each in result_list:
        #     de, wavelength, cross_section = each
        #     file_handle.write('%.4f   %.4f     %.8f\n' % (de, wavelength, cross_section))
        file_handle.close()

    def _gauss(self, E: float, dE_0n: float) -> float:
        # coeff = 1 / (c.delta * math.sqrt(pi / 2))
        # devided_num = - c.delta**2 / 2
        to_be_exp = ((E - dE_0n) ** 2) / self.devided_num
        return self.coeff * math.exp(to_be_exp)

    def _read_all_data(self, data_path: str):
        for idx in range(1, self.max_state + 1):
            self.E_data_list.append(self._read_single_data(os.path.join(data_path, f'E{idx}.dat')))
            self.f_data_list.append(self._read_single_data(os.path.join(data_path, f'f{idx}.dat')))
  
    def _read_single_data(self, file_path: str) -> List[float]:
        with open(file_path, 'r') as f:
            tmp = f.read().splitlines()
        data = list(map(float, tmp))
        return data


class TD_cross_section(CrossSection_calc):
    def __init__(self, work_path: str) -> None:
        super().__init__(work_path)
        # self.TD_out_name = 'cross-section-TD.dat'
        # self.SPC_out_name = 'cross-section-SPC.dat'
        self.TD_out_name = 'cross-section_qc-nea.dat'
        self.SPC_out_name = 'cross-section_spc.dat'

    @logger('start calc QC cross section')
    def calc(self, data_path: str, max_state: int, delta: float=0.05, output_name: str='cross-section-TD.dat'):
        super()._calc_cs(data_path, max_state, delta, output_name=output_name)
        # self.cp(os.path.join(self.cs_path, 'cross-section-TD.dat'), self.work_path)
    
    def calc_SPC(self, E_list: List[float], f_list: List[float], delta: float, out_name: str):
        super()._calc_SPC(E_list, f_list, delta, out_name)
        # self.cp(os.path.join(self.cs_path, self.SPC_out_name), self.work_path)


class ML_cross_section(CrossSection_calc, ML_calc):
    def __init__(self, work_path: str, program_path: str) -> None:
        # super().__init__(work_path)
        CrossSection_calc.__init__(self, work_path)
        ML_calc.__init__(self, program_path, work_path)
        self.ML_path = os.path.join(self.work_path, 'ML')
        self.new_dir(self.ML_path)
        # if not self.exist_path(self.ML_path):
        #     self.mkdir(self.ML_path)
        self.crrent_path: str = ''
        self.data_path: str = ''
        self.max_state: int = 0
        self.TD_point: int = 0
        self.xyz_file: str = ''
        self.this: CtlPartAdapter = None
        self.use_model_pred_path: str = os.path.join(self.ML_path, 'useModel')
        self.new_dir(self.use_model_pred_path)
        self.pred_data_path: str = os.path.join(self.ML_path, 'pred_data')
        self.new_dir(self.pred_data_path)
        self.mix_data_path: str = os.path.join(self.ML_path, 'mix_data')
        self.new_dir(self.mix_data_path)

        self.train_name: str = 'itrain.dat'
        self.sub_name: str = 'isubtrain.dat'
        self.val_name: str = 'ivalidate.dat'
        self.eq_name: str = 'eq.xyz'
        self.xyz_name: str = 'xyz.dat'
        self.ML_out_name: str = 'cross-section_ml-nea.dat' # self.output_file_name: str = 'cross-section-ML.dat'

    def define_data_set(self, this: 'CtlPartAdapter', TD_point: int, xyz_file: str, max_state: int, delta: float=0.01):
        self.TD_point = TD_point
        self.xyz_file = xyz_file
        self.max_state = max_state
        self.delta = delta
        self.this = this
        # to separate each 

    # @logger('iteratively calc TD with gaussian')    
    def _iter_calc_QC(self, start: int, end: int):
        calc, start, end = self.this.recheck_QC_calc(start, end)
        if calc:
            for i in range(start, end):
                if not EnvCheck.check_gaussian():
                    exit()
                if self.this.gauss_calc_inp_exist:
                    self.this.gauss.gaussian_TD_calc_with_template(i, self.this.multi_xyz[i])
                else:
                    self.this.gauss.gaussain_TD_with_para(idx=i, molecule=self.this.multi_xyz[i], functionals=self.this.functionals,
                                                basis=self.this.basis, nStates=self.this.N_max_state)
            self.this.gauss.process_gaussian_log(self.this.gauss.all_log_path, self.this.N_max_state, pattern='', regex=False)
        else:
            self.this.gauss.process_top_n_E_f(self.this.gauss.all_log_path, self.this.N_max_state, end)

    @logger('use all QC points to run ML-NEA calculations')
    def ML_train_all(self):
        # todo 3: to to record log
        self._iter_calc_QC(0, self.TD_point)
        rmse = self._train_epoch(self.TD_point)
        print(f'\nRMSE_geom value for {self.TD_point} point: {rmse}')
        sys.stdout.flush()
        self._train_final(self.TD_point)
        self._collect_ML_result(self.use_model_pred_path)
        self.mix_multiple_file(self.max_state, self.pred_data_path, self.this.gauss.gauss_data_path, self.mix_data_path)
        self._calc_cs(self.mix_data_path, self.max_state, self.delta, self.ML_out_name)
        # self.cp(os.path.join(self.cs_path, 'cross-section-ML.dat'), self.work_path)

    @logger('run ML-NEA iteratively for spectrum generation')
    def ML_train_iter(self, start_point: int=50, incr_point: int=50, threshold: float=0.05):
        self.cd(self.ML_path)
        current_point = start_point
        self._iter_calc_QC(0, current_point)        # the 0th is the equilibrium geom, calc ( 0 - 49 )  all 50 point
        prev_rmse = 0.0
        iter_cnt = 1
        # attention: In fact, we have TD_point number: (self.TD_point + 1)
        while current_point < self.TD_point + 1:
            rmse = self._train_epoch(current_point)
            rRMSE = (rmse - prev_rmse) / rmse
            print(f'ML-NEA iteration {iter_cnt}: train_number = {current_point}; RMSE_geom = {rmse}; rRMSE = {rRMSE}\n')
            sys.stdout.flush()
            iter_cnt += 1
            # todo 4: print log about the RMSE variation
            if abs(rRMSE) <= threshold:
                # calc cross section
                print(f'ML-NEA iteration ended after {iter_cnt - 1} iteration!')
                sys.stdout.flush()
                # todo 5: extract all data and mix data
                self._train_final(current_point)
                self._collect_ML_result(self.use_model_pred_path)
                self.mix_multiple_file(self.max_state, self.pred_data_path, self.this.gauss.gauss_data_path, self.mix_data_path)
                self._calc_cs(self.mix_data_path, self.max_state, self.delta, self.ML_out_name)
                break
            else:
                if current_point + incr_point >= self.TD_point + 1:
                    self._iter_calc_QC(current_point, self.TD_point + 1)
                    current_point = self.TD_point + 1
                    print(f'max training number exceeded, use all point for ML-NEA training\n')
                    rmse = self._train_epoch(current_point)
                    print(f'ML-NEA iteration {iter_cnt}: train_number = {current_point}; RMSE_geom = {rmse}; rRMSE = {rRMSE}\n')
                    sys.stdout.flush()
                    self._train_final(current_point)
                    self._collect_ML_result(self.use_model_pred_path)
                    self.mix_multiple_file(self.max_state, self.pred_data_path, self.this.gauss.gauss_data_path, self.mix_data_path)
                    self._calc_cs(self.mix_data_path, self.max_state, self.delta, self.ML_out_name)
                else:
                    self._iter_calc_QC(current_point, current_point + incr_point)
                    current_point += incr_point
                    prev_rmse = rmse
        # self.cp(os.path.join(self.cs_path, 'cross-section-ML.dat'), self.work_path)
    
    def _train_epoch(self, train_num: int) -> float:
        self.cd(self.ML_path)
        train_num_path = os.path.join(self.ML_path, str(train_num))
        if self.exist_path(train_num_path):
            self.rm(train_num_path)
        self.mkdir(train_num_path); self.cd(train_num_path)
        Molecule.multi_print_xyz(self.this.multi_xyz, self.xyz_name, train_num)
        _, sub, val = self._create_idx_file(train_num)
        rmse_list = []
        for task_sym in ['E', 'f']:
            task_path = os.path.join(train_num_path, task_sym)
            self.new_dir(task_path)
            for state in range(1, self.max_state + 1):
                sym_state = task_sym + str(state)
                self.cd(task_path); current_dir = os.path.join(task_path, str(state)); self.mkdir(current_dir); self.cd(current_dir)
                self.ln(os.path.join(train_num_path, self.xyz_name), self.xyz_name)
                self.ln(os.path.join(self.this.gauss.gauss_data_path, f'{sym_state}.dat'), f'{sym_state}')
                self.ln(os.path.join(train_num_path, self.train_name), self.train_name)
                self.ln(os.path.join(train_num_path, self.sub_name), self.sub_name)
                self.ln(os.path.join(train_num_path, self.val_name), self.val_name)
                self.ln(os.path.join(self.ML_path, self.eq_name), self.eq_name)
                
                self.set_task_type(createMLmodel=True)
                self.set_data_set_inp_file(XYZfile='xyz.dat', Yfile=f'{sym_state}')
                self.set_est_out(YestFile=f'{sym_state}.est')
                self.set_kernel(Gaussian=True)
                self.set_model_out(MLmodelOut=f'{sym_state}.unf')
                self.add_other('molDescriptor', 'RE'); self.add_other('molDescrType', 'unsorted')
                self.set_hyper_parameter(Lambda='opt', NlgLambda=11, lgLambdaL=-25, lgLambdaH=-1, Sigma='opt', lgSigmaL=-10, lgSigmaH=9, NlgSigma=11, lgOptDepth=2)
                self.add_other('minimizeError', 'RMSE')
                self.set_sampling(user_defined=True)
                self.set_index_num(Ntrain=train_num, NSubtrain=sub, Nvalidate=val)
                self.set_index_file(iTrainin=self.train_name, iSubtrainin=self.sub_name, iValidatein=self.val_name)
                self.run_ML(current_dir)
                rmse_value = self.collect_validate_RMSE()
                if rmse_value != 0.0:
                    rmse_list.append(rmse_value)
        return (reduce(lambda x, y: x * y, rmse_list)) ** (1 / len(rmse_list))
    
    def _train_final(self, train_num: int):
        self.cd(self.use_model_pred_path)
        Molecule.multi_print_xyz(self.this.multi_xyz, os.path.join(self.ML_path, self.xyz_name))
        for sym, state in product(['E', 'f'], range(1, self.max_state + 1)):
            self.cd(self.use_model_pred_path)
            sym_state = sym + str(state)
            self.mkdir(sym); self.cd(sym); self.mkdir(str(state)); self.cd(str(state))
            self.ln(os.path.join(self.ML_path, self.xyz_name), self.xyz_name)
            self.ln(os.path.join(self.ML_path, self.eq_name), self.eq_name)
            self.ln(os.path.join(self.ML_path, str(train_num), sym, str(state), f'{sym_state}.unf'), f'{sym_state}.unf')

            self.set_task_type(useMLmodel=True)
            self.set_data_set_inp_file(XYZfile=self.xyz_name)
            self.set_est_out(YestFile=f'{sym_state}.est')
            self.set_kernel(Gaussian=True)
            self.set_model_inp(MLmodelIn=f'{sym_state}.unf')
            self.add_other('molDescriptor', 'RE'); self.add_other('molDescrType', 'unsorted')
            self.run_ML(os.getcwd())

    def _collect_ML_result(self, final_path: str):
        for sym, state in product(['E', 'f'], range(1, self.max_state + 1)):
            sym_state = sym + str(state)
            self.ln(os.path.join(final_path, sym, str(state), f'{sym_state}.est'), os.path.join(self.pred_data_path, f'{sym_state}.dat'))

    def _create_idx_file(self, train_num: int, train_ratio: float=0.8) -> Tuple[int, int, int]:
        sub = int(train_num * train_ratio)
        val = train_num - sub
        self._write_seq(train_num, self.train_name)
        self._write_seq(sub, self.sub_name)
        self._write_seq(start=sub + 1, end=train_num, name=self.val_name)
        return train_num, sub, val
    
    def _write_seq(self, end: int, name: str, start: int=1):
        with open(name, 'w') as f:
            for i in range(start, end + 1):
                f.write(str(i) + '\n')


class SPC_cross_section():
    h_evs = 4.13566733E-15      # h eV.s
    light = 299792458           # speed of light m/s
    ev2nm = 1E9 * h_evs * light
    SPC_out_name = 'cross-section_spc.dat'
    def __init__(self) -> None:
        self.Es: List[float] = []
        self.fs: List[float] = []
        self.x_min: float = 0.0
        self.x_max: float = 0.0
        self.delta: float = 0.0
    
    def read_data_from_file(self, file_path: str) -> bool:
        try:
            with open(file_path) as f:
                data = f.read()
            for line in data:
                delta_E, f_osci = line.split()
                delta_E = float(delta_E)
                f_osci = float(f_osci)
                self.Es.append(delta_E)
                self.fs.append(f_osci)
        except:
            return False
        return True
    
    def read_data_from_para(self, E_list: List[float], f_list: List[float]):
        self.Es = E_list
        self.fs = f_list
        
    def set_para(self, delta: float, x_min: float, x_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.delta = delta
        
    @staticmethod
    def arange(start: float, end: float, step: float) -> Iterable[float]:
        if end <= start:
            yield start
        else:
            current = start
            while current <= end:
                yield current
                current += step
    
    @staticmethod
    def SPCfunc(delta: float, peak_position: float, peak_height: float, x: float):
        # delta: broadening parameter; ff: oscillator strength; EE: current inputed E to calculate SPC
        nn = 1.0
        E_deviation = 0.0
        result = ( 0.619 * nn * peak_height / delta
          *   math.exp( -(x - peak_position + E_deviation) ** 2 / (delta ** 2) ) )
        return result
    
    def calc(self, out_name: str):
        step = 0.002
        try:
            with open(out_name, 'w') as f:
                f.write('DE/eV    lambda/nm    sigma/A^2\n')
                for x in self.arange(self.x_min, self.x_max, step):
                    sigma: float = 0.0
                    for idx, peak_position in enumerate(self.Es):
                        peak_height = self.fs[idx]
                        sigma += self.SPCfunc(self.delta, peak_position, peak_height, x)
                    wavelength = self.ev2nm / x
                    # f.write(f'{x}\t{wavelength}\t{sigma}\n')
                    f.write('%.4f   %.4f     %.8f\n' % (x, wavelength, sigma))
        except:
            err_print('SPC-cross section calculate fail')
        

class EnvCheck(LinuxInterface):
    @classmethod
    def check_NX(cls) -> bool:
        _, output = cls.exec('echo $NX')
        if output.strip() != '':
            return True
        cls.print_warning('$NX variable not set!!!!!')
        return False

    @classmethod
    def check_mlatom(cls) -> bool:
        if cls.exist_path('MLatom.py'):
            return True
        cls.print_warning('ML-NEA.py is not at the same directory with MLatom.py!!!!!')
        return False

    @classmethod
    def check_gaussian(cls) -> bool:
        _, output = cls.exec(f'export | grep GAUSS_EXEDIR')
        if output.strip() != '':
            return True
        cls.print_warning('Gaissian software not exists or not be source, $GAUSS_EXEDIR variable not exists!!!!!')
        return False

    @classmethod
    def check_gaussian_version(cls) -> str:
        # attention: use which output may be empty or " /usr/bin/which: no *** in "
        _, output = cls.exec('export | grep GAUSS_EXEDIR | grep g16')
        if output.strip() != '':
            return 'g16'
        _, output = cls.exec('export | grep GAUSS_EXEDIR | grep g09')
        if output.strip() != '':
            return 'g09'
        # cls.print_warning('Gaussian 09 or Gaussian 16 not exists!!!!! If you use old version of Gaussian ,please update it!!!!!')
        # exit()
        return ''
        # todo 3: raise an error that no Gaussian exists

    @classmethod
    def check_python3(cls) -> bool:
        _, out = cls.exec('echo $python3')
        if out.strip() != '':
            return True
        # print('$python3 variable not found, be attention, it may run into error at some situation!!!')
        return False
    
    @staticmethod
    def check_matplotlib() -> bool:
        if dir('plt'):
            return True
        return False


class Dialog(LinuxInterface):
    def __init__(self):
        self.forbidden_list: List[str] = []
        pass

    def _judge_inp(self, inp: str) -> Tuple[type, Any]:
        if inp.strip().lower() == 'true':
            return bool, True
        elif inp.strip().lower() == 'false':
            return bool, False
        else:
            try:
                tmp = int(inp)
            except:
                try:
                    tmp = float(inp)
                except:
                    return str, inp
                else:
                    return float, tmp
            else:
                return int, tmp

    def set_forbidden(self, forbidden_list: List[str]):
        self.forbidden_listj = forbidden_list

    class Decorator():
        @staticmethod
        def defer_print(content: str) -> Callable[[Any], Any]:
            def defer_decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
                # @wraps(func)
                def defer_wrap(self: object, *args: Any, **kwargs: Any) -> Any:
                    result = func(self, *args, **kwargs)
                    print(content)
                    return result
                return defer_wrap
            return defer_decorator

    @Decorator.defer_print('-' * 80)
    def ask(self, idx: float, prompt: str,
                          description: str, default: str, require_type: type,
                          upper_lmt: _Int_or_Float=0, lower_lmt: _Int_or_Float=0) -> Any:
        # inp_value = input(f'step {idx}: {prompt}' + \
        #                   f'\n    {description}' + \
        #                   f'\n    default input: {default}' + \
        #                   f'\n    your value: ')
        print('')
        inp_value = input(f'{prompt}' + \
                          f'\n\t{description}' + \
                          f'\ndefault input: {default}\tyour value: ').strip()
        while True:
            if inp_value == '':
                if require_type == bool and default.strip().lower() == 'true':
                    return True
                elif require_type == bool and default.strip().lower() == 'false':
                    return False
                else:
                    return require_type(default)
            if inp_value in self.forbidden_list:
                inp_value = input('this input contradicts with some internal variable, please re-input: ').strip()
                continue
            inp_type, trans_value = self._judge_inp(inp_value)
            if inp_type != require_type:
                inp_value = input('it requires a ' + repr(require_type).split("\'")[1] + ' type, please re-input: ').strip()
                continue
            if require_type == int or require_type == float:
                if trans_value > upper_lmt or trans_value < lower_lmt:
                    inp_value = input(f'the value must meet the requirement: {lower_lmt} <= x <= {upper_lmt} , please re-input: ').strip()
                    continue
            break
        return trans_value


class Config(Dialog):
    def __init__(self):
        super(Config, self).__init__()
        self.config_name: str           = 'config'
        self.start_point: int           = 0
        self.has_config: bool           = False
        ##### xyz file path #####
        self.xyz_file: str              = 'geom.xyz'
        self.multi_xyz_file: str        = 'xyz.dat'
        self.multi_xyz_usage: int       = -1
        ##### gauss input file path #####
        self.gauss_opt_inp_file: str    = 'opt.com'
        self.gauss_opt_inp_exist: bool  = False
        self.gauss_calc_inp_file: str   = 'calc.com'
        self.gauss_calc_inp_exist: bool = False
        ##### gauss inp detail #####
        self.cpu_cores: int             = 0
        self.memory_limit: int          = 0
        self.functionals: str           = ''
        self.basis: str                 = ''
        self.charge: int                = 0
        self.multiplicity: int          = 0
        ##### ML-NEA defination part #####
        self.N_TD_NEA_point: int        = 0
        self.N_ML_NEA_point: int        = 0
        self.N_max_state: int           = 0
        self.use_all_point: bool        = False
        ##### ML-NEA procedure part #####
        self.train_start_point: int     = 0
        self.train_incr_point: int      = 0
        self.rRMSE_threshould: float    = 0.0
        ##### cross section parameter #####
        self.TD_delta: float  = 0.0
        self.ML_delta: float  = 0.0

        if self.exist_path(self.config_name):
            self.read_config()
            self.has_config = True
        else:
            self.ask_config()
            self.write_config()
            print('#' * 80 + f'\nconfiguration has been written as {self.config_name}, please re-run this program to calculate!!!\n' + '#' * 80)
    
    def read_config(self):
        with open(self.config_name) as f:
            config_dict = json.loads(f.read())
        self.start_point          = config_dict['start_point']
        self.has_config           = config_dict['has_config']
        self.xyz_file             = config_dict['xyz_file']
        self.multi_xyz_file       = config_dict['multi_xyz_file']
        self.multi_xyz_usage      = config_dict['multi_xyz_usage']
        self.gauss_opt_inp_file   = config_dict['gauss_opt_inp_file']
        self.gauss_opt_inp_exist  = config_dict['gauss_opt_inp_exist']
        self.gauss_calc_inp_file  = config_dict['gauss_calc_inp_file']
        self.gauss_calc_inp_exist = config_dict['gauss_calc_inp_exist']
        self.cpu_cores            = config_dict['cpu_cores']
        self.memory_limit         = config_dict['memory_limit']
        self.functionals          = config_dict['functionals']
        self.basis                = config_dict['basis']
        self.charge               = config_dict['charge']
        self.multiplicity         = config_dict['multiplicity']
        self.N_TD_NEA_point       = config_dict['N_TD_NEA_point']
        self.N_ML_NEA_point       = config_dict['N_ML_NEA_point']
        self.N_max_state          = config_dict['N_max_state']
        self.use_all_point        = config_dict['use_all_point']
        self.train_start_point    = config_dict['train_start_point']
        self.train_incr_point     = config_dict['train_incr_point']
        self.rRMSE_threshould     = config_dict['rRMSE_threshould']
        self.TD_delta             = config_dict['TD_delta']
        self.ML_delta             = config_dict['ML_delta']

    def write_config(self):
        config_dict = {
            'start_point'           : self.start_point,
            'has_config'            : self.has_config,
            'xyz_file'              : self.xyz_file,
            'multi_xyz_file'        : self.multi_xyz_file,
            'multi_xyz_usage'       : self.multi_xyz_usage,
            'gauss_opt_inp_file'    : self.gauss_opt_inp_file,
            'gauss_opt_inp_exist'   : self.gauss_opt_inp_exist,
            'gauss_calc_inp_file'   : self.gauss_calc_inp_file,
            'gauss_calc_inp_exist'  : self.gauss_calc_inp_exist,
            'cpu_cores'             : self.cpu_cores,
            'memory_limit'          : self.memory_limit,
            'functionals'           : self.functionals,
            'basis'                 : self.basis,
            'charge'                : self.charge,
            'multiplicity'          : self.multiplicity,
            'N_TD_NEA_point'        : self.N_TD_NEA_point,
            'N_ML_NEA_point'        : self.N_ML_NEA_point,
            'N_max_state'           : self.N_max_state,
            'use_all_point'         : self.use_all_point,
            'train_start_point'     : self.train_start_point,
            'train_incr_point'      : self.train_incr_point,
            'rRMSE_threshould'      : self.rRMSE_threshould,
            'TD_delta'              : self.TD_delta,
            'ML_delta'              : self.ML_delta,
        }
        with open(self.config_name, 'w') as f:
            f.write(json.dumps(config_dict))

    def ask_config(self):
        self._judge_task_type()
        if self.start_point   == 0:
            self.ask_task_type_0()
        elif self.start_point == 1:
            # self.ask_task_type_1()
            pass
        elif self.start_point == 2:
            pass
        elif self.start_point == 3:
            pass
        else:
            pass
    
    def ask_task_type_0(self):
        self._ask_max_state()
        self._ask_TD_NEA_point()
        self._ask_ML_NEA_point()
        self._ask_use_all_point()
        if self.exist_path(self.gauss_opt_inp_file) and GaussCalc.check_opt_freq_inp(self.gauss_opt_inp_file):
            # todo 5: gaussian inp file check integrition : to Gauss_calc part check
            self.gauss_opt_inp_exist = True
        else:
            if not self.exist_path(self.xyz_file):
                self._ask_xyz_file()
        if self.exist_path(self.gauss_calc_inp_file) and GaussCalc.check_TD_inp(self.gauss_calc_inp_file):
            # todo 5: gaussian inp file check integrition : to Gauss_calc part check
            self.gauss_calc_inp_exist = True
        if (not self.gauss_opt_inp_exist) or (not self.gauss_calc_inp_exist):
            self._ask_gauss_inp_detail()
        self._ask_ML_iteration_info()
        self._ask_delta_value()
        # NX part
    
    def ask_task_type_1(self):
        self._ask_max_state()
        self._ask_TD_NEA_point()
        self._ask_use_all_point() # remember to set the ML point number as the inputed xyz.dat data number
        if not self.exist_path(self.multi_xyz_file):
            self._ask_multi_xyz_file()
        if not self.exist_path(self.gauss_calc_inp_file):
            self._ask_gauss_inp_detail()
        else:
            self.gauss_calc_inp_exist = True
            # todo 5: remember to check whether the gauss inp file is correct
        self._ask_ML_iteration_info()
        self._ask_delta_value()
        # self._ask_multi_xyz_used_for()   ##### only use all point for ML prediction, and use very little for QC calc
        # if self.multi_xyz_usage == 0:    # all point used for ML prediction
        #     self._ask_TD_NEA_point()
        #     # todo 4: to check whether ML_point > TD_point
        # elif self.multi_xyz_usage == 1:  # all point used for TD calc reference
        #     self._ask_ML_NEA_point()
        #     # todo 5: need to run NX-MD for additional point
        #     # todo 4: to check whether ML_point > TD_point
    
    def ask_task_type_2(self):
        pass

    def ask_task_type_3(self):
        pass

    def _judge_task_type(self):
        self.start_point = self.ask(idx=0.0, prompt='select where you want to start with',
                                    description='0. I only have one xyz file (not need to be eqilibrium one)\n' +
                                                '\t1. I have multiple xyz file (you should ensure the point is enough)\n' +
                                                '\t2. I have multiple gaussian calculation output file (file name should be ends with .log)\n' +
                                                '\t3. I have E and f data of many excitation state calculated from multiple xyz geometry file',
                                    default='0', require_type=int, upper_lmt=3, lower_lmt=0)
    
    def _ask_xyz_file(self):
        self.xyz_file = self.ask(idx=0.1, prompt='define the xyz file name', 
                                  description='Please input the path/file_name of the xyz file(no need to be optimized geometry)',
                                  default='geom.xyz', require_type=str)

    def _ask_au_unit(self):
        self.au_unit = self.ask(idx=0.2, prompt='define whether your xyz file is a.u. unit', 
                                description='Please input False(xyz file is angstrom unit) or True(xyz file is a.u. unit)',
                                default='False', require_type=bool)
        
    def _ask_TD_NEA_point(self):
        self.N_TD_NEA_point = self.ask(idx=0.5, prompt='define the max number of QC-NEA point',
                                       description='Please input the max number of molecules you want to calculate in order to get a QC cross-section spectra, ' + \
                                                   '\n\tthis program will not perform all the calculation unless ML training is not converged at the max QC-NEA point traiing.',
                                       default='10', require_type=int, upper_lmt=999999999, lower_lmt=1)   # orig_vale: 300
            
    def _ask_max_state(self):
        self.N_max_state = self.ask(idx=0.0, prompt='define the number of excited states',
                                    description='Please input how many excited state you want to calculate with gaussian',
                                    default='3', require_type=int, upper_lmt=10000, lower_lmt=1)  # orig_value: 10

    def _ask_ML_NEA_point(self):
        self.N_ML_NEA_point = self.ask(idx=0.0, prompt='define the number of ML-NEA point',
                                       description='Please input how many point you want to predict and generate cross-section spectra with ML method',
                                       default='50', require_type=int, upper_lmt=999999999, lower_lmt=1)  # orig_value: 300
    
    def _ask_gauss_inp_detail(self):
        self.cpu_cores = self.ask(idx=0.0, prompt='define the cpu cores', require_type=int, 
                                  description='Please input how many cores you want to calculate using Gaussian software',
                                  upper_lmt=9999999, lower_lmt=1, default='4')
        self.memory_limit = self.ask(idx=0.0, prompt='define the memory usage',
                                     description='Please input the memory (unit: GB) you want to calculate using Gaussian software',
                                     upper_lmt=9999999, lower_lmt=1, require_type=int, default='4')
        self.functionals = self.ask(idx=0.0, prompt='define the functionls of Gaussian input file', require_type=str,
                                    description='Please input the functionals', default='CAM-B3LYP')
        self.basis = self.ask(idx=0.0, prompt='define the basis of Gaussain input file', require_type=str,
                              description='Please input the basis of Gaussian input file', default='def2TZVP')
        self.multiplicity = self.ask(idx=0.0, prompt='define the multiplicity of your molecule',
                                     description='Please input the multiplicity of your molecule.',
                                     require_type=int, default='1', upper_lmt=99999, lower_lmt=1)
        self.charge = self.ask(idx=0.0, prompt='define the charge of your molecule',
                               description='Please input the charge of your molecule.',
                               require_type=int, default='0', upper_lmt=99999, lower_lmt=-9999)

    def _ask_use_all_point(self):
        self.use_all_point = self.ask(idx=0.0, prompt='define whether to use all point to train ML directly', default='False', require_type=bool,
                                      description='Please input True(it will use all point for ML training, no iteration) \n\tor False(will run on-the-fly to determin optima QC point to calc)')

    def _ask_ML_iteration_info(self):
        self.train_start_point = self.ask(idx=0.0, prompt='define the ML starting training num', default='5', require_type=int,        # default: 50
                                          description='Please input how many point you want to start with training', upper_lmt=99999999999999, lower_lmt=1)
        self.train_incr_point = self.ask(idx=0.0, prompt='define the increasement of ML training', default='2', require_type=int,      # default: 50
                                         description='Please input how many point you want to increase in each ML iteration', upper_lmt=999999999, lower_lmt=1)
        self.rRMSE_threshould = self.ask(idx=0.0, prompt='define the threshould for rRMSE', default='0.1', require_type=float,
                                         description='Please input the threshould of rRMSE, which is the convergence criteria', upper_lmt=1.0, lower_lmt=1e-10)
                                        
    def _ask_delta_value(self):
        self.TD_delta = self.ask(idx=0.0, prompt='define the delta value (broadening parameter) of TD-NEA', require_type=float,
                                 description='Please input the delta value for TD-NEA, unit: eV', default='0.05', upper_lmt=1.0, lower_lmt=1e-10)
        self.ML_delta = self.ask(idx=0.0, prompt='define the delta value (broadening parameter) of ML-NEA', require_type=float,
                                 description='Please input the delta value for ML-NEA, unit: eV', default='0.02', upper_lmt=1.0, lower_lmt=1e-10)

    def _ask_multi_xyz_file(self):  # entrance point 1 & 3: multiple xyz
        self.multi_xyz_file = self.ask(idx=1.0, prompt='define the file name/path of the multiple xyz file', 
                                       description='Please input the name/path of the multiple xyz file, this file should be the consistance of pure xyz file' + \
                                                   '\n\tbe conscious, the next xyz file and current xyz file should only be one line feed.',
                                       default='xyz.dat', require_type=str)
    
    def _ask_multi_xyz_used_for(self):  # entrance point 1: multiple xyz
        self.multi_xyz_usage = self.ask(idx=0.0, prompt='define the multiple xyz geometry is used for',
                                        description='Please input 0(used for ML prediction), or 1(used for QC calc reference)',
                                        default='0', require_type=int, upper_lmt=1, lower_lmt=0)

    def _ask_gauss_folder(self):    # entrance point 2: multiple gaussian output log 
        self.gauss_folder_path = self.ask(idx=2.0, prompt='define the folder of multiple gauss output file',
                                          description='Please input the folder path where stores the multiple gaussian output file',
                                          default='gauss_log', require_type=str)
        self.gauss_log_pattern_type = self.ask(idx=2.1, prompt='define the pattern type of gaussian log file, you can choose globing ' + \
                                               'or regular expression.', default='0', require_type=str, upper_lmt=1, lower_lmt=0,
                                               description='you can input 0(globing pattern) or 1(regular expression), if you don\'t know about it, press Enter!')
        
    def _ask_globing_pattern(self): # entrance point 2: multiple gaussian output log  
        self.gauss_log_pattern = self.ask(idx=2.5, prompt='define the globing pattern of gaussian log file', default='*.log',
                                          require_type=str, description='please input the globing pattern, "*" stands for any string, ' + \
                                          '"?" stands for any one character, you can also input nothing, \n\t' + \
                                          'In default, it will search for all the file like 1.log 2.log 3.log ...')
        
    def _ask_regex_pattern(self): # entrance point 2: multiple gaussian output log  
        self.auss_log_pattern = self.ask(idx=2.5, prompt='define the regular expression of gaussian log file', default=r'[0-9]+?\.log',
                                         require_type=str, description='please input the regular expression pattern, ' + \
                                         ' "*" stands for any string, "?" stands for any one character')

    def _ask_E_f_path(self):    # entrance point 3: ask E and f file
        self.E_fdata_path = self.ask(idx=3.3, prompt='please input the folder of E and f file', default='data', require_type=str,
                                     description='Please ensure the file name is E1.dat ... E9.dat and f1.dat ... f9.dat')


class ProcedureCtl(Config):
    def __init__(self):
        super(ProcedureCtl, self).__init__()
        if self.has_config:
            # todo 5: Environment check
            self.run_init()
            self.launch_task_according_flag()

    def run_init(self):             # todo 4: to determine whether the initialize should be put at here
        self.init_molecule: Molecule   = Molecule()
        self.opt_molecule: Molecule    = Molecule()
        self.py_script_path   = os.path.abspath(__file__[:__file__.rfind('/')])
        self.work_path        = os.getcwd()
        self.gauss            = GaussCalc(self.py_script_path, self.work_path)
        self.nx               = NewtonXCalc(self.work_path)
        self.cs_TD            = TD_cross_section(self.work_path)
        self.cs_ML            = ML_cross_section(self.work_path, self.py_script_path)
        self.multi_xyz: List[Molecule] = []

    def launch_task_according_flag(self):       # todo 4: to determine whether to add non-ML part
        if self.start_point == 0:
            if EnvCheck.check_gaussian() and EnvCheck.check_NX() and EnvCheck.check_mlatom():
                self.gauss            = GaussCalc(self.py_script_path, self.work_path)
                self.step0_opt_freq_calc()       # todo 4: to determine whether the initialize should be put at her
            else:
                # raise error on Environment not complete
                pass
        elif self.start_point == 1:
            if EnvCheck.check_gaussian() and EnvCheck.check_mlatom():
                self.step1_multiple_xyz_read_calc()
            else:
                # raise error on Environment not complete
                pass
        elif self.start_point == 2:   # todo 4: to determined whether calc cross-section without ML
            if EnvCheck.check_mlatom():
                self.step2_grep_gauss_log()
            else:
                # raise error
                pass
        elif self.start_point == 3:
            self.step3_trans_user_E_f_file()  # todo 4: to determined whether calc cross-section without ML
        else:
            # raise start point error
            pass

    def step0_opt_freq_calc(self):
        if self.gauss_opt_inp_exist:
            self.cp(self.gauss_opt_inp_file, os.path.join(self.gauss.temp_path, 'opt.com') )
            self.cd(self.gauss.temp_path)
            self.gauss.gaussian_opt_calc()
        else:
            self.init_molecule.read_xyz(self.xyz_file, self.au_unit)
            self.gauss.gaussain_opt(molecule=self.init_molecule, functools=self.functionals, basis=self.basis,
                                    cores=self.cpu_cores, memory=self.memory_limit, 
                                    charge=self.charge, multiplicity=self.multiplicity)
        # to check whehther TD_inp file is usable
        if self.gauss_calc_inp_exist:
            if not self.gauss.parse_gauss_inp_file(os.path.join(self.work_path, self.gauss_calc_inp_file)):
                # customise gaussian TD in Gaussian class
                # todo 4: raise error on parse gauss TD inp
                pass
        self.opt_molecule = self.gauss.extract_xyz(os.path.join(self.gauss.gauss_path, 'opt.log'))
        self.opt_molecule.print_xyz(os.path.join(self.cs_ML.ML_path, 'eq.xyz'))
        self.step0_1_NX_MD()

    def step0_1_NX_MD(self):
        self.nx.cd(self.nx.NX_path)
        self.opt_molecule.print_xyz('opt.xyz')
        self.cp(os.path.join(self.gauss.gauss_path, 'opt.log'), self.nx.NX_path)
        self.nx.invoke_NX('opt.xyz', 'opt.log', self.opt_molecule.atom_num, self.N_ML_NEA_point)
        self.nx.cd(self.nx.NX_path)
        # todo 2: to remove other temporary file # DEBUG  TEMP  final_output  freq.out  geom  initqp_input  opt.xyz  samp_param  samp_points
        self.multi_xyz = self.nx.multi_xyz
        self.cd(self.work_path)
        self.step0_2_NX_xyz_gauss_calc()
        
    @logger('calc all the TD point')
    def calc_all_TD_point(self):
        for idx, mole in enumerate(self.multi_xyz):
            if self.gauss_calc_inp_exist:
                self.gauss.gaussian_TD_calc_with_template(idx, mole)
            else:
                self.gauss.gaussain_TD_with_para(idx, mole, self.functionals, self.basis, self.N_max_state)
            # self.gauss.gaussain_TD_calc(idx, mole, self.calc_functionals, self.calc_basis, self.N_excit_state)

    def step0_2_NX_xyz_gauss_calc(self):
        # todo 3: calc SPC spectra
        if self.use_all_point:
            self.calc_all_TD_point()
            self.gauss.process_gaussian_log(self.gauss.all_log_path, self.N_max_state, pattern='', regex=False)
            self.cs_TD.calc(self.gauss.gauss_data_path, self.N_max_state, self.TD_delta)
            self.cs_ML.define_data_set(self, self.N_TD_NEA_point, 'xyz.dat', self.N_max_state, self.ML_delta)
            self.cs_ML.ML_train_all()
        else:
            self.cs_ML.define_data_set(self, self.N_TD_NEA_point, 'xyz.dat', self.N_max_state, self.ML_delta)
            # todo 5: in test, use start=5, incr=2
            self.cs_ML.ML_train_iter(start_point=self.train_start_point, incr_point=self.train_incr_point, threshold=self.rRMSE_threshould)
            self.cs_TD.calc(self.gauss.gauss_data_path, self.N_max_state, self.TD_delta)
        opt_E_list, opt_f_list = self.gauss.process_one_gauss_log(os.path.join(self.gauss.all_log_path, '0.log'))
        self.cs_TD.calc_SPC(opt_E_list, opt_f_list, 0.5, 'cross-section-SPC.dat')
        self.step0_3_draw()
            
    def step0_3_draw(self):
        self.cd(self.work_path)
        plot = Plot()
        plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for QC-NEA', grid=True)
        plot.plot_file(self.cs_TD.TD_out_name, 'TD-cross-section.png', x_column=1, y_column=3)
        plot = Plot()
        plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for ML-NEA', grid=True)
        # plot.set_info('Energy / eV', 'cross-section', 'cross-section for ML-NEA', grid=True)
        plot.plot_file(self.cs_ML.ML_out_name, 'ML-cross-section.png', x_column=1, y_column=3)
        plot = Plot()
        plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for QC-SPC', grid=True)
        plot.plot_file(self.cs_TD.SPC_out_name, 'SPC-cross-section.png', x_column=1, y_column=3)


    def step1_multiple_xyz_read_calc(self):
        # to check whehther TD_inp file is usable
        if self.gauss_calc_inp_exist:
            if not self.gauss.parse_gauss_inp_file(os.path.join(self.work_path, self.gauss_calc_inp_file)):
                # customise gaussian TD in Gaussian class
                # todo 4: raise error on parse gauss TD inp
                pass
        self.step0_2_NX_xyz_gauss_calc()

    def step2_grep_gauss_log(self):
        gauss_folder_path = self.ask(idx=2.0, prompt='define the folder of multiple gauss output file',
                                     description='Please input the folder path where stores the multiple gaussian output file',
                                     default='gauss_log', require_type=str)
        gauss_log_pattern_type = self.ask(idx=2.1, prompt='define the pattern type of gaussian log file, you can choose globing ' + \
                                          'or regular expression.', default='0', require_type=str, upper_lmt=1, lower_lmt=0,
                                          description='you can input 0(globing pattern) or 1(regular expression)')
        self.ask_use_all_point(2.2)
        self.ask_state_NEA(2.3, 2.4)
        if gauss_log_pattern_type == 0:
            # globing
            gauss_log_pattern = self.ask(idx=2.5, prompt='define the globing pattern of gaussian log file', default='*.log',
                                         require_type=str, description='please input the globing pattern, "*" stands for any string, ' + \
                                         '"?" stands for any one character, you can also input nothing, and it will use all the file in the folder')
            self.gauss.process_gaussian_log(gauss_folder_path, self.N_excit_state, gauss_log_pattern, regex=False)
        elif gauss_log_pattern_type == 1:
            # regualr expression
            gauss_log_pattern = self.ask(idx=2.5, prompt='define the regular expression of gaussian log file', default=r'[0-9]+?\.log',
                                         require_type=str, description='please input the regular expression pattern, ' + \
                                         ' "*" stands for any string, "?" stands for any one character')
            self.gauss.process_gaussian_log(gauss_folder_path, self.N_excit_state, gauss_log_pattern, regex=True)
        else:
            # todo 3: to raise a error 
            pass
        # todo 5: get multi-xyz variable from gauss log, the same for fN_excit_state, N_TD_NEA
        # todo 5: ask for the ML iteration or train all
        pass

    def step3_trans_user_E_f_file(self):
        self.ask_use_all_point(3.0)
        self.ask_state_NEA(3.1, 3.2)
        data_path = self.ask(idx=3.3, prompt='please input the folder of E and f file', default='data', require_type=str,
                             description='Please ensure the file name is E1.dat ... E9.dat and f1.dat ... f9.dat')
        self.cp(data_path + '/*', self.gauss.gauss_data_path)
        # todo 5: get multi-xyz variable from gauss log, the same for fN_excit_state, N_TD_NEA
        # todo 5: ask for the ML iteration or train all


class ArgParser():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.set_arg()
        self.args = self.parser.parse_args()
        
    def set_arg(self) -> None:
        self.parser.add_argument('-Nexcitations', default=3, type=int, help='define how many excited state you want to calculate. (default=3)')
        self.parser.add_argument('-nQMpoints', default=0, type=int, help='define the point you want to directly calculate with ML-NEA method. (default=0, will not calculate all point)')
        self.parser.add_argument('-nMaxPoints', default=10000, type=int, help='define the max QC calculation point in the ML procedure. (default=10000)')
        self.parser.add_argument('-MLpoints', default=50000, type=int, help='define how many point you want to predict with ML. (default=50000)')
        self.parser.add_argument('-plotQCNEA', action='store_true', help='define whether to draw QC-NEA plot, default=False')
        self.parser.add_argument('-plotQCSPC', action='store_true', help='define whether to draw QC-SPC plot, default=False')
        self.parser.add_argument('-deltaQCNEA', default=0.01, type=float, help='define the delta value you want to use in QC-NEA. (default=0.01)')
        # self.parser.add_argument('-useAllPoint', default=False, type=bool, help='if True, will use all NX point to calc, or will run it iterately(default)')


class CtlPartAdapter(LinuxInterface):
    paper_adapter: int = 1
    custom_adapter: int = 2
    def __init__(self, handler: Union['RecurPaperProcedure', 'ProcedureCtl'], adapter_type: int):
        self.adapter: Union['RecurPaperProcedure', 'ProcedureCtl'] = handler
        self.type = adapter_type
    
    def recheck_QC_calc(self, start: int, end: int) -> Tuple[bool, int, int]:
        if end < start:
            err_print('inner error: iteratively calc TD encountered error end < start')
            exit()
        if self.type == self.paper_adapter:  # need to check whether it has E*.dat and f*.dat
            handle: RecurPaperProcedure = self.adapter
            if handle.exist_E_f_data:  # exists E f data   # end is not included
                if start > handle.E_f_max_line:  #  max  [start   end)
                    return True, start, end
                elif start <= handle.E_f_max_line < end:  #  [start max  end)
                    return True, handle.E_f_max_line, end
                else:
                    return False, start, end
            else:  # not exists E*.dat f*.dat
                return True, start, end
        elif self.type == self.custom_adapter:
            return True, start, end
    
    @property
    def gauss(self):
        return self.adapter.gauss

    @property
    def multi_xyz(self):
        return self.adapter.multi_xyz
    
    @property
    def gauss_calc_inp_exist(self) -> bool:
        if self.type == self.paper_adapter:
            paper: RecurPaperProcedure = self.adapter
            return self.exist_path(os.path.join(paper.work_path, paper.gauss_TD_inp_name))
        elif self.type == self.custom_adapter:
            custom: ProcedureCtl = self.adapter
            return custom.gauss_calc_inp_exist
    
    @property
    def functionals(self) -> str:
        if self.type == self.paper_adapter:
            err_print('error invoke by paper adapter, exit!')
            exit()
        elif self.type == self.custom_adapter:
            custom: ProcedureCtl = self.adapter
            return custom.functionals
    
    @property
    def basis(self) -> str:
        if self.type == self.paper_adapter:
            err_print('error invoke by paper adapter, exit!')
            exit()
        elif self.type == self.custom_adapter:
            custom: ProcedureCtl = self.adapter
            return custom.basis
        
    @property
    def N_max_state(self) -> int:
        if self.type == self.paper_adapter:
            paper: RecurPaperProcedure = self.adapter
            return paper.max_state
        elif self.type == self.custom_adapter:
            custom: ProcedureCtl = self.adapter
            return custom.N_max_state
    

class RecurPaperProcedure(ArgParser, LinuxInterface):
# ╭─────────────────────────────────────────────────────────────────────────────────────────────────╮
# │ ◎ ○ ○ ░░░░░░░░░░░░░░░░░░░░░░░░░ Control Part for Benzene 50k point ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
# ├─────────────────────────────────────────────────────────────────────────────────────────────────┤
# │                                                                                                 │
# │               .───────────────.                              .───────────────.                  │
# │       ┌─yes──(   has xyz.dat   )──no──┐             ┌──yes──(   has opt.log   )───no─┐          │
# │       │       `───────────────'       │             │        `───────────────'       │          │
# │       │                               │             │                                ▼          │
# │       │                               │             │                        .───────────────.  │
# │       │                               ▼             │               ┌──no───(   has opt.com   ) │
# │       │                        .─────────────.      │               │        `───────────────'  │
# │       │             ┌─────────(  NX-MD calc   )     │               ▼                │          │
# │       │             │          `─────────────'      │       .───────────────.       yes         │
# │       │             │                 ▲             │      (   error, exit   )       │          │
# │       │             │                 │             │       `───────────────'        ▼          │
# │       │             │                 │             │                         .─────────────.   │
# │ ┌ ─ ─ ┼ ─ ─ ─ ┐     │                 │             │                        (  run opt.com  )  │
# │       ▼             │                 │             ▼                         `─────────────'   │
# │ │ .───────.   │     │                 │     .───────────────.                        │          │
# │  ( ML-NEA  )◀───────┘                 └────(  get opt freq   )◀──────────────────────┘          │
# │ │ `───────'   │                             `───────────────'                                   │
# │       ▲                                             │                                           │
# │ └ ─ ─ ┼ ─ ─ ─ ┘                                     │                                           │
# │       └─────────────────────────────────────────────┘                                           │
# │                                                                                                 │
# └─────────────────────────────────────────────────────────────────────────────────────────────────┘
    def __init__(self) -> None:
        pass
    
    def check_env(self) -> None:
        pass
    
    def calc(self) -> None:
        # self.QC_calc_pre_check()
        self.detect_E_f_and_calc()
        self.opt_freq_calc()
        self.NX_multi_xyz_calc()
        if self.use_all_point:
            self.ML_NEA_calc_all()
        else:
            self.ML_NEA_calc_iter()
        self.QC_cross_section_calc()
        # self.draw_cs()   # todo 5: add environment check
        
    def invoke_by_self(self):
        super().__init__()
        self.pre_define()
        ##### args #####
        self.max_state: int = self.args.Nexcitations
        self.TD_point: int = self.args.nMaxPoints
        self.ML_point: int = self.args.MLpoints
        if self.args.nQMpoints:
            self.use_all_point: bool = True
            self.TD_point = self.args.nQMpoints
        else:
            self.use_all_point: bool = False
        #self.use_all_point: bool = self.args.useAllPoint
        self.delta_QC = self.args.deltaQCNEA
        self.calc()
        plotQCNEA = self.args.plotQCNEA
        plotQCSPC = self.args.plotQCSPC
        if not EnvCheck.check_matplotlib():
            err_print('not exists matplotlib module, will not plot!')
        else:
            self.draw_cs(plotQCNEA=plotQCNEA, plotQCSPC=plotQCSPC)
        print_citation()
        
    def invoke_by_API(self, Nexcitations: int=3, nMaxPoints: int=10000, nNEApoints: int=0, MLpoints: int=500, deltaQCNEA: float=0.05):
        #print_citation()
        self.pre_define()
        ##### args #####
        self.max_state: int = Nexcitations
        self.TD_point: int = nMaxPoints
        self.ML_point: int = MLpoints
        if nNEApoints:
            self.use_all_point: bool = True
            self.TD_point = nNEApoints
        else:
            self.use_all_point: bool = False
        self.delta_QC = deltaQCNEA
        self.calc()
        
    def pre_define(self) -> None:
        ##### file existance #####
        self.work_path = os.getcwd()
        self.py_script_path   = os.path.dirname(os.path.abspath(__file__))  #self.py_script_path   = os.path.abspath(__file__[:__file__.rfind('/')])
        global py_script_path
        py_script_path = self.py_script_path
        self.gauss_opt_inp_name: str = 'gaussian_optfreq.com' # self.gauss_opt_inp_name: str = 'opt-freq.com'
        self.gauss_TD_inp_name: str = 'gaussian_ef.com' # self.gauss_TD_inp_name: str = 'TD.com'
        self.gauss_opt_log_name: str = 'gaussian_optfreq.log' # self.gauss_opt_log_name: str = 'opt-freq.log'
        self.nx_nea_geom_name: str = 'nea_geoms.xyz' # self.nx_nea_geom_name: str = 'xyz.dat'
        self.eq_init_xyz_name: str = 'eq.xyz'
        self.eq_xyz_name: str = 'eq.xyz'
        self.E_files_name: str = 'E%s.dat'
        self.f_files_name: str = 'f%s.dat'
        self.exist_E_f_data: bool = False
        self.E_f_max_line: int = 0
        ##### calc handler class #####
        self.gauss    = GaussCalc(self.py_script_path, self.work_path)
        self.nx       = NewtonXCalc(self.work_path)
        self.cs_TD    = TD_cross_section(self.work_path)
        self.cs_ML    = ML_cross_section(self.work_path, self.py_script_path)
        self.cs_SPC   = SPC_cross_section()
        ##### intermediate variable #####
        self.opt_molecule: Molecule    = Molecule()
        self.multi_xyz: List[Molecule] = []
        ##### cross section parameter #####
        self.delta_ML = 0.01
        self.delta_QC = 0.01
        self.delta_SPC = 0.3
        
    def opt_freq_calc(self, force_calc: bool=False):
        self.cd(self.work_path)
        if not force_calc:
            if self.exist_path(self.eq_init_xyz_name) and self.exist_path(self.nx_nea_geom_name):
                self.opt_molecule.read_xyz(self.eq_init_xyz_name) 
                return
        if not EnvCheck.check_NX():
            exit()
        self.cd(self.work_path)
        if not self.exist_path(self.gauss_opt_log_name): # no exists opt-freq.log
            if self.exist_path(self.gauss_opt_inp_name): # exists opt-freq.com, will calc it
                if not EnvCheck.check_gaussian():
                    exit()
                self.gauss.gaussian_opt_calc(self.gauss_opt_inp_name)
            else:  # not exists opt-freq.com, raise error
                err_print(f'not exists {self.gauss_opt_inp_name} or {self.gauss_opt_log_name}, exit')
                exit()
        else:
            self.cp(self.gauss_opt_log_name, os.path.join(self.gauss.gauss_path, self.gauss_opt_log_name))
        # now exists opt-freq.log, extract opt.xyz & freq
        if not force_calc:
            self.opt_molecule = self.gauss.extract_xyz(
                os.path.join(self.gauss.gauss_path, self.gauss_opt_log_name))
            # self.opt_molecule.print_xyz(self.eq_xyz_name)  # no needed, don't pollute the working directory

    def NX_multi_xyz_calc(self):
        def NX_gen_ensemble():
            if not EnvCheck.check_NX():
                exit()
            self.cd(self.nx.NX_path)
            self.opt_molecule.print_xyz('opt.xyz')
            self.cp(os.path.join(self.gauss.gauss_path, self.gauss_opt_log_name),
                    os.path.join(self.nx.NX_path, 'opt.log'))
            self.nx.invoke_NX('opt.xyz', 'opt.log', self.opt_molecule.atom_num, self.ML_point)
            self.multi_xyz = self.nx.multi_xyz
        
        self.cd(self.work_path)
        if self.exist_path(self.nx_nea_geom_name): # exists nea geom path
            self.multi_xyz = Molecule.read_multi_xyz(self.nx_nea_geom_name)
            if len(self.multi_xyz) - 1 > self.ML_point:
                self.multi_xyz = self.multi_xyz[:self.ML_point + 1]
            elif len(self.multi_xyz) - 1 < self.ML_point:
                orig_xyz = self.multi_xyz
                self.opt_freq_calc(force_calc=True)
                NX_gen_ensemble()
                self.multi_xyz = orig_xyz + self.multi_xyz[len(orig_xyz): ]
        else:  # not exists multiple nea geom data, need to calc it with NX
            NX_gen_ensemble()

    def ML_NEA_calc_all(self):
        self.cd(self.cs_ML.ML_path)
        self.opt_molecule.print_xyz(self.eq_xyz_name)
        # for idx, mole in enumerate(self.multi_xyz):
        #     self.gauss.gaussian_TD_calc_with_template(idx, mole)
        # self.gauss.process_gaussian_log(self.gauss.all_log_path, self.max_state, pattern='', regex=False)
        ctl_adapter = CtlPartAdapter(self, CtlPartAdapter.paper_adapter)
        self.cs_ML.define_data_set(this=ctl_adapter, TD_point=self.TD_point, xyz_file=self.nx_nea_geom_name,
                                   max_state=self.max_state, delta=self.delta_ML)
        self.cs_ML.ML_train_all()
        
    def ML_NEA_calc_iter(self):
        self.cd(self.cs_ML.ML_path)
        self.opt_molecule.print_xyz(self.eq_xyz_name)
        ctl_adapter = CtlPartAdapter(self, CtlPartAdapter.paper_adapter)
        self.cs_ML.define_data_set(this=ctl_adapter, TD_point=self.TD_point, xyz_file=self.nx_nea_geom_name,
                                   max_state=self.max_state, delta=self.delta_ML)
        # self.cs_ML.ML_train_iter(start_point=5, incr_point=2, threshold=0.2)  # threshold=0.1
        self.cs_ML.ML_train_iter(start_point=50, incr_point=50, threshold=0.1)  # threshold=0.1

    def QC_calc_pre_check(self):
        self.cd(self.work_path)
        if not self.gauss.parse_gauss_inp_file(self.gauss_TD_inp_name):     # this procedure will also make a template for TD calculation
            err_print(f'has error on {self.gauss_TD_inp_name}, exit!')
            exit()
    
    def QC_cross_section_calc(self):
        self.cd(self.cs_TD.cs_path)
        self.cs_TD.calc(self.gauss.gauss_data_path, self.max_state, self.delta_QC, self.cs_TD.TD_out_name)  # use default delta=0.05
        opt_E_list, opt_f_list = self.gauss.process_one_gauss_log(os.path.join(self.gauss.all_log_path, '0.log'))
        self.opt_E_list, self.opt_f_list = opt_E_list, opt_f_list
        # self.cs_TD.calc_SPC(opt_E_list, opt_f_list, self.delta_SPC, self.cs_TD.SPC_out_name)
        self.cs_SPC.read_data_from_para(opt_E_list, opt_f_list)
        SPC_delta = 0.3
        x_min = min(self.cs_TD.x_min, self.cs_ML.x_min) - SPC_delta / 2
        x_max = max(self.cs_TD.x_max, self.cs_ML.x_max) + SPC_delta / 2
        self.cs_SPC.set_para(delta=SPC_delta, x_min=x_min, x_max=x_max)
        self.cs_SPC.calc(self.cs_SPC.SPC_out_name)
    
    def draw_cs(self, plotQCNEA: bool=True, plotQCSPC: bool=True):          # todo 4: plot in a figure
        qc_nea_name = 'cross-section_qc-nea.png'
        ml_nea_name = 'cross-section_ml-nea.png'
        spc_nea_name = 'cross-section_spc.png'
        self.cd(self.cs_ML.cs_path)
        if False:
            plot = Plot()
            plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for QC-NEA', grid=True)
            plot.plot_file(self.cs_TD.TD_out_name, qc_nea_name, x_column=1, y_column=3)
            # plot.plot_file(self.cs_TD.TD_out_name, 'TD-cross-section.png', x_column=1, y_column=3)
            plot = Plot()
            plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for ML-NEA', grid=True)
            plot.plot_file(self.cs_ML.ML_out_name, ml_nea_name, x_column=1, y_column=3)
            # plot.plot_file(self.cs_ML.output_file_name, 'ML-cross-section.png', x_column=1, y_column=3)
            plot = Plot()
            plot.set_info('Energy, eV', r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', 'cross-section for QC-SPC', grid=True)
            plot.plot_file(self.cs_TD.SPC_out_name, spc_nea_name, x_column=1, y_column=3)
            # plot.plot_file(self.cs_TD.SPC_out_name, 'SPC-cross-section.png', x_column=1, y_column=3)
        else:
            file_names: List[str] = []
            legend_names: List[str] = []
            colors: List[str] = []
            plot = Plot()
            if self.exist_path(os.path.join(self.work_path, 'cross-section_ref.dat')):
                file_names.append('cross-section_ref.dat')
                legend_names.append('ref')
                colors.append('k')
                self.cp(os.path.join(self.work_path, 'cross-section_ref.dat'), self.cs_ML.cs_path)
            if self.exist_path(os.path.join(self.cs_ML.cs_path, self.cs_TD.TD_out_name)) and plotQCNEA:
                file_names.append(self.cs_TD.TD_out_name)
                legend_names.append('QC-NEA')
                colors.append('b')
            if self.exist_path(os.path.join(self.cs_ML.cs_path, self.cs_ML.ML_out_name)):
                file_names.append(self.cs_ML.ML_out_name)
                legend_names.append('ML-NEA')
                colors.append('r')
            if self.exist_path(os.path.join(self.cs_ML.cs_path, self.cs_SPC.SPC_out_name)) and plotQCSPC:
                file_names.append(self.cs_SPC.SPC_out_name)
                legend_names.append('QC-SPC')
                colors.append('g')
                plot.plot_spc_bar(self.opt_E_list, self.opt_f_list)
                # plot.ax.set_xlim(min(self.opt_E_list), max(self.opt_E_list))
            if False:
                if self.exist_path(os.path.join(self.work_path, 'cross-section_ref.dat')):
                    # file_names.append('cross-section_ref.dat')
                    file_names = ['cross-section_ref.dat'] + file_names
                    # legend_names.append('ref-NEA')
                    legend_names = ['ref-NEA'] + legend_names
                    # colors.append('k')
                    colors = ['k'] + colors
                    self.cp(os.path.join(self.work_path, 'cross-section_ref.dat'), self.cs_ML.cs_path)
                file_names = [self.cs_TD.TD_out_name, self.cs_ML.ML_out_name, self.cs_TD.SPC_out_name]
                legend_names = ['QC-NEA', 'ML-NEA', 'QC-SPC']
                colors = ['b', 'r', 'g']
            if not file_names:
                err_print('not caluclated cross-section file exists. Exit!')
                exit()
            plot.set_info(xlabel='Energy, eV', ylabel=r'Cross section, $\mathrm{\AA^2}$ molecule$^{–1}$', grid=False)
            plot.plot_multiple_file(file_names, legend_names, colors, x_column=1, y_column=3)
            plot.ax.set_ylim(bottom=0.0)
            plot.ax.margins(x=0.0)
            plot.save_fig('plot.png')

    def detect_E_f_and_calc(self):
        for i in range(1, self.max_state + 1):
            if (not self.exist_path(f'E{i}.dat')) or (not self.exist_path(f'f{i}.dat')):
                # err_print('E*.dat and f*.dat is not compatible with max_excited_state')
                self.QC_calc_pre_check()
                return
        self.exist_E_f_data = True
        self.fabricate_gauss_data()

    def fabricate_gauss_data(self):     # the 0th should be the equilibrium geom
        def read_line_data(file_name: str):
            with open(file_name) as f:
                data = f.read().splitlines()
            return list(map(float, data))

        def fabricate_gauss_log(idx: int, Es: List[float], fs: List[float]):
            with open(os.path.join(self.gauss.all_log_path, f'{idx}.log'), 'w') as file:
                for i, E in enumerate(Es):
                    f = fs[i]
                    file.write(template % (i + 1, E, f) + '\n')

        template = ' Excited State   %d:      Singlet-   %f eV   0.000 nm  f=%f  <S**2>=0.000'
        E_data = [];  f_data = []
        for i in range(1, self.max_state + 1):
            E_data.append(read_line_data(f'E{i}.dat'))
            f_data.append(read_line_data(f'f{i}.dat'))
        self.E_f_max_line = len(E_data[0])
        if len(E_data[0]) != len(f_data[0]):  # todo 3: need to check every file's line
            err_print('E*.dat and f*.dat doesn\'t match, exit!')
            exit()
        for idx, _ in enumerate(E_data[0]):
            Es = [ x[idx] for x in E_data]
            fs = [ x[idx] for x in f_data]
            fabricate_gauss_log(idx, Es, fs)
        

Nexcitations_str = 'Nexcitations'
nQMpoints_str = 'nQMpoints'
nMaxPoints_str = 'nMaxPoints'
MLpoints_str = 'MLpoints'
deltaQCNEA_str = 'deltaQCNEA'
   
   
class ControlPartProxy():
    def __init__(self):
        self.start_benzene_50k_ML_calc()
        # judge this execution is define work input file or directly execute
        # define work need some special argument with the execution command
    @classmethod
    @logger('Main procedure starts')
    def invoke_paper_recur_api(cls, Nexcitations: int=3, nMaxPoints: int=10000, nTDpoints: int=0, MLpoints: int=50000, deltaQCNEA: float=0.01, plotQCNEA: bool=False, plotQCSPC: bool=False):
        cwd = os.getcwd()
        Nexcitations = 3 if Nexcitations == 0 else Nexcitations
        nMaxPoints = 10000 if nMaxPoints == 0 else nMaxPoints
        MLpoints = 50000 if MLpoints == 0 else MLpoints
        deltaQCNEA = 0.01 if deltaQCNEA == 0.0 else deltaQCNEA
        handle = RecurPaperProcedure()
        handle.invoke_by_API(Nexcitations, nMaxPoints, nTDpoints, MLpoints, deltaQCNEA)
        if not EnvCheck.check_matplotlib():
            err_print('not exists matplotlib module, will not plot!')
        else:
            handle.draw_cs(plotQCNEA, plotQCSPC)
        print_citation()
        LinuxInterface.cd(cwd)
    
    @classmethod
    @logger('Main procedure starts')
    def invoke_paper_recur_self(cls):
        handle = RecurPaperProcedure()
        handle.invoke_by_self()
    
    @logger('Main procedure starts')
    def start_benzene_50k_ML_calc(self):
        RecurPaperProcedure()
        # super(Main, self).__init__()
    
    @logger('Main procedure starts')
    def start_user_custom_calc(self):
        ProcedureCtl()

    @classmethod
    def copyright_printer(cls):
        pass

def parse_api(args: List[str]):

    def transfer(data: str, var_expr: str, require_type: type):
        try:
            tmp = require_type(data)
            return tmp
        except:
            err_print( '%s requires a %s type. exit!' % (var_expr, repr(type).split("'")[1]) )
            exit()

    plotQCSPC: bool = False
    plotQCNEA: bool = False
    Nexcitations: int = 0
    nQMpoints: int = 0
    nMaxPoints: int = 0
    MLpoints: int = 0
    deltaQCNEA: float = 0.0
    
    for arg in args:
        splited = arg.split('=')
        if len(splited) == 1:
            if splited[0].strip() == 'plotQCNEA':
                plotQCNEA = True
            elif splited[0].strip() == 'plotQCSPC':
                plotQCSPC = True
            elif splited[0].strip() in ['crossSection', 'cross-section', 'cross_section', 'MLNEA', 'ML-NEA', 'ML_NEA']:
                pass
            else:
                err_print('unrecognised parameter: %s. exit!' % splited[0])
                exit()
        if len(splited) == 2:
            if splited[0].strip().lower() == Nexcitations_str.lower():
                Nexcitations = transfer(splited[1], 'Nexcitations', int)
            elif splited[0].strip().lower() == nMaxPoints_str.lower():
                nMaxPoints = transfer(splited[1], 'nMaxPoints', int)
            elif splited[0].strip().lower() == nQMpoints_str.lower():
                nQMpoints = transfer(splited[1], 'nQMpoints', int)
            elif splited[0].strip().lower() in [MLpoints_str.lower(), 'nNEpoints'.lower()]:
                MLpoints = transfer(splited[1], 'nNEpoints', int)
            # elif splited[0].strip().lower() == MLpoints_str.lower():
            #     MLpoints = transfer(splited[1], 'MLpoints', int)
            elif splited[0].strip().lower() == deltaQCNEA_str.lower():
                deltaQCNEA = transfer(splited[1], 'deltaQCNEA', float)
            else:
                err_print('unrecognised parameter: %s. exit!' % splited[0])
                exit()
        if len(splited) >= 3:
            err_print('has more than one "=" symbol, exit!')
            exit()
    ControlPartProxy.invoke_paper_recur_api(Nexcitations, nMaxPoints, nQMpoints, MLpoints, deltaQCNEA, plotQCNEA, plotQCSPC)

def print_help():
    help='''
-------------------------------------------------------------------------------
  Options for ML-NEA:
    usage: MLatom.py cross-section [optional arguments]

  optional arguments:
    nExcitations=N      number of excited states to calculate. (default=3)
    nQMpoints=N         user-defined number of QM calculations for training ML. 
                        (default=0, number of QM calculations will be determined iteratively)
    plotQCNEA           requests plotting QC-NEA cross section
    deltaQCNEA=float    define the broadening parameter of QC-NEA cross section
    plotQCSPC           requests plotting cross section obtained via single point convolution
  
  advanced arguments:
    nMaxPoints=N        maximum number of QC calculations in the iterative procedure. (default=10000)
    nNEpoints=N          number of ML ensemble prediction. (default=50000)
  
  Environment setting (no needed)
    $NX                     Newton-X environment
    configured gaussian     Gaussian environment (see Gaussian manual for more detail)
  
  file that needs prepared
  1. mandatory file
    gaussian_optfreq.com    file that will execute gaussian opt and freq calc (alternatively with eq.xyz and nea_geoms.xyz)
    gaussian_ef.com         template file that will execute QC calculations
    
  2. optional file
    cross-section_ref.dat   reference cross-section file calculated with Newton-X
    eq.xyz                  optimized geometry file (to be used together with nea_geoms.xyz)
    nea_geoms.xyz           file with all geometries in nuclear ensemble (to be used together with eq.xyz)
    E1.dat E2.dat ...       files that store the excitation energy and oscillator strength
    f1.dat f2.dat ...       per line which corresponds to nea_geoms.xyz
    
  
  output file
    cross-section/cross-section_ml-nea.dat        cross-section spectra calculated with ML-NEA method
    cross-section/cross-section_qc-nea.dat        cross-section spectra calculated with QC-NEA method
    cross-section/cross-section_spc.dat           cross-section spectra calculated with single-point-convolution
    cross-section/plot.png                        the plot of cross-section calculated with different kinds of methods
  
  citation for ML-NEA:
    B.-X. Xue, M. Barbatti, P. O. Dral, J. Phys. Chem. A 2020, 124, 7199. DOI: 10.1021/acs.jpca.0c05310
  
  Since it uses Newton-X, please also cite Newton-X appropriately
-------------------------------------------------------------------------------
    '''
    print(help)

def print_citation():
    citation     = '  B.-X. Xue, M. Barbatti, P. O. Dral, J. Phys. Chem. A 2020, 124, 7199. DOI: 10.1021/acs.jpca.0c05310'
    nearef       = '  R. Crespo-Otero, M. Barbatti, Theor. Chem. Acc. 2012, 131, 1237,  DOI: 10.1007/s00214-012-1237-4'
    wignerref    = '  Schinke, R. Photodissociation Dynamics: Spectroscopy and Fragmentation of Small Polyatomic Molecules; Cambridge University Press: Cambridge, U.K., 1995'
    cs_algorithm = '  M. Barbatti, G. Granucci, M. Ruckenbauer, F. Plasser, \n' \
                   '  R. Crespo-Otero, J. Pittner, M. Persico, H. Lischka, NEWTON-X: \n' \
                   '  a package for Newtonian dynamics close to the crossing seam, \n' \
                   '  version 2.2, www.newtonx.org (2017). \n' \
                   '  \n' \
                   '  M. Barbatti, M. Ruckenbauer, F. Plasser, J. Pittner, G. Granucci, \n' \
                   '  M. Persico, H. Lischka, WIREs: Comp. Mol. Sci., 4, 26 (2014).' 
    print('*'*80)
    print(' You are going to use feature(s) listed below. \n Please cite corresponding work(s) in your paper:')
    print('\n %s:\n\t%s'%('ML-NEA',citation))
    print('\n %s:\n\t%s'%('NEA',   nearef))
    print('\n %s:\n\t%s'%('Samling from a Wigner distribution',wignerref))
    print('\n %s:\n\t%s'%('NEWTON-X',cs_algorithm))
    print('*'*80)


if __name__ == "__main__":
    ControlPartProxy.invoke_paper_recur_self()
    # ControlPartProxy.invoke_paper_recur_api(nTDpoints=200, plotQCNEA=True, deltaQCNEA=0.08)
    
