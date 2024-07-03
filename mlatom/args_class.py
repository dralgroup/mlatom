#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! args_class: handling input of MLatom                                      ! 
  ! Implementations by: Bao-Xin Xue                                           ! 
  !---------------------------------------------------------------------------! 
'''

import os
from collections import defaultdict, UserDict
from os import replace
from typing import Callable, DefaultDict, Sequence, Tuple, Union
from typing import List, Tuple, Dict, Set, Any
from copy import copy
import re
from . import stopper
from .doc import Doc
from .models import methods

default_MLprog={
    'kreg': 'mlatomf',
    'id': 'mlatomf',
    'kid': 'mlatomf',
    'ukid': 'mlatomf',
    'akid': 'mlatomf',
    'pkid': 'mlatomf',
    'krr-cm': 'mlatomf',
    'gap': 'gap',
    'gap-soap': 'gap',
    'sgdml': 'sgdml',
    'gdml':'sgdml',
    'dpmd': 'deepmd-kit',
    'deeppot-se': 'deepmd-kit',
    'physnet': 'physnet',
    'mace': 'mace',
    'ani': 'torchani',
    'ani-aev':'torchani',
    'ani1x': 'torchani',
    'ani1ccx': 'torchani',
    'ani2x': 'torchani',
    'ani-tl':'torchani',
    'mlqd': 'mlqd',   
}

class KeyNotFoundException(Exception):
    pass


class DuplicatedKeyException(Exception):
    pass


class AttributeDict(UserDict):
    data: Dict[Any, Any]
    lower_key_set: Set[Any]
    lower_key_to_key: Dict[str, str]
    attribute_dict_variable = ['data', 'lower_key_set', 'lower_key_to_key', 'attribute_dict_variable']
    
    def __init__(self) -> None:
        self.__dict__['data'] = {}
        self.__dict__['lower_key_set'] = set()
        self.__dict__['lower_key_to_key'] = {}
        self.__dict__['writable_flag'] = False
    
    @classmethod
    def merge_dict(cls, my_dict: 'AttributeDict', other: Dict[str, Any]) -> None:
        for other_key in other.keys():
            other_key_strip = other_key.strip()        # for the usage of my_dict
            other_key_lower = other_key_strip.lower()  # for the usage of my_dict.lower_key_set
            if other_key_lower in my_dict.lower_key_set:
                if type(other[other_key]) in [dict, cls]:
                    if type(my_dict[other_key_strip]) is not cls:
                        my_dict[other_key_strip] = AttributeDict()
                    cls.merge_dict(my_dict[other_key_strip], other[other_key])
                else:
                    my_dict[other_key_strip] = other[other_key]
            else:
                my_dict.lower_key_set.add(other_key_lower)
                my_dict.lower_key_to_key[other_key_lower] = other_key_strip
                my_dict[other_key_strip] = other[other_key]
    
    def key_in_dict(self, name: str) -> bool:
        lower_key = name.lower().strip()
        return lower_key in self.lower_key_set

    @classmethod
    def dict_to_attributedict(cls, d: Dict[str, Any]) -> 'AttributeDict':
        new_dict = cls()
        for key, value in d.items():
            if type(value) is dict:
                new_value = cls.dict_to_attributedict(value)
                new_dict[key] = new_value
            else:
                new_dict[key] = value
        return new_dict
    
    @classmethod
    def normal_dict(cls, d: 'AttributeDict') -> Dict[str, Any]:
        new_dict = {}
        for key, value in d.items():
            if type(value) in [cls, dict]:
                new_dict[key] = cls.normal_dict(value)
            else:
                new_dict[key] = value
        return new_dict
    
    def _raise_internal_key_error(self, name: str) -> None:
        if name.lower() in self.attribute_dict_variable:
            raise KeyError(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('__') or (name in self.__dict__) or (name in dir(self)):
            super(AttributeDict, self).__setattr__(name, value)
        else:
            self.__setitem__(name, value)
    
    # def __getattribute__(self, name: str) -> Any:
    #     if name in super(AttributeDict, self).__getattribute__('attribute_dict_variable'):
    #         raise KeyError(f'{name} can not appears on the args_class.{self.__class__}')
    #     return super(AttributeDict, self).__getattribute__(name)
    
    def __getattr__(self, name: str) -> Any:
        if not self.key_in_dict(name):
            self[name] = self.__class__()
        return self.__getitem__(name)
    
    def __setitem__(self, name: str, value: Any) -> Any:
        self._raise_internal_key_error(name)
        key_strip = name.strip()
        key_lower = key_strip.lower()
        if type(value) is dict:
            new_value = self.dict_to_attributedict(value)
        else:
            new_value = copy(value)
        if key_lower in self.lower_key_set:
            true_key = self.lower_key_to_key[key_lower]
            self.data[true_key] = new_value
        else:
            self.lower_key_set.add(key_lower)
            self.lower_key_to_key[key_lower] = key_strip
            self.data[key_strip] = new_value
    
    def __getitem__(self, name: str) -> Any:
        self._raise_internal_key_error(name)
        if name.startswith('__') or (name in self.__dict__):
            return self.__dict__[name]
        lower_key = name.lower().strip()
        if lower_key in self.lower_key_set:
            return self.data[self.lower_key_to_key[lower_key]]
        raise KeyError(name)


class ArgsBase():
    """As a argument parsing base class to be extended.
    
        basic usage:
            1. extend this class in your class argument class
            2. use add_default_dict_args or add_dict_args or add_keyword_value_args to add default key-value pair
            3. use parse_input_file or parse_input_content to parse argument content
            4. use self.args_dict to get a argument dict  OR  self.args_string_list(excluse) to get a list of argument string
        
        Other feature:
            1. set_ignore_keyword_list: to exclusive some keyword
            2. exists_key: to judge whether key exists (for safely check)  (now only support first layer check)
            3. set_keyword_alias(standard_key, [alias_key1, alias_key2, ...])
            4. can use add_dict_args to add a dict to the ArgsBase
        
        Attention:
            1. only when set dict_args and read file content can set new argument
            2. user can use args.xxx = yyy refresh the value, but only when the key exists, the dict will refresh, 
                OR it will only add a new member var on args, so please combine it with `exists_key` to check it.
    """
    ignore_list: List[str] = []
    keyword_alias_dict: Dict[str, List[str]] = {}
    writable: bool = True
    data: AttributeDict
    def __init__(self) -> None:
        self.__dict__['data'] = AttributeDict()
        self.__dict__['writable'] = False
    
    def add_default_dict_args(self, keyword_list: List[str], type_or_Callable_or_value: Union[type, Callable[..., Any], Any]):
        self.writable = True
        if type(type_or_Callable_or_value) in [type, Callable]:
            dd: DefaultDict[str, Any] = defaultdict(type_or_Callable_or_value)
            for key in keyword_list:
                self.data[key] = dd[key]
        else:
            for key in keyword_list:
                self.data[key] = copy(type_or_Callable_or_value)
        self.writable = False
    
    # @singledispatch
    def add_dict_args(self, dict: Dict[str, Any]):
        AttributeDict.merge_dict(self.data, dict)
    
    def add_keyword_value_args(self, keyword_list: List[str], value_list: List[Any]):
        if len(keyword_list) != len(value_list):
            self._raise_error(f'Internal Error: ArgsBase.add_keyword_value_args: the length is not equal in keyword_list and value_list')
        d = { k: v for k, v in zip(keyword_list, value_list)}
        self.add_dict_args(d)
    
    @staticmethod
    def _args_extractor(string: str) -> List[str]:
        pair_dict = {'(': ')', '[': ']', '{': '}', '<': '>', "'": "'", '"': '"'}
        pair_right: List[str] = []
        tmp = ''
        comment = False
        out_string_list: List[str] = []
        for chr in string:
            pair_right_last = pair_right[-1] if pair_right else ''
            if chr == '#':
                comment = True
            if chr == '\n':
                comment = False
            if comment:
                continue
            if chr in pair_dict.keys() and chr != pair_right_last:
                pair_right.append(pair_dict[chr])
            elif chr == pair_right_last:
                pair_right.pop()
            if pair_right:
                tmp += chr
            elif chr in [' ', '\n'] and tmp:  # end
                out_string_list.append(tmp.strip())
                tmp = ''
            else:
                tmp += chr.strip()
        if pair_right:
            stopper.stopMLatom(f'pair character unmatched args: "{tmp}"')
        else:
            if tmp: out_string_list.append(tmp)
        
        return out_string_list
    
    def parse_input_file(self, file: str):
        with open(file) as f:
            content = self._args_extractor(f.read())
        return content
    
    def parse_input_content(self, content: Union[List[str], str]):
        if type(content) is str:
            content = [content]
        _content = []
        for c in content:
            _content.extend(self._args_extractor(c) if '\n' not in c else [c])
        for c in _content:
            splitted = c.split('=', 1)
            if len(splitted) == 1:
                key = splitted[0]
                if key.lower() in ['help', '-help', '-h', '--help']:
                    self._print_doc()
                elif key.lower() in self.ignore_list:
                    pass
                else:
                    restored_key = self._restore_alias_key(key)
                    self.data[restored_key] = True
            elif len(splitted) == 2:
                key, value = tuple(map(lambda x: x.strip(), splitted))
                key, value = self._multi_level_dict(key, value)  # has changed the type of value
                key = self._restore_alias_key(key)
                if type(value) in [dict, AttributeDict]:
                    AttributeDict.merge_dict(self.data, {key: value})
                    continue
                else:
                    self.data[key] = value
            else:
                self._raise_error(f'error happends at your input file, error content:\n    {c}')
    
    @classmethod
    def _multi_level_dict(cls, key: str, value: Any) -> Tuple[str, Union[AttributeDict, Dict[str, Union[str, Dict[str, Any]]]]]:
        split_key = key.split('.')
        real_key = split_key[0]
        inner_value = cls._convert_to_value_with_type(copy(value))
        for sub_key in split_key[1:][::-1]:
            outter_dict = AttributeDict()
            outter_dict[sub_key] = inner_value
            inner_value = outter_dict
        return real_key, inner_value
    
    def _print_doc(self):
        try: Doc.printDoc(self.args_dict)
        except: self._warning_print('Doc not imported')
        stopper.stopMLatom('')

    @property
    def args_dict(self) -> Dict[str, Any]:
        return AttributeDict.normal_dict(self.data)
    
    def args_string_list(self, exclusive: List[Any]=[]) -> List[str]:
        l = []
        for key, value in self.args_dict.items():
            if exclusive:
                if value in exclusive:
                    continue
            if value is True:
                l.append(key)
            elif value is False:
                continue
            else:
                if type(value) is dict:
                    l.extend(self._dict_dot_expression(value, key))
                else:
                    l.append(f'{key}={value}')
                # l.append(f'{key}={value}')
        return l
    
    def _dict_dot_expression(self, d: Dict[str, Union[Dict[str, Any], Any]], key_pre_str: str='') -> List[str]:
        str_list = []
        for key, value in d.items():
            if type(value) is dict:
                str_list.extend(self._dict_dot_expression(value, f'{key_pre_str}.{key}'))
            else:
                str_list.append(f'{key_pre_str}.{key}={value}')
        return str_list
        
    
    def exists_key(self, name: str) -> bool:
        return self.exists_key(name)
    
    def set_ignore_keyword_list(self, kw_or_list: Union[str, List[str]]):
        if type(kw_or_list) in [list, tuple]:
            self.ignore_list.extend(kw_or_list)
        elif type(kw_or_list) is str:
            self.ignore_list.append(kw_or_list)
        else:
            self._raise_error(f'Internal error: "set_ignore_keyword_list" can not be {kw_or_list}')
    
    def __getattr__(self, name: str) -> Any:
        restored_name = self._restore_alias_key(name)
        return self.data[restored_name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('__') or (name in self.__dict__) or (name in dir(self)):
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)
        # restored_name = self._restore_alias_key(name)
        # if self.data.key_in_dict(restored_name):
        #     self.data[restored_name] = value
        # else:
        #     self.__dict__[name] = value
    
    def set_keyword_alias(self, standard: str, alias: Union[str, Sequence[str]]) -> None:
        if type(alias) is str:
            self.keyword_alias_dict[standard] = [alias.lower()]
        else:
            self.keyword_alias_dict[standard] = list(map(lambda x: x.lower(), alias))
    
    def _restore_alias_key(self, key: str) -> str:
        for std_key, alias_list in self.keyword_alias_dict.items():
            if key.lower().strip() in alias_list:
                return std_key
        return key
    
    def __getitem__(self, name: str) -> Any:
        restored_name = self._restore_alias_key(name)
        return self.data[restored_name]
    
    def __setitem__(self, name: str, value: Any) -> None:
        restored_name = self._restore_alias_key(name)
        if self.data.key_in_dict(restored_name):
            self.data[restored_name] = value
        else:
            self.__dict__[name] = value
    
    @staticmethod
    def _raise_error(content: str) -> None:
        print('WARNING ' * 10)
        print(' <!> %s <!>' % content)
        print('WARNING ' * 10)
        exit()
    
    @staticmethod
    def _warning_print(warning: str) -> None:
        print('-' * 80)
        print(' warning:  %s  :warning' % warning)
        print('-' * 80)
    
    @staticmethod
    def _convert_to_value_with_type(value: Any) -> Any:
        if type(value) in [dict, AttributeDict]:
            return value
        if value.strip().lower() == 'true':
            return True
        elif value.strip().lower() == 'false':
            return False
        else:
            try:
                tmp = int(value)
            except:
                try:
                    tmp = float(value)
                except:
                    return value
                else:
                    return tmp
            else:
                return tmp


class mlatom_args(ArgsBase):
    argsraw = ''
    ignore_list: List[str] = []
    keyword_alias_dict: Dict[str, List[str]] = {}
    writable: bool = True
    data: AttributeDict
    def __init__(self):
        super().__init__()
        self._task = None
        self._task_list = [
            # ML tasks
                'useMLmodel', 'createMLmodel', 'estAccMLmodel',
                'selfCorrect', 'learningCurve',
            # Data tasks
                # Conversion
                'XYZ2X', 'XYZ2SMI', 'SMI2XYZ', 
                # Analysis
                'analyze',
                # Sampling
                'sample', 'sampleFromSlices', 'mergeSlices','slice',
            # Simulation tasks 
                # PES tasks
                'geomopt', 'ts', 'freq', 'irc',
                # Dynamics
                'MD', 
                # Vibrational spectra
                'MD2vibr', 
                # ML NEA
                'crossSection', # Interfaces
                'callNXinterface', # DEVELOPMENT VERSION
                # MLQD
                'MLQD',
                # MLTPA
                'MLTPA',
                # Acive learning
                'al',
        ] # case should be exactly the same with corresponding method in MLtasks
        # Pre-defined methods
        # self._method_list = [
        #     'ODM2', 'ODM2star', 'CCSDTstarCBS', 'gfn2xtb',
        #     'AIQM1', 'AIQM1DFT', 'AIQM1DFTstar',
        #     'ani1x', 'ani2x', 'ani1ccx', 'ani1xd4', 'ani2xd4', 
        # ]
        self._method_list = list(methods.known_methods())
        # task aliases
        self.set_keyword_alias('crossSection', ['ML-NEA', 'ML_NEA', 'crossSection', 'cross-section', 'cross_section','MLNEA'])
        # self.set_keyword_alias('AIQM1DFTstar', ['AIQM1@DFT*'])
        # self.set_keyword_alias('AIQM1DFT', ['AIQM1@DFT'])
        # self.set_keyword_alias('ODM2star', ['ODM2*'])
        # self.set_keyword_alias('CCSDTstarCBS', ['CCSD(T)*/CBS'])
        # self.set_keyword_alias('ani1ccx', ['ANI-1ccx'])
        # self.set_keyword_alias('ani1x', ['ANI-1x'])
        # self.set_keyword_alias('ani2x', ['ANI-2x'])
        # self.set_keyword_alias('ani1xd4', ['ANI-1x-D4'])
        # self.set_keyword_alias('ani2xd4', ['ANI-2x-D4'])
        # self.set_keyword_alias('gfn2xtb', ['GFN2-xTB'])
        self.set_keyword_alias('AIQM1@DFT*', ['AIQM1DFTstar'])
        self.set_keyword_alias('AIQM1@DFT', ['AIQM1DFT'])
        self.set_keyword_alias('ODM2*', ['ODM2star'])
        self.set_keyword_alias('ODM3*', ['ODM3star'])
        self.set_keyword_alias('CCSD(T)*/CBS', ['CCSDTstarCBS'])
        self.set_keyword_alias('ANI-1ccx', ['ani1ccx'])
        self.set_keyword_alias('ANI-1x', ['ani1x'])
        self.set_keyword_alias('ANI-2x', ['ani2x'])
        self.set_keyword_alias('ANI-1x-D4', ['ani1xd4'])
        self.set_keyword_alias('ANI-2x-D4', ['ani2xd4'])
        self.set_keyword_alias('ANI-1xnr', ['ani1xnr'])
        self.set_keyword_alias('AIMNet2@B973c', ['aimnet2atb973c'])
        self.set_keyword_alias('AIMNet2@wb97M-D3', ['aimnet2atwb97md3'])
        self.set_keyword_alias('GFN2-xTB', ['gfn2xtb'])
        self.set_keyword_alias('MNDO/dH', ['mndodh'])
        self.set_keyword_alias('MNDO/H', ['mndoh'])
        self.set_keyword_alias('MNDO/d', ['mndod'])
        self.set_keyword_alias('SCC-DFTB', ['sccdftb'])
        self.set_keyword_alias('SCC-DFTB-heats', ['sccdftbheats'])
        self.set_keyword_alias('MINDO/3', ['mindo3'])
        self.set_keyword_alias('CNDO/2', ['cndo2'])
        # set to False
        self.add_default_dict_args(self._task_list, bool)
        self.add_default_dict_args(self._method_list, bool)
        self.add_default_dict_args(['deltaLearn', 'CVtest', 'CVopt'], bool)
        
        self.add_default_dict_args([
            # program control
            'method',
            'nthreads',
            # data IO
            'XYZfile', 'XfileIn',
            'Yfile', 'YestFile', 'Yb', 'Yt', 'YestT',
            'YgradFile', 'YgradEstFile', 'YgradB', 'YgradT', 'YgradEstT',
            'YgradXYZfile', 'YgradXYZestFile', 'YgradXYZb', 'YgradXYZt', 'YgradXYZestT',
            'HessianEstFile', 
            'charges', 'multiplicities', # ???
            # model IO
            'MLmodelIn', 'MLmodelOut',
            # model settings
            'MLmodelType', 'MLprog',
            'mndokeywords','QMprogramKeywords',
            # sampling
            'Nuse', 'Ntrain', 'Nsubtrain', 'Nvalidate', 'Ntest',
            ],
            None
        )
        # for blank strings
        self.add_default_dict_args(
            [
                'XYZfile', 'XfileIn',
                'Yfile', 'YestFile', 'Yb', 'Yt', 'YestT',
                'iTrainIn', 'iTestIn', 'iSubtrainIn', 'iValidateIn',
                'iTrainOut', 'iTestOut', 'iSubtrainOut', 'iValidateOut',
                'YgradFile', 'YgradEstFile', 'YgradB', 'YgradT', 'YgradEstT',
                'YgradXYZfile', 'YgradXYZestFile', 'YgradXYZb', 'YgradXYZt', 'YgradXYZestT',
                'MLprog', "MLmodelType",            
                'qmprog',
                'mndokeywords', 'QMprogramKeywords', 'charges', 'multiplicities',
                'optProg',
                'freqProg',
            ], ''
        )
        self.add_dict_args({
            'eqXYZfileIn': None,
            'Sampling': 'random',
            'molDescriptor': 'RE',
            'kernel': 'Gaussian', 
        })
        # selfCorrect
        self.parse_input_content([
            'nlayers=4'])
        # hyperopt
        self.parse_input_content([
            'hyperopt.max_evals=8',
            'hyperopt.algorithm=tpe',
            'hyperopt.losstype=geomean',
            'hyperopt.w_y=1',
            'hyperopt.w_ygrad=1',
            'hyperopt.points_to_evaluate=0',
        ])
        # cross validation
        self.add_dict_args({
            'NcvTestFolds': 5,
            'iCVtestPrefIn': '',
            'iCVtestPrefOut': '',
            'NcvOptFolds': 5,
            'iCVoptPrefIn': '',
            'iCVoptPrefOut': '',
        })
        # learning curve
        self.add_dict_args({
            'lcDir': 'learningCurve',
            'lcNtrains': "100,250,500,1000",
            "lcNrepeats": 5,
        })
        # geomopt/ts/freq/irc
        self.add_dict_args({
            'optXYZ': "optgeoms.xyz",
        })
        # ASE
        self.parse_input_content([
            'ase.fmax=0.02',
            'ase.steps=200',
            'ase.optimizer=',
            'ase.linear=',
            'ase.symmetrynumber='
        ])
        # SMI
        self.add_dict_args({
            'SMIin': "input.smi",
            "SMIout": "output.smi",
            "XYZout": "output.xyz",
        })
        # MD
        self.add_dict_args({
            'dt':0.1,                      # Time step; Unit: fs
            'trun':1000,                   # Length of trajectory; Unit: fs
            'initXYZ':'',                  # File containing initial geometry
            'initVXYZ':'',                 # File containing initial velocity
            'initcond':'',                 # How to generate initial condition
            'nmfile':'',                   # Gaussian ouput file containing normal modes
            'optXYZfile':'',               # File containing XYZ file with optimized geometry
            'trajH5MDout':'traj.h5',       # Output file: H5MD format
            'trajTime':'traj.t',           # Output file containing time
            'trajXYZout':'traj.xyz',       # Output file containing geometries
            'trajVXYZout':'traj.vxyz',     # Output file containing velocities
            'trajEpot':'traj.epot',        # Output file containing potential energies
            'trajEkin':'traj.ekin',        # Output file containing kinetic energies
            'trajEtot':'traj.etot',        # Output file containing total energies 
            'trajEgradXYZ':'traj.grad',    # Output file containing energy gradients
            'trajDipoles':'traj.dp',       # Output file containing dipole moments
            'trajTemperature':'traj.temp', # Output file containing instantaneous temperatures
            'trajTextOut':'traj',          # Output file
            'MLenergyUnits':'',            # Energy unit in ML model
            'MLdistanceUnits':'',          # Distance unit in ML model
            'ensemble':'nve',
            'thermostat':'',               # Thermostat
            'gamma':0.2,                   # Option for Anderson thermostat
            'initTemp':None,                # Initial temperature 
            'initEkin':None,                # Initial kinetic energy 
            'eliminateAngularMomentum':False,
            'temp':300,                    # Thermostat temperature
            'initVXYZout':'',              # Output file containing initial velocity
            'initXYZout':'',               # Output file containing initial geometry
            'NHClength':3,                 # Nose-Hoover chain length
            'Nc':3,                        # Multiple time step
            'Nys':7,                       # Number of Yoshida Suzuki steps used in NHC (1,3,5,7)
            'NHCfreq':16,
            'noang':0,
            'DOF':-6,
            'linear':0,
        })
        # geomopt output
        self.add_default_dict_args(
            [
                'printall',                # print out all information in geometry optimization
                'printmin',                # print out minimal infomation in geometry optimization        
                'dumpopttrajs'             # whether to dump optimization trajectory
            ], ''
        )
    
        self.defualt_args2pass = self.args_string_list(['', None])

    @property
    def args2pass(self):
        return [arg for arg in self.args_string_list(['', None]) if arg not in self.defualt_args2pass]

    def parse(self, argsraw):
        if len(argsraw) == 0:
            Doc.printDoc({})
            stopper.stopMLatom('At least one option should be provided')
        elif len(argsraw) == 1 and os.path.exists(argsraw[0]):
            argsraw = self.parse_input_file(argsraw[0])
            
        self.argsraw = argsraw
        self.parse_input_content(argsraw)
        
        self._post_operations()
            
    def _post_operations(self):
        if not self.MLprog:
            if self.MLmodelType :
                try: self.MLprog = default_MLprog[self.MLmodelType.lower()]
                except: stopper.stopMLatom('Unkown MLmodelType')
            else:
                self.MLprog = 'MLatomF'
        self._checkArgs()
        self._check_hyperopt()
        self._multi_lines_to_file()

    def _multi_lines_to_file(self):
        import hashlib
        for arg in self.args2pass:
            if '\n' in arg:
                key, value = arg.split('=', 1)
                tmpfile = f"{key}_{hashlib.md5(value.encode('utf-8')).hexdigest()[:6]}"
                if 'xyz' in key.lower():
                    tmpfile += '.xyz'
                with open(tmpfile, 'w') as f:
                    f.write(value.strip("'").strip('"').strip() + '\n')
                self.parse_input_content([f'{key}={tmpfile}'])

    def _check_hyperopt(self):
        self.hyperparameter_optimization = {
            'optimization_algorithm': None,
            'hyperparameters': [],
        }
        self._hyperopt_str_dict = {}
        self.hyperparameter_optimization['maximum_evaluations'] = int(self.hyperopt.max_evals)  
        for arg in self.args2pass:
            if bool(re.search('hyperopt\..+?\(.+?\)',arg)):
                key, value = arg.split('=', 1)
                self.hyperparameter_optimization['hyperparameters'].append(key.split('.')[-1])
                self._hyperopt_str_dict[key.split('.')[-1]] = value
        if self._hyperopt_str_dict:
            self.hyperparameter_optimization['optimization_algorithm'] = self.hyperopt.algorithm

    def _checkArgs(self):
        if not self._task:
            tasks = []
            for task in self._task_list:
                if self.data[task.lower()]:
                    tasks.append(task)
            if len(tasks) == 1:
                self._task = tasks[0]
            elif len(tasks) > 1:
                if self.selfCorrect:
                    self._task = 'selfCorrect'
                else:
                    if 'useMLmodel' in tasks:
                        tasks.remove('useMLmodel')
                        print('useMLmodel omitted')
                    self._task = tasks.pop(0)
                    for task in tasks:
                        self.data[task] = False
                    print(f' multiple tasks detected in the input. the first one ({self._task}) will be used')
        if self.method:
            if self.method in self._method_list:
                self.data[self.method] = True
        else:
            for method in self._method_list:
                if self.data[method]:
                    self.method = method
                    break
        if not self._task:
            if not self.method:
                Doc.printDoc({})
                stopper.stopMLatom(' a task or a method should be requested')
            else:
                self._task = 'useMLmodel'

    def copy(self, task=None, keys=None):
        new_args = mlatom_args()
        if keys is None:
            new_args.parse(self.args2pass)
        else:
            lower_keys = [key.lower() for key in keys]
            argsraw = [arg for arg in self.args_string_list(['', None]) if arg.split('=')[0].lower() in lower_keys]
            argsraw.append(task if task else self._task)
            new_args.parse(argsraw)
        return new_args
    
    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value in self._task_list:
            self.add_default_dict_args(self._task_list, bool)
            self.data[value] = True
            self._task = value
        else:
            print('unknow task, task not changed')

if __name__ == '__main__':
    try:
        args = ArgsBase()
        args.parse_input_file('inp')
    except: pass
    print(  'warning: you are directly running the module: args_class, '\
            'which should be extended by a class, not run it.')
