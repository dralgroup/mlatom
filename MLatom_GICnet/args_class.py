#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! args_class: handling input of MLatom                                      ! 
  ! Implementations by: Bao-Xin Xue                                           ! 
  !---------------------------------------------------------------------------! 
'''

from collections import defaultdict, UserDict
from os import replace
from typing import Callable, DefaultDict, Sequence, Tuple, Union
from typing import List, Tuple, Dict, Set, Any
from copy import copy
try:
    from . import stopper
    from .doc import Doc
except:
    import stopper
    from doc import Doc

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
        if name.startswith('__') or (name in self.__dict__):
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
        pair_level = 0
        pair_right: List[str] = []
        tmp = ''
        prev = ''
        out_string_list: List[str] = []
        for chr in string:
            pair_right_last = pair_right[-1] if pair_right else ''
            if chr in pair_dict.keys() and chr != pair_right_last:
                pair_level += 1
                pair_right.append(pair_dict[chr])
            elif chr == pair_right_last:
                pair_level -= 1
                pair_right.pop()
            elif chr == ' ' and pair_level == 0 and prev.strip():  # end
                out_string_list.append(tmp)
                tmp = ''
                prev = chr
                continue
            if '"' in pair_right or "'" in pair_right:
                tmp += chr
            else:
                tmp += chr.strip()
            prev = chr
        if pair_level:
            stopper.stopMLatom(f'pair character unmatched args: "{tmp}"')
        else:
            if tmp: out_string_list.append(tmp)
        return out_string_list
    
    def parse_input_file(self, file: str):
        try:
            with open(file) as f:
                content = f.read().splitlines()
        except:
            self._raise_error(f'can not open file {file}! exit!'); exit()
        self.parse_input_content(content)
    
    def parse_input_content(self, content: Union[List[str], str]):
        help = False
        if type(content) is str:
            content = content.splitlines()
        for c in self._clean_input_content(content):
            splitted = c.split('=')
            if len(splitted) == 1:
                key = splitted[0]
                if key.lower() in ['help', '-help', '-h', '--help']:
                    help = True
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
        if help:
            self._print_doc()
    
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
    
    @classmethod
    def _clean_input_content(cls, content: List[str]) -> List[str]:
        to_return: List[str] = []
        for c in content:
            usable = c.split('#')[0].strip()
            while usable.find(' =') != -1:
                usable = usable.replace(' =', '=')
            while usable.find('= ') != -1:
                usable = usable.replace('= ', '=')
            to_return.extend(cls._args_extractor(usable))
        return to_return
    
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


if __name__ == '__main__':
    try:
        args = ArgsBase()
        args.parse_input_file('inp')
    except: pass
    print(  'warning: you are directly running the module: args_class, '\
            'which should be extended by a class, not run it.')
