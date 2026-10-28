"""Test the real prediction method's input contract with a captured backend."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,unittest
import numpy as np

class Tensor:
    def __init__(self,data): self.data=np.asarray(data)
    def to(self,*args): return self

class PeriodicInput(unittest.TestCase):
    def prediction(self,pbc,cell):
        source=Path(__file__).parents[1]/'mlatom/interfaces/aimnet2_interface.py'
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='aimnet2_methods_single')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
        namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'real_predict','exec'),namespace)
        captured={}
        def backend(inputs,**kwargs):captured.update(inputs);return {}
        fake=SimpleNamespace(float='float',long='long',as_tensor=lambda x:Tensor(x),tensor=lambda x,**kw:Tensor(x))
        mol=SimpleNamespace(xyz_coordinates=np.array([[0.,0.,0.]]),atomic_numbers=np.array([1]),charge=0,multiplicity=2,pbc=pbc,cell=cell)
        with patch.dict(sys.modules,{'torch':fake}):namespace['predict'](SimpleNamespace(device='cpu',model=backend),mol)
        return captured
    def test_periodic_cell_reaches_calculator(self):
        cell=np.diag([8.,9.,10.]); result=self.prediction([True]*3,cell)
        self.assertIn('cell',result);np.testing.assert_equal(result['cell'].data,cell)
    def test_isolated_molecule_keeps_nonperiodic_input(self):
        self.assertNotIn('cell',self.prediction(None,None))
    def test_partial_periodicity_is_not_silently_promoted(self):
        with self.assertRaises(NotImplementedError):self.prediction([True,False,True],np.eye(3)*8)
    def test_invalid_periodic_cells_are_rejected(self):
        for cell in [None,np.zeros((3,3)),np.ones((2,2)),np.full((3,3),np.nan)]:
            with self.subTest(cell=cell),self.assertRaises(ValueError):self.prediction([True]*3,cell)
if __name__=='__main__':unittest.main()
