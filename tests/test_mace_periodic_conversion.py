"""Exercise the exact ASE conversion expressions without loading a MACE model."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import ase

class PeriodicConversion(unittest.TestCase):
    def test_all_molecule_to_ase_conversions_preserve_boundaries(self):
        source=Path(__file__).parents[1]/'mlatom/interfaces/mace_interface.py'
        tree=ast.parse(source.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='ase' and n.func.attr=='Atoms' and any(k.arg=='positions' and isinstance(k.value,ast.Attribute) and k.value.attr=='xyz_coordinates' for k in n.keywords)]
        self.assertEqual(len(calls),2)
        for pbc in ([True]*3,[True,False,True],None):
            mol=SimpleNamespace(xyz_coordinates=np.array([[0.,0.,0.],[1.,0.,0.]]),atomic_numbers=np.array([1,1]),cell=None if pbc is None else np.diag([8.,9.,10.]),pbc=pbc)
            for call in calls:
                atoms=eval(compile(ast.Expression(call),'conversion','eval'),{'ase':ase,'mol':mol,'energy_key':None})
                np.testing.assert_array_equal(atoms.pbc,np.array(pbc if pbc is not None else [False]*3))
                np.testing.assert_allclose(atoms.cell.array,mol.cell if mol.cell is not None else np.zeros((3,3)))
                np.testing.assert_allclose(atoms.positions,mol.xyz_coordinates)

if __name__=='__main__': unittest.main()
