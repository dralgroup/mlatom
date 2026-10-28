"""A large periodic cell must not retain every atom object at every MD step."""
import numpy as np

from mlatom.data import molecule
from mlatom.md import md


class FlatPotential:
    def predict(self, molecule, calculate_energy=True, calculate_energy_gradients=True):
        molecule.energy = 0.0
        molecule.add_xyz_vectorial_property(
            np.zeros((len(molecule.atoms), 3)), 'energy_gradients'
        )


def test_bounded_retention_matches_default_dynamics():
    xyz = '3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n'
    initial = molecule.from_xyz_string(xyz)
    initial.cell = np.eye(3) * 20.0
    initial.pbc = True
    for atom in initial.atoms:
        atom.xyz_velocities = np.array([0.01, 0.02, 0.03])
    assert 0.9 < initial.get_nuclear_masses()[1] < 1.1
    common = dict(model=FlatPotential(), molecule=initial, ensemble='NVE',
                  time_step=0.25, maximum_propagation_time=0.5)
    full = md(**common)
    bounded = md(**common, retain_trajectory=False)
    assert len(full.molecular_trajectory.steps) == 3
    assert len(bounded.molecular_trajectory.steps) == 1
    assert bounded.molecular_trajectory.steps[-1].step == 2
    np.testing.assert_allclose(
        full.molecular_trajectory.steps[-1].molecule.xyz_coordinates,
        bounded.molecular_trajectory.steps[-1].molecule.xyz_coordinates,
    )
    assert np.all(bounded.molecular_trajectory.steps[-1].molecule.pbc)
