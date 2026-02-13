"""
Interface for pysisyphus

Link to tutorial: https://pysisyphus.readthedocs.io/en/latest/overview.html
Link to github: https://github.com/eljost/pysisyphus

NOTE:

- The energy, gradients and hessian are handled separatedly in pysisyphus, which is not good for those methods that will calculate energy and gradients together.
> - pysisphus stores charge and multiplicity into calculator instead of geometry. 

WARNING:

potential problem: if the initial step (whatever irc or other opt task) doesn't retrieve all three results from TS, it will be kept to later steps and leads to faults. Because the energy retriever doesn't tell if it's TS or later steps, there is no better way to let pysisphus to use existing energy, gradients and hessian.
"""


from .. import constants, data
import numpy as np
import importlib, os, logging, sys
from datetime import datetime as dt
import tempfile
from typing import List

_find_pysisyphus = importlib.util.find_spec('pysisyphus')
if _find_pysisyphus is None:
    raise ValueError('pysisphus installation not found. Please install pysisphus via https://github.com/eljost/pysisyphus')
from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.Geometry import Geometry
from pysisyphus.run import IRC_DICT

class mlatom_calculator(Calculator):

    mlatom_results = {}

    def __init__(self, 
                 model=None, model_predict_kwargs=None,
                 init_hessian:np.ndarray=None, init_gradients:np.ndarray=None, init_energy:float=None, 
                 working_directory:str=None, reset:bool=True, **kwargs): # kwargs give charge and multiplicity

        self.model = model 
        self.working_directory = os.path.join(working_directory, "calc") # calculation file to calcs directory
        self.model_predict_kwargs = model_predict_kwargs
        kwargs["out_dir"] = self.working_directory


        # check if any property in the transition state
        if init_hessian is not None:
            self.mlatom_results["hessian"] = init_hessian / (constants.Angstrom2Bohr**2)
        if init_gradients is not None:
            self.mlatom_results["forces"] = -init_gradients / constants.Angstrom2Bohr
        if init_energy is not None:
            self.mlatom_results["energy"] = init_energy
        
        logger = logging.getLogger('calculator'); logger.handlers = [] # remove log file
        super().__init__(**kwargs)

    def run(self, atoms, coords, model_predict_kwargs):

        # load geometry
        coords = coords.reshape(-1, 3) * constants.Bohr2Angstrom
        species = np.array([data.element_symbol2atomic_number[aa.upper()] for aa in atoms])
        mlatom_mol = data.molecule.from_numpy(
            coordinates=coords,
            species=species
        )
        mlatom_mol.charge = self.charge; mlatom_mol.multiplicity = self.mult

        # calculate
        self.model.predict(molecule=mlatom_mol, **model_predict_kwargs)

        # store results
        if model_predict_kwargs['calculate_energy']: 
            self.mlatom_results["energy"] = mlatom_mol.energy
        if model_predict_kwargs['calculate_energy_gradients']:
            self.mlatom_results["forces"] = -mlatom_mol.get_energy_gradients() / constants.Angstrom2Bohr
        if model_predict_kwargs['calculate_hessian']:
            self.mlatom_results["hessian"] = mlatom_mol.hessian / (constants.Angstrom2Bohr**2)

    def get_energy(self, atoms, coords):
        if "energy" not in self.mlatom_results:
            model_predict_kwargs = {
                "calculate_energy":True, 
                "calculate_energy_gradients":False,
                "calculate_hessian":False}
            model_predict_kwargs.update(self.model_predict_kwargs)
            self.run(atoms, coords, model_predict_kwargs) 
        energy = self.mlatom_results["energy"]
        del self.mlatom_results["energy"]
        return {"energy": energy}

    def get_forces(self, atoms, coords):
        if "forces" not in self.mlatom_results:
            model_predict_kwargs = {
                "calculate_energy":False, 
                "calculate_energy_gradients":True,
                "calculate_hessian":False}
            model_predict_kwargs.update(self.model_predict_kwargs)
            self.run(atoms, coords, model_predict_kwargs)
        forces = self.mlatom_results["forces"]
        forces = forces.flatten()
        del self.mlatom_results["forces"]
        return {"forces": forces}

    def get_hessian(self, atoms, coords):
        if "hessian" not in self.mlatom_results:
            model_predict_kwargs = {
                "calculate_energy":False, 
                "calculate_energy_gradients":False,
                "calculate_hessian":True}
            model_predict_kwargs.update(self.model_predict_kwargs)
            self.run(atoms, coords, model_predict_kwargs)
        hessian = self.mlatom_results["hessian"]
        del self.mlatom_results["hessian"]
        return {"hessian": hessian}

def generate_irc(
    model, molecule, model_predict_kwargs, working_directory, direction, program_kwargs, verbose
):
    
    irc_kwargs = {"forward":False, "backward":False,
        "hessian_init":'calc'} # we always use hessian from mlatom calculation instead of estimated one.
    
    if direction.lower() in ["forward", "backward"]:
        irc_kwargs.update({direction.lower():True})
    if direction.lower() == 'both':
        irc_kwargs['forward'] = True; irc_kwargs['backward'] = True
    
    program_kwargs_copy = program_kwargs.copy()
    algorithm = program_kwargs_copy.pop('algorithm', 'eulerpc')

    irc_kwargs.update(program_kwargs_copy)

    # pysisyphus use BOHR as the distance unit for geometry
    atoms = molecule.atomic_numbers; coords = molecule.xyz_coordinates * constants.Angstrom2Bohr
    geom = Geometry(atoms, coords.flatten(), coord_type="cart")

    # check if any property in the transition state
    init_energy=None; init_gradients=None; init_hessian=None
    if 'energy' in molecule.__dict__: init_energy = molecule.energy
    if 'energy_gradients' in molecule.atoms[0].__dict__: init_gradients = molecule.get_energy_gradients()
    if 'hessian' in molecule.__dict__: init_hessian = molecule.hessian

    mlatom_calc = mlatom_calculator(
        model=model,
        model_predict_kwargs=model_predict_kwargs,    
        init_energy=init_energy,
        init_gradients=init_gradients,
        init_hessian=init_hessian,
        working_directory=working_directory,
        charge=molecule.charge, mult=molecule.multiplicity)
    geom.set_calculator(mlatom_calc, clear=False)

    with tempfile.TemporaryDirectory() as tmpdirname:
        import contextlib, io
        irc_kwargs.update({"out_dir":tmpdirname}) # move output files of pysisphus to tmp
        irc_logger = logging.getLogger('irc'); irc_logger.handlers = []  # remove log file
        irc = IRC_DICT[algorithm.lower()](geometry=geom, **irc_kwargs)
        if not verbose:
            with contextlib.redirect_stdout(io.StringIO()): # move all the printed messages from pysisyphus to null
                irc.run()
        else:
            print("Information from pysisyphus:\n")
            irc.run()
            print("\nEnd of information from pysisyphus:\n")
            
    # retrieve results
    # the order of traj from pysisyphus is: forward (reversed) - TS - backward
    # the order of gaussian is: backward (reversed) - TS - forward
    results = irc.get_full_irc_data()
    rcoords = results['lengths'] # BOHR in mw coords
    energies = results["energies"]
    coords = results["coords"] * constants.Bohr2Angstrom
    coords = coords.reshape(coords.shape[0], -1, 3)
    gradients = results['gradients'] / constants.Bohr2Angstrom

    # to molecular database
    trajdb = data.molecular_database(
        molecules=[molecule.copy() for _ in range(len(rcoords))])
    trajdb.xyz_coordinates = coords
    trajdb.add_scalar_properties(energies,"energy")
    trajdb.add_xyz_derivative_properties(gradients,xyz_derivative_property="energy_gradients")
    trajdb.add_scalar_properties(rcoords, "reaction_coordinates")

    if direction.lower() == 'forward':
        trajdb.molecules = trajdb.molecules[::-1]
    elif direction.lower() == 'backward':
        trajdb.add_scalar_properties(
            -trajdb.get_properties("reaction_coordinates"), "reaction_coordinates")
    else:
        trajdb.molecules = trajdb.molecules[::-1]
    
    return trajdb