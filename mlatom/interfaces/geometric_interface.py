import geometric 
from .. import constants, data, xyz
import sys, os
import logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if logger.name in ['geometric.nifty', 'geometric']:
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

'''
TODO 
- [] Check how to use self-generated hessian at each step
- [] add error handeler in irc
'''

class MLatomEngine(geometric.engine.Engine):

    def __init__(self, mlatom_molecule, model, model_predict_kwargs, save_traj=False):
        
        # coord unit for input molecule is Angstroms
        self.mlatom_molecule = mlatom_molecule 
        self.model = model
        self.model_predict_kwargs = model_predict_kwargs

        self.save_traj = save_traj
        if self.save_traj: self.trajdb = [] # xinxin: ugly imp to keep intermediate information

        # build geometric molecule
        molecule = geometric.molecule.Molecule()
        molecule.elem = mlatom_molecule.element_symbols.tolist()
        molecule.xyzs = [mlatom_molecule.xyz_coordinates]
        molecule.charge = mlatom_molecule.charge 
        molecule.mult = mlatom_molecule.multiplicity
        molecule.build_topology(force_bonds=False)
        super(MLatomEngine, self).__init__(molecule)

    def calc_new(self, coords, dirname):
        mol = self.mlatom_molecule.copy(atomic_labels=[],molecular_labels=[])
        updated_coord = coords.reshape(-1,3)*constants.Bohr2Angstrom

        self.M.xyzs[0] = updated_coord
        mol.update_xyz_vectorial_properties('xyz_coordinates', updated_coord)
        self.model._predict_geomopt(molecule=mol, **self.model_predict_kwargs)

        if self.save_traj:
            self.trajdb.append(mol) # xinxin: ugly imp to keep intermediate information  

        energy = mol.energy
        gradients = mol.get_energy_gradients()/constants.Angstrom2Bohr
        return {"energy": energy, "gradient": gradients.ravel()}
    
def optimize_geometry(
        model=None, 
        model_predict_kwargs:dict = {},
        molecule:data.molecule = None,
        ts:bool=False,
        maximum_number_of_steps:int = 200, 
        geometric_kwargs:dict = {}):
    
    import tempfile
    tmpdir = tempfile.TemporaryDirectory()
    tmpdirname = os.path.abspath(tmpdir.name)
    
    if ts:
        print('Start calculating hessian on trainsition state as first step ...'); sys.stdout.flush()
        model.predict(molecule=molecule, calculate_hessian=True)
        print('Finish calculating hessian and start optimizing geometry ...'); sys.stdout.flush()

        hess = molecule.hessian
        hessdir = f'{tmpdirname}.tmp/hessian'
        if not os.path.exists(hessdir):
            os.makedirs(hessdir)

        import numpy as np 
        np.savetxt(f'{hessdir}/hessian.txt',hess)
        molecule.write_file_with_xyz_coordinates(f'{hessdir}/coords.xyz')

    mlatom_engine = MLatomEngine(molecule, model, model_predict_kwargs, save_traj=True)
    
    try:
        results = geometric.optimize.run_optimizer(
            customengine=mlatom_engine, input=tmpdirname, transition=ts, 
            maxiter=maximum_number_of_steps, **geometric_kwargs)
    except Exception as ex:
        if type(ex) == geometric.errors.GeomOptNotConvergedError:
            print('Warning: Geometry optimization with geometric failed to converge.The last geometry will be used as the optimized molecule')
            optimization_trajectory = data.molecular_trajectory(
                steps=[data.molecular_trajectory_step(step=istep, molecule=imol) for istep, imol in enumerate(mlatom_engine.trajdb)])
            return optimization_trajectory, True
        else:
            print('Warning: Geometry optimization with geometric failed.')
        print('The initial geometry will be used as the optimized molecule.')
        return None, False

    tmpdir.cleanup()
    optimization_trajectory = results_to_traj(results, None, mlatom_engine.trajdb)
    return optimization_trajectory, True

def results_to_traj(results, molecule=None, trajdb=None):
    traj_xyz = results.xyzs 
    traj_energies = results.qm_energies
    traj_grads = results.qm_grads
    traj_steps = [float(com.split()[1]) for com in results.comms] # for irc

    if trajdb is not None: 
        # for geometry optimization
        # xinxin: there are some potential bugs here and will fix if any people meet it
        if len(traj_xyz) != len(trajdb): # some steps are rejected here and try to align them
            optimization_trajectory = data.molecular_database()
            i_mol_result = 0
            for imol in trajdb:
                mol_result = imol.copy()
                mol_result.xyz_coordinates = traj_xyz[i_mol_result]
                if xyz.rmsd(imol, mol_result) < 1e-8:
                    imol.iteration = traj_steps[i_mol_result]
                    optimization_trajectory.molecules.append(imol)
                    i_mol_result += 1
        else:
            optimization_trajectory = data.molecular_database(molecules=trajdb)
            optimization_trajectory.add_scalar_properties(traj_steps, "iteration")

        optimization_trajectory = data.molecular_trajectory(
        steps=[data.molecular_trajectory_step(step=istep, molecule=imol) for istep, imol in enumerate(optimization_trajectory)])
    
    else: # IRC
        optimization_trajectory = data.molecular_trajectory()
        for istep in range(len(traj_xyz)):
            step_molecule = molecule.copy(
                atomic_labels=[], molecular_labels=[]
            )
            step_molecule.xyz_coordinates = traj_xyz[istep]
            step_molecule.energy = traj_energies[istep]
            step_molecule.add_xyz_derivative_property(traj_grads[istep], 'energy_gradient')
            step_molecule.iteration = traj_steps[istep] # for irc
            optimization_trajectory.steps.append(
                data.molecular_trajectory_step(
                    step=istep, molecule=step_molecule
                )
            )

    return optimization_trajectory

def generate_irc(
    model, molecule, model_predict_kwargs, working_directory, direction, program_kwargs, verbose
):
    
    import tempfile
    import numpy as np 
    
    tmpdir = tempfile.TemporaryDirectory()
    tmpdirname = os.path.abspath(tmpdir.name)

    ### generate hessian first
    hess = molecule.hessian
    hessdir = f'{tmpdirname}.tmp/hessian'
    if not os.path.exists(hessdir): os.makedirs(hessdir)
    np.savetxt(f'{hessdir}/hessian.txt',hess)
    molecule.write_file_with_xyz_coordinates(f'{hessdir}/coords.xyz')

    ### start irc generation
    default_model_predict_kwargs = {"calculate_energy":True, "calculate_energy_gradients":True}
    default_model_predict_kwargs.update(model_predict_kwargs)
    mlatom_engine = MLatomEngine(molecule, model, default_model_predict_kwargs)
    results = geometric.optimize.run_optimizer(
        customengine=mlatom_engine, input=tmpdirname, irc=True, **program_kwargs)
    
    trajdb = results_to_traj(results, molecule).to_database()

    ### get reaction coordinates from log file
    # deprecate the code for now which might be useful in the future - because geometric will perform optimization in the last few steps and there is no reaction coordination information in that case.
    # from itertools import accumulate
    # geometric_log = open(tmpdirname+'.log', 'r').readlines()
    # step_indexs = []; step_sizes = []
    # for isll, ll in enumerate(geometric_log):
    #     if "Step" in ll:
    #         reject = False
    #         for iell in range(isll, len(geometric_log)):
    #             if "Rejecting step" in geometric_log[iell]:
    #                 reject = True 
    #             if "Total step dy" in geometric_log[iell]:
    #                 step_size = float(geometric_log[iell].split()[-1])
    #                 break 
    #         if not reject:
    #             step_index = int(ll.split()[1])
    #         else: continue
    #         step_indexs.append(step_index)
    #         step_sizes.append(step_size)
    # step_indexs = np.array(step_indexs); step_sizes = np.array(step_sizes)
    # ts_index = np.argwhere(step_indexs==0).flatten()
    # forward_step_sizes = np.array(list(accumulate(step_sizes[0:ts_index[1]])))
    # forward_step_sizes = -forward_step_sizes[::-1]
    # backward_step_sizes = np.array(list(accumulate(step_sizes[ts_index[1]:])))

    # directly use index for reaction coordinates  
    rcoords = trajdb.get_properties('iteration')
    ts_index = np.argwhere(rcoords==0).flatten()[0].item()
    trajdb.add_scalar_properties(rcoords, 'reaction_coordinates')

    ftrajdb = trajdb[:ts_index+1].copy()
    ftrajdb.molecules = ftrajdb.molecules[::-1]
    btrajdb = trajdb[ts_index:].copy()
    
    return ftrajdb, btrajdb