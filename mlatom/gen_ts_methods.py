import os, sys
from typing import NamedTuple, List
from . import data

class gen_ts_results(NamedTuple):
    ts:List[data.molecule] 
    avg_ts:data.molecule
    path:List[data.molecular_database]
    avg_path:data.molecular_database

def gen_ts_ects(
    reactant=None, product=None, 
    path=False,
    n_ts=10, n_path=None, avg_ts=True, avg_path=False,
    program_kwargs=None,
    working_directory=None
):
    
    if not working_directory: working_directory = os.path.abspath('./ects')
    if len(program_kwargs) == 0:
        raise ValueError('Please provide program_kwargs or program_kwargs_json for loading settings of ects model.')

    # check device and set local rank
    import torch
    if 'device' in program_kwargs: device = program_kwargs['device']
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.lower() not in ['cuda', 'cpu']:
        raise ValueError('device should be either "cuda" or "cpu".')
    if device.lower() == 'cuda':
        envdict = {'RANK':'0','WORLD_SIZE':'1','MASTER_ADDR':'localhost','MASTER_PORT':'12345','LOCAL_RANK':'0'}
        for ee in envdict:
            if ee not in os.environ:
                os.environ[ee] = envdict[ee]
                print(f'{ee} is not set in environment and will be set to default value {envdict[ee]}'); sys.stdout.flush()
    if 'LOCAL_RANK' not in os.environ: local_rank = 0
    else: local_rank = int(os.environ['LOCAL_RANK'])
    
    # check installation
    import importlib
    _find_ects = importlib.util.find_spec('EcTs')
    if _find_ects is None:
        raise ValueError('EcTs installation not found. Please install EcTs via https://github.com/AI4Reactions/ECTS')
    
    # load module
    from EcTs.comparm import GP,Loaddict2obj
    from EcTs.utils import xyz2mol
    from EcTs.graphs import RP_pair
    from EcTs.model import EcTs_Model

    # load settings for model
    Loaddict2obj(program_kwargs,GP)

    ratoms = reactant.atomic_numbers.tolist(); patoms = product.atomic_numbers.tolist()
    rcharge = reactant.charge; pcharge = product.charge
    rxyz = reactant.xyz_coordinates; pxyz = product.xyz_coordinates

    rmol = xyz2mol(ratoms,rxyz,charge=rcharge)[0]
    pmol = xyz2mol(patoms,pxyz,charge=pcharge)[0]
    rp_pair = RP_pair(rmol=rmol,pmol=pmol,idx='ects')

    ects_model = EcTs_Model(modelname="EcTs_Model",local_rank=local_rank)
    
    # create results tuple
    results_ts = [] 
    results_path = [] if path else None
    results_avg_ts = None; results_avg_path = None

    if path:
        ects_model.Sample_Path(rp_pair, path_num=n_path, savepath=working_directory)
        for np in range(n_path):
            path_moldb = data.molecular_database()
            for ii in range(2*GP.n_mid_states+1):
                path_moldb.append(data.molecule.from_xyz_file(os.path.join(working_directory, f'{local_rank}-{np}/{ii}.xyz')))
            results_path.append(path_moldb)

            results_ts.append(data.molecule.from_xyz_file(
                os.path.join(working_directory,f'{local_rank}-{np}-ts.xyz')))
    else:
        ects_model.Sample_TS(rp_pair, ts_num_per_mol=n_ts, savepath=working_directory)
        for nt in range(n_ts):
            results_ts.append(data.molecule.from_xyz_file(
                os.path.join(working_directory,f'{local_rank}-{nt}.xyz')))
    if avg_ts:
        results_avg_ts = data.molecule.from_xyz_file(os.path.join(working_directory,f'{local_rank}-average-ts.xyz'))
    
    if avg_path:
        print('Average path is not implemented yet.')

    return gen_ts_results(results_ts, results_avg_ts, results_path, results_avg_path)