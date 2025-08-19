import os 
from ..model_cls import torch_model, downloadable_model, method_model
from .. import data, constants

class mace_methods(torch_model, downloadable_model, method_model):

    '''
    Universal foundational models with MACE. For the full list of models, please refer to https://github.com/ACEsuit/mace#pretrained-foundation-models. 

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'MACE-OFF23'``, ``'MACE-OFF23s'``,``'MACE-OFF23m'``,``'MACE-OFF23l'``,``'MACE-OFF24'``,``'MACE-OFF24m'``.
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        nthreads (int): define number of threads to be used during inference.

    '''

    supported_methods = ['mace-off24', 'mace-off24m', 'mace-off23', 'mace-off23s', 'mace-off23m', 'mace-off23bm', 'mace-off23l'] # currently only MACE-OFF series are supported
    methods_info = {
        'mace-off23s':
            ['MACE-OFF23_small.model',"https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true"],
        'mace-off23m':
            ['MACE-OFF23_medium.model',"https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true"],
        'mace-off23l':
            ['MACE-OFF23_large.model',"https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model?raw=true"],
        'mace-off23bm': 
            ['MACE-OFF23b_medium.model', "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23b_medium.model?raw=true"],
        'mace-off23':
            ['MACE-OFF23_medium.model',"https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true"],
        'mace-off24':
            ['MACE-OFF24_medium.model', "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model?raw=true"],
        'mace-off24m':
            ['MACE-OFF24_medium.model', "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model?raw=true"]
    }

    def __init__(self, method=None, device=None, model_kwargs=None, nthreads=1):

        import torch 
        self.method = method
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.nthreads = nthreads
        if model_kwargs is None: self.model_kwargs = {}
        else: self.model_kwargs = model_kwargs
        self.load()

    def load(self):

        import importlib
        _find_ects = importlib.util.find_spec('mace')
        if _find_ects is None:
            raise ValueError('MACE installation not found. Please install MACE via https://github.com/ACEsuit/mace')

        from mace.calculators import mace_off # MACE ASE calculator
        model_dir = 'mace_off_model'
        model_files = self.methods_info[self.method.lower()][0]
        download_links = self.methods_info[self.method.lower()][1]

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        mlatom_model_path = os.path.join(mlatom_model_dir, model_files)
        if to_download: self.download(download_links, mlatom_model_path, extract=False, flatten=False)

        self.model = mace_off(model=mlatom_model_path, device=self.device, **self.model_kwargs)

    def predict(
            self,
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            calculate_energy: bool = True,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
    ):
        from ase import Atoms
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        for mol in molDB.molecules:
            atoms = Atoms(numbers=mol.atomic_numbers, positions=mol.xyz_coordinates)
            atoms.calc = self.model
            if calculate_energy:
                mol.energy = atoms.get_total_energy() * constants.eV2Hartree
            if calculate_energy_gradients:
                forces = atoms.get_forces() * constants.eV2Hartree
                mol.add_xyz_derivative_property(-forces, xyz_derivative_property='energy_gradients')
            if calculate_hessian:
                hess = self.model.get_hessian(atoms=atoms) * constants.eV2Hartree
                mol.hessian = hess.reshape(len(mol)*3, len(mol)*3)