import os 
from ..model_cls import torch_model, downloadable_model, method_model
from .. import data, constants

class mace_methods(torch_model, downloadable_model, method_model):

    '''
    Universal foundational models with MACE. For the full list of models, please refer to https://github.com/ACEsuit/mace#pretrained-foundation-models. 

    Arguments:
        method (str): A string that specifies the method. Available choices: ``'MACE-OFF23'``, ``'MACE-OFF23s'``,``'MACE-OFF23m'``,``'MACE-OFF23l'``,``'MACE-OFF24'``,``'MACE-OFF24m'``, and the universal models for materials ``'MACE-MP-0'`` and ``'MACE-MPA-0'`` (both the medium models). Periodic boundary conditions set with ``molecule.pbc`` and ``molecule.cell`` are passed to all of them.
        device (str, optional): Indicate which device the calculation will be run on, i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        nthreads (int): define number of threads to be used during inference.

    '''

    supported_methods = ['mace-off24', 'mace-off24m', 'mace-off23', 'mace-off23s', 'mace-off23m', 'mace-off23bm', 'mace-off23l', 'mace-mp-0', 'mace-mpa-0'] # MACE-OFF series and the MACE-MP universal models
    methods_info = {
        'mace-off23s':
            ['MACE-OFF23_small.model'],
        'mace-off23m':
            ['MACE-OFF23_medium.model'],
        'mace-off23l':
            ['MACE-OFF23_large.model'],
        'mace-off23bm': 
            ['MACE-OFF23b_medium.model'],
        'mace-off23':
            ['MACE-OFF23_medium.model'],
        'mace-off24':
            ['MACE-OFF24_medium.model'],
        'mace-off24m':
            ['MACE-OFF24_medium.model'],
        'mace-mp-0':
            ['2023-12-03-mace-128-L1_epoch-199.model'],
        'mace-mpa-0':
            ['mace-mpa-0-medium.model']
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

        if self.method.lower().startswith('mace-mp'):
            from mace.calculators import mace_mp as mace_calculator # MACE ASE calculator for materials
            model_dir = 'mace_mp_model'
        else:
            from mace.calculators import mace_off as mace_calculator # MACE ASE calculator for organic molecules
            model_dir = 'mace_off_model'
        model_files = self.methods_info[self.method.lower()][0]
        # download_links = self.methods_info[self.method.lower()][1]

        mlatom_model_dir, to_download = self.check_model_path(model_dir, model_files)
        mlatom_model_path = os.path.join(mlatom_model_dir, model_files)
        # if to_download: self.download(download_links, mlatom_model_path, extract=False, flatten=False)
        if to_download:
            raise ValueError(f"MACE model {model_files} not found in {mlatom_model_dir}. Please download it manually and place it at {mlatom_model_path}")

        # mace_off() and mace_mp() set torch's PROCESS-GLOBAL default precision and never put it
        # back, so every model built later in the same session inherits it. A model that
        # casts its own inputs to single precision then fails with "mat1 and mat2 must
        # have the same dtype" - OMNI-P1 does exactly that. Ask for the precision
        # explicitly, and restore the global one so nothing outside this call sees it.
        import torch
        model_kwargs = dict(self.model_kwargs) # a copy: popping from the caller's dict made a second load() fall back to float64
        self.default_dtype = model_kwargs.pop('default_dtype', 'float64')
        saved_default_dtype = torch.get_default_dtype()
        try:
            self.model = mace_calculator(model=mlatom_model_path, device=self.device,
                                         default_dtype=self.default_dtype, **model_kwargs)
        finally:
            torch.set_default_dtype(saved_default_dtype)

    def predict(
            self,
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            calculate_energy: bool = True,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
    ):
        import torch
        from ase import Atoms
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        # The calculator builds its input tensors in torch's default precision, which load()
        # gave back to the caller - hold the model's own precision for the calculation only.
        saved_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(getattr(torch, self.default_dtype))
        try:
            for mol in molDB.molecules:
                if mol.pbc is not None and mol.pbc.any() and mol.cell is None:
                    raise ValueError('molecule.pbc is set but molecule.cell is not - a periodic calculation needs the cell')
                atoms = Atoms(numbers=mol.atomic_numbers, positions=mol.xyz_coordinates,
                              cell=mol.cell, pbc=mol.pbc if mol.pbc is not None else False)
                atoms.calc = self.model
                if calculate_energy:
                    mol.energy = atoms.get_total_energy() * constants.eV2Hartree
                if calculate_energy_gradients:
                    forces = atoms.get_forces() * constants.eV2Hartree
                    mol.add_xyz_derivative_property(-forces, xyz_derivative_property='energy_gradients')
                if calculate_hessian:
                    hess = self.model.get_hessian(atoms=atoms) * constants.eV2Hartree
                    mol.hessian = hess.reshape(len(mol)*3, len(mol)*3)
        finally:
            torch.set_default_dtype(saved_default_dtype)