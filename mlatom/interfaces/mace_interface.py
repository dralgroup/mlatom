'''
.. code-block::

  !---------------------------------------------------------------------------! 
  ! Interface_MACE: Interface between MACE and MLatom                         ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''

from __future__ import annotations
from typing import Any, Union, Dict, List, Callable, Tuple
from .. import data
from .. import models
from .. import constants
from ..decorators import doc_inherit

import os
import uuid
import ast
import json
import logging
from typing import Optional

import numpy as np
import torch.nn.functional
import ase
from e3nn import o3
from torch.optim.swa_utils import SWALR, AveragedModel
from torch_ema import ExponentialMovingAverage

import mace
from mace import data as mace_data
from mace import modules, tools
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.checkpoint import CheckpointBuilder, CheckpointIO
from mace.tools.scripts_utils import (
    SubsetCollection,
    LRScheduler,
    create_error_table,
)

def load_from_molDB(
    molecular_database: data.molecular_database,
    config_type_weights: Dict,
    energy_key: str = "",
    gradients_key: str = "",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], mace_data.Configurations]:
    atoms_list = []
    for mol in molecular_database:
        atoms_list.append(ase.Atoms(positions=mol.xyz_coordinates, numbers=mol.atomic_numbers, info={'energy': getattr(mol, energy_key) * constants.Hartree2eV} if energy_key else None))
        if gradients_key:
            atoms_list[-1].new_array('forces', -1 * mol.get_xyz_vectorial_properties(gradients_key) * constants.Hartree2eV)

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1:
                isolated_atom_config = atoms.info.get("config_type") == "IsolatedAtom"
                if isolated_atom_config:
                    if energy_key in atoms.info.keys():
                        atomic_energies_dict[
                            atoms.get_atomic_numbers()[0]
                        ] = atoms.info[energy_key]
                    else:
                        logging.warning(
                            f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                            "but does not contain an energy."
                        )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

    configs = mace_data.config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
    )
    return atomic_energies_dict, configs

def get_dataset_from_molDB(
    train_db: data.molecular_database,
    valid_db: data.molecular_database,
    valid_fraction: float = 0.1,
    config_type_weights: Dict = {"Default": 1.0},
    energy_key: str = "",
    gradients_key: str = "",
) -> Tuple[SubsetCollection, Optional[Dict[int, float]]]:
    """Load training and test dataset from molecular database"""
    
    atomic_energies_dict, train_configs = load_from_molDB(
        molecular_database=train_db,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        gradients_key=gradients_key,
        extract_atomic_energies=True,
    )
    _, valid_configs = load_from_molDB(
        molecular_database=valid_db,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        gradients_key=gradients_key,
        extract_atomic_energies=False,
    )
    
    return (
        SubsetCollection(train=train_configs, valid=valid_configs, tests=[]),
        atomic_energies_dict,
    )

class mace(models.ml_model, models.torch_model):
    '''
    Create an `MACE <https://doi.org/10.48550/arXiv.2206.07697>`_  model object. 
    
    Interfaces to `MACE program <https://github.com/ACEsuit/mace>`_.

    Arguments:
        model_file (str, optional): The filename that the model to be saved with or loaded from.
        device (str, optional): Indicate which device the calculation will be run on. i.e. 'cpu' for CPU, 'cuda' for Nvidia GPUs. When not speficied, it will try to use CUDA if there exists valid ``CUDA_VISIBLE_DEVICES`` in the environ of system.
        hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        verbose (int, optional): 0 for silence, 1 for verbosity.
    '''
    hyperparameters = models.hyperparameters({
        #### Training ####
        'batch_size':           models.hyperparameter(value=10, minval=1, maxval=1024, optimization_space='linear', dtype=int),
        'valid_batch_size':     models.hyperparameter(value=10, minval=1, maxval=1024, optimization_space='linear', dtype=int),
        'max_num_epochs':     models.hyperparameter(value=2048, minval=1, maxval=4096, optimization_space='linear', dtype=int),
        'r_max':                models.hyperparameter(value=5.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'energy_weight':                models.hyperparameter(value=1.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'forces_weight':                models.hyperparameter(value=100.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'swa_energy_weight':                models.hyperparameter(value=1000.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'swa_forces_weight':                models.hyperparameter(value=100.0, minval=1, maxval=16, optimization_space='linear', dtype=float),
        'num_radial_basis':     models.hyperparameter(value=8, minval=1, maxval=32, optimization_space='linear', dtype=int),
        'num_cutoff_basis':     models.hyperparameter(value=5, minval=1, maxval=32, optimization_space='linear', dtype=int),
        'max_ell':              models.hyperparameter(value=3, minval=1, maxval=4, optimization_space='linear', dtype=int),
        'correlation':              models.hyperparameter(value=3, minval=1, maxval=4, optimization_space='linear', dtype=int),
        'radial_MLP':          models.hyperparameter(value=[64, 64, 64]),
        'radial_type':          models.hyperparameter(value='bessel', choices=["bessel", "gaussian"], dtype=str),
        'interaction':          models.hyperparameter(value='RealAgnosticResidualInteractionBlock', choices=["RealAgnosticResidualInteractionBlock", "RealAgnosticAttResidualInteractionBlock", "RealAgnosticInteractionBlock"], dtype=str),
        'interaction_first':          models.hyperparameter(value='RealAgnosticResidualInteractionBlock', choices=["RealAgnosticResidualInteractionBlock", "RealAgnosticInteractionBlock"], dtype=str),
        'scaling':          models.hyperparameter(value='rms_forces_scaling', choices=["std_scaling", "rms_forces_scaling", "no_scaling"], dtype=str),
        'loss':          models.hyperparameter(value='rms_forces_scaling', choices=["weighted", "forces_only"], dtype=str),
        'gate':          models.hyperparameter(value='silu', choices=["silu", "tanh", "abs", "None"], dtype=str),
        'num_interactions':              models.hyperparameter(value=2, minval=1, maxval=4, optimization_space='linear', dtype=int),
        'avg_num_neighbors':              models.hyperparameter(value=1.0, minval=0.1, maxval=4, optimization_space='linear', dtype=float),
        'num_polynomial_cutoff':     models.hyperparameter(value=10, minval=1, maxval=1024, optimization_space='linear', dtype=float),
        'weight_decay':     models.hyperparameter(value=5e-7, minval=1e-7, maxval=1e-6, optimization_space='linear', dtype=float),
        'lr':     models.hyperparameter(value=0.01, minval=1e-4, maxval=0.1, optimization_space='linear', dtype=float),
        'amsgrad':     models.hyperparameter(value=True, dtype=bool),
        'scheduler':     models.hyperparameter(value="ReduceLROnPlateau"),
        'lr_factor':     models.hyperparameter(value=0.8),
        'scheduler_patience':     models.hyperparameter(value=50),
        'patience':     models.hyperparameter(value=2048),
        'lr_scheduler_gamma':     models.hyperparameter(value=0.9993),
        'clip_grad':     models.hyperparameter(value=10.0),
        'error_table':     models.hyperparameter(value="TotalRMSE"),
        'save_cpu':     models.hyperparameter(value=False),
        'wandb':     models.hyperparameter(value=False),
        'wandb_name':     models.hyperparameter(value=""),
        'wandb_entity':     models.hyperparameter(value=""),
        'wandb_project':     models.hyperparameter(value=""),
        'wandb_log_hypers':     models.hyperparameter(value=[
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
        ],),
        'wandb':     models.hyperparameter(value=False),
        'ema':     models.hyperparameter(value=False),
        'ema_decay':     models.hyperparameter(value=0.99),
        'ema':     models.hyperparameter(value=False),
        'start_swa':     models.hyperparameter(value=None),
        'max_L':     models.hyperparameter(value=None),
        'restart_latest':     models.hyperparameter(value=False),
        'checkpoints_dir':     models.hyperparameter(value='MACE_ckpt'),
        'eval_interval':     models.hyperparameter(value=2),
        'swa_lr':     models.hyperparameter(value=1e-3),
        'swa_dipole_weight':     models.hyperparameter(value=1.0),
        'swa_virials_weight':     models.hyperparameter(value=10.0),
        'swa_stress_weight':     models.hyperparameter(value=10.0),
        'dipole_weight':     models.hyperparameter(value=1.0),
        'virials_weight':     models.hyperparameter(value=10.0),
        'stress_weight':     models.hyperparameter(value=10.0),
        'optimizer':     models.hyperparameter(value='adam'),        
        'E0s':     models.hyperparameter(value='average'),
        'num_channels':     models.hyperparameter(value=None),
        'hidden_irreps':     models.hyperparameter(value='128x0e + 128x1o'),
        'MLP_irreps':     models.hyperparameter(value='16x0e'),
        'compute_avg_num_neighbors': models.hyperparameter(value=True),
        'compute_forces': models.hyperparameter(value=True),
        'compute_stress': models.hyperparameter(value=False),
        'keep_checkpoints': models.hyperparameter(value=False),
        'swa': models.hyperparameter(value=False),
        'config_type_weights': models.hyperparameter(value={"Default": 1.0}),
        'model': models.hyperparameter(value='MACE'),
        'default_dtype': models.hyperparameter(value='float64'),
        'seed': models.hyperparameter(value=1234),
        'log_dir': models.hyperparameter(value='MACE_logs'),
        'log_level': models.hyperparameter(value='INFO'),
    })
    
    def __init__(self, model_file: str = None, device: str = None, hyperparameters: Union[Dict[str,Any], models.hyperparameters]={}, verbose=True):
        self.verbose = verbose
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.hyperparameters.default_dtype = 'float32'
        self.device = tools.init_device(device)
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        if not self.verbose:
            self.hyperparameters.log_level = 'ERROR'
        if model_file: 
            if os.path.isfile(model_file):
                self.load(model_file)
            else:
                if self.verbose: print(f'the trained MACE model will be saved in {model_file}')
            self.model_file = model_file
        else:
            self.model_file = f'mace_{str(uuid.uuid4())}.pt'
            
    def parse_args(self, args):
        super().parse_args(args)
        for hyperparam in self.hyperparameters:
            if hyperparam in args.hyperparameter_optimization['hyperparameters']:
                self.parse_hyperparameter_optimization(args, hyperparam)
            elif hyperparam in args.data:
                self.hyperparameters[hyperparam].value = args.data[hyperparam]
            elif 'mace' in args.data and hyperparam in args.mace.data:
                self.hyperparameters[hyperparam].value = args.mace.data[hyperparam]
    
    def load(self, model_file: str):
        self.model = torch.load(model_file, map_location=self.device).to(self.device)
        
    def save(self, model_file: str):
        if not model_file:
            model_file =f'mace_{str(uuid.uuid4())}.pt'
            self.model_file = model_file
        torch.save(self.model, model_file)
    
    @doc_inherit        
    def train(
        self, 
        molecular_database: data.molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database: Union[data.molecular_database, str, None] = 'sample_from_molecular_database',
        spliting_ratio: float = 0.8,
        hyperparameters: Union[Dict[str,Any], models.hyperparameters] = {},
    ) -> None:
        r'''
            validation_molecular_database (:class:`mlatom.data.molecular_database` | str, optional): Explicitly defines the database for validation, or use ``'sample_from_molecular_database'`` to make it sampled from the training set.
            hyperparameters (Dict[str, Any] | :class:`mlatom.models.hyperparameters`, optional): Updates the hyperparameters of the model with provided.
        '''
        self.hyperparameters.update(hyperparameters)
            
        
        args = self.hyperparameters
        
        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            molecular_database, validation_molecular_database = [molecular_database[i_split] for i_split in np.split(idx, [int(len(idx) * spliting_ratio)])]
        elif not validation_molecular_database:
            raise NotImplementedError("please specify validation_molecular_database or set it to 'sample_from_molecular_database'")
        
        tag = tools.get_tag(name=args.model, seed=args.seed)
        tools.set_seeds(args.seed)
        if self.verbose: tools.setup_logger(level=args.log_level, tag=tag, directory=args.log_dir)
        
        config_type_weights = ast.literal_eval(args.config_type_weights) if type(args.config_type_weights) == 'str' else  args.config_type_weights
        
        tools.set_default_dtype(args.default_dtype)
            
        collections, self.atomic_energies_dict = get_dataset_from_molDB(
            train_db=molecular_database,
            valid_db=validation_molecular_database,
            config_type_weights=config_type_weights,
            energy_key=property_to_learn,
            gradients_key=xyz_derivative_property_to_learn,
        )
        
        logging.info(
            f"Total number of configurations: train={len(collections.train)}, valid={len(collections.valid)}, "
            f"tests=[{', '.join([name + ': ' + str(len(test_configs)) for name, test_configs in collections.tests])}]"
        )
        
        self.z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
        # yapf: enable
        logging.info(self.z_table)
        if args.model == "AtomicDipolesMACE":
            atomic_energies = None
            dipole_only = True
            compute_dipole = True
            compute_energy = False
            args.compute_forces = False
            compute_virials = False
            args.compute_stress = False
        else:
            dipole_only = False
            if args.model == "EnergyDipolesMACE":
                compute_dipole = True
                compute_energy = True
                args.compute_forces = True
                compute_virials = False
                args.compute_stress = False
            else:
                compute_energy = True
                compute_dipole = False
            if self.atomic_energies_dict is None or len(self.atomic_energies_dict) == 0:
                if args.E0s is not None:
                    logging.info(
                        "Atomic Energies not in training file, using command line argument E0s"
                    )
                    if isinstance(args.E0s, dict):
                        self.atomic_energies_dict = args.E0s
                    elif type(args.E0s) == str and args.E0s != 'average':
                        self.atomic_energies_dict = ast.literal_eval(args.E0s)
                    else:
                        logging.info(
                            "Computing average Atomic Energies using least squares regression"
                        )
                        self.atomic_energies_dict = mace_data.compute_average_E0s(
                            collections.train, self.z_table
                        )
                else:
                    raise RuntimeError(
                        "E0s not found in training file and not specified in command line"
                    )
            atomic_energies: np.ndarray = np.array(
                [self.atomic_energies_dict[z] for z in self.z_table.zs]
            )
            logging.info(f"Atomic energies: {atomic_energies.tolist()}")

        train_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_data.AtomicData.from_config(config, z_table=self.z_table, cutoff=args.r_max)
                for config in collections.train
            ],
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        valid_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_data.AtomicData.from_config(config, z_table=self.z_table, cutoff=args.r_max)
                for config in collections.valid
            ],
            batch_size=args.valid_batch_size,
            shuffle=False,
            drop_last=False,
        )

        loss_fn: torch.nn.Module
        if args.loss == "weighted":
            loss_fn = modules.WeightedEnergyForcesLoss(
                energy_weight=args.energy_weight, forces_weight=args.forces_weight
            )
        elif args.loss == "forces_only":
            loss_fn = modules.WeightedForcesLoss(forces_weight=args.forces_weight)
        elif args.loss == "virials":
            loss_fn = modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
                virials_weight=args.virials_weight,
            )
        elif args.loss == "stress":
            loss_fn = modules.WeightedEnergyForcesStressLoss(
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
                stress_weight=args.stress_weight,
            )
        elif args.loss == "huber":
            loss_fn = modules.WeightedHuberEnergyForcesStressLoss(
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
                stress_weight=args.stress_weight,
                huber_delta=args.huber_delta,
            )
        elif args.loss == "dipole":
            assert (
                dipole_only is True
            ), "dipole loss can only be used with AtomicDipolesMACE model"
            loss_fn = modules.DipoleSingleLoss(
                dipole_weight=args.dipole_weight,
            )
        elif args.loss == "energy_forces_dipole":
            assert dipole_only is False and compute_dipole is True
            loss_fn = modules.WeightedEnergyForcesDipoleLoss(
                energy_weight=args.energy_weight,
                forces_weight=args.forces_weight,
                dipole_weight=args.dipole_weight,
            )
        else:
            # Unweighted Energy and Forces loss by default
            loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
        logging.info(loss_fn)

        if args.compute_avg_num_neighbors:
            args.avg_num_neighbors = modules.compute_avg_num_neighbors(train_loader)
        logging.info(f"Average number of neighbors: {args.avg_num_neighbors}")

        # Selecting outputs
        compute_virials = False
        if args.loss in ("stress", "virials", "huber"):
            compute_virials = True
            args.compute_stress = True
            args.error_table = "PerAtomRMSEstressvirials"

        output_args = {
            "energy": compute_energy,
            "forces": args.compute_forces,
            "virials": compute_virials,
            "stress": args.compute_stress,
            "dipoles": compute_dipole,
        }
        logging.info(f"Selected the following outputs: {output_args}")

        # Build model
        logging.info("Building model")
        if args.num_channels is not None and args.max_L is not None:
            assert args.num_channels > 0, "num_channels must be positive integer"
            assert args.max_L >= 0, "max_L must be non-negative integer"
            args.hidden_irreps = o3.Irreps(
                (args.num_channels * o3.Irreps.spherical_harmonics(args.max_L))
                .sort()
                .irreps.simplify()
            )

        assert (
            len({irrep.mul for irrep in o3.Irreps(args.hidden_irreps)}) == 1
        ), "All channels must have the same dimension, use the num_channels and max_L keywords to specify the number of channels and the maximum L"

        logging.info(f"Hidden irreps: {args.hidden_irreps}")
        model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(self.z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=self.z_table.zs,
        )

        self.model: torch.nn.Module

        if args.model == "MACE":
            if args.scaling == "no_scaling":
                std = 1.0
                logging.info("No scaling selected")
            else:
                mean, std = modules.scaling_classes[args.scaling](
                    train_loader, atomic_energies
                )
            self.model = modules.ScaleShiftMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=0.0,
                radial_MLP=ast.literal_eval(args.radial_MLP) if type(args.radial_MLP) == str else args.radial_MLP,
                radial_type=args.radial_type,
            )
        elif args.model == "ScaleShiftMACE":
            mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
            self.model = modules.ScaleShiftMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[args.interaction_first],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=mean,
                radial_MLP=ast.literal_eval(args.radial_MLP) if type(args.radial_MLP) == str else args.radial_MLP,
                radial_type=args.radial_type,
            )
        elif args.model == "ScaleShiftBOTNet":
            mean, std = modules.scaling_classes[args.scaling](train_loader, atomic_energies)
            self.model = modules.ScaleShiftBOTNet(
                **model_config,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[args.interaction_first],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=mean,
            )
        elif args.model == "BOTNet":
            self.model = modules.BOTNet(
                **model_config,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[args.interaction_first],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
            )
        elif args.model == "AtomicDipolesMACE":
            # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
            assert args.loss == "dipole", "Use dipole loss with AtomicDipolesMACE model"
            assert (
                args.error_table == "DipoleRMSE"
            ), "Use error_table DipoleRMSE with AtomicDipolesMACE model"
            self.model = modules.AtomicDipolesMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
                # dipole_scale=1,
                # dipole_shift=0,
            )
        elif args.model == "EnergyDipolesMACE":
            # std_df = modules.scaling_classes["rms_dipoles_scaling"](train_loader)
            assert (
                args.loss == "energy_forces_dipole"
            ), "Use energy_forces_dipole loss with EnergyDipolesMACE model"
            assert (
                args.error_table == "EnergyDipoleRMSE"
            ), "Use error_table EnergyDipoleRMSE with AtomicDipolesMACE model"
            self.model = modules.EnergyDipolesMACE(
                **model_config,
                correlation=args.correlation,
                gate=modules.gate_dict[args.gate],
                interaction_cls_first=modules.interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=o3.Irreps(args.MLP_irreps),
            )
        else:
            raise RuntimeError(f"Unknown model: '{args.model}'")

        self.model.to(self.device)

        # Optimizer
        decay_interactions = {}
        no_decay_interactions = {}
        for name, param in self.model.interactions.named_parameters():
            if "linear.weight" in name or "skip_tp_full.weight" in name:
                decay_interactions[name] = param
            else:
                no_decay_interactions[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": self.model.node_embedding.parameters(),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": args.weight_decay,
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "products",
                    "params": self.model.products.parameters(),
                    "weight_decay": args.weight_decay,
                },
                {
                    "name": "readouts",
                    "params": self.model.readouts.parameters(),
                    "weight_decay": 0.0,
                },
            ],
            lr=args.lr,
            amsgrad=args.amsgrad,
        )

        optimizer: torch.optim.Optimizer
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(**param_options)
        else:
            optimizer = torch.optim.Adam(**param_options)

        logger = tools.MetricsLogger(directory='MACE_results', tag=tag + "_train")

        lr_scheduler = LRScheduler(optimizer, args)

        swa: Optional[tools.SWAContainer] = None
        swas = [False]
        if args.swa:
            assert dipole_only is False, "swa for dipole fitting not implemented"
            swas.append(True)
            if args.start_swa is None:
                args.start_swa = (
                    args.max_num_epochs // 4 * 3
                )  # if not set start swa at 75% of training
            if args.loss == "forces_only":
                logging.info("Can not select swa with forces only loss.")
            elif args.loss == "virials":
                loss_fn_energy = modules.WeightedEnergyForcesVirialsLoss(
                    energy_weight=args.swa_energy_weight,
                    forces_weight=args.swa_forces_weight,
                    virials_weight=args.swa_virials_weight,
                )
            elif args.loss == "stress":
                loss_fn_energy = modules.WeightedEnergyForcesStressLoss(
                    energy_weight=args.swa_energy_weight,
                    forces_weight=args.swa_forces_weight,
                    stress_weight=args.swa_stress_weight,
                )
            elif args.loss == "energy_forces_dipole":
                loss_fn_energy = modules.WeightedEnergyForcesDipoleLoss(
                    args.swa_energy_weight,
                    forces_weight=args.swa_forces_weight,
                    dipole_weight=args.swa_dipole_weight,
                )
                logging.info(
                    f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight}, dipole weight : {args.swa_dipole_weight} and learning rate : {args.swa_lr}"
                )
            else:
                loss_fn_energy = modules.WeightedEnergyForcesLoss(
                    energy_weight=args.swa_energy_weight,
                    forces_weight=args.swa_forces_weight,
                )
                logging.info(
                    f"Using stochastic weight averaging (after {args.start_swa} epochs) with energy weight : {args.swa_energy_weight}, forces weight : {args.swa_forces_weight} and learning rate : {args.swa_lr}"
                )
            swa = tools.SWAContainer(
                model=AveragedModel(self.model),
                scheduler=SWALR(
                    optimizer=optimizer,
                    swa_lr=args.swa_lr,
                    anneal_epochs=1,
                    anneal_strategy="linear",
                ),
                start=args.start_swa,
                loss_fn=loss_fn_energy,
            )

        checkpoint_handler = CheckpointHandler_modified(
            directory=args.checkpoints_dir,
            tag=tag,
            keep=args.keep_checkpoints,
            swa_start=args.start_swa,
        )

        start_epoch = 0
        if args.restart_latest:
            try:
                opt_start_epoch = checkpoint_handler.load_latest(
                    state=tools.CheckpointState(self.model, optimizer, lr_scheduler),
                    swa=True,
                    device=self.device,
                )
            except Exception:  # pylint: disable=W0703
                opt_start_epoch = checkpoint_handler.load_latest(
                    state=tools.CheckpointState(self.model, optimizer, lr_scheduler),
                    swa=False,
                    device=self.device,
                )
            if opt_start_epoch is not None:
                start_epoch = opt_start_epoch

        ema: Optional[ExponentialMovingAverage] = None
        if args.ema:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=args.ema_decay)

        logging.info(self.model)
        logging.info(f"Number of parameters: {tools.count_parameters(self.model)}")
        logging.info(f"Optimizer: {optimizer}")

        if args.wandb:
            logging.info("Using Weights and Biases for logging")
            import wandb

            wandb_config = {}
            args_dict = vars(args)
            args_dict_json = json.dumps(args_dict)
            for key in args.wandb_log_hypers:
                wandb_config[key] = args_dict[key]
            tools.init_wandb(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=wandb_config,
            )
            wandb.run.summary["params"] = args_dict_json

        tools.train(
            model=self.model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            eval_interval=args.eval_interval,
            start_epoch=start_epoch,
            max_num_epochs=args.max_num_epochs,
            logger=logger,
            patience=args.patience,
            output_args=output_args,
            device=self.device,
            swa=swa,
            ema=ema,
            max_grad_norm=args.clip_grad,
            log_errors=args.error_table,
            log_wandb=args.wandb,
        )

        # Evaluation on test datasets
        logging.info("Computing metrics for training, validation, and test sets")

        all_collections = [
            ("train", collections.train),
            ("valid", collections.valid),
        ] + collections.tests

        for swa_eval in swas:
            epoch = checkpoint_handler.load_latest(
                state=tools.CheckpointState(self.model, optimizer, lr_scheduler),
                swa=swa_eval,
                device=self.device,
            )
            self.model.to(self.device)
            logging.info(f"Loaded model from epoch {epoch}")

            for param in self.model.parameters():
                param.requires_grad = False
            table = create_error_table(
                table_type=args.error_table,
                all_collections=all_collections,
                z_table=self.z_table,
                r_max=args.r_max,
                valid_batch_size=args.valid_batch_size,
                model=self.model,
                loss_fn=loss_fn,
                output_args=output_args,
                log_wandb=args.wandb,
                device=self.device,
            )
            logging.info("\n" + str(table))

            # Save entire model
            logging.info(f"Saving model to {self.model_file}")
            if args.save_cpu:
                self.model = self.model.to("cpu")
            self.save(self.model_file)

        logging.info("Done")
        
    @doc_inherit
    def predict(
            self, 
            molecular_database: data.molecular_database = None, 
            molecule: data.molecule = None,
            calculate_energy: bool = False,
            calculate_energy_gradients: bool = False, 
            calculate_hessian: bool = False,
            property_to_predict: Union[str, None] = 'estimated_y', 
            xyz_derivative_property_to_predict: Union[str, None] = None, 
            hessian_to_predict: Union[str, None] = None, 
            batch_size: int = 8,
        ) -> None:
        '''
            batch_size (int, optional): The batch size for batch-predictions.
        '''
        molDB, property_to_predict, xyz_derivative_property_to_predict, hessian_to_predict = \
            super().predict(molecular_database=molecular_database, molecule=molecule, calculate_energy=calculate_energy, calculate_energy_gradients=calculate_energy_gradients, calculate_hessian=calculate_hessian, property_to_predict = property_to_predict, xyz_derivative_property_to_predict = xyz_derivative_property_to_predict, hessian_to_predict = hessian_to_predict)
        
        configs =[mace_data.config_from_atoms(ase.Atoms(positions=mol.xyz_coordinates, numbers=mol.atomic_numbers)) for mol in molDB]
        
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                mace_data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=float(self.model.r_max)
                )
                for config in configs
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        energies_list = []
        forces_collection = []
        for batch in data_loader:
            batch = batch.to(self.device)
            try:
                output = self.model(batch.to_dict(), compute_force=xyz_derivative_property_to_predict)
            except:
                self.model = self.model.float()
                output = self.model(batch.to_dict(), compute_force=xyz_derivative_property_to_predict)
            energies_list.append(torch_tools.to_numpy(output["energy"]))
            
            if xyz_derivative_property_to_predict:
                forces = np.split(
                    torch_tools.to_numpy(output["forces"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )
                forces_collection.append(forces[:-1])  # drop last as its empty

        energies = np.concatenate(energies_list, axis=0)
        if property_to_predict:
            molDB.add_scalar_properties(energies / constants.Hartree2eV, property_to_predict)
        if xyz_derivative_property_to_predict:
            forces_list = np.array([
                -1 / constants.Hartree2eV * forces for forces_list in forces_collection for forces in forces_list
            ])
            molDB.add_xyz_vectorial_properties(forces_list, xyz_derivative_property_to_predict)

class CheckpointHandler_modified(tools.CheckpointHandler):

    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()
        self.old_cpk_path = None

    def save(
        self, state: tools.CheckpointState, epochs: int, keep_last: bool = False
    ) -> None:

        if self.old_cpk_path and not keep_last:
            os.remove(self.old_cpk_path)
 
        checkpoint = self.builder.create_checkpoint(state)
        self.io.save(checkpoint, epochs, keep_last)
        filename = self.io._get_checkpoint_filename(epochs, self.io.swa_start)
        latest_model_path = os.path.join(self.io.directory, filename.replace('epoch', 'model_epoch'))
        torch.save(state.model, latest_model_path)
        self.old_cpk_path = latest_model_path
            
def printHelp():
    helpText = __doc__.replace('.. code-block::\n\n', '') + '''
  To use Interface_MACE, please install MACE and its dependencies by following the instructions on  https://github.com/ACEsuit/mace

  Arguments with their default values:
    MLprog=MACE                enables this interface
    MLmodelType=MACE           requests ANI model

  For specifying options of MACE, use mace.<option_name>=<value>. E.g.:
    mace.max_num_epochs=1024
  
  check https://mace-docs.readthedocs.io/en/latest/index.html for available option names.

  Cite MACE:
    I. Batatia, D. P. Kovács, G. N. C. Simm, C. , G. Csányi, https://openreview.net/forum?id=YPpSngE-ZU, 2022.
    I. Batatia, S. Batzner, D. P. Kovács, A. Musaelian, G. N. C. Simm, C. Ortner, B. Kozinsky,  G. Csányi, arXiv:2205.06643 2022.
'''
    print(helpText)
