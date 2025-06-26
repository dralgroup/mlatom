import os
import time
import uuid
import math
import copy
import torch
import numpy as np
import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
import torch.utils
import torch.utils.data
from . import data
from . import model_cls
from . import models
import itertools
from .EEEDDDD_nn import batch_jacobian, MLP, ModelEnsemble, StackModelEnsemble, ModelCollection
from .EEEDDDD_descriptor import self_edge_vector_matrix, self_distance_matrix
from typing import Any, List, Dict, Union, Iterable, Optional

# torch.autograd.set_detect_anomaly(True)

@compile_mode("script")
class E3AttnBlk(torch.nn.Module):
    def __init__(
        self,
        lmax,
        sp_embed_dim,
        t_embed_dim,
        radial_cutoff,
        d_embed_dim,
        num_heads,
        irreps_key,
        irreps_query,
        irreps_value,
        key_weight_hidden_neurons,
        value_weight_hidden_neurons,
        drop_out,
    ) -> None:
        super().__init__()
        self.irreps_key = o3.Irreps(' + '.join([irreps_key] * num_heads))
        self.irreps_query = o3.Irreps(' + '.join([irreps_query] * num_heads))
        self.irreps_value = o3.Irreps(' + '.join([irreps_value] * num_heads))
        self.irreps_io = o3.Irreps(f'{sp_embed_dim}x0e + 1x1o + 1x1o + {t_embed_dim}x0e')
        self.radial_cutoff = radial_cutoff
        self.d_embed_args = [0, radial_cutoff, d_embed_dim, 'gaussian', False]
        self.d_embed_dim = d_embed_dim
        self.num_heads = num_heads
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.W_query = o3.Linear(self.irreps_io, self.irreps_query)
        self.W_out = o3.Linear(self.irreps_value, self.irreps_io)
        self.tp_key = o3.FullyConnectedTensorProduct(self.irreps_io, self.irreps_sh, self.irreps_key, shared_weights=False)
        self.tp_value = o3.FullyConnectedTensorProduct(self.irreps_io, self.irreps_sh, self.irreps_value, shared_weights=False)
        self.mlp_key_weight = MLP([d_embed_dim + t_embed_dim] + key_weight_hidden_neurons + [self.tp_key.weight_numel])
        self.mlp_value_weight = MLP([d_embed_dim + t_embed_dim] + value_weight_hidden_neurons + [self.tp_value.weight_numel])
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")
        self.drop_out = torch.nn.Dropout(drop_out)
    
    def split_heads(self, x):
        batch_size, num_atoms, num_adj, dim_x = x.shape
        return x.view(batch_size, num_atoms, num_adj, self.num_heads, dim_x // self.num_heads).transpose(1, 2).transpose(1, 3)
    
    def join_heads(self, x):
        batch_size, num_heads, num_atoms, dim_head = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, num_atoms, num_heads * dim_head)

    def forward(self, sp, coord, veloc, t, mask=None, idx=None, attn=None, pbc=None, cell=None):
        num_atoms = sp.shape[1]
        coord_edge = self_edge_vector_matrix(coord, pbc=pbc, cell=cell)
        coord_d = coord_edge.norm(dim=-1, keepdim=True)
        if mask is None:
            mask = torch.zeros_like(coord_d, dtype=bool)
        l = (~mask).sum(-2).max()
        if idx is None:
            _, idx = coord_d.topk(l, dim=-2, largest=False)
        mask = mask.gather(-2, idx).unsqueeze(1)
        coord_d = coord_d.gather(-2, idx)
        coord_edge = coord_edge.gather(-2, idx.repeat_interleave(3, -1))
        veloc_edge = self_edge_vector_matrix(veloc).gather(-2, idx.repeat_interleave(3, -1))
        t_ = t.unsqueeze(-2).repeat_interleave(l, -2)
        d_embed = e3nn.math.soft_one_hot_linspace(coord_d.squeeze(-1), *self.d_embed_args)
        d_t_embed = torch.cat((d_embed, t_), -1)
        x_src = torch.cat((sp.unsqueeze(-2).repeat_interleave(l, -2), coord_edge / -2, veloc_edge / -2, t_), -1)
        x_dst = torch.cat((sp.unsqueeze(-3).repeat_interleave(num_atoms, -3).gather(-2, idx.repeat_interleave(sp.shape[-1], -1)), coord_edge / 2, veloc_edge / 2, t.unsqueeze(-3).repeat_interleave(num_atoms, -3).gather(-2, idx.repeat_interleave(t.shape[-1], -1))), -1)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, coord_edge, True, normalization='norm')
        query = self.split_heads(self.W_query(x_dst))
        key = self.split_heads(self.tp_key(x_src, edge_sh, self.mlp_key_weight(d_t_embed)))
        value = self.split_heads(self.tp_value(x_src, edge_sh, self.mlp_value_weight(d_t_embed)))
        a = self.dot(query, key) / key.shape[-1]**0.5
        attn = a.masked_fill(mask, -torch.inf).softmax(-2)
        x = (attn * value).sum(-2)
        x = self.drop_out(self.W_out(self.join_heads(x)))
        sp, coord, veloc, t = [x[:, :, idx] for idx in self.irreps_io.slices()]
        return sp, coord, veloc, t, attn

@compile_mode("unsupported")
class EEENNDDDD(torch.nn.Module):
    def __init__(
        self,
        num_species,
        num_attn_blocks,
        lmax,
        sp_embed_dim,
        time_cutoff,
        t_embed_dim,
        radial_cutoff,
        d_embed_dim,
        num_heads,
        irreps_key,
        irreps_query,
        irreps_value,
        key_weight_hidden_neurons,
        value_weight_hidden_neurons,
        drop_out: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_species = num_species
        self.sp_embed_dim = sp_embed_dim
        self.num_attn_blocks = num_attn_blocks
        self.sp_embed = torch.nn.Embedding(num_species, sp_embed_dim)
        self.t_embed_args = [0, time_cutoff, t_embed_dim, 'gaussian', False]
        self.radial_cutoff = radial_cutoff
        self.Txs = torch.nn.ModuleList(
            [
                E3AttnBlk(
                    lmax=lmax,
                    sp_embed_dim=sp_embed_dim,
                    t_embed_dim=t_embed_dim,
                    radial_cutoff=radial_cutoff,
                    d_embed_dim=d_embed_dim,
                    num_heads=num_heads,
                    irreps_key=irreps_key,
                    irreps_query=irreps_query,
                    irreps_value=irreps_value,
                    key_weight_hidden_neurons=key_weight_hidden_neurons,
                    value_weight_hidden_neurons=value_weight_hidden_neurons,
                    drop_out=drop_out,
                )
                for _ in range(num_attn_blocks)
            ]
        )
    
    @property
    def num_parameter(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_base(self):
        for Blk in self.Txs:
            Blk.W_query = copy.deepcopy(Blk.W_query)
            Blk.tp_key = copy.deepcopy(Blk.tp_key)
            Blk.tp_value = copy.deepcopy(Blk.tp_value)
            Blk.W_out = copy.deepcopy(Blk.W_out)

    def forward(
        self,
        species_id: torch.Tensor,
        coordinates: torch.Tensor,
        velocities: torch.Tensor,
        time: torch.Tensor,
        pbc: torch.Tensor = None, 
        cell: torch.Tensor = None,
    ):
        species = self.sp_embed(species_id)
        time = e3nn.math.soft_one_hot_linspace(time, *self.t_embed_args).unsqueeze(1).repeat(1, species.shape[1], 1)
        sp = species
        coord = coordinates
        veloc = velocities
        t = time
        attn = None
        
        dist = self_distance_matrix(coord, pbc=pbc, cell=cell).unsqueeze(-1)
        mask = (dist > self.radial_cutoff) + (dist == 0)
        _, idx = dist.masked_fill(mask, torch.inf).topk((~mask).sum(-2).max(), dim=-2, largest=False)

        for Blk in self.Txs:
            sp, coord, veloc, t, attn = Blk(sp, coord, veloc, t, mask=mask, idx=idx, attn=attn, pbc=pbc, cell=cell)
            sp += species
            coord += coordinates
            veloc += velocities
            t += time
        
        if pbc is not None and cell is not None:
            cell_coord = coord @ cell.inverse()
            cell_coord -= cell_coord.floor() * pbc
            coord = cell_coord @ cell
        
        return coord
        # return {
            # 'species': sp, 
            # 'coordinates': coord, 
            # 'velocities': veloc, 
            # 'time': t, 
            # 'attention': attn,
        # }


class EEEDDDD(model_cls.ml_model, model_cls.torch_model,model_cls.downloadable_model):
    hyperparameters = model_cls.hyperparameters(
        {
            "seed": model_cls.hyperparameter(value=0),
            "num_attn_blocks": model_cls.hyperparameter(value=8),
            "lmax": model_cls.hyperparameter(value=3),
            "sp_embed_dim": model_cls.hyperparameter(value=16),
            "time_cutoff": model_cls.hyperparameter(value=10.0),
            "t_embed_dim": model_cls.hyperparameter(value=4),
            "radial_cutoff": model_cls.hyperparameter(value=4.0),
            "d_embed_dim": model_cls.hyperparameter(value=16),
            "num_heads": model_cls.hyperparameter(value=4),
            "irreps_key": model_cls.hyperparameter(value="8x0e + 8x1o"),
            "irreps_query": model_cls.hyperparameter(value="8x0e + 8x1o"),
            "irreps_value": model_cls.hyperparameter(value="8x0e + 8x1o"),
            "key_weight_hidden_neurons": model_cls.hyperparameter(value=[128]),
            "value_weight_hidden_neurons": model_cls.hyperparameter(value=[128]),
            "drop_out": model_cls.hyperparameter(value=0.0),
        }
    )

    model_file = None
    model = None

    def __init__(
        self,
        species: List = [],
        model_file: Optional[str] = None,
        hyperparameters: Union[Dict[str, Any], model_cls.hyperparameters] = {},
        device: Optional[str] = None,
        verbose=True,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose
        self.hyperparameters = self.hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)

        self.species = np.unique(species)

        if model_file:
            if not isinstance(model_file, str) or os.path.isfile(model_file):
                self.load(model_file)
            elif model_file.casefold() == 'MDtrajNet1'.casefold():
                self.load_special(model_file)
            else:
                if self.verbose:
                    print(f"the trained 4D model will be saved in {model_file}")
            self.model_file = model_file

    def save(self, model_file: str = "") -> None:
        if not model_file:
            model_file = f"4D_{str(uuid.uuid4())}.pt"
            self.model_file = model_file
        model_dict = {
            'state':  self.model.state_dict(),
            'args': {
                'species': self.species,
                'num_attn_blocks': self.hyperparameters.num_attn_blocks,
                'lmax': self.hyperparameters.lmax,
                'sp_embed_dim': self.hyperparameters.sp_embed_dim,
                'time_cutoff': self.hyperparameters.time_cutoff,
                't_embed_dim': self.hyperparameters.t_embed_dim,
                'radial_cutoff': self.hyperparameters.radial_cutoff,
                'd_embed_dim': self.hyperparameters.d_embed_dim,
                'num_heads': self.hyperparameters.num_heads,
                'irreps_key': self.hyperparameters.irreps_key,
                'irreps_query': self.hyperparameters.irreps_query,
                'irreps_value': self.hyperparameters.irreps_value,
                'key_weight_hidden_neurons': self.hyperparameters.key_weight_hidden_neurons,
                'value_weight_hidden_neurons': self.hyperparameters.value_weight_hidden_neurons,
                'drop_out': self.hyperparameters.drop_out,
            }
        }
        torch.save(model_dict, model_file)
        if self.verbose:
            print(f"model saved in {model_file}")

    def _load_model(self, model_file: str):
        model_dict = torch.load(model_file, map_location=self.device, weights_only=False)
        self.species = model_dict['args'].pop('species')
        model = self._new_model(model_dict['args'])
        model.load_state_dict(model_dict['state'])
        model.eval()
        if self.verbose:
            print(f"model loaded from {model_file}")
        return model

    def load(self, model_file: Union[str, Iterable[str]], ensemble=ModelCollection, reset_base=True) -> None:
        if isinstance(model_file, str):
            self.model = self._load_model(model_file)
        elif isinstance(model_file, Iterable):
            models=[self._load_model(m) for m in model_file]
            kwargs = {}
            if issubclass(ensemble, ModelEnsemble):
                kwargs['in_dims'] = [None]*4
                kwargs['reset_base'] = reset_base
            self.model = ensemble(models=models, **kwargs)
    
    def load_special(self,model_file):
        model_name, model_path, download = self.check_model_path(model_file)
        if download: self.download(model_name,model_path)
        model_paths = [os.path.join(model_path,f'MDtrajNet-1.{ii}.pt') for ii in range(4)]
        self.load(model_paths)

    def specie_to_index(self, species: np.ndarray) -> torch.Tensor:
        idx = -torch.ones(*species.shape, dtype=int)
        for i, specie in enumerate(self.species):
            idx[torch.tensor(species == specie)] = i
        if torch.min(idx) < 0:
            raise ValueError("unsupported elemet found")
        return idx

    def _new_model(self, hyperparameters={}):
        self.hyperparameters.update(hyperparameters)
        return EEENNDDDD(
            num_species=len(self.species),
            num_attn_blocks=self.hyperparameters.num_attn_blocks,
            lmax=self.hyperparameters.lmax,
            sp_embed_dim=self.hyperparameters.sp_embed_dim,
            time_cutoff=self.hyperparameters.time_cutoff,
            t_embed_dim=self.hyperparameters.t_embed_dim,
            radial_cutoff=self.hyperparameters.radial_cutoff,
            d_embed_dim=self.hyperparameters.d_embed_dim,
            num_heads=self.hyperparameters.num_heads,
            irreps_key=self.hyperparameters.irreps_key,
            irreps_query=self.hyperparameters.irreps_query,
            irreps_value=self.hyperparameters.irreps_value,
            key_weight_hidden_neurons=self.hyperparameters.key_weight_hidden_neurons,
            value_weight_hidden_neurons=self.hyperparameters.value_weight_hidden_neurons,
            drop_out=self.hyperparameters.drop_out,
        ).to(self.device)

    def train(
        self,
        trajectories: Optional[List] = None,
        dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 64,
        num_train: Union[int, float] = 0.8,
        num_valid: Union[int, float] = 0.2,
        max_num_train_batch: int = 2**10,
        max_num_valid_batch: int = 1,
        max_num_pool_batch: int = 1,
        max_epochs: int = 1024,
        velocity_weight: float = 0,
        LR_patience: int = 16,
        AL_interval: int = 0,
        AL_threshold: float = 3.0,
        validate_whole_training_set: bool = False,
        numerical_velocities: float = 0.0001,
        init_learning_rate: float =0.001,
        seed: Optional[int] = None,
        hyperparameters: Union[Dict[str, Any], model_cls.hyperparameters] = {},
        pbc: torch.Tensor = None, 
        cell: torch.Tensor = None,
    ):
        self.hyperparameters.update(hyperparameters)

        if not self.model:
            self.model = self._new_model()
            
        if isinstance(pbc, torch.Tensor):
            pbc = pbc.to(self.device)
        if isinstance(cell, torch.Tensor):
            cell = cell.float().to(self.device)

        dataset = trajs_to_dataset(
            trajectories,
            self.hyperparameters.time_cutoff,
            self.specie_to_index,
        ) if dataset is None else dataset
        
        provider = DataProvider(
            dataset,
            batch_size=batch_size,
            num_train=num_train,
            num_valid=num_valid,
            bucket_fn=lambda x: len(x['sp']),
            max_num_train_batch=max_num_train_batch,
            max_num_valid_batch=max_num_valid_batch,
            max_num_pool_batch=max_num_pool_batch,
            seed=self.hyperparameters.seed if seed is None else seed,          
        )
        
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=init_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=LR_patience, threshold=0
        )

        loss_weights = (
            torch.tensor(
                [1, velocity_weight]
            )
            .float()
            .to(self.device)
        )

        def batch_point_mse(batch, veloc=False, weight_by_time=False):
            sp = batch["sp"].to(self.device)
            init_coord = batch["init_coord"].to(self.device)
            init_veloc = batch["init_veloc"].to(self.device)
            final_coord = batch["final_coord"].to(self.device)
            final_veloc = batch["final_veloc"].to(self.device)
            t = batch["t"].to(self.device).requires_grad_(True)

            if weight_by_time:
                w = weight_by_time(t).view(-1, 1, 1)
            else:
                w = 1

            batch_size = sp.shape[0]
            coord_est = self.model(sp, init_coord, init_veloc, t, pbc=pbc, cell=cell)
            coord_err = coord_est - final_coord
            if pbc is not None:
                dist = []
                for i in range(- (pbc[0] + 0), pbc[0] + 1):
                    for j in range(- (pbc[1] + 0), pbc[1] + 1):
                        for k in range(- (pbc[2] + 0), pbc[2] + 1):
                            dist.append(coord_est - final_coord + i * cell[0] + j * cell[1] + k * cell[2])
                dist = torch.stack(dist)
                idx = dist.norm(dim=-1).argmin(0, True) 
                coord_err = dist.gather(0, torch.stack([idx] * 3, -1)).squeeze(0)
            coord_mse = (coord_err * w).square().mean((-2, -1))
            
            if torch.isnan(coord_mse).sum():
                pass
            if veloc:
                # if self.hyperparameters.minmax_time_velocities:
                #     lb, ub = self.hyperparameters.minmax_time_velocities
                #     mask = (t >= lb).logical_and(t <= ub)
                #     sp, init_coord, init_veloc, t, final_veloc = sp[mask], init_coord[mask], init_veloc[mask], t[mask], final_veloc[mask]
                if numerical_velocities:
                    veloc_est = numerical_differentiation(lambda x: self.model(sp, init_coord, init_veloc, x, pbc=pbc, cell=cell), t, numerical_velocities, side=2, fx=coord_est)
                else:
                    veloc_est = batch_jacobian(coord_est.reshape(batch_size, -1), t).view_as(coord_est)
                veloc_err = veloc_est - final_veloc
                veloc_mse = (veloc_err * w).square().mean((-2, -1))
            else:
                veloc_mse = torch.zeros_like(coord_mse, device=self.device)
            return torch.stack([coord_mse, veloc_mse])
            
        def epoch_point_mse(loader, veloc=False):
            mses = []
            for batch in loader:
                if veloc and not numerical_velocities:
                    p_mse = batch_point_mse(batch, veloc)
                else:
                    with torch.no_grad():
                        p_mse = batch_point_mse(batch, veloc)
                mses.append(p_mse)
            return torch.cat(mses, -1)
            

        def validate(loader, veloc=False):
            return torch.sqrt(epoch_point_mse(loader, veloc).mean(-1))
        
        def pool_AL(provider):
            p_mse = epoch_point_mse(provider.pool_loader)
            p_loss = (p_mse.T @ loss_weights).sqrt()
            avg = p_loss.mean()
            sd = p_loss.std()
            threshold = avg + AL_threshold * sd
            idx = torch.where(p_loss > threshold)[0]
            provider.update(idx)
            print('-' * 80)
            print('pool loss: %8.6f ± %8.6f threshold: %8.6f' % (avg, sd, threshold))
            print(f'updated {len(idx)} point(s) to training set from pool')
            provider.info()
            

        def epoch(validate_whole_training_set=False, training=True):
            results = {}
            if training: 
                self.model.train()
                # i_batch = 0
                for batch in provider.train_loader:
                    losses = batch_point_mse(
                        batch=batch, 
                        veloc=velocity_weight,
                        # weight_by_time=lambda t: 1 + (t / self.hyperparameters.time_cutoff).sqrt()
                    ).mean(-1)
                    loss = losses.dot(loss_weights)
                    if not loss.isnan().sum():
                        optimizer.zero_grad()
                        loss.backward()
                        # for k, v in dict(self.model.named_parameters()).items():
                        #     if v.isnan().sum():
                        #         print(k)
                        optimizer.step()
                    # i_batch += 1
                    # print(f'{i_batch}/{len(train_loader)}', end='\r')
                self.model.eval()
                if validate_whole_training_set:
                    train_losses = validate(
                        provider.train_loader, 
                        veloc=velocity_weight,
                    )
                    train_loss = train_losses.dot(loss_weights)
                else:
                    train_losses = losses
                    train_loss = loss
                results.update(
                    {
                        "training_losses": train_losses.cpu().detach().numpy(),
                        "training_loss": float(train_loss),
                    }
                )
            valid_losses = validate(
                provider.valid_loader, 
                # veloc=velocity_weight,
                veloc=True,
            )
            valid_loss = valid_losses.dot(loss_weights)
            scheduler.step(valid_loss)
            results.update(
                {
                    "validation_losses": valid_losses.cpu().detach().numpy(),
                    "validation_loss": float(valid_loss),
                }
            )
            return results

        def summary(i_epoch, time_epoch, lr, results):
            lines = []
            lines.append("-" * 80)
            lines.append(
                "epoch %-8d:: lr: %16.2g time:%16.2f s" % (i_epoch, lr, time_epoch)
            )
            if "training_loss" in results:
                coord_rmse, veloc_rmse = results["training_losses"]
                loss = results["training_loss"]
                lines.append(
                    "    training   RMSEs::   geometry : %12.6f Å velocity : %12.6f Å/fs"
                    % (coord_rmse, veloc_rmse)
                )
                lines.append("    training   loss ::   %-12.6f" % loss)
            if "validation_loss" in results:
                coord_rmse, veloc_rmse = results["validation_losses"]
                loss = results["validation_loss"]
                lines.append(
                    "    validation RMSEs::   geometry : %12.6f Å velocity : %12.6f Å/fs"
                    % (coord_rmse, veloc_rmse)
                )
                lines.append("    validation loss ::   %-12.6f" % loss)
            print("\n".join(lines), flush=True)

        best_loss = torch.inf
        for i_epoch in range(max_epochs + 1):
            if i_epoch:
                t_start = time.time()
                results = epoch(validate_whole_training_set)
                time_epoch = time.time() - t_start
            else:
                results = epoch(validate_whole_training_set, False)
                time_epoch = 0.0
            lr = optimizer.param_groups[0]["lr"]
            summary(i_epoch, time_epoch, lr, results)
            if results["validation_loss"] < best_loss:
                best_loss = results["validation_loss"]
                self.save(self.model_file)
            if math.isnan(results["validation_loss"]) or any([torch.isnan(p).sum() for p in self.model.parameters()]):
                break
                self.save('nan_'+self.model_file)
                self.model.load_state_dict(torch.load(self.model_file, map_location=self.device)['state'])
            if lr < 1e-6:
                break
            if AL_interval and i_epoch % AL_interval == 0:
                pool_AL(provider)

    def predict(
        self,
        molecular_database: data.molecular_database,
        time: Union[np.ndarray, float],
        predict_velocities: bool = True,
        numerical_velocities: Union[bool, float] = -0.0001,
        batch_size: int = 0,
    ):  
        molecular_database = data.molecular_database(molecular_database)
        time = torch.tensor(time if isinstance(time, Iterable) else [time] * len(molecular_database), device=self.device).float()
        if batch_size:
            preds = [
                self.predict(
                    molecular_database=molecular_database[i::batch_size],
                    time=time[i::batch_size],
                    predict_velocities=predict_velocities,
                    numerical_velocities=numerical_velocities,
                ) for i in range(batch_size)
            ]
            if isinstance(self.model, ModelCollection):
                molDBs = []
                for j in range(len(self.model)):
                    molDBs.append(data.molecular_database([preds[i % batch_size][j][i // batch_size] for i in range(len(molecular_database))]))
                return molDBs
            else:
                return data.molecular_database([preds[i % batch_size][i // batch_size] for i in range(len(molecular_database))])
        else:
            self.model.eval()
            species = self.specie_to_index(molecular_database.element_symbols).to(self.device)
            n_mol, num_at = species.shape
            coordinates = torch.tensor(molecular_database.xyz_coordinates, device=self.device).float()
            velocities = torch.tensor(molecular_database.get_xyz_vectorial_properties('xyz_velocities'), device=self.device).float()
            pbc = torch.tensor(molecular_database.pbc).to(self.device) if molecular_database.pbc is not None else None
            cell = torch.tensor(molecular_database.cell).float().to(self.device) if molecular_database.pbc is not None else None

            if predict_velocities:
                if numerical_velocities:
                    species = species.repeat(2, 1)
                    coordinates = coordinates.repeat(2, 1, 1)
                    velocities = velocities.repeat(2, 1, 1)
                    time = torch.cat([time, time + numerical_velocities], dim=0)
                    with torch.no_grad():
                        coord, coord_ = self.model(species, coordinates, velocities, time, pbc=pbc, cell=cell).split(n_mol, dim=-3)
                        veloc = (coord_ - coord) / numerical_velocities
                else:
                    time.requires_grad = True
                    coord = self.model(species, coordinates, velocities, time, pbc=pbc, cell=cell)
                    veloc = batch_jacobian(coord.reshape(-1, num_at * 3), time).view_as(coord)
            else:
                with torch.no_grad():
                    coord = self.model(species, coordinates, velocities, time, pbc=pbc, cell=cell)
                veloc = torch.zeros_like(coord)
            
            coord = coord.detach().cpu().numpy()
            veloc = veloc.detach().cpu().numpy()
            assert not np.sum(np.isnan(coord) + np.isnan(veloc)), 'nan value(s) predicted'
            
            if isinstance(self.model, ModelCollection):
                coord = np.transpose(coord, (1, 2, 3, 0))
                veloc = np.transpose(veloc, (1, 2, 3, 0))
                
            molDB = molecular_database.copy()
            molDB.xyz_coordinates = coord
            if predict_velocities:
                molDB.add_xyz_vectorial_properties(veloc, 'xyz_velocities')
            return molDB
        
        
    def propagate(
        self,
        molecule: data.molecule = None,
        molecular_database: data.molecular_database = None,
        time: float = 1000.0,
        time_step: float = 1.0,
        time_segment: float = 10.0,
        numerical_velocities=-0.0001,
        rescale_velocities=False,
        potential_model: models.model = None,
        potential_model_kwargs: dict = {},
        predict_potential: bool = False,
        print_eshift: bool =False,
        batch_size: int = 0,
    ):
        if molecule:
            molDB = data.molecular_database([molecule])
        elif molecular_database:
            molDB = molecular_database
        if rescale_velocities:
            assert potential_model, "provide a potential x model to rescale velocities"
            potential_model.predict(molecular_database=molDB, calculate_energy=True, **potential_model_kwargs)
            total_energy_target = molDB.energy + molDB.kinetic_energy
        n_traj = len(molDB)
        trajs = [data.molecular_trajectory() for _ in range(n_traj)]
        for i, traj in enumerate(trajs):
            traj.steps.append(data.molecular_trajectory_step(molecule=molDB[i]))
        molDBs = [molDB[[i]] for i in range(n_traj)]
        while time > 0:
            t_seg = min(time_segment, time)
            t = np.clip(np.arange(0, t_seg, time_step) + time_step, 0, t_seg)
            molDB = data.molecular_database([molDB[-1].copy() for _ in t for molDB in molDBs])
            t=np.repeat(t, n_traj)
            flat_DBs = self.predict(
                molecular_database=molDB,
                time=t, 
                numerical_velocities=numerical_velocities,
                batch_size=batch_size,
            )
            molDBs = [flat_DBs[i::n_traj] for i in range(n_traj)]
            
            if isinstance(self.model, ModelCollection):
                n_model = len(self.model)
                last_steps = data.molecular_database([molDB[-1].copy() for molDB in molDBs for _ in range(n_model)])
                for i in range(n_model):
                    last_steps[i::n_model].xyz_coordinates = last_steps[i::n_model].xyz_coordinates[..., i]
                    last_steps[i::n_model].add_xyz_vectorial_properties(last_steps[i::n_model].get_xyz_vectorial_properties('xyz_velocities')[..., i], 'xyz_velocities')
                potential_model.predict(molecular_database=last_steps, calculate_energy=True, **potential_model_kwargs)
                eshifts = (last_steps.energy + last_steps.kinetic_energy).T.reshape(-1, n_traj) - total_energy_target
                c = np.argmin(np.abs(eshifts), axis=0)
                for i in range(n_traj):
                    molDBs[i].xyz_coordinates = molDBs[i].xyz_coordinates[..., c[i]]
                    molDBs[i].add_xyz_vectorial_properties(molDBs[i].get_xyz_vectorial_properties('xyz_velocities')[..., c[i]], 'xyz_velocities')
            
            if rescale_velocities:
                for molDB in molDBs: molDB[-1].adjust_momenta()
                last_steps = data.molecular_database([molDB[-1] for molDB in molDBs])
                potential_model.predict(molecular_database=flat_DBs if predict_potential else last_steps, calculate_energy=True, **potential_model_kwargs)
                eshifts = (last_steps.get_properties('energy') + last_steps.kinetic_energy) - total_energy_target
                if print_eshift: print(eshifts * 627.5) 
                ek_target = np.clip(total_energy_target - last_steps.energy, 0, np.inf)
                # assert ek_target, 'could not conserve total energy'                    
                last_steps.add_xyz_vectorial_properties(np.sqrt(ek_target / last_steps.kinetic_energy)[:, np.newaxis,np.newaxis] * last_steps.get_xyz_vectorial_properties('xyz_velocities'), 'xyz_velocities')
            for i in range(n_traj):
                for mol in molDBs[i]:
                    trajs[i].steps.append(data.molecular_trajectory_step(molecule=mol))
            time = time - time_segment
            if self.verbose: print(f'{time} fs remaining                      ', end='\r')
        return trajs if n_traj - 1 else trajs[0]

    __call__ = propagate


class BucketSampler(torch.utils.data.Sampler):
    def __init__(
        self, bucket_key, batch_size, max_num_batch=torch.inf, shuffle=True, drop_last=False, generator=None, remember_last=False,
    ):
        self.buckets = {}
        self.batch_size = batch_size
        self.max_num_batch = max_num_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.remember_last = remember_last
        self.point_samplers = {}
        self.batch_samplers = {}
        self.update(bucket_key) 
    
    def new_point_sampler(self, key):
        if self.shuffle:
            self.point_samplers[key] = torch.utils.data.sampler.RandomSampler(
                data_source=self.buckets[key],
                generator=self.generator,
            )
        else:
            self.point_samplers[key] = torch.utils.data.sampler.SequentialSampler(
                data_source=self.buckets[key],
            )
    
    def new_batch_sampler(self, key):
        self.batch_samplers[key] = torch.utils.data.sampler.BatchSampler(
            sampler=self.point_samplers[key],
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        
    def update_selector(self):
        self.batch_id_to_key = []
        for key in self.buckets:
            self.batch_id_to_key.extend([key] * len(self.batch_samplers[key]))
        if self.shuffle:
            self.batch_selector = torch.utils.data.sampler.RandomSampler(
                data_source=range(len(self)),
                generator=self.generator,
            )
        else:
            self.batch_selector = torch.utils.data.sampler.SequentialSampler(
                data_source=range(len(self))
            )
    
    def update(self, bucket_key, mode='+'):
        for i, key in bucket_key.items():
            if mode == '+':
                if key not in self.buckets:
                    self.buckets[key] = [i]
                    self.new_point_sampler(key)
                    self.new_batch_sampler(key)
                else:
                    self.buckets[key].append(i)
            elif mode == '-' and i in self.buckets[key]:
                self.buckets[key].pop(self.buckets[key].index(i))
        self.update_selector()
    
        
    def __iter__(self):        
        batch_iters = {key: iter(self.batch_samplers[key]) for key in self.buckets}
        self.last_idx = []
        for i, i_batch in enumerate(self.batch_selector):
            if i >= self.max_num_batch:
                break
            key = self.batch_id_to_key[i_batch]
            idx = [self.buckets[key][i] for i in next(batch_iters[key])]
            if self.remember_last: self.last_idx.extend(idx)
            yield idx
    
    def __len__(self):
        return sum([len(self.batch_samplers[key]) for key in self.buckets])
    
class DataProvider():
    def __init__(
            self,
            dataset,
            batch_size,
            num_train=0.8,
            num_valid=0.2,
            bucket_fn=None,
            max_num_train_batch=torch.inf,
            max_num_valid_batch=torch.inf,
            max_num_pool_batch=torch.inf,
            seed=0,
        ) -> None:
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(seed)
        self.bucket_fn = bucket_fn
        
        num_pool = (1 if num_train < 1 else len(dataset)) - num_train - num_valid 
        train_idx, valid_idx, pool_idx = torch.utils.data.random_split(range(len(dataset)), [num_train, num_valid, num_pool], generator=self.generator)
        train_sampler = BucketSampler(
            bucket_key=self.bucket_key(train_idx),
            batch_size=batch_size,
            max_num_batch=max_num_train_batch,
            generator=self.generator,
        )
        valid_sampler = BucketSampler(
            bucket_key=self.bucket_key(valid_idx),
            batch_size=batch_size,
            max_num_batch=max_num_valid_batch,
            generator=self.generator,
        )
        pool_sampler = BucketSampler(
            bucket_key=self.bucket_key(pool_idx),
            batch_size=batch_size,
            max_num_batch=max_num_pool_batch,
            generator=self.generator,
            remember_last=True,
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_sampler=train_sampler
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=valid_sampler
        )
        self.pool_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=pool_sampler
        )
        
    def bucket_key(self, idx):
        if self.bucket_fn:
            return {i: self.bucket_fn(self.dataset[i]) for i in idx}
        else:
            return {i: 0 for i in range(len(self.dataset))}
    
    def update(self, idx):
        AL_idx = [self.pool_loader.batch_sampler.last_idx[i] for i in idx]
        self.train_loader.batch_sampler.update(self.bucket_key(AL_idx))
        self.pool_loader.batch_sampler.update(self.bucket_key(AL_idx), mode='-')
    
    def info(self):
        n_train = {k: len(v) for k, v in self.train_loader.batch_sampler.buckets.items()}
        n_valid = {k: len(v) for k, v in self.valid_loader.batch_sampler.buckets.items()}
        n_pool = {k: len(v) for k, v in self.pool_loader.batch_sampler.buckets.items()}
        print(f'train: {n_train} points in {len(self.train_loader)} batches')
        print(f'valid: {n_valid} points in {len(self.valid_loader)} batches')
        print(f'pool:  {n_pool} points in {len(self.pool_loader)} batches')
        

class interList:
    def __init__(self, lists):
        self.lists = lists
        self.idxmap = [(i,j) for i, l in enumerate(lists) for j in range(len(l))]
        self.idxmap = torch.cat([torch.stack([torch.ones(len(l), dtype=int) * i, torch.arange(len(l))]).T for i, l in enumerate(lists)])
    
    def __len__(self):
        return sum([len(l) for l in self.lists])
    
    def __iter__(self):
        for data in self.lists:
            yield from data
            
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(itertools.islice(self, idx.start, idx.stop, idx.step))
        else:
            i, j = self.idxmap[idx]
            return self.lists[i][j]


def trajs_to_dataset(
    trajectories,
    time_cutoff,
    species_converter,
    time_minimum=0,
):
    datasets = {
        "sp": [],
        "init_coord": [],
        "init_veloc": [],
        "final_coord": [],
        "final_veloc": [],
        "t": [],
    }
    for trajectory in trajectories:
        if isinstance(trajectory, data.molecular_trajectory):
            trajectory = convert_mlatom_traj(trajectory)
        num_steps = len(trajectory["t"])
        cutoff = np.sum(trajectory["t"] <= time_cutoff)
        gate = np.sum(trajectory["t"][:cutoff] < time_minimum)
        x_idx, y_idx = sample_trajectory(num_steps, cutoff, gate=gate)
        datasets["sp"].append(species_converter(trajectory["sp"][x_idx]))
        datasets["init_coord"].append(torch.tensor(trajectory["xyz"][x_idx], dtype=torch.float32))
        datasets["init_veloc"].append(torch.tensor(trajectory["v"][x_idx], dtype=torch.float32))
        datasets["final_coord"].append(torch.tensor(trajectory["xyz"][y_idx], dtype=torch.float32) )
        datasets["final_veloc"].append(torch.tensor(trajectory["v"][y_idx], dtype=torch.float32))
        datasets["t"].append(torch.tensor(trajectory["t"][y_idx] - trajectory["t"][x_idx], dtype=torch.float32))
    return torch.utils.data.StackDataset(
        **{key: interList(value) for key, value in datasets.items()}
    )

def convert_mlatom_traj(trajectory):
    mdb = trajectory.to_database()
    return {
        'sp': mdb.element_symbols,
        'xyz': mdb.xyz_coordinates,
        'v': mdb.get_xyz_vectorial_properties("xyz_velocities"),
        't': np.array([step.time for step in trajectory.steps]),
    }
    

def sample_trajectory(num_steps, cutoff, gate=0, mode="sparse"):
    idx = torch.arange(num_steps)
    if mode == "sparse":
        y_idx = idx
        x_idx = y_idx // cutoff * cutoff
        if gate:
            mask = y_idx % cutoff >= gate
            x_idx = x_idx[mask]
            y_idx = y_idx[mask]
    else:
        x_idx = idx.repeat(cutoff)
        y_idx = x_idx + torch.arange(cutoff).repeat_interleave(num_steps)
        mask = y_idx < num_steps
        x_idx = x_idx[mask]
        y_idx = y_idx[mask]
    return x_idx, y_idx


def numerical_differentiation(f, x, dx, side=2, fx=None):
    if side == 2:
        return (f(x + dx) - f(x - dx)) / dx /2
    if side == 1:
        return (f(x + dx) - fx) / dx 