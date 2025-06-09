import torch
from typing import Iterable, Dict, List, Tuple


class MLP(torch.nn.ModuleList):
    def __init__(self, neurons, activation_function=torch.nn.GELU):
        layers = []
        for i in range(len(neurons) - 2):
            layers.append(torch.nn.Linear(neurons[i], neurons[i + 1]))
            layers.append(activation_function())
        layers.append(torch.nn.Linear(neurons[i + 1], neurons[i + 2]))
        super().__init__(layers)
    
    def forward(self, x):
        for module in self:
            x = module(x)
        return x

class ModelCollection(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def __len__(self):
        return len(self.models)
    
    def __iter__(self):
        for model in self.models:
            yield model 
    
    def __getitem__(self, item):
        return self.models[item]
    
    def forward(self, *args, **kwargs):
        results = [model(*args, **kwargs) for model in self.models]
        if isinstance(results[0], torch.Tensor):
            return torch.stack(results)
        elif isinstance(results[0], List):
            return [torch.stack(result) for result in zip(*results)]
        elif isinstance(results[0], Tuple):
            return tuple(torch.stack(result) for result in zip(*results))
        elif isinstance(results[0], Dict):
            keys = list(results[0].keys())
            values = (result.values() for result in results)
            return {keys[i]: torch.stack(result) for i, result in enumerate(zip(*values))}

class ModelEnsemble(ModelCollection):
    def __init__(self, models, in_dims, reset_base=False):
        super().__init__(models)
        from torch.func import stack_module_state
        import copy
        self.params, self.buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(models[0]).to('meta')
        self.in_dims = tuple([0, 0] + in_dims) 
        self.reset_base = reset_base

    def fmodel(self, params, buffers, *args, **kwargs):
        from torch.func import functional_call
        if self.reset_base:
            if 'reset_base' in self.base_model.__dir__():
                self.base_model.reset_base()
            else:
                import copy
                self.base_model = copy.deepcopy(self.base_model)
        return functional_call(self.base_model, (params, buffers), args=args, kwargs=kwargs)
        
    def model(self, *args, **kwargs):
        from torch import vmap
        return vmap(self.fmodel, in_dims=self.in_dims)(self.params, self.buffers, *args, **kwargs)
        
    def forward(self, *args, **kwargs):
        results = self.model(*args, **kwargs)
        if isinstance(results, torch.Tensor):
            return self._algorithm(results, dim=0)
        elif isinstance(results, List):
            return [self._algorithm(result, dim=0) for result in results]
        elif isinstance(results, Tuple):
            return tuple(self._algorithm(result, dim=0) for result in results)
        elif isinstance(results, Dict):
            return {k: self._algorithm(v, dim=0) for k, v in results.items()}
        
class StackModelEnsemble(ModelEnsemble):
    @staticmethod
    def _algorithm(x, dim=0):
        return x.transpose(0, dim)
    
class SumModelEnsemble(ModelEnsemble):
    _algorithm = torch.sum

class MeanModelEnsemble(ModelEnsemble):
    _algorithm = torch.nanmean

class MedianModelEnsemble(ModelEnsemble):
    _algorithm = torch.nanmedian


class Attention(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, embed_dim=0, output_dim=0, num_heads=1):
        super().__init__()
        assert embed_dim % num_heads == 0, "make sure embed_dim % n_head == 0"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim if embed_dim else query_dim
        self.output_dim = output_dim if output_dim else query_dim
        
        self.W_q = torch.nn.Linear(self.query_dim, self.embed_dim, bias=False)
        self.W_k = torch.nn.Linear(self.key_dim, self.embed_dim, bias=False)
        self.W_v = torch.nn.Linear(self.value_dim, self.embed_dim, bias=False)
        self.W_o = torch.nn.Linear(self.embed_dim, self.output_dim, bias=False)
    
    def forward(self, Q, K, V, side_attn=0):
        bat_size, seq_len, _ = V.shape
        Q = (
            self.W_q(Q.transpose(0, 1))
            .view(seq_len, bat_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        K = (
            self.W_k(K.transpose(0, 1))
            .view(seq_len, bat_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        V = (
            self.W_v(V.transpose(0, 1))
            .view(seq_len, bat_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        
        dp_attn = torch.bmm(Q, K.transpose(-2, -1))
        total_attn = dp_attn + side_attn
        attn = torch.nn.functional.softmax(total_attn, dim=-1)
        
        O = torch.bmm(attn, V)
        O = O.transpose(0, 1).contiguous().view(seq_len, bat_size, self.embed_dim)
        O = self.W_o(O).transpose(0, 1)
        return O, total_attn


class SelfAttention(Attention):
    def __init__(self, input_dim, embed_dim, output_dim, num_heads=1):
        super().__init__(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_heads=num_heads,
        )
        
    def forward(self, X, side_attn=0):
        return super().forward(Q=X, K=X, V=X, side_attn=side_attn)
    
class GaussianOneHot(torch.nn.Module):
    def __init__(self, start=0., end=8, step=0.25, sigma=1.0) -> None:
        super().__init__()
        self.Xs = torch.range(start=start, end=end, step=step)
        self.sigma = sigma
        # self.register_parameter('sigma', torch.nn.parameter.Parameter(sigma))
    
    def dim(self):
        return len(self.Xs)
    
    def forward(self, X):
        return torch.exp(-((self.Xs - X.unsqueeze(-1))).square())

def batch_jacobian(y, x):
    y = y.sum(dim=0)
    jac = [torch.autograd.grad(yi, x, retain_graph=True)[0] for yi in y]
    jac = torch.stack(jac, dim=1)
    return jac

