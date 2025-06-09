import torch
import torchani


class ID(torch.nn.Module):
    def __init__(self, n_atom, weights=None, optimizable=False):
        super().__init__()
        self.register_buffer("n_atom", torch.tensor(n_atom))
        self.register_buffer("triu_idx", torch.triu_indices(n_atom, n_atom, 1))

        if weights is None:
            weights = torch.ones(self.triu_idx.shape[-1])
        else:
            assert (
                weights.shape == self.triu_idx[0].shape
            ), "the length of the weights does not match the length of the descriptor"
            weights = weights.float()

        if optimizable:
            self.register_parameter("weights", torch.nn.parameter.Parameter(weights))
        else:
            self.register_buffer("weights", weights)

    def forward(self, xyz):
        return (
            self.weights
            / distance_matrix(xyz, xyz)[:, self.triu_idx[0], self.triu_idx[1]]
        )

    def back_convert(self, descr):
        n = descr.shape[-2]
        d_mat = torch.zeros(n, self.n_atom, self.n_atom)
        distances = self.weights / descr
        d_mat[:, self.triu_idx[0], self.triu_idx[1]] = distances
        d_mat = torch.transpose(d_mat, -2, -1)
        d_mat[:, self.triu_idx[0], self.triu_idx[1]] = distances
        d2 = d_mat**2
        c = torch.eye(self.n_atom).repeat(n, 1, 1) - torch.ones_like(d2) / self.n_atom
        b = -0.5 * c.matmul(d2.matmul(c))
        U, S, _ = torch.linalg.svd(b)
        xyz = S[:, None, :3].sqrt() * U[:, :, :3]
        return xyz


class RE(ID):
    def __init__(self, eqxyz, optimizable=False):
        n_atom = eqxyz.shape[-2]
        weights = distance_matrix(eqxyz, eqxyz)[
            torch.triu_indices(n_atom, n_atom, 1).unbind()
        ]
        super().__init__(n_atom, weights, optimizable)


class CM(ID):
    def __init__(self, z, optimizable=False):
        n_atom = z.shape[-1]
        weights = z.outer(z)[torch.triu_indices(n_atom, n_atom, 1).unbind()]
        super().__init__(n_atom, weights, optimizable)


class AEV(torchani.AEVComputer):
    def __init__(
        self,
        Rcr=5.2000e00,
        Rca=3.5000e00,
        EtaR=torch.tensor([1.6000000e01]),
        ShfR=torch.tensor(
            [
                9.0000000e-01,
                1.1687500e00,
                1.4375000e00,
                1.7062500e00,
                1.9750000e00,
                2.2437500e00,
                2.5125000e00,
                2.7812500e00,
                3.0500000e00,
                3.3187500e00,
                3.5875000e00,
                3.8562500e00,
                4.1250000e00,
                4.3937500e00,
                4.6625000e00,
                4.9312500e00,
            ]
        ),
        EtaA=torch.tensor([8.0000000e00]),
        Zeta=torch.tensor([3.2000000e01]),
        ShfA=torch.tensor([9.0000000e-01, 1.5500000e00, 2.2000000e00, 2.8500000e00]),
        ShfZ=torch.tensor(
            [
                1.9634954e-01,
                5.8904862e-01,
                9.8174770e-01,
                1.3744468e00,
                1.7671459e00,
                2.1598449e00,
                2.5525440e00,
                2.9452431e00,
            ]
        ),
        num_species=4,
    ):
        super().__init__(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)

def edge_vector_matrix(x, y, pbc=None, cell=None):
    if pbc is None:
        return (y.unsqueeze(-3) - x.unsqueeze(-2))
    # elif cell == torch.diag(cell.diagonal()):
        
    else:
        assert cell is not None, 'provide cell info'
        matrices = []
        for i in torch.arange(- (pbc[0] + 0), pbc[0] + 1):
            for j in torch.arange(- (pbc[1] + 0), pbc[1] + 1):
                for k in torch.arange(- (pbc[2] + 0), pbc[2] + 1):
                    matrices.append(edge_vector_matrix(x, y + i * cell[0] + j * cell[1] + k * cell[2]))
        matrices = torch.stack(matrices)
        idx = matrices.norm(dim=-1).argmin(0, True) 
        return matrices.gather(0, torch.stack([idx] * x.shape[-1], -1)).squeeze(0)

def self_edge_vector_matrix(x, pbc=None, cell=None):
    return edge_vector_matrix(x, x, pbc=pbc, cell=cell)
                    
def distance_matrix(x, y, pbc=None, cell=None):
    return edge_vector_matrix(x, y, pbc=pbc, cell=cell).norm(dim=-1)

def self_distance_matrix(x, pbc=None, cell=None):
    return distance_matrix(x, x, pbc=pbc, cell=cell)