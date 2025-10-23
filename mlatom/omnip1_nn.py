import torch, torchani 
import numpy as np 

class omnip1_NN():
    
    def __init__(self, model_path:str=None, level:str="cc"):

        self.model_path = model_path
        self.level = level
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.species_order = [1,6,7,8]

        self.AEV_setup()
        self.NN_setup()
        self.model_setup()
        self.load()

    def AEV_setup(self):

        self.Rcr = 5.2000e+00
        self.Rca = 4.0
        nshiftA = 4
        self.ShfA = torch.linspace(0.9, self.Rca, nshiftA+1, device=self.device)[:-1]
        self.EtaR = torch.tensor([1.6000000e+01], device=self.device)
        self.ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=self.device)
        self.Zeta = torch.tensor([3.2000000e+01], device=self.device)
        self.ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=self.device)
        self.EtaA = torch.tensor([8.0000000e+00], device=self.device)
        self.aev_computer = torchani.AEVComputer(self.Rcr, self.Rca, self.EtaR, self.ShfR, self.EtaA, self.Zeta, self.ShfA, self.ShfZ, len(self.species_order))

    def load(self):

        self.modeldict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.nn.load_state_dict(self.modeldict['nn'])
        
    def NN_setup(self):
        
        aev_dim = self.aev_computer.aev_length + 1
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.GELU(),
            torch.nn.Linear(224, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )

        N_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1)
        )

        O_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1)
        )

        self.nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
         
    def predict(self, 
        molecule, 
        calculate_energy=True,
        calculate_energy_gradients=False,
        calculate_hessian=False,
        ):
        
        if self.level.lower() == 'cc'.lower():
            self.method = np.array([1])
            self.sae = self.modeldict['cc_sae']
        elif self.level.lower() == 'dft'.lower():
            self.method = np.array([0])
            self.sae = self.modeldict['dft_sae']

        from torchani.utils import ChemicalSymbolsToInts
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)
        atomic_numbers = np.array([atom.atomic_number for atom in molecule.atoms])
        coordinates = torch.tensor(np.array(molecule.xyz_coordinates).astype('float')).to(self.device).requires_grad_(bool(calculate_energy_gradients or calculate_hessian))

        coordinates = coordinates.unsqueeze(0)
        species = species_to_tensor(atomic_numbers).to(self.device).unsqueeze(0)
        method_vector = torch.tensor([self.method], device=self.device)
        _, ANI_NN_energies = self.model((species, coordinates, method_vector))
        predicted_energies = ANI_NN_energies.item()
        for ss in species[0]:
            predicted_energies += self.sae[ss.item()] 
        molecule.energy = predicted_energies

        if calculate_energy_gradients or calculate_hessian:
            ANI_NN_energy_gradients = torch.autograd.grad(ANI_NN_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0][0]
            if calculate_energy_gradients:
                grads = ANI_NN_energy_gradients.detach().cpu().numpy()
                molecule.add_xyz_vectorial_property(grads, 'energy_gradients')
            if calculate_hessian:
                ANI_NN_hessians = torchani.utils.hessian(coordinates, energies=ANI_NN_energies)
                molecule.add_scalar_property(ANI_NN_hessians.detach().cpu().numpy()[0], 'hessian')
        
    def model_setup(self):
        self.model = Sequential_modified(self.aev_computer, self.nn).to(self.device)

class Sequential_modified(torch.nn.ModuleList):

    def __init__(self, *modules):
        super(Sequential_modified, self).__init__(modules)

    def forward(self, input_):
        input_1_ = self[0](input_[:2])
        aev_vector = input_1_.aevs
        methods_vector = input_[-1]
        methods_vector_expand = methods_vector.unsqueeze(1).repeat(1,aev_vector.shape[1],1)
        aev_vector_method = torch.cat((aev_vector, methods_vector_expand),2)

        species = input_1_.species
        from torchani.aev import SpeciesAEV
        aev_vector_method = aev_vector_method.float()
        species = species.float()
        input_species_AEV_method = SpeciesAEV(species, aev_vector_method)
        input_ = self[1](input_species_AEV_method)
        return input_