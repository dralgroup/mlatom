import torch, torchani 
import numpy as np 

class omnip1_NN():
    
    def __init__(self, model_path:str=None, level="cc"):

        self.model_path = model_path
        # A label source is a category, not a coordinate: 'cc' and 'dft', or the
        # integer the model encodes them as. The released model was trained at
        # cc=1 and dft=0, and it cannot be evaluated between them - a value it
        # was not trained at leaves the atomic-energy shift unlookupable, so
        # predict() refuses until train() has fitted one.
        self.level = level
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.species_order = ['H','C','N','O']

        self.AEV_setup()
        self.NN_setup()
        self.model_setup()
        if model_path is not None:
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

    def load(self, model_path: str = None):
        if model_path is not None:
            self.model_path = model_path
        if self.model_path is None:
            raise ValueError('No model path provided for loading OMNI-P1 NN.')

        self.modeldict = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.nn.load_state_dict(self.modeldict['nn'])
        # Saved models may carry extra keys; keep the ones we need.
        if 'species_order' in self.modeldict:
            self.species_order = self.modeldict['species_order']
        if 'level' in self.modeldict:
            self.level = self.modeldict['level']
        # {label source name: the integer this model encodes it as} - the same
        # shape OMNI-P2x uses for its one-hot index.
        self.source_index = dict(self.modeldict.get('source_index', {}) or {})
        self._setup_sae()
        self.model.eval()
        
    def save(self, model_file: str = '') -> None:
        if not model_file:
            raise ValueError('A model file path must be provided to save OMNI-P1 NN.')
        import os
        os.makedirs(os.path.dirname(os.path.abspath(model_file)) if os.path.dirname(model_file) else '.', exist_ok=True)
        state = {
            'nn': self.nn.state_dict(),
            'cc_sae': self.modeldict['cc_sae'],
            'dft_sae': self.modeldict['dft_sae'],
            'species_order': self.species_order,
            'level': self.level,
            # {label source name: the integer this model encodes it as}, so a
            # model fine-tuned at a named source reloads knowing what it means.
            'source_index': dict(getattr(self, 'source_index', {}) or {}),
        }
        # Save whatever shift this model is actually using. Storing it only for a
        # a level outside the trained ones meant a fine-tuned model reloaded
        # with the pretrained cc_sae/dft_sae instead of the one fitted to the
        # user's data - a constant error per composition, silently.
        if self.sae is not None:
            state['custom_sae'] = self.sae
        torch.save(state, model_file)
        self.model_path = model_file

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
        
        if self.sae is None:
            raise RuntimeError(
                'OMNI-P1 SAE has not been set. For a level the released model '
                'was not trained at, run train() first to fit the SAE.'
            )
        self.method = np.array([float(self._level_index())])

        from torchani.utils import ChemicalSymbolsToInts
        species_to_tensor = ChemicalSymbolsToInts(self.species_order)
        element_symbols = np.array(molecule.get_element_symbols())
        coordinates = torch.tensor(np.array(molecule.xyz_coordinates).astype('float')).to(self.device).requires_grad_(bool(calculate_energy_gradients or calculate_hessian))

        coordinates = coordinates.unsqueeze(0)
        species = species_to_tensor(element_symbols).to(self.device).unsqueeze(0)
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

    def set_num_threads(self, nthreads=0):
        if nthreads:
            self.nthreads = nthreads
            import torch
            torch.set_num_threads(nthreads)

    # ------------------------------------------------------------------
    # Transfer-learning helpers
    # ------------------------------------------------------------------
    def _level_index(self):
        """
        The integer this model encodes its label source as.

        Unlike OMNI-P2x, OMNI-P1 is told its level when it is CONSTRUCTED -
        ``omnip1(level='cc')``, ``'dft'``, or the integer directly - and does not
        read ``mol.label_source``: the level is one value for the whole model,
        not a per-molecule input. An earlier version of this docstring claimed
        the data declares the name here, which is true of OMNI-P2x and was never
        true of OMNI-P1. Whether the two should be unified is a design decision,
        recorded in docs/todo.md rather than settled by a docstring.

        ``source_index`` maps a name to an index for callers who set it by hand;
        nothing populates it automatically on this model. An integer level is
        accepted directly, which is what both OMNI-P models have always been
        underneath: OMNI-P1 was trained at cc=1 and dft=0, and OMNI-P2x's
        one-hot is an index.
        """
        if isinstance(self.level, (int, np.integer)) and not isinstance(self.level, bool):
            return int(self.level)
        named = getattr(self, 'source_index', None) or {}
        if self.level in named:
            return int(named[self.level])
        if str(self.level).lower() == 'cc':
            return 1
        elif str(self.level).lower() == 'dft':
            return 0
        else:
            raise ValueError(
                f"OMNI-P1 has no encoding for label source '{self.level}'. Use "
                f"'cc' or 'dft', give the integer directly as level=, or declare "
                f"it with source_index={{'{self.level}': <int>}}.")

    def _setup_sae(self):
        """Set self.sae from the loaded checkpoint or pretrained defaults."""
        if 'custom_sae' in self.modeldict:
            # A shift fitted to someone's data wins over the pretrained defaults.
            self.sae = self.modeldict['custom_sae']
            return
        # Switch on the encoded index, never on the raw self.level. level is
        # documented as 'cc', 'dft' OR an integer (omnip1.py:17), and
        # _level_index() is the one place that resolves all three; reading
        # self.level again here disagreed with it and broke two of the three
        # documented forms. level=1 and level=0 raised AttributeError, because
        # an integer has no .lower() - and those are precisely the two integers
        # the released model was trained at, so the documented integer form
        # worked only for a level the model has never seen. A name resolved
        # through source_index raised ValueError for the same reason.
        # Neither was reachable when the checkpoint carries custom_sae, which
        # is why this survived: the released pretrained model has none, so this
        # is the path every fine-tune actually starts on.
        index = self._level_index()
        if index == 1:
            self.sae = self.modeldict['cc_sae']
        elif index == 0:
            self.sae = self.modeldict['dft_sae']
        else:
            self.sae = None  # a level the released model was not trained at

    def _fit_sae_from_data(self, *data_loaders):
        """Fit per-species SAE from total-energy residuals.

        OMNI-P1's NN returns a total molecular energy, so we solve the linear
        system

            E_target - E_NN = sum_s (N_s * SAE_s)

        across the provided data loaders using weighted least squares
        (weights = 1 / number of atoms, giving a per-atom average residual).
        """
        import torch
        self.model.eval()
        method_value = float(self._level_index())
        nspecies = len(self.species_order)
        species_indices = torch.arange(nspecies, device=self.device)

        ctc = torch.zeros((nspecies, nspecies), dtype=torch.float64, device='cpu')
        ctr = torch.zeros(nspecies, dtype=torch.float64, device='cpu')

        with torch.no_grad():
            for loader in data_loaders:
                if loader is None:
                    continue
                for properties in loader:
                    true_energies = properties['energies'].to(self.device).float()
                    species = properties['species'].to(self.device)
                    coordinates = properties['coordinates'].to(self.device).float()
                    num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                    weights = 1.0 / num_atoms.clamp(min=1.0)

                    method_vector = torch.full(
                        (species.shape[0], 1), method_value,
                        dtype=torch.float32, device=self.device
                    )
                    _, nn_total = self.model((species, coordinates, method_vector))
                    residual = (true_energies - nn_total.squeeze(-1)) * weights

                    # counts[i, s] = number of atoms of species s in molecule i
                    counts = (species.unsqueeze(-1) == species_indices).sum(dim=1).float()
                    counts = counts * weights.unsqueeze(-1)

                    ctc += (counts.T @ counts).cpu().double()
                    ctr += (counts.T @ residual).cpu().double()

        ctc = ctc.numpy()
        ctr = ctr.numpy()
        try:
            sae = np.linalg.solve(ctc, ctr)
        except np.linalg.LinAlgError:
            sae = np.linalg.lstsq(ctc, ctr, rcond=None)[0]

        self.sae = {int(j): float(sae[j]) for j in range(nspecies)}

        # Warn for species that never appeared.
        diag = np.diag(ctc)
        for j, count in enumerate(diag):
            if count <= 0:
                print(f'Warning: species {self.species_order[j]} did not appear in the '
                      f'training data; its fitted SAE is set to 0.0.')
                self.sae[j] = 0.0

        self.model.train()

    def _sae_tensor(self, species):
        """Compute per-molecule SAE from self.sae."""
        if self.sae is None:
            raise RuntimeError(
                'OMNI-P1 SAE has not been set. For a level the released model '
                'was not trained at, run train() first to fit the SAE.'
            )
        sae_per_atom = torch.zeros_like(species, dtype=torch.float32)
        mask = species >= 0
        if mask.any():
            sae_per_atom[mask] = torch.tensor(
                [self.sae[s.item()] for s in species[mask]],
                dtype=torch.float32,
                device=species.device,
            )
        return sae_per_atom.sum(dim=1)

    def _optimizer_setup(self, learning_rate, lr_reduce_factor, lr_reduce_patience, lr_reduce_threshold):
        import torch
        wlist = []
        blist = []
        for net_idx, (key, network) in enumerate(self.nn.items()):
            # Each network is a Sequential of (Linear, activation, Linear, ...).
            n_layers = (len(network) + 1) // 2
            for j in range(n_layers):
                layer = network[j * 2]
                if not isinstance(layer, torch.nn.Linear):
                    continue
                if j == 0 or j == n_layers - 1:
                    wlist.append({'params': [layer.weight]})
                else:
                    wlist.append({'params': [layer.weight], 'weight_decay': 0.0001 / 10 ** j})
                blist.append({'params': [layer.bias]})

        self.AdamW = torch.optim.AdamW(wlist, lr=learning_rate)
        self.SGD = torch.optim.SGD(blist, lr=learning_rate)
        self.AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.AdamW, factor=lr_reduce_factor, patience=lr_reduce_patience, threshold=lr_reduce_threshold)
        self.SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.SGD, factor=lr_reduce_factor, patience=lr_reduce_patience, threshold=lr_reduce_threshold)

    def fix_layers(self, layers_to_fix):
        """Freeze selected NN layers. `layers_to_fix` may be a flat list of layer
        indices applied to all networks (e.g. [0,4]), a single per-network list
        broadcast to all networks (e.g. [[0,4]], the ANI convention), or one list
        per network."""
        if not layers_to_fix:
            return
        # Flat list -> apply to all networks; a single nested sub-list -> broadcast
        # to all networks (so [[0,4]] behaves the same here as in the ANI interface);
        # a list of per-network lists -> used as-is.
        if not isinstance(layers_to_fix[0], list):
            layers_to_fix = [layers_to_fix] * len(self.nn)
        elif len(layers_to_fix) == 1:
            layers_to_fix = layers_to_fix * len(self.nn)
        for name, parameter in self.model.named_parameters():
            parts = name.split('.')
            if len(parts) < 2:
                continue
            try:
                net_idx = int(parts[-3])
                layer_idx = int(parts[-2])
            except ValueError:
                continue
            if layer_idx in layers_to_fix[net_idx]:
                parameter.requires_grad = False

    def _data_setup(self, molecular_database, validation_molecular_database, spliting_ratio,
                    property_to_learn, xyz_derivative_property_to_learn, batch_size):
        from .interfaces.torchani_interface import molDB2ANIdata, PADDING

        # OMNI-P1 only supports H, C, N, O.
        data_atomic_numbers = list(np.sort(np.unique(np.concatenate(molecular_database.atomic_numbers))))
        data_element_symbols = [self._atomic_number2symbol(z) for z in data_atomic_numbers]
        supported = set(self.species_order)
        if not set(data_element_symbols).issubset(supported):
            raise ValueError(
                f"OMNI-P1 transfer-learning supports only {self.species_order}, "
                f"but the database contains {data_element_symbols}."
            )

        if validation_molecular_database == 'sample_from_molecular_database':
            idx = np.arange(len(molecular_database))
            np.random.shuffle(idx)
            split = int(len(idx) * spliting_ratio)
            train_idx, val_idx = idx[:split], idx[split:]
            train_db = molecular_database[train_idx]
            val_db = molecular_database[val_idx]
        else:
            train_db = molecular_database
            val_db = validation_molecular_database

        self.subtraining_set = molDB2ANIdata(
            train_db, property_to_learn, xyz_derivative_property_to_learn
        ).species_to_indices(self.species_order).collate(batch_size, PADDING)

        self.validation_set = molDB2ANIdata(
            val_db, property_to_learn, xyz_derivative_property_to_learn
        ).species_to_indices(self.species_order).collate(batch_size, PADDING)

    @staticmethod
    def _atomic_number2symbol(z):
        from . import data
        return data.atomic_number2element_symbol[z]

    def train(
        self,
        molecular_database,
        property_to_learn: str = 'energy',
        xyz_derivative_property_to_learn: str = None,
        validation_molecular_database = 'sample_from_molecular_database',
        hyperparameters: dict = None,
        spliting_ratio: float = 0.8,
        save_model: bool = True,
        file_to_save_model: str = None,
        use_last_model: bool = False,
        reset_energy_shifter: bool = True,
        verbose: int = 1,
    ) -> None:
        """
        Fine-tune the OMNI-P1 NN on a molecular database.

        ``reset_energy_shifter`` refits the per-element atomic-energy shift from
        the training data, which is what the other universal models do when they
        are fine-tuned.
        """
        import torch
        from .interfaces.torchani_interface import PADDING

        hyperparameters = dict(hyperparameters) if hyperparameters is not None else {}
        hp = {
            'batch_size': 8,
            'max_epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_learning_rate': 1.0e-5,
            'lr_reduce_patience': 64,
            'lr_reduce_factor': 0.5,
            'lr_reduce_threshold': 0.0,
            'force_coefficient': 0.1,
            'loss_type': 'weighted',
            'fixed_layers': False,
        }
        hp.update(hyperparameters)

        if save_model and not file_to_save_model:
            raise ValueError('file_to_save_model must be provided when save_model=True.')
        if file_to_save_model:
            self.model_file = file_to_save_model

        self._data_setup(
            molecular_database, validation_molecular_database, spliting_ratio,
            property_to_learn, xyz_derivative_property_to_learn, hp['batch_size'],
        )

        # Refit the atomic-energy shift from the training data, as every other
        # universal model does when fine-tuned (they force reset_energy_shifter).
        # Keeping the pretrained cc_sae/dft_sae would leave the network to absorb
        # the difference between its own atomic references and the user's level,
        # which is a constant per composition and not what the network is for.
        if reset_energy_shifter or self.sae is None:
            self._fit_sae_from_data(self.subtraining_set, self.validation_set)

        self._optimizer_setup(
            hp['learning_rate'], hp['lr_reduce_factor'], hp['lr_reduce_patience'], hp['lr_reduce_threshold'])

        if 'fixed_layers' in hp and hp['fixed_layers']:
            # hp may be a model_cls.hyperparameters container, whose __setitem__
            # wraps values in a hyperparameter object - fix_layers wants the
            # list itself. See the same unwrap in torchani_interface.ani.train.
            self.fix_layers(getattr(hp['fixed_layers'], 'value', hp['fixed_layers']))

        self.model.train()

        method_value = float(self._level_index())

        def loss_function(prediction, reference, reduction='none'):
            return torch.nn.functional.mse_loss(prediction, reference, reduction=reduction)

        def validate():
            total_error = 0.0
            count = 0
            for properties in self.validation_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
                coordinates = properties['coordinates'].to(self.device).float()
                method_vector = torch.full((species.shape[0], 1), method_value, dtype=torch.float32, device=self.device)
                _, predicted_energies = self.model((species, coordinates, method_vector))
                predicted_total = predicted_energies + self._sae_tensor(species)
                total_error += (loss_function(predicted_total, true_energies) / num_atoms.sqrt()).nanmean().item()
                count += 1
            return total_error / count if count else 0.0

        best_state = None
        best_validation_loss = float('inf')

        if verbose:
            print('OMNI-P1 transfer-learning starting.')

        for epoch in range(1, hp['max_epochs'] + 1):
            validation_loss = validate()
            if verbose:
                print(f'  validation loss: {validation_loss:.6e} at epoch {epoch}')

            learning_rate = self.AdamW.param_groups[0]['lr']
            if learning_rate < hp['early_stopping_learning_rate']:
                if verbose:
                    print('  early stopping: learning rate below threshold.')
                break

            # Save the best model to memory and optionally to disk.
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {k: v.cpu().clone() for k, v in self.nn.state_dict().items()}
                if save_model:
                    self.save(self.model_file)

            self.AdamW_scheduler.step(validation_loss)
            self.SGD_scheduler.step(validation_loss)

            for properties in self.subtraining_set:
                true_energies = properties['energies'].to(self.device).float()
                species = properties['species'].to(self.device)
                num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)

                if xyz_derivative_property_to_learn:
                    coordinates = properties['coordinates'].to(self.device).float().requires_grad_(True)
                    true_forces = properties['forces'].to(self.device).float()
                    method_vector = torch.full((species.shape[0], 1), method_value, dtype=torch.float32, device=self.device)
                    _, predicted_energies = self.model((species, coordinates, method_vector))
                    predicted_total = predicted_energies + self._sae_tensor(species)
                    forces = -torch.autograd.grad(predicted_total.sum(), coordinates, create_graph=True, retain_graph=True)[0]

                    energy_loss = (loss_function(predicted_total, true_energies) / num_atoms.sqrt()).nanmean()

                    nan_mask = torch.isnan(true_forces)
                    true_forces[nan_mask] = 0
                    forces[nan_mask] = 0
                    force_loss = (loss_function(true_forces, forces).sum(dim=(1, 2)) / num_atoms).nanmean()

                    if hp['loss_type'] == 'geometric':
                        loss = (energy_loss * force_loss) ** 0.5
                    else:
                        loss = energy_loss + hp['force_coefficient'] * force_loss
                else:
                    coordinates = properties['coordinates'].to(self.device).float()
                    method_vector = torch.full((species.shape[0], 1), method_value, dtype=torch.float32, device=self.device)
                    _, predicted_energies = self.model((species, coordinates, method_vector))
                    predicted_total = predicted_energies + self._sae_tensor(species)
                    loss = (loss_function(predicted_total, true_energies) / num_atoms.sqrt()).nanmean()

                self.AdamW.zero_grad()
                self.SGD.zero_grad()
                loss.backward()
                self.AdamW.step()
                self.SGD.step()

        if save_model and not use_last_model and best_state is not None:
            self.nn.load_state_dict(best_state)
            self.save(self.model_file)

        self.model.eval()
        if verbose:
            print('OMNI-P1 transfer-learning finished.')


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
