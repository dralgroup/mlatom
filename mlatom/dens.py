from . import data, models, constants
from multiprocessing import cpu_count

class dens(models.model, metaclass=models.meta_method):

    """ 
    DFT ensemble methods. Preprint on ChemRxiv: https://doi.org/10.26434/chemrxiv-2024-2g7zr

    Arguments:

        method (str): [densxN]/[basis set]

    .. code-block:: 

        # Initialize molecule
        mol = ml.data.molecule()
        mol.read_from_xyz_file(filename='ethanol.xyz')
        # Run DENS calculation
        dens = ml.models.methods(method='dens24px3/6-31g*')
        dens.predict(molecule=mol, calculate_energy=True, calculate_energy_gradients=True, calculate_hessian=True) 
        energy = mol.energy
        gradient = mol.get_energy_gradients()
        hess = mol.hessian

    """ 

    def __init__(
        self,
        method: str = 'dens24px3/6-31g*',
        nthreads = None
    ):

        self.method = method.lower() 
        self.program = 'orca' if 'o' in method else 'pyscf'
        self.nfunctional = int(method.replace('dens24px','').split('/')[0]) if self.program=='pyscf' else int(method.replace('dens24ox','').split('/')[0])
        self.basis = method.split('/')[-1] 
        
        if nthreads is None:
                self.nthreads = cpu_count()
        else:
            self.nthreads = nthreads

    @classmethod
    def is_available_method(cls, method):
        # methods can be used without `method` keywords:
        # DFT, HF, MP2, FCI, CISD, CCSD, CCSD(T) / [basis set]
        method = method.split('/')[0]
        available_methods_pyscf = [f'dens24px{ii}' for ii in range(2,55)]
        available_methods_orca = [f'dens24ox{ii}' for ii in range(2,29)] 
        available_methods = available_methods_pyscf + available_methods_orca
        cls.available_methods = available_methods
        if method.lower() in cls.available_methods:
            return True
        return False

    def load(self):
        if self.program == 'pyscf':
            functional_dict = {pyscf_functionals[ii]:pyscf_weights[self.nfunctional-2][ii] for ii in range(self.nfunctional)}
        else:
            functional_dict = {orca_functionals[ii]:orca_weights[self.nfunctional-2][ii] for ii in range(self.nfunctional)}
        
        functional_nodes = []
        dispersion_nodes = []
        for ff, ww in functional_dict.items():
            functional_node = models.model_tree_node(
                    name=ff,
                    model=models.methods(method=f'{ff}/{self.basis}', program=self.program, nthreads=self.nthreads),
                    operator='predict')
            functional_node.weight = ww 
            functional_nodes.append(functional_node)

            if '-' in ff:
                f = ff.split('-')[0]
                disp = ff.split('-')[1]
                if disp.lower() in ['v', 'd3']:
                    continue
            dispersion_node = models.model_tree_node(
                    name=f'{f}_d3',
                    model=models.methods(method=f'd3bj', functional=f),
                    operator='predict') 
            dispersion_node.weight = ww 
            dispersion_nodes.append(dispersion_node)

        self.ensemble = models.model_tree_node(
            name=self.method,
            children=functional_nodes+dispersion_nodes,
            operator='weighted_sum'
        )
    
    def predict(self, molecular_database=None, molecule=None,
                calculate_energy=True, calculate_energy_gradients=False, calculate_hessian=False, calculate_dipole_derivatives=False, calculate_polarizability_derivatives=False,):
        molDB = super().predict(molecular_database=molecular_database, molecule=molecule)

        self.load()
        self.ensemble.predict(
            molecular_database=molDB,
            calculate_energy=calculate_energy, 
            calculate_energy_gradients=calculate_energy_gradients, 
            calculate_hessian=calculate_hessian, calculate_dipole_derivatives=calculate_dipole_derivatives,
            calculate_polarizability_derivatives=calculate_polarizability_derivatives,)

pyscf_functionals = hyb_fun = ['wB97X-V', 'M052X', 'B1LYP', 'X3LYP', 'B3LYP', 'N12SX', 'HSE06', 'TPSS0', 'B3P86', 'PW91', 'BLYP', 'B97-D', 'VV10', 'TPSSh', 'BPBE', 'OLYP', 'HSE03', 'B3PW91', 'revPBE', 'B97-1', 'PBE', 'PBE0', 'TPSS', 'N12', 'MPWB1K', 'SOGGA11X', 'BP86', 'M05', 'XLYP', 'M062X', 'wB97X-D3', 'PW6B95', 'MPW1B95', 'MN12SX', 'MN12L', 'MN15', 'M11L', 'revTPSS', 'SCAN', 'OPBE', 'RPBE', 'MN15L', 'B1B95', 'B97-2', 'TPSS1KCIS', 'M06L', 'MPWKCIS1K', 'PBE1KCIS', 'MPW1KCIS', 'revTPSSh', 'M06', 'O3LYP', 'mPW1LYP', 'PKZB']

pyscf_weights = [[0.7128, 0.2852], [0.6163, 0.284, 0.0982], [0.6448, 0.2592, 0.4398, -0.3408], [0.6257, 0.2544, 0.4517, -0.4341, 0.1064], [0.6385, 0.2848, 0.417, -0.3554, 0.1307, -0.1099], [0.6221, 0.2554, 0.607, -0.674, 0.1326, -0.2474, 0.314], [0.6199, 0.2741, 0.7015, -0.7667, 0.1493, -0.2979, 0.5016, -0.1723], [0.6144, 0.3007, 0.6964, -0.7528, 0.116, -0.3389, 0.4957, -0.2207, 0.0977], [0.5849, 0.3046, 0.2315, -0.1984, 0.12, -0.3891, 0.6524, -0.2982, 0.1782, -0.1829], [0.5812, 0.2858, 0.0445, -0.1282, 0.5534, -0.3788, 0.9548, -0.2287, -0.065, -0.403, -0.2106], [0.6078, 0.3016, 0.4303, -0.6737, 1.0618, -0.4011, 1.0557, -0.1547, -0.2191, -0.341, -0.4089, -0.2509], [0.6198, 0.2851, 0.3747, -0.682, 0.9445, -0.4434, 1.191, -0.1549, -0.2463, -0.4715, -0.3664, -0.1997, 0.1584], [0.6174, 0.2814, 0.3926, -0.712, 0.9711, -0.4445, 1.1745, -0.0784, -0.2515, -0.4529, -0.3786, -0.198, 0.1668, -0.0787], [0.605, 0.2773, 0.6939, -1.1133, 1.087, -0.421, 1.1879, -0.0585, -0.2048, -0.399, -0.4417, -0.1886, 0.198, -0.0939, -0.1174], [0.5922, 0.271, 0.7135, -1.134, 1.0436, -0.4017, 1.2204, -0.1546, -0.168, -0.4607, -0.4199, -0.2139, 0.2496, -0.0285, -0.1639, 0.0661], [0.6015, 0.2765, 0.6583, -1.1693, 1.1163, -0.398, 1.1584, -0.1873, -0.132, -0.5031, -0.4515, -0.1974, 0.267, 0.0071, -0.2173, 0.075, 0.1058], [0.5559, 0.2622, 0.7382, -1.3354, 1.0872, -0.4164, 0.893, -0.4875, -0.2549, -0.3925, -0.4385, -0.2323, 0.3534, 0.239, -0.6395, 0.0871, 0.1697, 0.8196], [0.5823, 0.2502, 0.8072, -1.3903, 1.0161, -0.4992, 0.9469, -0.683, -0.5041, -0.3005, -0.4217, -0.1043, 0.4236, 0.4194, -0.6197, 0.259, 0.1237, 1.2845, -0.5879], [0.5473, 0.2402, 0.8027, -1.464, 1.0336, -0.5539, 0.9568, -0.7648, -0.5118, -0.3034, -0.4245, -0.1456, 0.4735, 0.5019, -0.6855, 0.3239, 0.0356, 1.5089, -0.7415, 0.1706], [0.5347, 0.2441, 0.4536, -1.3517, 1.3365, -0.5414, 0.8846, -0.5106, -0.4397, -0.0068, -0.4901, -0.1993, 0.5372, 0.2613, -0.4149, 0.3548, 0.217, 1.1759, -0.6872, 0.271, -0.6301], [0.5329, 0.2427, 0.5475, -1.3894, 1.2986, -0.5371, 0.865, -0.5031, -0.4197, 0.0236, -0.5289, -0.1852, 0.5574, 0.2537, -0.4126, 0.3606, 0.1804, 1.1395, -0.7092, 0.2629, -0.6903, 0.1098], [0.5913, 0.236, 0.4874, -0.4999, 0.6313, -0.4912, 1.3072, -0.4424, -0.1161, -0.2337, -0.6323, -0.1636, 0.3621, -2.8078, -0.7429, 0.2588, -0.2521, 1.2464, -0.3238, 0.1708, -1.2594, 0.9254, 2.7489], [0.6086, 0.2342, 0.73, -0.7687, 0.6104, -0.621, 1.3703, -0.5956, 0.0062, -0.2425, -0.6676, -0.1682, 0.3005, -2.6911, -0.8526, 0.2176, -0.2819, 1.2315, -0.2937, 0.2542, -1.2393, 0.9991, 2.746, 0.1155], [0.604, 0.2217, 0.7113, -0.7574, 0.6181, -0.6331, 1.3994, -0.5913, 0.0297, -0.2355, -0.6674, -0.1665, 0.3063, -2.7763, -0.8449, 0.2151, -0.2853, 1.2075, -0.2876, 0.261, -1.2751, 0.9992, 2.8045, 0.1172, 0.0276], [0.6243, 0.2118, 0.7011, -0.7872, 0.6669, -0.6389, 1.4103, -0.5523, 0.0506, -0.2724, -0.6699, -0.1838, 0.31, -2.9, -0.8476, 0.2075, -0.2679, 1.2337, -0.2888, 0.3035, -1.2618, 0.9653, 2.8775, 0.1243, 0.0383, -0.0527], [0.6286, 0.2116, 0.7252, -0.7761, 0.5881, -0.6237, 1.4693, -0.5934, -0.113, -0.1868, -0.6606, -0.193, 0.3044, -2.9303, -1.023, 0.2012, -0.3432, 1.4227, -0.2395, 0.3056, -1.3977, 0.999, 2.9208, 0.1226, 0.0422, -0.0542, 0.1951], [0.628, 0.212, 0.7223, -0.7737, 0.5884, -0.6241, 1.4679, -0.5919, -0.1156, -0.1859, -0.6603, -0.194, 0.3048, -2.9306, -1.0217, 0.2017, -0.3433, 1.4256, -0.2416, 0.3065, -1.3988, 0.9983, 2.9206, 0.1224, 0.0421, -0.0547, 0.1964, 0.0014], [0.6119, 0.2215, 0.8035, -0.7915, 0.5793, -0.6252, 1.4024, -0.5583, -0.1259, -0.1576, -0.6515, -0.2121, 0.3249, -2.9786, -0.9821, 0.2204, -0.3232, 1.384, -0.2497, 0.3286, -1.3972, 0.9925, 2.9443, 0.1167, 0.0366, -0.0551, 0.2035, 0.0044, -0.0643], [0.6139, 0.2351, 0.7766, -0.7564, 0.58, -0.6325, 1.3762, -0.5656, -0.1552, -0.144, -0.6576, -0.212, 0.3276, -3.0286, -1.0116, 0.215, -0.3306, 1.4284, -0.2433, 0.3238, -1.4386, 1.0443, 2.9916, 0.116, 0.0458, -0.0472, 0.2295, 0.009, -0.0626, -0.0251], [0.678, 0.2295, 0.7641, -0.7495, 0.6039, -0.6201, 1.3906, -0.569, -0.0988, -0.1684, -0.655, -0.1813, 0.3066, -3.1153, -1.0244, 0.1797, -0.3113, 1.4061, -0.1871, 0.3179, -1.4123, 1.0157, 3.0603, 0.1185, 0.0502, -0.0367, 0.201, 0.0119, -0.0839, -0.0258, -0.0932], [0.6897, 0.2176, 0.8907, -1.221, 1.1472, -0.5908, 1.3456, -0.6595, 0.0839, -0.2594, -0.6984, -0.2268, 0.2921, -3.099, -0.9008, 0.1772, -0.1327, 0.9074, -0.1017, 0.4219, -1.3294, 0.9465, 3.1345, 0.0951, 0.3959, -0.1168, 0.1497, 0.0207, -0.1082, 0.0255, -0.1209, -0.372], [0.6899, 0.2166, 0.9195, -1.2541, 1.1576, -0.5913, 1.3407, -0.6687, 0.0987, -0.2453, -0.7051, -0.2281, 0.2898, -3.1274, -0.8995, 0.1813, -0.1328, 0.9088, -0.1103, 0.4194, -1.3259, 0.9448, 3.1659, 0.0945, 0.4421, -0.1171, 0.1383, 0.0214, -0.108, 0.0248, -0.1202, -0.3639, -0.0526], [0.6886, 0.2152, 0.9337, -1.2641, 1.1588, -0.5824, 1.3616, -0.672, 0.1191, -0.2609, -0.707, -0.2285, 0.2888, -3.1248, -0.8916, 0.1852, -0.1306, 0.8736, -0.1017, 0.4201, -1.3146, 0.921, 3.1651, 0.0921, 0.4442, -0.1149, 0.1333, 0.0194, -0.1141, 0.0344, -0.121, -0.3538, -0.0572, -0.0113], [0.5973, 0.1354, 0.7193, -0.9621, 0.8896, -0.5717, 1.3263, -0.5219, -0.2002, -0.2976, -0.7293, -0.2118, 0.4876, -3.0858, -1.0511, 0.1314, -0.2992, 1.1853, -0.0203, 0.3121, -1.5279, 1.1606, 2.965, 0.1223, 0.4271, -0.117, 0.3762, -0.0331, -0.0199, 0.0075, -0.0833, -0.311, 0.1567, 0.3319, -0.2803], [0.5712, 0.1096, 0.9276, -1.1695, 0.9099, -0.5652, 1.2517, -0.529, -0.1029, -0.1968, -0.74, -0.1923, 0.5054, -2.8745, -0.9773, 0.1483, -0.3109, 1.0416, -0.0437, 0.3293, -1.5629, 1.2061, 2.7797, 0.1322, 0.3702, -0.0653, 0.2951, -0.0423, -0.0532, 0.0021, -0.1126, -0.326, 0.188, 0.3202, -0.3057, 0.0921], [0.551, 0.1496, 0.8322, -1.0911, 0.9556, -0.6869, 1.2752, -0.5363, 0.0562, -0.3048, -0.7399, -0.1892, 0.5119, -3.1741, -0.8804, 0.161, -0.2886, 0.8737, -0.0802, 0.3866, -1.5269, 1.29, 3.0268, 0.1572, 0.481, -0.0343, 0.2568, -0.0116, -0.0971, -0.0593, -0.1262, -0.348, 0.0942, 0.4184, -0.2892, 0.0837, -0.0895], [0.5591, 0.1974, 0.8849, -1.2028, 0.8869, -0.7254, 1.2793, -0.6476, -0.1071, -0.4218, -0.7496, -0.2288, 0.5795, -2.9529, -0.8978, 0.2645, -0.3557, 0.8943, -0.2252, 0.3952, -1.4757, 1.4154, 3.342, 0.1617, 0.5836, -0.0061, 0.3326, -0.0033, -0.0599, -0.1103, -0.0948, -0.3351, -0.0713, 0.4691, -0.2934, 0.1159, -0.0869, -0.3025], [0.5807, 0.2184, 0.9776, -1.3112, 0.7911, -0.7151, 1.2029, -0.5767, -0.1907, -0.2721, -0.7251, -0.2084, 0.6505, -2.7301, -0.9824, 0.2625, -0.407, 0.9637, -0.2352, 0.3436, -1.5196, 1.5084, 3.0615, 0.175, 0.6103, -0.0297, 0.4489, -0.0047, -0.0182, -0.1153, -0.114, -0.3565, -0.1189, 0.4873, -0.2811, 0.1048, -0.0794, -0.2822, -0.106], [0.6056, 0.2122, 0.9541, -1.3378, 0.6669, -0.7262, 1.1588, -0.5936, -0.214, -0.3018, -0.74, -0.2031, 0.6092, -2.6234, -0.9226, 0.413, -0.379, 1.0398, -0.1926, 0.352, -1.5325, 1.6345, 2.9578, 0.1853, 0.6141, -0.0459, 0.4717, 0.003, -0.0107, -0.1145, -0.1251, -0.3639, -0.1204, 0.489, -0.2813, 0.1051, -0.0806, -0.2728, -0.0987, -0.1832], [0.6739, 0.175, 0.8339, -0.8755, 0.9044, -0.7782, 1.0842, -0.5358, -0.2461, -0.1356, -0.7388, -0.1381, 0.5651, -2.6217, -0.7384, 0.1222, -0.4121, 0.8775, -0.3442, 0.2958, -1.6602, 1.426, 2.8545, 0.2244, 0.5349, -0.0043, 0.4889, 0.0254, -0.3332, -0.1111, -0.1745, -0.3422, -0.0467, 0.5019, -0.2789, 0.0959, -0.0856, -0.2748, -0.1095, -0.0556, 0.3651], [0.6779, 0.1713, 0.837, -0.8226, 0.9402, -0.7754, 1.0609, -0.5495, -0.2232, -0.1484, -0.7495, -0.1391, 0.5826, -2.4919, -0.7341, 0.0569, -0.4253, 0.775, -0.2998, 0.2972, -1.6432, 1.3955, 2.7218, 0.2213, 0.5464, -0.0008, 0.4691, 0.0325, -0.3472, -0.1078, -0.1832, -0.3346, -0.0665, 0.4923, -0.2587, 0.1194, -0.0799, -0.2559, -0.0948, 0.0085, 0.3711, -0.0367], [0.588, 0.194, 1.3287, -1.4746, 0.9982, -0.743, 0.791, -0.4592, -0.2617, -0.2452, -0.7644, -0.1766, 0.5966, -2.715, -0.5019, 0.0593, -0.3793, 0.9665, -0.5135, 0.2681, -1.6024, 1.359, 2.839, 0.2061, 0.2886, 0.0494, 0.5189, 0.03, -0.1908, -0.1132, -0.1027, -0.3197, 1.2674, 0.4652, -0.2542, 0.1341, -0.0616, -0.2362, -0.0976, 0.1015, 0.3269, -0.0274, -1.1292], [0.6774, 0.1998, 1.2331, -1.8509, 1.1879, -0.7441, 1.3031, -0.3727, -0.2389, -0.3268, -0.7182, 0.1159, 0.5319, -2.7268, -0.5465, -0.4581, -0.5688, 1.492, -1.4101, 0.5025, -1.3261, 1.0384, 2.7483, 0.1957, -0.0762, 0.1341, 0.0452, 0.06, 0.4647, -0.1229, -0.1283, -0.2273, 1.5031, 0.4414, -0.2429, 0.0933, -0.0548, -0.2096, -0.1113, 0.712, 0.5441, -0.0258, -1.029, -0.6998], [0.6935, 0.2139, 1.0897, -1.8258, 1.2538, -0.7826, 1.2345, -0.2997, -0.1015, -0.3776, -0.6833, 0.1667, 0.5084, -2.8473, -0.3079, -0.4049, -0.6, 1.4231, -1.4987, 0.5237, -1.2544, 0.9897, 2.6189, 0.2103, -0.1493, 0.1586, -0.0688, 0.0639, 0.4022, -0.1527, -0.1369, -0.2538, 1.7623, 0.4514, -0.2379, 0.082, -0.0639, -0.2124, -0.101, 0.6551, 0.5913, -0.0213, -1.1867, -0.8255, 0.3065], [0.6952, 0.2162, 1.0856, -1.8149, 1.2529, -0.781, 1.2361, -0.3019, -0.0922, -0.3838, -0.6841, 0.1662, 0.5005, -2.8625, -0.3073, -0.4124, -0.6036, 1.4141, -1.4876, 0.5232, -1.2587, 0.997, 2.6252, 0.2104, -0.152, 0.1594, -0.0682, 0.0629, 0.407, -0.1552, -0.1399, -0.2564, 1.7768, 0.4524, -0.239, 0.084, -0.0641, -0.2088, -0.1011, 0.6593, 0.5929, -0.0234, -1.1978, -0.8297, 0.3089, 0.0058], [0.6952, 0.2162, 1.0857, -1.8127, 1.2512, -0.7807, 1.2387, -0.3006, -0.093, -0.3831, -0.684, 0.1657, 0.5001, -2.8597, -0.3094, -0.4126, -0.6035, 1.4119, -1.4851, 0.5221, -1.2609, 0.9986, 2.621, 0.2102, -0.152, 0.1596, -0.0658, 0.0629, 0.4079, -0.1555, -0.1394, -0.2564, 1.775, 0.4525, -0.2389, 0.0842, -0.0642, -0.2093, -0.101, 0.6602, 0.5919, -0.0236, -1.1958, -0.8289, 0.3094, 0.0058, -0.004], [0.6038, 0.171, 0.4457, -0.8013, 1.1288, -0.7712, 1.2955, -0.341, 0.2018, -0.4369, -0.7451, 0.0401, 0.5651, -2.4763, -0.2679, -0.3635, -0.7268, 0.6839, -1.1533, 0.5419, -1.1031, 1.1247, 2.0336, 0.2047, -0.4716, 0.1467, 0.0479, 0.0571, 0.3705, -0.1464, -0.1143, -0.219, 1.9415, 0.4935, -0.2918, 0.0968, -0.0663, -0.2176, -0.1253, 0.6535, 0.5042, 0.0074, -1.03, -0.7453, 0.5771, 0.0155, 0.7046, -1.0352], [0.5935, 0.1898, 0.6097, -0.7526, 1.1752, -0.7293, 1.2657, -0.4139, 0.4784, -0.7193, -0.7356, 0.0665, 0.5684, -2.4324, 0.019, -0.3289, -0.5997, 0.1621, -1.1107, 0.5907, -0.8939, 1.1391, 2.1008, 0.1865, -0.2312, 0.145, -0.2249, 0.0782, 0.1022, -0.1747, -0.0871, -0.2202, 1.932, 0.5122, -0.3016, 0.0692, -0.0688, -0.2026, -0.1349, 0.6034, 0.4534, 0.0324, -1.2578, -0.8567, 0.5528, 0.0058, 0.4625, -1.5795, 0.9694], [0.6193, 0.179, 0.6187, -0.656, 1.0766, -0.7452, 1.3627, -0.404, 0.2956, -0.7213, -0.7368, 0.0439, 0.595, -2.3008, -0.1069, -0.2596, -0.5695, 0.1506, -1.1415, 0.57, -0.9997, 1.2404, 2.1903, 0.1856, -0.2858, 0.1565, -0.0359, 0.0822, 0.0631, -0.1927, -0.0995, -0.2143, 1.8023, 0.5446, -0.3187, 0.0746, -0.0727, -0.0193, -0.1342, 0.5878, 0.4337, 0.06, -1.07, -0.7769, 0.5592, -0.0208, 0.4978, -1.6043, 0.8916, -0.3871], [0.6177, 0.1884, 0.5481, -0.5933, 1.146, -0.7375, 1.2735, -0.4147, 0.2845, -0.6827, -0.7387, 0.064, 0.594, -2.2171, -0.0468, -0.3004, -0.5686, 0.1176, -1.1681, 0.5834, -1.042, 1.3029, 2.0975, 0.1826, -0.3069, 0.1506, -0.0531, 0.1048, 0.0632, -0.1867, -0.1074, -0.2862, 1.8303, 0.5377, -0.3133, 0.0922, -0.0798, -0.0233, -0.1445, 0.5922, 0.4699, 0.0649, -1.0584, -0.7962, 0.5017, 0.0126, 0.5464, -1.6321, 0.9283, -0.3339, -0.0555], [0.6313, 0.1743, 0.519, -0.2424, 0.9815, -0.7353, 1.0787, -0.462, 0.2898, -0.553, -0.7492, 0.0811, 0.5506, -2.1975, 0.0847, -0.3202, -0.5166, 0.0601, -1.4834, 0.529, -0.9868, 1.432, 2.1204, 0.1832, -0.4577, 0.1352, -0.0852, 0.1143, 0.1485, -0.1772, -0.1106, -0.2905, 2.0253, 0.5394, -0.3111, 0.0648, -0.0826, -0.0257, -0.1413, 0.7533, 0.4126, 0.0753, -1.0831, -0.7674, 0.4841, -0.0007, 0.6497, -1.5456, 0.8282, -0.3066, -0.0461, -0.2607], [0.621, 0.1748, 0.4793, -0.3181, 1.026, -0.7122, 1.0941, -0.4727, 0.2878, -0.518, -0.751, 0.0838, 0.5536, -2.0999, 0.1407, -0.3042, -0.5316, 0.0241, -1.5097, 0.5291, -0.9994, 1.4004, 2.0428, 0.155, -0.4058, 0.138, -0.0982, 0.1091, 0.1352, -0.1764, -0.1023, -0.2904, 1.913, 0.5347, -0.3178, 0.0599, -0.0785, -0.0211, -0.1452, 0.724, 0.4308, 0.083, -1.0215, -0.7822, 0.5057, -0.0018, 0.6815, -1.6183, 0.8745, -0.3186, -0.0398, -0.2194, 0.0595], [0.6456, 0.1932, 0.4229, -0.1785, 1.0384, -0.7313, 1.0795, -0.4579, 0.2828, -0.4804, -0.7467, 0.0816, 0.5589, -2.1059, 0.1334, -0.2861, -0.5358, -0.0427, -1.5285, 0.521, -1.0237, 1.4228, 2.0204, 0.1723, -0.3878, 0.1406, -0.0714, 0.1009, 0.1156, -0.1938, -0.1119, -0.301, 1.8779, 0.5302, -0.3142, 0.0659, -0.0719, -0.0169, -0.1482, 0.7536, 0.4343, 0.0862, -0.9984, -0.784, 0.5391, 0.0064, 0.6807, -1.6563, 0.8914, -0.3265, -0.0424, -0.2429, 0.0283, -0.032]]

orca_functionals = ['DSD-BLYP', 'wB97X-V', 'B1LYP', 'B2PLYP', 'TPSSh', 'DSD-PBEP86', 'PWPB95', 'DSD-PBEB95', 'B3PW91', 'MPW2PLYP', 'rPW86PBE', 'RPBE', 'OLYP', 'B3P86', 'B3LYP', 'B1P86', 'O3LYP', 'BPBE', 'wB97X-D3', 'XLYP', 'mPW1LYP', 'PW1PW', 'mPWLYP', 'TPSS0', 'X3LYP', 'OPBE', 'revPBE', 'TPSS']

orca_weights = [[0.5555, 0.4437], [0.6347, 0.4676, -0.104], [0.5193, 0.4317, -0.1747, 0.2219], [0.3854, 0.4313, -0.1537, 0.4247, -0.0897], [0.0837, 0.4142, -0.142, 0.5616, -0.1329, 0.213], [-0.1911, 0.3109, -0.134, 0.7231, -0.2, 0.3307, 0.1587], [-0.4025, 0.3075, -0.1096, 0.7345, -0.2762, 0.6576, 0.3946, -0.3072], [-0.5107, 0.3051, -0.0962, 0.807, -0.2575, 0.7296, 0.398, -0.3055, -0.0704], [-0.7167, 0.3624, -0.0832, 1.1629, -0.2122, 0.8968, 0.4104, -0.3394, -0.2031, -0.2766], [-1.3877, 0.3007, -0.0573, 1.9192, -0.158, 1.1673, 0.4516, -0.4051, -0.2521, -0.3765, -0.2002], [-1.6633, 0.3088, -0.0928, 2.3222, -0.2124, 1.2165, 0.385, -0.3835, -0.2713, -0.3966, -0.3085, 0.0976], [-2.2195, 0.3251, -0.2334, 3.18, -0.2129, 1.2043, 0.3058, -0.2678, -0.2396, -0.4749, -0.4652, 0.2906, -0.1936], [-2.486, 0.3333, -0.2408, 3.4154, -0.1847, 1.4027, 0.3799, -0.4095, -0.0707, -0.5137, -0.4714, 0.2767, -0.2082, -0.2213], [-2.4906, 0.3318, -0.2594, 3.4004, -0.1878, 1.4287, 0.3935, -0.425, -0.0791, -0.5345, -0.4838, 0.2842, -0.2126, -0.2285, 0.0647], [-2.5148, 0.3314, -0.2665, 3.4138, -0.1816, 1.4423, 0.3979, -0.4315, -0.0462, -0.5165, -0.5178, 0.2871, -0.2283, -0.0614, 0.0666, -0.1737], [-2.5208, 0.3331, -0.255, 3.3688, -0.1874, 1.4587, 0.4155, -0.4573, -0.0083, -0.4731, -0.5492, 0.3072, -0.2398, 0.1164, 0.0641, -0.3438, -0.0285], [-2.5176, 0.3332, -0.2457, 3.3301, -0.1916, 1.467, 0.429, -0.4722, -0.033, -0.4409, -0.5576, 0.3063, -0.2395, 0.1088, 0.0643, -0.3318, -0.0319, 0.0239], [-2.5155, 0.3372, -0.2463, 3.3291, -0.1928, 1.4665, 0.4289, -0.4722, -0.0252, -0.4434, -0.5536, 0.3045, -0.2385, 0.1073, 0.0643, -0.3315, -0.032, 0.0195, -0.0056], [-2.9536, 0.2503, -0.6375, 3.5981, -0.2052, 1.7912, 0.5823, -0.6634, 0.4315, -0.4048, -0.6403, 0.1283, -0.1998, -0.3294, 0.0596, -0.0609, -0.0347, -0.1811, 0.0932, 0.3751], [-2.9443, 0.2502, -0.6377, 3.5954, -0.2047, 1.784, 0.5797, -0.6604, 0.4277, -0.4074, -0.6419, 0.1305, -0.2009, -0.3242, 0.0596, -0.0634, -0.0346, -0.1788, 0.0937, 0.3732, 0.0032], [-2.68, 0.2819, -0.3105, 3.6319, -0.2, 1.5303, 0.465, -0.4992, -0.0349, -0.6068, -0.695, 0.0476, -0.1039, 0.4371, 0.0605, -0.6358, -0.233, -0.0944, 0.0622, 0.3659, -0.3025, 0.513], [-2.458, 0.2675, -0.36, 2.7048, -0.19, 1.4828, 0.7573, -0.9253, -0.3392, 0.5035, -0.4692, -0.0495, 0.0228, 0.8499, 0.0627, -0.98, -0.1849, 0.1839, 0.1123, 0.4556, -0.2054, 0.283, -0.5296], [-2.474, 0.2702, -0.3678, 2.6883, -0.108, 1.5005, 0.7731, -0.9467, -0.2773, 0.529, -0.4706, -0.0563, 0.0297, 0.7765, 0.0625, -0.9078, -0.1854, 0.1439, 0.1104, 0.4738, -0.2043, 0.2846, -0.5486, -0.1005], [-2.2756, 0.2615, -0.7048, 2.7488, -0.075, 1.4003, 0.7913, -0.968, -0.3194, 0.3143, -0.4777, -0.0336, 0.0497, 0.4445, 0.0599, -0.6285, -0.2205, 0.1783, 0.1316, 0.4181, -0.1984, 0.252, -0.5798, -0.1322, 0.5582], [-2.5002, 0.2596, -0.2821, 2.6632, -0.0169, 1.5222, 0.6967, -0.8512, -0.0491, 0.5002, -0.4925, 0.0106, 0.3072, 0.5894, 0.0574, -0.8031, -0.0823, 0.1885, 0.1323, 0.3925, -0.197, 0.282, -0.5846, -0.201, -0.2326, -0.3152], [-2.8544, 0.2245, -0.172, 2.8481, -0.0072, 1.778, 0.7077, -0.8128, -0.0682, 0.3946, -0.3589, -0.0528, -0.0221, 0.4934, 0.056, -0.8277, -0.3185, 0.1206, 0.1549, 0.5327, -0.2456, 0.3046, -0.1952, -0.2039, -0.0716, 0.3617, -0.7733], [-2.8125, 0.227, 0.1399, 2.6614, -0.8118, 1.6781, 0.621, -0.6667, 0.0416, 0.5711, -0.4127, -0.0279, -0.12, 0.5254, 0.0514, -0.7233, -0.3087, -0.0617, 0.1323, 0.4422, -0.2465, 0.2811, -0.1621, -0.2327, -0.2551, 0.4439, -0.7262, 0.7436]]