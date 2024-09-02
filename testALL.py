"""
Tests the MOD.py functionality
"""

# system imports
import os

# third party imports
import numpy as np

# local imports
from MOD import MOD
from KSVD import KSVD
from utils import py_test_csv



# Use the exact matrices provided for testing
Data = np.array([
    [0.814724, 0.632359, 0.957507, 0.957167, 0.421761, 0.655741, 0.678735, 0.655478, 0.276923, 0.694829, 0.438744, 0.186873, 0.709365, 0.655098, 0.959744, 0.751267],
    [0.905792, 0.097540, 0.964889, 0.485376, 0.915736, 0.035712, 0.757740, 0.171187, 0.046171, 0.317099, 0.381558, 0.489764, 0.754687, 0.162612, 0.340386, 0.255095],
    [0.126987, 0.278498, 0.157613, 0.800280, 0.792207, 0.849129, 0.743132, 0.706046, 0.097132, 0.950222, 0.765517, 0.445586, 0.276025, 0.118998, 0.585268, 0.505957],
    [0.913376, 0.546882, 0.970593, 0.141886, 0.959492, 0.933993, 0.392227, 0.031833, 0.823458, 0.034446, 0.795200, 0.646313, 0.679703, 0.498364, 0.223812, 0.699077]
])

InitialDictionary = np.array([
    [0.455898, -0.269588, 0.532811, -0.679457, -0.132875, 0.761190, -0.146094, 0.037887],
    [0.639648, -0.385544, -0.137048, -0.689199, 0.961934, -0.385685, 0.129283, 0.260924],
    [0.035595, 0.853635, 0.632905, 0.236550, 0.197621, 0.333982, -0.698863, 0.934044],
    [-0.617851, -0.223573, -0.544757, -0.085946, 0.134066, 0.400366, -0.688138, -0.240923]
])


"""
print('Original Data:')
print(f'Original Data shape: {Data.shape}')
printFormatted(Data)
print('Initial Dictionary:')
print(f'Initial Dictionary shape: {InitialDictionary.shape}')
printFormatted(InitialDictionary)
"""
K = int(2 * Data.shape[0])
InitialDictionary = np.random.randn(Data.shape[0], K)
for i in range(K):
    InitialDictionary[:, i] = \
        InitialDictionary[:, i] / np.linalg.norm(InitialDictionary[:, i])

# Define parameters
param = {
    'K': K,  # num of atoms dict, atom = basis function
    'L': 1,
    'num_iterations': 10,
    'preserve_dc_atom': 0,
    'initialization_method': 'DataElements',  # or 'GivenMatrix'
    'initial_dictionary': InitialDictionary  # random initialization of dictionary
}

# Normalize the initial dictionary





# Run 
mod_dict, coef_matrix = MOD(Data, param)
ksvd_dict, X = KSVD(Data, param)




# Prepare output files
OUTPUT_DIR = 'debugCsvPy'  # Directory where CSV files will be stored
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mod_dict_path = os.path.join(OUTPUT_DIR, 'py_MOD.csv')
np.savetxt(mod_dict_path, mod_dict, delimiter=',', fmt='%.6f')
ksvd_dict_path = os.path.join(OUTPUT_DIR, 'py_KSVD.csv')
np.savetxt(ksvd_dict_path, ksvd_dict, delimiter=',', fmt='%.6f')

