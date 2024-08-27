"""
Tests the MOD.py functionality
"""

import os
import numpy as np
from MOD import MOD
from KSVD import KSVD

# Use the exact matrices provided for testing

# Original Data
Data = np.array([
    [0.8147, 0.6324, 0.9575, 0.9572, 0.4218, 0.6557, 0.6787, 0.6555, 0.2769, 0.6948, 0.4387, 0.1869, 0.7094, 0.6551, 0.9597, 0.7513],
    [0.9058, 0.0975, 0.9649, 0.4854, 0.9157, 0.0357, 0.7577, 0.1712, 0.0462, 0.3171, 0.3816, 0.4898, 0.7547, 0.1626, 0.3404, 0.2551],
    [0.1270, 0.2785, 0.1576, 0.8003, 0.7922, 0.8491, 0.7431, 0.7060, 0.0971, 0.9502, 0.7655, 0.4456, 0.2760, 0.1190, 0.5853, 0.5060],
    [0.9134, 0.5469, 0.9706, 0.1419, 0.9595, 0.9340, 0.3922, 0.0318, 0.8235, 0.0344, 0.7952, 0.6463, 0.6797, 0.4984, 0.2238, 0.6991]
])


# Initial Dictionary
InitialDictionary = np.array([
    [0.1873, -1.7947, -0.5445, 0.7394, -0.8396, 0.1240, -1.2078, -1.0582],
    [-0.0825, 0.8404, 0.3035, 1.7119, 1.3546, 1.4367, 2.9080, -0.4686],
    [-1.9330, -0.8880, -0.6003, -0.1941, -1.0722, -1.9609, 0.8252, -0.2725],
    [-0.4390, 0.1001, 0.4900, -2.1384, 0.9610, -0.1977, 1.3790, 1.0984]
])



Data = Data  # scale data


"""
print('Original Data:')
print(f'Original Data shape: {Data.shape}')
printFormatted(Data)
print('Initial Dictionary:')
print(f'Initial Dictionary shape: {InitialDictionary.shape}')
printFormatted(InitialDictionary)
"""


# Define parameters
param = {
    'K': 2 * Data.shape[0],  # num of atoms dict, atom = basis function
    'L': 1,
    'numIterations': 10,
    'preserveDCAtom': 0,
    'InitializationMethod': 'DataElements',  # or 'GivenMatrix'
    'TrueDictionary': np.eye(2),
    'initialDictionary': InitialDictionary  # random initialization of dictionary
}

# Normalize the initial dictionary
for i in range(param['K']):
    param['initialDictionary'][:, i] = \
        param['initialDictionary'][:, i] / np.linalg.norm(param['initialDictionary'][:, i])




# Run 
mod_dict, coef_matrix = MOD(Data, param)
ksvd_dict, X = KSVD(Data, param)




# Prepare output files
OUTPUT_DIR = 'debugCsvPy'  # Directory where CSV files will be stored
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mod_dict_path = os.path.join(OUTPUT_DIR, 'py_MOD.csv')
np.savetxt(mod_dict, mod_dict_path, delimiter=',', fmt='%.6f')
ksvd_dict_path = os.pathjoin(OUTPUT_DIR, 'py_KSVD.csv')
np.savetxt(ksvd_dict, ksvd_dict_path, delimiter=',', fmt='%.6f')


