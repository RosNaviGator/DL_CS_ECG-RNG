import numpy as np
from MOD import OMP

# Display the separator lines
print('----------------------------------------')
print('----------------------------------------')
print('----------------------------------------')

# Define a 4x4 dictionary D (columns must be normalized)
D = np.array([
    [0.5, 0.4, 0.7, 0.1],
    [0.5, -0.8, 0.2, -0.3],
    [0.5, 0.2, -0.7, 0.9],
    [0.5, 0.4, 0.2, 0.2]
])

D = np.ones((4, 4)) - 2 * np.eye(4) + 0.1 * np.linalg.inv(np.ones((4, 4)) + 2 * np.eye(4))


# Display the dictionary D
print(f'Dictionary D:\n{D}')

# Normalize columns of D
D = D / np.linalg.norm(D, axis=0)

# Define a 4x1 signal X
X = np.array([0.8, -0.6, 0.3, 0.9])

# Set the maximum number of coefficients L
L = 3

# Run the OMP function
A = OMP(D, X.reshape(-1, 1), L)  # Reshape X to ensure it's 2D like in MATLAB

# Display the output sparse coefficient matrix A
print(f'Sparse coefficient matrix A:\n{A}')
