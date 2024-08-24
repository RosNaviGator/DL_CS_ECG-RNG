# Tests:
# 1. sparseSignal

import matplotlib.pyplot as plt
import numpy as np
from sparseSignalGenerator import sparseSignal
from sparseDictionaries import generate_DCT_dictionary
from measurementMatrix import generate_DBDD_matrix
from SL0 import SL0
from evaluation import plot_signals

# TO test let's do a compressive sensing without blocks (whole signal)

DIM = 2 ** 10
K = int(DIM * 0.34)

# Generate sparse Signal
signal, active_indices = sparseSignal(DIM, K=K, sigma_inactive=0., sigma_active=0., fixedActiveValue=1)
plt.title("Sparse signal")
plt.plot(signal)
plt.show()

# Find "non sparse" representation of signal
Dict_DCT = generate_DCT_dictionary(DIM)
x = Dict_DCT @ signal
plt.title("Non sparse signal")
plt.plot(x)
plt.show()

# Generate measurement matrix
CR = 1/4  # Compression ratio
DIM_comp = int(DIM * CR)
Phi = generate_DBDD_matrix(DIM_comp, DIM)

# Compress signal
y = Phi @ x

# Reconstruct signal
sigma_min = 0.001
s_d_f = 0.5
mu = 2
l = 3
Phi_pinv = np.linalg.pinv(Phi)
s_hat = SL0(y, Phi, sigma_min=sigma_min, sigma_decrease_factor=s_d_f, mu_0=mu, L=l, A_pinv=Phi_pinv, showProgress=True)

# evaluation
plot_signals(signal, s_hat, "Original", "After processing")





