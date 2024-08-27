
# system imports

# third party imports
import numpy as np

# local imports
from utils import py_test_csv



def OMP(D, X, L):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse coding.

    This function implements the OMP algorithm, which is used to find the sparse representation of a signal over a given dictionary.
    For more details, refer to the paper "Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition" by J. A. Tropp and A. C. Gilbert, 
    published in the Asilomar Conference on Signals, Systems, and Computers, 2005.

    Parameters
    ----------
    D : numpy.ndarray
        The dictionary to use for sparse coding. It should be a matrix of size (n x K), where n is the signal dimension and K is the number of atoms in the dictionary.
        (its columns MUST be normalized).
    
    X : numpy.ndarray
        The signals to represent using the dictionary. It should be a matrix of size (n x N), where N is the number of signals.
    
    L : int
        The maximum number of coefficients to use for representing each signal.
    
    Returns
    -------
    A : numpy.ndarray
        The sparse representation of the signals over the dictionary. It should be a matrix of size (K x N).
    """

    [_, P] = X.shape
    [_, K] = D.shape

    A = np.zeros((K, P))
    
    for k in range(P):
        x = X[:, k]
        residual = x
        indx = np.array([], dtype=int)
        for j in range(L):         
            proj = D.T @ residual
            pos = np.argmax(np.abs(proj))
            indx = np.append(indx, pos.astype(int))
            a = np.linalg.pinv(D[:, indx]) @ x
            residual = x - D[:, indx] @ a
            if np.sum(residual ** 2) < 1e-6:
                break
        temp = np.zeros((K,))
        temp[indx] = a
        A[:, k] = temp

    return A