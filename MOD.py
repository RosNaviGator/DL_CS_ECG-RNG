

# system imports
import os

# third party imports
import numpy as np
import scipy.sparse as sp

# local imports
from utils import *
from OMP import OMP







def I_findDistanceBetweenDictionaries(original, new):
    """
    Calculates the distance between two dictionaries.

    Parameters:
    ----------
    original : numpy.ndarray
        The original dictionary.

    new : numpy.ndarray
        The new dictionary.

    Returns:
    -------
    catchCounter : int
        The number of elements that satisfy the condition errorOfElement < 0.01.
    totalDistances : float
        The sum of all errorOfElement values.
    
    
    """

    # first: all the columns in the original start with positive values
    catchCounter = 0
    totalDistances = 0

    for i in range(new.shape[1]):
        new[:,i] = np.sign(new[0,i]) * new[:,i]

    for i in range(original.shape[1]):
        d = np.sign(original[0,i]) * original[:,i]
        distances = np.sum(new - np.tile(d, (1, new.shape[1])), axis=0)
        index = np.argmin(distances)
        errorOfElement = 1 - np.abs(new[:,index].T @ d)
        totalDistances += errorOfElement
        catchCounter += errorOfElement < 0.01

    ratio = catchCounter / original.shape[1]
    return ratio, totalDistances



def MOD(data, parameters):
    """
    MOD (Method of Optimal Directions) algorithm for dictionary learning.

    The MOD algorithm is a method for learning a dictionary for sparse representation of signals.
    It iteratively updates the dictionary to best represent the input data with sparse coefficients
    using Orthogonal Matching Pursuit (OMP).

    This implementation is based on the original MATLAB version by Hadi Zanddizari 
    and has been implemented in Python by RosNaviGator (2024).

    References
    ----------
    Engan, K.; Aase, S.O.; Hakon Husoy, J. (1999). "Method of optimal directions for frame design".
    1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No.99CH36258).
    Vol. 5, pp. 2443–2446. doi:10.1109/ICASSP.1999.760624.

    Parameters
    ----------
    data : numpy.ndarray
        An (n x N) matrix containing N signals, each of dimension n.
    
    parameters : dict
        A dictionary containing the parameters for the MOD algorithm:
            - K : int
                The number of dictionary elements (columns) to train.
            
            - numIterations : int
                The number of iterations to perform for dictionary learning.
            
            - InitializationMethod : str
                Method to initialize the dictionary. Options are:
                * 'DataElements' - Initializes the dictionary using the first K data signals.
                * 'GivenMatrix' - Initializes the dictionary using a provided matrix (requires 'initialDictionary' key).

            - initialDictionary : numpy.ndarray, optional
                The initial dictionary matrix to use if 'InitializationMethod' is set to 'GivenMatrix'.
                It should be of size (n x K).

            - L : int
                The number of non-zero coefficients to use in OMP for sparse representation of each signal.

    Returns
    -------
    dictionary : numpy.ndarray
        The trained dictionary of size (n x K), where each column is a dictionary element.

    coef_matrix : numpy.ndarray
        The coefficient matrix of size (K x N), representing the sparse representation of the input data
        using the trained dictionary.
    """


    if data.shape[1] < parameters['K']:
        print("MOD: Number of signals is smaller than the dictionary size. Trivial solution...")
        dictionary = data[:, :data.shape[1]]
        coef_matrix = np.eye(data.shape[1])  # trivial coefficients
        return dictionary, coef_matrix
    
    elif parameters['InitializationMethod'] == 'DataElements':
        dictionary = data[:,:parameters['K']]
    elif parameters['InitializationMethod'] == 'GivenMatrix':
        if 'initialDictionary' not in parameters:
            raise ValueError("initialDictionary parameter is required when \
                             InitializationMethod is set to 'GivenMatrix'.")
        dictionary = parameters['initialDictionary']

    # Data arrives here as int16, so we need to convert it to float64
    dictionary = dictionary.astype(np.float64)
    # normalize dictionary
    dictionary = dictionary @ np.diag(1. / np.sqrt(np.sum(dictionary ** 2, axis=0)))
    dictionary = dictionary * np.tile(np.sign(dictionary[0, :]), (dictionary.shape[0], 1))

    for iterNum in range(parameters['numIterations']):
        # find coeffs
        # should try and use the one from sklearn
        coef_matrix = OMP(dictionary, data, parameters['L'])  # use the one written by me
        # improve dictionary
        dictionary = data @ coef_matrix.T @ np.linalg.inv(
            coef_matrix @ coef_matrix.T + 1e-7 * sp.eye(coef_matrix.shape[0]))
        dictionary = np.asarray(dictionary)
        dictionary = dictionary @ np.diag(1 /  np.sqrt(np.sum(dictionary ** 2, axis=0)))

    
    return dictionary, coef_matrix
