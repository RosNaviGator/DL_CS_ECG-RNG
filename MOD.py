

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
    This implementation takes the input data and learns a dictionary that can sparsely represent 
    the signals using Orthogonal Matching Pursuit (OMP).

    Parameters
    ----------
    data : numpy.ndarray
        An (n x N) matrix containing N signals, each of dimension n.
    
    parameters : dict
        A dictionary containing the parameters for the MOD algorithm:
            - K : int
                The number of dictionary elements to train.
            
            - numIterations : int
                The number of iterations to perform.
            
            - InitializationMethod : str
                Method to initialize the dictionary. Options are:
                * 'DataElements' - Initializes the dictionary using the data signals.
                * 'GivenMatrix' - Initializes the dictionary using a provided matrix (requires 'initialDictionary').

            - initialDictionary : numpy.ndarray, optional
                The matrix to use for dictionary initialization if 'InitializationMethod' is set to 'GivenMatrix'.
            
            - L : int
                The number of non-zero coefficients to use in OMP for coefficient calculation.

    Returns
    -------
    dictionary : numpy.ndarray
        The trained dictionary of size (n x K).

    output : dict
        A dictionary containing the coefficient matrix ('coef_matrix') obtained after the final iteration.
    """

    # intialize
    output = {}
    dictionary = None


    if data.shape[1] < parameters['K']:
        print("Number of signals is smaller than the dictionary size. Trivial solution...")
        dictionary = data[:, :data.shape[1]]
        return dictionary, {}
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

    output['coef_matrix'] = coef_matrix
    return dictionary, output
