# system imports
import os

# third party imports
import numpy as np
from scipy.sparse.linalg import svds

# local imports
from OMP import OMP


def svds_vector(v):
    """
    IF YOU WANT TO USE svds AND NOT full svd:
    svds doesn't work if you apply it to a 1D array, you can handle the 
    case separately with this function

    Computes the singular value decomposition of a vector or 2D matrix with one dimension equal to 1.

    Parameters:
    v (numpy array): Input vector (1D or 2D with one dimension equal to 1).

    Returns:
    s (float): Singular value, which is the norm of the vector.
    u (numpy array): Normalized version of the original vector (left singular vector).
    vt (numpy array): 1x1 matrix with a single entry [1] (right singular vector).
    """
    
    # Ensure input is a numpy array
    v = np.asarray(v)
    
    # Check if the input is a 1D array or a 2D array with one dimension equal to 1
    if v.ndim == 1:
        # Reshape to a 2D column vector
        v = v.reshape(-1, 1)
    elif v.ndim == 2 and (v.shape[0] == 1 or v.shape[1] == 1):
        # It's already in the correct 2D form
        pass
    else:
        raise ValueError("Input must be a vector or a 2D array with one dimension equal to 1.")
    
    # Compute the singular value as the norm of the vector
    s = np.linalg.norm(v)

    # Compute the left singular vector (u) by normalizing the input vector
    if s > 0:
        u = v / s
    else:
        u = np.zeros_like(v)
    
    # The right singular vector (v) is just a 1x1 matrix [1]
    vt = np.array([[1]])

    return u, s, vt



def I_findBetterDictionaryElement(data, dictionary, j,  coeff_matrix, numCoefUsed=1):
    """
    """


    relevantDataIndices = np.nonzero(coeff_matrix[j, :])[0]

    if relevantDataIndices.size == 0:
        errorMat = data - dictionary @ coeff_matrix
        errorNormVec = np.sum(errorMat ** 2, axis=0)
        i = np.argmax(errorNormVec)
        betterDictionaryElement = data[:, i]
        betterDictionaryElement = betterDictionaryElement / np.sqrt(betterDictionaryElement.T @ betterDictionaryElement)
        betterDictionaryElement = betterDictionaryElement * np.sign(betterDictionaryElement[0])
        coeff_matrix[j, :] = 0
        newVectAdded = 1
        return betterDictionaryElement, coeff_matrix, newVectAdded
    
    newVectAdded = 0
    tmpCoefMatrix = coeff_matrix[:, relevantDataIndices]
    tmpCoefMatrix[j, :] = 0
    errors = data[:, relevantDataIndices] - dictionary @ tmpCoefMatrix

    # SVD
    if (np.min(errors.shape) <= 1):
        u, s, vt = svds_vector(errors)
        betterDictionaryElement = u
        singularValue = s
        betaVector = vt
    else:
        u, s, vt = svds(errors, k=1)
        betterDictionaryElement = u[:, 0]  # left singular vector (corresponding to the largest singular value)
        singularValue = s[0]               # largest singular value
        betaVector = vt[0, :]              # right singular vector

    coeff_matrix[j, relevantDataIndices] = singularValue * betaVector.T

    return betterDictionaryElement, coeff_matrix, newVectAdded



def I_clearDictionary(dictionary, coeff_matrix, data):
    """
    """

    T2 = 0.99
    T1 = 3
    K = dictionary.shape[1]
    Er = np.sum((data - dictionary @ coeff_matrix) ** 2, axis=0)
    G = dictionary.T @ dictionary
    G = G - np.diag(np.diag(G))
    for jj in range(K):
        
        # abs seems useless
        if np.max(G[jj, :]) > T2 or np.count_nonzero(np.abs(coeff_matrix[jj, :]) > 1e-7) <= T1:
            pos = np.argmax(Er)
            Er[pos] = 0
            dictionary[:, jj] = data[:, pos] / np.linalg.norm(data[:, pos])
            G = dictionary.T @ dictionary
            G = G - np.diag(np.diag(G))
    
    return dictionary





def KSVD(data, param):
    """
    """


    # Check if elements of dataset needs to be preserved
    #  as dictionary atom (useful for natural images)
    if(param['preserveDCAtom'] > 0):
        fixedDictElem = np.zeros((data.shape[0], 1))  
        fixedDictElem[:data.shape[0], 0] = 1 / np.sqrt(data.shape[0])
    else:
        fixedDictElem = np.empty((0, 0))

    # Dictionary Initialization
    if(data.shape[1] < param['K']):
        print('KSVD: Size of data is smaller than the dictionary size. Trivial solution...')
        dictionary = data[:, :data.shape[1]]
        coef_matrix = np.eye(data.shape[1])  # trivial coefficients
        return dictionary, coef_matrix
    
    dictionary = np.zeros((data.shape[0], param['K']), dtype=np.float64)    
    if(param['InitializationMethod'] == 'DataElements'):
        dictionary[:, :param['K'] - param['preserveDCAtom']] = \
            data[:, :param['K'] - param['preserveDCAtom']]
    elif(param['InitializationMethod'] == 'GivenMatrix'):
        dictionary[:, :param['K'] - param['preserveDCAtom']] = \
            param['initialDictionary'][:, :param['K'] - param['preserveDCAtom']]

    


    # Adjust dicitonary for fixed elements
    if(param['preserveDCAtom']):
        tmpMat = fixedDictElem @ np.linalg.lstsq(dictionary)
        dictionary = dictionary - fixedDictElem @ tmpMat

    # Normalize
    dictionary = dictionary @ np.diag(1 / np.sqrt(np.sum(dictionary ** 2, axis=0)))
    dictionary = dictionary * np.tile(np.sign(dictionary[0, :]), (dictionary.shape[0], 1))

    ## KSVD algorithm
    for iterNum in range(param['numIterations']):
        # Compute coefficients with OMP
        coef_matrix = OMP(
            np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary,
            data,
            param['L']
            )
        # Improve dict elems      
        #rand_perm = np.random.permutation(dictionary.shape[1])
        rand_perm = np.array([2, 4, 6, 3, 1, 7, 0, 5])
        for j in rand_perm:
            betterDictElem, coef_matrix, newVectAdded = I_findBetterDictionaryElement(
                data,
                np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary,
                j + fixedDictElem.shape[1],
                coef_matrix,
                param['L']
                )


            # substitute the j-th
            dictionary[:, j] = betterDictElem.ravel()
            # preserveDCAtom case
            if(param['preserveDCAtom']):
                tmpCoeff = fixedDictElem @ np.linalg.lstsq(betterDictElem)
                dictionary[:, j] = betterDictElem - fixedDictElem @ tmpCoeff
                dictionary[:, j] = dictionary[:, j] / np.sqrt(dictionary[:, j].T @ dictionary[:, j])

        # clear dictionary
        dictionary = I_clearDictionary(dictionary, coef_matrix[fixedDictElem.shape[1]:, :], data)

    # last hstack
    dictionary = np.hstack((fixedDictElem, dictionary)) if fixedDictElem.size > 0 else dictionary
    
    return dictionary, coef_matrix





    

