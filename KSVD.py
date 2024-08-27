

# third party imports
import numpy as np
from scipy.sparse.linalg import svds

# local imports
from OMP import OMP



def I_findBetterDictionaryElement(data, dictionary, j,  coeff_matrix, numCoefUsed=1):
    """
    """

    betterDictionaryElement = None

    relevantDataIndices = np.nonzero(coeff_matrix[j, :])[0]
    if not relevantDataIndices:
        errorMat = data - dictionary @ coeff_matrix
        errorNormVec = np.sum(errorMat ** 2, axis=0)
        i = np.argmax(errorNormVec)
        betterDictionaryElement = data[:, i]
        betterDictionaryElement = betterDictionaryElement / np.sqrt(betterDictionaryElement.T @ betterDictionaryElement)
        betterDictionaryElement = betterDictionaryElement * np.sign(betterDictionaryElement[0, 0])
        coeff_matrix[j, :] = 0
        newVectAdded = 1
        return betterDictionaryElement, coeff_matrix, newVectAdded
    
    newVectAdded = 0
    tmpCoefMatrix = coeff_matrix[:, relevantDataIndices]
    tmpCoefMatrix[j, :] = 0
    errors = data[:, relevantDataIndices] - dictionary @ tmpCoefMatrix

    # SVD
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
    Er = np.sum(data - dictionary @ coeff_matrix, axis=0)
    G = dictionary.T @ dictionary
    G = G - np.diag(np.diag(G))
    for jj in range(K):
        # abs seems useless
        if np.max(G[jj, :]) > T2 or np.count_nonzero(np.abs(coeff_matrix[jj, :]) > 1e-7 ):
            pos = np.argmax(Er)
            Er[pos] = 0
            dictionary[:, jj] = data[:, pos / np.linalg.norm(daata[:, pos])]
            G = dictionary.T @ dictionary
            G = G - np.diag(np.diag(G))
    
    return dictionary





def KSVD(data, param):
    """
    """

    # initialize dictionary
    dictionary = None

    # Check if elements of dataset needs to be preserved
    #  as dictionary atom (useful for natural images)
    if(param['preserveDCAtom'] > 0):
        fixedDictElem = None
        fixedDictElem[:data.shape[0], 0] = 1 / np.sqrt(data.shape[0])
    else:
        fixedDictElem = None

    # Dictionary Initialization
    if(data.shape[1] < param['K']):
        print('Size of data is smaller than the dictionary size. Trivial solution...')
        dictionary = data[:, :data.shape[1]]
        return dictionary
    elif(param['InitializationMethod'] == 'DataElements'):
        dictionary[:, :param['K'] - param['preserveDCAtom']] = \
            data[:, :param['K'] - param['preserveDCAtom']]
    elif(param['InitializationMethod'] == 'GivenMatrix'):
        dictionary[:, :param['K'] - param['preserveDCAtom']] = \
            param['initialDictionary'][:, :param['K'] - param['preserveDCAtom']]

    # --------------------------------
    # DEBUG SECTION
    # --------------------------------
    output_dir = 'debugCsvPy'  # Directory where CSV files will be stored
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    py_dict_path = os.path.join(output_dir, 'py_test.csv')
    np.savetxt(py_dict_path, dictionary, delimiter=',', fmt='%.6f')

    # Data arrives here as int16, so we need to convert it to float64
    #dictionary = dictionary.astype(np.float64)

    # Adjust dicitonary for fixed elements
    if(param['preserveDCAtom']):
        tmpMat = fixedDictElem @ np.linalg.lstsq(dictionary)
        dictionary = dictionary - fixedDictElem @ tmpMat

    # DEBUG
    print(f'type(dictionary): {type(dictionary)}, dictionary.dtype: {dictionary.dtype}')
    
    # Normalize
    dictionary = dictionary @ np.diag(1 / np.sqrt(np.sum(dictionary ** 2, axis=0)))
    dictionary = dictionary * np.tile(np.sign(dictionary[0, :]), (dictionary.shape[0], 1))


    ## KSVD algorithm
    for iterNum in range(param['numIterations']):
        # Compute coefficients with OMP
        coef_matrix = OMP(np.hstack((fixedDictElem, dictionary)), data, param['L'])
        rand_perm = np.random.permutation(dictionary.shape[1])
        for j in rand_perm:
            betterDictElem, coef_matrix, _ = \
                I_findBetterDictionaryElement( 
                    data, np.hstack((fixedDictElem, dictionary)),
                    j + dictionary.shape[1], coef_matrix, param['L']
                    )
        dictionary[:, j] = betterDictElem

        if(param['preserveDCAtom']):
            tmpCoeff = fixedDictElem @ np.linalg.lstsq(betterDictElem)
            dictionary[:, j] = betterDictElem - fixedDictElem @ tmpCoeff
            dictionary[:, j] = dictionary[:, j] / np.sqrt(dictionary[:, j].T @ dictionary[:, j])

    
    # clear dictionary
    dictionary = I_clearDictionary(dictionary, coef_matrix[fixedDictElem,shape[1]:, :], data)

    # last hstack
    dictionary = np.hstack((fixedDictElem, dictionary))

    return dictionary, coef_matrix





    

