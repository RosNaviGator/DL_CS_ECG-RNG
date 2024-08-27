import numpy as np
from utils import *
import scipy.sparse as sp
from utils import *
import os






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

    [n, P] = X.shape
    [n, K] = D.shape

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

    This function implements the MOD algorithm, which is used to learn a dictionary for sparse representation of signals.
    For more details, refer to the paper "Method of Optimal Directions for Frame Design" by K. Engan, S.O. Aase, and J.H. Husøy, 
    presented at the IEEE International Conference on Acoustics, Speech, and Signal Processing, 1999.

    Parameters
    ----------
    Data : numpy.ndarray
        An (n x N) matrix containing N signals (Y), each of dimension n.
    
    param : dict
        A dictionary that includes all required parameters for the MOD algorithm execution.
        Required fields are:
            - K : int
                The number of dictionary elements to train.
            
            - numIterations : int
                The number of iterations to perform.
            
            - errorFlag : int
                If set to 0, a fixed number of coefficients is used for representation of each signal.
                If set to 1, an arbitrary number of atoms represent each signal, until a specific representation error is reached.
                Depending on this flag:
                    - If errorFlag == 0, 'L' must be specified as the number of representing atoms.
                    - If errorFlag == 1, 'errorGoal' must be specified as the allowed error.
            
            - L : int, optional
                Maximum number of coefficients to use in OMP coefficient calculations (only used if errorFlag == 0).
            
            - errorGoal : float, optional
                Allowed representation error in representing each signal (only used if errorFlag == 1).
            
            - InitializationMethod : str
                Method to initialize the dictionary, which can be one of the following:
                * 'DataElements' (initialization by the signals themselves)
                * 'GivenMatrix' (initialization by a given matrix param['initialDictionary'])
            
            - initialDictionary : numpy.ndarray, optional
                The matrix to be used for dictionary initialization (only used if InitializationMethod == 'GivenMatrix').
            
            - preserveDCAtom : bool, optional
                If set to True, generates a DC atom (in which all entries are equal) that does not change throughout the training.
            
            - TrueDictionary : numpy.ndarray, optional
                If specified, the difference between this dictionary and the trained one is measured and displayed in each iteration.
            
            - displayProgress : bool, optional
                If set to True, progress information is displayed. If errorFlag == 0, the average representation error (RMSE) is displayed. 
                If errorFlag == 1, the average number of required coefficients for the representation of each signal is displayed.
    
    Returns
    -------
    Dictionary : numpy.ndarray
        The extracted dictionary of size (n x K).
    
    output : dict
        A dictionary containing information about the current run. It may include the following fields:
            - CoefMatrix : numpy.ndarray
                The final coefficients matrix. It should hold that Data is approximately equal to Dictionary @ output['CoefMatrix'].
            
            - ratio : numpy.ndarray, optional
                If the true dictionary was defined (in synthetic experiments), this parameter holds a vector of length numIterations that includes
                the detection ratios in each iteration.
            
            - totalerr : numpy.ndarray, optional
                The total representation error after each iteration (defined only if displayProgress == True and errorFlag == 0).
            
            - numCoef : numpy.ndarray, optional
                A vector of length numIterations that includes the average number of coefficients required for representation of each signal
                in each iteration (defined only if displayProgress == True and errorFlag == 1).
    """

    # intialize
    output = {}
    Dictionary = None


    if data.shape[1] < parameters['K']:
        print("Number of signals is smaller than the dictionary size. Trivial solution...")
        Dictionary = data[:, :data.shape[1]]
        return Dictionary, {}
    
    elif parameters['InitializationMethod'] == 'DataElements':
        Dictionary = data[:,:parameters['K']]
    
    # should be changed to if 'DataElements' in parameters['InitializationMethod'], remove 'GivenMatrix' thing
    elif parameters['InitializationMethod'] == 'GivenMatrix':  
        Dictionary = parameters['initialDictionary']







    for i in range(5): print()
    print(f'Type(Dictionary): {type(Dictionary)}, Dictionary.dtype: {Dictionary.dtype}')
    
    
    # Data arrives here as int16, so we need to convert it to float64
    Dictionary = Dictionary.astype(np.float64)
    # normalize dictionary
    Dictionary = Dictionary @ np.diag(1. / np.sqrt(np.sum(Dictionary ** 2, axis=0)))

    #print(f'Initialization: Dictionary shape: {Dictionary.shape}')
    #printFormatted(Dictionary)
    # Prepare output files
    output_dir = 'debugCsvPy'  # Directory where CSV files will be stored
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define file paths
    py_dict = os.path.join(output_dir, 'py_test.csv')
    # Save the dictionary
    np.savetxt(py_dict, Dictionary, delimiter=',', fmt='%.6f')
    

    Dictionary = Dictionary * np.tile(np.sign(Dictionary[0, :]), (Dictionary.shape[0], 1))
    K = Dictionary.shape[1]
    totalErr = np.zeros((1, parameters['numIterations']))

    # print
    print(f'Initialization: Dictionary shape: {Dictionary.shape}')
    #printFormatted(Dictionary)


    if parameters['TrueDictionary'].shape == Dictionary.shape:
        displayErrorWithTrueDictionary = False
        ErrorBetweenDictioanries = np.zeros((parameters['numIterations']+1, 1))
        ratio = np.zeros((parameters['numIterations']+1, 1))
    else:
        displayErrorWithTrueDictionary = False
    
    numCoef = parameters['L']

    for iterNum in range(parameters['numIterations']):
        # find coeffs
        if parameters['errorFlag'] == 0:
            # should try and use the one from sklearn
            CoefMatrix = OMP(Dictionary, data, parameters['L'])  # use the one written by me
            #print(f'Iteration {iterNum}: CoefMatrix shape: {CoefMatrix.shape}')
            #printFormatted(CoefMatrix)

        else:
            # non esisting implementation with errorFlag == 1
            raise ValueError("ErrorFlag == 1 not implemented yet")
            parameters['L'] = 1

    # improve dictionary
    Dictionary = data @ CoefMatrix.T @ np.linalg.inv(CoefMatrix @ CoefMatrix.T + 1e-7 * sp.eye(CoefMatrix.shape[0]))
    sumDictElems = np.sum(np.abs(Dictionary), axis=0)
    Dictionary = np.asarray(Dictionary)
    Dictionary = Dictionary @ np.diag(1 /  np.sqrt(np.sum(Dictionary ** 2, axis=0)))

    if (iterNum > 0) and (parameters['displayProgress']):
        if parameters['errorFlag'] == 0:
            output['totalErr'][iterNum-1] = \
                np.sqrt(
                    np.sum(
                        np.sum(data - Dictionary @ CoefMatrix) ** 2) / data.size
                    )
            print(f'Iteration {iterNum}: Total error: {output["totalErr"][iterNum-1]}')
        else:
            output['numCoef'][iterNum-1] =  np.count_nonzero(CoefMatrix) / data.shape[1]
            print(f'Iteration {iterNum}: Average number of coefficients: {output["numCoef"][iterNum-1]}')
        
    
    if displayErrorWithTrueDictionary:
        [ratio[iterNum+1], ErrorBetweenDictioanries[iterNum+1]] = I_findDistanceBetweenDictionaries(parameters['TrueDictionary'], Dictionary)
        print(f'Iteration {iterNum}: Ratio between dictionaries: {ratio[iterNum+1]}')

    
    output['CoefMatrix'] = CoefMatrix

    return Dictionary, output
            


