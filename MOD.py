import numpy as np
from utils import *
import scipy.sparse as sp





def paolo():
    return "Hello from Paolo!"


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

    [n,P] = X.shape
    [n,K] = D.shape

    A = np.zeros((K,P))
    print(f'size of A: {A.shape}')
    
    for k in range(P):
        a = np.zeros((L,))
        print(f'size of a: {a.shape}')
        x = X[:,k]
        print(f'size of x: {x.shape}')
        residual = x
        print(f'size of residual: {residual.shape}')
        indx = np.array([], dtype=int)
        for j in range(L):
            print()
            print()
            print('LOOP')            
            proj = D.T @ residual
            print(f'size of residual: {residual.shape}')
            print(f'residual: {residual}')
            print(f'size of proj: {proj.shape}')
            print(f'proj: {proj}')
            pos = np.argmax(np.abs(proj))
            print(f'size of pos: {pos}')
            print(f'pos AFTER: {pos}')
            indx = np.append(indx, pos.astype(int)) 
            print(f'indx: {indx}')
            print(f'D[:,indx].shape: {D[:,indx].shape}')
            print(f'D[:,indx]: \n {D[:,indx]}')
            a = np.linalg.pinv(D[:,indx]) @ x
            residual = x - D[:, indx] @ a
            if np.sum(residual ** 2) < 1e-6:
                break
        
        print()
        print()
        print('OUT OF LOOP')
        temp = np.zeros((K,))
        temp[indx] = a
        print(f'temp: {temp}')

        A[:,k] = temp

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
            
            - numIteration : int
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
                If the true dictionary was defined (in synthetic experiments), this parameter holds a vector of length numIteration that includes
                the detection ratios in each iteration.
            
            - totalerr : numpy.ndarray, optional
                The total representation error after each iteration (defined only if displayProgress == True and errorFlag == 0).
            
            - numCoef : numpy.ndarray, optional
                A vector of length numIteration that includes the average number of coefficients required for representation of each signal
                in each iteration (defined only if displayProgress == True and errorFlag == 1).
    """

    # intialize
    output = {}
    Dictionary = None


    if data.shape[1] < parameters['K']:
        raise ValueError("Number of signals is smaller than the dictionary size. Trivial solution...")
        Dictionary = Data[:, 1:data.shape[1]]
        return Dictionary, {}
    
    elif parameters['InitializationMethod'] == 'DataElements':
        Dictionary = data[:,1:parameters['K']]
    
    # should be changed to if 'DataElements' in parameters['InitializationMethod'], remove 'GivenMatrix' thing
    elif parameters['InitializationMethod'] == 'GivenMatrix':  
        Dictionary = parameters['initialDictionary']


    # normalize dictionary
    Dictionary = Dictionary @ np.diag(1. / np.sqrt(np.sum(Dictionary ** 2, axis=0)))
    Dictionary = Dictionary * np.tile(np.sign(Dictionary[0, :]), (Dictionary.shape[0], 1))
    K = Dictionary.shape[1]
    totalErr = np.zeros((1, parameters['numIteration']))


    if parameters['TrueDictionary'].shape == Dictionary.shape:
        displayErrorWithTrueDictionary = False
        ErrorBetweenDictioanries = np.zeros((parameters['numIterations']+1, 1))
        ratio = np.zeros((parameters['numIterations']+1, 1))
    else:
        displayErrorWithTrueDictionary = False
    
    numCoef = parameters['L']

    for iterNum in range(parameters['numIteration']):
        # find coeffs
        if parameters['errorFlag'] == 0:
            # should try and use the one from sklearn
            CoefMatrix = OMP(Dictionary, data, parameters['L'])  # use the one written by me
        else:
            # non esisting implementation with errorFlag == 1
            raise ValueError("ErrorFlag == 1 not implemented yet")
            parameters['L'] = 1

    # improve dictionary
    Dictionary = data @ CoefMatrix.T @ np.linalg.inv(CoefMatrix @ CoefMatrix.T + 1e-7 * sp.eye(CoefMatrix.shape[0]))
    sumDictElems = np.sum(np.abs(Dictionary), axis=0)
    Dictionary = Dictionary @ np.diag(1 /  np.sqrt(sum(Dictionary ** 2, axis=0)))

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
            




import numpy as np
from utils import *  # Assuming utils has required methods
import scipy.sparse as sp

# Use the exact matrices provided for testing

# Original Data
Data = np.array([
    [0.8147, 0.6324, 0.9575, 0.9572, 0.4218, 0.6557, 0.6787, 0.6555, 0.2769, 0.6948, 0.4387, 0.1869, 0.7094, 0.6551, 0.9597, 0.7513],
    [0.9058, 0.0975, 0.9649, 0.4854, 0.9157, 0.0357, 0.7577, 0.1712, 0.0462, 0.3171, 0.3816, 0.4898, 0.7547, 0.1626, 0.3404, 0.2551],
    [0.1270, 0.2785, 0.1576, 0.8003, 0.7922, 0.8491, 0.7431, 0.7060, 0.0971, 0.9502, 0.7655, 0.4456, 0.2760, 0.1190, 0.5853, 0.5060],
    [0.9134, 0.5469, 0.9706, 0.1419, 0.9595, 0.9340, 0.3922, 0.0318, 0.8235, 0.0344, 0.7952, 0.6463, 0.6797, 0.4984, 0.2238, 0.6991]
])

print('Original Data:')
print(Data)

# True Dictionary
TrueDictionary = np.array([
    [1.1006, -0.7423, 0.7481, -1.4023, -0.1961, 1.5877, -0.2437, 0.1049],
    [1.5442, -1.0616, -0.1924, -1.4224, 1.4193, -0.8045, 0.2157, 0.7223],
    [0.0859, 2.3505, 0.8886, 0.4882, 0.2916, 0.6966, -1.1658, 2.5855],
    [-1.4916, -0.6156, -0.7648, -0.1774, 0.1978, 0.8351, -1.1480, -0.6669]
])

print('True Dictionary:')
print(TrueDictionary)

# Initial Dictionary
InitialDictionary = np.array([
    [0.1873, -1.7947, -0.5445, 0.7394, -0.8396, 0.1240, -1.2078, -1.0582],
    [-0.0825, 0.8404, 0.3035, 1.7119, 1.3546, 1.4367, 2.9080, -0.4686],
    [-1.9330, -0.8880, -0.6003, -0.1941, -1.0722, -1.9609, 0.8252, -0.2725],
    [-0.4390, 0.1001, 0.4900, -2.1384, 0.9610, -0.1977, 1.3790, 1.0984]
])

print('Initial Dictionary:')
print(InitialDictionary)

# Define parameters for MOD function
param = {
    'K': 2 * Data.shape[0],  # num of atoms dict, atom = basis function
    'L': 1,
    'numIteration': 10,
    'errorFlag': 0,
    'preserveDCAtom': 0,
    'displayProgress': 0,
    'InitializationMethod': 'DataElements',
    'TrueDictionary': TrueDictionary,
    'initialDictionary': InitialDictionary  # random initialization of dictionary
}

# Normalize the initial dictionary
for i in range(param['K']):
    param['initialDictionary'][:, i] = param['initialDictionary'][:, i] / np.linalg.norm(param['initialDictionary'][:, i])

# Run MOD function
Dictionary, output = MOD(Data, param)

# Print results
print('Final Dictionary:')
print(Dictionary)

# Display all outputs
print('Output:')
print(output)

if 'totalerr' in output:
    print('Total Error after each iteration:')
    print(output['totalerr'])

if 'numCoef' in output:
    print('Average number of coefficients per signal after each iteration:')
    print(output['numCoef'])
