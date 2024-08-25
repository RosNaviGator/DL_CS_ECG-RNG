import scipy.io
import numpy as np
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt
import time
from utils import printFormatted
import sparseDictionaries as sd
import ksvdSyanga as syanga
import sparseDictionaries as spardic
import measurementMatrix as mesmat
import evaluation as eval
from ksvd import ApproximateKSVD

# avoid plot blocking
#plt.ion()

# Choose how overcomplete the dictionary should be
ovComp = 1


# initialize Signal to Noise Ratios
SNR_DCTT=0
SNR_MODD=0
SNR_KSVDD=0


repeat = 1  # number of times to repeat the experiment
for rep in range(repeat):

    # Load the data
    data0 = scipy.io.loadmat('100m.mat')   # contains val: [2x650000 double]
    # data0.val(1,:) are MLII ECG data
    # data0.val(2,:) are V5 ECG data

    # Train and Test data
    MULT = 128  # default 128
    TRAIN_NUM = 400  # default 400
    TEST_NUM = 25  # default 25
    
    start_train = 0
    end_train = TRAIN_NUM*MULT

    start_test = end_train
    end_test = start_test + TEST_NUM*MULT

    trainSet = data0['val'][0][start_train:end_train]
    testSet = data0['val'][0][start_test:end_test]

    print('trainSet.shape:', trainSet.shape)
    print(trainSet)
    print('testSet.shape:', testSet.shape)
    #print(testSet)
    # plot test set with label for name
    plt.plot(testSet)
    plt.title('Test Set')
    plt.show()



    # Create training matrix

    # Ensure that the signal length N is divisible by n_block
    l = TRAIN_NUM  # Number of training blocks
    n = MULT  # Length of each block
    # Reshape the 1D signal in 2D (blocks)
    trainMat = trainSet.reshape(l, n)
    print('trainMat.shape:', trainMat.shape)


    # Create test matrix
    testMat = testSet.reshape(TEST_NUM,n)



    # Dictionary Learning 
    num_comp = n-1  # Number of dictionary atoms (basis functions) to learn. Set to the number of features if you want a square dictionary matrix.


    # Start time
    start_time = time.time()

    dict_learner = DictionaryLearning(
        n_components=num_comp,  # Number of dictionary atoms (basis functions) to learn. Set to the number of features if you want a square dictionary matrix.
        alpha=1,  # Regularization parameter controlling the sparsity of the representation. Higher values enforce more sparsity (fewer non-zero coefficients).
        max_iter=1,  # Maximum number of iterations for the optimization algorithm during the dictionary learning process.
        tol=1e-08,  # Tolerance for the stopping condition. If the cost function change is smaller than this value, the optimization stops.
        fit_algorithm='lars',  # Algorithm used for dictionary learning. 'cd' stands for Coordinate Descent, which updates one coordinate at a time to minimize the cost function.
        transform_algorithm='omp',  # Algorithm used for the sparse coding step (transforming the data). 'omp' stands for Orthogonal Matching Pursuit, which finds the sparsest representation of the data.
        transform_n_nonzero_coefs=int(0.1*n),  # If not None, specifies the maximum number of non-zero coefficients to use for sparse coding. Overrides alpha if set.
        transform_alpha=None,  # Regularization parameter for the transform step. If set, it overrides the alpha value during transformation to control sparsity.
        n_jobs=None,  # Number of parallel jobs to run for the computation. None means 1 unless in a joblib.parallel_backend context. Useful for speeding up computations.
        code_init=None,  # Initial value for the sparse code (coefficients). If None, it is initialized randomly. Can be used to provide a starting point for optimization.
        dict_init=None,  # Initial value for the dictionary atoms. If None, it is initialized randomly. Can be used to provide a starting point for the dictionary.
        callback=None,  # A callable that gets called every five iterations of the algorithm to monitor convergence or log progress.
        verbose=False,  # If True, it prints information about the progress of the optimization algorithm during fitting. Useful for debugging.
        split_sign=False,  # If True, the dictionary will include both the positive and negative parts of the data separately, effectively doubling its size.
        random_state=42,  # Seed for the random number generator. This ensures reproducibility by producing the same results every time the model is run with the same data and parameters.
        positive_code=False,  # If True, enforces the sparse code (coefficients) to be non-negative. This is useful in cases where negative values do not make sense.
        positive_dict=False,  # If True, enforces the dictionary atoms to be non-negative. Useful in cases where dictionary elements should not have negative values.
        transform_max_iter=1000  # Maximum number of iterations to run the transform algorithm. Controls how long the sparse coding step can take.
    )
    # Fit the model to the training matrix
    dict_learner.fit(trainMat)
    # End time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time


    # Output the learned dictionary matrix
    sklearn_dict = dict_learner.components_
    sklearn_dict = sklearn_dict.T
    
    # Visualize
    print('sklearn_dict.shape:', sklearn_dict.shape)
    #printFormatted(sklearn_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(sklearn_dict)
    alpha = dict_learner.alpha
    print('alpha:', alpha)
    print('elapsed_time:', elapsed_time)



    # learn dictionary using K-SVD of syanga
    syanga_dict, X, E = syanga.ksvd(Data=trainMat, num_atoms=num_comp, 
                                    sparsity=int(0.1*n), initial_D=None, 
                                    maxiter=10, etol=1e-8, approx=False, 
                                    debug=True)
    
    # Visualize
    print('syanga_dict.shape:', syanga_dict.shape)
    #printFormatted(syanga_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(syanga_dict)



    # load matlab ksvd dictionary
    data1 = scipy.io.loadmat('./matlabDir/debug/DicKSVD_output.mat')
    matlab_dict = data1['DicKSVD']
    print('matlab_dict.shape:', matlab_dict.shape)
    #printFormatted(matlab_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(matlab_dict)



    # ksvd package dictionary
    ksvd = ApproximateKSVD(n_components = n-1,  # num of dictionary atoms
                            max_iter = 10,  # maximum number of iterationsto learn the dictionary
                            tol = 1e-6,  # tolerance for stopping criterion
                            transform_n_nonzero_coefs = int(0.1*n))  # aim for 5 non-zero coefficients in the sparse coding

    # Fit the KSVD model to the data
    # convert to float
    trainMat = trainMat.astype(float)
    ksvd.fit(trainMat)
    # Access the learned dictionary
    ksvd_dict = ksvd.components_
    ksvd_dict = ksvd_dict.T
    print('ksvd_dict.shape:', ksvd_dict.shape)
    #printFormatted(ksvd_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(ksvd_dict)



    # dct dictionary
    dct_dict = spardic.dct_dictionary(n)
    print('dct_dict.shape:', dct_dict.shape)
    #printFormatted(dct_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(dct_dict)



    ##TESTING

    x = testSet

    N = MULT  # Length of each block
    CR = 1/4  # Compression ratio
    M = int(N*CR)  # Number of measurements


    # Create measurement matrix
    Phi = mesmat.generate_DBDD_matrix(M, N)
    # Phi = mesmat.generate_random_matrix(M, N, 'scaled_binary')
    print('Phi.shape:', Phi.shape)
    #printFormatted(Phi, decimals = 4)


    # Theta matrix
    sklearn_theta = Phi @ sklearn_dict
    syanga_theta = Phi @ syanga_dict
    matlab_theta = Phi @ matlab_dict
    ksvd_theta = Phi @ ksvd_dict
    dct_theta = Phi @ dct_dict

    # shapes
    print('sklearn_theta.shape:', sklearn_theta.shape)
    print('syanga_theta.shape:', syanga_theta.shape)
    print('matlab_theta.shape:', matlab_theta.shape)
    print('ksvd_theta.shape:', ksvd_theta.shape)
    print('dct_theta.shape:', dct_theta.shape)


    # SL0 parameters
    sigma_off = 0.001
    sklearn_theta_pinv = np.linalg.pinv(sklearn_theta)
    syanga_theta_pinv = np.linalg.pinv(syanga_theta)
    matlab_theta_pinv = np.linalg.pinv(matlab_theta)
    ksvd_theta_pinv = np.linalg.pinv(ksvd_theta)
    dct_theta_pinv = np.linalg.pinv(dct_theta)
    mu = 2
    sig_dec_fact = 0.5
    l = 3
    if sigma_off > 0:
        sigma_min = sigma_off * 4
    else:
        sigma_min = 0.00001

    
    u_piezz = 19
    current_dict = sklearn_dict
    # test reconstruction of test set
    # print x shape
    print('x.shape:', x.shape)
    x_test = testMat[u_piezz]
    print('x_test.shape:', x_test.shape)
    #print(x_test)
    s_train = np.linalg.pinv(current_dict) @ x_test
    print('s_train.shape:', s_train.shape)
    #print(s_train)
    # find median values, threshold is median
    treshold = np.median(np.abs(s_train))
    # eliminate below treshold
    s_train[np.abs(s_train) < treshold] = 0
    print('s_train.shape:', s_train.shape)
    #print(s_train)
    x_test_rec = current_dict @ s_train
    print('x_test_rec.shape:', x_test_rec.shape)
    #print(x_test_rec)
    # eval
    eval.plot_signals(x_test, x_test_rec, 'x_test', 'x_test_rec')

    # test reconstruction of training set
    x_train = trainMat[u_piezz]
    print('x_train.shape:', x_train.shape)
    #print(x_train)
    s_train = np.linalg.pinv(current_dict) @ x_train
    print('s_train.shape:', s_train.shape)
    #print(s_train)
    # find median values, threshold is median
    treshold = np.median(np.abs(s_train))
    # eliminate below treshold
    s_train[np.abs(s_train) < treshold] = 0
    print('s_train.shape:', s_train.shape)
    #print(s_train)
    x_train_rec = current_dict @ s_train
    print('x_train_rec.shape:', x_train_rec.shape)
    #print(x_train_rec)
    # eval
    eval.plot_signals(x_train, x_train_rec, 'x_train', 'x_train_rec')




    # Algo
    for i in range(len(testSet) // N):


        y = Phi @ x[i*N:(i+1)*N]
    
        



    

    










