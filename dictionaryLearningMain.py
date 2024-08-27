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
from SL0 import SL0
import csv
import datetime
from MOD import MOD



# avoid plot blocking
#plt.ion()

# Choose how overcomplete the dictionary should be
ovComp = 1


# initialize Signal to Noise Ratios
sklearn_snr = 0
syanga_snr = 0
matlab_ksvd_snr = 0
matlab_mod_snr = 0
ksvd_snr = 0
dct_snr = 0



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
    #print('testSet.shape:', testSet.shape)
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
    num_comp = 2*n  # Number of dictionary atoms (basis functions) to learn. Set to the number of features if you want a square dictionary matrix.


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
        random_state=None,  # Seed for the random number generator. This ensures reproducibility by producing the same results every time the model is run with the same data and parameters.
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
    matlab_ksvd_dict = data1['DicKSVD']
    print('matlab_ksvd_dict.shape:', matlab_ksvd_dict.shape)
    #printFormatted(matlab_ksvd_dict, decimals = 4)
    # Test the dictionary
    sd.check_matrix_properties(matlab_ksvd_dict)

    # load matlab mod dictionary
    #data2 = scipy.io.loadmat('./matlabDir/debug/DicMod_output.mat')
    #matlab_mod_dict = data2['DicMod']
    #print('matlab_mod_dict.shape:', matlab_mod_dict.shape)
    #printFormatted(matlab_mod_dict, decimals = 4)
    # Test the dictionary
    #sd.check_matrix_properties(matlab_mod_dict)


    # use my function instead MATLAB/FORTRAN ==> COL-MAJOR
    
    # initialize TrueDictionary and InitialDictionary at random with shape of TrainMat
    TrueDictionary = np.random.rand(trainMat.T.shape[0],  2 * trainMat.T.shape[0])
    
    # Define parameters for MOD function
    param = {
        'K': 2 * trainMat.T.shape[0],  # num of atoms dict, atom = basis function
        'L': 1,
        'numIterations': 10,
        'errorFlag': 0,
        'preserveDCAtom': 0,
        'displayProgress': 0,
        'InitializationMethod': 'DataElements',
        'TrueDictionary': TrueDictionary,
        'initialDictionary': None  # random initialization of dictionary is futher on
    }

    # initialize InitialDictionary
    InitialDictionary = np.random.rand(trainMat.T.shape[0], param['K'])
    # assign to param
    param['initialDictionary'] = InitialDictionary
    # Normalize the initial dictionary
    for i in range(param['K']):
        param['initialDictionary'][:, i] = param['initialDictionary'][:, i] / np.linalg.norm(param['initialDictionary'][:, i])

    # Run MOD function
    matlab_mod_dict, output = MOD(trainMat.T, param)

    # Test the dictionary
    sd.check_matrix_properties(matlab_mod_dict)





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
    #Phi = mesmat.generate_DBDD_matrix(M, N)
    #string = 'DBDD'

    Phi = mesmat.generate_random_matrix(M, N, 'scaled_binary')
    string = 'scaled_binary'

    print('Phi.shape:', Phi.shape)
    #printFormatted(Phi, decimals = 4)


    # Theta matrix
    sklearn_theta = Phi @ sklearn_dict
    syanga_theta = Phi @ syanga_dict
    matlab_theta = Phi @ matlab_ksvd_dict
    matlab_mod_theta = Phi @ matlab_mod_dict 
    ksvd_theta = Phi @ ksvd_dict
    dct_theta = Phi @ dct_dict

    # shapes
    print('sklearn_theta.shape:', sklearn_theta.shape)
    print('syanga_theta.shape:', syanga_theta.shape)
    print('matlab_theta.shape:', matlab_theta.shape)
    print('matlab_mod_theta.shape:', matlab_mod_theta.shape)
    print('ksvd_theta.shape:', ksvd_theta.shape)
    print('dct_theta.shape:', dct_theta.shape)


    # SL0 parameters
    sigma_off = 0.001
    sklearn_theta_pinv = np.linalg.pinv(sklearn_theta)
    syanga_theta_pinv = np.linalg.pinv(syanga_theta)
    matlab_theta_pinv = np.linalg.pinv(matlab_theta)
    matlab_mod_theta_pinv = np.linalg.pinv(matlab_mod_theta)
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
    current_dict = matlab_mod_dict

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
    eval.plot_signals(reconstructed_signal=x_test_rec, original_signal=x_test,
                      snr=None,
                      reconstructed_name='x_test_rec',
                      original_name='x_test'
                      )

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
    eval.plot_signals(reconstructed_signal=x_train_rec, original_signal=x_train,
                      snr=None,
                      reconstructed_name='x_train_rec',
                      original_name='x_train'
                      )




    # Algorithm

    # Initialize the sparse code
    sklearn_x = np.zeros(len(x))
    syanga_x = np.zeros(len(x))
    matlab_ksvd_x = np.zeros(len(x))
    matlab_mod_x = np.zeros(len(x))
    ksvd_x = np.zeros(len(x))
    dct_x = np.zeros(len(x))


    for i in range(len(x) // N):

        # Sampling Phase
        y = Phi @ x[i*N:(i+1)*N]
        #print('y.shape:', y.shape)
        #print(y)


        # SL0: Sparse reconstruction
        sklearn_xp = SL0(y, sklearn_theta, sigma_min, sig_dec_fact, 
                         mu, l, sklearn_theta_pinv, showProgress=False)
        syanga_xp = SL0(y, syanga_theta, sigma_min, sig_dec_fact,
                        mu, l, syanga_theta_pinv, showProgress=False)
        matlab_ksvd_xp = SL0(y, matlab_theta, sigma_min, sig_dec_fact,
                        mu, l, matlab_theta_pinv, showProgress=False)
        matlab_mod_xp = SL0(y, matlab_mod_theta, sigma_min, sig_dec_fact,
                        mu, l, matlab_mod_theta_pinv, showProgress=False)
        ksvd_xp = SL0(y, ksvd_theta, sigma_min, sig_dec_fact,
                        mu, l, ksvd_theta_pinv, showProgress=False)
        dct_xp = SL0(y, dct_theta, sigma_min, sig_dec_fact,
                        mu, l, dct_theta_pinv, showProgress=False)

        
        # Recovery Phase
        sklearn_x[i*N:(i+1)*N] = sklearn_dict @ sklearn_xp
        syanga_x[i*N:(i+1)*N] = syanga_dict @ syanga_xp
        matlab_ksvd_x[i*N:(i+1)*N] = matlab_ksvd_dict @ matlab_ksvd_xp
        matlab_mod_x[i*N:(i+1)*N] = matlab_mod_dict @ matlab_mod_xp
        ksvd_x[i*N:(i+1)*N] = ksvd_dict @ ksvd_xp
        dct_x[i*N:(i+1)*N] = dct_dict @ dct_xp
    
    
    # Evaluation
    sklearn_snr += eval.calculate_snr(testSet, sklearn_x)
    syanga_snr += eval.calculate_snr(testSet, syanga_x)
    matlab_ksvd_snr += eval.calculate_snr(testSet, matlab_ksvd_x)
    matlab_mod_snr += eval.calculate_snr(testSet, matlab_mod_x)
    ksvd_snr += eval.calculate_snr(testSet, ksvd_x)
    dct_snr += eval.calculate_snr(testSet, dct_x)



# Average the SNRs
sklearn_snr /= repeat
syanga_snr /= repeat
matlab_ksvd_snr /= repeat
matlab_mod_snr /= repeat
ksvd_snr /= repeat
dct_snr /= repeat



# Print the average over e of the snr
print(f'Average SNR over {repeat} repetitions:')
print('sklearn_snr:', sklearn_snr)
print('syanga_snr:', syanga_snr)
print('matlab_ksvd_snr:', matlab_ksvd_snr)
print('matlab_mod_snr:', matlab_mod_snr)
print('ksvd_snr:', ksvd_snr)
print('dct_snr:', dct_snr)



# append on a csv at each run, each column is a different dictionary
# first col is measurement matrix used in string
# second column put date Y M D H M S IN A SINGLE COMMA SEPARATED STRING
# third column is the number of repetitions
# then the snrs, go:
    # Append results to a CSV file
with open('results.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([string, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), repeat, sklearn_snr, syanga_snr, matlab_ksvd_snr, matlab_mod_snr, ksvd_snr, dct_snr])


# plot
eval.plot_signals(reconstructed_signal=sklearn_x, original_signal=testSet,
                    snr=sklearn_snr,
                    reconstructed_name='sklearn_x',
                    original_name='testSet'
                    )

eval.plot_signals(reconstructed_signal=syanga_x, original_signal=testSet,
                    snr=syanga_snr,
                    reconstructed_name='syanga_x',
                    original_name='testSet'
                    )

eval.plot_signals(reconstructed_signal=matlab_ksvd_x, original_signal=testSet,
                    snr=matlab_ksvd_snr,
                    reconstructed_name='matlab_ksvd_x',
                    original_name='testSet'
                    )

eval.plot_signals(reconstructed_signal=matlab_mod_x, original_signal=testSet,
                    snr=matlab_mod_snr,
                    reconstructed_name='matlab_mod_x',
                    original_name='testSet'
                    )

eval.plot_signals(reconstructed_signal=ksvd_x, original_signal=testSet,
                    snr=ksvd_snr,
                    reconstructed_name='ksvd_x',
                    original_name='testSet'
                    )

eval.plot_signals(reconstructed_signal=dct_x, original_signal=testSet,
                    snr=dct_snr,
                    reconstructed_name='dct_x',
                    original_name='testSet'
                    )




    
        



    

    










