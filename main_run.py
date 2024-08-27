"""
Compressed Sensing of ECG - Kronecker Technicque - Dictionary Learning
"""

import datetime
import csv

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import sparseDictionaries as spardic
import measurementMatrix as mesmat
import evaluation as evaluat
from SL0 import SL0
from MOD import MOD
from KSVD import KSVD
from utils import py_test_csv


# avoid plot blocking
#plt.ion()

# initialize Signal to Noise Ratios
KSVD_SNR = 0
MOD_SNR = 0
DCT_SNR = 0
KSVD_KRON_SNR = 0
MOD_KRON_SNR = 0
DCT_KRON_SNR = 0

# Load the data
data0 = scipy.io.loadmat('100m.mat')   # contains val: [2x650000 double]
# data0.val(1,:) are MLII ECG data
# data0.val(2,:) are V5 ECG data

# Train and Test data
MULT = 128  # default 128
TRAIN_NUM = 400  # default 400
TEST_NUM = 25  # default 25

TRAIN_LEN = TRAIN_NUM*MULT
TEST_LEN = TEST_NUM*MULT

START_TRAIN = 0
END_TRAIN = TRAIN_LEN

START_TEST = END_TRAIN
END_TEST = START_TEST + TEST_LEN

trainSet = data0['val'][0][START_TRAIN:END_TRAIN]
testSet = data0['val'][0][START_TEST:END_TEST]

plt.plot(testSet)
plt.title('Test Set')
plt.show()

## Testing Phase Parameters (DO NOT CONFUSE WITH TRAINING PARAMETERS)
x = testSet
if len(x) != TEST_LEN:
    print('ERROR: x has not the correct length, check inital parameters')
    exit()

N = 16  # Length of each block
CR = 1/4  # Compression ratio
M = int(N * CR)  # Length of compressed measurement
SIGNAL_BLOCKS = len(x) // N
KRON_FACT = 4  # kronecker factor
N_KRON = KRON_FACT * N  # kron block length
M_KRON = int(N_KRON * CR)  # kron compressed meas. length
SIGNAL_BLOCKS_KRON = len(x) // N_KRON



REPEAT = 1  # number of times to REPEAT the experiment
for rep in range(REPEAT):


    ## --------------------------------
    ## Adaptive Dictionary Learning
    ## --------------------------------

    ## NON KRONECKER
    ## --------------------------------
    T_LEN = N  # Length of each training sample
    T_NUM = TRAIN_LEN // T_LEN  # Number of training samples
    trainMat = trainSet.reshape(T_NUM, T_LEN)
    REDUND = 1  # 1 for no redundancy
    print(trainMat.T.shape)

    param = {
        'K': REDUND * trainMat.T.shape[0],  # num of atoms dict, atom = basis function
        'L': 1,
        'numIterations': 10,
        'preserveDCAtom': 0,
        'InitializationMethod': 'DataElements', # 'DataElements' or 'GivenMatrix'
        'initialDictionary': None  # random initialization of dictionary is futher on
    }
    # initialize InitialDictionary (only used if 'GivenMatrix' is set)
    InitialDictionary = np.random.rand(trainMat.T.shape[0], param['K'])
    for i in range(param['K']):
        InitialDictionary[:, i] = InitialDictionary[:, i] / np.linalg.norm(InitialDictionary[:, i])
    param['initialDictionary'] = InitialDictionary

    # non-kron dicts
    mod_dict, output = MOD(trainMat.T, param)
    ksvd_dict, X = KSVD(trainMat.T, param)
    print('mod')
    spardic.check_matrix_properties(mod_dict)
    print('ksvd')
    spardic.check_matrix_properties(ksvd_dict)


    ## KRONECKER
    ## --------------------------------
    T_LEN = N_KRON  # Length of each sample
    T_NUM = TRAIN_LEN // T_LEN # Number of training samples
    trainMat = trainSet.reshape(T_NUM, T_LEN)
    REDUND = 1  # 1 for no redundancy
    print(trainMat.T.shape)



    param = {
        'K': REDUND * trainMat.T.shape[0],  # num of atoms dict, atom = basis function
        'L': 1,
        'numIterations': 10,
        'preserveDCAtom': 0,
        'InitializationMethod': 'DataElements', # 'DataElements' or 'GivenMatrix'
        'initialDictionary': None  # random initialization of dictionary is futher on
    }
    # initialize InitialDictionary (only used if 'GivenMatrix' is set)
    InitialDictionary = np.random.rand(trainMat.T.shape[0], param['K'])
    for i in range(param['K']):
        InitialDictionary[:, i] = InitialDictionary[:, i] / np.linalg.norm(InitialDictionary[:, i])
    param['initialDictionary'] = InitialDictionary

    # kron
    mod_dict_kron, output = MOD(trainMat.T, param)
    ksvd_dict_kron, X = KSVD(trainMat.T, param)
    print('Kron mod')
    spardic.check_matrix_properties(mod_dict_kron)
    print('Kron svd')
    spardic.check_matrix_properties(ksvd_dict_kron)



    ## Fixed dct dictionary (non-adaptive)
    ## --------------------------------
    dct_dict = spardic.dct_dictionary(N)
    dct_dict_kron = spardic.dct_dictionary(N_KRON)
    spardic.check_matrix_properties(dct_dict)
    spardic.check_matrix_properties(dct_dict_kron)






    ## --------------------------------
    ##TESTING
    ## --------------------------------
 

    ## Create measurement matrix
    ## --------------------------------
    PHI_STRING = None
    # deterministic DBBD
    #Phi = mesmat.generate_DBBD_matrix(M, N)
    #PHI_STRING = 'DBBD'

    # randomic measurement
    if(PHI_STRING == 'DBBD'):
        raise ValueError('DBBD is already active')
        exit(-1)
    PHI_STRING = 'scaled_binary' # need for print at the end: 'scaled_binary', 'binary', 'gaussian'
    Phi = mesmat.generate_random_matrix(M, N, PHI_STRING)

    ## Create kron measurement matrix
    ## --------------------------------
    Phi_kron = np.kron(np.eye(KRON_FACT), Phi)


    ## Theta matrix
    ## --------------------------------
    mod_theta = Phi @ mod_dict
    ksvd_theta = Phi @ ksvd_dict
    dct_theta = Phi @ dct_dict

    ## Theta kron matrix
    ## --------------------------------
    mod_theta_kron = Phi_kron @ mod_dict_kron
    ksvd_theta_kron = Phi_kron @ ksvd_dict_kron  
    dct_theta_kron = Phi_kron @ dct_dict_kron

    

    # SL0 parameters
    sigma_off = 0.001
    mod_theta_pinv = np.linalg.pinv(mod_theta)
    ksvd_theta_pinv = np.linalg.pinv(ksvd_theta)
    dct_theta_pinv = np.linalg.pinv(dct_theta)
    mod_theta_kron_pinv = np.linalg.pinv(mod_theta_kron)
    ksvd_theta_kron_pinv = np.linalg.pinv(ksvd_theta_kron)
    dct_theta_kron_pinv = np.linalg.pinv(dct_theta_kron)
    mu_SL0 = 2
    L_SL0 = 3
    sig_dec_fact = 0.5
    if sigma_off > 0:
        sigma_min = sigma_off * 4
    else:
        sigma_min = 0.00001

    

    # Algorithm

    # Initialize the sparse code
    mod_x = np.zeros(len(x))
    ksvd_x = np.zeros(len(x))
    dct_x = np.zeros(len(x))
    mod_kron_x = np.zeros(len(x))
    ksvd_kron_x = np.zeros(len(x))
    dct_kron_x = np.zeros(len(x))

    # Sampling Phase
    Y = np.zeros((M, SIGNAL_BLOCKS))
    for i in range(len(x) // N):

        Y[:,i] = Phi @ x[i*N:(i+1)*N]

    # non-kron recovery
    for i in range(SIGNAL_BLOCKS):
        
        y = Y[:,i]

        # SL0: Sparse reconstruction
        mod_xp = SL0(y, mod_theta, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, mod_theta_pinv, showProgress=False)
        ksvd_xp = SL0(y, ksvd_theta, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, ksvd_theta_pinv, showProgress=False)
        dct_xp = SL0(y, dct_theta, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, dct_theta_pinv, showProgress=False)
        # Recovery Phase
        mod_x[i*N:(i+1)*N] = mod_dict @ mod_xp
        ksvd_x[i*N:(i+1)*N] = ksvd_dict @ ksvd_xp
        dct_x[i*N:(i+1)*N] = dct_dict @ dct_xp

    # kron recovery
    for i in range(SIGNAL_BLOCKS_KRON):
        y = Y[:, i*KRON_FACT : (i+1)*KRON_FACT].flatten(order='F')

        # SL0
        mod_kron_xp = SL0(y, mod_theta_kron, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, mod_theta_kron_pinv, showProgress=False)
        ksvd_kron_xp = SL0(y, ksvd_theta_kron, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, ksvd_theta_kron_pinv, showProgress=False)
        dct_kron_xp = SL0(y, dct_theta_kron, sigma_min, sig_dec_fact,
                        mu_SL0, L_SL0, dct_theta_kron_pinv, showProgress=False)


    # Evaluation (in rep loop, NOT in i loop)
    MOD_SNR += evaluat.calculate_snr(testSet, mod_x)
    KSVD_SNR += evaluat.calculate_snr(testSet, ksvd_x)
    DCT_SNR += evaluat.calculate_snr(testSet, dct_x)
    MOD_KRON_SNR += evaluat.calculate_snr(testSet, mod_kron_x)
    KSVD_KRON_SNR += evaluat.calculate_snr(testSet, ksvd_kron_x)
    DCT_KRON_SNR += evaluat.calculate_snr(testSet, dct_kron_x)


# Average the SNRs
MOD_SNR /= REPEAT
KSVD_SNR /= REPEAT
DCT_SNR /= REPEAT
MOD_KRON_SNR /= REPEAT
KSVD_KRON_SNR /= REPEAT
DCT_KRON_SNR /= REPEAT


# Print the average over e of the snr
print(f'Average SNR over {REPEAT} repetitions:')
print(f'MOD SNR: {MOD_SNR}')
print(f'MOD KRON SNR: {MOD_KRON_SNR}')
print(f'KSVD SNR: {KSVD_SNR}')
print(f'KSVD KRON SNR: {KSVD_KRON_SNR}')
print(f'DCT SNR: {DCT_SNR}')
print(f'DCT KRON SNR: {DCT_KRON_SNR}')


# CSV to store old results
with open('results.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([PHI_STRING, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), REPEAT, MOD_SNR, MOD_KRON_SNR, KSVD_SNR, KSVD_KRON_SNR, DCT_SNR, DCT_KRON_SNR])

# plot
evaluat.plot_signals(reconstructed_signal=mod_x, original_signal=testSet,
                    snr=MOD_SNR,
                    reconstructed_name='mod_x',
                    original_name='testSet'
                    )
evaluat.plot_signals(reconstructed_signal=mod_kron_x, original_signal=testSet,
                    snr=MOD_KRON_SNR,
                    reconstructed_name='mod_kron_x',
                    original_name='testSet'
                    )

evaluat.plot_signals(reconstructed_signal=ksvd_x, original_signal=testSet,
                    snr=KSVD_SNR,
                    reconstructed_name='ksvd_x',
                    original_name='testSet'
                    )
evaluat.plot_signals(reconstructed_signal=ksvd_kron_x, original_signal=testSet, 
                    snr=KSVD_KRON_SNR,
                    reconstructed_name='ksvd_kron_x',
                    original_name='testSet'
                    )

evaluat.plot_signals(reconstructed_signal=dct_x, original_signal=testSet,
                    snr=DCT_SNR,
                    reconstructed_name='dct_x',
                    original_name='testSet'
                    )   
evaluat.plot_signals(reconstructed_signal=dct_kron_x, original_signal=testSet,
                    snr=DCT_KRON_SNR,
                    reconstructed_name='dct_kron_x',
                    original_name='testSet'
                    )
