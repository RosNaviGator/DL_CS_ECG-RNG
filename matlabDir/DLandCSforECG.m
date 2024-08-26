clear; clc; close all;

% initialize Signal to Noise Ratios
SNR_DCTT=0;
SNR_MODD=0;
SNR_KSVDD=0;


repeat=1;  % process will be repeated 'repeat' times

for iter = 1:repeat
    
    %% ------------------------------------------------
    %% DATA
    %% ------------------------------------------------
    
    %% Load ECG data
    
    data0 = load('./100m.mat');  % contains val: [2x650000 double]
    % display shape of data0.val
    %disp('Shape of data0.val');
    %disp(size(data0.val));
    % data0.val(1,:) are MLII ECG data
    % data0.val(2,:) are V5 ECG data
    
    
    %% Extract training and testing Sets from ECG data
    
    MULT = 128;  % We'll extract multiples of MULT samples                   DEFAULT: 128
    TRAIN_NUM = 400;  % Number of times we take MULT samples for training    DEFAULT: 400
    TEST_NUM = 25;  % Number of times we take MULT samples for testing       DEFAULT: 25
    
    start_train = 1;  % we start extracting from this sample for training
    end_train = TRAIN_NUM*MULT;  % we end extracting at this sample for training
    
    % To avoid bias, we start extracting from the next sample after training
    start_test = end_train+1;
    end_test = start_test+(TEST_NUM*MULT)-1;


    trainSet = data0.val(1,start_train:end_train);
    testSet = data0.val(1,start_test:end_test);
    %disp('Shape of trainingSet');
    %disp(size(trainSet));
    %disp(trainSet(1:3));
    %disp('Shape of testSet');
    %disp(size(testSet));
    %disp(testSet(1:3));
    
    %% ------------------------------------------------
    %% TRAINING
    %% ------------------------------------------------
    
    % create Training Matrix from trainingSet
    l=TRAIN_NUM;  %Number of trained signal
    n=MULT;  %length of signal
    %Data = zeros(n,l);
    TrainMat = zeros(n,l);
    for i=1:l
        TrainMat(:,i) = trainSet( (i-1)*n+1 : i*n );
        %plot(TrainMat(:,i)); pause(0.1);
    end

    %disp('Shape of TrainMat');
    %disp(size(TrainMat));
    %disp(TrainMat(1:3,1:3));
    

    %% Dictionary learning

    
    param.K = 2*n;  % num of atoms dict, atom = basis function
    param.L = 1;
    param.numIteration = 10;
    param.preserveDCAtom = 0;
    param.InitializationMethod = 'DataElements';
    %param.InitializationMethod = 'GivenMatrix';
    param.TrueDictionary = randn(n,2*n);    
    iniMat = randn(n,param.K);  % random initialization of dictionary
    for i =1: param.K
        iniMat(:,i) = iniMat(:,i)/norm(iniMat(:,i));  % normalizie each atom (column)
    end
    param.initialDictionary = iniMat;
    
    % Dictionary learning using MOD and KSVD
    [DicMod, outputMod] = MOD(TrainMat,param);
    [DicKSVD,X] = KSVD(TrainMat,param);

    % Define the directory and output file name
    output_dir = './debug';
    % Check if the directory exists, if not, create it
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Save the dictionary to a .mat file in the specified directory
    output_filename = fullfile(output_dir, 'DicKSVD_output.mat');
    save(output_filename, 'DicKSVD');

    % save also the the MOD dictionary
    output_filename = fullfile(output_dir, 'DicMod_output.mat');
    save(output_filename, 'DicMod');

    
    
    
    %% ------------------------------------------------
    %% TESTING
    %% ------------------------------------------------
    
    % Test signal
    x=testSet';
    % print shape of x, print x
    %disp('Shape of x');
    %disp(size(x));
    %disp(x(1:16));
    
    % Parameters
    N = MULT;  % length of signal block
    CR = 1/4;  % compression ratio
    M = N*CR;  % length of compressed block
    
    
    %% Measurement matrix (A)
    
    %A = GenerateMatrix(M,N); ???
    %A = randn(M,N);  % random matrix
    
    % binomial random matrix
    A = ones(M,N);
    A = binornd(A,.5);
    A = A-.5;A=1/sqrt(M)*A;
    
    % BDDB matrix
    %------ Deterministic matrix: DBBD matrix 
    %A = zeros(M,N);
    %m=N/M;
    %for i=1:M 
    %    A(i,1+(i-1)*m:(i)*m) = 1;
    %end
    
    
    %isp('Shape of A');
    %disp(size(A));
    %disp(A);
    
    
    %% Dictionaries
    
    %dict_DCT = wmpdictionary(N,'LstCpt',{'dct'});  % DCT dictionary (benchmark)
    dict_DCT = wmpdictionary(N, 'LstCpt', repmat({'dct'}, 1, 2)); % Creates a dictionary with twice the number of DCT atoms
    dict_MOD = DicMod;  % MOD dictionary (previously learned)
    dict_KSVD = DicKSVD;  % KSVD dictionary (previously learned)

    %disp('Shape of dict_DCT');
    %disp(size(dict_DCT));
    %disp(dict_DCT(1:3,1:3));
    %disp('Shape of dict_MOD');
    %disp(size(dict_MOD));
    %disp(dict_MOD(1:3,1:3));
    %disp('Shape of dict_KSVD');
    %disp(size(dict_KSVD));
    %disp(dict_KSVD(1:3,1:3));

    
    
    %% Theta matrix, A*dict
    
    A1_DCT=A*dict_DCT;
    A1_MOD=A*dict_MOD;
    A1_KSVD=A*dict_KSVD;
    
    
    %--------------------Sl0 Parameters
    for i=1:1
        
        sigma_off = 0.001;
        A_pinv_DCT = pinv(A1_DCT);
        A_pinv_MOD = pinv(A1_MOD);
        A_pinv_KSVD = pinv(A1_KSVD);
        mu_0 = 2;
        sigma_decrease_factor = 0.5;
        L = 3;
        
        %true_s = sparseSigGen4plusNoise(9,floor(27/4),sigma_off);
        if sigma_off>0
            sigma_min = sigma_off*4;
        else
            sigma_min = 0.00001;
        end
    end
    
    
    
    % Seems it doesn't use kroencker
    for i=1:length(testSet)/N
        
        j=i;
        
        y=A*x((i-1)*N+1:N*i,1);  % compressed signal block
        % shape of y
        %disp('Shape of y');
        %disp(size(y));
        
        xp_DCT = SL0(A1_DCT, y, sigma_min, sigma_decrease_factor, mu_0, L, A_pinv_DCT);
        xp_MOD = SL0(A1_MOD, y, sigma_min, sigma_decrease_factor, mu_0, L, A_pinv_MOD);
        xp_KSVD = SL0(A1_KSVD, y, sigma_min, sigma_decrease_factor, mu_0, L, A_pinv_KSVD);
        
        zm_DCT = dict_DCT*xp_DCT;
        zz_DCT(N*j-(N-1):N*j) = zm_DCT(:);
        
        zm_MOD = dict_MOD*xp_MOD;
        zz_MOD(N*j-(N-1):N*j) = zm_MOD(:);
        
        zm_KSVD = dict_KSVD*xp_KSVD;
        zz_KSVD(N*j-(N-1):N*j) = zm_KSVD(:);
    end
    
    
    % DCT over original signal
    plot(testSet);
    hold on;
    plot(zz_DCT,'b');
    % title
    title('DCT over original signal');
    % labels
    xlabel('Samples');
    ylabel('Amplitude');
    % legend
    legend('Original Signal','DCT');
    
    
    
    figure;
    plot(testSet);
    hold on;
    plot(zz_MOD,'r');
    % title
    title('MOD over original signal');
    % labels
    xlabel('Samples');
    ylabel('Amplitude');
    % legend
    legend('Original Signal','MOD');
    
    
    figure;
    plot(testSet);
    hold on;
    plot(zz_KSVD,'g');
    % title
    title('KSVD over original signal');
    % labels
    xlabel('Samples');
    ylabel('Amplitude');
    % legend
    legend('Original Signal','KSVD');
     

    err_DCT = zz_DCT-testSet;
    SNR_DCT = 20*log10(norm(testSet)/norm(err_DCT));
    err_MOD = zz_MOD-testSet;
    SNR_MOD = 20*log10(norm(testSet)/norm(err_MOD));
    err_KSVD = zz_KSVD-testSet;
    SNR_KSVD = 20*log10(norm(testSet)/norm(err_KSVD));

    SNR_DCTT=SNR_DCTT+SNR_DCT;
    SNR_MODD=SNR_MODD+SNR_MOD;
    SNR_KSVDD=SNR_KSVDD+SNR_KSVD;
    
end


fprintf('mean SNR_DCT, mean SNR_MOD, mean SNR_KSVD, over %d times\n', repeat);
fprintf('DCT: %f\n', SNR_DCTT/repeat);
fprintf('MOD: %f\n', SNR_MODD/repeat);
fprintf('KSVD: %f\n', SNR_KSVDD/repeat); 





