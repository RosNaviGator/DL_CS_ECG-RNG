clc; clear; close all;



% Set the random seed for reproducibility
rng(0); % Fixed seed for reproducibility

% Define dimensions
n = 4; % Dimension of each signal
N = 16; % Number of signals
% Generate random Data matrix with fixed seed
Data = rand(n, N); % Random matrix of size n x N with reproducibility

% print original
disp('Original Data:');
disp(Data);

True = randn(n,2*n);
disp('True Dictionary:');
disp(True);

Init = randn(n,2*n);
disp('Initial Dictionary:');
disp(Init);



param.K = 2*n;  % num of atoms dict, atom = basis function
    param.L = 1;
    param.numIteration = 10;
    param.errorFlag = 0;
    param.preserveDCAtom =0;
    param.displayProgress = 0;
    param.InitializationMethod = 'DataElements';
    param.TrueDictionary = True;
    %param.InitializationMethod = 'DataElements';
    %param.InitializationMethod = 'GivenMatrix';
    
    iniMat = Init;  % random initialization of dictionary
    for i =1: param.K
        iniMat(:,i) = iniMat(:,i)/norm(iniMat(:,i));  % normalizie each atom (column)
    end
    param.initialDictionary = iniMat;




% Run MOD function
[Dictionary, output] = MOD(Data, param);

% Print results
disp('Final Dictionary:');
disp(Dictionary);

% Display all outputs
disp('Output:');
disp(output);


if isfield(output, 'totalerr')
    disp('Total Error after each iteration:');
    disp(output.totalerr);
end

if isfield(output, 'numCoef')
    disp('Average number of coefficients per signal after each iteration:');
    disp(output.numCoef);
end
