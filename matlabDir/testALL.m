clc; clear; close all;

% Set the random seed for reproducibility
rng(0); % Fixed seed for reproducibility

% Define dimensions
n = 4; % Dimension of each signal
N = 16; % Number of signals
% Generate random Data matrix with fixed seed
Data = rand(n, N); % Random matrix of size n x N with reproducibility


% print original
%disp('Original Data:');
%disp(Data);
%Init = randn(n,2*n);
%disp('Initial Dictionary:');
%disp(Init);

param.K = fix(2*n);  % num of atoms dict, atom = basis function
param.L = 1;
param.numIteration = 10;
param.preserveDCAtom = 0;
param.InitializationMethod = 'DataElements';
%param.InitializationMethod = 'GivenMatrix';
iniMat = randn(n,param.K);  % random initialization of dictionary
for i =1: param.K
    iniMat(:,i) = iniMat(:,i)/norm(iniMat(:,i));  % normalizie each atom (column)
end
param.initialDictionary = iniMat;




% Run MOD function
[Dictionary, CoefMatrix] = MOD(Data, param);
% Run KSVD function
[DicKSVD, X] = KSVD(Data, param);

mat_test_csv(DicKSVD)




outputDir = 'debugCsvMAT';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
outputFilename = fullfile(outputDir, 'mat_MOD.csv');
saveMatrixWithPrecision(Dictionary, outputFilename, '6');
outputFilename = fullfile(outputDir, 'mat_KSVD.csv');
saveMatrixWithPrecision(DicKSVD, outputFilename, '6');

% Just to check if the script is running
disp("Arrived at the end")








