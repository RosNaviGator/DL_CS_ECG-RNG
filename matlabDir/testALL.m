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
%True = randn(n,2*n);
%disp('True Dictionary:');
%disp(True);
%Init = randn(n,2*n);
%disp('Initial Dictionary:');
%disp(Init);

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




% Run MOD function
[Dictionary, CoefMatrix] = MOD(Data, param);
% Run KSVD function
[DicKSVD, X] = KSVD(Data, param);




outputDir = 'debugCsvMAT';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
outputFilename = fullfile(outputDir, 'mat_MOD.csv');
saveMatrixWithPrecision(Dictionary, outputFilename, '6');
outputFilename = fullfile(outputDir, 'CoefMatrix.csv');
saveMatrixWithPrecision(CoefMatrix, outputFilename, '6');
outputFilename = fullfile(outputDir, 'mat_KSVD.csv');
saveMatrixWithPrecision(DicKSVD, outputFilename, '6');
%outputFilename = fullfile(outputDir, 'X.csv');
%saveMatrixWithPrecision(X, outputFilename, '6');


% Just to check if the script is running
disp("Arrived at the end")






% ----------------------------------------
%% FUNCTIONS
% ----------------------------------------

% Function to save matrix with specific precision using fprintf
function saveMatrixWithPrecision(matrix, filename, precision)
    fileID = fopen(filename, 'w');
    formatSpec = [repmat(['%', precision, 'f,'], 1, size(matrix, 2)-1), '%', precision, 'f\n'];
    fprintf(fileID, formatSpec, matrix.');
    fclose(fileID);
end





