function [Dictionary, CoefMatrix] = MOD(Data, param)
    % =========================================================================
    %                          MOD algorithm (Simplified Version)
    % =========================================================================
    % Originally developed in 2008 by Hadi Zanddizari.
    % Simplified and revised in 2024 by RosNaviGator.
    % For details on the original method, see the paper:
    % "Method of optimal directions for frame design", by K. Engan,
    % S.O. Aase, and J.H. Husfy, IEEE International Conference on Acoustics,
    % Speech, and Signal Processing, 1999.
    % =========================================================================
    % INPUT ARGUMENTS:
    % Data                         an n x N matrix that contains N signals, each of dimension n.
    % param                        structure that includes required parameters:
    %    K                         number of dictionary elements to train.
    %    numIteration              number of iterations to perform for dictionary learning.
    %    InitializationMethod      method to initialize the dictionary:
    %                              * 'DataElements' - initialization using the first K signals themselves.
    %                              * 'GivenMatrix' - initialization using a provided matrix (param.initialDictionary).
    %    initialDictionary         (optional) an n x K matrix for initializing the dictionary, required if
    %                              'InitializationMethod' is 'GivenMatrix'.
    %    L                         the sparsity level, i.e., the maximum number of non-zero elements
    %                              allowed in each column of the coefficient matrix.
    % =========================================================================
    % OUTPUT ARGUMENTS:
    %  Dictionary                  The trained dictionary of size n x K.
    %  CoefMatrix                  The final coefficients matrix of size K x N, such that Data ≈ Dictionary * CoefMatrix.
    % =========================================================================
    
    % Check if the size of the data is smaller than the dictionary size
    if (size(Data,2) < param.K)
        disp('Size of data is smaller than the dictionary size. Trivial solution...');
        Dictionary = Data(:,1:size(Data,2));
        CoefMatrix = eye(size(Data,2)); % Trivial coefficients (identity matrix)
        return;
    elseif strcmp(param.InitializationMethod,'DataElements')
        % Initialize dictionary using data elements
        Dictionary = Data(:,1:param.K);
    elseif strcmp(param.InitializationMethod,'GivenMatrix')
        % Initialize dictionary using a given matrix
        Dictionary = param.initialDictionary;
    else
        error('Invalid InitializationMethod specified.');
    end


    
    % Normalize the dictionary
    Dictionary = Dictionary * diag(1 ./ sqrt(sum(Dictionary .^ 2)));
    Dictionary = Dictionary .* repmat(sign(Dictionary(1,:)), size(Dictionary, 1), 1); 
    

    % Main loop for dictionary optimization
    for iterNum = 1:param.numIteration
        % Find the coefficients using OMP (Orthogonal Matching Pursuit)
        CoefMatrix = OMP(Dictionary, Data, param.L);
    
        % Improve the dictionary
        Dictionary = Data * CoefMatrix' / (CoefMatrix * CoefMatrix' + 1e-7 * speye(size(CoefMatrix, 1)));
        sumDictElems = sum(abs(Dictionary));
        zerosIdx = find(sumDictElems < eps);       
        % Reinitialize any zero columns with random values
        Dictionary(:, zerosIdx) = randn(size(Dictionary, 1), length(zerosIdx));
        % Normalize the dictionary again
        Dictionary = Dictionary * diag(1 ./ sqrt(sum(Dictionary .^ 2)));
    end
    
    % Return the final dictionary and coefficient matrix
end
