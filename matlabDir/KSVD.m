function [Dictionary, CoefMatrix] = KSVD(Data, param)
    % =========================================================================
    %                          Simplified K-SVD Algorithm
    % =========================================================================
    % The K-SVD algorithm finds a dictionary for the linear representation of
    % signals. Given a set of signals, it searches for the best dictionary that
    % can sparsely represent each signal. The original algorithm and its details 
    % can be found in "The K-SVD: An Algorithm for Designing of Overcomplete 
    % Dictionaries for Sparse Representation" by M. Aharon, M. Elad, and A.M. 
    % Bruckstein, IEEE Trans. on Signal Processing, Vol. 54, no. 11, pp. 4311-4322, 
    % November 2006.
    %
    % This code is a simplified version of the original K-SVD algorithm by Hadi 
    % Zanddizari (2008). It was modified and simplified by RosNaviGator in 2024.
    % Several parameters and functionalities have been removed for simplification.
    % =========================================================================
    % INPUT ARGUMENTS:
    % Data                         an nXN matrix that contains N signals, each of dimension n.
    % param                        structure that includes all required parameters for the K-SVD execution.
    %    K                         the number of dictionary elements to train.
    %    numIteration              number of iterations to perform.
    %    errorFlag                 if =0, a fixed number of coefficients is used for representation of each signal.
    %                              If =1, an arbitrary number of atoms represent each signal until a specific 
    %                              representation error is reached.
    %    preserveDCAtom            if =1, the first atom in the dictionary is set to be constant.
    %    L                         maximum coefficients to use in OMP coefficient calculations (used if errorFlag = 0).
    %    InitializationMethod      method to initialize the dictionary:
    %                              'DataElements' - initialization by the signals themselves,
    %                              'GivenMatrix' - initialization by a given matrix 'initialDictionary'.
    %    initialDictionary         (optional) if 'InitializationMethod' is 'GivenMatrix', this is the matrix to use.
    % =========================================================================
    % OUTPUT ARGUMENTS:
    %  Dictionary                  The extracted dictionary of size nX(param.K).
    %  CoefMatrix                  The final coefficients matrix.
    % =========================================================================
    
    % Check if DC atom needs to be preserved
    if (param.preserveDCAtom > 0)
        FixedDictionaryElement(1:size(Data, 1), 1) = 1 / sqrt(size(Data, 1));
    else
        FixedDictionaryElement = [];
    end
    
    % Dictionary Initialization
    if (size(Data, 2) < param.K)
        disp('KSVD: Size of data is smaller than the dictionary size. Trivial solution...');
        Dictionary = Data(:, 1:size(Data, 2));
        CoefMatrix = eye(size(Data,2)); % Trivial coefficients (identity matrix)
        return;
    elseif (strcmp(param.InitializationMethod, 'DataElements'))
        Dictionary(:, 1:param.K - param.preserveDCAtom) = Data(:, 1:param.K - param.preserveDCAtom);
    elseif (strcmp(param.InitializationMethod, 'GivenMatrix'))
        Dictionary(:, 1:param.K - param.preserveDCAtom) = param.initialDictionary(:, 1:param.K - param.preserveDCAtom);
    end

    


    % Adjust dictionary for fixed elements
    if (param.preserveDCAtom)
        tmpMat = FixedDictionaryElement \ Dictionary;
        Dictionary = Dictionary - FixedDictionaryElement * tmpMat;
    end

    % Normalize
    Dictionary = Dictionary * diag(1 ./ sqrt(sum(Dictionary .* Dictionary)));
    Dictionary = Dictionary .* repmat(sign(Dictionary(1, :)), size(Dictionary, 1), 1);

    % Start of the K-SVD algorithm
    for iterNum = 1:param.numIteration
        % Coefficient calculation using OMP
        CoefMatrix = OMP([FixedDictionaryElement, Dictionary], Data, param.L);
        % Improve dict elems
        rPerm = randperm(size(Dictionary, 2));
        for j = rPerm
            [betterDictionaryElement, CoefMatrix, NewVectorAdded] = I_findBetterDictionaryElement(Data, [FixedDictionaryElement, Dictionary], j + size(FixedDictionaryElement, 2), CoefMatrix, param.L);
            
            % substitute the j-th
            Dictionary(:, j) = betterDictionaryElement;
            % preserveDCAtom case
            if (param.preserveDCAtom)
                tmpCoef = FixedDictionaryElement \ betterDictionaryElement;
                Dictionary(:, j) = betterDictionaryElement - FixedDictionaryElement * tmpCoef;
                Dictionary(:, j) = Dictionary(:, j) / sqrt(Dictionary(:, j)' * Dictionary(:, j));
            end
        end

        % Clear dictionary
        Dictionary = I_clearDictionary(Dictionary, CoefMatrix(size(FixedDictionaryElement, 2) + 1:end, :), Data);
        
    end
    
    Dictionary = [FixedDictionaryElement, Dictionary];
    
    % END OF K-SVD FUNCTION
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  I_findBetterDictionaryElement
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [betterDictionaryElement, CoefMatrix, NewVectorAdded] = I_findBetterDictionaryElement(Data, Dictionary, j, CoefMatrix, numCoefUsed)
    if nargin < 5
        numCoefUsed = 1;
    end


    relevantDataIndices = find(CoefMatrix(j, :));  
    if isempty(relevantDataIndices)
        ErrorMat = Data - Dictionary * CoefMatrix;
        ErrorNormVec = sum(ErrorMat.^2);
        [~, i] = max(ErrorNormVec);
        betterDictionaryElement = Data(:, i);
        betterDictionaryElement = betterDictionaryElement / sqrt(betterDictionaryElement' * betterDictionaryElement);
        betterDictionaryElement = betterDictionaryElement * sign(betterDictionaryElement(1));
        CoefMatrix(j, :) = 0;
        NewVectorAdded = 1;
        return;
    end
    
    NewVectorAdded = 0;
    tmpCoefMatrix = CoefMatrix(:, relevantDataIndices);
    tmpCoefMatrix(j, :) = 0;
    errors = Data(:, relevantDataIndices) - Dictionary * tmpCoefMatrix;
    [betterDictionaryElement, singularValue, betaVector] = svds(errors, 1);
    CoefMatrix(j, relevantDataIndices) = singularValue * betaVector';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  I_clearDictionary
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function Dictionary = I_clearDictionary(Dictionary, CoefMatrix, Data)
    T2 = 0.99;
    T1 = 3;
    K = size(Dictionary, 2);
    Er = sum( (Data - Dictionary * CoefMatrix).^2, 1);
    G = Dictionary' * Dictionary;
    G = G - diag(diag(G));
    for jj = 1:K
        
        if max(G(jj, :)) > T2 || length(find(abs(CoefMatrix(jj, :)) > 1e-7)) <= T1
            [~, pos] = max(Er);
            Er(pos(1)) = 0;
            Dictionary(:, jj) = Data(:, pos(1)) / norm(Data(:, pos(1)));
            G = Dictionary' * Dictionary;
            G = G - diag(diag(G));
        end
    end
    