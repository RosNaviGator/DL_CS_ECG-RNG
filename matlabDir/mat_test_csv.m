function mat_test_csv(object)
    % mat_test_csv - Save a matrix object to a CSV file with specified precision.
    %
    %   mat_test_csv(object)
    %
    %   This function takes a matrix object as input and saves it to a CSV file
    %   with a specified precision. The CSV file is saved in the 'debugCsvMAT'
    %   directory, and the filename is 'mat_test.csv'.
    %
    %   Input:
    %   - object: The matrix object to be saved as a CSV file.
    %
    %   See also saveMatrixWithPrecision
    outputDir = 'debugCsvMAT';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    outputFilename = fullfile(outputDir, 'mat_test.csv');
    saveMatrixWithPrecision(object, outputFilename, '6');
end
