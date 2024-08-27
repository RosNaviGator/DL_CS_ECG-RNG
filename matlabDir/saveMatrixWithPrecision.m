% Function to save matrix with specific precision using fprintf
function saveMatrixWithPrecision(matrix, filename, precision)
    fileID = fopen(filename, 'w');
    formatSpec = [repmat(['%', precision, 'f,'], 1, size(matrix, 2)-1), '%', precision, 'f\n'];
    fprintf(fileID, formatSpec, matrix.');
    fclose(fileID);
end

