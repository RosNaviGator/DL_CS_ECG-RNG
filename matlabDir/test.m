clc; clear; close all;



% Define a reproducible matrix Dictionary
Dictionary = [1 -2 0 3; 4 5 -6 -7; 8 -9 10 11];

% Display the original matrix
disp('Original Matrix:');
disp(Dictionary);

% Apply the given expression
result = repmat(sign(Dictionary(1,:)), size(Dictionary, 1), 1);

% Display the result
disp('Sgneck:');
disp(result);


Dictionary = Dictionary .* result;

% Display the result
disp('Resulting Matrix:');
disp(Dictionary);
