clc; clear; close all;


disp('----------------------------------------');
disp('----------------------------------------');
disp('----------------------------------------');


% Define a 4x4 dictionary D (columns must be normalized)
D = [
    0.5, 0.4, 0.7, 0.1;
    0.5, -0.8, 0.2, -0.3;
    0.5, 0.2, -0.7, 0.9;
    0.5, 0.4, 0.2, 0.2
];

D = ones(4) - 2 * eye(4) + 0.1 * inv(ones(4) + 2 * eye(4));
disp(D);


% Normalize columns of D
D = D ./ vecnorm(D);

% Define a 4x1 signal X
X = [0.8; -0.6; 0.3; 0.9];

% Set the maximum number of coefficients L
L = 3;

% Run the OMP function
A = OMP(D, X, L);

% Display the output sparse coefficient matrix A
disp(A);
