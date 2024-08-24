% Just an utility to disable/enable toolboxes in MATLAB
% https://it.mathworks.com/matlabcentral/fileexchange/60347-toggletoolbox

clc; clear; close all;

%% Use toggleToolbox 
% Visualize active toolboxes names in current session
M = toggleToolbox('names');
disp("Active toolboxes names:");
disp(M);
% query if on/off all toolboxes
ALL = toggleToolbox('all');
disp("Query all toolboxes:");
disp(ALL);
% query if on/off specific toolbox
LIST = toggleToolbox({'matlab', 'wavelet'}, 'query');
disp("Query specific toolboxes:")
disp(LIST);
disp("-----------------------------");


% turn off specific toolbox
toggleToolbox({'wavelet'}, 'off');
disp("Toggled one off:");
ALL = toggleToolbox('all');
disp(ALL);
disp("-----------------------------");


% turn on specific toolbox
toggleToolbox({'wavelet'}, 'on');
disp("Toggles back on:");
ALL = toggleToolbox('all');
disp(ALL);
disp("-----------------------------");


% disable all toolboxes
toggleToolbox('all', 'off');
disp("Toggled all off:");
ALL = toggleToolbox('all');
disp(ALL);
disp("-----------------------------");

% enable all toolboxes
toggleToolbox('all', 'on');
disp("Toggled all on:");
ALL = toggleToolbox('all');
disp(ALL);
disp("-----------------------------");
