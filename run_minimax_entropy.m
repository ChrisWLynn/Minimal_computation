% Example script to compute the complete maximum entropy model (that is,
% the maximum entropy model with the minimum number of inputs needed to
% capture all direct dependencies) for a single neuron

% Load data:
data = load('data_mouse_hippocampus.mat'); % Hippocampus
% data = load('data_mouse_visual_responding.mat'); % Visual cortex (responding)
% data = load('data_mouse_visual_spontaneous.mat'); % Visual cortex (spontaneous)
% data = load('data_C_elegans.mat'); % C. elegans

% Time-series data:
X = data.X;

% Neuron to analyze:
y = 13;

% Run greedy algorithm to find optimal inputs and complete model:
minimax_entropy(X, y);