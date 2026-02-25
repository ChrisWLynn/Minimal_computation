function MI = mutual_information(m, C)
% Inputs: Nx1 vector of magnetizations m and NxN matrix of pairwise
% correlations C for an Ising system with 0,1 variables, where N is the
% number of nodes in the network.
%
% Outputs: Mutual informations MI between nodes

% Size of system:
N = length(m);

% Four contributions to mutual information:
MI = zeros(N);

MI_temp = C.*log2(C./(m*m'));
MI(C ~= 0) = MI_temp(C ~= 0);

MI_temp = (m - C).*log2((m - C)./(m - m*m'));
MI(m - C ~= 0) = MI(m - C ~= 0) + MI_temp(m - C ~= 0);

MI_temp = (m' - C).*log2((m' - C)./(m' - m*m'));
MI(m' - C ~= 0) = MI(m' - C ~= 0) + MI_temp(m' - C ~= 0);

MI_temp = (1 - m - m' + C).*log2((1 - m - m' + C)./(1 - m - m' + m*m'));
MI(1 - m - m' + C ~= 0) = MI(1 - m - m' + C ~= 0) + MI_temp(1 - m - m' + C ~= 0);

% Four contributions to mutual information:
% MI_1 = C.*log2(C./(m*m'));
% MI_2 = (m - C).*log2((m - C)./(m - m*m'));
% MI_3 = (m' - C).*log2((m' - C)./(m' - m*m'));
% MI_4 = (1 - m - m' + C).*log2((1 - m - m' + C)./(1 - m - m' + m*m'));

% Compute mutual information while correcting for zero probabilities:
% MI = zeros(N);
% MI(C ~= 0) = MI_1(C ~= 0);
% MI(m - C ~= 0) = MI(m - C ~= 0) + MI_2(m - C ~= 0);
% MI(m' - C ~= 0) = MI(m' - C ~= 0) + MI_3(m' - C ~= 0);
% MI(1 - m - m' + C ~= 0) = MI(1 - m - m' + C ~= 0) + MI_4(1 - m - m' + C ~= 0);

% Correct diagonal:
MI(logical(eye(N))) = -m.*log2(m) - (1-m).*log2(1-m);

