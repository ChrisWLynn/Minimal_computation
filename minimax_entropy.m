function [] = minimax_entropy(activity, neuron_id)
% Function to perform minimax entropy analysis on an individual neuron. The
% variable "activity" is an NxT matrix of binarized time-series activity,
% where N is the number of neurons in a recording and T is the length of
% the recording. The variable "neuron_id" specifies the neuron to analyze.
% This function greedily finds the minimum number of inputs needed for the
% maximum entropy model to predict all of the pairwise correlations with
% other neurons (within eperimental errors). We only allow inputs that
% spike at least once with the output neuron.

% Neuron to analyze:
y = neuron_id;

% Activity time-series:
X = full(activity);

% Number of cells:
N = size(X,1);

% Number of time bins:
T = size(X,2);

% Threshold for errors in predicted correlations relative to noise:
Cerr_thresh = 2;

% Compute averages, correlations, and mutual informations of real data (binary):
m = sum(X, 2)/T;
C = X*X'/T;
MI = mutual_information(m, C);

% Independent entropies:
Ss_tot = -m.*log2(m) - (1-m).*log2(1-m);
S_tot = Ss_tot(y);

% Numbers of inputs to consider:
nums_in = [1:10, 15:5:50, 60:10:100, 150:50:1400, N-1]; % Use for hippocampus data
% nums_in = [1:10, 15:5:50, 60:10:100, 150:50:500, 600:200:2000, 3000:1000:11000 N-1]; % Use for visual cortex data
% nums_in = 1:(N-1); % Use for C. elegans data

% Possible inputs (all neurons that activated with output at least once):
inds_Xall = setdiff(find(C(y,:) > 0), y);

nums_in = [nums_in(nums_in < length(inds_Xall)), length(inds_Xall)];
num_nums = length(nums_in);

% Inputs with positive correlations:
inds_Cpos = setdiff(find(C(y,:) > 0), y);

% Things to compute:
bs = zeros(1, num_nums); % Biases
Ws = cell(1, num_nums); % Input weights
Ss = zeros(1, num_nums); % Model entropies
inputs = []; % Inputs
complete = zeros(1, num_nums); % Indicator if gradient descent converges
num_iter = zeros(1, num_nums); % Number of gradient descent steps
Cerr_max = zeros(1, num_nums); % Maximum errors in predicted correlations (normalized)

% Initialize states:
X_temp = X;
X_b = [ones(1,T); X_temp];
Px_temp = ones(1,T)/T;

% Start with no inputs:
inputs = [];

% Initial guesses at parameters:
b = log(m(y)/(1-m(y)));
W = [];

% Exponents to use for gradient decent step size scaling:
exponents = [4/3, 1, 3/4, 1/2, 1/4, 0, -1/4];

% Loop over number of inputs:
for i = 1:num_nums

    tic

    % Possible inputs:
    inds_Xall_temp = setdiff(inds_Xall, inputs);

    % Estimate entropy drops analytically:
    if i == 1
        dSs = MI(inds_Xall_temp, y); % Exact for first input
    else
        Py_temp = (exp(-b - W'*X_temp(inputs,:)) + 1).^(-1);
        C_temp = X_b*(Px_temp.*Py_temp)';
        inputs_b = [1, inputs + 1];
        A = (X_b.*repmat(Px_temp.*Py_temp, N+1, 1))*X_b' - (X_b.*repmat(Px_temp.*(Py_temp.^2), N+1, 1))*X_b';
        dW = -A(inputs_b, inputs_b)\A(inputs_b, inds_Xall_temp + 1);
        dC_dw = diag(A(inds_Xall_temp + 1, inds_Xall_temp + 1) + A(inputs_b, inds_Xall_temp + 1)'*dW);
        dSs = 1/2*(dC_dw.^(-1)).*(C(inds_Xall_temp,y) - C_temp(inds_Xall_temp + 1)).^2;
    end

    % Pick inputs that yield largest approximate entropy drops:
    if i == 1
        num_in_new = nums_in(i);
    else
        num_in_new = nums_in(i) - nums_in(i-1);
    end
    [~, inds_new] = maxk(dSs, num_in_new);
    inputs_new = inds_Xall_temp(inds_new);
    inputs = [inputs, inputs_new];

    % Observed distribution:
    [X_unique, ~, ic] = unique(X_temp(inputs,:)', 'rows');
    X_unique = X_unique';
    Px_unique = histcounts(categorical(ic), 'Normalization', 'count')/T;

    % Fit maximum entropy model:
    for j = 1:length(exponents)
        [b_temp, W_temp, check, iter, ~] = maxEnt_neuron(m(y), C(inputs, y),...
            X_unique, Px_unique, exponents(j), log(m(y)/(1-m(y))), zeros(length(inputs), 1));

        if check == 1
            break;
        end
    end

    % Record things:
    b = b_temp;
    W = W_temp;
    bs(i) = b;
    Ws{i} = W;
    complete(i) = check;
    num_iter(i) = iter;

    % Compute entropy:
    Py = (exp(-b - W'*X(inputs,:)) + 1).^(-1);
    Ss_temp = -Py.*log2(Py) - (1-Py).*log2(1-Py);
    Ss_temp(Py == 0) = 0;
    Ss_temp(Py == 1) = 0;
    Ss(i) = mean(Ss_temp);

    % Compute max and avg error in predicted correlations:
    C_temp = X(setdiff(inds_Cpos, inputs),:)*Py'/T;
    Cerr_max(i) = max([0; abs(C(setdiff(inds_Cpos, inputs), y) - C_temp)./sqrt(C(setdiff(inds_Cpos, inputs), y)/T)]);

    % Print some things:
    % y
    % nums_in(i)
    % j
    % toc
    % complete(i)
    % Cerr_max(i)

end

% Search for smallest number of inputs to predict all correlations within errors:

% Things to record:
nums_in_search = [];
bs_search = [];
Ws_search = {};
Ss_search = [];
inputs_complete = [];
complete_search = [];
num_iter_search = [];
Cerr_max_search = [];

% Index to begin search:
ind_corr = find(Cerr_max < Cerr_thresh, 1, 'first');

if ~isempty(ind_corr)

    % Initialize things:
    num_in_good = nums_in(ind_corr);
    num_in_bad = nums_in(ind_corr - 1);
    inputs_search = inputs(1:num_in_bad);
    inputs_complete = inputs(1:num_in_good);
    W = Ws{ind_corr - 1};
    b = bs(ind_corr - 1);

    % Loop until convergence:
    while num_in_good - num_in_bad > 1

        tic

        % Possible inputs:
        inds_Xall_temp = setdiff(inds_Xall, inputs_search);

        % Compute entropy drops:
        Py_temp = (exp(-b - W'*X_temp(inputs_search,:)) + 1).^(-1);
        C_temp = X_b*(Px_temp.*Py_temp)';
        inputs_b = [1, inputs_search + 1];
        A = (X_b.*repmat(Px_temp.*Py_temp, N+1, 1))*X_b' - (X_b.*repmat(Px_temp.*(Py_temp.^2), N+1, 1))*X_b';
        dW = -A(inputs_b, inputs_b)\A(inputs_b, inds_Xall_temp + 1);
        dC_dw = diag(A(inds_Xall_temp + 1, inds_Xall_temp + 1) + A(inputs_b, inds_Xall_temp + 1)'*dW);
        dSs = 1/2*(dC_dw.^(-1)).*(C(inds_Xall_temp,y) - C_temp(inds_Xall_temp + 1)).^2;

        % Largest approximate entropy drops:
        num_in_new = round((num_in_good - num_in_bad)/2);
        [~, inds_new] = maxk(dSs, num_in_new);
        inputs_new = inds_Xall_temp(inds_new);
        inputs_temp = [inputs_search, inputs_new];

        % Observed distribution:
        [X_unique, ~, ic] = unique(X_temp(inputs_temp,:)', 'rows');
        X_unique = X_unique';
        Px_unique = histcounts(categorical(ic), 'Normalization', 'count')/T;

        % Fit max ent model:
        for j = 1:length(exponents)
            [b_temp, W_temp, check, iter, ~] = maxEnt_neuron(m(y), C(inputs_temp, y),...
                X_unique, Px_unique, exponents(j), log(m(y)/(1-m(y))), zeros(length(inputs_temp), 1));

            if check == 1
                break;
            end
        end

        % Record things:
        b_new = b_temp;
        W_new = W_temp;
        nums_in_search = [nums_in_search, length(inputs_temp)];
        bs_search = [bs_search, b_new];
        Ws_search = [Ws_search, W_new];
        complete_search = [complete_search, check];
        num_iter_search = [num_iter_search, iter];

        % Compute entropy:
        Py = (exp(-b_new - W_new'*X(inputs_temp,:)) + 1).^(-1);
        Ss_temp = -Py.*log2(Py) - (1-Py).*log2(1-Py);
        Ss_temp(Py == 0) = 0;
        Ss_temp(Py == 1) = 0;
        Ss_search = [Ss_search, mean(Ss_temp)];

        % Compute max and avg error in predicted correlations:
        C_temp = X(setdiff(inds_Cpos, inputs_temp),:)*Py'/T;
        Cerr_temp = max(abs(C(setdiff(inds_Cpos, inputs_temp), y) - C_temp)./sqrt(C(setdiff(inds_Cpos, inputs_temp), y)/T));
        Cerr_max_search = [Cerr_max_search, Cerr_temp];

        % Print some things:
        % y
        % length(inputs_temp)
        % j
        % toc
        % check
        % Cerr_temp

        % Set new upper or lower index:
        if Cerr_temp > Cerr_thresh
            num_in_bad = length(inputs_temp);
            inputs_search = inputs_temp;
            b = b_new;
            W = W_new;
        else
            num_in_good = length(inputs_temp);
            inputs_complete = inputs_temp;
        end

    end

    % Re-order and combine all search results:
    [~, inds_search] = sort(nums_in_search, 'ascend');

    nums_in = [nums_in(1:(ind_corr-1)), nums_in_search(inds_search), nums_in(ind_corr:end)];
    bs = [bs(1:(ind_corr-1)), bs_search(inds_search), bs(ind_corr:end)];
    Ws = [Ws(1:(ind_corr-1)), Ws_search(inds_search), Ws(ind_corr:end)];
    Ss = [Ss(1:(ind_corr-1)), Ss_search(inds_search), Ss(ind_corr:end)];
    complete = [complete(1:(ind_corr-1)), complete_search(inds_search), complete(ind_corr:end)];
    num_iter = [num_iter(1:(ind_corr-1)), num_iter_search(inds_search), num_iter(ind_corr:end)];
    Cerr_max = [Cerr_max(1:(ind_corr-1)), Cerr_max_search(inds_search), Cerr_max(ind_corr:end)];

end

% Save results:
save(['minimax_entropy_neuron_', num2str(y)], 'y', 'N', 'nums_in', 'Ss', 'S_tot',...
    'bs', 'Ws', 'inputs', 'inputs_complete', 'Cerr_max','complete', 'num_iter');

