function [b, W, complete, t, err] = maxEnt_neuron(y_obs, corr_obs, X_obs, P_obs, exponent, b0, W0)
% Inputs: average value of the output y_obs and Nx1 vector of pairwise
% correlations between the output and the n inputs corr_obs, NxT matrix of
% states of the N inputs X_obs (where T is the number of samples), and 1xT
% vector P_obs of observed probabilities for the inputs. We consider a
% system with binary (0,1) variables, one output, N inputs, and T samples.
% We also take as input initial guesses at the bias b0 and weights W0.
% The variable exponent
%
% Outputs: bias b on the output and Nx1 vector W of the interactions
% between the input and outputs of the max ent model on the output that
% reproduces the average output yavg_obs and pairwise correlations corr_obs
% with the inputs.

% Algorithm parameters:
stepSize_b = 1;
stepSize_W = 1;
inertia = 0.99;
% threshold = 10^(-5); % Use this for mouse visual cortex and C elegans
threshold = 10^(-6); % Use this for mouse hippocampus
maxSteps = 10^4; % Use this for all data
steps_check = 1000; % Use this
complete = 0;

% Number of inputs and samples:
N = size(X_obs, 1);
T = size(X_obs, 2);

% Initialize variables:
b = b0;
W = W0;

% Initalize gradient step:
db_new = 0;
dW_new = zeros(N,1);

% Keep track of errors:
errs = zeros(1, maxSteps);

% Loop over maximum number of steps:
for t = 1:maxSteps

    % Record old gradient step:
    db_old = db_new;
    dW_old = dW_new;

    % Compute average output and correlations of max ent model:
    y_avg = P_obs*(exp(-b - X_obs'*W) + 1).^(-1);
    corr = X_obs*(P_obs'.*(exp(-b - X_obs'*W) + 1).^(-1));

    % New gradients:
    dy_avg = y_obs - y_avg;
    dcorr = corr_obs - corr;
    err = max(abs([dy_avg; dcorr]));

    % Terminate if gradients are sufficiently small:
    if err < threshold
        complete = 1;
        return;
    end

    % Check if gradients have grown:
    errs(t) = err;
    if t > steps_check
        if err > errs(t - steps_check)
            return;
        end
    end

    % New gradient step:
    db_new = stepSize_b*(t^exponent)*dy_avg + inertia*db_old;
    dW_new = stepSize_W*(t^exponent)*dcorr + inertia*dW_old;

    % Update parameters:
    b = b + db_new;
    W = W + dW_new;

end