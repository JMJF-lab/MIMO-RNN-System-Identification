clear
close all
clc
rng(42,'twister');
%% ==========================================================
% NARX - LOCAL CLOSED-LOOP STABILITY ANALYSIS
%
% Final NARX architecture:
%   2 input variables
%   2 output variables
%   input delays:    1:10
%   feedback delays: 1:10
%   hidden units:    50
%   hidden transfer: tansig
%   output transfer: purelin
%
% State:
%   20 delayed input variables
%   20 delayed output variables
%
% Total state dimension = 40
%
% The local Jacobian corresponds to the closed-loop NARX
% predictor, where predicted outputs are fed back.
%% ==========================================================

load net
load dataset_validation.mat
load C
load S

%% ================== VALIDATION DATA ==================

dataset_validation_n = normalize(dataset, ...
    "center",C,"scale",S);

XValidation = dataset_validation_n(:,1:2)';
TValidation = dataset_validation_n(:,3:4)';

T_len = size(XValidation,2);

%% ================== NETWORK PARAMETERS ==================

N_hidden = net.layers{1}.size;

N_input_delays = length(net.inputWeights{1,1}.delays);
N_feedback_delays = length(net.inputWeights{1,2}.delays);

N_state = 2*N_input_delays + ...
          2*N_feedback_delays;

fprintf('\n=============================================\n');
fprintf('       NARX LOCAL STABILITY ANALYSIS\n');
fprintf('=============================================\n');

fprintf('Hidden units       = %d\n',N_hidden);
fprintf('Input delays       = %d\n',N_input_delays);
fprintf('Feedback delays    = %d\n',N_feedback_delays);
fprintf('State dimension    = %d\n',N_state);

%% ================== EXTRACT WEIGHTS ==================

W_x = net.IW{1,1};
W_y = net.IW{1,2};

W_out = net.LW{2,1};

b_hidden = net.b{1};
b_out = net.b{2};

fprintf('\nWeight dimensions:\n');
fprintf('W_x       = %d x %d\n',size(W_x,1),size(W_x,2));
fprintf('W_y       = %d x %d\n',size(W_y,1),size(W_y,2));
fprintf('W_out     = %d x %d\n',size(W_out,1),size(W_out,2));

%% ================== INITIAL STATE ==================

% First 10 samples are used to initialize the delay state.

x_delay = zeros(2*N_input_delays,1);
y_delay = zeros(2*N_feedback_delays,1);

for k = 1:N_input_delays

    x_delay(2*(k-1)+1:2*k) = ...
        XValidation(:,N_input_delays-k+1);

    y_delay(2*(k-1)+1:2*k) = ...
        TValidation(:,N_feedback_delays-k+1);

end

state = [x_delay; y_delay];

%% ================== LOCAL JACOBIANS ==================

rho_all = zeros(T_len-N_input_delays,1);

for t = N_input_delays+1:T_len

    %% Current delayed input/output state

    x_del = state(1:2*N_input_delays);

    y_del = state(2*N_input_delays+1:end);

    %% Hidden state

    z = W_x*x_del + W_y*y_del + b_hidden;

    h = tansig(z);

    %% Predicted output

    y = W_out*h + b_out;

    %% ------------------------------------------------------
    % Jacobian of hidden layer with respect to NARX state
    % ------------------------------------------------------

    D = diag(1-h.^2);

    J_hidden = D*[W_x W_y];

    %% Jacobian of predicted output

    J_y = W_out*J_hidden;

    %% ------------------------------------------------------
    % State-transition Jacobian
    %
    % New state:
    %
    % x(t), x(t-1), ... x(t-9)
    % y(t), y(t-1), ... y(t-9)
    %
    % ------------------------------------------------------

    A = zeros(N_state,N_state);

    % Shift input delays
    A(1:2*(N_input_delays-1),1:2*N_input_delays-2) = ...
        eye(2*(N_input_delays-1));

    % New input x(t) is exogenous -> zero derivatives

    % Shift output delays
    row_start = 2*N_input_delays + 1;
    row_end = row_start + 2*(N_feedback_delays-1) - 1;

    col_start = 2*N_input_delays + 1;
    col_end = col_start + 2*(N_feedback_delays-1) - 1;

    A(row_start+2:row_end+2,col_start:col_end) = ...
        eye(2*(N_feedback_delays-1));

    % New predicted output
    A(row_start:row_start+1,:) = J_y;

    %% Spectral radius

    lambda = eig(A);

    rho_all(t-N_input_delays) = max(abs(lambda));

    %% Update NARX state

    state = [...
        XValidation(:,t);
        x_delay(1:end-2);
        y;
        y_delay(1:end-2)];

end

%% ================== STABILITY METRICS ==================

mean_rho   = mean(rho_all);
std_rho    = std(rho_all);
median_rho = median(rho_all);
p95_rho    = prctile(rho_all,95);
max_rho    = max(rho_all);
min_rho    = min(rho_all);

stable_percentage = mean(rho_all < 1)*100;
unstable_percentage = mean(rho_all >= 1)*100;

%% ================== RESULTS ==================

fprintf('\n---------------------------------------------\n');
fprintf('LOCAL JACOBIAN METRICS\n');
fprintf('---------------------------------------------\n');

fprintf('Mean rho(A)         = %.6f\n',mean_rho);
fprintf('Std rho(A)          = %.6f\n',std_rho);
fprintf('Median rho(A)       = %.6f\n',median_rho);
fprintf('95th percentile     = %.6f\n',p95_rho);
fprintf('Maximum rho(A)      = %.6f\n',max_rho);
fprintf('Minimum rho(A)      = %.6f\n',min_rho);

fprintf('\n---------------------------------------------\n');
fprintf('STABILITY OCCURRENCE\n');
fprintf('---------------------------------------------\n');

fprintf('rho(A) < 1          = %.2f %%\n',stable_percentage);
fprintf('rho(A) >= 1         = %.2f %%\n',unstable_percentage);

fprintf('\n=============================================\n');

%% ================== PLOT ==================

figure('Color','w');

plot(N_input_delays+1:T_len,rho_all,'LineWidth',1.8);

hold on

yline(1,'--','LineWidth',1.5);

xlabel('Validation sample');
ylabel('\rho(A_k)');
title('NARX Local Spectral Radius');

grid on

%% ================== SAVE ==================

save NARX_stability_results.mat ...
    rho_all ...
    mean_rho std_rho median_rho ...
    p95_rho max_rho min_rho ...
    stable_percentage unstable_percentage
