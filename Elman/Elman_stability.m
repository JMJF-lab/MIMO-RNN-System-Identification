clear
close all
clc

rng(42,'twister');

%% ==========================================================
% ELMAN - LOCAL STABILITY ANALYSIS
%
% Final Elman architecture:
%   input delays = 1:2
%   hidden units = 16
%   transfer function = tansig
%
% Context state dimension = 16*2 = 32
%
% Local state:
%   c(k) = [h(k-1); h(k-2)]
%
% Jacobian:
%
% A(k) = [D(k)*Wrec
%         I       0   ]
%
% where
%   D(k) = diag(1-h(k).^2)
%% ==========================================================

%% ================== LOAD MODEL ==================

load elm_net
load dataset_validation.mat
load C
load S

%% ================== VALIDATION DATA ==================

dataset_validation_n = normalize(dataset, ...
    "center",C,"scale",S);

XValidation = [dataset_validation_n(:,1) ...
               dataset_validation_n(:,2)]';

TValidation = [dataset_validation_n(:,3) ...
               dataset_validation_n(:,4)]';

T_len = size(XValidation,2);

%% ================== NETWORK INFORMATION ==================

N_hidden = elm_net.layers{1}.size;
N_delays = length(elm_net.layerWeights{1,1}.delays);

N_state = N_hidden*N_delays;

fprintf('\nHidden units       = %d\n',N_hidden);
fprintf('Number of delays   = %d\n',N_delays);
fprintf('State dimension    = %d\n',N_state);

%% ================== WEIGHTS ==================

W_in  = elm_net.IW{1,1};
W_rec = elm_net.LW{1,1};
b     = elm_net.b{1};

fprintf('\nW_in size          = %d x %d\n',size(W_in,1),size(W_in,2));
fprintf('W_rec size         = %d x %d\n',size(W_rec,1),size(W_rec,2));

%% ================== INITIAL CONTEXT ==================

% MATLAB Elman network uses two delayed hidden states.
%
% Initial context is set to zero for the local trajectory analysis.

context = zeros(N_state,1);

rho_all = zeros(T_len,1);

eig_all = zeros(N_state,T_len);

%% ================== STATE TRAJECTORY ==================

for t = 1:T_len

    %% ---------- Current input ----------

    u = XValidation(:,t);

    %% ---------- Current hidden state ----------

    z = W_in*u + W_rec*context + b;

    h = tansig(z);

    %% ---------- Local derivative ----------

    D = diag(1-h.^2);

    %% ---------- Jacobian of context dynamics ----------

    A = zeros(N_state,N_state);

    % New hidden state
    A(1:N_hidden,:) = D*W_rec;

    % Shift previous hidden state
    A(N_hidden+1:end,1:N_state-N_hidden) = ...
        eye(N_state-N_hidden);

    %% ---------- Eigenvalues ----------

    lambda = eig(A);

    eig_all(:,t) = lambda;

    rho_all(t) = max(abs(lambda));

    %% ---------- Update context ----------

    context = [h; context(1:N_state-N_hidden)];

end

%% ================== STABILITY METRICS ==================

rho_eff = rho_all;

mean_rho   = mean(rho_eff);
std_rho    = std(rho_eff);
median_rho = median(rho_eff);
p95_rho    = prctile(rho_eff,95);
max_rho    = max(rho_eff);
min_rho    = min(rho_eff);

stable_percentage = mean(rho_eff < 1)*100;
unstable_percentage = mean(rho_eff >= 1)*100;

%% ================== RESULTS ==================

fprintf('\n=============================================\n');
fprintf('       ELMAN LOCAL STABILITY ANALYSIS\n');
fprintf('=============================================\n');

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

plot(1:T_len,rho_eff,'LineWidth',1.8);

hold on

yline(1,'--','LineWidth',1.5);

xlabel('Validation sample');
ylabel('\rho(A_k)');
title('Elman Local Spectral Radius');

grid on

%% ================== SAVE ==================

save Elman_stability_results.mat ...
    rho_all eig_all ...
    mean_rho std_rho median_rho ...
    p95_rho max_rho min_rho ...
    stable_percentage unstable_percentage
