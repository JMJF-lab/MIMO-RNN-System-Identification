clear
close all
clc

rng(42,'twister');

%% ==========================================================
% ESN - LOCAL STABILITY ANALYSIS
%
% Same ESN architecture and OSA conditions as the final model
%
% State:
%   x(k) = reservoir state
%
% Dynamics:
%   x(k+1) = (1-alpha)x(k) +
%            alpha*tanh(Win*u(k) + W*x(k))
%
% Local Jacobian:
%   A(k) = (1-alpha)I +
%          alpha*diag(1-tanh(z(k)).^2)*W
%
% Stability criterion:
%   rho(A(k)) < 1
%% ==========================================================

%% ================== LOAD MODEL ==================

load esn_best_model.mat
load dataset_validation.mat

%% ================== PARAMETERS ==================

N_res = 50;
alpha = 0.10;
washout = 50;

%% ================== NORMALIZATION ==================

dataset_validation_n = normalize(dataset, ...
    "center",C,"scale",S);

XValidation = dataset_validation_n(:,1:2)';
TValidation = dataset_validation_n(:,3:4)';

T_len = size(XValidation,2);

%% ================== OSA STATE TRAJECTORY ==================

x = zeros(N_res,1);

rho_all = zeros(T_len,1);

max_eig_all = zeros(T_len,1);

eig_all = zeros(N_res,T_len);

for t = 1:T_len

    % OSA / teacher forcing
    u_aug = [XValidation(:,t); TValidation(:,t); 1];

    % Pre-activation
    z = Win*u_aug + W*x;

    % Reservoir state update
    x = (1-alpha)*x + alpha*tanh(z);

    %% ---------- Local Jacobian ----------

    D = diag(1 - tanh(z).^2);

    A = (1-alpha)*eye(N_res) + alpha*D*W;

    %% ---------- Eigenvalues ----------

    lambda = eig(A);

    eig_all(:,t) = lambda;

    rho_all(t) = max(abs(lambda));

    max_eig_all(t) = rho_all(t);

end

%% ================== REMOVE WASHOUT ==================

rho_eff = rho_all(washout:end);

eig_eff = eig_all(:,washout:end);

%% ================== STABILITY METRICS ==================

mean_rho = mean(rho_eff);

std_rho = std(rho_eff);

median_rho = median(rho_eff);

p95_rho = prctile(rho_eff,95);

max_rho = max(rho_eff);

min_rho = min(rho_eff);

stable_percentage = mean(rho_eff < 1)*100;

unstable_percentage = mean(rho_eff >= 1)*100;

%% ================== RESULTS ==================

fprintf('\n=============================================\n');
fprintf('       ESN LOCAL STABILITY ANALYSIS\n');
fprintf('=============================================\n');

fprintf('\nReservoir size       = %d\n',N_res);
fprintf('Spectral radius W   = %.4f\n',max(abs(eig(W))));
fprintf('Leaking rate        = %.4f\n',alpha);
fprintf('Washout             = %d\n',washout);

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

plot(washout:T_len,rho_eff,'LineWidth',1.8);

hold on

yline(1,'--','LineWidth',1.5);

xlabel('Validation sample');
ylabel('\rho(A_k)');
title('ESN Local Spectral Radius');

grid on

%% ================== SAVE ==================

save ESN_stability_results.mat ...
    rho_all rho_eff eig_all ...
    mean_rho std_rho median_rho ...
    p95_rho max_rho min_rho ...
    stable_percentage unstable_percentage
