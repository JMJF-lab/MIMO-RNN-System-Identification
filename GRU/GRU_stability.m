clear
close all
clc
rng(42,'twister');
%% ==========================================================
% GRU - LOCAL STABILITY ANALYSIS
%
% Final architecture:
%   2 inputs
%   150 GRU hidden units
%   batch normalization
%   2 outputs
%
% Local Jacobian:
%   A_k = d h_k / d h_{k-1}
%% ==========================================================

load net
load dataset_validation.mat
load C
load S

%% ================== VALIDATION DATA ==================

dataset_validation_n = normalize(dataset, ...
    "center",C,"scale",S);

XValidation = [dataset_validation_n(:,1) ...
               dataset_validation_n(:,2)];

T_len = size(XValidation,1);

%% ================== GRU LAYER ==================

gru = net.Layers(2);

N_hidden = gru.NumHiddenUnits;

fprintf('\n=============================================\n');
fprintf('       GRU LOCAL STABILITY ANALYSIS\n');
fprintf('=============================================\n');

fprintf('Hidden units       = %d\n',N_hidden);
fprintf('Reset gate mode    = %s\n',gru.ResetGateMode);
fprintf('State activation   = %s\n',gru.StateActivationFunction);
fprintf('Gate activation    = %s\n',gru.GateActivationFunction);

%% ================== EXTRACT LEARNABLES ==================

idxW = find(net.Learnables.Layer == "gru" & ...
            net.Learnables.Parameter == "InputWeights");

idxR = find(net.Learnables.Layer == "gru" & ...
            net.Learnables.Parameter == "RecurrentWeights");

idxB = find(net.Learnables.Layer == "gru" & ...
            net.Learnables.Parameter == "Bias");

W_all = double(extractdata(net.Learnables.Value{idxW}));
R_all = double(extractdata(net.Learnables.Value{idxR}));
b_all = double(extractdata(net.Learnables.Value{idxB}));

%% ================== SPLIT GRU PARAMETERS ==================

% MATLAB order:
%   1. Reset gate
%   2. Update gate
%   3. Candidate state

Wr = W_all(1:N_hidden,:);
Wz = W_all(N_hidden+1:2*N_hidden,:);
Wh = W_all(2*N_hidden+1:3*N_hidden,:);

Rr = R_all(1:N_hidden,:);
Rz = R_all(N_hidden+1:2*N_hidden,:);
Rh = R_all(2*N_hidden+1:3*N_hidden,:);

br = b_all(1:N_hidden);
bz = b_all(N_hidden+1:2*N_hidden);
bh = b_all(2*N_hidden+1:3*N_hidden);

%% ================== CHECK DIMENSIONS ==================

fprintf('\nParameter dimensions:\n');
fprintf('Wr = %d x %d\n',size(Wr,1),size(Wr,2));
fprintf('Wz = %d x %d\n',size(Wz,1),size(Wz,2));
fprintf('Wh = %d x %d\n',size(Wh,1),size(Wh,2));

fprintf('Rr = %d x %d\n',size(Rr,1),size(Rr,2));
fprintf('Rz = %d x %d\n',size(Rz,1),size(Rz,2));
fprintf('Rh = %d x %d\n',size(Rh,1),size(Rh,2));

%% ================== GRU STATE FUNCTION ==================

gru_state = @(x,h) ...
    (1 - 1./(1+exp(-(Wz*x + bz + Rz*h)))) .* ...
    tanh(Wh*x + bh + ...
         1./(1+exp(-(Wr*x + br + Rr*h))) .* (Rh*h)) ...
    + 1./(1+exp(-(Wz*x + bz + Rz*h))) .* h;

%% ================== LOCAL JACOBIANS ==================

rho_all = zeros(T_len,1);

% Numerical perturbation
epsilon = 1e-6;

h = zeros(N_hidden,1);

for t = 1:T_len

    x = XValidation(t,:)';

    % Current hidden state
    h_new = gru_state(x,h);

    % ------------------------------------------------------
    % Numerical Jacobian:
    %
    % A(:,j) = d h_new / d h_j
    % ------------------------------------------------------

    A = zeros(N_hidden,N_hidden);

    for j = 1:N_hidden

        hp = h;
        hm = h;

        hp(j) = hp(j) + epsilon;
        hm(j) = hm(j) - epsilon;

        fp = gru_state(x,hp);
        fm = gru_state(x,hm);

        A(:,j) = (fp - fm)/(2*epsilon);

    end

    % Spectral radius
    lambda = eig(A);

    rho_all(t) = max(abs(lambda));

    % Update hidden state
    h = h_new;

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

plot(1:T_len,rho_all,'LineWidth',1.8);

hold on

yline(1,'--','LineWidth',1.5);

xlabel('Validation sample');
ylabel('\rho(A_k)');
title('GRU Local Spectral Radius');

grid on

%% ================== SAVE ==================

save GRU_stability_results.mat ...
    rho_all ...
    mean_rho std_rho median_rho ...
    p95_rho max_rho min_rho ...
    stable_percentage unstable_percentage
