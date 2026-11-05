%% ================== ESN TEST WITH NOISE ==================
clear;
close all;
clc;

%% MODEL AND DATA LOADING
load('esn_best_model.mat');
load('dataset_test.mat');

%% NORMALIZATION
datasetn = normalize(dataset, "center", C, "scale", S);

%% REPRODUCIBLE NOISE: INPUTS ONLY
rng(42,'twister');
noise_levels = [0.01 0.05];
noise_base = randn(size(datasetn(:,1:2)));

%% PARAMETERS
N_res = size(Win,1);
lr = leaking_rate;
washout = 50;

%% LOOP FOR 1% AND 5%
for n = 1:length(noise_levels)
    noise_level = noise_levels(n);

    % Add noise to p and v; h1 and h4 remain clean
    datasetn_noise = datasetn;
    datasetn_noise(:,1:2) = datasetn(:,1:2) + ...
        noise_level*noise_base;
    XTest = datasetn_noise(:,1:2)';
    TTest = datasetn(:,3:4)';
    T_len = size(XTest,2);

    %% OSA SIMULATION WITH TEACHER FORCING
    x = zeros(N_res,1);
    Y_prev = zeros(2,1);
    Y_pred_all = zeros(2,T_len);
    for t = 1:T_len
        u_aug = [XTest(:,t); Y_prev; 1];
        x = (1-lr)*x + lr*tanh(Win*u_aug + W*x);
        Y_pred_all(:,t) = Wout*[x; XTest(:,t); 1];

        % Use the clean ground-truth output for the next step
        Y_prev = TTest(:,t);
    end

   %% DENORMALIZATION
    Y_pred = zeros(2,T_len);
    Y_true = zeros(2,T_len);
    Y_pred(1,:) = Y_pred_all(1,:)*S(3) + C(3);
    Y_pred(2,:) = Y_pred_all(2,:)*S(4) + C(4);
    Y_true(1,:) = TTest(1,:)*S(3) + C(3);
    Y_true(2,:) = TTest(2,:)*S(4) + C(4);

    %% WASHOUT
    Y_pred = Y_pred(:,washout:end);
    Y_true = Y_true(:,washout:end);

    %% METRICS
    fprintf('\n=====================================\n');
    fprintf('ESN TEST WITH NOISE: %.0f %%\n',noise_level*100);
    fprintf('=====================================\n');

    for k = 1:2
        % Error
        e = Y_true(k,:) - Y_pred(k,:);

        % Accuracy metrics
        rmse_k = sqrt(mean(e.^2));
        mae_k = mean(abs(e));
        medae_k = median(abs(e));
        r2_k = 1 - sum(e.^2) / ...
            sum((Y_true(k,:) - mean(Y_true(k,:))).^2);

        % Directional Accuracy (%)
        da_k = mean( ...
            sign(diff(Y_true(k,:))) == ...
            sign(diff(Y_pred(k,:))) ...
            ) * 100;
        if k == 1
            label = 'h1';
        else
            label = 'h4';
        end
        fprintf(['%s -> RMSE: %.6f | MAE: %.6f | ' ...
                 'MedAE: %.6f | R2: %.6f | DA: %.2f%%\n'], ...
                 label, rmse_k, mae_k, medae_k, r2_k, da_k);
    end
end
