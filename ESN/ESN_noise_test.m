clear; close all; clc;

rng(42,'twister'); % Reproducibility

%% ================== LOAD MODEL ==================

load('esn_best_model.mat');   % contains best, C, S
load('dataset_test.mat');

%% ================== NORMALIZATION ==================

datasetn = normalize(dataset,"center",C,"scale",S);

%% ================== NOISE LEVELS ==================

noise_levels = [0.01 0.05];   % 1% and 5%

%% ================== ESN PARAMETERS ==================
N_res = size(Win,1);
lr   = 0.10;

washout = 50;

%% ================== NOISE LOOP ==================

for n = 1:length(noise_levels)

    noise_level = noise_levels(n);

    %% ----- Add noise ONLY to the inputs -----

    datasetn_noise = datasetn;

    datasetn_noise(:,1:2) = datasetn(:,1:2) + ...
        noise_level*randn(size(datasetn(:,1:2)));

    %% ----- Inputs and outputs -----

    XTest = datasetn_noise(:,1:2)';
    TTest = datasetn(:,3:4)';      % real outputs without noise

    T_len = size(XTest,2);

    %% ================== ESN SIMULATION ==================

    x = zeros(N_res,1);
    %Y_prev = zeros(2,1);

    X_reservoir = zeros(N_res,T_len);
    Y_pred_all  = zeros(2,T_len);

    for t = 1:T_len

        %u_aug = [XTest(:,t); Y_prev; 1];
        u_aug = [XTest(:,t); TTest(:,t); 1];

        x = (1-lr)*x + lr*tanh(Win*u_aug + W*x);

        X_reservoir(:,t) = x;

        y = Wout * [x; XTest(:,t); 1];

        Y_pred_all(:,t) = y;

        %Y_prev = y;

    end

    %% ================== DENORMALIZATION ==================

    Y_pred = zeros(2,T_len);
    Y_true = zeros(2,T_len);

    Y_pred(1,:) = Y_pred_all(1,:)*S(3) + C(3);
    Y_pred(2,:) = Y_pred_all(2,:)*S(4) + C(4);

    Y_true(1,:) = TTest(1,:)*S(3) + C(3);
    Y_true(2,:) = TTest(2,:)*S(4) + C(4);

    %% ================== WASHOUT ==================

    Y_pred = Y_pred(:,washout:end);
    Y_true = Y_true(:,washout:end);

    %% ================== ERRORS ==================

    err_h1 = Y_true(1,:) - Y_pred(1,:);
    err_h4 = Y_true(2,:) - Y_pred(2,:);

    %% ================== METRICS h1 ==================

    mse_h1   = mean(err_h1.^2);
    rmse_h1  = sqrt(mse_h1);
    mae_h1   = mean(abs(err_h1));
    medae_h1 = median(abs(err_h1));

    r2_h1 = 1 - sum(err_h1.^2) / ...
        sum((Y_true(1,:) - mean(Y_true(1,:))).^2);

    da_h1 = mean(sign(diff(Y_true(1,:))) == ...
                 sign(diff(Y_pred(1,:)))) * 100;

    %% ================== METRICS h4 ==================

    mse_h4   = mean(err_h4.^2);
    rmse_h4  = sqrt(mse_h4);
    mae_h4   = mean(abs(err_h4));
    medae_h4 = median(abs(err_h4));

    r2_h4 = 1 - sum(err_h4.^2) / ...
        sum((Y_true(2,:) - mean(Y_true(2,:))).^2);

    da_h4 = mean(sign(diff(Y_true(2,:))) == ...
                 sign(diff(Y_pred(2,:)))) * 100;

    %% ================== RESULTS ==================

    fprintf('\n=====================================\n');
    fprintf('NOISE LEVEL = %.0f %%\n',noise_level*100);
    fprintf('=====================================\n');

    fprintf('\n--- h1 ---\n');
    fprintf('MSE   : %.6f\n', mse_h1);
    fprintf('RMSE  : %.6f\n', rmse_h1);
    fprintf('MAE   : %.6f\n', mae_h1);
    fprintf('MedAE : %.6f\n', medae_h1);
    fprintf('R2    : %.6f\n', r2_h1);
    fprintf('DA    : %.2f %%\n', da_h1);

    fprintf('\n--- h4 ---\n');
    fprintf('MSE   : %.6f\n', mse_h4);
    fprintf('RMSE  : %.6f\n', rmse_h4);
    fprintf('MAE   : %.6f\n', mae_h4);
    fprintf('MedAE : %.6f\n', medae_h4);
    fprintf('R2    : %.6f\n', r2_h4);
    fprintf('DA    : %.2f %%\n', da_h4);

    %% ================== FFT ==================

    FFT_real_h1 = fft(Y_true(1,:));
    FFT_pred_h1 = fft(Y_pred(1,:));

    fft_error_h1 = norm(abs(FFT_real_h1) - abs(FFT_pred_h1)) ...
                 / norm(abs(FFT_real_h1));

    FFT_real_h4 = fft(Y_true(2,:));
    FFT_pred_h4 = fft(Y_pred(2,:));

    fft_error_h4 = norm(abs(FFT_real_h4) - abs(FFT_pred_h4)) ...
                 / norm(abs(FFT_real_h4));

    fprintf('\nFFT Error h1 = %.6f\n', fft_error_h1);
    fprintf('FFT Error h4 = %.6f\n', fft_error_h4);

    %% ================== PLOTS ==================

    figure;

    subplot(2,1,1)
    plot(Y_pred(1,:),'r','LineWidth',2); hold on
    plot(Y_true(1,:),'b','LineWidth',2)
    title(sprintf('h1 - Noise %.0f%%',noise_level*100))
    legend('pred','true')
    grid on

    subplot(2,1,2)
    plot(Y_pred(2,:),'g','LineWidth',2); hold on
    plot(Y_true(2,:),'k','LineWidth',2)
    title(sprintf('h4 - Noise %.0f%%',noise_level*100))
    legend('pred','true')
    grid on

end

%% ================== RESERVOIR STATES ==================

figure;
plot(X_reservoir','LineWidth',1)
title('Reservoir States')
xlabel('Time Step')
ylabel('Activation')
grid on
