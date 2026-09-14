clear
close all
clc
rng(42,'twister');

%% ================== DATA ==================

load dataset_entreno.mat
[datasetn,C,S] = normalize(dataset);

XTrain = datasetn(:,1:2)';
TTrain = datasetn(:,3:4)';

load dataset_validation.mat
dataset_validation_n = normalize(dataset,"center",C,"scale",S);

XValidation = dataset_validation_n(:,1:2)';
TValidation = dataset_validation_n(:,3:4)';

%% ================== BEST PARAMETERS ==================

spectral_radius = 0.85;
leaking_rate    = 0.10;
input_scaling   = 0.6;
lambda          = 0.01;
density         = 0.01;

N_res = 50;
washout = 50;

%% ================== RESERVOIR ==================

Win_base = rand(N_res, (2 + 2 + 1)) - 0.5;

W_base = sprandn(N_res,N_res,density);
W_base = full(W_base);

rhoW = abs(eigs(W_base,1,'largestabs'));
W_base = W_base / rhoW;

W  = spectral_radius * W_base;
Win = input_scaling * Win_base;

%% ================== TRAINING ==================

tic;

x = zeros(N_res,1);
T = size(XTrain,2);
X = zeros(N_res,T);

Y_prev = zeros(2,1);

for t = 1:T
    u_aug = [XTrain(:,t); Y_prev; 1];

    x = (1-leaking_rate)*x + leaking_rate*tanh(Win*u_aug + W*x);
    X(:,t) = x;

    Y_prev = TTrain(:,t);
end

X_aug = [X;
    XTrain;
    ones(1,T)];

X_eff = X_aug(:,washout:end);
T_eff = TTrain(:,washout:end);

Ireg = eye(size(X_eff,1));

Wout = T_eff * X_eff' / (X_eff*X_eff' + lambda*Ireg);

trainingTime = toc;

fprintf('ESN training time: %.2f s\n', trainingTime);

%% ================== VALIDATION ==================

T_len = size(XValidation,2);

x = zeros(N_res,1);
Y_pred_all = zeros(2,T_len);

for t = 1:T_len

    % OSA (teacher forcing)
    u_aug = [XValidation(:,t); TValidation(:,t); 1];

    x = (1-leaking_rate)*x + leaking_rate*tanh(Win*u_aug + W*x);

    y = Wout * [x; XValidation(:,t); 1];

    Y_pred_all(:,t) = y;

end

%% ================== DENORMALIZATION ==================

Y_pred = zeros(2,T_len);
Y_true = zeros(2,T_len);

Y_pred(1,:) = Y_pred_all(1,:)*S(3) + C(3);
Y_pred(2,:) = Y_pred_all(2,:)*S(4) + C(4);

Y_true(1,:) = TValidation(1,:)*S(3) + C(3);
Y_true(2,:) = TValidation(2,:)*S(4) + C(4);

%% ================== WASHOUT ==================

Y_pred = Y_pred(:,washout:end);
Y_true = Y_true(:,washout:end);

%% ================== ERRORS ==================

err = Y_true - Y_pred;

err_h1 = Y_true(1,:) - Y_pred(1,:);
err_h4 = Y_true(2,:) - Y_pred(2,:);

%% ================== GLOBAL METRICS ==================

mse  = mean(err(:).^2);
rmse = sqrt(mse);

mae = mean(abs(err(:)));

medae = median(abs(err(:)));

r2 = 1 - sum(err(:).^2) / sum((Y_true(:)-mean(Y_true(:))).^2);

da = mean(sign(diff(Y_true(:))) == sign(diff(Y_pred(:))))*100;

%% ================== h1 ==================

mse_h1 = mean(err_h1.^2);
rmse_h1 = sqrt(mse_h1);

mae_h1 = mean(abs(err_h1));

medae_h1 = median(abs(err_h1));

r2_h1 = 1 - sum(err_h1.^2) / sum((Y_true(1,:) - mean(Y_true(1,:))).^2);

da_h1 = mean(sign(diff(Y_true(1,:))) == sign(diff(Y_pred(1,:))))*100;

%% ================== h4 ==================

mse_h4 = mean(err_h4.^2);
rmse_h4 = sqrt(mse_h4);

mae_h4 = mean(abs(err_h4));

medae_h4 = median(abs(err_h4));

r2_h4 = 1 - sum(err_h4.^2) / sum((Y_true(2,:) - mean(Y_true(2,:))).^2);

da_h4 = mean(sign(diff(Y_true(2,:))) == sign(diff(Y_pred(2,:))))*100;

%% ================== RESULTS ==================

fprintf('\n===== VALIDATION =====\n');

fprintf('\n=== h1 ===\n');
fprintf('RMSE: %.6f | MAE: %.6f | MedAE: %.6f | R2: %.6f | DA: %.2f%%\n', ...
    rmse_h1, mae_h1, medae_h1, r2_h1, da_h1);

fprintf('\n=== h4 ===\n');
fprintf('RMSE: %.6f | MAE: %.6f | MedAE: %.6f | R2: %.6f | DA: %.2f%%\n', ...
    rmse_h4, mae_h4, medae_h4, r2_h4, da_h4);

%% ================== FFT ==================

FFT_real_h1 = fft(Y_true(1,:));
FFT_pred_h1 = fft(Y_pred(1,:));

fft_error_h1 = norm(abs(FFT_real_h1)-abs(FFT_pred_h1)) ...
             / norm(abs(FFT_real_h1));

FFT_real_h4 = fft(Y_true(2,:));
FFT_pred_h4 = fft(Y_pred(2,:));

fft_error_h4 = norm(abs(FFT_real_h4)-abs(FFT_pred_h4)) ...
             / norm(abs(FFT_real_h4));

fprintf('\n===== FFT ANALYSIS =====\n');

fprintf('FFT Error h1 = %.6f\n', fft_error_h1);
fprintf('FFT Error h4 = %.6f\n', fft_error_h4);

%% ================== SAVE ==================

save('esn_best_model.mat','Wout','Win','W','C','S');
