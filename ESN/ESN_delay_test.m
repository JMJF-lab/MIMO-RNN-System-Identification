
%% ================== TEST ESN CON RETARDO d=1 ==================
clear;
close all;
clc;

%% CARGA DEL MODELO Y LOS DATOS
load('esn_best_model.mat');
load('dataset_test.mat');

%% NORMALIZACIÓN
datasetn = normalize(dataset, "center", C, "scale", S);

XTest_raw = datasetn(:,1:2)';
TTest_raw = datasetn(:,3:4)';

%% ALINEACIÓN PARA RETARDO DE UN PASO
XTest = XTest_raw(:,1:end-1);
TTest = TTest_raw(:,2:end);

T_len = size(XTest,2);

%% PARÁMETROS
N_res = size(Win,1);
lr = leaking_rate;
washout = 50;

%% SIMULACIÓN OSA CON TEACHER FORCING
x = zeros(N_res,1);
Y_prev = TTest_raw(:,1);
Y_pred_all = zeros(2,T_len);

for t = 1:T_len
    u_aug = [XTest(:,t); Y_prev; 1];
    x = (1-lr)*x + lr*tanh(Win*u_aug + W*x);

    Y_pred_all(:,t) = Wout*[x; XTest(:,t); 1];

    % Salida objetivo real del instante alineado actual
    Y_prev = TTest(:,t);
end

%% DESNORMALIZACIÓN
Y_pred = zeros(2,T_len);
Y_true = zeros(2,T_len);

Y_pred(1,:) = Y_pred_all(1,:)*S(3) + C(3);
Y_pred(2,:) = Y_pred_all(2,:)*S(4) + C(4);
Y_true(1,:) = TTest(1,:)*S(3) + C(3);
Y_true(2,:) = TTest(2,:)*S(4) + C(4);

%% WASHOUT
Y_pred = Y_pred(:,washout:end);
Y_true = Y_true(:,washout:end);

%% MÉTRICAS POR SALIDA
for k = 1:2
    e = Y_true(k,:) - Y_pred(k,:);
    mse_k = mean(e.^2);
    rmse_k = sqrt(mse_k);
    mae_k = mean(abs(e));
    medae_k = median(abs(e));
    r2_k = 1 - sum(e.^2)/ ...
        sum((Y_true(k,:)-mean(Y_true(k,:))).^2);
    da_k = mean(sign(diff(Y_true(k,:))) == ...
                sign(diff(Y_pred(k,:))))*100;

    if k == 1
        label = 'h1';
    else
        label = 'h4';
    end

    fprintf('\n===== TEST DELAY d=1: %s =====\n',label);
    fprintf('MSE: %.6f | RMSE: %.6f | MAE: %.6f\n', ...
        mse_k,rmse_k,mae_k);
    fprintf('MedAE: %.6f | R2: %.6f | DA: %.2f%%\n', ...
        medae_k,r2_k,da_k);
end

%% FFT
FFT_real_h1 = fft(Y_true(1,:));
FFT_pred_h1 = fft(Y_pred(1,:));
fft_error_h1 = norm(abs(FFT_real_h1)-abs(FFT_pred_h1)) / ...
               norm(abs(FFT_real_h1));

FFT_real_h4 = fft(Y_true(2,:));
FFT_pred_h4 = fft(Y_pred(2,:));
fft_error_h4 = norm(abs(FFT_real_h4)-abs(FFT_pred_h4)) / ...
               norm(abs(FFT_real_h4));

fprintf('\n===== FFT ERROR: DELAY d=1 =====\n');
fprintf('FFT Error h1 = %.6f\n',fft_error_h1);
fprintf('FFT Error h4 = %.6f\n',fft_error_h4);

%% GRÁFICAS
figure;

subplot(2,1,1);
plot(Y_pred(1,:),'r','LineWidth',1.5);
hold on;
plot(Y_true(1,:),'b--','LineWidth',1.5);
title('h1 - Delay d=1');
legend('Prediction','Real');
grid on;

subplot(2,1,2);
plot(Y_pred(2,:),'g','LineWidth',1.5);
hold on;
plot(Y_true(2,:),'k--','LineWidth',1.5);
title('h4 - Delay d=1');
legend('Prediction','Real');
grid on;
