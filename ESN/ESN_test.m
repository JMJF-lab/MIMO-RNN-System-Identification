%% ================== TEST ESTÁNDAR (ESN) ==================
clear;
close all;
clc;

load('esn_best_model.mat');
load('dataset_test.mat');

%% ================== NORMALIZACIÓN ==================
datasetn = normalize(dataset, "center", C, "scale", S);
XTest = datasetn(:,1:2)';
TTest = datasetn(:,3:4)';

%% ================== PARÁMETROS ==================
N_res = size(Win,1);
lr    = leaking_rate;          % Usa el leaking_rate real del modelo
T_len = size(XTest,2);

%% ================== ESTADO Y CAUSALIDAD ==================
x = zeros(N_res,1);
Y_prev = zeros(2,1);           % Inicialización y(0) = 0
X_reservoir = zeros(N_res,T_len);
Y_pred_all  = zeros(2,T_len);

%% ================== SIMULACIÓN ==================
for t = 1:T_len
    % Causalidad estricta y(t-1)
    u_aug = [XTest(:,t); Y_prev; 1];
    
    x = (1-lr)*x + lr*tanh(Win*u_aug + W*x);
    X_reservoir(:,t) = x;
    
    y = Wout * [x; XTest(:,t); 1];
    Y_pred_all(:,t) = y;
    
    % Teacher Forcing: actualizar Y_prev con la salida real target en t
    Y_prev = TTest(:,t);
end

%% ================== DESNORMALIZACIÓN ==================
Y_pred = zeros(2,T_len);
Y_true = zeros(2,T_len);
Y_pred(1,:) = Y_pred_all(1,:)*S(3) + C(3);
Y_pred(2,:) = Y_pred_all(2,:)*S(4) + C(4);
Y_true(1,:) = TTest(1,:)*S(3) + C(3);
Y_true(2,:) = TTest(2,:)*S(4) + C(4);

%% ================== WASHOUT ==================
washout = 50;
Y_pred = Y_pred(:,washout:end);
Y_true = Y_true(:,washout:end);

%% ================== ERRORES ==================
err = Y_true - Y_pred;
err_h1 = Y_true(1,:) - Y_pred(1,:);
err_h4 = Y_true(2,:) - Y_pred(2,:);

%% ================== MÉTRICAS GLOBALES ==================
mse   = mean(err(:).^2);
rmse  = sqrt(mse);
mae   = mean(abs(err(:)));
medae = median(abs(err(:)));
r2    = 1 - sum(err(:).^2) / sum((Y_true(:) - mean(Y_true(:))).^2);
da    = mean(sign(diff(Y_true(:))) == sign(diff(Y_pred(:)))) * 100;

%% ================== MÉTRICAS POR SALIDA ==================
% ---- h1 ----
mse_h1   = mean(err_h1.^2);
rmse_h1  = sqrt(mse_h1);
mae_h1   = mean(abs(err_h1));
medae_h1 = median(abs(err_h1));
r2_h1    = 1 - sum(err_h1.^2) / sum((Y_true(1,:) - mean(Y_true(1,:))).^2);
da_h1    = mean(sign(diff(Y_true(1,:))) == sign(diff(Y_pred(1,:)))) * 100;

% ---- h4 ----
mse_h4   = mean(err_h4.^2);
rmse_h4  = sqrt(mse_h4);
mae_h4   = mean(abs(err_h4));
medae_h4 = median(abs(err_h4));
r2_h4    = 1 - sum(err_h4.^2) / sum((Y_true(2,:) - mean(Y_true(2,:))).^2);
da_h4    = mean(sign(diff(Y_true(2,:))) == sign(diff(Y_pred(2,:)))) * 100;

%% ================== RESULTADOS ==================
fprintf('\n=== TEST RESULTS (GLOBAL) ===\n');
fprintf('RMSE  : %.6f\n', rmse);
fprintf('MAE   : %.6f\n', mae);
fprintf('MedAE : %.6f\n', medae);
fprintf('R2    : %.6f\n', r2);
fprintf('DA    : %.2f %%\n', da);
fprintf('\n=== h1 ===\n');
fprintf('RMSE: %.6f | MAE: %.6f | MedAE: %.6f | R2: %.6f | DA: %.2f%%\n', ...
    rmse_h1, mae_h1, medae_h1, r2_h1, da_h1);
fprintf('\n=== h4 ===\n');
fprintf('RMSE: %.6f | MAE: %.6f | MedAE: %.6f | R2: %.6f | DA: %.2f%%\n', ...
    rmse_h4, mae_h4, medae_h4, r2_h4, da_h4);

%% ================== FFT ANALYSIS ==================
% ---- FFT h1 ----
FFT_real_h1  = fft(Y_true(1,:));
FFT_pred_h1  = fft(Y_pred(1,:));
fft_error_h1 = norm(abs(FFT_real_h1) - abs(FFT_pred_h1)) / norm(abs(FFT_real_h1));

% ---- FFT h4 ----
FFT_real_h4  = fft(Y_true(2,:));
FFT_pred_h4  = fft(Y_pred(2,:));
fft_error_h4 = norm(abs(FFT_real_h4) - abs(FFT_pred_h4)) / norm(abs(FFT_real_h4));

fprintf('\n===== FFT ANALYSIS =====\n');
fprintf('FFT Error h1 = %.6f\n', fft_error_h1);
fprintf('FFT Error h4 = %.6f\n', fft_error_h4);

%% ================== PLOTS (Q1 STYLE - ESN) ==================
lw_plot  = 2.8;
lw_axes  = 2.0;
fs_axes  = 18;
fs_label = 28;
fs_title = 30;

N = size(Y_pred,2);
time_axis = 1:N;

f = figure('Units','inches', ...
           'Position',[0 0 12 12], ...
           'PaperUnits','inches', ...
           'PaperPosition',[0 0 12 12], ...
           'PaperSize',[12 12], ...
           'Color','w');

%% Subplot 1 – h1
ax1 = subplot(2,1,1);
hold on
plot(time_axis, Y_pred(1,:), 'r', 'LineWidth', lw_plot)
plot(time_axis, Y_true(1,:), 'b', 'LineWidth', lw_plot)
xlabel('Time step', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold')
ylabel('$h_1$', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold')
title('\fontsize{30}{0}\selectfont ESN Prediction vs Real Data -- $h_1$', ...
      'Interpreter', 'latex', 'FontSize', fs_title, 'FontWeight', 'bold')
legend('$\hat{h}_1$', '$h_1$', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold', 'Location', 'best')
xlim(ax1, [1 N])
grid on
set(ax1, 'FontSize', fs_axes, 'LineWidth', lw_axes, 'Box', 'on', 'TickLabelInterpreter', 'latex')

%% Subplot 2 – h4
ax2 = subplot(2,1,2);
hold on
plot(time_axis, Y_pred(2,:), 'g', 'LineWidth', lw_plot)
plot(time_axis, Y_true(2,:), 'k', 'LineWidth', lw_plot)
xlabel('Time step', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold')
ylabel('$h_4$', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold')
title('\fontsize{30}{0}\selectfont ESN Prediction vs Real Data -- $h_4$', ...
      'Interpreter', 'latex', 'FontSize', fs_title, 'FontWeight', 'bold')
legend('$\hat{h}_4$', '$h_4$', 'Interpreter', 'latex', 'FontSize', fs_label, 'FontWeight', 'bold', 'Location', 'best')
xlim(ax2, [1 N])
grid on
set(ax2, 'FontSize', fs_axes, 'LineWidth', lw_axes, 'Box', 'on', 'TickLabelInterpreter', 'latex')

%% COMPACT JOURNAL STYLE
drawnow
ti1 = ax1.TightInset;
ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];
ti2 = ax2.TightInset;
ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

%% EXPORT VECTORIAL
exportgraphics(f, 'ESN_Prediction_h1_h4.pdf', 'ContentType', 'vector');
