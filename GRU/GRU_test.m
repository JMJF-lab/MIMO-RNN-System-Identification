clear

close all

clc

%% =====================================================

%  LOAD NETWORK AND DATA

%% =====================================================

rng(42,'twister');

load net net; 

load C; 

load S;

load dataset_test.mat % contains 'dataset'

inicio = 1;

%% =====================================================

%  NORMALIZATION

%% =====================================================

datasetn = normalize(dataset, "center", C, "scale", S);

% Normalized inputs and outputs

XTest = [datasetn(inicio:end,1) datasetn(inicio:end,2)]; % pump and valve

TTest = [datasetn(inicio:end,3) datasetn(inicio:end,4)]; % h1 and h4

%% =====================================================

%  GRU PREDICTION

%% =====================================================

YTestn = minibatchpredict(net, XTest, MiniBatchSize=1, UniformOutput=false);

% Denormalize prediction

YTest(:,1) = YTestn{1}(:,1) * S(3) + C(3); % h1

YTest(:,2) = YTestn{1}(:,2) * S(4) + C(4); % h4

% Denormalize TTest

TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3); % h1

TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4); % h4

%% =====================================================

%  SINGLE FIGURE – GRU h1 and h4 (FINAL Q1 FORMAT)

% =====================================================

%% Global visual parameters

lw_plot  = 2.8;  

lw_axes  = 2.0;  

fs_axes  = 18;   

fs_label = 28;   

fs_title = 30;   

%% Signals

h1_pred = YTest(:,1);

h1_real = TTest_desnorm(:,1);

h4_pred = YTest(:,2);

h4_real = TTest_desnorm(:,2);

%% Create vertical journal-style figure

f = figure('Units','inches', ...

           'Position',[0 0 12 12], ...

           'PaperUnits','inches', ...

           'PaperPosition',[0 0 12 12], ...

           'PaperSize',[12 12], ...

           'Color','w', ...

           'ToolBar','none', ...

           'MenuBar','none');

%% =====================================================

%  Subplot 1 – h1

% =====================================================

ax1 = subplot(2,1,1);

hold on

plot(h1_pred,'r','LineWidth',lw_plot)

plot(h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_1$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('\fontsize{30}{0}\selectfont GRU Prediction vs Real Data -- $h_1$',...

      'Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$',...

       'Interpreter','latex','FontSize',fs_label,...

       'FontWeight','bold','Location','best')

xlim([1 length(h1_pred)])

ylim([min([h1_pred(:); h1_real(:)]), ...

      max([h1_pred(:); h1_real(:)])])

grid on

set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes,...

    'Box','on','TickLabelInterpreter','latex')

%% =====================================================

%  Subplot 2 – h4

%% =====================================================

ax2 = subplot(2,1,2);

hold on

plot(h4_pred,'g','LineWidth',lw_plot)

plot(h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_4$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('\fontsize{30}{0}\selectfont GRU Prediction vs Real Data -- $h_4$',...

      'Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$',...

       'Interpreter','latex','FontSize',fs_label,...

       'FontWeight','bold','Location','best')

xlim([1 length(h4_pred)])

ylim([min([h4_pred(:); h4_real(:)]), ...

      max([h4_pred(:); h4_real(:)])])

grid on

set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes,...

    'Box','on','TickLabelInterpreter','latex')

%% =====================================================

%  Fine journal-style adjustment (clean spacing)

%% =====================================================

drawnow

ti1 = ax1.TightInset;

ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;

ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

%% Export as vector graphics

exportgraphics(f,'GRU_Prediction_h1_h4.pdf','ContentType','vector')

%% =====================================================

%  METRICS CALCULATION

%% =====================================================

% h1

mse_val1 = mean((TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)).^2);

rmse_val1 = rmse(TTest_desnorm(inicio:end,1), YTest(inicio:end,1));

mae_val1  = mean(abs(TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)));

% >>> NEW

medae_val1 = median(abs(TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)));

da_val1 = mean(sign(diff(TTest_desnorm(inicio:end,1))) == ...

                sign(diff(YTest(inicio:end,1))))* 100;

ss_res    = sum((TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)).^2);

ss_tot    = sum((TTest_desnorm(inicio:end,1) - mean(TTest_desnorm(inicio:end,1))).^2);

r2_val1   = 1 - (ss_res / ss_tot);

fprintf("MSE-h1: %.4f\nRMSE-h1: %.4f\nMAE-h1: %.4f\nMedAE-h1: %.4f\nDA-h1: %.4f\nR2-h1: %.4f\n", ...

    mse_val1, rmse_val1, mae_val1, medae_val1, da_val1, r2_val1);

disp(repmat('-', 1, 50));

% h4

mse_val2 = mean((TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)).^2);

rmse_val2 = rmse(TTest_desnorm(inicio:end,2), YTest(inicio:end,2));

mae_val2  = mean(abs(TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)));

% >>> NEW

medae_val2 = median(abs(TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)));

da_val2 = mean(sign(diff(TTest_desnorm(inicio:end,2))) == ...

                sign(diff(YTest(inicio:end,2))))* 100;

ss_res    = sum((TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)).^2);

ss_tot    = sum((TTest_desnorm(inicio:end,2) - mean(TTest_desnorm(inicio:end,2))).^2);

r2_val2   = 1 - (ss_res / ss_tot);

fprintf("MSE-h4: %.4f\nRMSE-h4: %.4f\nMAE-h4: %.4f\nMedAE-h4: %.4f\nDA-h4: %.4f\nR2-h4: %.4f\n", ...

    mse_val2, rmse_val2, mae_val2, medae_val2, da_val2, r2_val2);

%% =====================================================

%  FFT ANALYSIS

%% =====================================================

% -------------------

% FFT h1

% -------------------

FFT_real_h1 = fft(TTest_desnorm(:,1));

FFT_pred_h1 = fft(YTest(:,1));

fft_error_h1 = norm(abs(FFT_real_h1) - abs(FFT_pred_h1)) ...

    / norm(abs(FFT_real_h1));

% -------------------

% FFT h4

% -------------------

FFT_real_h4 = fft(TTest_desnorm(:,2));

FFT_pred_h4 = fft(YTest(:,2));

fft_error_h4 = norm(abs(FFT_real_h4) - abs(FFT_pred_h4)) ...

    / norm(abs(FFT_real_h4));

fprintf('\n===== FFT ANALYSIS =====\n');

fprintf('FFT Error h1 = %.6f\n', fft_error_h1);

fprintf('FFT Error h4 = %.6f\n', fft_error_h4);

%% =====================================================

%  WINDOW OUTPUT AND NETWORK ANALYSIS

%% =====================================================

mensaje = sprintf([ ...

    '--- h1 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | R² = %.4f\n' ...

    '--- h4 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | R² = %.4f\n\n' ...

    '¿Deseas abrir la vista de analyzeNetwork(net)?'], ...

    mse_val1, rmse_val1, mae_val1, r2_val1, ...

    mse_val2, rmse_val2, mae_val2, r2_val2);

opcion = questdlg(mensaje, 'Resultados del Modelo y Visualización', 'Sí', 'No', 'No');

if strcmpi(opcion, 'Sí')

    analyzeNetwork(net);

end
