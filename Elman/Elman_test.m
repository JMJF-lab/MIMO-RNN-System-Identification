clear

close all

clc

%% =====================================================

%  TEST DATASET (ELMAN - FINAL EVALUATION)

%% =====================================================

rng(42,'twister');

load elm_net

load C

load S

load dataset_test.mat

%% =====================================================

%  NORMALIZATION (USING TRAINING STATISTICS)

%% =====================================================

datasetn = normalize(dataset,"center",C,"scale",S);

XTest = [datasetn(:,1) datasetn(:,2)]';

TTest = [datasetn(:,3) datasetn(:,4)]';

%% =====================================================

%  PREDICTION

%% =====================================================

uTest = num2cell(XTest,1);

yTest = elm_net(uTest);

yTest_mat = cell2mat(yTest);

%% =====================================================

%  DENORMALIZATION

%% =====================================================

YTest(:,1) = yTest_mat(1,:)' * S(3) + C(3);

YTest(:,2) = yTest_mat(2,:)' * S(4) + C(4);

TTest_denorm(:,1) = TTest(1,:)' * S(3) + C(3);

TTest_denorm(:,2) = TTest(2,:)' * S(4) + C(4);

%% =====================================================

%  PERFORMANCE METRICS

%% =====================================================

% -------------------

% h1

% -------------------

mse_h1 = mean((TTest_denorm(:,1) - YTest(:,1)).^2);

rmse_h1 = sqrt(mse_h1);

mae_h1 = mean(abs(TTest_denorm(:,1) - YTest(:,1)));

ss_res = sum((TTest_denorm(:,1) - YTest(:,1)).^2);

ss_tot = sum((TTest_denorm(:,1) - mean(TTest_denorm(:,1))).^2);

r2_h1 = 1 - ss_res/ss_tot;

medae_h1 = median(abs(TTest_denorm(:,1) - YTest(:,1)));

delta_real_h1 = diff(TTest_denorm(:,1));

delta_pred_h1 = diff(YTest(:,1));

da_h1 = mean(sign(delta_real_h1) == sign(delta_pred_h1)) * 100;

% -------------------

% h4

% -------------------

mse_h4 = mean((TTest_denorm(:,2) - YTest(:,2)).^2);

rmse_h4 = sqrt(mse_h4);

mae_h4 = mean(abs(TTest_denorm(:,2) - YTest(:,2)));

ss_res = sum((TTest_denorm(:,2) - YTest(:,2)).^2);

ss_tot = sum((TTest_denorm(:,2) - mean(TTest_denorm(:,2))).^2);

r2_h4 = 1 - ss_res/ss_tot;

medae_h4 = median(abs(TTest_denorm(:,2) - YTest(:,2)));

delta_real_h4 = diff(TTest_denorm(:,2));

delta_pred_h4 = diff(YTest(:,2));

da_h4 = mean(sign(delta_real_h4) == sign(delta_pred_h4)) * 100;

%% =====================================================

%  DISPLAY TEST RESULTS

%% =====================================================

fprintf('\n===== TEST (ELMAN FINAL) =====\n');

fprintf(['\n--- h1 ---\n' ...

         'MSE   = %.4f\n' ...

         'RMSE  = %.4f\n' ...

         'MAE   = %.4f\n' ...

         'R2    = %.4f\n' ...

         'MedAE = %.4f\n' ...

         'DA    = %.2f%%\n'], ...

         mse_h1, rmse_h1, mae_h1, r2_h1, medae_h1, da_h1);

fprintf(['\n--- h4 ---\n' ...

         'MSE   = %.4f\n' ...

         'RMSE  = %.4f\n' ...

         'MAE   = %.4f\n' ...

         'R2    = %.4f\n' ...

         'MedAE = %.4f\n' ...

         'DA    = %.2f%%\n'], ...

         mse_h4, rmse_h4, mae_h4, r2_h4, medae_h4, da_h4);

%% =====================================================

%  FFT ANALYSIS

%% =====================================================

% -------------------

% FFT h1

% -------------------

FFT_real_h1 = fft(TTest_denorm(:,1));

FFT_pred_h1 = fft(YTest(:,1));

fft_error_h1 = norm(abs(FFT_real_h1) - abs(FFT_pred_h1)) ...

    / norm(abs(FFT_real_h1));

% -------------------

% FFT h4

% -------------------

FFT_real_h4 = fft(TTest_denorm(:,2));

FFT_pred_h4 = fft(YTest(:,2));

fft_error_h4 = norm(abs(FFT_real_h4) - abs(FFT_pred_h4)) ...

    / norm(abs(FFT_real_h4));

fprintf('\n===== FFT ANALYSIS =====\n');

fprintf('FFT Error h1 = %.6f\n', fft_error_h1);

fprintf('FFT Error h4 = %.6f\n', fft_error_h4);

%% =====================================================

%  PLOTS

%% =====================================================

lw_plot  = 2.8;

lw_axes  = 2.0;

fs_axes  = 18;

fs_label = 28;

fs_title = 30;

N = size(YTest,1);

x = 1:N;

f = figure('Units','inches', ...

           'Position',[0 0 12 12], ...

           'PaperUnits','inches', ...

           'PaperPosition',[0 0 12 12], ...

           'PaperSize',[12 12], ...

           'Color','w');

%% =====================================================

%  h1

%% =====================================================

ax1 = subplot(2,1,1);

hold on

plot(x,YTest(:,1),'r','LineWidth',lw_plot)

plot(x,TTest_denorm(:,1),'b','LineWidth',lw_plot)

xlabel('\fontsize{24}{0}\selectfont Time step',...

       'Interpreter','latex',...

       'FontWeight','bold')

ylabel('\fontsize{24}{0}\selectfont $h_1$',...

       'Interpreter','latex',...

       'FontWeight','bold')

title('\fontsize{30}{0}\selectfont Elman Prediction vs Real Data -- $h_1$', ...

      'Interpreter','latex', ...

      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$', ...

       'Interpreter','latex', ...

       'FontSize',fs_label, ...

       'Location','best')

xlim(ax1,[1 N])   % Restrict the x-axis to the available test samples

grid on

set(ax1,'FontSize',fs_axes, ...

        'LineWidth',lw_axes, ...

        'Box','on', ...

        'TickLabelInterpreter','latex')

%% =====================================================

%  h4

%% =====================================================

ax2 = subplot(2,1,2);

hold on

plot(x,YTest(:,2),'g','LineWidth',lw_plot)

plot(x,TTest_denorm(:,2),'k','LineWidth',lw_plot)

xlabel('\fontsize{24}{0}\selectfont Time step',...

       'Interpreter','latex',...

       'FontWeight','bold')

ylabel('\fontsize{24}{0}\selectfont $h_4$',...

       'Interpreter','latex',...

       'FontWeight','bold')

title('\fontsize{30}{0}\selectfont Elman Prediction vs Real Data -- $h_4$', ...

      'Interpreter','latex', ...

      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$', ...

       'Interpreter','latex', ...

       'FontSize',fs_label, ...

       'Location','best')

xlim(ax2,[1 N])   % Apply the same x-axis restriction to the second subplot

grid on

set(ax2,'FontSize',fs_axes, ...

        'LineWidth',lw_axes, ...

        'Box','on', ...

        'TickLabelInterpreter','latex')

%% =====================================================

%  COMPACT JOURNAL STYLE (CONSISTENT WITH THE LSTM FIGURE)

%% =====================================================

drawnow

ti1 = ax1.TightInset;

ax1.Position = [ti1(1), 0.57, ...

               1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;

ax2.Position = [ti2(1), 0.10, ...

               1 - ti2(1) - ti2(3), 0.32];

%% =====================================================

%  EXPORT VECTOR FIGURE

%% =====================================================

exportgraphics(f,'Elman_Prediction_h1_h4.pdf', ...

               'ContentType','vector');
