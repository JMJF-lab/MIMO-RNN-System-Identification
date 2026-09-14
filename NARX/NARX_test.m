clear

close all

clc

%% =====================================================

%  NARX TEST

%  OPEN-LOOP FAIR COMPARISON WITH LSTM

% =====================================================

rng(42,'twister');

%% =====================================================

%  LOAD NETWORK

% =====================================================

load net net

load C

load S

%% =====================================================

%  LOAD TEST DATA

% =====================================================

load dataset_test.mat

datasetn = normalize( ...

    dataset,...

    "center",C,...

    "scale",S);

inicio = 1;

%% =====================================================

%  INPUTS AND TARGETS

% =====================================================

XTest = datasetn(inicio:end,1:2);

TTest = datasetn(inicio:end,3:4);

%% =====================================================

%  CONVERT TO CELL FORMAT

% =====================================================

Xcell = con2seq(XTest');

Tcell = con2seq(TTest');

%% =====================================================

%  OPEN-LOOP PREDICTION

% =====================================================

[Xs,Xi,Ai,Ts] = preparets( ...

    net,...

    Xcell,...

    {},...

    Tcell);

Ypred_n = net(Xs,Xi,Ai);

%% =====================================================

%  CONVERT TO MATRIX

% =====================================================

Ypred_mat = cell2mat(Ypred_n)';

TTest_mat = cell2mat(Ts)';

%% =====================================================

%  DENORMALIZATION

% =====================================================

YTest(:,1) = Ypred_mat(:,1)*S(3) + C(3);

YTest(:,2) = Ypred_mat(:,2)*S(4) + C(4);

TTest_desnorm(:,1) = TTest_mat(:,1)*S(3) + C(3);

TTest_desnorm(:,2) = TTest_mat(:,2)*S(4) + C(4);

%% =====================================================

%  METRICS

% =====================================================

%% h1

mse_h1  = mean((TTest_desnorm(:,1)-YTest(:,1)).^2);

rmse_h1 = rmse(TTest_desnorm(:,1),YTest(:,1));

mae_h1  = mean(abs(TTest_desnorm(:,1)-YTest(:,1)));

ss_res = sum((TTest_desnorm(:,1)-YTest(:,1)).^2);

ss_tot = sum((TTest_desnorm(:,1)-mean(TTest_desnorm(:,1))).^2);

r2_h1  = 1 - ss_res/ss_tot;

%% h4

mse_h4  = mean((TTest_desnorm(:,2)-YTest(:,2)).^2);

rmse_h4 = rmse(TTest_desnorm(:,2),YTest(:,2));

mae_h4  = mean(abs(TTest_desnorm(:,2)-YTest(:,2)));

ss_res = sum((TTest_desnorm(:,2)-YTest(:,2)).^2);

ss_tot = sum((TTest_desnorm(:,2)-mean(TTest_desnorm(:,2))).^2);

r2_h4  = 1 - ss_res/ss_tot;

%% =====================

% EXTRA METRICS (paper)

%% =====================

% --- h1 ---

medae_h1 = median(abs(TTest_desnorm(:,1)-YTest(:,1)));

da_h1 = mean(sign(diff(TTest_desnorm(:,1))) == sign(diff(YTest(:,1)))) * 100;

% --- h4 ---

medae_h4 = median(abs(TTest_desnorm(:,2)-YTest(:,2)));

da_h4 = mean(sign(diff(TTest_desnorm(:,2))) == sign(diff(YTest(:,2)))) * 100;

%% =====================================================

%  PRINT RESULTS

% =====================================================

fprintf('\n============= TEST =============\n');

fprintf('\n--- h1 ---\n');

fprintf('MSE  : %.4f\n',mse_h1);

fprintf('RMSE : %.4f\n',rmse_h1);

fprintf('MAE  : %.4f\n',mae_h1);

fprintf('MedAE: %.4f\n',medae_h1);

fprintf('R2   : %.4f\n',r2_h1);

fprintf('DA   : %.2f%%\n',da_h1);

fprintf('\n--- h4 ---\n');

fprintf('MSE  : %.4f\n',mse_h4);

fprintf('RMSE : %.4f\n',rmse_h4);

fprintf('MAE  : %.4f\n',mae_h4);

fprintf('MedAE: %.4f\n',medae_h4);

fprintf('R2   : %.4f\n',r2_h4);

fprintf('DA   : %.2f%%\n',da_h4);

fprintf('================================\n');

%% =====================================================

%  FFT ANALYSIS

% =====================================================

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

%  SINGLE FIGURE – NARX h1 and h4 (FINAL Q1 FORMAT)

% =====================================================

%% Global visual parameters

lw_plot  = 2.8;

lw_axes  = 2.0;

fs_axes  = 18;

fs_label = 28;

fs_title = 30;

%% Extract signals

h1_pred = YTest(:,1);

h1_real = TTest_desnorm(:,1);

h4_pred = YTest(:,2);

h4_real = TTest_desnorm(:,2);

%% Temporal information

Ntotal = size(dataset,1);

Npred  = length(h1_pred);

xNarx = (Ntotal - Npred + 1):Ntotal;

%% Vertical journal-style figure

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

plot(xNarx,h1_pred,'r','LineWidth',lw_plot)

plot(xNarx,h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_1$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('\fontsize{30}{0}\selectfont NARX Prediction vs Real Data -- $h_1$',...

      'Interpreter','latex',...

      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$',...

       'Interpreter','latex',...

       'FontSize',fs_label,...

       'FontWeight','bold',...

       'Location','best')

xlim([1 Ntotal])

ylim([min([h1_pred; h1_real]) ...

      max([h1_pred; h1_real])])

grid on

set(ax1,'FontSize',fs_axes,...

        'LineWidth',lw_axes,...

        'Box','on',...

        'TickLabelInterpreter','latex')

%% =====================================================

%  Subplot 2 – h4

% =====================================================

ax2 = subplot(2,1,2);

hold on

plot(xNarx,h4_pred,'g','LineWidth',lw_plot)

plot(xNarx,h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_4$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('\fontsize{30}{0}\selectfont NARX Prediction vs Real Data -- $h_4$',...

      'Interpreter','latex',...

      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$',...

       'Interpreter','latex',...

       'FontSize',fs_label,...

       'FontWeight','bold',...

       'Location','best')

xlim([1 Ntotal])

ylim([min([h4_pred; h4_real]) ...

      max([h4_pred; h4_real])])

grid on

set(ax2,'FontSize',fs_axes,...

        'LineWidth',lw_axes,...

        'Box','on',...

        'TickLabelInterpreter','latex')

%% =====================================================

%  Fine Q1-style adjustment (compact and aligned)

% =====================================================

drawnow

ti1 = ax1.TightInset;

ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;

ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

%% =====================================================

%  Export vector PDF

% =====================================================

exportgraphics(f,...

               'NARX_Prediction_h1_h4.pdf',...

               'ContentType','vector');
