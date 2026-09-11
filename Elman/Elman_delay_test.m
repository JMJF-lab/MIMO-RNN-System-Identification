%% ============================

% ELMAN TEST WITH INPUT DELAY

%% ============================

clear

close all

clc

rng(42,'twister');

%% ----------------------------

% LOAD NETWORK AND TEST DATA

%% ----------------------------

load elm_net

load C

load S

load dataset_test.mat

%% ----------------------------

% NORMALIZE DATASET

%% ----------------------------

datasetn = normalize(dataset,"center",C,"scale",S);

inicio = 1;

XTest = [datasetn(inicio:end,1), datasetn(inicio:end,2)];

TTest = [datasetn(inicio:end,3), datasetn(inicio:end,4)];

%% ----------------------------

% APPLY INPUT DELAY

%% ----------------------------

d = 1; % One-step input delay

XTest_delay = [ ...

    zeros(d,size(XTest,2));

    XTest(1:end-d,:)

];

%% ----------------------------

% PREPARE ELMAN SEQUENCE

%% ----------------------------

uTest = num2cell(XTest_delay',1);

%% ----------------------------

% ELMAN PREDICTION WITH DELAY

%% ----------------------------

yTest = elm_net(uTest);

yTest_mat = cell2mat(yTest);

%% ----------------------------

% DENORMALIZE PREDICTIONS

%% ----------------------------

YTest(:,1) = yTest_mat(1,:)'*S(3) + C(3); % h1

YTest(:,2) = yTest_mat(2,:)'*S(4) + C(4); % h4

TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3);

TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4);

%% ----------------------------

% PERFORMANCE METRICS

%% ----------------------------

% ======================

% h1

% ======================

mse_h1  = mean((TTest_desnorm(:,1)-YTest(:,1)).^2);

rmse_h1 = sqrt(mse_h1);

mae_h1  = mean(abs(TTest_desnorm(:,1)-YTest(:,1)));

ss_res  = sum((TTest_desnorm(:,1)-YTest(:,1)).^2);

ss_tot  = sum((TTest_desnorm(:,1)-mean(TTest_desnorm(:,1))).^2);

r2_h1   = 1 - ss_res/ss_tot;

medae_h1 = median(abs(TTest_desnorm(:,1)-YTest(:,1)));

da_h1 = mean( ...

    sign(diff(TTest_desnorm(:,1))) == ...

    sign(diff(YTest(:,1))) ...

    ) * 100;

% ======================

% h4

% ======================

mse_h4  = mean((TTest_desnorm(:,2)-YTest(:,2)).^2);

rmse_h4 = sqrt(mse_h4);

mae_h4  = mean(abs(TTest_desnorm(:,2)-YTest(:,2)));

ss_res  = sum((TTest_desnorm(:,2)-YTest(:,2)).^2);

ss_tot  = sum((TTest_desnorm(:,2)-mean(TTest_desnorm(:,2))).^2);

r2_h4   = 1 - ss_res/ss_tot;

medae_h4 = median(abs(TTest_desnorm(:,2)-YTest(:,2)));

da_h4 = mean( ...

    sign(diff(TTest_desnorm(:,2))) == ...

    sign(diff(YTest(:,2))) ...

    ) * 100;

%% ----------------------------

% DISPLAY RESULTS

%% ----------------------------

fprintf('\n=== ELMAN Performance Metrics with Delay d=%d ===\n', d);

fprintf(['h1: MSE=%.4f | RMSE=%.4f | MAE=%.4f | ', ...

         'R²=%.4f | MedAE=%.4f | DA=%.2f%%\n'], ...

         mse_h1, rmse_h1, mae_h1, ...

         r2_h1, medae_h1, da_h1);

fprintf(['h4: MSE=%.4f | RMSE=%.4f | MAE=%.4f | ', ...

         'R²=%.4f | MedAE=%.4f | DA=%.2f%%\n'], ...

         mse_h4, rmse_h4, mae_h4, ...

         r2_h4, medae_h4, da_h4);

%% ----------------------------

% JOURNAL-STYLE PLOTS

%% ----------------------------

lw_plot  = 2.8;

lw_axes  = 2.0;

fs_axes  = 18;

fs_label = 28;

fs_title = 30;

f = figure( ...

    'Units','inches', ...

    'Position',[0 0 12 12], ...

    'PaperUnits','inches', ...

    'PaperPosition',[0 0 12 12], ...

    'PaperSize',[12 12], ...

    'Color','w');

%% ----------------------------

% h1

%% ----------------------------

ax1 = subplot(2,1,1);

hold on

plot(YTest(:,1),'r','LineWidth',lw_plot)

plot(TTest_desnorm(:,1),'b','LineWidth',lw_plot)

xlabel('Time step', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold')

ylabel('$h_1$', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold')

title(sprintf('ELMAN Prediction vs Real Data -- $h_1$ (d=%d)',d), ...

    'Interpreter','latex', ...

    'FontSize',fs_title, ...

    'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold', ...

    'Location','best')

xlim([1 length(YTest(:,1))])

ylim([ ...

    min([YTest(:,1); TTest_desnorm(:,1)]), ...

    max([YTest(:,1); TTest_desnorm(:,1)]) ...

])

grid on

set(ax1, ...

    'FontSize',fs_axes, ...

    'LineWidth',lw_axes, ...

    'Box','on', ...

    'TickLabelInterpreter','latex')

%% ----------------------------

% h4

%% ----------------------------

ax2 = subplot(2,1,2);

hold on

plot(YTest(:,2),'g','LineWidth',lw_plot)

plot(TTest_desnorm(:,2),'k','LineWidth',lw_plot)

xlabel('Time step', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold')

ylabel('$h_4$', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold')

title(sprintf('ELMAN Prediction vs Real Data -- $h_4$ (d=%d)',d), ...

    'Interpreter','latex', ...

    'FontSize',fs_title, ...

    'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$', ...

    'Interpreter','latex', ...

    'FontSize',fs_label, ...

    'FontWeight','bold', ...

    'Location','best')

xlim([1 length(YTest(:,2))])

ylim([ ...

    min([YTest(:,2); TTest_desnorm(:,2)]), ...

    max([YTest(:,2); TTest_desnorm(:,2)]) ...

])

grid on

set(ax2, ...

    'FontSize',fs_axes, ...

    'LineWidth',lw_axes, ...

    'Box','on', ...

    'TickLabelInterpreter','latex')

%% ----------------------------

% JOURNAL-STYLE FINE ADJUSTMENTS

%% ----------------------------

drawnow

ti1 = ax1.TightInset;

ax1.Position = [ ...

    ti1(1), ...

    0.57, ...

    1-ti1(1)-ti1(3), ...

    0.32];

ti2 = ax2.TightInset;

ax2.Position = [ ...

    ti2(1), ...

    0.10, ...

    1-ti2(1)-ti2(3), ...

    0.32];

%% ----------------------------

% EXPORT VECTOR FIGURE

%% ----------------------------

exportgraphics( ...

    f, ...

    'ELMAN_Prediction_h1_h4_delay.pdf', ...

    'ContentType','vector');
