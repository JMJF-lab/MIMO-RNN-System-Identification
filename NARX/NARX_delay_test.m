clear

close all

clc

rng(42,'twister');

%% ============================

% NARX test with delay = 1 step

%% ============================

delay = 1;

% Load network and normalization parameters

load net net; 

load C; load S;

load dataset_test.mat

% Normalize inputs

datasetn = normalize(dataset, "center", C, "scale", S);

inicio = 1;

% Normalized inputs and outputs

XTest = [datasetn(inicio:end,1) datasetn(inicio:end,2)];

TTest = [datasetn(inicio:end,3) datasetn(inicio:end,4)];

% ----------------------------

% Apply input delay

% ----------------------------

XTest_delay = [zeros(delay,size(XTest,2)); XTest(1:end-delay,:)];

% Convert to cell format

Xcell = con2seq(XTest_delay');

Tcell = con2seq(TTest');

% Close loop

net_narx_closed = closeloop(net);

net_narx_closed.name = [net.name ' - Closed Loop'];

% Simulation

Ypred_n = net_narx_closed(Xcell);

% Convert to matrix

Ypred_mat = cell2mat(Ypred_n)';

%% =====================================================

%  OUTPUTS (UNCHANGED)

%% =====================================================

YTest(:,1) = Ypred_mat(:,1)*S(3) + C(3);

YTest(:,2) = Ypred_mat(:,2)*S(4) + C(4);

TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3);

TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4);

%% =====================================================

%  FIGURE (UNCHANGED)

%% =====================================================

lw_plot  = 2.8;

lw_axes  = 2.0;

fs_axes  = 18;

fs_label = 28;

fs_title = 30;

h1_pred = YTest(:,1); h1_real = TTest_desnorm(:,1);

h4_pred = YTest(:,2); h4_real = TTest_desnorm(:,2);

f = figure('Units','inches','Position',[0 0 12 12], ...

           'PaperUnits','inches','PaperPosition',[0 0 12 12], ...

           'PaperSize',[12 12],'Color','w','ToolBar','none','MenuBar','none');

ax1 = subplot(2,1,1); hold on

plot(h1_pred,'r','LineWidth',lw_plot)

plot(h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

ylabel('$h_1$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

title('NARX Prediction vs Real Data -- $h_1$','Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

grid on

set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes,'Box','on','TickLabelInterpreter','latex')

ax2 = subplot(2,1,2); hold on

plot(h4_pred,'g','LineWidth',lw_plot)

plot(h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

ylabel('$h_4$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

title('NARX Prediction vs Real Data -- $h_4$','Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

grid on

set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes,'Box','on','TickLabelInterpreter','latex')

drawnow

ti1 = ax1.TightInset;

ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;

ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];

exportgraphics(f,'NARX_Prediction_h1_h4_delay1.pdf','ContentType','vector');

%% =====================================================

%  METRICS (CORRECTED)

%% =====================================================

% --- h1 ---

e1 = TTest_desnorm(:,1) - YTest(:,1);

mse_h1  = mean(e1.^2);

rmse_h1 = sqrt(mse_h1);

mae_h1  = mean(abs(e1));

medae_h1 = median(abs(e1));

r2_h1 = 1 - sum(e1.^2) / sum((TTest_desnorm(:,1) - mean(TTest_desnorm(:,1))).^2);

da_h1 = mean(sign(diff(TTest_desnorm(:,1))) == sign(diff(YTest(:,1)))) * 100;

% --- h4 ---

e2 = TTest_desnorm(:,2) - YTest(:,2);

mse_h4  = mean(e2.^2);

rmse_h4 = sqrt(mse_h4);

mae_h4  = mean(abs(e2));

medae_h4 = median(abs(e2));

r2_h4 = 1 - sum(e2.^2) / sum((TTest_desnorm(:,2) - mean(TTest_desnorm(:,2))).^2);

da_h4 = mean(sign(diff(TTest_desnorm(:,2))) == sign(diff(YTest(:,2)))) * 100;

%% =====================================================

%  PRINT FINAL

%% =====================================================

fprintf('\n--- h1 ---\n');

fprintf('MSE=%.3f | RMSE=%.3f | MAE=%.3f | MedAE=%.3f | R2=%.3f | DA=%.2f%%\n', ...

    mse_h1, rmse_h1, mae_h1, medae_h1, r2_h1, da_h1);

fprintf('\n--- h4 ---\n');

fprintf('MSE=%.3f | RMSE=%.3f | MAE=%.3f | MedAE=%.3f | R2=%.3f | DA=%.2f%%\n', ...

    mse_h4, rmse_h4, mae_h4, medae_h4, r2_h4, da_h4);
