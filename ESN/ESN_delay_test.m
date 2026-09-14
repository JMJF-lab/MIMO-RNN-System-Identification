clear; close all; clc;

rng(42,'twister');

%% ============================
% ESN TEST WITH ONE-STEP DELAY
%% ============================

delay = 1;

%% ================== LOAD MODEL ==================

load('esn_best_model.mat');
load('dataset_test.mat');

%% ================== NORMALIZATION ==================

datasetn = normalize(dataset,"center",C,"scale",S);

XTest = datasetn(:,1:2)';   % Inputs: pump and valve
TTest = datasetn(:,3:4)';   % Ground-truth outputs: h1 and h4

%% ================== INPUT DELAY ==================

XTest_delay = [zeros(size(XTest,1), delay), ...
               XTest(:,1:end-delay)];

%% ================== ESN PARAMETERS ==================

N_res = size(Win,1);
lr   = 0.10;

washout = 50;
T_len = size(XTest_delay,2);

%% ================== ESN SIMULATION ==================

x = zeros(N_res,1);
%Y_prev = zeros(2,1);

X_reservoir = zeros(N_res,T_len);
Y_pred_all  = zeros(2,T_len);

for t = 1:T_len

    % Augmented input: inputs + feedback + bias
    %u_aug = [XTest_delay(:,t); Y_prev; 1];
    %u_aug = [XTest(:,t); TTest(:,t); 1];
    u_aug = [XTest_delay(:,t); TTest(:,t); 1];

    % Reservoir state update
    x = (1-lr)*x + lr*tanh(Win*u_aug + W*x);

    X_reservoir(:,t) = x;

    % ESN output
    y = Wout * [x; XTest_delay(:,t); 1];

    Y_pred_all(:,t) = y;

    % Autoregressive feedback
    %Y_prev = y;

end

%% ================== DENORMALIZATION ==================

YTest = zeros(2,T_len);
TTest_desnorm = zeros(2,T_len);

YTest(1,:) = Y_pred_all(1,:)*S(3) + C(3);
YTest(2,:) = Y_pred_all(2,:)*S(4) + C(4);

TTest_desnorm(1,:) = TTest(1,:)*S(3) + C(3);
TTest_desnorm(2,:) = TTest(2,:)*S(4) + C(4);

%% ================== WASHOUT ==================

YTest = YTest(:,washout:end);
TTest_desnorm = TTest_desnorm(:,washout:end);

%% ================== MAIN FIGURE ==================

lw_plot  = 2.8;
lw_axes  = 2.0;
fs_axes  = 18;
fs_label = 28;
fs_title = 32;

h1_pred = YTest(1,:);
h1_real = TTest_desnorm(1,:);
h4_pred = YTest(2,:);
h4_real = TTest_desnorm(2,:);

f = figure('Units','inches','Position',[0 0 12 12], ...
           'PaperUnits','inches','PaperPosition',[0 0 12 12], ...
           'PaperSize',[12 12],'Color','w','ToolBar','none','MenuBar','none');

% ================= h1 =================
ax1 = subplot(2,1,1); hold on
plot(h1_pred,'r','LineWidth',lw_plot)
plot(h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
ylabel('$h_1$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

title('ESN Prediction vs Real Data -- $h_1$','Interpreter','latex', ...
      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold','Location','best')

xlim([1 length(h1_pred)])
ylim([min([h1_pred h1_real]) max([h1_pred h1_real])])

grid on
set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes,'Box','on','TickLabelInterpreter','latex')

% ================= h4 =================
ax2 = subplot(2,1,2); hold on
plot(h4_pred,'g','LineWidth',lw_plot)
plot(h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')
ylabel('$h_4$','Interpreter','latex','FontSize',fs_label,'FontWeight','bold')

title('ESN Prediction vs Real Data -- $h_4$','Interpreter','latex', ...
      'FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold','Location','best')

xlim([1 length(h4_pred)])
ylim([min([h4_pred h4_real]) max([h4_pred h4_real])])

grid on
set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes,'Box','on','TickLabelInterpreter','latex')

drawnow

ti1 = ax1.TightInset;
ax1.Position = [ti1(1), 0.60, 1 - ti1(1) - ti1(3), 0.34];

ti2 = ax2.TightInset;
ax2.Position = [ti2(1), 0.08, 1 - ti2(1) - ti2(3), 0.34];

exportgraphics(f,'ESN_Prediction_h1_h4_delay1.pdf','ContentType','vector');

%% ================== METRICS ==================

calc_metrics = @(y_true,y_pred) struct( ...
    "MSE", mean((y_true - y_pred).^2), ...
    "RMSE", sqrt(mean((y_true - y_pred).^2)), ...
    "MAE", mean(abs(y_true - y_pred)), ...
    "R2", 1 - sum((y_true - y_pred).^2)/sum((y_true - mean(y_true)).^2), ...
    "MedAE", median(abs(y_true - y_pred)), ...
    "DA", mean(sign(diff(y_true)) == sign(diff(y_pred)))*100 ...
);

metrics_h1 = calc_metrics(h1_real,h1_pred);
metrics_h4 = calc_metrics(h4_real,h4_pred);

fprintf('\n--- h1 ---\n');
fprintf('MSE=%.6f | RMSE=%.6f | MAE=%.6f | R2=%.6f | MedAE=%.6f | DA=%.2f%%\n', ...
        metrics_h1.MSE, metrics_h1.RMSE, metrics_h1.MAE, metrics_h1.R2, metrics_h1.MedAE, metrics_h1.DA);

fprintf('\n--- h4 ---\n');
fprintf('MSE=%.6f | RMSE=%.6f | MAE=%.6f | R2=%.6f | MedAE=%.6f | DA=%.2f%%\n', ...
        metrics_h4.MSE, metrics_h4.RMSE, metrics_h4.MAE, metrics_h4.R2, metrics_h4.MedAE, metrics_h4.DA);

%% ================== RESERVOIR STATES ==================

figure;
plot(X_reservoir','LineWidth',1)
title('ESN Reservoir States')
xlabel('Time Step')
ylabel('Activation')
grid on
