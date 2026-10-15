
clear
close all
clc
rng(42,'twister');

%% ============================
% Test NARX con delay = 1 paso
% OPEN-LOOP (mismo protocolo que LSTM)
%% ============================

delay = 1;

%% Cargar red y datos
load net net
load C
load S
load dataset_test.mat

%% Normalizar dataset
datasetn = normalize(dataset,"center",C,"scale",S);

inicio = 1;

XTest = [datasetn(inicio:end,1), datasetn(inicio:end,2)];
TTest = [datasetn(inicio:end,3), datasetn(inicio:end,4)];

%% Aplicar retardo en entradas
XTest_delay = [zeros(delay,size(XTest,2)); ...
               XTest(1:end-delay,:)];

%% Convertir a formato de celdas
Xcell = con2seq(XTest_delay');
Tcell = con2seq(TTest');

%% ============================
% Predicción OPEN-LOOP
%% ============================

[Xs,Xi,Ai,Ts] = preparets( ...
    net,...
    Xcell,...
    {},...
    Tcell);

Ypred_n = net(Xs,Xi,Ai);

%% Convertir a matriz
Ypred_mat = cell2mat(Ypred_n)';
TTest_mat = cell2mat(Ts)';

%% ============================
% Desnormalización
%% ============================

YTest = zeros(size(Ypred_mat));
TTest_desnorm = zeros(size(TTest_mat));

YTest(:,1) = Ypred_mat(:,1)*S(3) + C(3);
YTest(:,2) = Ypred_mat(:,2)*S(4) + C(4);

TTest_desnorm(:,1) = TTest_mat(:,1)*S(3) + C(3);
TTest_desnorm(:,2) = TTest_mat(:,2)*S(4) + C(4);

%% ============================
% Figura
%% ============================

lw_plot  = 2.8;
lw_axes  = 2.0;
fs_axes  = 18;
fs_label = 28;
fs_title = 30;

h1_pred = YTest(:,1);
h1_real = TTest_desnorm(:,1);

h4_pred = YTest(:,2);
h4_real = TTest_desnorm(:,2);

Npred = length(h1_pred);
xNarx = 1:Npred;

f = figure('Units','inches','Position',[0 0 12 12], ...
           'PaperUnits','inches','PaperPosition',[0 0 12 12], ...
           'PaperSize',[12 12],'Color','w', ...
           'ToolBar','none','MenuBar','none');

%% Subplot 1 - h1
ax1 = subplot(2,1,1);
hold on

plot(xNarx,h1_pred,'r','LineWidth',lw_plot)
plot(xNarx,h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')
ylabel('$h_1$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

title(sprintf('NARX Prediction vs Real Data -- $h_1$ (d=%d)',delay), ...
      'Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$', ...
       'Interpreter','latex','FontSize',fs_label, ...
       'FontWeight','bold','Location','best')

xlim([1 Npred])
ylim([min([h1_pred;h1_real]) max([h1_pred;h1_real])])

grid on
set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes, ...
        'Box','on','TickLabelInterpreter','latex')

%% Subplot 2 - h4
ax2 = subplot(2,1,2);
hold on

plot(xNarx,h4_pred,'g','LineWidth',lw_plot)
plot(xNarx,h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')
ylabel('$h_4$','Interpreter','latex', ...
       'FontSize',fs_label,'FontWeight','bold')

title(sprintf('NARX Prediction vs Real Data -- $h_4$ (d=%d)',delay), ...
      'Interpreter','latex','FontSize',fs_title,'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$', ...
       'Interpreter','latex','FontSize',fs_label, ...
       'FontWeight','bold','Location','best')

xlim([1 Npred])
ylim([min([h4_pred;h4_real]) max([h4_pred;h4_real])])

grid on
set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes, ...
        'Box','on','TickLabelInterpreter','latex')

%% Ajuste de figura
drawnow

ti1 = ax1.TightInset;
ax1.Position = [ti1(1),0.57,1-ti1(1)-ti1(3),0.32];

ti2 = ax2.TightInset;
ax2.Position = [ti2(1),0.10,1-ti2(1)-ti2(3),0.32];

%% Exportar figura vectorial
exportgraphics(f,'NARX_Prediction_h1_h4_delay1.pdf', ...
               'ContentType','vector');

%% ============================
% Métricas
%% ============================

% h1
e1 = h1_real - h1_pred;

mse_h1   = mean(e1.^2);
rmse_h1  = sqrt(mse_h1);
mae_h1   = mean(abs(e1));
medae_h1 = median(abs(e1));

ss_tot_h1 = sum((h1_real-mean(h1_real)).^2);
r2_h1 = 1 - sum(e1.^2)/ss_tot_h1;

da_h1 = mean(sign(diff(h1_real)) == ...
             sign(diff(h1_pred)))*100;

% h4
e4 = h4_real - h4_pred;

mse_h4   = mean(e4.^2);
rmse_h4  = sqrt(mse_h4);
mae_h4   = mean(abs(e4));
medae_h4 = median(abs(e4));

ss_tot_h4 = sum((h4_real-mean(h4_real)).^2);
r2_h4 = 1 - sum(e4.^2)/ss_tot_h4;

da_h4 = mean(sign(diff(h4_real)) == ...
             sign(diff(h4_pred)))*100;

%% Mostrar resultados
fprintf('\n=== NARX TEST DELAY d=%d (OPEN-LOOP) ===\n',delay);

fprintf('\n--- h1 ---\n');
fprintf('MSE   = %.6f\n',mse_h1);
fprintf('RMSE  = %.6f\n',rmse_h1);
fprintf('MAE   = %.6f\n',mae_h1);
fprintf('MedAE = %.6f\n',medae_h1);
fprintf('R2    = %.6f\n',r2_h1);
fprintf('DA    = %.2f%%\n',da_h1);

fprintf('\n--- h4 ---\n');
fprintf('MSE   = %.6f\n',mse_h4);
fprintf('RMSE  = %.6f\n',rmse_h4);
fprintf('MAE   = %.6f\n',mae_h4);
fprintf('MedAE = %.6f\n',medae_h4);
fprintf('R2    = %.6f\n',r2_h4);
fprintf('DA    = %.2f%%\n',da_h4);

fprintf('\n====================================\n');
