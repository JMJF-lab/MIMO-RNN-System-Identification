clear

close all

clc

rng(42,'twister');

%% =====================================================

%  LOAD NETWORK AND DATA

%% =====================================================

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

%  APPLY DELAY (d = 1)

%% =====================================================

d = 1; % one-step delay

XTest_delay = [zeros(d,size(XTest,2)); XTest(1:end-d,:)];

%% =====================================================

%  GRU PREDICTION (WITH DELAY)

%% =====================================================

YTestn = minibatchpredict(net, XTest_delay, ...

                          MiniBatchSize=1, ...

                          UniformOutput=false);

% Convert cell to matrix

YTestn = YTestn{1};

% Denormalize prediction

YTest(:,1) = YTestn(:,1) * S(3) + C(3); % h1

YTest(:,2) = YTestn(:,2) * S(4) + C(4); % h4

% Denormalize TTest

TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3); % h1

TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4); % h4

%% =====================================================

%  SINGLE FIGURE – GRU h1 and h4 (FINAL Q1 FORMAT)

%% =====================================================

lw_plot  = 2.8;  

lw_axes  = 2.0;  

fs_axes  = 18;   

fs_label = 28;   

fs_title = 32;   

h1_pred = YTest(:,1);

h1_real = TTest_desnorm(:,1);

h4_pred = YTest(:,2);

h4_real = TTest_desnorm(:,2);

f = figure('Units','inches', ...

           'Position',[0 0 12 12], ...

           'PaperUnits','inches', ...

           'PaperPosition',[0 0 12 12], ...

           'PaperSize',[12 12], ...

           'Color','w', ...

           'ToolBar','none', ...

           'MenuBar','none');

%% Subplot h1

ax1 = subplot(2,1,1);

hold on

plot(h1_pred,'r','LineWidth',lw_plot)

plot(h1_real,'b','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_1$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('GRU Prediction vs Real Data -- $h_1$',...

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

%% Subplot h4

ax2 = subplot(2,1,2);

hold on

plot(h4_pred,'g','LineWidth',lw_plot)

plot(h4_real,'k','LineWidth',lw_plot)

xlabel('Time step','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

ylabel('$h_4$','Interpreter','latex',...

       'FontSize',fs_label,'FontWeight','bold')

title('GRU Prediction vs Real Data -- $h_4$',...

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

drawnow

%% =====================================================

%  VECTOR PDF EXPORT

%% =====================================================

exportgraphics(f,'GRU_Prediction_h1_h4_delay1.pdf','ContentType','vector')

%% =====================================================

%  METRICS CALCULATION

%% =====================================================

% h1

mse_val1  = mean((TTest_desnorm(:,1) - YTest(:,1)).^2);

rmse_val1 = sqrt(mse_val1);

mae_val1  = mean(abs(TTest_desnorm(:,1) - YTest(:,1)));

medae_val1 = median(abs(TTest_desnorm(:,1) - YTest(:,1)));

ss_res    = sum((TTest_desnorm(:,1) - YTest(:,1)).^2);

ss_tot    = sum((TTest_desnorm(:,1) - mean(TTest_desnorm(:,1))).^2);

da_h1 = mean(sign(diff(TTest_desnorm(:,1))) == sign(diff(YTest(:,1)))) * 100;

r2_val1   = 1 - (ss_res / ss_tot);

fprintf("MSE-h1: %.4f\nRMSE-h1: %.4f\nMAE-h1: %.4f\nMedAE-h1: %.4f\nR2-h1: %.4f\nDA-h1: %.2f%%\n", ...

        mse_val1, rmse_val1, mae_val1, medae_val1, r2_val1, da_h1);

disp(repmat('-', 1, 50));

% h4

mse_val2  = mean((TTest_desnorm(:,2) - YTest(:,2)).^2);

rmse_val2 = sqrt(mse_val2);

mae_val2  = mean(abs(TTest_desnorm(:,2) - YTest(:,2)));

medae_val2 = median(abs(TTest_desnorm(:,2) - YTest(:,2)));

ss_res    = sum((TTest_desnorm(:,2) - YTest(:,2)).^2);

ss_tot    = sum((TTest_desnorm(:,2) - mean(TTest_desnorm(:,2))).^2);

da_h4 = mean(sign(diff(TTest_desnorm(:,2))) == sign(diff(YTest(:,2)))) * 100;

r2_val2   = 1 - (ss_res / ss_tot);

fprintf("MSE-h4: %.4f\nRMSE-h4: %.4f\nMAE-h4: %.4f\nMedAE-h4: %.4f\nR2-h4: %.4f\nDA-h4: %.2f%%\n", ...

        mse_val2, rmse_val2, mae_val2, medae_val2, r2_val2, da_h4);

%% =====================================================

%  ANALYZE NETWORK OPTION

%% =====================================================

mensaje = sprintf([ ...

    '--- h1 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | MedAE = %.4f | R² = %.4f\n' ...

    '--- h4 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | MedAE = %.4f | R² = %.4f\n\n' ...

    '¿Deseas abrir analyzeNetwork(net)?'], ...

    mse_val1, rmse_val1, mae_val1, medae_val1, r2_val1, ...

    mse_val2, rmse_val2, mae_val2, medae_val2, r2_val2);

opcion = questdlg(mensaje, 'Resultados del Modelo', 'Sí', 'No', 'No');

if strcmpi(opcion, 'Sí')

    analyzeNetwork(net);

end
