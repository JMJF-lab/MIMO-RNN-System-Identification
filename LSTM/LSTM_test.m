%% 
clear
close all
clc
rng(42,'twister');

load net; load C; load S;

load dataset_test.mat % Test dataset
datasetn=normalize(dataset,"center",C,"scale",S);

inicio=1;

XTest=[datasetn(inicio:end,1) datasetn(inicio:end,2);]; % Inputs: p (pump) and v (valve)
TTest=[datasetn(inicio:end,3) datasetn(inicio:end,4)]; % Outputs: h1 and h4 (tank levels)

YTestn = minibatchpredict(net,XTest,MiniBatchSize=1,UniformOutput=false);
[YTtn, activations] = predict(net,XTest);

disp(activations);

% Filter activations corresponding to the LSTM layer
lstmActivations = activations(strcmp(activations.Layer, 'lstm'), :);

% Extract HiddenState and CellState values
HiddenState = lstmActivations{strcmp(lstmActivations.Parameter, 'HiddenState'), 'Value'};
CellState = lstmActivations{strcmp(lstmActivations.Parameter, 'CellState'), 'Value'};

YTest(:,1)=YTestn{1}(:,1)*S(3)+C(3);
YTest(:,2)=YTestn{1}(:,2)*S(4)+C(4);


%% =====================================================
%  SINGLE FIGURE – LSTM h1 AND h4
% =====================================================

%% Global visual parameters

lw_plot  = 2.8;
lw_axes  = 2.0;
fs_axes  = 18;
fs_label = 28;
fs_title = 30;


%% Extract signals

h1_pred = YTest(:,1);             % LSTM prediction h1
h1_real = dataset(inicio:end,3);  % Real h1 data

h4_pred = YTest(:,2);             % LSTM prediction h4
h4_real = dataset(inicio:end,4);  % Real h4 data


%% Journal-style vertical figure

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

xlabel('\fontsize{24}{0}\selectfont Time step',...
       'Interpreter','latex',...
       'FontWeight','bold')

ylabel('\fontsize{24}{0}\selectfont $h_1$',...
       'Interpreter','latex',...
       'FontWeight','bold')

title('\fontsize{30}{0}\selectfont LSTM Prediction vs Real Data -- $h_1$',...
      'Interpreter','latex',...
      'FontWeight','bold')

legend('$\hat{h_1}$','$h_1$',...
       'Interpreter','latex',...
       'FontSize',fs_label,...
       'FontWeight','bold',...
       'Location','best')

xlim([1 length(h1_pred)])
ylim([min([h1_pred; h1_real]) max([h1_pred; h1_real])])

grid on

set(ax1,'FontSize',fs_axes,'LineWidth',lw_axes,...
    'Box','on','TickLabelInterpreter','latex')


%% =====================================================
%  Subplot 2 – h4
% =====================================================

ax2 = subplot(2,1,2);
hold on

plot(h4_pred,'g','LineWidth',lw_plot)
plot(h4_real,'k','LineWidth',lw_plot)

xlabel('\fontsize{24}{0}\selectfont Time step',...
       'Interpreter','latex',...
       'FontWeight','bold')

ylabel('\fontsize{24}{0}\selectfont $h_4$',...
       'Interpreter','latex',...
       'FontWeight','bold')

title('\fontsize{30}{0}\selectfont LSTM Prediction vs Real Data -- $h_4$',...
      'Interpreter','latex',...
      'FontWeight','bold')

legend('$\hat{h_4}$','$h_4$',...
       'Interpreter','latex',...
       'FontSize',fs_label,...
       'FontWeight','bold',...
       'Location','best')

xlim([1 length(h4_pred)])
ylim([min([h4_pred; h4_real]) max([h4_pred; h4_real])])

grid on

set(ax2,'FontSize',fs_axes,'LineWidth',lw_axes,...
    'Box','on','TickLabelInterpreter','latex')


%% =====================================================
%  Fine adjustment
% =====================================================

drawnow

ti1 = ax1.TightInset;
ax1.Position = [ti1(1), 0.57, 1 - ti1(1) - ti1(3), 0.32];

ti2 = ax2.TightInset;
ax2.Position = [ti2(1), 0.10, 1 - ti2(1) - ti2(3), 0.32];


%% Export vector figure

exportgraphics(f,'LSTM_Prediction_h1_h4.pdf','ContentType','vector')


% Denormalize TTest to match the scale of YTest

TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3); % h1
TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4); % h4


% Mean squared error (MSE)

mse_val1 = mean((TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)).^2);


% Root mean squared error (RMSE)

rmse_val1 = rmse(TTest_desnorm(inicio:end,1), YTest(inicio:end,1));


% Mean absolute error (MAE)

mae_val1 = mean(abs(TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)));


% Coefficient of determination (R²)

ss_res = sum((TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)).^2);
ss_tot = sum((TTest_desnorm(inicio:end,1) - mean(TTest_desnorm(inicio:end,1))).^2);
r2_val1 = 1 - (ss_res / ss_tot);

fprintf("MSE_h1: %.4f\nRMSE_h1: %.4f\nMAE_h1: %.4f\nR2_h1: %.4f\n", mse_val1, rmse_val1, mae_val1, r2_val1);

linea = repmat('-', 1, 50); % Create a line of 50 dashes
disp(linea);


% Mean squared error (MSE)

mse_val2 = mean((TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)).^2);


% Root mean squared error (RMSE)

rmse_val2 = rmse(TTest_desnorm(inicio:end,2), YTest(inicio:end,2));


% Mean absolute error (MAE)

mae_val2 = mean(abs(TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)));


% Coefficient of determination (R²)

ss_res = sum((TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)).^2);
ss_tot = sum((TTest_desnorm(inicio:end,2) - mean(TTest_desnorm(inicio:end,2))).^2);
r2_val2 = 1 - (ss_res / ss_tot);

fprintf("MSE_h4: %.4f\nRMSE_h4: %.4f\nMAE_h4: %.4f\nR2_h4: %.4f\n", mse_val2, rmse_val2, mae_val2, r2_val2);


% Median absolute error (MedAE)

medae_h1 = median(abs(TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)));
medae_h4 = median(abs(TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)));


% Directional Accuracy (DA)
% Percentage of times the model correctly predicts the direction of change

delta_real_h1 = diff(TTest_desnorm(:,1));
delta_pred_h1 = diff(YTest(:,1));

da_h1 = mean(sign(delta_real_h1) == sign(delta_pred_h1)) * 100;

delta_real_h4 = diff(TTest_desnorm(:,2));
delta_pred_h4 = diff(YTest(:,2));

da_h4 = mean(sign(delta_real_h4) == sign(delta_pred_h4)) * 100;

fprintf('--- h1 ---\nMedAE: %.4f | DA: %.2f%%\n', medae_h1, da_h1);
fprintf('--- h4 ---\nMedAE: %.4f | DA: %.2f%%\n', medae_h4, da_h4);


%% =====================================================
%  FFT ANALYSIS (Frequency-Domain Validation)
% =====================================================

% FFT h1

FFT_real_h1 = fft(TTest_desnorm(:,1));
FFT_pred_h1 = fft(YTest(:,1));

fft_error_h1 = norm(abs(FFT_real_h1) - abs(FFT_pred_h1)) ...
    / norm(abs(FFT_real_h1));


% FFT h4

FFT_real_h4 = fft(TTest_desnorm(:,2));
FFT_pred_h4 = fft(YTest(:,2));

fft_error_h4 = norm(abs(FFT_real_h4) - abs(FFT_pred_h4)) ...
    / norm(abs(FFT_real_h4));

fprintf('\n===== FFT ANALYSIS =====\n');
fprintf('FFT Error h1 = %.6f\n', fft_error_h1);
fprintf('FFT Error h4 = %.6f\n', fft_error_h4);


%% =====================================================
%  FFT FIGURE
% =====================================================

N = length(FFT_real_h1);
f = (0:N-1);

figure('Color','w');

subplot(2,1,1)

plot(f,abs(FFT_real_h1),'b','LineWidth',2)
hold on
plot(f,abs(FFT_pred_h1),'r','LineWidth',2)

title('FFT Comparison -- $h_1$','Interpreter','latex')
xlabel('Frequency Index','Interpreter','latex')
ylabel('Magnitude','Interpreter','latex')

legend('$FFT(h_1)$','$FFT(\hat{h}_1)$',...
    'Interpreter','latex','Location','best')

grid on


subplot(2,1,2)

plot(f,abs(FFT_real_h4),'k','LineWidth',2)
hold on
plot(f,abs(FFT_pred_h4),'g','LineWidth',2)

title('FFT Comparison -- $h_4$','Interpreter','latex')
xlabel('Frequency Index','Interpreter','latex')
ylabel('Magnitude','Interpreter','latex')

legend('$FFT(h_4)$','$FFT(\hat{h}_4)$',...
    'Interpreter','latex','Location','best')

grid on


% Display results in a dialog window

mensaje = sprintf([ ...
    '--- h1 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | R² = %.4f\n' ...
    '--- h4 ---\nMSE = %.4f | RMSE = %.4f | MAE = %.4f | R² = %.4f\n\n' ...
    '¿Deseas abrir la vista de analyzeNetwork(net)?'], ...
    mse_val1, rmse_val1, mae_val1, r2_val1, ...
    mse_val2, rmse_val2, mae_val2, r2_val2);

opcion = questdlg(mensaje, ...
                  'Resultados del Modelo y Visualización', ...
                  'Sí', 'No', 'No');


% AnalyzeNetwork

if strcmpi(opcion, 'Sí')
    analyzeNetwork(net);
end


% Extract and convert states to vectors for plotting

hidden = extractdata(HiddenState{1}); % hidden is 150x1
cell = extractdata(CellState{1});     % cell is also 150x1


% Plot states

figure;

subplot(2,1,1);

plot(hidden, 'b.-');
title('HiddenState (LSTM Hidden State)');
xlabel('LSTM Unit');
ylabel('Value');
grid on;


subplot(2,1,2);

plot(cell, 'r.-');
title('CellState (LSTM Memory State)');
xlabel('LSTM Unit');
ylabel('Value');
grid on;
