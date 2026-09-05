%% ==========================
% LSTM script with 1% and 5% noise
%% ==========================

clear
close all
clc

rng(999,'twister');  % For reproducibility

load net; load C; load S;
load dataset_test.mat  % Test dataset

datasetn = normalize(dataset,"center",C,"scale",S);
inicio = 1;

% Noise levels to evaluate: 1% and 5%
noise_levels = [0.01, 0.05];

for n = 1:length(noise_levels)

    noise_level = noise_levels(n);

    fprintf('\n==============================\n');
    fprintf('Noise level: %.0f%%\n', noise_level*100);
    fprintf('==============================\n');

    % Add noise
    datasetn_noise = datasetn + noise_level * randn(size(datasetn));

    % Prepare inputs and outputs

    % Noisy input (sensor noise)
    XTest = [datasetn_noise(inicio:end,1) datasetn_noise(inicio:end,2)];

    % Clean output (ground truth)
    TTest = [datasetn(inicio:end,3) datasetn(inicio:end,4)];

    % Prediction
    YTestn = minibatchpredict(net,XTest,MiniBatchSize=1,UniformOutput=false);
    [~, activations] = predict(net,XTest);

    % Extract LSTM activations
    lstmActivations = activations(strcmp(activations.Layer,'lstm'), :);

    HiddenState = lstmActivations{strcmp(lstmActivations.Parameter,'HiddenState'),'Value'};
    CellState   = lstmActivations{strcmp(lstmActivations.Parameter,'CellState'),'Value'};

    % Denormalize predictions
    YTest(:,1) = YTestn{1}(:,1)*S(3) + C(3);  % h1
    YTest(:,2) = YTestn{1}(:,2)*S(4) + C(4);  % h4


    % =======================
    % Metrics
    % =======================

    TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3);
    TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4);


    % --- h1 ---

    mse_h1 = mean((TTest_desnorm(inicio:end,1)-YTest(inicio:end,1)).^2);

    rmse_h1 = rmse(TTest_desnorm(inicio:end,1), YTest(inicio:end,1));

    mae_h1 = mean(abs(TTest_desnorm(inicio:end,1)-YTest(inicio:end,1)));

    ss_res = sum((TTest_desnorm(inicio:end,1)-YTest(inicio:end,1)).^2);

    ss_tot = sum((TTest_desnorm(inicio:end,1)-mean(TTest_desnorm(inicio:end,1))).^2);

    r2_h1 = 1 - ss_res/ss_tot;

    medae_h1 = median(abs(TTest_desnorm(:,1)-YTest(:,1)));

    da_h1 = mean(sign(diff(TTest_desnorm(:,1)))==sign(diff(YTest(:,1))))*100;


    % --- h4 ---

    mse_h4 = mean((TTest_desnorm(inicio:end,2)-YTest(inicio:end,2)).^2);

    rmse_h4 = rmse(TTest_desnorm(inicio:end,2), YTest(inicio:end,2));

    mae_h4 = mean(abs(TTest_desnorm(inicio:end,2)-YTest(inicio:end,2)));

    ss_res = sum((TTest_desnorm(inicio:end,2)-YTest(inicio:end,2)).^2);

    ss_tot = sum((TTest_desnorm(inicio:end,2)-mean(TTest_desnorm(inicio:end,2))).^2);

    r2_h4 = 1 - ss_res/ss_tot;

    medae_h4 = median(abs(TTest_desnorm(:,2)-YTest(:,2)));

    da_h4 = mean(sign(diff(TTest_desnorm(:,2)))==sign(diff(YTest(:,2))))*100;


    % --- Display metrics ---

    fprintf('--- h1 ---\nMSE: %.4f | RMSE: %.4f | MAE: %.4f | R2: %.4f | MedAE: %.4f | DA: %.2f%%\n', ...
        mse_h1, rmse_h1, mae_h1, r2_h1, medae_h1, da_h1);

    fprintf('--- h4 ---\nMSE: %.4f | RMSE: %.4f | MAE: %.4f | R2: %.4f | MedAE: %.4f | DA: %.2f%%\n', ...
        mse_h4, rmse_h4, mae_h4, r2_h4, medae_h4, da_h4);


    % =======================
    % LSTM activations
    % =======================

    hidden = extractdata(HiddenState{1});
    cell = extractdata(CellState{1});

    figure;

    subplot(2,1,1);
    plot(hidden,'b.-');
    title('HiddenState (LSTM)');
    xlabel('Unit');
    ylabel('Value');
    grid on;

    subplot(2,1,2);
    plot(cell,'r.-');
    title('CellState (LSTM)');
    xlabel('Unit');
    ylabel('Value');
    grid on;

end


% Ask whether to open analyzeNetwork

opcion = questdlg('¿Deseas abrir la vista de analyzeNetwork(net)?','Visualización','Sí','No','No');

if strcmpi(opcion,'Sí')
    analyzeNetwork(net);
end
