clear
close all
clc
rng(42,'twister');

load net net
load C
load S
load dataset_test.mat

% Normalizar con los parametros del entrenamiento
datasetn = normalize(dataset, "center", C, "scale", S);

noise_levels = [0.01, 0.05];
inicio = 1;

% Misma base de ruido para ambos niveles y las diez semillas
noise_base = randn(size(datasetn(:,1:2)));

for n = 1:length(noise_levels)
    noise_level = noise_levels(n);

    % Ruido solo en las entradas p y v
    XTest = datasetn(inicio:end,1:2) ...
        + noise_level * noise_base(inicio:end,:);

    % Objetivos limpios
    TTest = datasetn(inicio:end,3:4);

    % Prediccion
    YTestn = minibatchpredict(net, XTest, ...
        MiniBatchSize=1, UniformOutput=false);

    % Desnormalizar predicciones
    YTest = zeros(size(TTest));
    YTest(:,1) = YTestn{1}(:,1) * S(3) + C(3);
    YTest(:,2) = YTestn{1}(:,2) * S(4) + C(4);

    % Desnormalizar objetivos limpios
    TTest_desnorm = zeros(size(TTest));
    TTest_desnorm(:,1) = TTest(:,1) * S(3) + C(3);
    TTest_desnorm(:,2) = TTest(:,2) * S(4) + C(4);

    % Metricas h1
    err_h1 = TTest_desnorm(:,1) - YTest(:,1);
    mse_h1 = mean(err_h1.^2);
    rmse_h1 = sqrt(mse_h1);
    mae_h1 = mean(abs(err_h1));
    medae_h1 = median(abs(err_h1));
    ss_res = sum(err_h1.^2);
    ss_tot = sum((TTest_desnorm(:,1) - mean(TTest_desnorm(:,1))).^2);
    r2_h1 = 1 - ss_res/ss_tot;
    da_h1 = mean(sign(diff(TTest_desnorm(:,1))) == ...
        sign(diff(YTest(:,1)))) * 100;

    % Metricas h4
    err_h4 = TTest_desnorm(:,2) - YTest(:,2);
    mse_h4 = mean(err_h4.^2);
    rmse_h4 = sqrt(mse_h4);
    mae_h4 = mean(abs(err_h4));
    medae_h4 = median(abs(err_h4));
    ss_res = sum(err_h4.^2);
    ss_tot = sum((TTest_desnorm(:,2) - mean(TTest_desnorm(:,2))).^2);
    r2_h4 = 1 - ss_res/ss_tot;
    da_h4 = mean(sign(diff(TTest_desnorm(:,2))) == ...
        sign(diff(YTest(:,2)))) * 100;

    fprintf('\n===== GRU: NOISE %.0f%% =====\n', noise_level*100);
    fprintf(['h1: MSE=%.4f | RMSE=%.4f | MAE=%.4f | ' ...
        'MedAE=%.4f | R2=%.4f | DA=%.2f%%\n'], ...
        mse_h1, rmse_h1, mae_h1, medae_h1, r2_h1, da_h1);
    fprintf(['h4: MSE=%.4f | RMSE=%.4f | MAE=%.4f | ' ...
        'MedAE=%.4f | R2=%.4f | DA=%.2f%%\n'], ...
        mse_h4, rmse_h4, mae_h4, medae_h4, r2_h4, da_h4);

    % Grafica de predicciones y objetivos limpios
    figure
    plot(YTest(:,1), 'r', 'LineWidth', 1);
    hold on
    plot(TTest_desnorm(:,1), 'b', 'LineWidth', 1);
    plot(YTest(:,2), 'g', 'LineWidth', 1);
    plot(TTest_desnorm(:,2), 'k', 'LineWidth', 1);
    legend('$\hat{h_1}$', '$h_1$', '$\hat{h_4}$', '$h_4$', ...
        'FontSize', 11, 'Interpreter', 'latex');
    title(sprintf('GRU Prediction vs Real Data (%.0f%% Noise)', ...
        noise_level*100));
    xlabel('Time step');
    ylabel('Output');
    grid on
end
