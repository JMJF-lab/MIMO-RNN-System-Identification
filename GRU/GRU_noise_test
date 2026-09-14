clear
close all
clc

rng(42,'twister');  % Set seed for reproducibility

load net net; load C; load S;
load dataset_test.mat % test dataset

% Normalize using the same parameters as training
datasetn = normalize(dataset, "center", C, "scale", S);

% Noise levels to evaluate
noise_levels = [0.01, 0.05]; % 1% and 5%
inicio = 1;

for n = 1:length(noise_levels)
    noise_level = noise_levels(n);
    datasetn_noise = datasetn + noise_level * randn(size(datasetn));

% Inputs with noise
XTest = [datasetn_noise(inicio:end,1) datasetn_noise(inicio:end,2)];

% Clean outputs (ground truth)
TTest = [datasetn(inicio:end,3) datasetn(inicio:end,4)];

    % Perform prediction
    YTestn = minibatchpredict(net, XTest, MiniBatchSize=1, UniformOutput=false);

    % Denormalize prediction
    YTest(:,1) = YTestn{1}(:,1) * S(3) + C(3); % h1
    YTest(:,2) = YTestn{1}(:,2) * S(4) + C(4); % h4

    % Plot
    figure
    plot(YTest(:,1), 'LineWidth', 1, 'Color', 'r')
    hold on
    plot(dataset(inicio:end,3), 'LineWidth', 1, 'Color', 'b')
    plot(YTest(:,2), 'LineWidth', 1, 'Color', 'g')
    plot(dataset(inicio:end,4), 'LineWidth', 1, 'Color', 'k')
    legend('$\hat{h_1}$', '$h_1$', '$\hat{h_4}$', '$h_4$', 'FontSize', 11, 'Interpreter', 'latex')
    title(sprintf('Noise GRU Prediction vs Real Data (%.0f%% Noise)', noise_level*100), 'FontSize', 9)

    % Denormalize TTest to match the scale of YTest
    TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3); % h1
    TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4); % h4

    % --- h1 metrics ---
    mse_val = mean((TTest_desnorm(inicio:end,1) - YTest(inicio:end,1)).^2);
    rmse_val = rmse(TTest_desnorm(inicio:end,1), YTest(inicio:end,1));
    mae_val = mean(abs(TTest_desnorm(:,1) - YTest(inicio:end,1)));
    ss_res = sum((TTest_desnorm(:,1) - YTest(inicio:end,1)).^2);
    ss_tot = sum((TTest_desnorm(:,1) - mean(TTest_desnorm(:,1))).^2);
    r2_val = 1 - (ss_res / ss_tot);

    % MedAE
    medae_val = median(abs(TTest_desnorm(:,1) - YTest(:,1)));

    % Directional Accuracy (DA)
    delta_real = diff(TTest_desnorm(:,1));
    delta_pred = diff(YTest(:,1));
    da_val = mean(sign(delta_real) == sign(delta_pred)) * 100;

    fprintf("=== Noise level: %.0f%% ===\n", noise_level*100);
    fprintf('--- h1 ---\nMSE: %.4f | RMSE: %.4f | MAE: %.4f | R2: %.4f | MedAE: %.4f | DA: %.2f%%\n', ...
        mse_val, rmse_val, mae_val, r2_val, medae_val, da_val);

    linea = repmat('-', 1, 50);  % Create a 50-character separator line
    disp(linea);

    % --- h4 metrics ---
    mse_val = mean((TTest_desnorm(inicio:end,2) - YTest(inicio:end,2)).^2);
    rmse_val = rmse(TTest_desnorm(inicio:end,2), YTest(inicio:end,2));
    mae_val = mean(abs(TTest_desnorm(:,2) - YTest(inicio:end,2)));
    ss_res = sum((TTest_desnorm(:,2) - YTest(inicio:end,2)).^2);
    ss_tot = sum((TTest_desnorm(:,2) - mean(TTest_desnorm(:,2))).^2);
    r2_val = 1 - (ss_res / ss_tot);

    % MedAE
    medae_val = median(abs(TTest_desnorm(:,2) - YTest(:,2)));

    % Directional Accuracy (DA)
    delta_real = diff(TTest_desnorm(:,2));
    delta_pred = diff(YTest(:,2));
    da_val = mean(sign(delta_real) == sign(delta_pred)) * 100;

    fprintf('--- h4 ---\nMSE: %.4f | RMSE: %.4f | MAE: %.4f | R2: %.4f | MedAE: %.4f | DA: %.2f%%\n', ...
        mse_val, rmse_val, mae_val, r2_val, medae_val, da_val);
end

analyzeNetwork(net);
