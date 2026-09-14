clear
close all
clc

rng(42,'twister');

load net net
load C
load S
load dataset_test.mat

datasetn = normalize(dataset,"center",C,"scale",S);

inicio = 1;

noise_levels = [0.01, 0.05];
noise_labels = {'1%','5%'};

% Network in closed-loop mode (free-running prediction)
net_narx_closed = closeloop(net);
net_narx_closed.name = [net.name ' - Closed Loop'];

for n = 1:length(noise_levels)

    ruido = noise_levels(n);

    fprintf('\n==================== Noise: %s ====================\n', noise_labels{n});

    %% =========================
    % Noise ONLY on inputs
    %% =========================
    datasetn_noise = datasetn + ruido*randn(size(datasetn));

    XTest = datasetn_noise(inicio:end,1:2); % noisy inputs
    TTest = datasetn(inicio:end,3:4);       % clean outputs (ground truth)

    %% =========================
    % NARX format
    %% =========================
    Xcell = con2seq(XTest');

    %% =========================
    % Prediction
    %% =========================
    Ypred = net_narx_closed(Xcell);
    Ypred = cell2mat(Ypred)';

    %% =========================
    % Denormalization
    %% =========================
    YTest = zeros(size(Ypred));

    YTest(:,1) = Ypred(:,1)*S(3) + C(3);
    YTest(:,2) = Ypred(:,2)*S(4) + C(4);

    TTest_desnorm(:,1) = TTest(:,1)*S(3) + C(3);
    TTest_desnorm(:,2) = TTest(:,2)*S(4) + C(4);

    %% =========================
    % h1 metrics
    %% =========================
    mse_h1 = mean((TTest_desnorm(:,1)-YTest(:,1)).^2);
    rmse_h1 = sqrt(mse_h1);
    mae_h1 = mean(abs(TTest_desnorm(:,1)-YTest(:,1)));
    medae_h1 = median(abs(TTest_desnorm(:,1)-YTest(:,1)));
    r2_h1 = 1 - sum((TTest_desnorm(:,1)-YTest(:,1)).^2) / ...
        sum((TTest_desnorm(:,1)-mean(TTest_desnorm(:,1))).^2);
    da_h1 = mean(sign(diff(TTest_desnorm(:,1)))==sign(diff(YTest(:,1))))*100;

    %% =========================
    % h4 metrics
    %% =========================
    mse_h4 = mean((TTest_desnorm(:,2)-YTest(:,2)).^2);
    rmse_h4 = sqrt(mse_h4);
    mae_h4 = mean(abs(TTest_desnorm(:,2)-YTest(:,2)));
    medae_h4 = median(abs(TTest_desnorm(:,2)-YTest(:,2)));
    r2_h4 = 1 - sum((TTest_desnorm(:,2)-YTest(:,2)).^2) / ...
        sum((TTest_desnorm(:,2)-mean(TTest_desnorm(:,2))).^2);
    da_h4 = mean(sign(diff(TTest_desnorm(:,2)))==sign(diff(YTest(:,2))))*100;

    %% =========================
    % Display results
    %% =========================
    fprintf('h1: MSE=%.4f | RMSE=%.4f | MAE=%.4f | R2=%.4f | MedAE=%.4f | DA=%.2f%%\n', ...
        mse_h1, rmse_h1, mae_h1, r2_h1, medae_h1, da_h1);

    fprintf('h4: MSE=%.4f | RMSE=%.4f | MAE=%.4f | R2=%.4f | MedAE=%.4f | DA=%.2f%%\n', ...
        mse_h4, rmse_h4, mae_h4, r2_h4, medae_h4, da_h4);

end

view(net)
