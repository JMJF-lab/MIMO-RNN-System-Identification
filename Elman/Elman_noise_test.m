%% ==========================
% ELMAN SCRIPT WITH 1% AND 5% NOISE
%% ==========================
clear
close all
clc

rng(42,'twister');  % Reproducibility

load elm_net
load C
load S
load dataset_test.mat

%% =====================================================
%  NOISE LEVELS
%% =====================================================

noise_levels = [0.01, 0.05];

for n = 1:length(noise_levels)

    noise_level = noise_levels(n);

    fprintf('\n==============================\n');
    fprintf('Noise level: %.0f%%\n', noise_level*100);
    fprintf('==============================\n');

    %% =====================================================
    %  NORMALIZATION
    %% =====================================================

    datasetn = normalize(datasetn,"center",C,"scale",S);

    %% =====================================================
    %  ADD NOISE
    %% =====================================================

    datasetn_noise = datasetn + noise_level * randn(size(datasetn));

    %% =====================================================
    %  INPUTS AND TARGETS
    %% =====================================================

    XTest = [datasetn_noise(:,1) datasetn_noise(:,2)]';
    TTest = [datasetn_noise(:,3) datasetn_noise(:,4)]';

    %% =====================================================
    %  ELMAN PREDICTION
    %% =====================================================

    uTest = num2cell(XTest,1);

    yTest = elm_net(uTest);

    yTest_mat = cell2mat(yTest);

    %% =====================================================
    %  DENORMALIZATION
    %% =====================================================

    YTest(:,1) = yTest_mat(1,:)' * S(3) + C(3);
    YTest(:,2) = yTest_mat(2,:)' * S(4) + C(4);

    TTest_denorm(:,1) = TTest(1,:)' * S(3) + C(3);
    TTest_denorm(:,2) = TTest(2,:)' * S(4) + C(4);

    %% =====================================================
    %  h1 FIGURE
    %% =====================================================

    f1 = figure;

    set(f1,'Units','inches');
    set(f1,'Position',[0 0 8 4]);

    plot(YTest(:,1),'r','LineWidth',1.5);
    hold on
    plot(TTest_denorm(:,1),'b','LineWidth',1.5);

    title(sprintf('ELMAN Prediction vs Real Data (h_1) %.0f%% Noise', ...
        noise_level*100),'FontSize',12);

    legend('$\hat{h_1}$','$h_1$', ...
        'Interpreter','latex', ...
        'FontSize',11);

    xlabel('Time step');
    ylabel('h_1');

    grid on

    xlim([1 length(YTest(:,1))]);

    ylim([ ...
        min([YTest(:,1); TTest_denorm(:,1)]), ...
        max([YTest(:,1); TTest_denorm(:,1)]) ...
    ]);

    set(f1,'PaperUnits','inches');
    set(f1,'PaperPosition',[0 0 12 4]);
    set(f1,'PaperSize',[12 4]);

    print(f1, ...
        sprintf('ELMAN_h1_%.0fperc.png',noise_level*100), ...
        '-dpng','-r600');

    %% =====================================================
    %  h4 FIGURE
    %% =====================================================

    f2 = figure;

    set(f2,'Units','inches');
    set(f2,'Position',[0 0 8 4]);

    plot(YTest(:,2),'g','LineWidth',1.5);
    hold on
    plot(TTest_denorm(:,2),'k','LineWidth',1.5);

    title(sprintf('ELMAN Prediction vs Real Data (h_4) %.0f%% Noise', ...
        noise_level*100),'FontSize',12);

    legend('$\hat{h_4}$','$h_4$', ...
        'Interpreter','latex', ...
        'FontSize',11);

    xlabel('Time step');
    ylabel('h_4');

    grid on

    xlim([1 length(YTest(:,2))]);

    ylim([ ...
        min([YTest(:,2); TTest_denorm(:,2)]), ...
        max([YTest(:,2); TTest_denorm(:,2)]) ...
    ]);

    set(f2,'PaperUnits','inches');
    set(f2,'PaperPosition',[0 0 12 4]);
    set(f2,'PaperSize',[12 4]);

    print(f2, ...
        sprintf('ELMAN_h4_%.0fperc.png',noise_level*100), ...
        '-dpng','-r600');

    %% =====================================================
    %  h1 METRICS
    %% =====================================================

    mse_h1 = mean((TTest_denorm(:,1)-YTest(:,1)).^2);

    rmse_h1 = sqrt(mse_h1);

    mae_h1 = mean(abs(TTest_denorm(:,1)-YTest(:,1)));

    ss_res = sum((TTest_denorm(:,1)-YTest(:,1)).^2);

    ss_tot = sum((TTest_denorm(:,1)-mean(TTest_denorm(:,1))).^2);

    r2_h1 = 1 - ss_res/ss_tot;

    medae_h1 = median(abs(TTest_denorm(:,1)-YTest(:,1)));

    da_h1 = mean( ...
        sign(diff(TTest_denorm(:,1))) == ...
        sign(diff(YTest(:,1))) ...
    ) * 100;

    %% =====================================================
    %  h4 METRICS
    %% =====================================================

    mse_h4 = mean((TTest_denorm(:,2)-YTest(:,2)).^2);

    rmse_h4 = sqrt(mse_h4);

    mae_h4 = mean(abs(TTest_denorm(:,2)-YTest(:,2)));

    ss_res = sum((TTest_denorm(:,2)-YTest(:,2)).^2);

    ss_tot = sum((TTest_denorm(:,2)-mean(TTest_denorm(:,2))).^2);

    r2_h4 = 1 - ss_res/ss_tot;

    medae_h4 = median(abs(TTest_denorm(:,2)-YTest(:,2)));

    da_h4 = mean( ...
        sign(diff(TTest_denorm(:,2))) == ...
        sign(diff(YTest(:,2))) ...
    ) * 100;

    %% =====================================================
    %  DISPLAY RESULTS
    %% =====================================================

    fprintf('\n--- h1 ---\n');

    fprintf(['MSE: %.4f | RMSE: %.4f | MAE: %.4f | ', ...
             'R2: %.4f | MedAE: %.4f | DA: %.2f%%\n'], ...
             mse_h1, rmse_h1, mae_h1, ...
             r2_h1, medae_h1, da_h1);

    fprintf('\n--- h4 ---\n');

    fprintf(['MSE: %.4f | RMSE: %.4f | MAE: %.4f | ', ...
             'R2: %.4f | MedAE: %.4f | DA: %.2f%%\n'], ...
             mse_h4, rmse_h4, mae_h4, ...
             r2_h4, medae_h4, da_h4);

end
