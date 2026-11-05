clear
close all
clc

%% =====================================================
% NARX ROBUSTNESS TEST - OPEN LOOP
% Noise in p,v; clean h1,h4 reference signals
%% =====================================================

% Dedicated seed for test noise generation
rng(42,'twister');

%% MODEL AND DATA LOADING
load net net
load C
load S
load dataset_test.mat

datasetn = normalize(dataset,"center",C,"scale",S);

noise_levels = [0.01 0.05];
noise_labels = {'1%','5%'};

% The same base perturbation for both noise levels
noise_base = randn(size(datasetn(:,1:2)));

%% EVALUATION
for n = 1:length(noise_levels)

    noise_level = noise_levels(n);

    fprintf('\n====================================\n');
    fprintf('NARX OPEN-LOOP - NOISE %s\n',noise_labels{n});
    fprintf('====================================\n');

    % Add noise only to p and v
    datasetn_noise = datasetn;
    datasetn_noise(:,1:2) = datasetn(:,1:2) + noise_level*noise_base;

    XTest = datasetn_noise(:,1:2)';
    TTest = datasetn(:,3:4)';

    % Prepare sequences using the network delays
    Xcell = con2seq(XTest);
    Tcell = con2seq(TTest);

    [Xs,Xi,Ai,Ts] = preparets(net,Xcell,{},Tcell);

   % Open-loop prediction
    Ycell = net(Xs,Xi,Ai);

    Ypred = cell2mat(Ycell)';
    Tref  = cell2mat(Ts)';

    % Denormalize
    YTest = zeros(size(Ypred));
    TTest_desnorm = zeros(size(Tref));

    YTest(:,1) = Ypred(:,1)*S(3) + C(3);
    YTest(:,2) = Ypred(:,2)*S(4) + C(4);

    TTest_desnorm(:,1) = Tref(:,1)*S(3) + C(3);
    TTest_desnorm(:,2) = Tref(:,2)*S(4) + C(4);

    %% h1 METRICS
    err_h1 = TTest_desnorm(:,1) - YTest(:,1);

    mse_h1   = mean(err_h1.^2);
    rmse_h1  = sqrt(mse_h1);
    mae_h1   = mean(abs(err_h1));
    medae_h1 = median(abs(err_h1));

    ss_tot_h1 = sum((TTest_desnorm(:,1) - ...
        mean(TTest_desnorm(:,1))).^2);
    r2_h1 = 1 - sum(err_h1.^2)/ss_tot_h1;

    da_h1 = mean(sign(diff(TTest_desnorm(:,1))) == ...
        sign(diff(YTest(:,1))))*100;

    %% h4 METRICS
    err_h4 = TTest_desnorm(:,2) - YTest(:,2);

    mse_h4   = mean(err_h4.^2);
    rmse_h4  = sqrt(mse_h4);
    mae_h4   = mean(abs(err_h4));
    medae_h4 = median(abs(err_h4));

    ss_tot_h4 = sum((TTest_desnorm(:,2) - ...
        mean(TTest_desnorm(:,2))).^2);
    r2_h4 = 1 - sum(err_h4.^2)/ss_tot_h4;

    da_h4 = mean(sign(diff(TTest_desnorm(:,2))) == ...
        sign(diff(YTest(:,2))))*100;

    %% RESULTS
    fprintf('\nh1: MSE=%.6f | RMSE=%.6f | MAE=%.6f | MedAE=%.6f | R2=%.6f | DA=%.2f%%\n', ...
        mse_h1,rmse_h1,mae_h1,medae_h1,r2_h1,da_h1);

    fprintf('h4: MSE=%.6f | RMSE=%.6f | MAE=%.6f | MedAE=%.6f | R2=%.6f | DA=%.2f%%\n', ...
        mse_h4,rmse_h4,mae_h4,medae_h4,r2_h4,da_h4);

end
