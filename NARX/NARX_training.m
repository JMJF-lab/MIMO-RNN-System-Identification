clear

close all

clc

%% =====================================================

%  NARX TRAINING

%  Q1 FAIR COMPARISON WITH LSTM

% =====================================================

%% Reproducibility

rng(42,'twister');

%% =====================================================

%  LOAD TRAINING DATA

% =====================================================

load dataset_entreno.mat

% Z-score normalization

[datasetn,C,S] = normalize(dataset);

save C C

save S S

%% Inputs and outputs

XTrain = datasetn(:,1:2)';   % p, v

TTrain = datasetn(:,3:4)';   % h1, h4

%% =====================================================

%  LOAD VALIDATION DATA

% =====================================================

load dataset_validation.mat

% IMPORTANT:

% use training mean and standard deviation

dataset_validation_n = normalize( ...

    dataset,...

    "center",C,...

    "scale",S);

XValidation = dataset_validation_n(:,1:2)';

TValidation = dataset_validation_n(:,3:4)';

%% =====================================================

%  NARX ARCHITECTURE

% =====================================================

inputDelays    = 1:10;

feedbackDelays = 1:10;

hiddenSizes    = 50;

net = narxnet(inputDelays,...

              feedbackDelays,...

              hiddenSizes);

%% =====================================================

%  FAIR COMPARISON SETTINGS

% =====================================================

% No internal data splitting

net.divideFcn = 'dividetrain';

% No automatic preprocessing

net.inputs{1}.processFcns = {};

net.inputs{2}.processFcns = {};

net.outputs{2}.processFcns = {};

%% =====================================================

%  TRAINING CONFIGURATION

% =====================================================

% Bayesian Regularization

% Better generalization

net.trainFcn = 'trainbr';

net.trainParam.epochs   = 1000;

net.trainParam.min_grad = 1e-7;

%% =====================================================

%  PREPARE TRAINING DATA

% =====================================================

[Xs,Xi,Ai,Ts] = preparets( ...

    net,...

    con2seq(XTrain),...

    {},...

    con2seq(TTrain));

%% =====================================================

%  TRAIN NETWORK

% =====================================================

tic;

[net,tr] = train(net,Xs,Ts,Xi,Ai);

trainingTime = toc;

fprintf('\n====================================\n');

fprintf('Training finished\n');

fprintf('Training time: %.2f s\n',trainingTime);

fprintf('====================================\n');

%% =====================================================

%  SAVE NETWORK

% =====================================================

save net net

%% =====================================================

%  EXTERNAL VALIDATION (OPEN-LOOP)

% =====================================================

[Xval_s,Xival,Aival,Tval_s] = preparets( ...

    net,...

    con2seq(XValidation),...

    {},...

    con2seq(TValidation));

Yval = net(Xval_s,Xival,Aival);

%% =====================================================

%  METRICS

% =====================================================

Yval_mat = cell2mat(Yval)';

Tval_mat = cell2mat(Tval_s)';

% Denormalization

Yval_desnorm(:,1) = Yval_mat(:,1)*S(3) + C(3);

Yval_desnorm(:,2) = Yval_mat(:,2)*S(4) + C(4);

Tval_desnorm(:,1) = Tval_mat(:,1)*S(3) + C(3);

Tval_desnorm(:,2) = Tval_mat(:,2)*S(4) + C(4);

%% h1

rmse_h1 = rmse(Tval_desnorm(:,1),Yval_desnorm(:,1));

mae_h1  = mean(abs(Tval_desnorm(:,1)-Yval_desnorm(:,1)));

ss_res = sum((Tval_desnorm(:,1)-Yval_desnorm(:,1)).^2);

ss_tot = sum((Tval_desnorm(:,1)-mean(Tval_desnorm(:,1))).^2);

r2_h1  = 1 - ss_res/ss_tot;

%% h4

rmse_h4 = rmse(Tval_desnorm(:,2),Yval_desnorm(:,2));

mae_h4  = mean(abs(Tval_desnorm(:,2)-Yval_desnorm(:,2)));

ss_res = sum((Tval_desnorm(:,2)-Yval_desnorm(:,2)).^2);

ss_tot = sum((Tval_desnorm(:,2)-mean(Tval_desnorm(:,2))).^2);

r2_h4  = 1 - ss_res/ss_tot;

%% Display results

fprintf('\n=========== VALIDATION ===========\n');

fprintf('\n--- h1 ---\n');

fprintf('RMSE : %.4f\n',rmse_h1);

fprintf('MAE  : %.4f\n',mae_h1);

fprintf('R2   : %.4f\n',r2_h1);

fprintf('\n--- h4 ---\n');

fprintf('RMSE : %.4f\n',rmse_h4);

fprintf('MAE  : %.4f\n',mae_h4);

fprintf('R2   : %.4f\n',r2_h4);

fprintf('==================================\n');
