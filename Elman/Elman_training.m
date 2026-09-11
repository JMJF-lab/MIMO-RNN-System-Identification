clear

close all

clc

%% =====================================================

%  ELMAN NETWORK - FAIR COMPARISON WITH LSTM

%% =====================================================

rng(42,'twister');

%% =====================================================

%  LOAD TRAINING DATA

%% =====================================================

load dataset_entreno.mat

% Normalization consistent with the LSTM and NARX models

[datasetn,C,S] = normalize(dataset);

save C C

save S S

XTrain = [datasetn(:,1) datasetn(:,2)]';

TTrain = [datasetn(:,3) datasetn(:,4)]';

%% =====================================================

%  LOAD VALIDATION DATA (EXTERNAL - SAME AS LSTM)

%% =====================================================

load dataset_validation.mat

dataset_validation_n = normalize(dataset,"center",C,"scale",S);

XValidation = [dataset_validation_n(:,1) dataset_validation_n(:,2)]';

TValidation = [dataset_validation_n(:,3) dataset_validation_n(:,4)]';

%% =====================================================

%  CREATE ELMAN NETWORK

%% =====================================================

hiddenSize = 16;

elm_net = elmannet(1:2, hiddenSize);

elm_net.layers{1}.transferFcn = 'tansig';

elm_net.layers{2}.transferFcn = 'purelin';

%% =====================================================

%  FAIR COMPARISON SETTINGS

%% =====================================================

% Disable the internal data division to use the external validation set

elm_net.divideFcn = 'dividetrain';

% Disable automatic input and output preprocessing

elm_net.inputs{1}.processFcns = {};

elm_net.outputs{2}.processFcns = {};

%% =====================================================

%  TRAINING CONFIGURATION

%% =====================================================

elm_net.trainFcn = 'trainlm';

elm_net.trainParam.epochs = 1700;

elm_net.trainParam.min_grad = 1e-10;

% Levenberg-Marquardt optimization parameters

elm_net.trainParam.mu       = 0.05;

elm_net.trainParam.mu_inc   = 5;

elm_net.trainParam.mu_dec = 0.1;

elm_net.trainParam.mu_max   = 1e20;

%% =====================================================

%  PREPARE INPUT AND TARGET SEQUENCES

%% =====================================================

u = num2cell(XTrain,1);

y = num2cell(TTrain,1);

%% =====================================================

%  TRAINING

%% =====================================================

tic;

%elm_net = configure(elm_net,u,y);

%elm_net = init(elm_net);

[elm_net, tr] = train(elm_net, u, y);

trainingTime = toc;

fprintf('Training time: %.2f s\n',trainingTime);

save elm_net elm_net

%% =====================================================

%  EXTERNAL VALIDATION

%% =====================================================

uVal = num2cell(XValidation,1);

yVal = num2cell(TValidation,1);

yPred = elm_net(uVal);

yPred_mat = cell2mat(yPred);

%% =====================================================

%  DENORMALIZATION

%% =====================================================

YPred(:,1) = yPred_mat(1,:)' * S(3) + C(3);

YPred(:,2) = yPred_mat(2,:)' * S(4) + C(4);

YTrue(:,1) = TValidation(1,:)' * S(3) + C(3);

YTrue(:,2) = TValidation(2,:)' * S(4) + C(4);

%% =====================================================

%  PERFORMANCE METRICS

%% =====================================================

rmse_h1 = rmse(YTrue(:,1),YPred(:,1));

rmse_h4 = rmse(YTrue(:,2),YPred(:,2));

mae_h1 = mean(abs(YTrue(:,1)-YPred(:,1)));

mae_h4 = mean(abs(YTrue(:,2)-YPred(:,2)));

ss_res = sum((YTrue(:,1)-YPred(:,1)).^2);

ss_tot = sum((YTrue(:,1)-mean(YTrue(:,1))).^2);

r2_h1 = 1 - ss_res/ss_tot;

ss_res = sum((YTrue(:,2)-YPred(:,2)).^2);

ss_tot = sum((YTrue(:,2)-mean(YTrue(:,2))).^2);

r2_h4 = 1 - ss_res/ss_tot;

%% =====================================================

%  DISPLAY VALIDATION RESULTS

%% =====================================================

fprintf('\n===== VALIDATION (ELMAN) =====\n');

fprintf('\nh1:\nRMSE = %.4f | MAE = %.4f | R2 = %.4f\n',rmse_h1,mae_h1,r2_h1);

fprintf('\nh4:\nRMSE = %.4f | MAE = %.4f | R2 = %.4f\n',rmse_h4,mae_h4,r2_h4);
