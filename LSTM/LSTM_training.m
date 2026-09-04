clear
close all
clc

% Set seed for reproducibility
rng(42,'twister');

load dataset_entreno.mat % Training dataset
[datasetn,C,S]=normalize(dataset);
save C C; % Save normalization center
save S S; % Save normalization scale

% Input and output data are stored as column vectors.
% Each column corresponds to one input or output variable.

XTrain=[datasetn(:,1) datasetn(:,2);]; % Inputs: p (pump) and v (valve)
TTrain=[datasetn(:,3) datasetn(:,4)]; % Outputs: h1 and h4 (tank levels)


load dataset_validation.mat % Validation dataset
dataset_validation_n = normalize(dataset, "center", C, "scale", S);

XValidation=[dataset_validation_n(:,1) dataset_validation_n(:,2);]; % Inputs: p (pump) and v (valve)
TValidation=[dataset_validation_n(:,3) dataset_validation_n(:,4)]; % Outputs: h1 and h4 (tank levels)


% Define the network architecture
numFeatures=size(XTrain,2);
numResponses = size(TTrain,2);
numHiddenUnits = 150;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,OutputMode="sequence")
    batchNormalizationLayer  % Batch normalization
    fullyConnectedLayer(numResponses)];


% Specify training options
maxEpochs = 1200;
miniBatchSize = 12;

options = trainingOptions("adam", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=0.001, ...
    GradientThreshold=5, ...
    Shuffle="never", ...    
    Metrics="rmse", ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=10, ...
    OutputNetwork="best-validation", ...
    Plots="training-progress", ...
    Verbose=0);


% Train the network
[net,info] = trainnet(XTrain,TTrain,layers,"mse",options);

save net net


%% ================= VALIDATION =================

YPredVal = minibatchpredict(net,XValidation);

% Convert predictions to a matrix
YPredVal = gather(extractdata(YPredVal));

% Compute prediction errors
err = TValidation - YPredVal;


%% h1

mse_h1_val   = mean(err(:,1).^2);
rmse_h1_val  = sqrt(mse_h1_val);
mae_h1_val   = mean(abs(err(:,1)));
medae_h1_val = median(abs(err(:,1)));

DA_h1_val = mean(sign(diff(TValidation(:,1))) == sign(diff(YPredVal(:,1))))*100;

SSres = sum((TValidation(:,1)-YPredVal(:,1)).^2);
SStot = sum((TValidation(:,1)-mean(TValidation(:,1))).^2);
R2_h1_val = 1 - SSres/SStot;


%% h4

mse_h4_val   = mean(err(:,2).^2);
rmse_h4_val  = sqrt(mse_h4_val);
mae_h4_val   = mean(abs(err(:,2)));
medae_h4_val = median(abs(err(:,2)));

DA_h4_val = mean(sign(diff(TValidation(:,2))) == sign(diff(YPredVal(:,2))))*100;

SSres = sum((TValidation(:,2)-YPredVal(:,2)).^2);
SStot = sum((TValidation(:,2)-mean(TValidation(:,2))).^2);
R2_h4_val = 1 - SSres/SStot;


%% Display results

fprintf('\n===== VALIDATION (LSTM) =====\n\n');

fprintf('h1:\n');
fprintf('MSE   = %.4f\n',mse_h1_val);
fprintf('RMSE  = %.4f\n',rmse_h1_val);
fprintf('MAE   = %.4f\n',mae_h1_val);
fprintf('MedAE = %.4f\n',medae_h1_val);
fprintf('DA    = %.2f %%\n',DA_h1_val);
fprintf('R2    = %.4f\n\n',R2_h1_val);

fprintf('h4:\n');
fprintf('MSE   = %.4f\n',mse_h4_val);
fprintf('RMSE  = %.4f\n',rmse_h4_val);
fprintf('MAE   = %.4f\n',mae_h4_val);
fprintf('MedAE = %.4f\n',medae_h4_val);
fprintf('DA    = %.2f %%\n',DA_h4_val);
fprintf('R2    = %.4f\n',R2_h4_val);
