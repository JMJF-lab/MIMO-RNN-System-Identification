clear
clc
rng(1,'twister')

%% ==========================================================
% LSTM - 10 SEEDS (Orden: 0, 1, 2, 7, 13, 27, 42, 73, 101, 999)
%% ==========================================================
RMSE_h1_LSTM = [0.4586 0.4807 0.3885 0.4339 0.4697 0.4191 0.3936 0.4175 0.4865 0.4542];
RMSE_h4_LSTM = [0.4163 0.4308 0.3991 0.4381 0.4083 0.4367 0.3329 0.4637 0.4228 0.3883];

MAE_h1_LSTM = [0.3307 0.3640 0.2866 0.3175 0.3482 0.3067 0.2842 0.3152 0.3621 0.3324];
MAE_h4_LSTM = [0.3181 0.3322 0.3072 0.3325 0.3201 0.3454 0.2648 0.3578 0.3262 0.2976];

MedAE_h1_LSTM = [0.2449 0.2923 0.2199 0.2357 0.2722 0.2259 0.2119 0.2524 0.2814 0.2422];
MedAE_h4_LSTM = [0.2449 0.2519 0.2416 0.2570 0.2643 0.2981 0.2208 0.2743 0.2731 0.2381];

R2_h1_LSTM = [0.9995 0.9994 0.9996 0.9995 0.9994 0.9996 0.9996 0.9996 0.9994 0.9995];
R2_h4_LSTM = [0.9994 0.9994 0.9995 0.9994 0.9994 0.9994 0.9996 0.9993 0.9994 0.9995];

DA_h1_LSTM = [98.70 98.60 98.55 98.50 98.75 98.65 98.10 98.40 98.25 98.25];
DA_h4_LSTM = [98.05 97.40 97.25 97.60 97.50 97.70 97.25 97.45 97.65 97.65];

FFT_h1_LSTM = [0.006509 0.006722 0.005372 0.006083 0.007259 0.006058 0.005297 0.005758 0.006947 0.006748];
FFT_h4_LSTM = [0.006755 0.006910 0.006405 0.007275 0.007094 0.007341 0.005412 0.007748 0.006511 0.006254];

% Bootstrap CI (95%)
CI_RMSE_h1_LSTM = bootci(10000,@mean,RMSE_h1_LSTM);
CI_RMSE_h4_LSTM = bootci(10000,@mean,RMSE_h4_LSTM);
CI_MAE_h1_LSTM = bootci(10000,@mean,MAE_h1_LSTM);
CI_MAE_h4_LSTM = bootci(10000,@mean,MAE_h4_LSTM);
CI_MedAE_h1_LSTM = bootci(10000,@mean,MedAE_h1_LSTM);
CI_MedAE_h4_LSTM = bootci(10000,@mean,MedAE_h4_LSTM);
CI_R2_h1_LSTM = bootci(10000,@mean,R2_h1_LSTM);
CI_R2_h4_LSTM = bootci(10000,@mean,R2_h4_LSTM);
CI_DA_h1_LSTM = bootci(10000,@mean,DA_h1_LSTM);
CI_DA_h4_LSTM = bootci(10000,@mean,DA_h4_LSTM);
CI_FFT_h1_LSTM = bootci(10000,@mean,FFT_h1_LSTM);
CI_FFT_h4_LSTM = bootci(10000,@mean,FFT_h4_LSTM);

% Display
fprintf('\n================= LSTM BOOT_CI =================\n\n');
fprintf('RMSE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(RMSE_h1_LSTM),std(RMSE_h1_LSTM),CI_RMSE_h1_LSTM(1),CI_RMSE_h1_LSTM(2));
fprintf('RMSE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(RMSE_h4_LSTM),std(RMSE_h4_LSTM),CI_RMSE_h4_LSTM(1),CI_RMSE_h4_LSTM(2));
fprintf('MAE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(MAE_h1_LSTM),std(MAE_h1_LSTM),CI_MAE_h1_LSTM(1),CI_MAE_h1_LSTM(2));
fprintf('MAE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(MAE_h4_LSTM),std(MAE_h4_LSTM),CI_MAE_h4_LSTM(1),CI_MAE_h4_LSTM(2));
fprintf('MedAE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(MedAE_h1_LSTM),std(MedAE_h1_LSTM),CI_MedAE_h1_LSTM(1),CI_MedAE_h1_LSTM(2));
fprintf('MedAE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(MedAE_h4_LSTM),std(MedAE_h4_LSTM),CI_MedAE_h4_LSTM(1),CI_MedAE_h4_LSTM(2));
fprintf('R2 h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(R2_h1_LSTM),std(R2_h1_LSTM),CI_R2_h1_LSTM(1),CI_R2_h1_LSTM(2));
fprintf('R2 h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(R2_h4_LSTM),std(R2_h4_LSTM),CI_R2_h4_LSTM(1),CI_R2_h4_LSTM(2));
fprintf('DA h1 = %.2f ± %.2f CI=[%.2f %.2f]\n', mean(DA_h1_LSTM),std(DA_h1_LSTM),CI_DA_h1_LSTM(1),CI_DA_h1_LSTM(2));
fprintf('DA h4 = %.2f ± %.2f CI=[%.2f %.2f]\n\n', mean(DA_h4_LSTM),std(DA_h4_LSTM),CI_DA_h4_LSTM(1),CI_DA_h4_LSTM(2));
fprintf('FFT h1 = %.6f ± %.6f CI=[%.6f %.6f]\n', mean(FFT_h1_LSTM),std(FFT_h1_LSTM),CI_FFT_h1_LSTM(1),CI_FFT_h1_LSTM(2));
fprintf('FFT h4 = %.6f ± %.6f CI=[%.6f %.6f]\n', mean(FFT_h4_LSTM),std(FFT_h4_LSTM),CI_FFT_h4_LSTM(1),CI_FFT_h4_LSTM(2));

%% ==========================================================
% ELMAN - 10 SEEDS (Orden: 0, 1, 2, 7, 13, 27, 42, 73, 101, 999)
%% ==========================================================
RMSE_h1_ELMAN = [0.0707 0.0975 0.0363 0.1012 0.0567 2.3418 0.1083 0.0836 0.1168 0.1090];
RMSE_h4_ELMAN = [0.0629 0.0791 0.1336 0.0484 0.0523 0.2523 0.0637 0.0702 0.2205 0.0906];

MAE_h1_ELMAN = [0.0376 0.0547 0.0211 0.0506 0.0288 0.5688 0.0613 0.0396 0.0409 0.0537];
MAE_h4_ELMAN = [0.0329 0.0499 0.0528 0.0283 0.0266 0.0719 0.0333 0.0291 0.0620 0.0285];

MedAE_h1_ELMAN = [0.0198 0.0314 0.0147 0.0228 0.0127 0.0268 0.0275 0.0180 0.0146 0.0199];
MedAE_h4_ELMAN = [0.0186 0.0296 0.0143 0.0188 0.0132 0.0197 0.0200 0.0153 0.0170 0.0139];

R2_h1_ELMAN = [1.0000 1.0000 1.0000 1.0000 1.0000 0.9860 1.0000 1.0000 1.0000 1.0000];
R2_h4_ELMAN = [1.0000 1.0000 0.9999 1.0000 1.0000 0.9998 1.0000 1.0000 0.9998 1.0000];

DA_h1_ELMAN = [99.60 99.60 99.70 99.65 99.65 99.50 99.60 99.65 99.60 99.70];
DA_h4_ELMAN = [99.15 99.30 99.35 99.25 99.25 99.10 99.30 99.20 99.25 99.10];

FFT_h1_ELMAN = [0.001060 0.001312 0.000493 0.001378 0.000910 0.034973 0.001368 0.001286 0.001668 0.001586];
FFT_h4_ELMAN = [0.000890 0.001102 0.002002 0.000687 0.000727 0.003817 0.000885 0.001049 0.003330 0.001331];

% Bootstrap CI (95%)
CI_RMSE_h1_ELMAN = bootci(10000,@mean,RMSE_h1_ELMAN);
CI_RMSE_h4_ELMAN = bootci(10000,@mean,RMSE_h4_ELMAN);
CI_MAE_h1_ELMAN  = bootci(10000,@mean,MAE_h1_ELMAN);
CI_MAE_h4_ELMAN  = bootci(10000,@mean,MAE_h4_ELMAN);
CI_MedAE_h1_ELMAN= bootci(10000,@mean,MedAE_h1_ELMAN);
CI_MedAE_h4_ELMAN= bootci(10000,@mean,MedAE_h4_ELMAN);
CI_R2_h1_ELMAN   = bootci(10000,@mean,R2_h1_ELMAN);
CI_R2_h4_ELMAN   = bootci(10000,@mean,R2_h4_ELMAN);
CI_DA_h1_ELMAN   = bootci(10000,@mean,DA_h1_ELMAN);
CI_DA_h4_ELMAN   = bootci(10000,@mean,DA_h4_ELMAN);
CI_FFT_h1_ELMAN  = bootci(10000,@mean,FFT_h1_ELMAN);
CI_FFT_h4_ELMAN  = bootci(10000,@mean,FFT_h4_ELMAN);

% Display
fprintf('\n================= ELMAN_BOOTSTRAP CI =================\n\n');
fprintf('RMSE h1   = %.4f ± %.4f   CI=[%.4f %.4f]\n',mean(RMSE_h1_ELMAN),std(RMSE_h1_ELMAN),CI_RMSE_h1_ELMAN(1),CI_RMSE_h1_ELMAN(2));
fprintf('RMSE h4   = %.4f ± %.4f   CI=[%.4f %.4f]\n\n',mean(RMSE_h4_ELMAN),std(RMSE_h4_ELMAN),CI_RMSE_h4_ELMAN(1),CI_RMSE_h4_ELMAN(2));
fprintf('MAE h1    = %.4f ± %.4f   CI=[%.4f %.4f]\n',mean(MAE_h1_ELMAN),std(MAE_h1_ELMAN),CI_MAE_h1_ELMAN(1),CI_MAE_h1_ELMAN(2));
fprintf('MAE h4    = %.4f ± %.4f   CI=[%.4f %.4f]\n\n',mean(MAE_h4_ELMAN),std(MAE_h4_ELMAN),CI_MAE_h4_ELMAN(1),CI_MAE_h4_ELMAN(2));
fprintf('MedAE h1  = %.4f ± %.4f   CI=[%.4f %.4f]\n',mean(MedAE_h1_ELMAN),std(MedAE_h1_ELMAN),CI_MedAE_h1_ELMAN(1),CI_MedAE_h1_ELMAN(2));
fprintf('MedAE h4  = %.4f ± %.4f   CI=[%.4f %.4f]\n\n',mean(MedAE_h4_ELMAN),std(MedAE_h4_ELMAN),CI_MedAE_h4_ELMAN(1),CI_MedAE_h4_ELMAN(2));
fprintf('R2 h1     = %.4f ± %.4f   CI=[%.4f %.4f]\n',mean(R2_h1_ELMAN),std(R2_h1_ELMAN),CI_R2_h1_ELMAN(1),CI_R2_h1_ELMAN(2));
fprintf('R2 h4     = %.4f ± %.4f   CI=[%.4f %.4f]\n\n',mean(R2_h4_ELMAN),std(R2_h4_ELMAN),CI_R2_h4_ELMAN(1),CI_R2_h4_ELMAN(2));
fprintf('DA h1     = %.2f ± %.2f   CI=[%.2f %.2f]\n',mean(DA_h1_ELMAN),std(DA_h1_ELMAN),CI_DA_h1_ELMAN(1),CI_DA_h1_ELMAN(2));
fprintf('DA h4     = %.2f ± %.2f   CI=[%.2f %.2f]\n\n',mean(DA_h4_ELMAN),std(DA_h4_ELMAN),CI_DA_h4_ELMAN(1),CI_DA_h4_ELMAN(2));
fprintf('FFT h1    = %.6f ± %.6f   CI=[%.6f %.6f]\n',mean(FFT_h1_ELMAN),std(FFT_h1_ELMAN),CI_FFT_h1_ELMAN(1),CI_FFT_h1_ELMAN(2));
fprintf('FFT h4    = %.6f ± %.6f   CI=[%.6f %.6f]\n',mean(FFT_h4_ELMAN),std(FFT_h4_ELMAN),CI_FFT_h4_ELMAN(1),CI_FFT_h4_ELMAN(2));

%% ==========================================================
% GRU - 10 SEEDS (Orden: 0, 1, 2, 7, 13, 27, 42, 73, 101, 999)
%% ==========================================================
RMSE_h1_GRU = [0.6835 0.6106 0.6383 0.5004 0.7292 0.6594 0.7011 0.4672 0.7867 0.6549];
RMSE_h4_GRU = [0.5420 0.5212 0.5053 0.6398 0.6881 0.7302 0.7057 0.5177 0.5867 0.5498];

MAE_h1_GRU = [0.5129 0.5087 0.4524 0.4903 0.3618 0.5370 0.5015 0.3499 0.6009 0.4868];
MAE_h4_GRU = [0.5606 0.3990 0.4018 0.3996 0.5003 0.5030 0.5898 0.3917 0.4646 0.4060];

MedAE_h1_GRU = [0.3802 0.3686 0.3524 0.3904 0.2670 0.4233 0.4025 0.2619 0.4659 0.3687];
MedAE_h4_GRU = [0.4818 0.2939 0.3143 0.3329 0.4097 0.3780 0.5170 0.2972 0.3879 0.3040];

R2_h1_GRU = [0.9987 0.9988 0.9991 0.9990 0.9994 0.9986 0.9989 0.9994 0.9984 0.9989];
R2_h4_GRU = [0.9984 0.9990 0.9991 0.9992 0.9986 0.9984 0.9982 0.9991 0.9989 0.9990];

DA_h1_GRU = [98.30 98.40 98.70 98.65 98.25 98.20 98.35 98.60 98.55 97.80];
DA_h4_GRU = [98.05 97.75 98.05 97.95 97.90 97.70 97.90 97.80 97.80 97.40];

FFT_h1_GRU = [0.011394 0.010729 0.008569 0.009241 0.006367 0.010293 0.009079 0.005300 0.013149 0.009803];
FFT_h4_GRU = [0.010758 0.007356 0.007267 0.007133 0.007998 0.010572 0.011582 0.007354 0.008988 0.007159];

% Bootstrap CI (95%)
CI_RMSE_h1_GRU   = bootci(10000,@mean,RMSE_h1_GRU);
CI_RMSE_h4_GRU   = bootci(10000,@mean,RMSE_h4_GRU);
CI_MAE_h1_GRU    = bootci(10000,@mean,MAE_h1_GRU);
CI_MAE_h4_GRU    = bootci(10000,@mean,MAE_h4_GRU);
CI_MedAE_h1_GRU  = bootci(10000,@mean,MedAE_h1_GRU);
CI_MedAE_h4_GRU  = bootci(10000,@mean,MedAE_h4_GRU);
CI_R2_h1_GRU     = bootci(10000,@mean,R2_h1_GRU);
CI_R2_h4_GRU     = bootci(10000,@mean,R2_h4_GRU);
CI_DA_h1_GRU     = bootci(10000,@mean,DA_h1_GRU);
CI_DA_h4_GRU     = bootci(10000,@mean,DA_h4_GRU);
CI_FFT_h1_GRU    = bootci(10000,@mean,FFT_h1_GRU);
CI_FFT_h4_GRU    = bootci(10000,@mean,FFT_h4_GRU);

% Display
fprintf('\n================= GRU_BOOTSTRAP CI =================\n\n');
fprintf('RMSE h1   = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(RMSE_h1_GRU),std(RMSE_h1_GRU),CI_RMSE_h1_GRU(1),CI_RMSE_h1_GRU(2));
fprintf('RMSE h4   = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(RMSE_h4_GRU),std(RMSE_h4_GRU),CI_RMSE_h4_GRU(1),CI_RMSE_h4_GRU(2));
fprintf('MAE h1    = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(MAE_h1_GRU),std(MAE_h1_GRU),CI_MAE_h1_GRU(1),CI_MAE_h1_GRU(2));
fprintf('MAE h4    = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(MAE_h4_GRU),std(MAE_h4_GRU),CI_MAE_h4_GRU(1),CI_MAE_h4_GRU(2));
fprintf('MedAE h1  = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(MedAE_h1_GRU),std(MedAE_h1_GRU),CI_MedAE_h1_GRU(1),CI_MedAE_h1_GRU(2));
fprintf('MedAE h4  = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(MedAE_h4_GRU),std(MedAE_h4_GRU),CI_MedAE_h4_GRU(1),CI_MedAE_h4_GRU(2));
fprintf('R2 h1     = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(R2_h1_GRU),std(R2_h1_GRU),CI_R2_h1_GRU(1),CI_R2_h1_GRU(2));
fprintf('R2 h4     = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(R2_h4_GRU),std(R2_h4_GRU),CI_R2_h4_GRU(1),CI_R2_h4_GRU(2));
fprintf('DA h1     = %.2f ± %.2f   CI=[%.2f %.2f]\n', mean(DA_h1_GRU),std(DA_h1_GRU),CI_DA_h1_GRU(1),CI_DA_h1_GRU(2));
fprintf('DA h4     = %.2f ± %.2f   CI=[%.2f %.2f]\n\n', mean(DA_h4_GRU),std(DA_h4_GRU),CI_DA_h4_GRU(1),CI_DA_h4_GRU(2));
fprintf('FFT h1    = %.6f ± %.6f   CI=[%.6f %.6f]\n', mean(FFT_h1_GRU),std(FFT_h1_GRU),CI_FFT_h1_GRU(1),CI_FFT_h1_GRU(2));
fprintf('FFT h4    = %.6f ± %.6f   CI=[%.6f %.6f]\n', mean(FFT_h4_GRU),std(FFT_h4_GRU),CI_FFT_h4_GRU(1),CI_FFT_h4_GRU(2));

%% ==========================================================
% NARX - 10 SEEDS (Orden: 0, 1, 2, 7, 13, 27, 42, 73, 101, 999)
%% ==========================================================
RMSE_h1_NARX = [0.0071 0.0082 0.0076 0.0073 0.0064 0.0146 0.0103 0.0071 0.0078 0.0074];
RMSE_h4_NARX = [0.0050 0.0045 0.0039 0.0059 0.0049 0.0082 0.0047 0.0041 0.0058 0.0082];

MAE_h1_NARX = [0.0017 0.0025 0.0020 0.0018 0.0020 0.0046 0.0030 0.0023 0.0022 0.0018];
MAE_h4_NARX = [0.0015 0.0020 0.0015 0.0018 0.0017 0.0039 0.0025 0.0018 0.0020 0.0016];

MedAE_h1_NARX = [0.0007 0.0011 0.0010 0.0008 0.0009 0.0023 0.0012 0.0011 0.0010 0.0007];
MedAE_h4_NARX = [0.0008 0.0012 0.0009 0.0010 0.0010 0.0024 0.0017 0.0011 0.0011 0.0008];

R2_h1_NARX = [1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000];
R2_h4_NARX = [1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000];

DA_h1_NARX = [100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00 100.00];
DA_h4_NARX = [99.75 99.85 99.80 99.75 99.75 99.80 99.75 99.75 99.80 99.65];

FFT_h1_NARX = [0.000097 0.000109 0.000107 0.000100 0.000087 0.000195 0.000137 0.000095 0.000106 0.000100];
FFT_h4_NARX = [0.000075 0.000066 0.000060 0.000091 0.000076 0.000118 0.000071 0.000058 0.000086 0.000129];

% Bootstrap CI (95%)
CI_RMSE_h1_NARX   = bootci(10000,@mean,RMSE_h1_NARX);
CI_RMSE_h4_NARX   = bootci(10000,@mean,RMSE_h4_NARX);
CI_MAE_h1_NARX    = bootci(10000,@mean,MAE_h1_NARX);
CI_MAE_h4_NARX    = bootci(10000,@mean,MAE_h4_NARX);
CI_MedAE_h1_NARX  = bootci(10000,@mean,MedAE_h1_NARX);
CI_MedAE_h4_NARX  = bootci(10000,@mean,MedAE_h4_NARX);
CI_R2_h1_NARX     = bootci(10000,@mean,R2_h1_NARX);
CI_R2_h4_NARX     = bootci(10000,@mean,R2_h4_NARX);
CI_DA_h1_NARX     = bootci(10000,@mean,DA_h1_NARX);
CI_DA_h4_NARX     = bootci(10000,@mean,DA_h4_NARX);
CI_FFT_h1_NARX    = bootci(10000,@mean,FFT_h1_NARX);
CI_FFT_h4_NARX    = bootci(10000,@mean,FFT_h4_NARX);

% Display
fprintf('\n================= NARX_BOOTSTRAP CI =================\n\n');
fprintf('RMSE h1   = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(RMSE_h1_NARX),std(RMSE_h1_NARX),CI_RMSE_h1_NARX(1),CI_RMSE_h1_NARX(2));
fprintf('RMSE h4   = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(RMSE_h4_NARX),std(RMSE_h4_NARX),CI_RMSE_h4_NARX(1),CI_RMSE_h4_NARX(2));
fprintf('MAE h1    = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(MAE_h1_NARX),std(MAE_h1_NARX),CI_MAE_h1_NARX(1),CI_MAE_h1_NARX(2));
fprintf('MAE h4    = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(MAE_h4_NARX),std(MAE_h4_NARX),CI_MAE_h4_NARX(1),CI_MAE_h4_NARX(2));
fprintf('MedAE h1  = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(MedAE_h1_NARX),std(MedAE_h1_NARX),CI_MedAE_h1_NARX(1),CI_MedAE_h1_NARX(2));
fprintf('MedAE h4  = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(MedAE_h4_NARX),std(MedAE_h4_NARX),CI_MedAE_h4_NARX(1),CI_MedAE_h4_NARX(2));
fprintf('R2 h1     = %.4f ± %.4f   CI=[%.4f %.4f]\n', mean(R2_h1_NARX),std(R2_h1_NARX),CI_R2_h1_NARX(1),CI_R2_h1_NARX(2));
fprintf('R2 h4     = %.4f ± %.4f   CI=[%.4f %.4f]\n\n', mean(R2_h4_NARX),std(R2_h4_NARX),CI_R2_h4_NARX(1),CI_R2_h4_NARX(2));
fprintf('DA h1     = %.2f ± %.2f   CI=[%.2f %.2f]\n', mean(DA_h1_NARX),std(DA_h1_NARX),CI_DA_h1_NARX(1),CI_DA_h1_NARX(2));
fprintf('DA h4     = %.2f ± %.2f   CI=[%.2f %.2f]\n\n', mean(DA_h4_NARX),std(DA_h4_NARX),CI_DA_h4_NARX(1),CI_DA_h4_NARX(2));
fprintf('FFT h1    = %.6f ± %.6f   CI=[%.6f %.6f]\n', mean(FFT_h1_NARX),std(FFT_h1_NARX),CI_FFT_h1_NARX(1),CI_FFT_h1_NARX(2));
fprintf('FFT h4    = %.6f ± %.6f   CI=[%.6f %.6f]\n', mean(FFT_h4_NARX),std(FFT_h4_NARX),CI_FFT_h4_NARX(1),CI_FFT_h4_NARX(2));

%% ==========================================================
% ESN - 10 SEEDS (Orden: 0, 1, 2, 7, 13, 27, 42, 73, 101, 999)
%% ==========================================================

RMSE_h1_ESN = [ ...
    1.279778 1.120149 0.847544 0.943610 1.201114 ...
    0.902882 0.792935 0.773719 0.995606 1.220001];

RMSE_h4_ESN = [ ...
    1.248378 1.083499 0.912250 0.945796 1.296590 ...
    1.025135 0.976853 0.941557 1.003312 1.215070];

MAE_h1_ESN = [ ...
    0.966812 0.853807 0.659877 0.746739 0.922690 ...
    0.705938 0.632346 0.608150 0.771324 0.927579];

MAE_h4_ESN = [ ...
    0.956620 0.828838 0.722695 0.757788 0.950006 ...
    0.822811 0.776031 0.738960 0.810032 0.928219];

MedAE_h1_ESN = [ ...
    0.752158 0.706335 0.515867 0.616102 0.756765 ...
    0.592602 0.538007 0.494890 0.636132 0.720840];

MedAE_h4_ESN = [ ...
    0.760270 0.685511 0.589688 0.652218 0.721268 ...
    0.710438 0.650929 0.610249 0.698073 0.722382];

R2_h1_ESN = [ ...
    0.995663 0.996677 0.998098 0.997642 0.996180 ...
    0.997841 0.998335 0.998415 0.997375 0.996059];

R2_h4_ESN = [ ...
    0.994444 0.995814 0.997033 0.996811 0.994006 ...
    0.996253 0.996598 0.996839 0.996411 0.994736];

DA_h1_ESN = [ ...
    95.69 96.62 96.72 96.67 96.10 ...
    96.92 96.98 96.67 96.62 95.85];

DA_h4_ESN = [ ...
    93.64 94.16 95.49 94.77 94.16 ...
    94.72 95.03 94.98 94.98 94.72];

FFT_h1_ESN = [ ...
    0.015101 0.014027 0.009712 0.011859 0.016707 ...
    0.011319 0.009921 0.009349 0.010471 0.017141];

FFT_h4_ESN = [ ...
    0.016884 0.014648 0.011995 0.012482 0.018141 ...
    0.013678 0.012264 0.011521 0.014559 0.017350];

% Bootstrap CI (95%)

CI_RMSE_h1_ESN = bootci(10000,@mean,RMSE_h1_ESN);
CI_RMSE_h4_ESN = bootci(10000,@mean,RMSE_h4_ESN);

CI_MAE_h1_ESN = bootci(10000,@mean,MAE_h1_ESN);
CI_MAE_h4_ESN = bootci(10000,@mean,MAE_h4_ESN);

CI_MedAE_h1_ESN = bootci(10000,@mean,MedAE_h1_ESN);
CI_MedAE_h4_ESN = bootci(10000,@mean,MedAE_h4_ESN);

CI_R2_h1_ESN = bootci(10000,@mean,R2_h1_ESN);
CI_R2_h4_ESN = bootci(10000,@mean,R2_h4_ESN);

CI_DA_h1_ESN = bootci(10000,@mean,DA_h1_ESN);
CI_DA_h4_ESN = bootci(10000,@mean,DA_h4_ESN);

CI_FFT_h1_ESN = bootci(10000,@mean,FFT_h1_ESN);
CI_FFT_h4_ESN = bootci(10000,@mean,FFT_h4_ESN);

% Display
fprintf('\n================= ESN =================\n\n');
fprintf('RMSE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(RMSE_h1_ESN),std(RMSE_h1_ESN),CI_RMSE_h1_ESN(1),CI_RMSE_h1_ESN(2));
fprintf('RMSE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(RMSE_h4_ESN),std(RMSE_h4_ESN),CI_RMSE_h4_ESN(1),CI_RMSE_h4_ESN(2));
fprintf('MAE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(MAE_h1_ESN),std(MAE_h1_ESN),CI_MAE_h1_ESN(1),CI_MAE_h1_ESN(2));
fprintf('MAE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(MAE_h4_ESN),std(MAE_h4_ESN),CI_MAE_h4_ESN(1),CI_MAE_h4_ESN(2));
fprintf('MedAE h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(MedAE_h1_ESN),std(MedAE_h1_ESN),CI_MedAE_h1_ESN(1),CI_MedAE_h1_ESN(2));
fprintf('MedAE h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(MedAE_h4_ESN),std(MedAE_h4_ESN),CI_MedAE_h4_ESN(1),CI_MedAE_h4_ESN(2));
fprintf('R2 h1 = %.4f ± %.4f CI=[%.4f %.4f]\n', mean(R2_h1_ESN),std(R2_h1_ESN),CI_R2_h1_ESN(1),CI_R2_h1_ESN(2));
fprintf('R2 h4 = %.4f ± %.4f CI=[%.4f %.4f]\n\n', mean(R2_h4_ESN),std(R2_h4_ESN),CI_R2_h4_ESN(1),CI_R2_h4_ESN(2));
fprintf('DA h1 = %.2f ± %.2f CI=[%.2f %.2f]\n', mean(DA_h1_ESN),std(DA_h1_ESN),CI_DA_h1_ESN(1),CI_DA_h1_ESN(2));
fprintf('DA h4 = %.2f ± %.2f CI=[%.2f %.2f]\n\n', mean(DA_h4_ESN),std(DA_h4_ESN),CI_DA_h4_ESN(1),CI_DA_h4_ESN(2));
fprintf('FFT h1 = %.6f ± %.6f CI=[%.6f %.6f]\n', mean(FFT_h1_ESN),std(FFT_h1_ESN),CI_FFT_h1_ESN(1),CI_FFT_h1_ESN(2));
fprintf('FFT h4 = %.6f ± %.6f CI=[%.6f %.6f]\n', mean(FFT_h4_ESN),std(FFT_h4_ESN),CI_FFT_h4_ESN(1),CI_FFT_h4_ESN(2));
