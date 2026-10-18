clear
close all
clc
rng(42,'twister');
%% ==========================================================
% LSTM LOCAL STABILITY ANALYSIS
% ==========================================================

load net
load C
load S
load dataset_validation.mat

dataset_validation_n = normalize(dataset, ...
    "center", C, "scale", S);

XValidation = dataset_validation_n(:,1:2)';
TValidation = dataset_validation_n(:,3:4)';

%% ----------------------------------------------------------
% LSTM dimensions
% ----------------------------------------------------------

N_hidden = 150;
N_state  = 2*N_hidden;

fprintf('\n=============================================\n');
fprintf('       LSTM LOCAL STABILITY ANALYSIS\n');
fprintf('=============================================\n');

fprintf('Hidden units       = %d\n',N_hidden);
fprintf('State dimension    = %d\n',N_state);

%% ----------------------------------------------------------
% Extract trained parameters
% ----------------------------------------------------------

W  = extractdata(net.Learnables.Value{1});
R  = extractdata(net.Learnables.Value{2});
B  = extractdata(net.Learnables.Value{3});

W  = double(W);
R  = double(R);
B  = double(B);

% MATLAB LSTM gate order:
% Input gate, Forget gate, Cell candidate, Output gate

Wi = W(1:N_hidden,:);
Wf = W(N_hidden+1:2*N_hidden,:);
Wc = W(2*N_hidden+1:3*N_hidden,:);
Wo = W(3*N_hidden+1:4*N_hidden,:);

Ri = R(1:N_hidden,:);
Rf = R(N_hidden+1:2*N_hidden,:);
Rc = R(2*N_hidden+1:3*N_hidden,:);
Ro = R(3*N_hidden+1:4*N_hidden,:);

bi = B(1:N_hidden);
bf = B(N_hidden+1:2*N_hidden);
bc = B(2*N_hidden+1:3*N_hidden);
bo = B(3*N_hidden+1:4*N_hidden);

fprintf('\nWeight dimensions:\n');
fprintf('Input weights     = %d x %d\n',size(W,1),size(W,2));
fprintf('Recurrent weights = %d x %d\n',size(R,1),size(R,2));
fprintf('Bias              = %d x %d\n',size(B,1),size(B,2));

%% ----------------------------------------------------------
% Numerical Jacobian
% State = [h; c]
% ----------------------------------------------------------

T_len = size(XValidation,2);

rho_values = zeros(T_len,1);

h = zeros(N_hidden,1);
c = zeros(N_hidden,1);

eps_fd = 1e-6;

for t = 1:T_len

    x = XValidation(:,t);

    % -------------------------------------------------------
    % Current state
    % -------------------------------------------------------

    state = [h; c];

    % -------------------------------------------------------
    % Exact LSTM state transition
    % -------------------------------------------------------

    h0 = state(1:N_hidden);
    c0 = state(N_hidden+1:end);

    ai = Wi*x + Ri*h0 + bi;
    af = Wf*x + Rf*h0 + bf;
    ac = Wc*x + Rc*h0 + bc;
    ao = Wo*x + Ro*h0 + bo;

    i = 1./(1+exp(-ai));
    f = 1./(1+exp(-af));
    g = tanh(ac);
    o = 1./(1+exp(-ao));

    c_new = f.*c0 + i.*g;
    h_new = o.*tanh(c_new);

    %% ------------------------------------------------------
    % Numerical Jacobian with respect to [h;c]
    % ------------------------------------------------------

    A = zeros(N_state,N_state);

    for j = 1:N_state

        state_p = state;
        state_m = state;

        state_p(j) = state_p(j) + eps_fd;
        state_m(j) = state_m(j) - eps_fd;

        hp = state_p(1:N_hidden);
        cp = state_p(N_hidden+1:end);

        hm = state_m(1:N_hidden);
        cm = state_m(N_hidden+1:end);

        % Positive perturbation
        aip = Wi*x + Ri*hp + bi;
        afp = Wf*x + Rf*hp + bf;
        acp = Wc*x + Rc*hp + bc;
        aop = Wo*x + Ro*hp + bo;

        ip = 1./(1+exp(-aip));
        fp = 1./(1+exp(-afp));
        gp = tanh(acp);
        op = 1./(1+exp(-aop));

        cp_new = fp.*cp + ip.*gp;
        hp_new = op.*tanh(cp_new);

        % Negative perturbation
        aim = Wi*x + Ri*hm + bi;
        afm = Wf*x + Rf*hm + bf;
        acm = Wc*x + Rc*hm + bc;
        aom = Wo*x + Ro*hm + bo;

        im = 1./(1+exp(-aim));
        fm = 1./(1+exp(-afm));
        gm = tanh(acm);
        om = 1./(1+exp(-aom));

        cm_new = fm.*cm + im.*gm;
        hm_new = om.*tanh(cm_new);

        state_p_new = [hp_new; cp_new];
        state_m_new = [hm_new; cm_new];

        A(:,j) = (state_p_new-state_m_new)/(2*eps_fd);
    end

    eigA = eig(A);
    rho_values(t) = max(abs(eigA));

    % Update state
    h = h_new;
    c = c_new;
end

%% ----------------------------------------------------------
% Remove initial transient
% ----------------------------------------------------------

washout = 50;

rho_eff = rho_values(washout+1:end);

%% ----------------------------------------------------------
% Statistics
% ----------------------------------------------------------

rho_mean   = mean(rho_eff);
rho_std    = std(rho_eff);
rho_median = median(rho_eff);
rho_95     = prctile(rho_eff,95);
rho_max    = max(rho_eff);
rho_min    = min(rho_eff);

pct_stable = 100*mean(rho_eff < 1);
pct_unstable = 100*mean(rho_eff >= 1);

%% ----------------------------------------------------------
% Results
% ----------------------------------------------------------

fprintf('\n---------------------------------------------\n');
fprintf('LOCAL JACOBIAN METRICS\n');
fprintf('---------------------------------------------\n');

fprintf('Mean rho(A)         = %.6f\n',rho_mean);
fprintf('Std rho(A)          = %.6f\n',rho_std);
fprintf('Median rho(A)       = %.6f\n',rho_median);
fprintf('95th percentile     = %.6f\n',rho_95);
fprintf('Maximum rho(A)      = %.6f\n',rho_max);
fprintf('Minimum rho(A)      = %.6f\n',rho_min);

fprintf('\n---------------------------------------------\n');
fprintf('STABILITY OCCURRENCE\n');
fprintf('---------------------------------------------\n');

fprintf('rho(A) < 1          = %.2f %%\n',pct_stable);
fprintf('rho(A) >= 1         = %.2f %%\n',pct_unstable);

fprintf('\n=============================================\n');
