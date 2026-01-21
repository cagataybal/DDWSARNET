% =========================================================================
% DDWSARNET: Deep Layer-Wise Dynamic-Tau WSAR Neural Network
% Official Benchmark Implementation
%
% This script reproduces the experimental comparison between:
%   1. DDWSARNET (Layer-Wise)
%   2. WSAR (Monolithic)
%   3. PSO (Particle Swarm)
%   4. BP (Backpropagation / Adam)
%   5. CNN (Convolutional Baseline)
%
% Paper: "DDWSARNET: A Layer-Wise Dynamic-Tau WSAR Framework..."
% Copyright (c) 2026 Cagatay Bal, PhD.
% Licensed under the MIT License.
% =========================================================================

clear; clc; close all;
addpath('algorithms');
addpath('utils');

%% 1. Configuration & Parameters
core_seed   = 32; 
rng(core_seed);

N           = 1000;
featNoise   = 0.30;
flipRate    = 0.30;
trainRatio  = 0.70;

% MLP Architecture
hidden_sizes = [16 16]; 
actHidden    = 'relu';
K            = 2;
LBUB         = 5;       % Weight bounds [-5, 5]

%% 2. Data Generation (Two-Moons with Noise)
fprintf('Generating Data (Noise=%.2f, Flip=%.2f)...\n', featNoise, flipRate);
[X, y_true] = make_moons(N, featNoise);

% Label Flipping (Aleatoric Uncertainty)
flipIdx = rand(N, 1) < flipRate;
y = y_true;
y(flipIdx) = 1 - y(flipIdx);

% Normalization (Z-Score)
mu = mean(X, 1); 
sd = std(X, [], 1) + 1e-12;
Xz = (X - mu) ./ sd;

% Train/Test Split
idx  = randperm(N);
ntr  = round(trainRatio * N);
trId = idx(1:ntr); 
teId = idx(ntr+1:end);

Xtr = Xz(trId, :); ytr = y(trId);
Xte = Xz(teId, :); yte = y(teId);

%% 3. Setup Problem Dimensions
in_dim   = size(Xtr, 2);
dim_list = [in_dim, hidden_sizes, K];
S = layer_slices(dim_list);
n_params = S.n_params;
LB = -LBUB * ones(1, n_params);
UB =  LBUB * ones(1, n_params);

% Objective Function Helper
obj_full = @(theta) nn_ce_wrapper(theta, Xtr, ytr, dim_list, actHidden);

%% 4. Algorithm 1: Monolithic WSAR (Baseline)
fprintf('Running Monolithic WSAR...\n');
opts_mono = wsarset('PopulationSize', 100, 'MaxIterations', 1000, ...
                    'Tao', -0.8, 'TaoMode', 'fixed', 'RandomSeed', core_seed);
[x_mono, f_mono, ~, out_mono] = wsar(obj_full, n_params, LB, UB, opts_mono);

%% 5. Algorithm 2: DDWSARNET (Layer-Wise, Proposed)
fprintf('Running DDWSARNET (Layer-Wise)...\n');
optsLW = struct();
optsLW.PopSize         = 100;
optsLW.MaxIterPerBlock = 100; % Short bursts per block
optsLW.Cycles          = 7;
optsLW.EliteK          = 8;
optsLW.Display         = 'none';

theta0 = LB + 0.1*(UB-LB).*rand(size(LB)); % Mild random init
[theta_LW, outLW] = wsar_layerwise(obj_full, theta0, dim_list, LB, UB, optsLW);

%% 6. Algorithm 3: PSO (Stochastic Baseline)
fprintf('Running PSO...\n');
rng(core_seed + 1);
opts_pso = optimoptions('particleswarm', 'SwarmSize', 100, ...
                        'MaxIterations', 1000, 'Display', 'none');
[theta_pso, fval_pso] = particleswarm(obj_full, n_params, LB, UB, opts_pso);

%% 7. Algorithm 4: Backpropagation (Gradient Baseline)
fprintf('Running Backpropagation (PatternNet)...\n');
% Note: PatternNet uses its own scaling/structure
[X_all_mm, ps_in] = mapminmax(Xz', -1, 1);
T_all = full(ind2vec(y' + 1, K));
net_bp = patternnet(hidden_sizes);
net_bp.trainParam.showWindow = 0;
net_bp.trainParam.epochs = 1000;
% Manual Indices
trMask = false(1,N); trMask(trId)=true;
teMask = false(1,N); teMask(teId)=true;
net_bp.divideFcn = 'divideind';
net_bp.divideParam.trainInd = find(trMask);
net_bp.divideParam.testInd  = find(teMask);
net_bp.divideParam.valInd   = [];
[net_bp, tr_bp] = train(net_bp, X_all_mm, T_all);

%% 8. Algorithm 5: CNN (Inductive Bias Baseline)
fprintf('Running CNN (Image Projection)...\n');
imgSize = 32; sigma = 1.2;
Ximg_all = zeros(imgSize, imgSize, 1, N, 'single');
for i = 1:N, Ximg_all(:,:,:,i) = point2img(Xz(i,:), imgSize, sigma); end
Ycat = categorical(y);
layers = [imageInputLayer([imgSize imgSize 1]), convolution2dLayer(3,8,'Padding','same'), ...
          reluLayer, maxPooling2dLayer(2,'Stride',2), fullyConnectedLayer(K), softmaxLayer, classificationLayer];
opts_cnn = trainingOptions('adam', 'MaxEpochs', 12, 'Verbose', false, 'Shuffle','every-epoch');
[net_cnn, info_cnn] = trainNetwork(Ximg_all(:,:,:,trId), Ycat(trId), layers, opts_cnn);

%% 9. Evaluation & Reporting
fprintf('\n--- Performance Evaluation ---\n');

% Helper to compute stats
eval_model = @(th, name) evaluate_metrics(th, Xte, yte, dim_list, actHidden, name);

res_mono = eval_model(x_mono, 'Monolithic WSAR');
res_lw   = eval_model(theta_LW, 'DDWSARNET');
res_pso  = eval_model(theta_pso, 'PSO');

% BP Eval
P_bp = net_bp(mapminmax('apply', Xte', ps_in))';
ce_bp = nn_loss(P_bp, yte, K);
fprintf('BP (PatternNet)   | CE: %.4f\n', ce_bp);

% CNN Eval
[~, scores_cnn] = classify(net_cnn, Ximg_all(:,:,:,teId));
ce_cnn = nn_loss(scores_cnn, yte, K);
fprintf('CNN (Adam)        | CE: %.4f\n', ce_cnn);

% Plot Convergence
figure('Name','Convergence Comparison');
plot(out_mono.besthistory, 'b-', 'LineWidth', 1.5); hold on;
plot(outLW.train_history, 'r-', 'LineWidth', 1.5);
legend('Monolithic WSAR', 'DDWSARNET (Layer-Wise)');
xlabel('Evaluations / Progress'); ylabel('Cross-Entropy Loss');
title('Optimization Landscape Trajectory');
grid on;

%% Local Helper for Wrapper
function ce = nn_ce_wrapper(theta, X, y, dim_list, act)
    [~, P] = forward_mlp(theta, X, dim_list, act);
    ce = nn_loss(P, y, dim_list(end));
end

function res = evaluate_metrics(theta, X, y, dim_list, act, name)
    [~, P] = forward_mlp(theta, X, dim_list, act);
    [ce, mse] = nn_loss(P, y, dim_list(end));
    [~, yhat] = max(P, [], 2); yhat = yhat - 1;
    acc = mean(yhat == y) * 100;
    fprintf('%-18s | Acc: %.2f%% | CE: %.4f | MSE: %.4f\n', name, acc, ce, mse);
    res.ce = ce; res.acc = acc;
end