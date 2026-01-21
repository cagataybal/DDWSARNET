function [theta_best, out] = wsar_layerwise(obj_full, theta0, dim_list, LB, UB, optsLW)
% WSAR_LAYERWISE Implements the Layer-Wise Block-Coordinate Optimization strategy.
%
%   DDWSARNET: Deep Layer-Wise Dynamic-Tau WSAR Neural Network
%   Copyright (c) 2026 Cagatay Bal, PhD.
%   Licensed under the MIT License.

    % Add utils to path if not present (handled by main usually)
    
    % 1. Decompose Parameter Vector into Blocks (Weights/Biases per layer)
    S = layer_slices(dim_list);
    blocks = {};
    for l = 1:S.L
        blocks{end+1} = S.W{l}; %#ok<AGROW>
        blocks{end+1} = S.b{l}; %#ok<AGROW>
    end
    B = numel(blocks);

    % 2. Parse Options
    if ~isfield(optsLW, 'PopSize'),     optsLW.PopSize = 100; end
    if ~isfield(optsLW, 'Cycles'),      optsLW.Cycles = 3; end
    if ~isfield(optsLW, 'EliteK'),      optsLW.EliteK = 10; end
    if ~isfield(optsLW, 'CoupleRatio'), optsLW.CoupleRatio = 0.35; end
    
    % Configure Inner WSAR (Dynamic Tau)
    wsar_opts = wsarset('PopulationSize', optsLW.PopSize, ...
                        'MaxIterations', optsLW.MaxIterPerBlock, ...
                        'Tao', -0.8, 'TaoMode', 'loss', 'Tao0', -0.8, ...
                        'AdaptWindow', 40, 'AdaptTol', 1e-4, ...
                        'Display', optsLW.Display);

    theta = theta0(:)';
    ELITE_BUF = zeros(0, numel(theta));
    train_hist = [];
    tau_hist   = [];

    % 3. Main Layer-Wise Loop
    for c = 1:optsLW.Cycles
        for b = 1:B
            slice = blocks{b};
            nvars = numel(slice);
            
            % Define Sub-Problem: Optimize Block 'b', freeze others
            obj_block = @(z_blk) obj_full( overwrite_slice(theta, slice, z_blk) );
            
            % Phase 1: Elite-Mix Generative Initialization
            if ~isempty(ELITE_BUF)
                X0 = init_from_elite(ELITE_BUF, slice, optsLW.PopSize, LB, UB, optsLW.CoupleRatio);
                wsar_opts.InitialPopulation = X0;
            else
                wsar_opts.InitialPopulation = [];
            end
            
            % Phase 2: Run WSAR on Block
            [zbest, ~, ~, outb] = wsar(obj_block, nvars, LB(slice), UB(slice), wsar_opts);
            
            % Update Global Vector
            theta(slice) = zbest;
            
            % Phase 3: Update Global Elite Archive
            Kkeep = min(optsLW.EliteK, size(outb.finalpopulation, 1));
            [~, ord] = sort(outb.finalscores, 'ascend');
            elites_blk = outb.finalpopulation(ord(1:Kkeep), :);
            ELITE_BUF = grow_elite_buffer(ELITE_BUF, theta, slice, elites_blk, Kkeep);
            
            % Log History
            train_hist = [train_hist; outb.besthistory(:)]; %#ok<AGROW>
            if isfield(outb, 'tao_history')
                tau_hist = [tau_hist; outb.tao_history(:)]; %#ok<AGROW>
            end
        end
    end
    
    theta_best = theta;
    out.train_history = train_hist;
    out.tau_history   = tau_hist;
end

% --- Helpers ---
function TH = overwrite_slice(theta, slice, z_blk)
    TH = theta; 
    TH(slice) = z_blk;
end

function ELITE_BUF = grow_elite_buffer(ELITE_BUF, theta_base, slice, elites_blk, Kkeep)
    % Appends new elite solutions to the global archive
    if isempty(ELITE_BUF), ELITE_BUF = zeros(0, numel(theta_base)); end
    for k = 1:Kkeep
        th = theta_base; 
        th(slice) = elites_blk(k, :);
        ELITE_BUF(end+1, :) = th; %#ok<AGROW>
    end
    % Limit Buffer Size
    MAXBUF = 200;
    if size(ELITE_BUF, 1) > MAXBUF
        ELITE_BUF = ELITE_BUF(end-MAXBUF+1:end, :);
    end
end

function X0 = init_from_elite(elite_thetas, slice, popsize, lb, ub, mix_ratio)
    % Generates initial population using Robust Statistics (IQR) from Elite Archive
    nvars = numel(slice);
    n_mix = max(1, round(popsize * mix_ratio));
    n_rand = popsize - n_mix;
    
    X0 = zeros(popsize, nvars);
    
    % Exploitation Part
    E = elite_thetas(:, slice);
    mu = mean(E, 1);
    
    % Robust Scale (IQR-based clipping)
    p25 = prctile(E, 25, 1);
    p75 = prctile(E, 75, 1);
    iqr_val = p75 - p25;
    sig_raw = std(E, [], 1);
    sig = min(sig_raw, 1.5 * iqr_val) + 1e-6; 
    
    mix_part = randn(n_mix, nvars) .* sig + mu;
    mix_part = min(max(mix_part, lb(slice)), ub(slice));
    X0(1:n_mix, :) = mix_part;
    
    % Exploration Part (Uniform Random)
    X0(n_mix+1:end, :) = lb(slice) + (ub(slice)-lb(slice)) .* rand(n_rand, nvars);
end