function [xbest, fbest, exitflag, output] = wsar(fun, nvars, lb, ub, opts)
% WSAR Core engine for Weighted Superposition Attraction-Repulsion algorithm.
%
%   Solves optimization problem min f(x) subject to lb <= x <= ub.
%
%   DDWSARNET: Deep Layer-Wise Dynamic-Tau WSAR Neural Network
%   Copyright (c) 2026 Cagatay Bal, PhD.
%   Licensed under the MIT License.

    if nargin < 5 || isempty(opts), opts = wsarset(); end
    
    % --- Initialization ---
    popsize  = opts.PopulationSize; 
    maxIter  = opts.MaxIterations;
    tao      = opts.Tao; 
    taoMode  = opts.TaoMode;
    tau0     = opts.Tao0; 
    Wlen     = opts.AdaptWindow; 
    tol      = opts.AdaptTol;
    rngSeed  = opts.RandomSeed;
    
    if ~isempty(rngSeed), rng(rngSeed); end
    
    % Expand bounds
    if isscalar(lb), lb = repmat(lb, 1, nvars); end
    if isscalar(ub), ub = repmat(ub, 1, nvars); end
    
    % Initial Population
    X0 = opts.InitialPopulation;
    if ~isempty(X0)
        agent = X0;
        if size(agent,1) < popsize
            needed = popsize - size(agent,1);
            agent = [agent; lb + (ub-lb).*rand(needed, nvars)];
        elseif size(agent,1) > popsize
            agent = agent(1:popsize, :);
        end
    else
        agent = lb + (ub-lb).*rand(popsize, nvars);
    end
    
    % Evaluate Initial Population
    agent_OBJ = local_eval(fun, agent);
    [fbest, i0] = min(agent_OBJ);
    xbest = agent(i0, :);
    
    % History Buffers
    BEST_hist = zeros(maxIter, 1);
    tao_hist  = zeros(maxIter, 1);
    
    % Reset Dynamic Controller
    compute_tau(0, [], Wlen, tol, tau0, 'reset');
    
    t_start = tic;
    
    % --- Main Loop ---
    for q = 1:maxIter
        % 1. Update Tau
        if strcmpi(taoMode, 'fixed')
            curTau = tao;
        else
            curTau = compute_tau(q, BEST_hist, Wlen, tol, tau0);
        end
        tao_hist(q) = curTau;
        
        % 2. Rank and Weighting
        [~, order] = sort(agent_OBJ, 'ascend');
        rank = zeros(popsize, 1);
        m = min(numel(order), popsize);
        rank(order(1:m)) = (1:m).';
        
        rrank  = popsize - rank + 1;
        BESTw  = rank .^ curTau;
        WORSTw = rrank.^ curTau;
        
        % 3. Generate Superposition Centers (BEST and WORST)
        rv = rand(1, nvars);
        BEST_center  = zeros(1, nvars);
        WORST_center = zeros(1, nvars);
        
        for d = 1:nvars
            % Vectorized Weighted Selection Logic
            % (Optimized for readability loop-wise here, but logic remains)
            c_b = []; w_b = [];
            c_w = []; w_w = [];
            for j = 1:popsize
                if BESTw(j) >= rv(d),  c_b(end+1) = agent(j,d); w_b(end+1) = BESTw(j);  end
                if WORSTw(j) >= rv(d), c_w(end+1) = agent(j,d); w_w(end+1) = WORSTw(j); end
            end
            BEST_center(d)  = local_choose(c_b, w_b, agent(:,d));
            WORST_center(d) = local_choose(c_w, w_w, agent(:,d));
        end
        
        % 4. Fitness of Centers & Swap
        best_c_obj  = local_eval(fun, BEST_center);
        worst_c_obj = local_eval(fun, WORST_center);
        
        if worst_c_obj < best_c_obj
            tmp = BEST_center; BEST_center = WORST_center; WORST_center = tmp;
            best_c_obj = worst_c_obj; % update known fitness
        end
        
        % 5. Update Positions (JAYA-like Attraction-Repulsion)
        nagent = agent;
        for j = 1:popsize
            if agent_OBJ(j) > best_c_obj
                r1 = rand(); r2 = rand();
                step = r1 * (BEST_center - abs(agent(j,:))) + ...
                       r2 * (abs(agent(j,:)) - WORST_center);
                nagent(j,:) = agent(j,:) + step;
            else
                % Random Move (Levy-flight like or simple stochastic)
                step = rand();
                nagent(j,:) = nagent(j,:) + (2*rand(1,nvars)-1)*step;
            end
            
            % Boundary Clamp
            nagent(j,:) = min(max(nagent(j,:), lb), ub);
        end
        
        % 6. Selection (Greedy)
        nagent_OBJ = local_eval(fun, nagent);
        better = nagent_OBJ <= agent_OBJ;
        agent(better, :)   = nagent(better, :);
        agent_OBJ(better) = nagent_OBJ(better);
        
        % 7. Update Global Best
        [curBest, curIdx] = min(agent_OBJ);
        if curBest < fbest
            fbest = curBest;
            xbest = agent(curIdx, :);
        end
        BEST_hist(q) = fbest;
        
        % Display
        if strcmpi(opts.Display, 'iter') && (mod(q, 50) == 0 || q == 1)
            fprintf('Iter %4d | Best: %.6e | Tau: %.3f\n', q, fbest, curTau);
        end
    end
    
    output.iterations  = maxIter;
    output.besthistory = BEST_hist;
    output.tao_history = tao_hist;
    output.elapsedtime = toc(t_start);
    output.finalpopulation = agent;
    output.finalscores = agent_OBJ;
    exitflag = 1;
end

% --- Local Helpers ---
function vals = local_eval(fun, X)
    % Evaluates function for population matrix X
    [pop, ~] = size(X);
    vals = zeros(pop, 1);
    for i = 1:pop
        vals(i) = fun(X(i, :));
    end
end

function val = local_choose(c, w, fallback)
    % Weighted random selection
    if isempty(c)
        val = fallback(randi(numel(fallback)));
        return;
    end
    w = max(0, w(:)); 
    s = sum(w);
    if s == 0
        val = fallback(randi(numel(fallback)));
    else
        w = w / s;
        cdf = cumsum(w);
        idx = find(rand() <= cdf, 1, 'first');
        if isempty(idx), idx = numel(c); end
        val = c(idx);
    end
end