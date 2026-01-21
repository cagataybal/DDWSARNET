function curTau = compute_tau(q, BEST_hist, Wlen, tol, tau0, mode)
% COMPUTE_TAU Calculates the dynamic selection pressure (tau) based on loss history.
%
%   This function implements the Bi-Directional Controller (Exploration vs Exploitation)
%   described in the DDWSARNET paper.
%
%   DDWSARNET: Deep Layer-Wise Dynamic-Tau WSAR Neural Network
%   Copyright (c) 2026 Cagatay Bal, PhD.
%   Licensed under the MIT License.

    persistent stagnation_count cooldown_left last_ref

    % Reset state if requested
    if nargin >= 6 && strcmp(mode, 'reset')
        stagnation_count = 0; 
        cooldown_left = 0; 
        last_ref = Inf; 
        curTau = tau0; 
        return
    end

    % Hyperparameters for the controller
    epsTol    = 0.07 * tol;                % Threshold for "flat" landscape
    patience  = max(8, round(1.5 * Wlen)); % Patience before triggering exploration
    coolIters = max(5, round(Wlen / 3));   % Cooldown period after switch

    % Early iterations: stick to baseline
    if q <= 2
        curTau = tau0;
        last_ref = BEST_hist(max(1, q));
        return
    end

    % Monitor the sliding window
    p0   = max(1, q - max(2, Wlen));
    prev = BEST_hist(p0); 
    last = BEST_hist(max(1, q-1));
    
    if prev == 0 || last == 0
        improv = Inf; 
    else
        improv = prev - last; 
    end

    % Check cooldown
    if cooldown_left > 0
        curTau = tau0; 
        cooldown_left = cooldown_left - 1; 
        last_ref = last; 
        return
    end

    % Detect Stagnation
    flat_change = abs(last - last_ref);
    if flat_change < epsTol
        stagnation_count = stagnation_count + 1;
    else
        stagnation_count = 0;
    end
    last_ref = last;

    % State 1: Trigger Exploration (Relax Tau)
    if stagnation_count >= patience
        curTau = tau0; % Reset first
        cooldown_left = coolIters - 1;
        stagnation_count = 0;
        return
    end

    % State 2: Trigger Intensification vs Exploration based on improvement
    if improv < tol
        curTau = 0.5 * tau0;   % Relax -> Explore
    else
        curTau = 1.5 * tau0;   % Intensify -> Exploit
    end
end