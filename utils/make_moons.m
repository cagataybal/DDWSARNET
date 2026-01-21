function [X, y] = make_moons(N, noise)
% MAKE_MOONS Generates the classic Two-Moons benchmark dataset.
    n2 = floor(N/2);
    
    % Moon 1 (Upper)
    t1 = pi * rand(n2, 1);
    x1 = [cos(t1), sin(t1)];
    
    % Moon 2 (Lower)
    t2 = pi * rand(n2, 1);
    x2 = [1 - cos(t2), -sin(t2) - 0.5];
    
    X = [x1; x2] + noise * randn(N, 2);
    y = [zeros(n2, 1); ones(n2, 1)];
    
    % Fill remainder if N is odd
    if size(X, 1) < N
        X(end+1, :) = [cos(0.5*pi), sin(0.5*pi)] + noise * randn(1, 2);
        y(end+1, 1) = 0;
    end
end