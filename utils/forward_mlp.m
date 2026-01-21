function [ZL, P] = forward_mlp(theta, X, dim_list, actHidden)
% FORWARD_MLP Computes the forward pass of a Multi-Layer Perceptron.
%   Returns logits (ZL) and Softmax Probabilities (P).
    S = layer_slices(dim_list);
    H = X;
    for l = 1:S.L
        Wl = reshape(theta(S.W{l}), dim_list(l), dim_list(l+1));
        bl = reshape(theta(S.b{l}), 1, dim_list(l+1));
        Zl = H * Wl + bl;
        
        if l < S.L
            % Hidden Layer Activation
            switch lower(actHidden)
                case 'sigmoid', H = 1 ./ (1 + exp(-Zl));
                case 'relu',    H = max(0, Zl);
                otherwise, error('Unknown activation: %s', actHidden);
            end
        else
            % Output Layer (Logits -> Softmax)
            ZL = Zl;
            % Stability trick: subtract max
            ZL = ZL - max(ZL, [], 2);
            P  = exp(ZL);
            P  = P ./ sum(P, 2);
        end
    end
end