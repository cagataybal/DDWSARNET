function [ce, mse] = nn_loss(P, y, K)
% NN_LOSS Computes Cross-Entropy (CE) and Mean Squared Error (MSE).
%   P: [N x K] Probability matrix
%   y: [N x 1] True class labels (0 to K-1)
    
    N = size(P, 1);
    
    % Cross-Entropy
    rows = (1:N)'; 
    cols = y(:) + 1; % Convert 0-based to 1-based index
    linearInd = sub2ind([N, K], rows, cols);
    pick = P(linearInd);
    ce = -mean(log(pick + eps));
    
    % MSE (Optional return)
    if nargout > 1
        Y_onehot = zeros(N, K);
        Y_onehot(linearInd) = 1;
        mse = mean(sum((Y_onehot - P).^2, 2));
    end
end