function S = layer_slices(dim_list)
% LAYER_SLICES Generates index maps for flattening/reshaping NN weights.
    L = numel(dim_list)-1;
    W = cell(1,L); 
    b = cell(1,L);
    idx = 0;
    for l = 1:L
        d_in  = dim_list(l); 
        d_out = dim_list(l+1);
        W{l} = idx + (1:d_in*d_out); idx = idx + d_in*d_out;
        b{l} = idx + (1:d_out);      idx = idx + d_out;
    end
    S.W = W; 
    S.b = b; 
    S.L = L; 
    S.n_params = idx;
end