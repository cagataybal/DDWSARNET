function I = point2img(x2, imgSize, sigma)
% POINT2IMG Converts 2D points into a Gaussian heatmap image (for CNN input).
    xy = max(min(x2, 3), -3); % Clip to [-3, 3]
    xy = (xy + 3) / 6;        % Normalize to [0, 1]
    
    cx = 1 + xy(1) * (imgSize - 1);
    cy = 1 + xy(2) * (imgSize - 1);
    
    [xg, yg] = meshgrid(1:imgSize, 1:imgSize);
    distSq = (xg - cx).^2 + (yg - cy).^2;
    
    I = exp(-distSq / (2 * sigma^2));
    I = single(reshape(I, imgSize, imgSize, 1));
end