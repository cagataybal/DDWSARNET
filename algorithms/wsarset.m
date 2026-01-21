function options = wsarset(varargin)
% WSARSET Create or modify WSAR optimization options structure.
%
%   OPTIONS = WSARSET('PARAM1',VALUE1,'PARAM2',VALUE2,...) creates a
%   structure with the specified parameters.
%
%   DDWSARNET: Deep Layer-Wise Dynamic-Tau WSAR Neural Network
%   Copyright (c) 2026 Cagatay Bal, PhD.
%   Licensed under the MIT License.
%
%   Parameters:
%       'PopulationSize' - Number of agents (default: 100)
%       'MaxIterations'  - Max number of iterations (default: 1000)
%       'Tao'            - Fixed selection pressure (default: -0.8)
%       'TaoMode'        - 'fixed' or 'loss' (adaptive)
%       'Tao0'           - Initial Tau for adaptive mode
%       'AdaptWindow'    - Window size for loss monitoring
%       'AdaptTol'       - Tolerance for stagnation detection
%       'RandomSeed'     - Fix RNG seed for reproducibility
%       'Display'        - Level of display: 'none' or 'iter'

    % Default Options
    options.PopulationSize = 100;
    options.MaxIterations  = 1000;
    options.Tao            = -0.8;
    options.TaoMode        = 'fixed';
    options.Tao0           = -0.8;
    options.AdaptWindow    = 25;
    options.AdaptTol       = 1e-4;
    options.UseVectorized  = false;
    options.Display        = 'none';
    options.RandomSeed     = [];
    options.InitialPopulation = [];

    % Parse Arguments
    for k = 1:2:numel(varargin)
        key = varargin{k};
        if k+1 > numel(varargin)
            error('wsarset:MissingValue', 'Missing value for parameter "%s".', key);
        end
        val = varargin{k+1};
        
        switch lower(key)
            case {'populationsize','popsize'}, options.PopulationSize = val;
            case {'maxiterations','maxiter'},  options.MaxIterations  = val;
            case 'tao',        options.Tao     = val;
            case 'taomode',    options.TaoMode = lower(val);
            case 'tao0',       options.Tao0    = val;
            case 'adaptwindow',options.AdaptWindow = val;
            case 'adapttol',   options.AdaptTol    = val;
            case 'usevectorized', options.UseVectorized = logical(val);
            case 'display',    options.Display = val;
            case 'randomseed', options.RandomSeed = val;
            case 'initialpopulation', options.InitialPopulation = val;
            otherwise
                error('wsarset:UnknownParameter', 'Unknown parameter "%s".', key);
        end
    end
end