function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% Initial X_poly as X for the p = 1
X_poly = X

% If p > 1, proceed to go into Loop to generate more X(i).^p elements, 
% and then add into matirx X_poly
if p > 1
    for i = 2:p
        X_poly_power_temp = X.^i
        X_poly = [ X_poly X_poly_power_temp ]
    end
end

% =========================================================================

end
