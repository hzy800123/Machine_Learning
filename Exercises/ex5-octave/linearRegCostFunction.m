function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Sample of linear regression without regularization
% J = sum((X * theta - y).^2)/(2*m);

% Sample of logistic regression with regularization
% J = sum( (-1) * y' * log(g)  - (1 - y)' * log(1 - g) ) / m + (lambda / (2*m) * sum(theta(2:end).^2))

J = sum((X * theta - y).^2)/(2*m) + (lambda / (2*m) * sum(theta(2:end).^2))

grad = X' * ( X * theta - y ) / m

% 将grad的 第二个元素到最后一个元素，都加上一个值：(lambda / m * theta(2:end))
grad(2:end) = grad(2:end) + (lambda / m * theta(2:end))


% =========================================================================

grad = grad(:);

end
