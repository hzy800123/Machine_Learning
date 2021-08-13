function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g = sigmoid(X * theta)
% For gOther, it needs to exclude the 1st column in Matrix X, and exclude 1st row in Matrix theta
% gOther = sigmoid(X(:,2:end) * theta(2:end))
% gZero = sigmoid(X(:,1) * theta(1))

J = sum( (-1) * y' * log(g)  - (1 - y)' * log(1 - g) ) / m + (lambda / (2*m) * sum(theta(2:end).^2))

grad = X' * ( g - y ) / m

% 将grad的 第二个元素到最后一个元素，都加上一个值：(lambda / m * theta(2:end))
grad(2:end) = grad(2:end) + (lambda / m * theta(2:end))

% Exclue the 1st column in Matrix X, and then calculate its 转置矩阵
% 或者：先求 X 的转置矩阵，然后去掉 第一行
% gradOther = (X(:,2:end))' * ( gOther - y ) / m + ( lambda * theta(2:end) / m )
% gradZero = (X(:,1))' * ( gZero - y ) / m

% grad = [gradZero; gradOther]


% p(hx>=0.5) = 1;   // Update the element in the vector hx to be 1, if the element >= 0.5


% =============================================================

end
