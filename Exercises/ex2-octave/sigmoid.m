function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Note:
% 1./2 = 1/2 = 0.5
% 1./[ 2 3; 4 5 ] =  [0.5 0.33; 0.25 0.2 ]

g = 1./( 1 + exp(z.*(-1)) )


% =============================================================

end
