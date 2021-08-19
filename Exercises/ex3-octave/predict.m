function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = X
% Add ones to the a1 data matrix
% X = [ones(m, 1) X];
a1Plus = [ones(m, 1) a1];
% Now the matrix a1Plus is 5000 x 401

% z2 = X * Theta1';
z2 = a1Plus * Theta1';
a2 = sigmoid(z2);

% Add ones to the a2 data matrix
a2Plus = [ones(m, 1) a2];

z3 = a2Plus * Theta2';
a3 = sigmoid(z3);

% size(a3)

% For Each Row Max value and corresponding index of them can be found as below
[ max_values, indices ] = max( a3, [] , 2 );

p = indices;

% size(p)



% =========================================================================


end
