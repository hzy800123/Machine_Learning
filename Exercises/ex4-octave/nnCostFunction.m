function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


%% Setup the parameters you will use for this exercise
% input_layer_size  = 400;  % 20x20 Input Images of Digits
% hidden_layer_size = 25;   % 25 hidden units
% num_labels = 10;          % 10 labels, from 1 to 10   
                            % (note that we have mapped "0" to label 10)


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
% m = 5000 samples in training set
% The Matix X is 5000 x 400


% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% Theta1 is 25 x 401
% Theta2 is 10 x 26


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




a1 = X
% Add ones to the a1 data matrix
a1Plus = [ones(m, 1) a1]
% Now the matrix a1Plus is 5000 x 401

z2 = a1Plus * Theta1'
a2 = sigmoid(z2)

% Add ones to the a2 data matrix
a2Plus = [ones(m, 1) a2]

z3 = a2Plus * Theta2'
a3 = sigmoid(z3)
g = a3

% a3 is Matrix of 5000 x 10
% y is Matrix of 5000 x 1

n = num_labels  % 10 labels, from 1 to 10   
yN = full(sparse(1:numel(y), y, 1, numel(y), n));
size(yN)
% y is Matrix of 5000 x 1
% yN is Matrix of 5000 x 10, it change number to vector
% E.g. 2 -> [ 0 1 0 0 0 0 0 0 0 0 ]

% J = sum( (-1) * y' * log(g)  - (1 - y)' * log(1 - g) ) / m + (lambda / (2*m) * sum(theta(2:end).^2))
J = sum(sum( (-1) * yN .* (log(g))  - (1 - yN) .* (log(1 - g)) )) / m

% sum(theta(2:end).^2))

% Theta1 is 25 x 401
% Theta2 is 10 x 26

Theta1Sum = sum(sum(Theta1(:, 2:end).^2))
Theta2Sum = sum(sum(Theta2(:, 2:end).^2))

J = J + lambda / (2*m) * ( Theta1Sum + Theta2Sum )

% Theta1 is 25 x 401
% Theta2 is 10 x 26

% Initial the DELTA1 is 25 x 401
% DELTA1 = zeros(25, 401)
DELTA1 = zeros(size(Theta1, 1), size(Theta1, 2))

% Initial the DELTA2 is 10 x 26
% DELTA2 = zeros(10, 26)
DELTA2 = zeros(size(Theta2, 1), size(Theta2, 2))

% Test dummy data
% input_layer_size = 3;
% hidden_layer_size = 5;
% num_labels = 3;
% m = 5;

% Loop from 1 to 5000
for t = 1:m
    
    % Step 1: computing the activations (z(2); a(2); z(3); a(3)) for layers 2 and 3

    % Matrix X is 5000 x 400
    % Select the each row from Matrix X
    a1Vector = X(t,:)
    % Add one Bias Unit to the a1Vector
    a1VectorPlus = [1 a1Vector]
    % Now the matrix a1VectorPlus is 1 x 401

    % Theta1 is 25 x 401
    z2Vector = a1VectorPlus * Theta1'
    a2Vector = sigmoid(z2Vector)
    % Now the matrix a2Vector is 1 x 25

    % Add one Bias Unit to the a2Vector
    a2VectorPlus = [1 a2Vector]
    % Now the matrix a2VectorPlus is 1 x 26

    % Theta2 is 10 x 26
    z3Vector = a2VectorPlus * Theta2'
    a3Vector = sigmoid(z3Vector)
    % Now the matrix z3Vector is 1 x 10    
    % gVector = a3Vector


    % Step 2: Calculate the delta3K in layer 3
    % yN is Matrix of 5000 x 10, it change number to vector
    y3K = yN(t,:)

    % delta3K is 1 x 10 in layer 3
    delta3K = a3Vector - y3K


    % Step 3: Calculate the delta2K in layer 2

    % The matrix z2Vector is 1 x 25
    % delta3K is 1 x 10    
    % Remove the delta2(0), it needs to remove the 1st column
    delta2K = (delta3K * Theta2)(2:end) .* sigmoidGradient(z2Vector)
    % delta2K = (delta3K * Theta2)(2:end) .* (a2Vector .* (1 - a2Vector))
    % delta2K is 1 x 25
    

    % Step 4: Accumulate the gradient from this example
    % delta3K is 1 x 10
    % a2VectorPlus is 1 x 26    
    % DELTA2 is 10 x 25
    DELTA2 = DELTA2 + delta3K' * a2VectorPlus
    % DELTA2 is 10 x 26
    % Theta2 is 10 x 26

    % delta2K is 1 x 25
    % a1VectorPlus is 1 x 401    
    % DELTA1 is 25 x 401
    DELTA1 = DELTA1 + delta2K' * a1VectorPlus    
    % DELTA1 is 25 x 401
    % Theta1 is 25 x 401

end

% Step 5: Obtain the (unregularized) gradient for the neural network cost function
Theta1_grad = DELTA1 / m
Theta2_grad = DELTA2 / m


% Step 6: Add regularization to the gradient
% 将grad 矩阵中的 每一行的 从第二个元素到最后一个元素，都加上一个值：(lambda / m * theta(:, 2:end))
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m * Theta1(:, 2:end))
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m * Theta2(:, 2:end))


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
