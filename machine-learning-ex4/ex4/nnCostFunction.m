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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
y_matrix =  zeros(size(y,1),num_labels);
for t=1:size(y,1)
  y_matrix(t,y(t))=1;
endfor

a_1 = [ones(m, 1) X];
# size(a_1) [5000    401]

z_2 = Theta1*a_1';
#size(z_2) [25   5000]

g_2 = sigmoid(z_2);
#size(g_2) [25   5000]

a_2 = [ones(1, size(g_2, 2)); g_2];
#size(a_2) [26   5000]

z_3= Theta2*a_2;
#size(z_3) [10   5000]

g_3 = sigmoid(z_3)';
a_3 = g_3;
#size(y_matrix(:,)) [5000      10]
#size(g_3(:,)) [5000      10]

for t = 1:num_labels
  J = J + [(-y_matrix(:,t)'*log(a_3(:,t)))-((1-y_matrix(:,t))'*log(1-a_3(:,t)))] ;
endfor


%J=J/m



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

#a_1,z_2,g_2,a_2,z_3,g_3

del_3 =  a_3' .- y_matrix'; # [10 x 5000] - [10 x 5000]

#size(del_3' * Theta2(:,2:end))) # [5000 x 10] [10 x 25]

#size(z_2) [25 x 5000]

del_2 = (del_3' * Theta2(:,2:end)).*sigmoidGradient(z_2'); #[5000x10] *[10x25] - [5000 x 25]


Delta_3 = zeros(size(Theta2));
Delta_2 = zeros(size(Theta1));

#Delta_3(:,2:end) = Delta_3(:,2:end) + ((a_3'*del_2));

Delta_3 = del_3*a_2';

Delta_2 = del_2'*a_1;

Theta1_grad = (1/m)*Delta_2;
Theta2_grad = (1/m)*Delta_3;

Temp1 = Theta1;
Temp2 = Theta2;

Temp1(:,1) = 0;
Temp2(:,1) = 0;

Theta1_grad = Theta1_grad + Temp1*lambda/m;
Theta2_grad = Theta2_grad + Temp2*lambda/m;

##Delta_3(:,2:end) = del_3*a_2(2:end,:)';
##
##Delta_2(:,2:end) = del_2'*a_1(:,2:end);

##X(1,:)*Theta1(1,:)';
##z_2(1,:) = Theta1(1,:)*X(1,:)';
##a_2(1,:) = sigmoid(z_2(1,:));
##a_2 = [1;a_2];
##z_3(1,:) = Theta2(1,:)*a_2(1,:);
##a_3(1,:) = sigmoid(z_3(1,:))'
##
##d_3 = a_3(1,:)-y(1,:)
##d_2 = Theta2(1,:)'*d_3.*sigmoidGradient(z_2(1,:))
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


reg_cost = (lambda/(2*m))* [(Theta1(:,2:size(Theta1,2))(:)'*Theta1(:,2:size(Theta1,2))(:)+(Theta2(:,2:size(Theta2,2))(:))'*Theta2(:,2:size(Theta2,2))(:))];
J=(J/m)+reg_cost;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
