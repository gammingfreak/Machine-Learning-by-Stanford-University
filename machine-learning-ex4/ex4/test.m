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

X = [ones(m, 1) X];
z_1 = Theta1*X';
g_1 = sigmoid(z_1);

g_1 = [ones(1, size(g_1, 2)); g_1];

z_2= Theta2*g_1;
g_2= sigmoid(z_2)';

for t = 1:num_labels
  J = J + [(-y_matrix(:,t)'*log(g_2(:,t)))-((1-y_matrix(:,t))'*log(1-g_2(:,t)))] ;
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
delta_2=Theta1_grad(:,2:size(Theta1_grad,2));
delta_3=Theta2_grad(:,2:size(Theta2_grad,2));
for t=1:m
  z_2=zeros(1:hidden_layer_size,1);
  a_2=zeros(1:hidden_layer_size,1);
  z_3=zeros(1:num_labels,1);
  a_3=zeros(1:num_labels,1);

    for i=1:hidden_layer_size
      
      z_2(i) = X(t,:)*Theta1(i,:)';
      a_2(i) = sigmoid(z_2(i));
      a_2 = [1 a_2];
    endfor
    
    for i=1:num_labels
      z_3(i) = Theta2(i,:)*a_2;
      a_3(i) = sigmoid(z_3(i))';
    endfor

    d_3 = a_3-y_matrix(t,:);

    d_2 = (Theta2(:,2:size(Theta2,2))'*d_3')' .*(sigmoidGradient(X(t,2:size(X,2))*Theta1(:,2:size(Theta1,2))'));

    delta_2 = delta_2+(d_2'*X(t,2:size(X,2)));
    
    delta_3 = delta_3+d_3'*a_2(2:length(a_2))';

    #delta_2 = delta_2+(d_2(2:length(d_2))'*X(t,2:size(X,2)))
    
    #delta_3 = delta_3+d_3(2:length(d_3))'*a_2(2:length(a_2));
   
    
endfor

Theta1_grad = [zeros(size(delta_2,1),1) delta_2]/(m);
Theta2_grad = [zeros(size(delta_3,1),1) delta_3]/(m);

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
