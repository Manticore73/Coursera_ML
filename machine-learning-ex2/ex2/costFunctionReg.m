function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m, p] = size(X); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));
costSum = 0;
Reg = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = zeros(m);
z = zeros(m);

z = X * theta;

for i = 1:1:m
    h(i) = sigmoid(z(i));
    costSum = costSum + (1/m).*( -y(i).*log(h(i)) - (1-y(i)).*log(1 - h(i)));
end

for j = 2:1:p
    Reg = Reg + (lambda/(2*m))*(theta(j)).^2;
end

J = costSum + Reg;

%%% Calculating the Gradience %%%

%%% Gradience for theta_0 %%%
for i = 1:1:m
        grad(1) = grad(1) + (1/m)*(h(i) - y(i)).*(X(i,1));
end

for j = 2:1:p
    for i = 1:1:m
        grad(j) = grad(j) + (1/m)*(h(i) - y(i)).*(X(i,j));
    end
    grad(j) = grad(j) + (lambda*theta(j))/(m);
end

% =============================================================

end
