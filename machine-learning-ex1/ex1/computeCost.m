function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples

%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

Hx = X*theta;
J = (1/(2*m))*(sum((Hx - y).^2));

% =========================================================================

end
