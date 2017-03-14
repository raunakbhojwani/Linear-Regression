function  theta = LinearRegWL2_train(Xtrain, Ytrain, alpha, w)

% Weighted Linear Regression: 
% Inputs (n: # of data points; d: number of features): 
%     -- Xtrain: (n X d) training features
%     -- Ytrain: (n X 1) training labels
%     -- alpha: (1 X 1) regularization
%     -- w:     (1 X d) weights in the L2 regularization
% 
% Outputs: 
%     -- theta: (1 X d) the parameter

% please complete the code
 
xT = Xtrain';
theta = (((xT*Xtrain) + (w*alpha)) \ (xT*Ytrain));
 
 
 
 
 
 



