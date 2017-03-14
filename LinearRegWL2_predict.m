function  hatYtest = LinearRegWL2_predict(Xtest, theta)
% Weighted Linear Regression: 
% Inputs (n: # of data points; d: number of features): 
%     -- Xtest: (n X d) features of the testing data
%     -- theta: (d X 1) parameter vector
%
% Outputs: 
%    -- hatYtest: (n X 1) prediction on the testing data points

%hatYtest =  ... please complete the code

hatYtest = Xtest*theta;