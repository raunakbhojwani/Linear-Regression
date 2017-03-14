%%% Load and plot data %%%% 
clear all; 
rng('default'); rng(1);
load curveData
figure; hold on;
plot(Xtrain, Ytrain, 'o');

%% Run polynomial Regression 

poly_order = 10; % order of the polynomial feature
alpha = 1;       % regularization parameter
w = [0, ones(1,poly_order)]; % weights in L2 norm; we set w(1)=0 so that we do not regularize on the constant term. 

% Apply linear regression on the polynomial feature to the data with a
% polynomial curve
theta = LinearRegWL2_train(poly_feature(Xtrain, poly_order),  Ytrain, alpha, w); % training 
hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order), theta); % testing
err_test = calculate_mse_Ytest_curveData(hatYtest); % calculate mse = mean((Ytest-hatYtest).^2);
fprintf('Testing Error is %d\n', err_test);

% plot the curve 
Xgrid = linspace(0,10,100)';
hatYgrid = LinearRegWL2_predict(poly_feature(Xgrid,poly_order), theta); % testing

figure; hold on;
plot(Xtrain, Ytrain, 'o');
plot(Xgrid, hatYgrid, '-');
legend('Training', 'Testing');


%% cross validation (Simple example):

% create a function handle that takes x and y as inputs and
% output the estimated parameter, the other parameters (poly_order, alpha, w) are
% are fixed and defined outside of the function handle. For example, you can run train_func(Xtrain, Ytrain),
% and get the same theta you had before. 
% For more info about matlab function handle, see 
% http://www.mathworks.com/help/matlab/matlab_prog/creating-a-function-handle.html
% 
Nfold = 2; % the code only works when Nfold=2 currently, please make work for general Nfold

train_func = @(x,y)LinearRegWL2_train(poly_feature(x, poly_order),  y, alpha, w);  
% create a function handle for calculaing the MSE on xtest with linear regression parameter th: 
evaluate_func = @(x, y, th)(mean((LinearRegWL2_predict(poly_feature(x, poly_order), th) - y).^2)); 
% cross validation: taking the data and train_func, predict_func as inputs and output the cross validation error 
err_xVar = cross_validation(Xtrain, Ytrain, Nfold, train_func, evaluate_func);


%% Use cross validation to select the best alpha:
Nfold = 5;
poly_order = 10; % order of the polynomial feature
w = [0, ones(1,poly_order)]; % weights in L2 norm; we set w(1)=0 so that we do not regularize on the constant term. 
alphaVec = [0,0.1,1,10,100];
for i = 1:length(alphaVec)
    alpha  = alphaVec(i);
    %... please complete the code
    
    theta = LinearRegWL2_train(poly_feature(Xtrain, poly_order),  Ytrain, alpha, w); % training 
    hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order), theta); % testing
    err_xVar_Vec_alpha(i) = calculate_mse_Ytest_curveData(hatYtest);
    %err_xVar_Vec(i) = ... please complete the code
end

[nothing, alphaBest] = min(err_xVar_Vec_alpha);

% Rerun your model with the best alpha and evaluate your performance 
% hatYtest = ....  please complete the code 
w = [0, ones(1,poly_order)];
theta = LinearRegWL2_train(poly_feature(Xtrain, poly_order),  Ytrain, alphaVec(alphaBest), w); % training 
hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order), theta); % testing
err_test = calculate_mse_Ytest_curveData(hatYtest); % calculate mse = mean((Ytest-hatYtest).^2);
fprintf('Testing Error is %d\n', err_test);
fprintf('Optimal alpha is %d\n', alphaVec(alphaBest+1));


%% Use cross validation to select the best polynomial order:
Nfold = 5;
alpha = .1;
poly_order_Vec = [1,2,3,4,5,10,20];
for i = 1:length(poly_order_Vec)
    poly_order  = poly_order_Vec(i);
    w = [0, ones(1,poly_order)]; % weights in L2 norm; we set w(1)=0 so that we do not regularize on the constant term.     
    %... please complete the code
    
    poly = poly_feature(Xtrain, poly_order);
    theta = LinearRegWL2_train(poly,  Ytrain, alpha, w); % training 
    hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order), theta); % testing
    err_xVar_Vec_PolyOrder(i) = calculate_mse_Ytest_curveData(hatYtest); % calculate mse = mean((Ytest-hatYtest).^2);
    
    %err_xVar_Vec(i) = ... % please complete the code
end

bestErr = err_xVar_Vec_PolyOrder(1);
bestPolyOrder = 1;
for i = 1:length(err_xVar_Vec_PolyOrder)
    if (err_xVar_Vec_PolyOrder(i) <= bestErr) 
        bestErr = err_xVar_Vec_PolyOrder(i);
        bestPolyOrder = i;
    end
end

%poly_order_best = ....
[nothing, polyBest] = min(err_xVar_Vec_PolyOrder);

% Rerun your model with the best polyorder and evaluate your performance 
% hatYtest = ....  please complete the code 
w = [0, ones(1,poly_order_Vec(polyBest))];
theta = LinearRegWL2_train(poly_feature(Xtrain, poly_order_Vec(polyBest)),  Ytrain, alpha, w); % training 
hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order_Vec(polyBest)), theta); % testing
err_test = calculate_mse_Ytest_curveData(hatYtest); % calculate mse = mean((Ytest-hatYtest).^2);
fprintf('Testing Error is %d\n', err_test);
fprintf('Optimal Poly Order is %d\n', poly_order_Vec(polyBest-1));


%% Use various combinations of polyorder and alpha
Nfold = 5;
poly_order_Vec = [1,2,3,4,5,10,20];
alphaVec = [0,0.1,1,10,100];
for i = 1:length(poly_order_Vec)
    poly_order  = poly_order_Vec(i);
    for i = 1:length(alphaVec)
        alpha  = alphaVec(i);

        theta = LinearRegWL2_train(poly_feature(Xtrain, poly_order),  Ytrain, alpha, w); % training 
        hatYtest = LinearRegWL2_predict(poly_feature(Xtest,poly_order), theta); % testing
        err_xVar_Vec_alphaPoly(i) = calculate_mse_Ytest_curveData(hatYtest);
        %err_xVar_Vec(i) = ... please complete the code
    end
end
