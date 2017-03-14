function  XX = poly_feature(X, poly_order)
% construc the polynomial feature [1, x, x.^2, ..., x.^poly_order]
% Inputs (n=# of data points):
%   -- X: (n X 1) the original 1D feature 
%   -- XX: (n X poly_order+1) the polynomial feature


XX = ones(size(X,1), poly_order+1);
for i = 0:poly_order
    XX(:,i+1) = X.^i;
end