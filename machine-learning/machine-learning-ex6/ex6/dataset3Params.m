function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%best_error = Inf;
%vals = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
%for i = 1:length(vals)
%  j = 3;
%
%  C2 = vals(i);
%  %for j = 1:length(vals)
%    sigma2 = vals(j);
%    model = svmTrain(X, y, C2, @(x1, x2) gaussianKernel(x1, x2, sigma2));
%    predictions = svmPredict(model, Xval);
%    error = mean(double(predictions ~= yval));
%
%    disp("MODEL:");
%    disp(C2);
%    disp(sigma2);
%    disp(error);
%
%    if error < best_error
%      disp("SET");
%      best_error = error;
%      C = C2;
%      sigma = sigma2;
%    end
%  %end
%end

% =========================================================================

end
