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

%tempC = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%tempSigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

%temp = 1;

%for i = 1:length(tempC)
%	for j = 1:length(tempSigma)
%		model = svmTrain(X, y, tempC(i), @(x1, x2) gaussianKernel(x1, x2, tempSigma(j)));
%		predictions = svmPredict(model, Xval);
%		temp2 = mean(double(predictions ~= yval));
%		if (temp2 < temp)
%			temp = temp2;
%			C = tempC(i);
%			sigma = tempSigma(j);
%			disp(C);
%			disp(sigma);
%		end
%	end
%end

%disp(C);
%disp(sigma);




% =========================================================================

end
