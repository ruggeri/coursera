% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, regularizedGradient] = InstanceNegLogLikelihood(X, y, theta, modelParams)
    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if Y_2 = 5 and Y_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    disp("Begin generating features");
    fflush(stdout);
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.

    nll = 0;

    % Calculate partition function by creating clique tree.

    % Turn features into factors.
    disp("Begin creating factors");
    fflush(stdout);
    F = struct();
    for feature=featureSet.features
      % TODO: Clearly suboptimal. Optimize.
      factor = struct(
                   'var', feature.var,
                   'card', repmat(modelParams.numHiddenStates, 1, length(feature.var)),
                   'val', []);
      factor.val = zeros(1, prod(factor.card));
      factor = SetValueOfAssignment(factor,
                                    feature.assignment,
                                    theta(feature.paramIdx));
      factor.val = exp(factor.val);
      F(end+1) = factor;
    end

    % Create the clique tree and calibrate so we can get the partition
    % function.
    disp("Begin creating clique tree");
    fflush(stdout);
    P = CreateCliqueTree(F);
    disp("Begin calibrating clique tree");
    fflush(stdout);
    [P, logZ] = CliqueTreeCalibrate(P, false);
    nll += logZ;

    % Calculate log-probability of the data.
    disp("Begin calculating log-probability of data");
    fflush(stdout);
    for i=1:length(featureSet.features)
      feature = featureSet.features(i);
      if y(feature.var) == feature.assignment
        nll -= theta(feature.paramIdx);
      end
    end

    % Factor in regularization of weights.
    disp("Begin performing regularization");
    fflush(stdout);
    nll += (modelParams.lambda / 2) * theta * theta';

    % Calculate gradient.
    disp("Beginning Gradient Calculation");
    fflush(stdout);

    featureCounts = zeros(size(theta));
    modelFeatureCounts = zeros(size(theta));

    disp("Begin counting empirical/model feature counts");
    fflush(stdout);
    for feature=featureSet.features
      % Account for model feature counts.
      % TODO: Clearly suboptimal. Optimize!
      for cliqueIdx=1:length(P.cliqueList)
        clique = P.cliqueList(cliqueIdx);
        if any(!ismember(feature.var, clique.var))
          continue;
        end

        % Normalize; may not be done already?? WTF? Why would you do
        % that, asshats?
        clique.val /= sum(clique.val);
        clique = FactorMarginalization(
                     clique,
                     setdiff(clique.var, feature.var));
        v = GetValueOfAssignment(
                clique,
                feature.assignment,
                feature.var);
        modelFeatureCounts(feature.paramIdx) += v;

        break;
      end

      % Account for empirical feature counts.
      if all(y(feature.var) == feature.assignment)
        featureCounts(feature.paramIdx) += 1;
      end
    end

    unregularizedGradient = modelFeatureCounts - featureCounts;

    disp("Beginning regularization");
    fflush(stdout);
    % Derivative: Regularization
    regularizedGradient = unregularizedGradient;
    for i=1:length(theta)
      regularizedGradient(i) += modelParams.lambda * theta(i);
    end
end
