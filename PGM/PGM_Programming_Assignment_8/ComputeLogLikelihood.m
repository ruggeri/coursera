function logProb = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
NUM_POSES = 10;
K = length(P.c); % number of classes

logProb = 0.0;

% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for dataIdx=1:N
  logProbs = [];
  for classIdx=1:K
    logProbK = log(P.c(classIdx));

    example = dataset(dataIdx, :, :);
    example = reshape(example, NUM_POSES, 3);

    for poseIdx=1:NUM_POSES
      if G(poseIdx, 1) == 0
        logProbK += ComputeRootLogProb(P, G, classIdx, example, poseIdx);
      else
        logProbK += ComputeChildLogProb(P, G, classIdx, example, poseIdx);
      end
    end

    logProbs(end+1) = logProbK;
  end

  logProb += log(sum(exp(logProbs)));
end

end

function logProb = ComputeRootLogProb(P, G, classIdx, example, poseIdx)
  pose = example(poseIdx, :);

  params = P.clg(poseIdx);

  logProbs = [];
  logProbs(end+1) = ...
    lognormpdf(pose(1), params.mu_y(classIdx), params.sigma_y(classIdx));
  logProbs(end+1) = ...
    lognormpdf(pose(2), params.mu_x(classIdx), params.sigma_x(classIdx));
  logProbs(end+1) = ...
    lognormpdf(pose(3), params.mu_angle(classIdx), params.sigma_angle(classIdx));

  logProb = sum(logProbs);
end

function logProb = ComputeChildLogProb(P, G, classIdx, example, poseIdx)
  parentPoseIdx = G(poseIdx, 2);
  parentPose = example(parentPoseIdx, :)(:);
  pose = example(poseIdx, :);

  % Calculate means.
  theta = P.clg(poseIdx).theta(classIdx, :);
  mu_y = theta(1:4)  * [1; parentPose];
  mu_x = theta(5:8)  * [1; parentPose];
  mu_angle = theta(9:12) * [1; parentPose];

  % Extract variances.
  params = P.clg(poseIdx);

  logProbs = [];
  logProbs(end+1) = lognormpdf(pose(1), mu_y, params.sigma_y(classIdx));
  logProbs(end+1) = lognormpdf(pose(2), mu_x, params.sigma_x(classIdx));
  logProbs(end+1) = lognormpdf(pose(3), mu_angle, params.sigma_angle(classIdx));

  logProb = sum(logProbs);
end
