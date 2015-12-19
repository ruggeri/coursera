function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

  N = size(dataset, 1);
  NUM_POSES = size(dataset, 2);
  K = size(labels, 2);

P.c = mean(labels);
P.clg = [];
for poseIdx=1:size(dataset, 2)
  if G(poseIdx, 1) == 0
    P.clg(poseIdx) = LearnRootCPD(dataset, G, labels, poseIdx);
  else
    P.clg(poseIdx) = LearnChildCPD(dataset, G, labels, poseIdx);
  end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);
fprintf('log likelihood: %f\n', loglikelihood);
end

function params = LearnRootCPD(dataset, G, labels, poseIdx)
  params = struct(
               "theta", [],
               "mu_y", [],
               "sigma_y", [],
               "mu_x", [],
               "sigma_x", [],
               "mu_angle", [],
               "sigma_angle", []);

  for labelIdx=1:size(labels, 2)
    examples = find(labels(:, labelIdx));
    poses = squeeze(dataset(examples, poseIdx, :));

    [params.mu_y(end+1), params.sigma_y(end+1)] = ...
      FitGaussianParameters(poses(:, 1));
    [params.mu_x(end+1), params.sigma_x(end+1)] = ...
      FitGaussianParameters(poses(:, 2));
    [params.mu_angle(end+1), params.sigma_angle(end+1)] = ...
      FitGaussianParameters(poses(:, 3));
  end
end

function params = LearnChildCPD(dataset, G, labels, poseIdx)
  params = struct(
               "theta", [],
               "mu_y", [],
               "sigma_y", [],
               "mu_x", [],
               "sigma_x", [],
               "mu_angle", [],
               "sigma_angle", []);

  for labelIdx=1:size(labels, 2)
    allTheta = [];

    examples = find(labels(:, labelIdx));
    poses = squeeze(dataset(examples, poseIdx, :));

    parentPoseIdx = G(poseIdx, 2);
    parentPoses = squeeze(dataset(examples, parentPoseIdx, :));

    [Theta sigma] = ...
      FitLinearGaussianParameters(poses(:, 1), parentPoses);
    Theta = [Theta(end), Theta(1:end-1)'];
    allTheta = [allTheta, Theta];
    params.sigma_y(end+1) = sigma;

    [Theta sigma] = ...
      FitLinearGaussianParameters(poses(:, 2), parentPoses);
    Theta = [Theta(end), Theta(1:end-1)'];
    allTheta = [allTheta, Theta];
    params.sigma_x(end+1) = sigma;

    [Theta sigma] = ...
      FitLinearGaussianParameters(poses(:, 3), parentPoses);
    Theta = [Theta(end), Theta(1:end-1)'];
    allTheta = [allTheta, Theta];
    params.sigma_angle(end+1) = sigma;

    params.theta(end+1, :) = allTheta;
  end
end
