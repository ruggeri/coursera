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
  NUM_CLASSES = size(labels, 2);

  if ndims(G) == 2
    G = repmat(G, 1, 1, NUM_CLASSES);
  end

  P.c = mean(labels);
  P.clg = [];
  for poseIdx=1:size(dataset, 2)
    P.clg(poseIdx) = LearnCPD(dataset, G, labels, poseIdx);
  end

  loglikelihood = ComputeLogLikelihood(P, G, dataset);
  fprintf('log likelihood: %f\n', loglikelihood);
end

function params = EmptyParams()
  params = struct(
               "theta", [],
               "mu_y", [],
               "sigma_y", [],
               "mu_x", [],
               "sigma_x", [],
               "mu_angle", [],
               "sigma_angle", []);
end

function params = LearnCPD(dataset, G, labels, poseIdx)
  params = EmptyParams();

  NUM_LABELS = size(labels, 2);
  for labelIdx=1:NUM_LABELS
    if G(poseIdx, 1, labelIdx) == 0
      params_ = LearnRootCPD(dataset, G, labels, poseIdx, labelIdx);
    else
      params_ = LearnChildCPD(dataset, G, labels, poseIdx, labelIdx);
    end

    params = MergeParams(params, params_);
  end
end

function params = MergeParams(params1, params2)
  params = struct(
               "theta", [params1.theta; params2.theta],
               "mu_y", [params1.mu_y, params2.mu_y],
               "sigma_y", [params1.sigma_y, params2.sigma_y],
               "mu_x", [params1.mu_x, params2.mu_x],
               "sigma_x", [params1.sigma_x, params2.sigma_x],
               "mu_angle", [params1.mu_angle, params2.mu_angle],
               "sigma_angle", [params1.sigma_angle, params2.sigma_angle]);
end

function params = LearnRootCPD(dataset, G, labels, poseIdx, labelIdx)
  params = EmptyParams();

  examples = find(labels(:, labelIdx));
  poses = squeeze(dataset(examples, poseIdx, :));

  [params.mu_y(end+1), params.sigma_y(end+1)] = ...
    FitGaussianParameters(poses(:, 1));
  [params.mu_x(end+1), params.sigma_x(end+1)] = ...
    FitGaussianParameters(poses(:, 2));
  [params.mu_angle(end+1), params.sigma_angle(end+1)] = ...
    FitGaussianParameters(poses(:, 3));
end

function params = LearnChildCPD(dataset, G, labels, poseIdx, labelIdx)
  params = EmptyParams();

  examples = find(labels(:, labelIdx));
  poses = squeeze(dataset(examples, poseIdx, :));

  parentPoseIdx = G(poseIdx, 2, labelIdx);
  parentPoses = squeeze(dataset(examples, parentPoseIdx, :));

  [Theta sigma] = ...
    FitLinearGaussianParameters(poses(:, 1), parentPoses);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_y(end+1) = sigma;

  [Theta sigma] = ...
    FitLinearGaussianParameters(poses(:, 2), parentPoses);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_x(end+1) = sigma;

  [Theta sigma] = ...
    FitLinearGaussianParameters(poses(:, 3), parentPoses);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_angle(end+1) = sigma;
end
