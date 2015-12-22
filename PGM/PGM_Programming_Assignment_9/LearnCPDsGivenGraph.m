function P = LearnCPDsGivenGraph(dataset, G, labelWeights)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

  N = size(dataset, 1);
  NUM_POSES = size(dataset, 2);
  NUM_LABELS = size(labelWeights, 2);

  if ndims(G) == 2
    G = repmat(G, 1, 1, NUM_LABELS);
  end

  P.c = mean(labelWeights);
  P.clg = [];
  for poseIdx=1:NUM_POSES
    P.clg(poseIdx) = LearnCPD(dataset, G, labelWeights, poseIdx);
  end
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

function params = LearnCPD(dataset, G, labelWeights, poseIdx)
  params = EmptyParams();

  NUM_LABELS = size(labelWeights, 2);
  for labelIdx=1:NUM_LABELS
    if G(poseIdx, 1, labelIdx) == 0
      params_ = LearnRootCPD(dataset, G, labelWeights, poseIdx, labelIdx);
    else
      params_ = LearnChildCPD(dataset, G, labelWeights, poseIdx, labelIdx);
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

function params = LearnRootCPD(dataset, G, labelWeights, poseIdx, labelIdx)
  params = EmptyParams();

  poses = squeeze(dataset(:, poseIdx, :));
  weights = labelWeights(:, labelIdx);

  [params.mu_y(end+1), params.sigma_y(end+1)] = ...
    FitG(poses(:, 1), weights);
  [params.mu_x(end+1), params.sigma_x(end+1)] = ...
    FitG(poses(:, 2), weights);
  [params.mu_angle(end+1), params.sigma_angle(end+1)] = ...
    FitG(poses(:, 3), weights);
end

function params = LearnChildCPD(dataset, G, labelWeights, poseIdx, labelIdx)
  params = EmptyParams();

  poses = squeeze(dataset(:, poseIdx, :));
  weights = labelWeights(:, labelIdx);

  parentPoseIdx = G(poseIdx, 2, labelIdx);
  parentPoses = squeeze(dataset(:, parentPoseIdx, :));

  [Theta sigma] = FitLG(poses(:, 1), parentPoses, weights);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_y(end+1) = sigma;

  [Theta sigma] = FitLG(poses(:, 2), parentPoses, weights);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_x(end+1) = sigma;

  [Theta sigma] = FitLG(poses(:, 3), parentPoses, weights);
  Theta = [Theta(end), Theta(1:end-1)'];
  params.theta = [params.theta, Theta];
  params.sigma_angle(end+1) = sigma;
end
