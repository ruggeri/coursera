function [A W] = LearnGraphStructure(dataset)

% Input:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
%
% Output:
% A: maximum spanning tree computed from the weight matrix W
% W: 10 x 10 weight matrix, where W(i,j) is the mutual information between
%    node i and j.
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

  NUM_EXAMPLES = size(dataset, 1);
  NUM_POSES = size(dataset, 2);
  NUM_COMPONENTS = size(dataset,3);

  W = zeros(NUM_POSES, NUM_POSES);
  for poseIdx1=1:NUM_POSES
    poses1 = squeeze(dataset(:, poseIdx1, :));
    for poseIdx2=(poseIdx1):NUM_POSES
      poses2 = squeeze(dataset(:, poseIdx2, :));

      w = GaussianMutualInformation(poses1, poses2);
      W(poseIdx1, poseIdx2) = w;
      W(poseIdx2, poseIdx1) = w;
    end
  end

  % Compute maximum spanning tree
  A = MaxSpanningTree(W);
end
