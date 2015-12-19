function [P G logProb] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

  NUM_EXAMPLES = size(dataset, 1);
  NUM_POSES = size(dataset, 2);
  NUM_CLASSES = size(labels, 2);

  % Initialization. Set everyone child of first vertex. Basically Naive
  % Bayes.
  G = zeros(NUM_POSES, 2, NUM_CLASSES); % graph structures to learn
  for classIdx=1:NUM_CLASSES
    G(2:end, :, classIdx) = ones(NUM_POSES - 1, 2);
  end

  % Estimate graph structure for each class
  for classIdx=1:NUM_CLASSES
    examples = find(labels(:, classIdx));
    dataset_ = squeeze(dataset(examples, :, :));

    [A, _] = LearnGraphStructure(dataset_);
    G(:, :, classIdx) = ConvertAtoG(A);
  end

  % Estimate parameters
  [P, logProb] = LearnCPDsGivenGraph(dataset, G, labels);

  fprintf('log likelihood: %f\n', logProb);
end
