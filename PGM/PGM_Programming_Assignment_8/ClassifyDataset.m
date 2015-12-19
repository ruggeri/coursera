function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

  NUM_EXAMPLES = size(dataset, 1);
  NUM_CLASSES = length(P.c);

  if ndims(G) == 2
    G = repmat(G, 1, 1, NUM_CLASSES);
  end

  numCorrectPredictions = 0;
  for exampleIdx=1:NUM_EXAMPLES
    example = squeeze(dataset(exampleIdx, :, :));
    predictedClass = ClassifyExample(P, G, example);

    numCorrectPredictions += labels(exampleIdx, predictedClass);
  end

  accuracy = numCorrectPredictions / NUM_EXAMPLES;
  fprintf('Accuracy: %.2f\n', accuracy);
end

function classIdx = ClassifyExample(P, G, example)
  NUM_CLASSES = length(P.c);

  logProbs = [];
  for classIdx=1:NUM_CLASSES
    G_ = squeeze(G(:, :, classIdx));
    logProbs(end+1) = ComputeExampleLogProb(P, G_, classIdx, example);
  end

  [_, classIdx] = max(logProbs);
end
