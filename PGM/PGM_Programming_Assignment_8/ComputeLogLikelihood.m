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

  NUM_EXAMPLES = size(dataset, 1); % number of examples
  NUM_CLASSES = length(P.c); % number of classes

  if ndims(G) == 2
    G = repmat(G, 1, 1, NUM_CLASSES);
  end

% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).

  logProb = 0.0;
  for dataIdx=1:NUM_EXAMPLES
    logProbs = [];
    for classIdx=1:NUM_CLASSES
      G_ = squeeze(G(:, :, classIdx));
      example = squeeze(dataset(dataIdx, :, :));
      logProbs(end+1) = ...
        ComputeExampleLogProb(P, G_, classIdx, example);
    end

    logProb += log(sum(exp(logProbs)));
  end
end
