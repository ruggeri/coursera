function [ForwardLogProbs, BackwardLogProbs] = ...
         RunForwardBackward(P, logEmissionProbs)
  NUM_POSES = size(logEmissionProbs, 1);
  NUM_CLASSES = length(P.c);

  logTransMatrix = log(P.transMatrix);

  ForwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
  ForwardLogProbs(1, :) = log(P.c) + logEmissionProbs(1, :);

  for poseIdx=2:NUM_POSES
    % Probability of state given observations up to poseIdx.
    ForwardLogProbs(poseIdx, :) = logsumexp((
      ForwardLogProbs(poseIdx - 1, :)'
      .+ logTransMatrix
      .+ logEmissionProbs(poseIdx, :))')';
  end

  BackwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
  % Redundant.
  BackwardLogProbs(NUM_POSES, :) = zeros(1, NUM_CLASSES);
  for poseIdx=fliplr(1:(NUM_POSES-1))
    % Probability of subsequence observations given state.
    BackwardLogProbs(poseIdx, :) = logsumexp(
      logTransMatrix .+
      logEmissionProbs(poseIdx + 1, :) .+
      BackwardLogProbs(poseIdx + 1, :))';
  end
end
