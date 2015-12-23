function [ClassProb PairProb] = ...
         ExtractClassAndPairProbs(P,
                                  ForwardLogProbs,
                                  BackwardLogProbs,
                                  logEmissionProbs)
  NUM_POSES = size(logEmissionProbs, 1);
  NUM_CLASSES = length(P.c);

  logTransMatrix = log(P.transMatrix);

  ClassProb = zeros(NUM_POSES, NUM_CLASSES);
  PairProb = zeros(NUM_POSES-1, NUM_CLASSES ^ 2);
  for poseIdx=1:NUM_POSES
    logProbs = ForwardLogProbs(poseIdx, :) .+ ...
      BackwardLogProbs(poseIdx, :);

    % Normalize.
    logProbs -= logsumexp(logProbs);
    ClassProb(poseIdx, :) = exp(logProbs);
  end

  % Forward & Backward are correct as all ClassProb correct.
  for poseIdx=1:(NUM_POSES-1)
    logProbs = (
      ForwardLogProbs(poseIdx, :)' .+
      logTransMatrix .+
      logEmissionProbs(poseIdx + 1, :) .+
      BackwardLogProbs(poseIdx + 1, :));

    logProbs = reshape(logProbs, 1, NUM_CLASSES ^ 2);

    % Normalize.
    logProbs -= logsumexp(logProbs);

    PairProb(poseIdx, :) = exp(logProbs);
  end
end
