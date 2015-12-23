function [ClassProb PairProb logLikelihood] = ...
         BaumWelch(P, actionData, poseData, logEmissionProbs)
  NUM_ACTIONS = length(actionData);
  NUM_CLASSES = length(P.c);
  NUM_POSES = size(poseData, 1);
  NUM_TRANSITIONS = NUM_POSES - NUM_ACTIONS;

  ClassProb = zeros(NUM_POSES, NUM_CLASSES);
  PairProb = zeros(NUM_TRANSITIONS, NUM_CLASSES ^ 2);
  logLikelihood = 0;

  logTransMatrix = log(P.transMatrix);

  for actionIdx=1:NUM_ACTIONS
    action = actionData(actionIdx);
    poseIdxs = action.marg_ind;

    NUM_POSES = length(poseIdxs);

    ForwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
    ForwardLogProbs(1, :) = log(P.c) + logEmissionProbs(poseIdxs(1), :);

    for poseIdxIdx=2:NUM_POSES
      % Probability of state given observations up to poseIdxIdx.
      ForwardLogProbs(poseIdxIdx, :) = logsumexp((
        ForwardLogProbs(poseIdxIdx - 1, :)'
        .+ logTransMatrix
        .+ logEmissionProbs(poseIdxs(poseIdxIdx), :))')';
    end

    BackwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
    BackwardLogProbs(NUM_POSES, :) = ones(1, NUM_CLASSES);
    for poseIdxIdx=fliplr(1:(NUM_POSES-1))
      % Probability of subsequence observations given state.
      BackwardLogProbs(poseIdxIdx, :) = logsumexp(
        logTransMatrix .+
        logEmissionProbs(poseIdxs(poseIdxIdx + 1), :) .+
        BackwardLogProbs(poseIdxIdx + 1, :))';
    end

    for poseIdxIdx=1:NUM_POSES
      logProbs = ForwardLogProbs(poseIdxIdx, :) .+ ...
        BackwardLogProbs(poseIdxIdx, :);

      % Accumulate logLikelihood of the data.
      if poseIdxIdx == 1
        logLikelihood += logsumexp(logProbs);
      end

      % Normalize.
      logProbs -= logsumexp(logProbs);
      ClassProb(poseIdxs(poseIdxIdx), :) = exp(logProbs);
    end

    % Forward & Backward are correct as all ClassProb correct.
    for poseIdxIdx=1:(NUM_POSES-1)
      logProbs = (
        ForwardLogProbs(poseIdxIdx, :)' .+
        logTransMatrix .+
        logEmissionProbs(poseIdxs(poseIdxIdx) + 1, :) .+
        BackwardLogProbs(poseIdxIdx + 1, :));

      logProbs = reshape(logProbs, 1, NUM_CLASSES ^ 2);

      % Normalize.
      logProbs -= logsumexp(logProbs);

      PairProb(action.pair_ind(poseIdxIdx), :) = exp(logProbs);
    end
  end
end
