function [ClassProb PairProb] = BaumWelch(P, actionData, poseData, logEmissionProbs)
  NUM_ACTIONS = size(actionData, 1);
  NUM_CLASSES = length(P.c);
  NUM_POSES = size(poseData, 1);
  NUM_TRANSITIONS = NUM_POSES - NUM_ACTIONS;

  ClassProb = zeros(NUM_POSES, NUM_CLASSES);
  PairProb = zeros(NUM_TRANSITIONS, NUM_CLASSES ^ 2);

  for actionIdx=1:NUM_ACTIONS
    action = actionData(actionIdx);
    poseIdxs = action.marg_ind;

    NUM_POSES = length(poseIdxs);

    ForwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
    ForwardLogProbs(1, :) = log(P.c) + logEmissionProbs(poseIdxs(1), :);

    for poseIdxIdx=2:NUM_POSES
      % Probability of state given observation and preceeding data.
      ForwardLogProbs(poseIdxIdx, :) = ...
        sum(ForwardLogProbs(poseIdxIdx - 1, :)' .+ P.transMatrix) .+ ...
        logEmissionProbs(poseIdxs(poseIdxIdx), :);

      % Normalize.
      ForwardLogProbs(poseIdxIdx, :) -= ...
        logsumexp(ForwardLogProbs(poseIdxIdx, :));
    end

    BackwardLogProbs = zeros(NUM_POSES, NUM_CLASSES);
    BackwardLogProbs(NUM_POSES, :) = ones(1, NUM_CLASSES);
    for poseIdxIdx=fliplr(1:(NUM_POSES-1))
      % Probability of state given observation and following data.
      BackwardLogProbs(poseIdxIdx, :) = ...
        sum(BackwardLogProbs(poseIdxIdx + 1, :) .+ P.transMatrix, 2)' .+ ...
        logEmissionProbs(poseIdxs(poseIdxIdx), :);

      % Normalize.
      BackwardLogProbs(poseIdxIdx, :) -= ...
        logsumexp(BackwardLogProbs(poseIdxIdx, :));
    end

    for poseIdxIdx=1:NUM_POSES
      logProbs = ForwardLogProbs(poseIdxIdx, :) .+ ...
        BackwardLogProbs(poseIdxIdx, :);
      logProbs -= logsumexp(logProbs);
      ClassProb(poseIdxs(poseIdxIdx), :) = exp(logProbs);
    end
  end
end
