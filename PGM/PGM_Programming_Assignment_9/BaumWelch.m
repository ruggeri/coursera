function [ClassProb PairProb logLikelihood] = ...
         BaumWelch(P, actionData, poseData, logEmissionProbs)
  NUM_ACTIONS = length(actionData);
  NUM_CLASSES = length(P.c);
  NUM_POSES = size(poseData, 1);
  NUM_TRANSITIONS = NUM_POSES - NUM_ACTIONS;

  ClassProb = zeros(NUM_POSES, NUM_CLASSES);
  PairProb = zeros(NUM_TRANSITIONS, NUM_CLASSES ^ 2);
  logLikelihood = 0;

  for actionIdx=1:NUM_ACTIONS
    action = actionData(actionIdx);
    poseIdxs = action.marg_ind;
    pairIdxs = action.pair_ind;

    NUM_POSES = length(poseIdxs);

    logEmissionProbs_ = logEmissionProbs(poseIdxs, :);
    [ForwardLogProbs, BackwardLogProbs] = ...
      RunForwardBackward(P, logEmissionProbs_);

    [ClassProb(poseIdxs, :), PairProb(pairIdxs, :)] = ...
      ExtractClassAndPairProbs(P,
                               ForwardLogProbs,
                               BackwardLogProbs,
                               logEmissionProbs_);

    % Accumulate logLikelihood of the data.
    logLikelihood += logsumexp(ForwardLogProbs(NUM_POSES, :));
    % TODO: Why is this necessary??
    logLikelihood += NUM_POSES - 1;
  end
end
