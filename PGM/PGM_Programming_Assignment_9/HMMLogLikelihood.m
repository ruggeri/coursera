function logProbs = HMMLogLikelihood(actionData, poseData, P, G)
  NUM_ACTIONS = length(actionData);
  NUM_CLASSES = length(P.c);
  NUM_POSES = length(poseData);

  P2 = P;
  P2.c = ones(NUM_CLASSES, 1);
  logEmissionProbs = zeros(NUM_POSES, NUM_CLASSES);
  for poseIdx=1:NUM_POSES
    example = squeeze(poseData(poseIdx, :, :));
    logEmissionProbs(poseIdx, :) = ComputeExampleLogProbs(P2, G, example);
  end

  logProbs = zeros(NUM_ACTIONS, 1);
  for actionIdx=1:NUM_ACTIONS
    poseIdxs = actionData(actionIdx).marg_ind;

    logEmissionProbs_ = logEmissionProbs(poseIdxs, :);
    [ForwardLogProbs, _] = RunForwardBackward(P, logEmissionProbs_);

    logProbs(actionIdx) = logsumexp(ForwardLogProbs(end, :));
  end
end
