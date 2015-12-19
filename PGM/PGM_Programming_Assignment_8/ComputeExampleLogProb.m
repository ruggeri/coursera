function logProb = ComputeExampleLogProb(P, G, classIdx, example)
  NUM_POSES = size(example, 1);

  logProb = log(P.c(classIdx));

  for poseIdx=1:NUM_POSES
    if G(poseIdx, 1) == 0
      logProb += ComputeRootLogProb(P, G, classIdx, example, poseIdx);
    else
      logProb += ComputeChildLogProb(P, G, classIdx, example, poseIdx);
    end
  end
end
