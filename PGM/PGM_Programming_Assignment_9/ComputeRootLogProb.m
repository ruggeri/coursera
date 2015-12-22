function logProb = ComputeRootLogProb(P, G, classIdx, example, poseIdx)
  pose = example(poseIdx, :);

  params = P.clg(poseIdx);

  logProbs = [];
  logProbs(end+1) = ...
    lognormpdf(pose(1), params.mu_y(classIdx), params.sigma_y(classIdx));
  logProbs(end+1) = ...
    lognormpdf(pose(2), params.mu_x(classIdx), params.sigma_x(classIdx));
  logProbs(end+1) = ...
    lognormpdf(pose(3), params.mu_angle(classIdx), params.sigma_angle(classIdx));

  logProb = sum(logProbs);
end
