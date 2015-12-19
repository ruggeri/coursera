function logProb = ComputeChildLogProb(P, G, classIdx, example, poseIdx)
  parentPoseIdx = G(poseIdx, 2);
  parentPose = example(parentPoseIdx, :)(:);
  pose = example(poseIdx, :);

  % Calculate means.
  theta = P.clg(poseIdx).theta(classIdx, :);
  mu_y = theta(1:4)  * [1; parentPose];
  mu_x = theta(5:8)  * [1; parentPose];
  mu_angle = theta(9:12) * [1; parentPose];

  % Extract variances.
  params = P.clg(poseIdx);

  logProbs = [];
  logProbs(end+1) = lognormpdf(pose(1), mu_y, params.sigma_y(classIdx));
  logProbs(end+1) = lognormpdf(pose(2), mu_x, params.sigma_x(classIdx));
  logProbs(end+1) = lognormpdf(pose(3), mu_angle, params.sigma_angle(classIdx));

  logProb = sum(logProbs);
end
