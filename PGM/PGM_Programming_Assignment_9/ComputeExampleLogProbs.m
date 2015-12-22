function logProbs = ComputeExampleLogProbs(P, G, example)
  NUM_CLASSES = length(P.c);
  logProbs = zeros(1, NUM_CLASSES);

  for classIdx=1:NUM_CLASSES
    if ndims(G) == 3
      G_ = squeeze(G(:, :, classIdx));
    else
      G_ = G;
    end

    logProbs(classIdx) = ComputeExampleLogProb(P, G_, classIdx, example);
  end
end
