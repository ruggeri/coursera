%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numVars = 0;
for i=1:length(F)
  numVars = max(numVars, max(F(i).var));
  F(i) = ObserveEvidence(F(i), E);
end

% You fucks. CreateCliqueTree expects evidence *in a different format*
% than the usual 2D format. WTF?!
P = CreateCliqueTree(F, []);
P = CliqueTreeCalibrate(P, isMax);

M = repmat(struct('var', [], 'card', [], 'val', []), 1, numVars);
for cliqueIdx=1:length(P.cliqueList)
  c = P.cliqueList(cliqueIdx);
  for varIdx=1:length(c.var)
    var = c.var(varIdx);
    if !isempty(M(var).val)
      % Don't recompute!
      continue
    end

    M(var) = FactorMarginalization(c, setdiff(c.var, var));
    M(var).val /= sum(M(var).val);
  end
end

end
