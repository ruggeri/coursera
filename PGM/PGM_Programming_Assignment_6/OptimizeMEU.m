% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

  MEU = 0;
  EUF = CalculateExpectedUtilityFactor(I);
  OptimalDecisionRule = D;

  indexToAssignmentsD = IndexToAssignment(1:length(D.val), D.card);

  i = 1;
  while i <= length(D.val)
    range = i:(i+D.card(1)-1);

    bestAssignment = i;
    bestUtils = EUF.val(i);

    for i2=range
      assignment = indexToAssignmentsD(i2, :);
      val = GetValueOfAssignment(EUF, assignment, D.var);
      if bestUtils < val
        bestAssignment = i2;
        bestUtils = val;
      end
    end

    OptimalDecisionRule.val(range) = 0;
    OptimalDecisionRule.val(bestAssignment) = 1;
    MEU += bestUtils;

    i += D.card(1);
  end
end
