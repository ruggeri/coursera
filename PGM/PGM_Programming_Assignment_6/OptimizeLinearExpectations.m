% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.
  D = I.DecisionFactors(1);
  MEU = 0;
  EUF = computeEUF(I);
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

function EUF = computeEUF(I)
  EUF = struct("var", [], "card", [], "val", []);

  for utilityIdx=1:length(I.UtilityFactors)
    I2 = struct("RandomFactors", I.RandomFactors,
                "DecisionFactors", I.DecisionFactors,
                "UtilityFactors", I.UtilityFactors(utilityIdx));
    EUF = FactorSum(EUF, CalculateExpectedUtilityFactor(I2));
  end
end
