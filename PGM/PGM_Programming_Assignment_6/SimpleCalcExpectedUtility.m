% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);

  utilVars = U.var;
  eliminatedVars = setdiff(unique([F.var]), utilVars);

  F = VariableElimination(F, eliminatedVars);

  f = F(1);
  for f2=F(2:end)
    f = FactorProduct(f, f2);
  end

  % Weight utils by probabilities.
  U2 = FactorProduct(f, U);
  EU = sum(U2.val);
end
