% Copyright (C) Daphne Koller, Stanford University, 2012

function EUF = CalculateExpectedUtilityFactor( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: A factor over the scope of the decision rule D from I that
  % gives the conditional utility given each assignment for D.var
  %
  % Note - We assume I has a single decision node and utility node.
  F = [I.RandomFactors I.UtilityFactors];

  eliminatedVars = setdiff(unique([F.var]), I.DecisionFactors(1).var);
  F = VariableElimination(F, eliminatedVars);

  f = F(1);
  for f2=F(2:end)
    f = FactorProduct(f, f2);
  end

  EUF = f;
end
