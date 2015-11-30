%BLOCKLOGDISTRIBUTION
%
%   LogBS = BlockLogDistribution(V, G, F, A) returns the log of a
%   block-sampling array (which contains the log-unnormalized-probabilities of
%   selecting each label for the block), given variables V to block-sample in
%   network G with factors F and current assignment A.  Note that the variables
%   in V must all have the same dimensionality.
%
%   Input arguments:
%   V -- an array of variable names
%   G -- the graph with the following fields:
%     .names - a cell array where names{i} = name of variable i in the graph 
%     .card - an array where card(i) is the cardinality of variable i
%     .edges - a matrix such that edges(i,j) shows if variables i and j 
%              have an edge between them (1 if so, 0 otherwise)
%     .var2factors - a cell array where var2factors{i} gives an array where the
%              entries are the indices of the factors including variable i
%   F -- a struct array of factors.  A factor has the following fields:
%       F(i).var - names of the variables in factor i
%       F(i).card - cardinalities of the variables in factor i
%       F(i).val - a vectorized version of the CPD for factor i (raw probability)
%   A -- an array with 1 entry for each variable in G s.t. A(i) is the current
%       assignment to variable i in G.
%
%   Each entry in LogBS is the log-probability that that value is selected.
%   LogBS is the P(V | X_{-v} = A_{-v}, all X_i in V have the same value), where
%   X_{-v} is the set of variables not in V and A_{-v} is the corresponding
%   assignment to these variables consistent with A.  In the case that |V| = 1,
%   this reduces to Gibbs Sampling.  NOTE that exp(LogBS) is not normalized to
%   sum to one at the end of this function (nor do you need to worry about that
%   in this function).
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function LogBS = BlockLogDistribution(V, G, F, A)
if length(unique(G.card(V))) ~= 1
    disp('WARNING: trying to block sample invalid variable set');
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute LogBS by multiplying (adding in log-space) in the correct values from
% each factor that includes some variable in V.  
%
% NOTE: As this is called in the innermost loop of both Gibbs and Metropolis-
% Hastings, you should make this fast.  You may want to make use of
% G.var2factors, repmat,unique, and GetValueOfAssignment.
%
% Also you should have only ONE for-loop, as for-loops are VERY slow in matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find factors involving the variables under consideration.
F = F([G.var2factors{V}]);

% Construct a factor we will return involving only the free vars.
res = struct('var', V, 'card', G.card(V), 'val', []);
res.val = ones(1, prod(res.card));

[varIndexs, assignments] = buildMaps(res, F, A);
LogBS = combineFactors(res, F, varIndexs, assignments);
LogBS = finalizeLogBS(LogBS, G, V);
endfunction

function [varIndexs, assignments] = buildMaps(res, F, A)
  % Record the index of each of the vars.
  varIndexs = [];
  varIndexs(res.var) = 1:length(res.var);

  % Calculate once a map of idx->assignment.
  % TODO: This commented code breaks when V consists of multiple vars?
  %assignments = floor(((1:length(res.val))' - 1) ./ ...
  %                    cumprod([1, res.card(1:end-1)])) + 1;
  assignments = IndexToAssignment(1:length(res.val), res.card);

  % Extend varIndexs and assignments maps with unassigned vars.
  unsampledVars = setdiff([F.var], res.var);
  numSampledVars = length(res.var);
  numUnsampledVars = length(unsampledVars);
  unsampledVarsIndxs = ...
    (numSampledVars+1):(numSampledVars+numUnsampledVars);
  varIndexs(unsampledVars) = unsampledVarsIndxs;
  assignments(:, unsampledVarsIndxs) = ...
    repmat(A(unsampledVars), size(assignments, 1), 1);
endfunction

function LogBS = combineFactors(res, F, varIndexs, assignments)
  % Iterate through factors. We'll need to add each in to res.
  for factorIdx=1:length(F)
    f = F(factorIdx);

    % Get only the relevant portions of the assignments.
    reducedAssignments = assignments(:, varIndexs(f.var));
    % Find where these assignments live in f.
    indexs = cumprod([1, f.card(1:(end - 1))]) * (reducedAssignments' - 1);
    indexs += 1;

    res.val = res.val .* f.val(indexs);
  end

  LogBS = res.val;
endfunction

function LogBS = finalizeLogBS(LogBS, G, V)
  % This bullshit calculates a diagonal. Fuck your 1-based indices.
  card = G.card(V(1));
  q = sum(card .^ (0:(length(V)-1)));
  idxs = ((0:(card-1))*q)+1;
  % Just take out the assignments to A that assign everyone the same val.
  LogBS = LogBS(idxs);

  % Finally, take the log.
  LogBS = log(LogBS);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Re-normalize to prevent underflow when you move back to probability
  % space
  LogBS = LogBS - min(LogBS);
endfunction
