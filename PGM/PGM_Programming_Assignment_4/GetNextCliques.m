%GETNEXTCLIQUES Find a pair of cliques ready for message passing
%   [i, j] = GETNEXTCLIQUES(P, messages) finds ready cliques in a given
%   clique tree, P, and a matrix of current messages. Returns indices i and j
%   such that clique i is ready to transmit a message to clique j.
%
%   We are doing clique tree message passing, so
%   do not return (i,j) if clique i has already passed a message to clique j.
%
%	 messages is a n x n matrix of passed messages, where messages(i,j)
% 	 represents the message going from clique i to clique j. 
%   This matrix is initialized in CliqueTreeCalibrate as such:
%      MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);
%
%   If more than one message is ready to be transmitted, return 
%   the pair (i,j) that is numerically smallest. If you use an outer
%   for loop over i and an inner for loop over j, breaking when you find a 
%   ready pair of cliques, you will get the right answer.
%
%   If no such cliques exist, returns i = j = 0.
%
%   See also CLIQUETREECALIBRATE
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function [i, j] = GetNextCliques(P, messages)
  numCliques = length(P.cliqueList);
  for i=1:numCliques
    for j=1:numCliques
      if !P.edges(i, j)
        % No connection!
        continue;
      end

      if !isempty(messages(i, j).var)
        % already sent this message!
        continue;
      end

      shouldSend = true;
      neighbors = find(P.edges(i, :));
      for neighbor=neighbors
        if neighbor == j
          continue;
        end

        if isempty(messages(neighbor, i).var)
          % Not ready, not all messages received
          shouldSend = false;
          break;
        end
      end

      if shouldSend
        return;
      end
    end
  end

  % No valid messages to send!
  i = 0;
  j = 0;

  return;
