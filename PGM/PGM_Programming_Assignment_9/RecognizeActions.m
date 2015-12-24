% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = ...
         RecognizeActions(datasetTrain, datasetTest, G, maxIter)

  % INPUTS
  % datasetTrain: dataset for training models, see PA for details
  % datasetTest: dataset for testing models, see PA for details
  % G: graph parameterization as explained in PA decription
  % maxIter: max number of iterations to run for EM.

  % OUTPUTS
  % accuracy: recognition accuracy, defined as
  % (#correctly classified examples / #total examples)
  % predicted_labels: N x 1 vector with the predicted labels for
  % each of the instances in datasetTest, with N being the number
  % of unknown test instances

  % Train a model for each action
  % Note that all actions share the same graph parameterization and
  % number of max iterations.

  models = struct("c", {},
                  "clg", {},
                  "transMatrix", {});

  NUM_ACTION_TYPES = length(datasetTrain);
  for actionTypeIdx=1:NUM_ACTION_TYPES
    actionTypeData = datasetTrain(actionTypeIdx);
    P = EM_HMM(actionTypeData.actionData,
               actionTypeData.poseData,
               G,
               actionTypeData.InitialClassProb,
               actionTypeData.InitialPairProb,
               maxIter);

    models(end+1) = P;
  end

  % Classify each of the instances in datasetTrain
  % Compute and return the predicted labels and accuracy
  % Accuracy is defined as (#correctly classified examples / #total examples)
  % Note that all actions share the same graph parameterization

  NUM_ACTIONS = length(datasetTest.actionData);
  logProbs = zeros(NUM_ACTIONS, NUM_ACTION_TYPES);
  for actionTypeIdx=1:NUM_ACTION_TYPES
    logProbs(:, actionTypeIdx) = ...
      HMMLogLikelihood(datasetTest.actionData,
                       datasetTest.poseData,
                       models(actionTypeIdx),
                       G);
  end

  [_, predicted_labels] = max(logProbs, [], 2);
  accuracy = ...
    sum(datasetTest.labels == predicted_labels) / length(datasetTest.labels)
end
