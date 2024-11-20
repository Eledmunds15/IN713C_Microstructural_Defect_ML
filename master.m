%% Master Document
% Ethan L. Edmunds, Nov 2024

clear; close all; clc; % Clear the current environment

datasetPath = 'defect_dataset'; % Identify the path with the datasets

num_classes = 4; % Initialize the number of classes
filter_size = 5; % Select filter size for CNN
imageSize = [200 200]; % Selecting the image Size

aux_params = [imageSize, num_classes, filter_size]; % Create parameters to be used across all CNNs
baseline_hyper_params = [4, 10]; % Define hyperparameters for baseline test

[trainImds, subset_trainImds, valImds, valDlarray, validationLabels, classNames] = dataCollectAndAugment(datasetPath, imageSize); % Get the dataset, split into training and validation datasets, and augment them

%% Baseline Model Training

filename_baseline = "baseline_model_1";
[baselineModel, baselineModelInfo] = trainBaselineModel(trainImds, valImds, aux_params, baseline_hyper_params, filename_baseline); % Train the baseline dataset
[baselineConfMat, baselineAccuracy, baselineAccuracy_classes] = evaluateModel(baselineModel, baseline_hyper_params, valDlarray, validationLabels, classNames, "model_information", filename_baseline); % Evaluate the baseline model

%% Gridlayer Search Optimisation

num_layers = [3 4 5 6 7 8 9]; % number of layers to explore during grid search
num_filters = [10 20 30 40 50 60 70 80]; % number of filters to explore during grid search

[gs_accuracy, gs_best_accuracy, gs_best_hyperparameters] = gridLayerSearch(subset_trainImds, num_layers, num_filters, aux_params, valImds, valDlarray, validationLabels, classNames); % call gridlayer search

filename_gridsearch = "gridsearch_optimised_model_1";
[gridsearchOptimisedModel, gridsearchModelInfo] = trainBaselineModel(trainImds, valImds, aux_params, gs_best_hyperparams); % Train a model using full training data and gridsearch optimised hyperparameters
[gridsearchConfMat, gridsearchAccuracy, gridsearchAccuracy_classes] = evaluateModel(gridsearchOptimisedModel, baseline_hyper_params, valDlarray, validationLabels, classNames, "gridsearch_information", filename_gridsearch); % Evaluate the baseline model

%% Bayesian Optimisation
[bayesOptModel, bayesOptModelInfo] = bayesianOptimisation(trainImds, valImds, valDlarray, validationLabels, classNames, aux_params);
