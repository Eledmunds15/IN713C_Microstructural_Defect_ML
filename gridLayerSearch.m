%% Gridlayer Search Function
% Ethan L. Edmunds, Nov 2024
function [gs_accuracy, gs_best_accuracy, gs_best_hyperparameters] = gridlayerSearch(trainData, num_layers_arr, num_filters_arr, aux_params, valData, valImages, valLabels, classNames)
    
    gs_best_accuracy = 0; % initialize the best accuracy
    gs_accuracy = zeros(length(num_layers_arr)*length(num_filters_arr), 3); % preallocate memory to store array of accuracies
    gs_best_hyperparameters = [0, 0];
    iteration = 0;

    networkOptions = trainingOptions("adam", ...
        "MiniBatchSize", 128, ...
        "InitialLearnRate", 0.01, ...
        "MaxEpochs", 10, ...
        "Shuffle", "every-epoch", ...
        "ValidationData", valData, ...
        "Verbose", true, ...
        "ValidationPatience", 100, ...
        "Metrics", ["accuracy", "recall"], ...
        "OutputNetwork", "best-validation", ...
        "ValidationFrequency", 5, ...
        "Plots", "none", ...
        "ExecutionEnvironment", "parallel");

    for i = 1:length(num_layers_arr)
        for j = 1:length(num_filters_arr)

            iteration = iteration + 1; % increase iteration
            
            % current hyperparameters
            hyperParams_current = [num_layers_arr(i), num_filters_arr(j)];
            networkArchitecture = createNetworkModel(hyperParams_current, aux_params);

            fprintf("Number of Layers: " + string(hyperParams_current(1)) + " | Number of Filters: " + string(hyperParams_current(2)) + "\n\n");

            current_layers = num_layers_arr(i); % Store 
            current_filters = num_filters_arr(j); % Store

            % train the network using network generated using
            % createNetworkModel
            [gs_model, gs_modelInfo] = trainnet(trainData, networkArchitecture, 'crossentropy', networkOptions);
            close(findall(groot, 'Tag', 'NNET_CNN_TAININGPLOT_UIFIGURE'));

            filename = "gridsearch_iteration_" + string(hyperParams_current(1) + "_" + string(hyperParams_current(2))); % Create a file name to store information about the model
            parent_folder = "gridsearch_information";

            fullfileName = fullfile(parent_folder, filename); % Create the full file name to be stored in a folder

            [~, iter_overallAccuracy, ~] = evaluateModel(gs_model, hyperParams_current, valImages, valLabels, classNames, "gridsearch_information", filename);

            save(fullfileName, "gs_model", "gs_modelInfo", "iter_overallAccuracy", "current_layers", "current_filters");

            % extract the validation accuracy of the model
            gs_accuracy(iteration,:) = [hyperParams_current(1), hyperParams_current(2), iter_overallAccuracy];

            % store parameters if they yield better accuracy than the
            % previous iteration
            if gs_accuracy(iteration,3) > gs_best_accuracy
                
                gs_best_accuracy = iter_overallAccuracy; % store the best accuracy
                gs_best_hyperparameters = hyperParams_current; % store the hyperparameters

            end
        end
    end

end