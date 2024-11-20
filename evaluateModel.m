%% Create predictions using a neural network
% Ethan L. Edmunds, Nov 2024
function [confMat, overallAccuracy, classAccuracy] = evaluateModel(mdl, hyper_params, validation_images, validation_labels, classNames, parent_folder, filename)
    
    num_layers = hyper_params(1); % Initialize number of layers
    num_filters = hyper_params(2); % Initialize number of filters

    % create predictions using the model and the validation images
    [predictions, ~] = predict(mdl, validation_images);

    % use on hot decode to convert the predictions back into a categorical
    % array and find its transpose
    predicted_labels = transpose(onehotdecode(predictions, classNames, 1));

    confMat = confusionmat(validation_labels, predicted_labels); % confusion matrix

    totalCorrect = sum(diag(confMat)); % calculate the total number of correct predictions
    totalInstances = sum(confMat(:)); % calculate the total instances in the conf mat

    overallAccuracy = totalCorrect/totalInstances; % calculate the overall accuracy of the deep learning algorithm

    numClasses = size(confMat, 1); % Calculate the number of classes
    classAccuracy = zeros(numClasses); % Preallocate memory to hold class accuracies

    for i = 1: numClasses
        classAccuracy(i) = (confMat(i,i)/sum(confMat(i,:)))*100; % Diagonal element divided by row sum
    end

    confusionchart(confMat, classNames) % Create a confusion chart for the plot

    filename = filename + "_eval_" + string(num_layers) + "_" + string(num_filters); % Declare filename (filename + number of layers + number of filters)

    fullFileName = fullfile(parent_folder, filename);

    save(fullFileName,"overallAccuracy", "classAccuracy", "confMat");
    cons=[];

end