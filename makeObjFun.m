function ObjFcn = makeObjFun(trainData, augImagesVal, labelsVal, classNames, networkOptions, aux_params, imageSize)

    ObjFcn = @valErrorFun;

    function [valError,cons,fileName] = valErrorFun(optVars)

        hyperParams = [optVars.Layers optVars.Filters];
        
        test_initialLearnRate = optVars.InitialLearnRate;
        test_learnRateDropFactor = optVars.LearnRateDropFactor;
        test_learnRateDropPeriod = optVars.LearnRateDropPeriod;

        % create layers using the bayesian optimised hyperparameters
        layers = createNetworkModel(hyperParams, aux_params);

        % change network options to match test options
        networkOptions.InitialLearnRate = test_initialLearnRate;
        networkOptions.LearnRateDropPeriod = test_learnRateDropPeriod;
        networkOptions.LearnRateDropFactor = test_learnRateDropFactor;
        tic;    % start the timer before training

        % train network
        trainedNet = trainnet(trainData, layers, 'crossentropy', networkOptions);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
        bayes_training_time = toc;    % Stop the timer after training
        clear tic;

        % classify the validation output using the trained network
        bayes_pred = createPredictions(trainedNet, augImagesVal, classNames);

        parent_folder = "bayesian_optimisation_information";
        fileName = num2str(hyperParams(1)) + "_" + num2str(hyperParams(2)) + ".mat";

        [predictions, ~] = predict(mdl, validation_images);

        % use one hot decode to conver the predictions back into a categorical
        % array and find it's transpose
        predictions = transpose(onehotdecode(predictions, classNames, 1));

        accuracy = mean(predictions == labelsVal);

        % calculate the validation error
        valError = 1 - accuracy;

        if ~exist(parent_folder, 'dir')
            mkdir(parent_folder)
        end

        % save the trained network as a '.mat' file using the validation
        % error as the filename
        fullFileName = fullfile(parent_folder, fileName);
        save(fullFileName,'trainedNet','valError', "test_initialLearnRate","test_learnRateDropPeriod","test_learnRateDropFactor","hyperParams",'networkOptions',...
            'bayes_training_time');
        cons=[];
    end
end