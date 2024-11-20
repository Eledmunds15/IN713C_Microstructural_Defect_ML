%% Data Collection Script
% Ethan L. Edmunds, Nov 2024
function [trainData, subset_trainData, valData, augImagesVal, validationLabels, classNames] = dataCollectAndAugment(path, imageSize) 
    
    % Get the folder path and output number of files in the dataset
    imageFolder = path;
    imds = imageDatastore(imageFolder, "IncludeSubfolders", true, "LabelSource", "foldernames"); % Create a dataset from the input folders

    tbl_count = countEachLabel(imds); % Count each label inside of the dataset
    datasetSize = sum(tbl_count{:,"Count"}); % Output the total size of the dataset
    fprintf("The number of defects in the dataset is " + string(datasetSize) + "\n");

    % Create the training and validation datasets
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 0.3, "randomized");

    tbl_count_train = countEachLabel(imdsTrain) % Count each label inside of the train dataset
    datasetSizeTrain = sum(tbl_count_train{:,"Count"}); % Output the total size of the train dataset
    fprintf("The number of defects in the dataset is " + string(datasetSizeTrain) + "\n");

    tbl_count_val = countEachLabel(imdsVal) % Count each label inside of the validation dataset
    datasetSizeVal = sum(tbl_count_val{:,"Count"}); % Output the total size of the validation dataset
    fprintf("The number of defects in the dataset is " + string(datasetSizeVal) + "\n");

    imageAugmenter = imageDataAugmenter(..., % Define the image augmentation configuration
        "RandXReflection", true, ...
        "RandYReflection", true, ...
        "RandRotation", [0, 360], ...
        "RandScale", [0.9 1.1]);

    augImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, "DataAugmentation", imageAugmenter); % Resize all the images of the train dataset and apply data augmentation
    augImdsVal = augmentedImageDatastore(imageSize, imdsVal, "DataAugmentation", imageAugmenter); % Resize all the images of the validation dataset and apply data augmentation

    augDataVal = readall(augImdsVal); % Turning the augmented validation dataset into a table

    % Create a cell array of validation dataset to be used with dlarray objects
    imagesVal = augDataVal{:, 1};
    imagesVal = cat(4, imagesVal{:});

    validationLabels = augDataVal{:,2}; % Get the full array of validation labels
    classNames = categories(validationLabels); % Identify the categories of validation labels

    augImagesVal = dlarray(single(imagesVal), 'SSCB'); % Turn the validataion table of images into a dlarray

    fprintf("Dataset collection, processing & augmentation completed!\n\n")

    % Assign outputs
    trainData = augImdsTrain;
    valData = augImdsVal;

    subset_train_index = randsample(1:datasetSizeTrain, int32(datasetSizeTrain*0.4));

    subset_train = subset(imdsTrain, subset_train_index);

    subset_trainData = augmentedImageDatastore(imageSize, subset_train, "DataAugmentation", imageAugmenter);

end