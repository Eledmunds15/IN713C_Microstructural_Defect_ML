%% Create Baseline Model
% Ethan L. Edmunds, Nov 2024
function networkArchitecture = createNetworkModel(hyper_params, aux_params);
    
    num_blocks = hyper_params(1); % number of layers to be used
    num_filters = hyper_params(2); % number of filters to be used

    imageSize = aux_params(1:2); % image size to be in the network
    num_classes = aux_params(3); % number of classes to be used
    filter_size = aux_params(4); % size of filters to be used in network design

    networkArchitecture = [imageInputLayer([imageSize 1])]; % Create the input layer for the CNN

    % Create the blocks for the CNN architecture each of which contain
    % convolution -> batch normalisation -> relu -> max pooling layers
    for i = 1:(num_blocks)
        networkArchitecture = [networkArchitecture
            convolution2dLayer(filter_size, num_filters, "Padding", "same")
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2, "Stride", 2, "Padding", "same")
        ];
    end

    % Output layers
    networkArchitecture = [networkArchitecture
        dropoutLayer(0.3)
        fullyConnectedLayer(num_classes)
        softmaxLayer
    ];

end