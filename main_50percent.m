clc;
clear;
close all;

% Test Image Path
TestPath = 'TestData';
if ~exist(TestPath, 'dir')
    mkdir(TestPath)
end

% Read files form pc. 
[file,path,indx] = uigetfile('./Input/*.jpg;*.jpeg;*.bmp',... 
                                    'Select an Input Image File');
if isequal(file,0)
   disp('User selected Cancel')
else
   disp(['Selected File Name: ', file])
   delete './TestData/*.jpg';
   In_Img = imread([path,file]);
   
   imwrite(In_Img,['./TestData/',file]);
end

% In_Img = imread([Path,File]);
% figure; imshow(I); title('Input Test Image');

% Image Resize
InImg = In_Img;
Re_Img = imresize(InImg, [227 227]);
In_Img = InImg;
figure; imshow(Re_Img); title(['Input Image: ',(file)]);

% Gray Conversion
% Get the dimensions of the image.  
[rows, columns, no_of_band] = size(InImg);
if isequal (no_of_band,3)
	% Convert it to gray scale 
	Gr_Img = rgb2gray(InImg);
    Gr_Img = imresize(Gr_Img,[227 227]);
else
    Gr_Img = imresize(InImg,[227 227]);
end
figure; imshow(Gr_Img); title('Gray  Image');

% Filter - Preprocessing
InImg = double(Re_Img);
Gs=fspecial('gaussian');
[rows1, columns1, no_of_band1] = size(InImg);
if isequal (no_of_band1,3)
	% Convert it to gray scale 
	In_fil(:,:,1)=imfilter(double(InImg(:,:,1)),Gs);
    In_fil(:,:,2)=imfilter(double(InImg(:,:,2)),Gs);
    In_fil(:,:,3)=imfilter(double(InImg(:,:,3)),Gs);

else
    In_fil=imfilter(double(InImg),Gs);
end
figure; imshow(uint8(In_fil)); title('Preprocessed Image');

% Binary Otsu Segmentation
% InImg = rgb2gray(uint8(In_fil));
InImg = uint8(In_fil);

% Specify initial contour location
mask = zeros(size(InImg));
mask(25:end-25,25:end-25) = 1;

% TestImg
TestImg = imageDatastore(TestPath,...
            'LabelSource',...
            'foldernames',...
            'IncludeSubfolders',true);

% Load pretrained network
net = resnet50();
lgraph = layerGraph(net);
netName = "squeezenet";
net1 = eval(netName);

inputSize = net1.Layers(1).InputSize(1:2);
classes = net1.Layers(end).Classes;
layerName = activationLayerName(netName);

% h = figure('Units','normalized','Position',[0.05 0.05 0.9 0.8],'Visible','on');

[rows2, columns2, no_of_band2] = size(Re_Img);
if isequal (no_of_band2,3)
	In_Img1 = Re_Img;
else
    In_Img1(:,:,1)= Re_Img;
    In_Img1(:,:,2)= Re_Img;
    In_Img1(:,:,3)= Re_Img;
end

im = In_Img1;
imResized = imresize(im,[inputSize(1), NaN]);
imageActivations = activations(net1,imResized,layerName);

   
scores = squeeze(mean(imageActivations,[1 2]));
if netName ~= "squeezenet"
    fcWeights = net1.Layers(end-2).Weights;
    fcBias = net1.Layers(end-2).Bias;
    scores =  fcWeights*scores + fcBias;
    [~,classIds] = maxk(scores,3);              
    weightVector = shiftdim(fcWeights(classIds(1),:),-1);
    classActivationMap = sum(imageActivations.*weightVector,3);
else
    [~,classIds] = maxk(scores,3);
    classActivationMap = imageActivations(:,:,classIds(1));
end
scores = exp(scores)/sum(exp(scores));
maxScores = scores(classIds);
labels = classes(classIds);

figure;
J=inimshow(im,classActivationMap);
title('Segmentation Mask');

% Inspect the first layer
net.Layers(1);
% Inspect the last layer
net.Layers(end);
% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames);

% Number of categories
numClasses = 2;
% New Learnable Layer
newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
   
% Replacing the last layers with new layers
lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
newsoftmaxLayer = softmaxLayer('Name','new_softmax');
lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    
% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
% image features are extracted using activations.
validationTestSet = augmentedImageDatastore(imageSize, TestImg, 'ColorPreprocessing', 'gray2rgb');


% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights');

featureLayer = 'fc1000';

% Extract test features using the CNN
testFeatures = activations(net, validationTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Preprocessing Technique
imdsTrain.ReadFcn = @(filename)preprocess_Xray(filename);
imdsTest.ReadFcn = @(filename)preprocess_Xray(filename);
    

    