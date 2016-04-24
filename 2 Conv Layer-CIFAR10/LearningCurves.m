%% Learning Curves of Convolution Neural Network
%Here we train the CNN and we show the accuracy curves when the training
%size grows
%% Clearing Workspace
clear all;
clc;

%% Loading Data
load cifar10
%Setting number of Batch Test
nbTestBatch=10000;
batchTest=testImages(:,:,:,1:nbTestBatch);
batchTestLabels=testLabels(1:nbTestBatch);

%%======================================================================
%% Architecture
%Input
imageDim = 32;
%Conv Layer I
filterDim1 = 5;
numFilters1 = 6;
poolDim1 = 2;
%Conv Layer II
filterDim2= 5 ;
numFilters2= 12;
poolDim2=2;
depth2=numFilters1;
%FcLayerI
hiddenLayer=100;
%OutputLayer
numClasses = 10;
%Initialization of parameters
theta =cnnInitParams(imageDim,filterDim1,numFilters1,poolDim1,filterDim2,depth2,numFilters2, poolDim2,hiddenLayer,numClasses);
%%======================================================================

%% Options
epochs = 10;
minibatch =50;
alpha = 1;
momentum = 0.5;

%%======================================================================
%% Learning and plotting accuracy curves

% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));
m = length(labels);

%Learning by SGD
it = 0;
nbIterations=epochs*floor(size(images,3)/minibatch);
accuracies=zeros(1,nbIterations);

for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;
        
        % increase momentum after momIncrease iterations
%         if it == momIncrease
%             mom = momentum;
%         end;
        
        % get next randomly selected minibatch
        mb_data = images(:,:,:,rp(s:s+minibatch-1));
        mb_labels = squeeze(labels(rp(s:s+minibatch-1)));
        
        % evaluate the objective function on the next minibatch
        pred=false;
        [cost,grad] = cnnCost(theta,mb_data,mb_labels,numClasses,...
            filterDim1,numFilters1,poolDim1,...
            filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,pred);
        
        % updating parameters using stochastic descent
%                 velocity = (mom.*velocity) + (alpha.*grad);
                theta = theta-(alpha.*grad);
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        
%                 % evaluating accuracy on training set
%         [~,~,trainPreds]=cnnCost(theta,images,labels,numClasses,...
%                                         filterDim1,numFilters1,poolDim1,...
%                                         filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,true);
%                 trainAcc = sum(trainPreds==labels)/length(trainPreds);
%                 fprintf('      Accuracy is %f\n',trainAcc);
%         
%                 evaluating accuracy on testing set
%         %
%                 [~,~,testPreds]=cnnCost(theta,batchTest,batchTestLabels,numClasses,...
%                                         filterDim1,numFilters1,poolDim1,...
%                                         filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,true);
%                     testAcc = sum(testPreds==batchTestLabels)/length(testPreds);
%                     fprintf('  Accuracy is %f\n',testAcc);
%                     accuracies(it)=testAcc;
    end;
    
    % aneal learning rate by factor of two after each epoch
%    alpha = alpha/(2.0);
    
end;

%Saving trained value of parameters
opttheta = theta;

%Plotting accuracy curve
% figure
% plot(minibatch*[1:nbIterations],accuracies*100)
% xlabel('Training size')
% ylabel('Accuracy % on batch test')
% 
