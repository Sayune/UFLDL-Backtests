function [Wc1,Wc2, Wd1,Wd2, bc1,bc2, bd1,bd2] = cnnParamsToStack(theta,imageDim,filterDim1,numFilters1,poolDim1,filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,numClasses)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
%  bc      -  bias for convolution layer of size numFilters x 1
%  bd      -  bias for dense layer of size hiddenSize x 1

outDim1 = (imageDim - filterDim1 + 1)/poolDim1;
outDim2 = (outDim1 - filterDim2 + 1) /poolDim2;
hiddenSize=outDim2^2*numFilters2;
%% Reshape theta
indS = 1;
indE = filterDim1^2*numFilters1*3;
Wc1 = reshape(theta(indS:indE),filterDim1,filterDim1,3,numFilters1);
indS = indE+1;
indE =indE+ filterDim2^2*numFilters2*depth2;
Wc2 = reshape(theta(indS:indE),filterDim2,filterDim2,depth2,numFilters2);
indS = indE+1;
indE = indE+hiddenSize*hiddenLayer;
Wd1 = reshape(theta(indS:indE),hiddenLayer,hiddenSize);
indS = indE+1;
indE = indE+hiddenLayer*numClasses;
Wd2 = reshape(theta(indS:indE),numClasses,hiddenLayer);
indS = indE+1;
indE = indE+numFilters1;
bc1 = theta(indS:indE);
indS = indE+1;
indE = indE+numFilters2;
bc2 = theta(indS:indE);
indS = indE+1;
indE = indE+hiddenLayer;
bd1 = theta(indS:indE);
bd2=theta(indE+1:end);
end