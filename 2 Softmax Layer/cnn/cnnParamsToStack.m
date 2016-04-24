function [Wd1, Wd2, bd1, bd2] = cnnParamsToStack(theta,numClasses,hiddenLayer,imageDim)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
%% Reshape theta
indS = 1;
indE = hiddenLayer*numClasses;
Wd1 =reshape( theta(indS:indE),numClasses,hiddenLayer);
indS = indE+1;
indE = indE+hiddenLayer*imageDim*imageDim;
Wd2 = reshape(theta(indS:indE),hiddenLayer,imageDim*imageDim);
indS = indE+1;
indE = indE+numClasses;
bd1 = theta(indS:indE);
bd2 = theta(indE+1:end);
end