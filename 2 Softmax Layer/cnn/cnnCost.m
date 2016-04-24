function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,hiddenLayer,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict

% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

[Wd1, Wd2,bd1,bd2] = cnnParamsToStack(theta,numClasses,hiddenLayer,imageDim);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
 %First FC Layer 
Wd1_grad = zeros(size(Wd1));
bd1_grad=zeros(size(bd1));

%Second FC Layer
Wd2_grad = zeros(size(Wd2)); 
bd2_grad=zeros(size(bd2));


%%======================================================================
%% Forward Propagation
%FC I Layer

activations=zeros(hiddenLayer,numImages);
for numImg=1:numImages
vectimg=images(:,:,numImg);
vectimg=squeeze(vectimg(:));
B=Wd2*vectimg;
B=bsxfun(@plus,B,bd2);
activations(:,numImg)=B;
end

activations=sigmf(activations,[1 0]);

%activations hiddenLayerxnumImages
%Wd1 numClasses x hiddenLayer
%% Softmax Layer I

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

A=Wd1*activations; %numClasses x numImages
%adding the bias
A=bsxfun(@plus,A,bd1);
%preventing large values
A=bsxfun(@minus,A,max(A));
A=exp(A); %element_wise exponential
S=sum(A);
probs=bsxfun(@rdivide,A,S); %numClasses x numImages


%======================================================================
%% Calculating Cost


cost = 0; % save objective into cost


%Construction of cost function J
I=sub2ind(size(probs), labels',1:numImages);
values = log(probs(I));

%Storing result in cost
cost=-sum(values)/numImages; 

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% Backpropagation

%Backpropagation Softmax Layer

errorsSoftmax=probs;
errorsSoftmax(I)=errorsSoftmax(I)-1;
errorsSoftmax=errorsSoftmax/numImages; %numClasses x numImages
%Wd is %numClasses x hiddenSize

errorsSL=(Wd1'*errorsSoftmax).*activations.*(1-activations); %hiddenLayer x numImages

%Grad SoftmaxLayer 
bd1_grad=sum(errorsSoftmax,2); %notice that this is the sum of gradients.
Wd1_grad=errorsSoftmax*activations';

%Backpropagatiaton FC Layer
% errorsFC=(Wd2'*errorsSL).*(images)*(1-images);

%Backpropagation Entry

%Grad FCLayer 
bd2_grad=sum(errorsSL,2);
img=reshape(images,imageDim*imageDim,numImages);
Wd2_grad=errorsSL*img';



%=======================================%

%% Unroll gradient into grad vector for minFunc
grad = [Wd1_grad(:) ; Wd2_grad(:) ; bd1_grad(:) ; bd2_grad(:)];

end
