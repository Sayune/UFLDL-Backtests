function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
    filterDim1,numFilters1,poolDim1,...
    filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,pred)
if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,4); % number of images
weightDecay = 0;

%% Reshape parameters and setup gradient matrices

[Wc1,Wc2, Wd1,Wd2, bc1,bc2, bd1,bd2] = cnnParamsToStack(theta,imageDim,filterDim1,numFilters1,poolDim1,filterDim2,depth2,numFilters2,poolDim2,hiddenLayer,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc1_grad = zeros(size(Wc1));
bc1_grad = zeros(size(bc1));
Wc2_grad = zeros(size(Wc2));
bc2_grad = zeros(size(bc2));
Wd1_grad = zeros(size(Wd1));
bd1_grad = zeros(size(bd1));
Wd2_grad = zeros(size(Wd2));
bd2_grad = zeros(size(bd2));

%%======================================================================
%% Forward Propagation

%% Convolutional Layer I

convDim1 = imageDim-filterDim1+1; % dimension of convolved output
outputDim1 = (convDim1)/poolDim1; % dimension of subsampled output

%ConvLayer

activations1=zeros(convDim1,convDim1,numFilters1,numImages);

for j=1:numFilters1
    z=zeros(convDim1,convDim1,numImages);
    for i=1:3
        kij=squeeze(Wc1(:,:,i,j));
        ai=squeeze(images(:,:,i,:));
        z=z+convn(ai,kij,'valid');
    end
    
    bj=squeeze(bc1(j,1));
    z=z+bj;
            activations1(:,:,j,:)=sigm(z); %sigmoid
%     activations1(:,:,j,:)=max(z,0); %ReLu
    
end

%Pooling

activationsPooled1=zeros(outputDim1,outputDim1,numFilters1,numImages);
for j = 1 : numFilters1
    z= convn(squeeze(activations1(:,:,j,:)), ones(poolDim1) / (poolDim1 ^ 2), 'valid');
    activationsPooled1(:,:,j,:) = z(1 : poolDim1 : end, 1 : poolDim1 : end, :);
end
%% Conv Layer II
convDim2 = outputDim1-filterDim2+1; % dimension of convolved output
outputDim2 = (convDim2)/poolDim2; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations2 = zeros(convDim2,convDim2,numFilters2,numImages);


for j=1:numFilters2
    z=zeros(convDim2,convDim2,numImages);
    for i=1:depth2
        kij=squeeze(Wc2(:,:,i,j));
        ai=squeeze(activationsPooled1(:,:,i,:));
        z=z+convn(ai,kij,'valid');
    end
    
    bj=squeeze(bc2(j,1));
    z=z+bj;
            activations2(:,:,j,:)=sigm(z); %sigmoid
%     activations2(:,:,j,:)=max(z,0); %ReLu
end

%Subsampling activations
activationsPooled2=zeros(outputDim2,outputDim2,numFilters2,numImages);

for j = 1 : numFilters2
    z= convn(squeeze(activations2(:,:,j,:)), ones(poolDim2) / (poolDim2 ^ 2), 'valid');
    activationsPooled2(:,:,j,:) = z(1 : poolDim2 : end, 1 : poolDim2 : end, :);
end

%Reshaping for FC layer I

fv = [];
for j = 1 : numFilters2
    fv = [fv ; reshape(squeeze(activationsPooled2(:,:,j,:)), outputDim2 * outputDim2, numImages)];
end

%% FC Layer I

%fv hiddenSizexnumImages
%Wd1 hiddenLayerxhiddenSize
O1=Wd1*fv + repmat(bd1, 1,numImages); %hiddenLayerxnumImages
O1=sigm(O1); %sigmoid
% O1=max(O1,0); %ReLu


%% Output Layer

%L2 Loss
% O=sigm(Wd2*O1 + repmat(bd2, 1,numImages));

%Softmax Loss

%preventing large values
%Wd2 numClassesxhiddenLayers

O=Wd2*O1 + repmat(bd2, 1,numImages);
O=bsxfun(@minus,O,max(O));
O=exp(O); %element_wise exponential
S=sum(O);
probs=bsxfun(@rdivide,O,S);
%%================================================================================================================
%% Backpropagation
%COnstructing the goundtruth matix related to labels
Y=zeros(numClasses,numImages);
for j=1:numImages
    Y(labels(j),j)=1;
end


%L2 Loss
% E=O-Y;
%Calculating Cost
% cost = 1/2* sum(E(:) .^ 2) / size(E,2);
% od=E.*(O.*(1-O));

weightDecayCost = .5 * weightDecay * (sum(Wd1(:) .^ 2)+sum(Wd2(:) .^ 2) + sum(Wc1(:) .^ 2)+sum(Wc2(:) .^ 2));
%Softmax Loss

% cost=-sum(values)/numImages + weightDecayCost;
cost=-sum(sum(Y.*log(probs)))/numImages + weightDecayCost;
E=probs-Y;
od=E.*(O.*(1-O));

if pred
    [~,preds] = max(O,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%% Backpropagation FC I
%od numClassesxnumImages
%Wd1 numClassesxhiddenLayer
%O1 hiddenLayerxnumImages
errorsFC1=(Wd2'*od);
%% Backpropatiaton Conv-Pooling Layer II
%S2 Layer

errorsPooled2=Wd1'*errorsFC1;
errorsPooled2 = reshape(errorsPooled2, [], outputDim2, numFilters2, numImages);

%C2 Layer
upsamplederrors2=zeros(convDim2,convDim2,numFilters2,numImages);

for imageNum=1:numImages
    for filterNum2=1:numFilters2
        delta=errorsPooled2(:,:,filterNum2,imageNum);
        upsamplederrors2(:,:,filterNum2,imageNum)=(1/poolDim2^2) * kron(delta,ones(poolDim2));
    end
end

errorsConv2=(upsamplederrors2.*activations2.*(1-activations2)); %sigmoid
% errorsConv2=upsamplederrors2.*double(activations2>0); %ReLu
%% Backpropagatiaton Conv-Pooling Layer I
%S Layer
errorsPooled1=zeros(outputDim1,outputDim1,numFilters1,numImages);

for i=1:numFilters1
    z=zeros(outputDim1,outputDim1,numImages);
    for j=1:numFilters2
        kij=squeeze(Wc2(:,:,i,j));
        dj=squeeze(errorsConv2(:,:,j,:));
        z=z+convn(dj,rot180(kij),'full');
    end
    errorsPooled1(:,:,i,:)=z;
end

%C Layer
upsamplederrors1=zeros(convDim1,convDim1,numFilters1,numImages);
for imageNum=1:numImages
    for filterNum=1:numFilters1
        delta=errorsPooled1(:,:,filterNum,imageNum);
        upsamplederrors1(:,:,filterNum,imageNum)=(1/poolDim1^2) * kron(delta,ones(poolDim1));
    end
end

errorsConv1=(upsamplederrors1.*activations1.*(1-activations1)); %sigmoid

% errorsConv1=upsamplederrors1.*double(activations1>0); %ReLu


%% Gradient Calculation
%================================================================================
%% Conv Layer I
%Filters weights gradients
for j=1:numFilters1
    dj=squeeze(errorsConv1(:,:,j,:));
    for i=1:3
        %Sum of gradients over all images for considered filter
        ai=squeeze(images(:,:,i,:));
        dkij=convn(flipall(ai),dj,'valid')/numImages;
        %Save value in Wc_grad before taking the next filter
        Wc1_grad(:,:,i,j)=dkij;
    end
    bc1_grad(j)=sum(dj(:))/numImages;
end
Wc1_grad=Wc1_grad+weightDecay * Wc1;

%% Conv Layer II
%Filters weights gradients
depth2=size(Wc2_grad,3);

for j=1:numFilters2
    dj=squeeze(errorsConv2(:,:,j,:));
    for i=1:depth2
        ai=squeeze(activationsPooled1(:,:,i,:));
        dkij=convn(flipall(ai),dj,'valid')/numImages;
        Wc2_grad(:,:,i,j)=dkij;
    end
    bc2_grad(j)=sum(dj(:))/numImages;
end
Wc2_grad=Wc2_grad+weightDecay * Wc2;
%% FC II
%FC weights gradients
Wd1_grad=errorsFC1*fv'/numImages;
Wd1_grad=Wd1_grad +weightDecay * Wd1;
%FC bias gradients
bd1_grad=mean(errorsFC1,2);


%% Output Layer
%FC weights gradients
Wd2_grad=od*(O1)'/numImages;
Wd2_grad=Wd2_grad+weightDecay * Wd2;
%FC bias gradients
bd2_grad=mean(od,2);


%Here, We compute the gradients for the ConvLayer

%% Unroll gradient into grad vector for minFunc
grad = [Wc1_grad(:) ;Wc2_grad(:) ; Wd1_grad(:) ; Wd2_grad(:); bc1_grad(:) ; bc2_grad(:) ;bd1_grad(:);bd2_grad(:)];

end
