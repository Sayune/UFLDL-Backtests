function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
    filterDim1,numFilters1,poolDim1,...
    filterDim2,depth2,numFilters2,poolDim2,pred)
if ~exist('pred','var')
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images
weightDecay = 1e-4;

%% Reshape parameters and setup gradient matrices

[Wc1,Wc2, Wd, bc1,bc2, bd] = cnnParamsToStack(theta,imageDim,filterDim1,numFilters1,poolDim1,filterDim2,depth2,numFilters2,poolDim2,numClasses);
Wd=gather(Wd);
bd=gather(bd);


% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc1_grad = zeros(size(Wc1));
bc1_grad = zeros(size(bc1));
Wc2_grad = zeros(size(Wc2));
bc2_grad = zeros(size(bc2));
Wd_grad = zeros(size(Wd));
bd_grad = zeros(size(bd));

%%======================================================================
%% Forward Propagation

%% Convolutional Layer I

convDim1 = imageDim-filterDim1+1; % dimension of convolved output
outputDim1 = (convDim1)/poolDim1; % dimension of subsampled output

%ConvLayer

activations1=zeros(convDim1,convDim1,numFilters1,numImages);

for j=1:numFilters1
    k1j=squeeze(Wc1(:,:,j));
    z=convn(images,k1j,'valid');
    bj=squeeze(bc1(j,1));
    activations1(:,:,j,:)=sigm(z+bj);
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
    activations2(:,:,j,:)=sigm(z+bj);
end

%Subsampling activations
activationsPooled2=zeros(outputDim2,outputDim2,numFilters2,numImages);

for j = 1 : numFilters2
    z= convn(squeeze(activations2(:,:,j,:)), ones(poolDim2) / (poolDim2 ^ 2), 'valid');
    activationsPooled2(:,:,j,:) = z(1 : poolDim2 : end, 1 : poolDim2 : end, :);
end

%Reshaping for FC layer

fv = [];
    for j = 1 : numFilters2
        fv = [fv ; reshape(squeeze(activationsPooled2(:,:,j,:)), outputDim2 * outputDim2, numImages)];
    end

%% FC Layer

%L2 Loss
    % O=sigm(Wd*fv + repmat(bd, 1,numImages));

%Softmax Loss
%preventing large values
    O=Wd*fv + repmat(bd, 1,numImages);
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

weightDecayCost = .5 * weightDecay * (sum(Wd(:) .^ 2) + sum(Wc1(:) .^ 2)+sum(Wc2(:) .^ 2));
%Softmax Loss
    I=sub2ind(size(probs), labels',1:numImages);
    values = log(probs(I));
    cost=-sum(values)/numImages + weightDecayCost; 
    od=probs-Y;

if pred
    [~,preds] = max(O,[],1);
    preds = preds';
    grad = 0;
    return;
end;
%% Backpropatiaton Conv-Pooling Layer II
%S2 Layer

errorsPooled2=Wd'*od;
errorsPooled2 = reshape(errorsPooled2, [], outputDim2, numFilters2, numImages);

%C2 Layer
upsamplederrors2=zeros(convDim2,convDim2,numFilters2,numImages);

for imageNum=1:numImages
    for filterNum2=1:numFilters2
        delta=errorsPooled2(:,:,filterNum2,imageNum);
        upsamplederrors2(:,:,filterNum2,imageNum)=(1/poolDim2^2) * kron(delta,ones(poolDim2));
    end
end

errorsConv2=(upsamplederrors2.*activations2.*(1-activations2));
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
    errorsPooled1(:,:,i,:)=gather(z);
end

%C Layer
upsamplederrors1=zeros(convDim1,convDim1,numFilters1,numImages);
for imageNum=1:numImages
    for filterNum=1:numFilters1
        delta=errorsPooled1(:,:,filterNum,imageNum);
        upsamplederrors1(:,:,filterNum,imageNum)=(1/poolDim1^2) * kron(delta,ones(poolDim1));
    end
end

errorsConv1=(upsamplederrors1.*activations1.*(1-activations1));


%% Gradient Calculation
%================================================================================
%% Conv Layer I
%Filters weights gradients
for j=1:numFilters1
    dj=squeeze(errorsConv1(:,:,j,:));
    %Sum of gradients over all images for considered filter
    dk1j=convn(flipall(images),dj,'valid')/numImages;
    %Save value in Wc_grad before taking the next filter
    Wc1_grad(:,:,j)=dk1j;
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
%% FC
%FC weights gradients
Wd_grad=od*(fv)'/numImages;
Wd_grad=Wd_grad+weightDecay * Wd;
%FC bias gradients
bd_grad=mean(od,2);


%Here, We compute the gradients for the ConvLayer

%% Unroll gradient into grad vector for minFunc
grad = [Wc1_grad(:) ;Wc2_grad(:) ; Wd_grad(:) ; bc1_grad(:) ; bc2_grad(:) ;bd_grad(:)];

end
