%% Testing results
%This code is aimed to see how the CNN behaves when parameters are learned
%Therefore it should be launched only after the opttheta has been learned.
%% Architecture
%Conv Layer I
filterDim1 = 5;   
numFilters1 = 6;   
poolDim1 = 2;       
%Conv Layer II
filterDim2= 5 ;
numFilters2= 12;
poolDim2=2;
depth2=numFilters1;

%% Setting the size of the batchtest

batchtest=100;
batchImages=testImages(:,:,1:batchtest);
batchLabels=testLabels(1:batchtest);

x=floor(sqrt(batchtest));
y=floor(batchtest/x)+1 ;
figure
for i=1:batchtest
    subplot(y,x,i)
    imshow(batchImages(:,:,i))
end
suptitle('Images de l''échantillon de test')

imageDim = size(batchImages,1); % height/width of image
numImages = size(batchImages,3); % number of images

%% Reshape parameter vector theta

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias

[Wc1,Wc2, Wd, bc1,bc2, bd] = cnnParamsToStack(opttheta,imageDim,filterDim1,numFilters1,poolDim1,filterDim2,depth2,numFilters2,poolDim2,numClasses);

%%======================================================================

%% Show Learned Convoltution Filters

x1=floor(sqrt(numFilters1));
y1=floor(numFilters1/x1)+1 ;
figure
for i=1:numFilters1
    subplot(y1,x1,i)
    imshow(squeeze(Wc1(:,:,1:3,i)))
    str=sprintf('Filtre %d',i);
    title(str);
end
suptitle('Filtres du CNN')

x2=floor(sqrt(numFilters2));
y2=floor(numFilters2/x2)+1 ;
figure
for i=1:numFilters2
    subplot(y2,x2,i)
    imshow(squeeze(Wc2(:,:,12,i)))
    str=sprintf('Filtre %d',i);
    title(str);
end
suptitle('Filtres du CNN')


%% Forward Propagation


%% Convolutional Layer I

convDim1 = imageDim-filterDim1+1; % dimension of convolved output
outputDim1 = (convDim1)/poolDim1; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations

% activations = zeros(convDim,convDim,numFilters,numImages);
activations1 = cnnConvolve(filterDim1, numFilters1, images, Wc1, bc1);
% outputDim x outputDim x numFilters x numImages tensor for storing


%Showing activations of an image as an example:
numImage=1; %Change this parameter to change the number of considered image
x1=floor(sqrt(numFilters1));
y1=floor(numFilters1/x1)+1 ;
figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(activations1(:,:,i,numImage))
    str=sprintf('Activation %d',i);
    title(str);
end
str1=sprintf('Activations de l''image numéro %d',numImage);
suptitle(str1)

% subsampled activations

activationsPooled1=zeros(outputDim1,outputDim1,numFilters1,numImages);
 for j = 1 : numFilters1
    z= convn(squeeze(activations1(:,:,j,:)), ones(poolDim2) / (poolDim2 ^ 2), 'valid'); 
    activationsPooled1(:,:,j,:) = z(1 : poolDim1 : end, 1 : poolDim1 : end, :);
end

%Showing MeanPooled activations of an image as an example:

figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(activationsPooled(:,:,i,numImage))
    str=sprintf('PooledActiv. %d',i);
    title(str);
end
str1=sprintf('Pooled Activations de l''image numéro %d',numImage);
suptitle(str1)


%% Conv Layer II
convDim2 = outputDim1-filterDim2+1; % dimension of convolved output
outputDim2 = (convDim2)/poolDim2; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations2 = zeros(convDim2,convDim2,numFilters2,numImages);


%Convolution between Pooled Activations from layer I and Filter volume
    for j=1:numFilters2
         bj=squeeze(bc2(j,1));
         z=zeros(convDim2,convDim2,numImages);
        for i=1:depth2
            %Selecting 2-D filter
            kij=squeeze(Wc2(:,:,i,j));
            %Selecting activation
            ai=squeeze(activationsPooled1(:,:,i,:));
            z=z+convn(ai,rot180(kij),'valid');
        end
        activations2(:,:,j,:)=sigmf(z+bj,[0,1]);
    end
    
    
%Showing activations of an image as an example:
numImage=1; %Change this parameter to change the number of considered image
x1=floor(sqrt(numFilters));
y1=floor(numFilters/x1)+1 ;
figure
for i=1:numFilters2
    subplot(y1,x1,i)
    imshow(activations2(:,:,i,numImage))
    str=sprintf('Activation %d',i);
    title(str);
end
str1=sprintf('Activations de l''image numéro %d',numImage);
suptitle(str1)

    
%Subsampling activations
activationsPooled2=zeros(outputDim2,outputDim2,numFilters2,numImages);

 for j = 1 : numFilters1
    z= convn(squeeze(activations2(:,:,j,:)), ones(poolDim2) / (poolDim2 ^ 2), 'valid'); 
    activationsPooled2(:,:,j,:) = z(1 : poolDim2 : end, 1 : poolDim2 : end, :);
 end

 %Reshaping for Softmax layer
activationsPooled2 = reshape(activationsPooled2,[],numImages);

%Showing MeanPooled activations of an image as an example:

figure
for i=1:numFilters
    subplot(y1,x1,i)
    imshow(activationsPooled(:,:,i,numImage))
    str=sprintf('PooledActiv. %d',i);
    title(str);
end
str1=sprintf('Pooled Activations de l''image numéro %d',numImage);
suptitle(str1)


%% Softmax Layer
% numClasses x numImages for storing probability that each image belongs to
% each class.
%probs = zeros(numClasses,numImages);
A=Wd*activationsPooled2; %numClasses x numImages
%adding the bias
B=bsxfun(@plus,A,bd);
%preventing large values
B=bsxfun(@minus,B,max(B));
B=exp(B); %element_wise exponential
S=sum(B);
probs=bsxfun(@rdivide,B,S); %numClasses x numImages

%%======================================================================
%% Calculate predictions
% Makes predictions given probs and returns without backproagating errors.

    [~,preds] = max(probs,[],1);
    preds = preds';
    preds(preds==10)=0;
 
% %Showing predictions 
% figure
% 
% for i=1:batchtest
%     subplot(y,x,i)
%     axis off
%     str=sprintf('%d',preds(i));
%     t = text(0.5,0.5,str);
%    s = t.FontSize;
%    t.FontSize = 12;
% 
% end
% suptitle('Prédiction des valeurs par le CNN')

%Showing and comparing labels and predictions


batchLabels(batchLabels==10)=0;
figure
for i=1:batchtest
    subplot(y,x,i)
    axis off
    
   if preds(i)==batchLabels(i)
       str=sprintf('%d',batchLabels(i));
    t = text(0.5,0.5,str);
   s = t.FontSize;
   t.FontSize = 12;
   t.Color='blue';
   else
       str=sprintf('%d|%d',batchLabels(i),preds(i));
    t = text(0.5,0.5,str);
   s = t.FontSize;
   t.FontSize = 12;
       t.Color='red';      
   end
end
suptitle('Labels des images | Predictions CNN')

%Showing probability distribution for mistaken predictions

Error=(preds~=batchLabels);
errorimages=find(Error);

for k=errorimages'
    prob=100*probs(:,k);
    copyprob=prob;
    prob(1)=copyprob(10);
    for i=1:9
        prob(i+1)=copyprob(i);
    end
    figure 
    imshow(batchImages(:,:,k))
    str=sprintf('Image numero %d', k);
    title(str)
     figure
    bar([0:9],prob')
    xlabel('Classes')
    ylabel('Probability %')
    str=sprintf('Probability distribution of image %d',k);
    title(str);

end

%% Calculating Accuracy
acc = sum(preds==batchLabels)/length(preds);
fprintf('Accuracy is %f\n',acc);

%%======================================================================