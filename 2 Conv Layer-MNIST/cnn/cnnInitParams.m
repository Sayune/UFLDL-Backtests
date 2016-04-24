function theta = cnnInitParams(imageDim,filterDim1,numFilters1, poolDim1,filterDim2,depth2,numFilters2, poolDim2,numClasses)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  numClasses -  number of classes to predict
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(filterDim1 < imageDim,'filterDim must be less that imageDim');

r1=sqrt(6)/sqrt((numFilters1+1)*filterDim2^2);
r2=sqrt(6)/sqrt((numFilters1+numFilters2)*filterDim2^2);
Wc1 = (rand(filterDim1,filterDim1,numFilters1)-0.5)*2*r1;
Wc2 = (rand(filterDim2,filterDim2,depth2,numFilters2)-0.5)*2*r2;

%Conv Layer I
convDim1 = imageDim - filterDim1 + 1; % dimension of convolved Image

%Conv Layer2
convDim2 = (convDim1/poolDim1) - filterDim2 + 1; % dimension of convolved Feature
outDim2=convDim2/poolDim2;


% we'll choose weights uniformly from the interval [-r, r]
hiddenSize=outDim2^2*numFilters2;
r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = (rand(numClasses, hiddenSize)-0.5)*2 * r;

bc1 = zeros(numFilters1, 1);
bc2 = zeros(numFilters2,1);
bd  = zeros(numClasses, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc1(:) ;Wc2(:); Wd(:) ; bc1(:);bc2(:); bd(:)];

end

