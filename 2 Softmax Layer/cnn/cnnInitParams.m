function theta = cnnInitParams(numClasses,hiddenLayer,imageDim)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
  
%First FC Layer

r1  = sqrt(6) / sqrt(numClasses+hiddenLayer+1);
Wd1 = rand(numClasses, hiddenLayer) * 2 * r1 - r1;
bd1 = zeros(numClasses, 1);

%Second Layer

r2  = sqrt(6) / sqrt(hiddenLayer+(imageDim*imageDim)+1);
Wd2=rand(hiddenLayer,(imageDim*imageDim))*2*r2-r2;
bd2=zeros(hiddenLayer,1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wd1(:) ; Wd2(:) ; bd1(:) ; bd2(:)];

end

