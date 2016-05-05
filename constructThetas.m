function [Thetas] = constructThetas(hiddenLayerSize)

dimensions = [8 hiddenLayerSize 10];


for i = 1:length(dimensions)-1
   Thetas{i} = randInitializeWeights(dimensions(i), dimensions(i+1))';
end

end

