function [outputLayer] = AllLayersFwdProp(Thetas, inputLayer)
    m = size(inputLayer,1);
    iterations = length(Thetas)+1; 
    A{1} = inputLayer;
    
    for i = 2:iterations
       Z{i} = [ones(m,1) A{i-1}] * Thetas{i-1};
       A{i} = sigmoid(Z{i});
    end
    
    outputLayer = A{iterations};
end


