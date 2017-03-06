% Author : David Chen
% ANN training logic, forward and backward propagation. Trained weights are stored in W matrix.

% W - Initial Weight
% X - Features
% y - Response Variable
% alpha - Learning Rate

function [W] = ANN(W, X, y, alpha)
    J = 0;
    m = size(X, 1);
    
    Z = cell(length(W)+1,1);
    A = cell(length(W)+1,1);
    D = cell(length(W)+1,1);
    G = cell(length(W),1);
    
    Z{1} = 0; % Sigmoid
    A{1} = X; % Activation
    D{1} = 0; % Delta

    for i = 1:length(W)
        G{i} = zeros(size(W{i}));
    end

    for i = 2:length(W)+1
        Z{i} = [ones(m, 1) A{i-1}] * W{i-1};
        A{i} = sigmoid(Z{i});
    end
    
    for i = 1:length(A)-1
        A{i} = [ones(m,1) A{i}];
    end
 
    for i=1:m
        for j = fliplr(1:length(D))

            if j == length(D)
                D{j} = (A{j}(i,:) - y(i,:))' ;

            elseif j == 1
                G{j} = G{j} + A{j}(i,:)' * D{j+1}' / m ;
            else
                D{j} = W{j} * D{j+1};
                D{j} = D{j}(2:end) .* sigmoidGradient(Z{j}(i,:))'; 
                G{j} = G{j} + A{j}(i,:)' * D{j+1}' / m ;
            end
        end 
    end
 
    for i = 1:length(W)
        W{i} = W{i} - alpha * G{i};
    end
end
 
 
