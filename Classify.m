function [p] = Classify(alphas)
numObs = size(alphas,1);
numClasses = size(alphas,2);
%classes = zeros(numObs,numClasses);
[ok,p] = max(alphas, [], 2 );
%classes(:,p) = 1;
end

