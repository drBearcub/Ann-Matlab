function [output] = FwdProp(input, theta)
m = size(input,1);
output = [ones(m,1) input]*theta;
output = sigmoid(output);
end

