function [ X ] = normalize( X )
    X =  (X -  ones(size(X,1),1) * min(X)) ./ (ones(size(X,1),1) * (max(X)-min(X)));
end