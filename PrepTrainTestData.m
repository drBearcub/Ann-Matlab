function [train_y, train_X, test_y, test_X] = PrepTrainTestData(X,y,cutpoint)

data = [X y];

% randomly permute data
data = data(randperm(size(data,1)),:);

% normalize data
data(:,1:end-1) = standardize(data(:,1:end-1));

%plotmatrix(data);


training_data = data(1:round(size(data,1)*cutpoint),:);
testing_data = data(round(size(data,1)*cutpoint)+1:end,:);

%  prepare training set
train_y = training_data(:,end);
train_X = training_data(:,1:end-1);

%  prepare testing set
test_y = testing_data(:,end);
test_X = testing_data(:,1:end-1);
end

