clear ; close all; clc
Data = load('mydata.csv');
Data = Data(randperm(size(Data,1)),:);

Data(:,1:8) = standardize(Data(:,1:8));

Training = Data(1:965,:);
Testing = Data(966:1484,:);
compare = Training(:,9);
compareTest = Testing(:,9);
Ytraining = zeros(10, size(Training(:,9),1));

for i = 1:size(Training(:,9),1)
    Ytraining(Training(i,9),i) = 1;
end

Ytraining = Ytraining';
inputLayer = Training(:,1:8);

%% Problem 1
Theta1 = randInitializeWeights(8,3)';
Theta2 = randInitializeWeights(3,10)';

thetabuddies{1}  = Theta1;
thetabuddies{2} = Theta2;

iter = 300;
weights1 = zeros(iter,3);
weights2 = zeros(iter,10);
error = zeros(iter,1);c
testingerror = zeros(iter,1);

outputyay = zeros(iter,10);
testingoutput = zeros(iter,10);

for i = 1:iter
    [thetabuddies] = ANNProblem1(thetabuddies, inputLayer, Ytraining, 2);
    curPred = AllLayersFwdProp(thetabuddies, inputLayer);
    curResult = Classify(curPred);
    error(i,1) = 1- sum(compare == curResult)/965;
    
    curTestPred = AllLayersFwdProp(thetabuddies, Testing(:,1:8));
    curTestResult = Classify(curTestPred);
    testingerror(i) = 1-sum(Testing(:,9) == curTestResult)/(1484-965);
    
    weights1(i,:) = thetabuddies{1}(2,:);
    weights2(i,:) = thetabuddies{2}(2,:);
    outputyay(i,:) = curPred(1,:);
    
end

errorPlot = plot(testingerror);
saveas(errorPlot, 'testingerror.jpg');

outputPlot = plot(outputyay(:,1));
hold on
for i = 2:10
    outputPlot = plot(outputyay(:,i));
end
hold off
saveas(outputPlot, 'output.jpg');

curPlot = plot(weights1(:,1))
hold on
for i = 2:3
    curPlot = plot(weights1(:,i))
end
hold off   
saveas(curPlot, 'node2layer1.jpg');   

curPlot2 = plot(weights2(:,1))
hold on
for i = 2:10
    curPlot2 = plot(weights2(:,i))
end
hold off
saveas(curPlot2, 'node2layer2.jpg');  

output = AllLayersFwdProp(thetabuddies, inputLayer);
finalresult = Classify(output);
sum(compare == finalresult)/965

output = AllLayersFwdProp(thetabuddies, Testing(:,1:8));
finalresult = Classify(output);
sum(Testing(:,9) == finalresult)/(1484-965)

%% Problem 2
thetabuddies{1} = randInitializeWeights(8,3)';
thetabuddies{2} = randInitializeWeights(3,10)';

Ytraining2 = zeros(10, size(Training(:,9),1));

for i = 1:size(Data(:,9),1)
    Ytraining2(Data(i,9),i) = 1;
end

Ytraining2 = Ytraining2';

for i = 1:800
    [thetabuddies] = ANN(thetabuddies, Data(:,1:8), Ytraining2, 2);
end

hidden = FwdProp(Data(:,1:8), thetabuddies{1});
output = FwdProp(hidden , thetabuddies{2});
finalresult = Classify(output);

1 - sum(Data(:,9) == finalresult)/1484
%% Problem 3
t1 = [0.1 0.2 0.3; 0.2 0.3 0.4; 0.5 0.6 0.7; 0.8 0.9 1.0; 1.1 1.2 1.3; 1.2 1.3 1.4; 1.5 1.6 1.7; 1.6 1.7 1.8; 1.9 2.0 2.1];
t2 = [1.0 2.2 3.3 1.2 3.3 4.4 5.5 6.6 7.7 8.8; 1.0 2.3 3.3 1.2 3.4 4.4 5.5 6.2 7.7 8.8; 1.1 2.2 3.9 2.4 1.1 2.3 4.4 5.5 7.5 9.9; 1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 1.2];

t1 = buddie1
t2 = buddie2
t1 = t1'
t2 = t2'


OneOb = load('mydata.csv');
OneOb = OneOb(1,:);

fun = zeros(10,1);
fun(OneOb(1,9),1) = 1;
fun = fun';

input3 = OneOb(1,1:8);
Z2 = [1 OneOb(1,1:8)] * t1
A2 = sigmoid(Z2)

Z3 = [1 A2] * t2
A3 = sigmoid(Z3)

A2 = [1 A2]
A1 = [1 OneOb(:,1:8)]

compare3 = zeros(1,10);
compare3(1,OneOb(1,9)) = 1;

Delta3 = (A3 - compare3)';
Delta2temp = t2 * Delta3;
woo = sigmoidGradient(Z2)';
Delta2 = Delta2temp(2:end) .* sigmoidGradient(Z2)';

Gradient2 = A2' * Delta3';
Gradient1 = A1' * Delta2';

t1 = t1 - 1*Gradient1;
t2 = t2 - 1*Gradient2;

thetabuddies2 = cell(2,1);
thetabuddies2{1} = [0.1 0.2 0.3; 0.2 0.3 0.4; 0.5 0.6 0.7; 0.8 0.9 1.0; 1.1 1.2 1.3; 1.2 1.3 1.4; 1.5 1.6 1.7; 1.6 1.7 1.8; 1.9 2.0 2.1];
thetabuddies2{2} = [1.0 2.2 3.3 1.2 3.3 4.4 5.5 6.6 7.7 8.8; 1.0 2.3 3.3 1.2 3.4 4.4 5.5 6.2 7.7 8.8; 1.1 2.2 3.9 2.4 1.1 2.3 4.4 5.5 7.5 9.9; 1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9 1.2];

thetabuddies2 = ANN(thetabuddies2, OneOb(1,1:8), fun ,1)

thetabuddies2{1} == t1
thetabuddies2{2} == t2
%% Problem 4
ANNConfigs = cell(3,4);
ANNThetas = cell(3,4);
ANNAccuracy = cell(3,4);
compare = Training(:,9);
compareTest = Testing(:,9);

for i = 1:4
    for j = 1:3
        ANNConfigs{j,i} = ones(1,j)*i*3;
        ANNThetas{j,i} = constructThetas(ANNConfigs{j,i})';
        
        for iter = 1:800
            ANNThetas{j,i} = ANN(ANNThetas{j,i}, inputLayer, Ytraining, 2);
        end
        
        output = AllLayersFwdProp(ANNThetas{j,i}, Testing(:,1:8));
        finalresult = Classify(output);
        ANNAccuracy{j,i} = sum(compareTest == finalresult)/(1484-965)
    end
end


%% Problem 5
problem5X = [0.50, 0.49, 0.52, 0.20, 0.55, 0.03, 0.50, 0.39];
problem5Pred = AllLayersFwdProp(thetabuddies, problem5X);
Amax = max(problem5Pred)


uncertainty = sum(1 +( problem5Pred - Amax))
Classify(problem5Pred)
