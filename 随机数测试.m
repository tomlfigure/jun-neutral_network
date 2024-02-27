
%%
clc
clear all



%% 导入随机数, 创建神经网络

inputDataPath = "E:\\项目\\python\\machineLearning\\inputData.csv";
outputDataPath = "E:\\项目\\python\\machineLearning\\outputData.csv";

inputTable = readtable(inputDataPath);
outputTable = readtable(outputDataPath);

inputData = (table2array(inputTable))';
outputData = (table2array(outputTable))';




[inputData_n, psi] = mapminmax(inputData, 0, 1);
[outputData_n, pso] = mapminmax(outputData, 0, 1);

% 创建网络
net = newff(inputData_n, outputData_n, 10);


%% 获取新的数据，然后继续训练

%获取新的数据

inputTable = readtable(inputDataPath);
outputTable = readtable(outputDataPath);

inputTrainData = (table2array(inputTable))';
outputTrainData = (table2array(outputTable))';

%将数据归一化
inputTrainData_g = mapminmax('apply', inputTrainData, psi);
outputTrainData_g = mapminmax('apply', outputTrainData, pso);


%设置训练参数
net.trainParam.epochs = 100;
net.trainParam.lr = 0.1;


% 训练
net = train(net, inputTrainData_g, outputTrainData_g);

%% 验证神经网络对于 3 to 2 based on randomNumbers 的预测能力

inputTable = readtable(inputDataPath);
outputTable = readtable(outputDataPath);

inputVerifyData = (table2array(inputTable))';
outputVerifyData_actual = (table2array(outputTable))';

inputVerifyData_g = mapminmax('apply', inputVerifyData, psi);


outputVerifyData_g = sim(net, inputVerifyData_g);
outputVerifyData = mapminmax('reverse', outputVerifyData_g, pso);


%% 展示结果

disp(['actual   ', 'predicted   ', 'error']);

error = abs(outputVerifyData - outputVerifyData_actual);

results = [outputVerifyData_actual' outputVerifyData' error'];
disp(results);









