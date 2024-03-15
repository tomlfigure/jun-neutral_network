
%% 导入psx, psy and previous result for display

load("data\keyData.mat");
load("data\result_display.mat")
%% 导入训练数据
table_x = readtable('data\\data_x.csv');
table_y = readtable('data\\data_y.csv');
data_x = table2array(table_x)';
[data_x_n, psx] = mapminmax(data_x, 0, 1);
data_y = table2array(table_y)';
[data_y_n, psy] = mapminmax(data_y, 0, 1);
save('data\\keyData.mat', 'psx', 'psy');

%% 模型训练
tic
clear net
InputSize = 5;
InputSize = 5;
InputSize = 5;
numHiddenUnits = 40;
In
% numHiddenUnits = 40;
FC1 = 30;
FC2 = 20;
FC3 = 10;
FC4 = 5;
MaxEpochs=1000;
InitialLearnRate=0.0025;
OutputSize = 3;
layers = [ ...
    sequenceInputLayer(InputSize,'Name','input1')  %输入层
    lstmLayer(120,'Name','lstm1')  %LSTM神经网络 
    lstmLayer(60,'Name','lstm2')  %LSTM神经网络
    fullyConnectedLayer(FC1,'Name','fc1')
    fullyConnectedLayer(FC2,'Name','fc2')%全连接层
    fullyConnectedLayer(FC3,'Name','fc3')
    fullyConnectedLayer(FC4,'Name','fc4')
    fullyConnectedLayer(OutputSize,'Name','fc5')  %输出层
    regressionLayer('Name','re_1')];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',1,...
    'Shuffle','never',...
    'GradientThreshold',1, ...
    'InitialLearnRate',InitialLearnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',50,...
    'LearnRateDropFactor',0.9,...
    'L2Regularization',0.0002,...
    'Verbose',false,...
    'Plots','training-progress');
rng('default')  %固定随机化，便于调参
net = trainNetwork(data_x_n, data_y_n, layers, options);


clear InputSize FC1 FC2 numHiddenUnits MaxEpochs InitialLearnRate OutputSize table_x table_y 
clear data_x data_x_n data_y data_y_n
toc


%% 继续训练
table_x = readtable('data\\data_x.csv');
table_y = readtable('data\\data_y.csv');
data_x = table2array(table_x)';
data_y = table2array(table_y)';
data_x_n = mapminmax('apply', data_x, psx);
data_y_n = mapminmax('apply', data_y, psy);
net = trainNetwork(data_x_n, data_y_n, net.Layers, options);
clear table_x table_y data_x data_x_n data_y data_y_n

%% 验证数据
table_x_v = readtable('data\\data_x_v.csv');
table_y_v = readtable('data\\data_y_v.csv');
%验证数据
data_x_v = table2array(table_x_v)';
data_y_v = table2array(table_y_v)';
%归一化
data_x_v_n = mapminmax('apply', data_x_v, psx);
%预测数据
data_y_p_n = predict(net, data_x_v_n);
data_y_p = mapminmax('reverse', data_y_p_n, psy);

%% 展示误差
errors = data_y_v - data_y_p;
errors = errors';
errors_percentage = errors ./ data_y_v' .* 100;
disp(['               actualt                       ', 'predicted                   ', 'error                  ', 'error_percentage'])
disp([data_y_v', data_y_p', errors, errors_percentage]);
clear table_x_v table_y_v data_x_v data_x_v_n data_y_p_n
save('data\\result_display.mat', 'data_y_v', 'data_y_p', 'errors', 'errors_percentage');







