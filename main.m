%%
clear all


%% 导入psx, psy and previous result for display

load("data\\keyData.mat");
load("data\\result_display.mat")
%% 导入训练数据
table_x = readtable('data\\inputData.xlsx');
table_y = readtable('data\\outputData.xlsx');

data_x = table2array(table_x)';
data_x = data_x(2:end, :);
[data_x_n, psx] = mapminmax(data_x, 0, 1);
data_y = table2array(table_y)';
data_y = data_y(2:end, :);
dp_data = data_y(1, :);
bpt_data = data_y(2, :);
te_data = data_y(3, :);
ce_data = data_y(4, :);
[dp_data_n, psdp] = mapminmax(dp_data, 0, 1);
[bpt_data_n, psbpt] = mapminmax(bpt_data, 0, 1);
[te_data_n, pste] = mapminmax(te_data, 0, 1);
[ce_data_n, psce] = mapminmax(ce_data, 0, 1);
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce');

%% prepare the train data
data_x_n_train = data_x_n(1:end, 1:500);
dp_data_n_train = dp_data_n(1:end, 1:500);
bpt_data_n_train = bpt_data_n(1:end, 1:500);
te_data_n_train = te_data_n(1:end, 1:500);
ce_data_n_train = ce_data_n(1:end, 1:500);

%% 模型训练

tic
clear net
InputSize = 5;
numHiddenUnits = 40;
FC1 = 30;
FC2 = 20;
FC3 = 10;
FC4 = 5;
MaxEpochs=400;
InitialLearnRate=0.0025;
OutputSize = 1;
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
net_dp = trainNetwork(data_x_n_train, dp_data_n_train, layers, options);
% net_bpt = trainNetwork(data_x_n_train, bpt_data_n_train, layers, options);
% net_te = trainNetwork(data_x_n_train, te_data_n_train, layers, options);
% net_ce = trainNetwork(data_x_n_train, ce_data_n_train, layers, options);

%保存网络，清理训练数据
save('net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
clear InputSize FC1 FC2 numHiddenUnits MaxEpochs InitialLearnRate OutputSize
clear table_x table_y data_y dp_data bpt_data te_data ce_data dp_data_n bpt_data_n te_data_n ce_data_n
clear data_x_n_train dp_data_n_train bpt_data_n_train te_data_n_train ce_data_n_train
toc


%% 继续训练
start_col = 501;
end_col = 600;

%加载训练数据
table_x = readtable('data\\inputData.xlsx');
table_y = readtable('data\\outputData.xlsx');
data_x = table2array(table_x)';
data_x = data_x(2:end, start_col:end_col);
data_y = table2array(table_y)';
data_y = data_y(2:end, start_col:end_col);
dp_data = data_y(1, :);
bpt_data = data_y(2, :);
te_data = data_y(3, :);
ce_data = data_y(4, :);

%normalize
data_x_n = mapminmax('apply', data_x, psx);
dp_data_n = mapminmax('apply', dp_data, psdp);
bpt_data_n = mapminmax('apply', bpt_data, psbpt);
te_data_n = mapminmax('apply', te_data, pste);
ce_data_n = mapminmax('apply', ce_data, psce);

%训练网络
net_dp = trainNetwork(data_x_n, dp_data_n, net_dp.Layers, options);
% net_bpt = trainNetwork(data_x_n, bpt_data_n, net_bpt.Layers, options);
% net_te = trainNetwork(data_x_n, te_data_n, net_te.Layers, options);
% net_ce = trainNetwork(data_x_n, ce_data_n, net_ce.Layers, options);

%保存网络，清理训练数据
save('net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
clear table_x table_y data_y dp_data bpt_data te_data ce_data dp_data_n bpt_data_n te_data_n ce_data_n
clear data_x_n_train dp_data_n_train bpt_data_n_train te_data_n_train ce_data_n_train

%% 导入神经网络
load('net.mat');

%% 验证数据
start_col = 700;
end_col = 800;
table_x = readtable('data\\inputData.xlsx');
table_y = readtable('data\\outputData.xlsx');

data_x = table2array(table_x)';
data_x_v = data_x(2:end, start_col:end_col);
data_y = table2array(table_y)';
data_y = data_y(2:end, start_col:end_col);
% 
%actual_data
dp_data_v = data_y(1, :);
bpt_data_v = data_y(2, :);
te_data_v = data_y(3, :);
ce_data_v = data_y(4, :);

% normalize
data_x_v_n = mapminmax('apply', data_x_v, psx);

% predict
dp_data_p_n = predict(net_dp, data_x_v_n);
% bpt_data_p_n = predict(net_bpt, data_x_v_n);
% te_data_p_n = predict(net_de, data_x_v_n);
% ce_data_p_n = predict(net_ce, data_x_v_n);

% reverse data
dp_data_p = mapminmax('reverse', dp_data_p_n, psdp);
% bpt_data_p = mapminmax('reverse', bpt_data_p_n, psbpt);
% te_data_p = mapminmax('reverse', te_data_p_n, pste);
% ce_data_p = mapminmax('reverse', ce_data_p_n, psce);

% save the result data
save("result_display.mat", "dp_data_p", "bpt_data_p", "te_data_p", "ce_data_p");

%% 展示误差
% actual -> dp_data_v
% predict -> dp_data_p
x = 1:1:101;
hold on
plot(x, dp_data_v);
plot(x, dp_data_p);
hold off







