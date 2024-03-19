

%% 构建网络
[data_x, dp_data, bpt_data, te_data, ce_data] = loadData(1, 1046);
[data_x_n, psx] = mapminmax(data_x, 0, 1);

[dp_data_n, psdp] = mapminmax(dp_data, 0, 1);
[bpt_data_n, psbpt] = mapminmax(bpt_data, 0, 1);
[te_data_n, pste] = mapminmax(te_data, 0, 1);
[ce_data_n, psce] = mapminmax(ce_data, 0, 1);
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce','version_id');
fclear();

% 模型训练
load('data\\keyData.mat');
[layers, options] = networkArgus();
rng('default')  %固定随机化，便于调参
% create the networks
start_col = 1;
end_col = 400;
version_id = 0;
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce','version_id');
[data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col);
net_dp = trainNetwork(data_x_n, dp_data_n, layers, options);
logMessage = sprintf("initialize the networks by data from %d to %d cols", start_col, end_col);
trainLog(logMessage, options);
save('data\\net.mat', "net_dp");
fclear();

%% 继续训练
load('data\\keyData.mat');
start_col = 401;
end_col = start_col + 99;
[data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col);
[layers, options] = networkArgus();
net_dp = trainNetwork(data_x_n, dp_data_n, net_dp.Layers, options);
save('data\\net.mat', 'net_dp');
logMessage = sprintf("continue training the network from %d to %d cols", start_col, end_col);
trainLog(logMessage, options);
fclear();

%% 数据预测
start_col_v = 600;
end_col_v = 700;
[data_x_v dp_data_v bpt_data_v te_data_v ce_data_v] = loadData(start_col_v, end_col_v);

% normalize
data_x_v_n = mapminmax('apply', data_x_v, psx);

% predict
dp_data_p_n = predict(net_dp, data_x_v_n);
% reverse data
dp_data_p = mapminmax('reverse', dp_data_p_n, psdp);

% save the result data
save("data\\result_display.mat", "dp_data_p","dp_data_v","start_col_v","end_col_v",'version_id');

% 误差分析

% actual -> dp_data_v
% predict -> dp_data_p

% 导入预测结果
load('data\\result_display.mat');

% max error percentage

%calculate errors
[dp_mep, dp_aep] = maxErrorPercent(dp_data_p, dp_data_v);
data = {};
data.dp_mep = dp_mep;
data.dp_aep = dp_aep;

%log errors
errorLog(data, start_col_v, end_col_v);
clear data
fclear();

