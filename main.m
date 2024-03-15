%%
clear all

%% 初始化数据, 加载ps密钥
[data_x, dp_data, bpt_data, te_data, ce_data] = loadData(1, 1046);
[data_x_n, psx] = mapminmax(data_x, 0, 1);
[dp_data_n, psdp] = mapminmax(dp_data, 0, 1);
[bpt_data_n, psbpt] = mapminmax(bpt_data, 0, 1);
[te_data_n, pste] = mapminmax(te_data, 0, 1);
[ce_data_n, psce] = mapminmax(ce_data, 0, 1);
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce');
fclear();

%% 模型训练

[data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(1, 300);
[layers, options] = networkArgus();
rng('default')  %固定随机化，便于调参
net_dp = trainNetwork(data_x_n, dp_data_n, layers, options);
net_bpt = trainNetwork(data_x_n, bpt_data_n, layers, options);
net_te = trainNetwork(data_x_n, te_data_n, layers, options);
net_ce = trainNetwork(data_x_n, ce_data_n, layers, options);

%保存网络，清理训练数据
save('net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
fclear();


%% 继续训练
start_col = 301;
end_col = 350;

%加载训练数据
[data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col);

%训练网络
net_dp = trainNetwork(data_x_n, dp_data_n, net_dp.Layers, options);
net_bpt = trainNetwork(data_x_n, bpt_data_n, net_bpt.Layers, options);
net_te = trainNetwork(data_x_n, te_data_n, net_te.Layers, options);
net_ce = trainNetwork(data_x_n, ce_data_n, net_ce.Layers, options);

%保存网络，清理训练数据
save('net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
fclear();

%% 导入神经网络
load('net.mat');

%% 预测数据
start_col = 351;
end_col = 370;
[data_x_v dp_data_v bpt_data_v te_data_v ce_data_v] = loadData(start_col, end_col);

% normalize
data_x_v_n = mapminmax('apply', data_x_v, psx);

% predict
dp_data_p_n = predict(net_dp, data_x_v_n);
bpt_data_p_n = predict(net_bpt, data_x_v_n);
te_data_p_n = predict(net_te, data_x_v_n);
ce_data_p_n = predict(net_ce, data_x_v_n);

% reverse data
dp_data_p = mapminmax('reverse', dp_data_p_n, psdp);
bpt_data_p = mapminmax('reverse', bpt_data_p_n, psbpt);
te_data_p = mapminmax('reverse', te_data_p_n, pste);
ce_data_p = mapminmax('reverse', ce_data_p_n, psce);

% save the result data
save("result_display.mat", "dp_data_p", "bpt_data_p", "te_data_p", "ce_data_p", "dp_data_v", "bpt_data_v", "te_data_v", "ce_data_v");
fclear();

%% 误差分析

% actual -> dp_data_v
% predict -> dp_data_p
%% 导入预测结果
load('result_display.mat');
%% 相关系数
correlation = corrcoef(dp_data_v, dp_data_p);
disp(correlation);
%% 折线图
x = 1:1:20;
hold on
plot(x, dp_data_v);
plot(x, dp_data_p);
hold off

%% 散点图
scatter(dp_data_v, dp_data_p);
xlabel('源数据');
ylabel('预测数据');
title('源数据 vs 预测数据');

%% 误差图
x = 1:101;  % X轴数据，假设为101个数据点
error = dp_data_p - dp_data_v;  % 计算误差
errorbar(x, dp_data_v, error);
xlabel('数据点');
ylabel('数值');
title('源数据 vs 预测数据误差');







