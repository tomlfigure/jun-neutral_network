%%
clear all

%% 初始化数据, 加载ps密钥
[data_x, dp_data, bpt_data, te_data, ce_data] = loadData(1, 1046);
[data_x_n, psx] = mapminmax(data_x, 0, 1);
[dp_data_n, psdp] = mapminmax(dp_data, 0, 1);
[bpt_data_n, psbpt] = mapminmax(bpt_data, 0, 1);
[te_data_n, pste] = mapminmax(te_data, 0, 1);
[ce_data_n, psce] = mapminmax(ce_data, 0, 1);
%用于记录网络的版本号
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce');
fclear();

% 模型训练
load('data\\keyData.mat');
[layers, options] = networkArgus();
rng('default')  %固定随机化，便于调参
% create the networks
start_col = 1;
end_col = 300;

version_id = 0;
save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce','version_id');

[data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col);

net_dp = trainNetwork(data_x_n, dp_data_n, layers, options);
net_bpt = trainNetwork(data_x_n, bpt_data_n, layers, options);
net_te = trainNetwork(data_x_n, te_data_n, layers, options);
net_ce = trainNetwork(data_x_n, ce_data_n, layers, options);

logMessage = sprintf("initialize the networks by data from %d to %d cols", start_col, end_col);
trainLog(logMessage, options);

%保存网络，清理训练数据
save('data\\net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
fclear();

%%
% 继续训练
for start_col
   
    end_col = start_col + 99;
    %加载训练数据
    [data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col);
    
    %训练网络
    net_dp = trainNetwork(data_x_n, dp_data_n, net_dp.Layers, options);
    net_bpt = trainNetwork(data_x_n, bpt_data_n, net_bpt.Layers, options);
    net_te = trainNetwork(data_x_n, te_data_n, net_te.Layers, options);
    net_ce = trainNetwork(data_x_n, ce_data_n, net_ce.Layers, options);
    
    %log
    logMessage = sprintf("continue training the network from %d to %d cols", start_col, end_col);
    trainLog(logMessage, options);
    
    %保存网络，清理训练数据
    save('data\\net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
    fclear();
    
    % 导入神经网络
    load('data\\net.mat');
    
    %%预测数据
    end_col = 330;
    start_col_v = end_col + 1;
    end_col_v = start_col_v + 30;
    [data_x_v dp_data_v bpt_data_v te_data_v ce_data_v] = loadData(start_col_v, end_col_v);
    
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
    save("data\\result_display.mat", "dp_data_p", "bpt_data_p", "te_data_p",...
        "ce_data_p", "dp_data_v", "bpt_data_v", "te_data_v", "ce_data_v", "start_col_v",...
        "end_col_v");
    fclear();
    
    % 误差分析
    
    % actual -> dp_data_v
    % predict -> dp_data_p
    
    % 导入预测结果
    load('data\\result_display.mat');
    
    % max error percentage
    
    %calculate errors
    [dp_mep, dp_aep] = maxErrorPercent(dp_data_p, dp_data_v);
    [bpt_mep, bpt_aep] = maxErrorPercent(bpt_data_p, bpt_data_v);
    [te_mep, te_aep] = maxErrorPercent(te_data_p, te_data_v);
    [ce_mep, ce_aep] = maxErrorPercent(ce_data_p, ce_data_v);
    data = {};
    data.dp_mep = dp_mep;
    data.dp_aep = dp_aep;
    data.bpt_mep = bpt_mep;
    data.bpt_aep = bpt_aep;
    data.te_mep = te_mep;
    data.te_aep = te_aep;
    data.ce_mep = ce_mep;
    data.ce_aep = ce_aep;
    
    %log errors
    errorLog(data, start_col_v, end_col_v);
    clear data
    fclear();
end

%% 保存最终的预测数据和网络
save('result\\net.mat', "net_dp", "net_bpt", "net_te", "net_ce");
%% 测试预测
load('data\\keyData.mat');
load('result\\net.mat');
start_col_v = 1031;
end_col_v = 1046;
[data_x_v dp_data_v bpt_data_v te_data_v ce_data_v] = loadData(start_col_v, end_col_v);
% normalize
data_x_v_n = mapminmax('apply', data_x_v, psx);

% predict
dp_data_p_n = predict(net_dp, data_x_v_n);
bpt_data_p_n = predict(net_bpt, data_x_v_n);
te_data_p_n = predict(net_te, data_x_v_n);
ce_data_p_n = predict(net_ce, data_x_v_n);

% reverse data
dp_data_p = mapminmax('reverse', dp_data_p_n, psdp);
te_data_p = mapminmax('reverse', te_data_p_n, pste);

clear data_x_v_n dp_data_p_n bpt_data_p_n te_data_p_n ce_data_p_n

%%

dp_r = calR(dp_data_v, dp_data_p);
te_r = calR(te_data_v, te_data_p);
ce_r = calR(ce_data_v, ce_data_p);
bpt_r = calR(bpt_data_v, bpt_data_p);
disp(dp_r);
disp(te_r);
disp(ce_r);
disp(bpt_r);



