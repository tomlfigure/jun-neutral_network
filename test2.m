
%%
close all;
clc;

%%
Y = @(y0, v0, g, t) y0 + (v0 .* t) + (g .* (t .^ 2)) ./ 2;
G1 = @(m, g, v0, t) (m ./ sqrt(1 - ((v0 + g .* t) ./ 2e2) .^ 2)) .* g;
G2 = @(m, g) m * g;

inputDataPath = "E:\\Projects\\jun-ai-prediction\\python\\machineLearning\\inputData.csv";

inputTable = readtable(inputDataPath);
% y0 v0 t m g
inputData = (table2array(inputTable))';
%%

g = 9.79;
actualY = Y(inputData(1,:),inputData(2,:),g,inputData(3,:));
actualG = G1(inputData(4, :), g, inputData(2, :), inputData(3, :));
outPutData = [actualY; actualG];

%%
% 归一化
actualG = actual

%%  设置网络
%节点个数
inputnum=4; % 输入层节点数量
hiddennum=10; % 隐含层节点数量
outputnum=2;  % 输出层节点数量

%% 构建BP神经网络
net=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm');% 建立模型，传递函数使用purelin，采用梯度下降法训练

W1= net. iw{1, 1};                  %输入层到中间层的权值
B1 = net.b{1};                      %中间各层神经元阈值

W2 = net.lw{2,1};                   %中间层到输出层的权值
B2 = net. b{2};                     %输出层各神经元阈值

%%  网络参数配置（ 训练次数，学习速率，训练目标最小误差等）
net.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net.trainParam.lr=0.01;             % 学习速率，这里设置为0.01
net.trainParam.goal=0.00001;        % 训练目标最小误差，这里设置为0.00001






