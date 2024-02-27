

%%
clc
clear all


%% 定义目标函数, 设置初始训练数据，创建BP神经网络

%定义目标函数
f = @(x, y) x .^ 2 - 2 * y;

f = @(x, y) x .^ 2 - 2 * y;

f = 
%设置训练数据
x_test = 2:2:20;
y_test = 2:2:20;
z_test_train = f(x_test, y_test);
z_test_train = f(x_test, y_test)

%数据归一化处理
[x_test_train, psx] = mapminmax(x_test, 0, 1);
[y_test_train, psy] = mapminmax(y_test, 0, 1);
[x_test_train, psx] = mapminmax(x_test, 0, 1)


%将x，y以列为组进行排列
data_test_train = [x_test_train; y_test_train];


%创建神经网络, 设置网络层级数为9
net = newff(data_test_train, z_test_train, 9);

%%

%设置训练参数
net.trainParam.goal = 1e-3;
net.trainParam.epochs = 100;
net.trainParam.lr = 0.01;


%设置新的训练数据

x_new_train = 1:1:1000;
y_new_train = 1:1:1000;

[x_new_train_g, psx] = mapminmax(x_new_train, 0, 1);
[y_new_train_g, psy] = mapminmax(y_new_train, 0, 1);

z_new_train = f(x_new_train, y_new_train);

data_new_train = [x_new_train_g; y_new_train_g];

%用初始数据对网络进行100轮训练
net = train(net, data_new_train, z_new_train);

%%

%设置验证数据
x_test_verify = 4:3:20;
y_test_verify = 4:3:20;
    % 计算验证数据的实际值
z_test_verify_actual = f(x_test_verify, y_test_verify);


%验证数据归一化处理
x_test_verify_g = mapminmax('apply', x_test_verify, psx);
y_test_verify_g = mapminmax('apply', y_test_verify, psy);


%进行模型预测
z_test_verify = sim(net, [x_test_verify_g; y_test_verify_g]);

%计算实际值与预测值的误差
error = abs(z_test_verify_actual - z_test_verify);

percentage = (error ./ z_test_verify_actual) .* 100;


%% 对结果进行展示
%以列的形式进行展示
results = [z_test_verify_actual' , z_test_verify', error', percentage'];

disp(['actual   ', 'predicted   ', 'error   ', 'error_percentage']);

disp(results);




disp(results)
% 固定误差 3-5 %

% 将数据分组训练 -> 一组练， 一组验证

% 先找demos ,能用则用, 比较 ,读懂

% 1000样本 5,6个参数 2-3 out 找个类似的  搞清楚参数

% 3 nets







