

%%
clc
clear all

%%

A = [1 2 3; 4 5 6];

 
%%

%定义目标函数
f = @(x, y) x .^ 2 - 2 * y;

%设置训练数据
x_test = 2:2:20;
y_test = 2:2:20;
z_test = f(x_test, y_test);

%数据归一化处理
[x_test_train, psx] = mapminmax(x_test, 0, 1);
[y_test_train, psy] = mapminmax(y_test, 0, 1);
[z_test_train, psz] = mapminmax(z_test, 0, 1);

%将x，y以列为组进行排列
data_test_train = [x_test_train; y_test_train];


%创建神经网络, 设置网络层级数为9
net = newff(data_test_train, z_test_train, 9);

%%

%设置训练参数
% net.trainParam.goal = 1e-3;
% net.trainParam.epochs = 100;
% net.trainParam.lr = 0.01;

%用初始数据对网络进行100轮训练
net = train(net,data_test_train, z_test_train);

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
z_test_verify_g = sim(net, [x_test_verify_g; y_test_verify_g]);

%将预测出的数据反归一化
z_test_verify = mapminmax('reverse', z_test_verify_g, psz);

%计算实际值与预测值的误差
error = abs(z_test_verify_actual - z_test_verify);

%以列的形式进行展示
results = [z_test_verify_actual' , z_test_verify', error'];










