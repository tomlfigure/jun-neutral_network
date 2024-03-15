% 绝对路径
addpath("C:\\Users\\23360\\Desktop\\jun-neutral_network");  % 添加函数文件所在的绝对路径
addpath("subfolder");

x = 1;
y = f(x);
disp(y);

dp_data_v = [10, 2, 3, 4, 5];
dp_data_p = [1, 2.2, 2.8, 3.9, 4.7];

[mse, mae] = calculateErrors(dp_data_v, dp_data_p);
disp([mse, mae]);