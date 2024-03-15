function [mse, mae] = calculateErrors(dp_data_v, dp_data_p)
    % 计算均方误差（MSE）
    mse = mean((dp_data_v - dp_data_p).^2);
    
    % 计算平均绝对误差（MAE）
    mae = mean(abs(dp_data_v - dp_data_p));
end