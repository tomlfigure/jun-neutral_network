

function [maxPercent, avgPercent] = maxErrorPercent(data1, data2)
    % 确保数据长度相同
    assert(length(data1) == length(data2), '数据长度不相同');
    
    % 计算百分比差异
    percentDiff = abs((data1 - data2) ./ data1) * 100;
    
    % 计算最大百分比差异
    maxPercent = max(percentDiff);
    
    % 计算平均百分比差异
    avgPercent = mean(percentDiff);
end