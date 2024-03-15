function [data_x, dp_data, bpt_data, te_data, ce_data] = loadData(start_col, end_col)
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
end

