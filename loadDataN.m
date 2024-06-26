

function [data_x_n, dp_data_n, bpt_data_n, te_data_n, ce_data_n] = loadDataN(start_col, end_col)
load("data\\keyData.mat");
[data_x, dp_data, bpt_data, te_data, ce_data] = loadData(start_col, end_col);
    data_x_n = mapminmax('apply', data_x, psx);
    dp_data_n = mapminmax('apply', dp_data, psdp);
    bpt_data_n = mapminmax('apply', bpt_data, psbpt);
    te_data_n = mapminmax('apply', te_data, pste);
    ce_data_n = mapminmax('apply', ce_data, psce);
    clear table_x table_y data_y dp_data bpt_data te_data ce_data
end