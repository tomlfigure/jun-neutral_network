

function y = fclear()
evalin('base', 'clear table_x table_y data_x data_y dp_data bpt_data te_data ce_data');
evalin('base', 'clear data_x_n dp_data_n bpt_data_n te_data_n ce_data_n');
evalin('base', 'clear data_x_v data_x_v_n dp_data_p bpt_data_p te_data_p ce_data_p');
evalin('base', 'clear dp_data_v bpt_data_v te_data_v ce_data_v')
evalin('base', 'clear dp_data_p_n bpt_data_p_n te_data_p_n ce_data_p_n')
evalin('base', 'clear start_col end_col');
evalin('base', 'clear logMessage');
evalin('base', 'clear dp_mep dp_aep bpt_mep bpt_aep te_mep te_aep ce_mep ce_aep')
end




