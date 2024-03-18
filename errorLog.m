

function y = errorLog(data, start_col, end_col)
    load("data\\keyData.mat");
    filename = 'errorLog.txt';
    fileID = fopen(filename, 'a');  % 打开文件以进行写入（如果文件不存在则创建）
    fprintf(fileID, 'the errors based on version %d:\n', version_id);
    fprintf(fileID, 'the cols from %d to %d have been used for prediction\n', start_col, end_col);
    fprintf(fileID, 'maxErrorPercent:\n\n');
    fprintf(fileID, 'drop pressure:\n');
    fprintf(fileID, '%f, %f\n\n', data.dp_mep, data.dp_aep);
    fprintf(fileID, 'baseplate temperature:\n');
    fprintf(fileID, '%f, %f\n', data.bpt_mep, data.bpt_aep);
    fprintf(fileID, '\n');
    fprintf(fileID, 'temperature even:\n');
    fprintf(fileID, '%f, %f\n', data.te_mep, data.te_aep);
    fprintf(fileID, '\n');
    fprintf(fileID, 'coefficent:\n');
    fprintf(fileID, '%f, %f\n\n', data.ce_mep, data.ce_aep);
    currentDateTime = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
    fprintf(fileID, "%s\n\n\n\n\n", currentDateTime);
    fclose(fileID);  % 关闭文件
end