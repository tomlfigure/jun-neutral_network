

function y = trainLog(message, options)
    load("data\\keyData.mat");
    filename = 'log.txt';
    fileID = fopen(filename, 'a');  % 打开文件以进行写入（如果文件不存在则创建）
    version_id = version_id + 1;
    fprintf(fileID, 'version %d:\n', version_id);
    fprintf(fileID, "LearnRateDropPeriod : %d\n", options.LearnRateDropPeriod );
    fprintf(fileID, "MaxEpochs : %d\n", options.MaxEpochs);
    fprintf(fileID, "InitialLearnRate : %d\n", options.InitialLearnRate);
    fprintf(fileID, "L2Regularization : %d\n", options.L2Regularization);
    fprintf(fileID, "LearnRateDropFactor : %d\n\n", options.LearnRateDropFactor);
    fprintf(fileID, '%s\n', message);  % 将文本写入文件，%s 表示字符串，\n 表示换行
    currentDateTime = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
    fprintf(fileID, "%s\n\n\n\n", currentDateTime);
    fclose(fileID);  % 关闭文件
    save('data\\keyData.mat', 'psx', 'psdp', 'psbpt',"pste", 'psce','version_id');
end