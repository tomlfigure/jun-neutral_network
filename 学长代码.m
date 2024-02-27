%% SOH validation
% 训练集设置  
%tic
clear train_soh_feature train_soh_y data_pro2 data_pro1 span_cr

train_soh_y=cell_rul_mit_data(1:train_num,4);

fc_num=5;  %对应特征组合

for i=1:train_num 
    switch fc_num
        case 1  % we propose
            train_soh_feature{i,1}=[feature_rawdata(i).dfeature01(:,1),feature_rawdata(i).dfeature01(:,2),feature_rawdata(i).dfeature01(:,5),feature_rawdata(i).dfeature01(:,6),feature_rawdata(i).dfeature01(:,10)];
        case 2  % top five largest PCC
            train_soh_feature{i,1}=[feature_rawdata(i).dfeature01(:,1),feature_rawdata(i).dfeature01(:,9),feature_rawdata(i).dfeature01(:,8),feature_rawdata(i).dfeature01(:,10),feature_rawdata(i).dfeature01(:,6)];
        case 3  % reference (PCC 0.9)
            train_soh_feature{i,1}=[feature_rawdata(i).dfeature01(:,1),feature_rawdata(i).dfeature01(:,10),feature_rawdata(i).dfeature01(:,5),feature_rawdata(i).dfeature01(:,3)];
        case 4  % abs PCC more than 0.9 
            train_soh_feature{i,1}=[feature_rawdata(i).dfeature01(:,1),feature_rawdata(i).dfeature01(:,8),feature_rawdata(i).dfeature01(:,9),feature_rawdata(i).dfeature01(:,10)];
        case 5  % all features
            train_soh_feature{i,1}=[feature_rawdata(i).dfeature01(:,1),feature_rawdata(i).dfeature01(:,2),feature_rawdata(i).dfeature01(:,3),feature_rawdata(i).dfeature01(:,4),feature_rawdata(i).dfeature01(:,5),feature_rawdata(i).dfeature01(:,6),feature_rawdata(i).dfeature01(:,7),feature_rawdata(i).dfeature01(:,8),feature_rawdata(i).dfeature01(:,9),feature_rawdata(i).dfeature01(:,10)];
    end
    train_soh_y{i,3}=train_soh_y{i,1}(1:length(train_soh_feature{i,1}));
end

for i=1:train_num  %删去前两行
    train_soh_y{i,3}(1:2,:)=[];
    train_soh_feature{i,1}(1:2,:)=[];
    clear delete_pro
end

% for i=1:train_num
%     span_cr(i,1)=max(train_soh_y{i,3}(:,1))-min(train_soh_y{i,3}(:,1));  %数据跨度统一
%     span_cr(i,2)=max(train_soh_y{i,3}(:,2))-min(train_soh_y{i,3}(:,2));
% end
% 
% span_cr1(1)=mean(span_cr(:,1));
% span_cr1(2)=mean(span_cr(:,2));
% clear span_cr
% 
% for i=1:train_num
%     train_soh_y{i,4}=[train_soh_y{i,3}(:,1)/span_cr1(1)-0.4,train_soh_y{i,3}(:,2)/span_cr1(2)-0.1];
% end


% for i=1:train_num  %删去全零元素行
%     train_soh_y{i,1}(all(train_soh_feature{i,1}==0,2),:)=[];
%     train_soh_feature{i,1}(all(train_soh_feature{i,1}==0,2),:)=[];
%     clear delete_pro
% end

for i=1:train_num  %修补空缺数据
    for j=1:size(train_soh_feature{i,1},1)
        if sum(train_soh_feature{i,1}(j,:))==0
            for k=1:100
                if j+k<=size(train_soh_feature{i,1},1)
            if sum(train_soh_feature{i,1}(j+k,:))~=0
                for f=1:size(train_soh_feature{i,1},2)
                train_soh_feature{i,1}(j:j+k-1,f)=interp1([1,k+2],[train_soh_feature{i,1}(j-1,f);train_soh_feature{i,1}(j+k,f)]',[2:k+1],'linear')';
                end
            end
            end
            end
        end
    end    
end

confusion_num=[36 12 87	65 5 83 4 59 32 25 40 78 34 73 61 9 79 49 53 48	28 14 18 46	62 26 84 60 63 76 42 20 47 16 13 70 56 85 72 2 75 41 58 77 64 44 33 24 11 69 68 57 17 6 30 52 3 10 82 51 19 55 21 22 23 1 37 7 74 15 43 38 86 29 54 39 50 71 66 45 27 8 80 67 81 35 31];

for i=1:87
    train_soh_feature{i,2}=train_soh_feature{confusion_num(i),1};
    train_soh_y{i,4}=train_soh_y{confusion_num(i),3};
end

xdata_pro=train_soh_feature(:,2);
ydata_pro=train_soh_y(:,4); 

% for n=1:train_num
%     data_pro1=xdata_pro{n,1};
%     for m=1:size(data_pro1,1)
%     if m<5
%         repeat_data=repmat(data_pro1(m,:),5,1);
%         repeat_data=repeat_data(:);
%     end
% 
%     if m>4
%         repeat_data=data_pro1(m-4:m,:);
%         repeat_data=repeat_data(:);
%     end
%     data_pro2(m,:)=repeat_data;
%     end
%     xdata_pro{n,2}=data_pro2;
%     train_soh_feature{n,3}=data_pro2;
%     clear data_pro1 data_pro2
% end

for i=1:train_num  %数据结构转变，以适应sw_sampling函数
    xdata_pro{i,2}=xdata_pro{i,1}';
end

train_soh_feature(:,3)=sw_sampling(xdata_pro(:,2),5);

train_xdata_s=[];
train_ydata_s=[];

length_1=size(train_soh_feature{1,1},2);
for i=1:train_num
    data_pro3=repmat(0,length_1*5,100);
    data_pro31=repmat(train_soh_feature{i,3}(1,:)',1,100);
    data_pro4=repmat(0,length_1*5,100);
    data_pro41=repmat(train_soh_feature{i,3}(end,:)',1,100);
    train_xdata_s=[train_xdata_s,data_pro3,data_pro31,train_soh_feature{i,3}',data_pro41,data_pro4];
%     train_xdata_s=[train_xdata_s,data_pro3,xdata_pro{i,2}',data_pro4];
    data_pro5=repmat(0,1,100);
    data_pro51=repmat(ydata_pro{i,1}(1,:)',1,100);
    data_pro6=repmat(0,1,100);
    data_pro61=repmat(ydata_pro{i,1}(end,:)',1,100);
    train_ydata_s=[train_ydata_s,data_pro5,data_pro51,ydata_pro{i,1}',data_pro61,data_pro6];
%     train_ydata_s=[train_ydata_s,data_pro5,ydata_pro{i,1}',data_pro6];
end

clear ydata_pro xdata_pro data_pro3 data_pro31 data_pro4 data_pro41 data_pro5 data_pro51 data_pro6 data_pro61
%% 模型训练
tic
clear net6
InputSize = size(train_xdata_s,1);
numHiddenUnits = 40;
FC1=240;
FC2=40;
MaxEpochs=2000;
InitialLearnRate=0.0025;
OutputSize = 1;
layers = [ ...
    sequenceInputLayer(InputSize,'Name','input1')  %输入层
    %fullyConnectedLayer(InputSize*4)
    lstmLayer(120,'Name','lstm1')  %LSTM神经网络 
    lstmLayer(60,'Name','lstm2')  %LSTM神经网络
    fullyConnectedLayer(30,'Name','fc1')
    fullyConnectedLayer(20,'Name','fc2')%全连接层
    fullyConnectedLayer(10,'Name','fc3')
    fullyConnectedLayer(5,'Name','fc4')
    fullyConnectedLayer(OutputSize,'Name','fc5')  %输出层
    regressionLayer('Name','re_1')];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',MaxEpochs, ...
    'MiniBatchSize',1,...
    'Shuffle','never',...
    'GradientThreshold',1, ...
    'InitialLearnRate',InitialLearnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',50,...
    'LearnRateDropFactor',0.9,...
    'L2Regularization',0.0002,...
    'Verbose',false,...
    'Plots','training-progress');
rng('default')  %固定随机化，便于调参
net6 = trainNetwork(train_xdata_s,train_ydata_s,layers,options);

clear InputSize FC1 FC2 numHiddenUnits MaxEpochs InitialLearnRate OutputSize train_xdata_s train_ydata_s
toc

%% 验证集设置
clear valid_soh_feature valid_soh_y test_xdata_s test_ydata_s data_pro2 data_pro1
valid_soh_y=cell_rul_validation(1:valid_num,1);  %原始数据
%valid_soh_y1=cell_rul_validation(1:valid_num,4);  %处理后的数据，用于p10

for i=1:valid_num
    switch fc_num
        case 1  % we propose
            valid_soh_feature{i,1}=[feature_validation(i).dfeature01(:,1),feature_validation(i).dfeature01(:,2),feature_validation(i).dfeature01(:,5),feature_validation(i).dfeature01(:,6),feature_validation(i).dfeature01(:,10)];
        case 2  % top five largest PCC
            valid_soh_feature{i,1}=[feature_validation(i).dfeature01(:,1),feature_validation(i).dfeature01(:,9),feature_validation(i).dfeature01(:,8),feature_validation(i).dfeature01(:,10),feature_validation(i).dfeature01(:,6)];
        case 3  % reference (PCC 0.9)
            valid_soh_feature{i,1}=[feature_validation(i).dfeature01(:,1),feature_validation(i).dfeature01(:,10),feature_validation(i).dfeature01(:,5),feature_validation(i).dfeature01(:,3)];
        case 4  % abs PCC more than 0.9 
            valid_soh_feature{i,1}=[feature_validation(i).dfeature01(:,1),feature_validation(i).dfeature01(:,8),feature_validation(i).dfeature01(:,9),feature_validation(i).dfeature01(:,10)];
        case 5  % all features
            valid_soh_feature{i,1}=[feature_validation(i).dfeature01(:,1),feature_validation(i).dfeature01(:,2),feature_validation(i).dfeature01(:,3),feature_validation(i).dfeature01(:,4),feature_validation(i).dfeature01(:,5),feature_validation(i).dfeature01(:,6),feature_validation(i).dfeature01(:,7),feature_validation(i).dfeature01(:,8),feature_validation(i).dfeature01(:,9),feature_validation(i).dfeature01(:,10)];            
    end
    valid_soh_y{i,2}=valid_soh_y{i,1}(1:length(valid_soh_feature{i,1}));
    %valid_soh_y1{i,2}=valid_soh_y1{i,1}(1:length(valid_soh_feature{i,1}));
end

for i=1:valid_num  %删去前两行
    valid_soh_y{i,2}(1:2,:)=[];
    %valid_soh_y1{i,2}(1:2,:)=[];
    valid_soh_feature{i,1}(1:2,:)=[];
    clear delete_pro
end

% for i=1:valid_num
%     valid_soh_y1{i,4}=[valid_soh_y1{i,3}(:,1)/span_cr1(1)-0.4,valid_soh_y1{i,3}(:,2)/span_cr1(2)-0.1];
% end

for i=1:valid_num  %修补空缺数据
    for j=1:size(valid_soh_feature{i,1},1)
        if sum(valid_soh_feature{i,1}(j,:))==0
            for k=1:100
                if j+k<=size(valid_soh_feature{i,1},1)
            if sum(valid_soh_feature{i,1}(j+k,:))~=0
                for f=1:size(valid_soh_feature{i,1},2)
                valid_soh_feature{i,1}(j:j+k-1,f)=interp1([1,k+2],[valid_soh_feature{i,1}(j-1,f);valid_soh_feature{i,1}(j+k,f)]',[2:k+1],'linear')';
                end
            end
            end
            end
        end
    end    
end

xdata_pro=valid_soh_feature;
ydata_pro=valid_soh_y(:,2); 

for i=1:valid_num  %数据结构转变，以适应sw_sampling函数
    xdata_pro{i,2}=xdata_pro{i,1}';
end

xdata_pro(:,3)=sw_sampling(xdata_pro(:,2),5);

test_xdata_s=xdata_pro(:,3);
test_ydata_s=ydata_pro;

clear ydata_pro xdata_pro




