

function R = calY(data1, data2)
    R_ = corrcoef(data1, data2);
    R = R_(1, 2);
end