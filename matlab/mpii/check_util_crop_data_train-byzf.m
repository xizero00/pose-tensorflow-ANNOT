function annolist2 = check_util_crop_data_train(p, annolist, img_id, rectidxs)


ncrops = 0;

numImgs = 0;

for imgidx = 1:length(annolist)
    % 打印进度
    fprintf('.');
    
    % 某个图像的人的框框
    rect = annolist(imgidx).annorect;

    
    % 遍历当前图片的每个框框
    for ridx = 1:length(rect)
        
        % 判断当前框框是否是需要处理的框框
        if (~isempty(setdiff(ridx,rectidxs{imgidx})))
            continue;
        end
        
        numImgs = numImgs + 1;
        
        % 判断是否存在annopoints结构，以及这个结构不为空
        if ~isfield(rect(ridx), 'annopoints') || isempty(rect(ridx).annopoints)
            continue;
        end


        ncrops = ncrops + 1;

        
    end
    

    
end
fprintf('\ndone\n');
annolist2 = [];
assert(numImgs == ncrops);

end