function annolist2 = util_crop_data_train(p, annolist, img_id, rectidxs)

saveDir = p.saveDir;
% 计算scale的高度
refHeight = p.refHeight;
% crop外扩的参数
deltaCrop = p.deltaCrop;
% 是否需要考虑crop内部是否有其他人的关节，如果有则内缩一下
bCropIsolated = p.bCropIsolated;

if (~exist(saveDir, 'dir'))
    mkdir(saveDir);
end

annolist2 = [];
ncrops = 1;

delta_x = deltaCrop;
delta_y = deltaCrop;

numImgs = 0;

for imgidx = 1:length(annolist)
    % 打印进度
    fprintf('.');
    
    % 某个图像的人的框框
    rect = annolist(imgidx).annorect;
    rect2 = annolist(imgidx).annorect;
    % 读取图像
    img = imread([p.imageDir annolist(imgidx).image.name]);
    
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
        
        % 大小为3x16
        pointsAll = [];
        % 获取当前框框内的人的关节点
        points = rect(ridx).annopoints.point;
        % 遍历每一个关节点，存起来
        for pid = 1:length(points)
            pp = [points(pid).x points(pid).y];
            pointsAll = [pointsAll; pp];
        end
        
        % 得到最大的和做小的坐标x,y
        minX = min(pointsAll(:,1));
        maxX = max(pointsAll(:,1));
        minY = min(pointsAll(:,2));
        maxY = max(pointsAll(:,2));
        
        % 如果设定参考高度则重新计算scale
        % 然后缩放图像
        if (refHeight > 0)
            sc = rect(ridx).scale*200/refHeight;
            %% rescale image
            img_sc = imresize(img,1/sc,'bicubic');
        else
            % 如果没有设定参考高度则将扩展crop也乘以scale这样变成缩放之后的scale
            sc = 1.0;
            img_sc = img;
            delta_x = round(deltaCrop*rect(ridx).scale);
            delta_y = round(deltaCrop*rect(ridx).scale);
        end
        
        posX1 = round(minX/sc);
        posX2 = round(maxX/sc);
        posY1 = round(minY/sc);
        posY2 = round(maxY/sc);
            
        %% crop image
        % 扩展一下crop的区域
        x1_new = round(max(1, posX1 - delta_x));
        x2_new = round(min(size(img_sc, 2), posX2 + delta_x));
        
        y1_new = max(1, posY1 - delta_y);
        y2_new = min(size(img_sc, 1), posY2 + delta_y);
        
        % bCropIsolated表示是否需要考虑其他人的关节，尽量保证不把其他人的关节crop到图片中
        if (bCropIsolated && length(rect)>1)
            %% compute the closest annotated joint positions of other people
            % 把其他所有的相关的人的关节坐标都放到points2All
            points2All = [];
            % 这里已经去掉了ridx
            for ridx2 = [1:ridx-1 ridx+1:length(rect2)]
                if (isfield(rect2(ridx2),'annopoints') && ~isempty(rect2(ridx2).annopoints) && ...
                    isfield(rect2(ridx2).annopoints,'point') && ~isempty(rect2(ridx2).annopoints.point))
                    points2 = rect2(ridx2).annopoints.point;
                    for pid = 1:length(points2)
                        pp = [points2(pid).x points2(pid).y];
                        points2All = [points2All; pp];
                    end
                end
            end
            % 如果存在其他的人的关节坐标,则将靠近左上角和右下角的坐标都搞到
            if (~isempty(points2All))
                % 将所有关节点都进行缩放
                points2All = points2All./sc;
                % 所有相邻关节点的x < 左上角的x
                % 与左上角最靠近的x
                d = points2All(:,1) - posX1; idx = find(d<0);
                [~,id] = max(d(idx)); posX1other = points2All(idx(id),1);
                % 所有相邻关节点的y < 左上角的x
                % 与左上角最靠近的y
                d = points2All(:,2) - posY1; idx = find(d<0);
                [~,id] = max(d(idx)); posY1other = points2All(idx(id),2);
                % 所有相邻关节点的x > 右下角的x
                % 与右下角最靠近的x
                d = posX2 - points2All(:,1); idx = find(d<0);
                [~,id] = max(d(idx)); posX2other = points2All(idx(id),1);
                % 所有相邻关节点的y > 右下角的y
                % 与右下角最靠近的y
                d = posY2 - points2All(:,2); idx = find(d<0);
                [~,id] = max(d(idx)); posY2other = points2All(idx(id),2);
                
                % 靠近左上角和右下角的其他人的关节点的坐标向外扩10个像素
                % delta2就是10个像素乘以scale之后的结果
                if (refHeight > 0)
                    delta2 = refHeight/200*10;
                else
                    delta2 = rect(ridx).scale*10;
                end
                % 将其他人的关节点的坐标向外扩10个像素（经过缩放的10个像素）
                if (~isempty(posX1other))
                    x1_new = round(max(x1_new, posX1other+delta2));
                end
                if (~isempty(posX2other))
                    x2_new = round(min(x2_new, posX2other-delta2));
                end
            end
        end
        % 将人抠出来，把与当前人的坐标也考虑进去了保证没人在框框里面
        imgCrop = img_sc(y1_new:y2_new, x1_new:x2_new,1);
        imgCrop(:,:,2) = img_sc(y1_new:y2_new, x1_new:x2_new,2);
        imgCrop(:,:,3) = img_sc(y1_new:y2_new, x1_new:x2_new,3);
        
        %% save image
        % 保存图片到文件
        % padZeros是在img_id(imgidx)之前填充几个0，保证是5位数字
        fname = [saveDir '/im' padZeros(num2str(img_id(imgidx)),5) '_' num2str(ridx) '.png'];
        imwrite(imgCrop, fname);
        image_size = [size(imgCrop, 1), size(imgCrop, 2)];
        
        % 文件名是T_00001_1.png
        % T_图像编号_人的编号.png
        fnameT = [saveDir '/T_' padZeros(num2str(img_id(imgidx)),5) '_' num2str(ridx)];
        % T是个变换矩阵？
        % x1_new和y1_new是左上角坐标
        T = sc*[1 0 x1_new; 0 1 y1_new; 0 0 1];
        save(fnameT,'T');
        
        %% transfer annotations
        % 把当前框框中人的关节点弄到crop之后的坐标上去（也变换一下）
        for pid = 1:length(points)
            points(pid).x = (points(pid).x)/sc - x1_new + 1;
            points(pid).y = (points(pid).y)/sc - y1_new + 1;
        end
        
        % 将关节点坐标存进去
        rect(ridx).annopoints.point = points;
        % 将当前头部框框的左上角和有效做坐标也变换一下
        rect(ridx).x1 = rect(ridx).x1/sc - x1_new + 1;
        rect(ridx).y1 = rect(ridx).y1/sc - y1_new + 1;
        rect(ridx).x2 = rect(ridx).x2/sc - x1_new + 1;
        rect(ridx).y2 = rect(ridx).y2/sc - y1_new + 1;
        % 将原始数据人的中心也变换一下
        rect(ridx).objpos.x = rect(ridx).objpos.x/sc - x1_new + 1;
        rect(ridx).objpos.y = rect(ridx).objpos.y/sc - y1_new + 1;
        
        % annotlist2是装处理之后的数据的
        % image.name是经过crop之后的图像名称
        % imgnum是图像的顺序编号
        % annorect是当前人的框框以及人的中心点以及人的关节点的坐标
        % image_size是经过crop之后的图像大小
        if (isempty(annolist2))
            annolist2.image.name = fname;
            annolist2.imgnum = 1;
            annolist2.annorect = rect(ridx);
            annolist2.image_size = image_size;
        else
            ncrops = ncrops + 1;
            annolist2(ncrops).image.name = fname;
            annolist2(ncrops).imgnum = ncrops;
            annolist2(ncrops).annorect = rect(ridx);
            annolist2(ncrops).image_size = image_size;
        end
        
    end
    
    % 处理100个就显示一下处理的个数
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,length(annolist));
    end
    
end
fprintf('\ndone\n');

%assert(numImgs == length(annolist2));

end