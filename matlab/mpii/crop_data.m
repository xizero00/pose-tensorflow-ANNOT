function annolist = crop_data(p)

fprintf('crop_data()\n');

bTrain = p.bTrain;
refHeight = p.refHeight;
deltaCrop = p.deltaCrop;
bSingle = p.bSingle;
bCropIsolated = p.bCropIsolated;
bMulti = p.bMulti;
bObjposOffset = p.bObjposOffset;
saveDir = p.saveDir;
DATASET = p.dataset;
annolist = DATASET.annolist;

func_crop_data_test = @util_crop_data;
func_crop_data_train = @util_crop_data_train;

fprintf('bTrain: %d\n',bTrain);
fprintf('refHeight: %d\n',refHeight);
fprintf('deltaCrop: %d\n',deltaCrop);
fprintf('bSingle: %d\n',bSingle);
fprintf('bCropIsolated: %d\n',bCropIsolated);
fprintf('bMulti: %d\n',bMulti);
fprintf('bObjposOffset: %d\n',bObjposOffset);

if (bTrain)
    split = 'train-v15';
else
    split = 'test-v15';
end

%{
if (bSingle)
    mode = 'singlePerson';
    rectidxs = DATASET.single_person;
else
    mode = 'multPerson';
    for imgidx = 1:length(DATASET.single_person)
        DATASET.mult_person{imgidx} = [DATASET.mult_person{imgidx} DATASET.borderline_person{imgidx}'];
    end
    rectidxs = DATASET.mult_person;
end
%}


if (bSingle)
    mode = 'singlePerson';
    % 每个图片里面单个人的索引
    rectidxs = DATASET.single_person;
else
	% 如果是多人训练模式
    mode = 'multPerson';
    rectidxs = cell(size(DATASET.single_person));
    for imgidx = 1:length(DATASET.single_person)
    	% 单人的框框的id
        single_person = DATASET.single_person{imgidx};
        single_person = reshape(single_person, [1, length(single_person)]);
        % setdiff(A,B)是找出在A不在B的元素
        % 这里就是找出多人的框框的id
        rectidxs{imgidx} = setdiff(1:length(annolist(imgidx).annorect), single_person);
    end
end

% annotlist的文件名为annotlist-singlePerson-h400.mat
annolistFullName = [saveDir '/annolist-' mode '-h' num2str(refHeight) '.mat'];
if exist(annolistFullName, 'file') == 2
    % 如果有才加载，否则就进行处理了
    load(annolistFullName, 'annolist');
    return;
end

% 去除多人框框id中为空的图片
imgidxs1 = find(cellfun(@isempty,rectidxs) == 0);
% 去除非训练的图片
imgidxs2 = find(DATASET.img_train == bTrain);
% 取交集
imgidxs = intersect(imgidxs1,imgidxs2);

% 实际上这里的imgidxs并没有去除那些有头部的坐标的但是没有关节坐标的
imgidxs_sub = imgidxs;

% set scale
% 给annotlist加入尺度，尺度的计算公式是
% scale = 人的头部大小*8/200
annolistSubset = util_set_scale(annolist(imgidxs_sub),200);

% crop images
if (~bTrain) % test
    assert(false);
    annolist = func_crop_data_test(annolistSubset, saveDir, refHeight, imgidxs_sub, rectidxs(imgidxs_sub), DATASET.groups(imgidxs_sub), deltaCrop, deltaCrop, 'symmetric', bObjposOffset);
else % train
    if (bMulti)
        assert(false);
        annolist = func_crop_data_test(annolistSubset, saveDir, refHeight, imgidxs_sub, rectidxs(imgidxs_sub), DATASET.groups(imgidxs_sub), deltaCrop, deltaCrop, 'symmetric', false);
    else
        % 对所有训练的图像（imgidxs_sub指定）和所有单人的框框（rectidxs(imgidxs_sub)指定）
        % 进行crop以及缩放并且计算crop之后的坐标
        annolist = func_crop_data_train(p, annolistSubset, imgidxs_sub, rectidxs(imgidxs_sub));
    end
end

save(annolistFullName, 'annolist');
