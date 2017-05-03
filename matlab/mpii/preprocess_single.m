function preprocess_single(datasetDir, datasetName, saveDir, imageDir)
% datasetDir 是数据集目录
% datasetName 是数据信息的mat文件名
% saveDir 保存处理之后的目录
% imageDir 图像的目录
if (nargin < 2)
    datasetName = 'mpii_human_pose_v1_u12_1';
end

if (nargin < 3)
    saveDir = fullfile(datasetDir, 'cropped');
end

if (nargin < 4)
    imageDir = fullfile(datasetDir, 'images');
end

p = struct();

%crop_data(1,16000,1,400,130,1,0,1,130,0) % MPII single train
% 是否是训练数据
p.bTrain = 1;
% 参考高度，计算scale用的
p.refHeight = 400;
p.deltaCrop = 130;
% 是否是单人
p.bSingle = 1;
p.bCropIsolated = 1;
% 是否是多人
p.bMulti = 0;
% 人中心的偏移量？
p.bObjposOffset = 1;
% 数据集目录
p.datasetDir = datasetDir;
% 数据集信息完整路径
p.datasetName = fullfile(p.datasetDir, datasetName);
% 保存处理之后的目录
p.saveDir = saveDir;
% 图像的目录
p.imageDir = [imageDir '/'];

% 加载数据集信息
load(p.datasetName, 'RELEASE');
p.dataset = RELEASE;

annolist1 = crop_data(p);

%crop_data(1,16000,1,400,65,0,0,0,65,0) % multiperson as single people, train
p.deltaCrop = 65;
p.bSingle = 0;
p.bCropIsolated = 0;

annolist2 = crop_data(p);

annolist = horzcat(annolist1, annolist2);

annolistFullName = [p.saveDir '/annolist-full-h' num2str(p.refHeight)];
save(annolistFullName, 'annolist');

prepare_training_data(annolist, p.saveDir);

end
