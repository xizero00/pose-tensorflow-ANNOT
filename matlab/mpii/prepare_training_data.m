function prepare_training_data(annolist, saveDir)

% 是否将关节类型设定为以0为起始索引的
zero_indexed_joint_ids = true;

% 14个关节？
pidxs = [0 2 4 5 7 9 12 14 16 17 19 21 22 23];
num_joints = length(pidxs);
%{
28个，有id和pos两个字段
id从0-27
pos是成对出现
0 2 4 5 7 9 12 14 16 17 19 21 22 23 实际上对应的就是
0	[0,0]%
1	[0,1]
2	[1,1]%
3	[1,2]
4	[2,2]%
5	[3,3]%
6	[3,4]
7	[4,4]%
8	[4,5]
9	[5,5]%
10	[6,7]
11	[8,9]
12	[10,10]%
13	[11,10]
14	[11,11]%
15	[12,11]
16	[12,12]%
17	[13,13]%
18	[13,14]
19	[14,14]%
20	[14,15]
21	[15,15]%
22	[8,8]%
23	[9,9]%
24	[6,6]
25	[7,7]
26	[16,16]
27	[17,17]
%}

%{
实际顺序是如下
0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle,
6 - r wrist, 7 - r elbow, 8 - r shoulder, 9 - l shoulder, 10 - l elbow, 11 - l wrist
12 - upper neck, 13 - head top
%}
load('parts.mat');

mkdir_if_missing(saveDir);

num_images = length(annolist);
channels = 3; % three channel images

dataset = struct('image',{}, 'size',{}, 'joints', {});

for i = 1:num_images
	% 每100个显示一次
    if mod(i, 100) == 0
        fprintf('processing image %d/%d \n', i, num_images);
    end
    % 文件名
    filename = annolist(i).image.name;
    
    joints = zeros(num_joints, 3);
    % 一个图片中的人数
    num_people = length(annolist(i).annorect);
    all_joints = cell(1,1);
    % 遍历该图片中的每个人
    for k = 1:num_people
    	% 获取每个人的关节，感觉没有必要用到parts
    	% 这里，可能作者还有其他用途吧，得到的关节顺序上面已经给出
        joint_list = get_anno_joints(annolist(i).annorect(k), pidxs, parts);
        
        % 将每个人的关节和类型写入到joints
        % joints的第一列是关节类型，第二列是x，第三列是y
        % 关节类型此时是以1为起始索引的
        n = 0;
        for j = 1:num_joints
            jnt = joint_list(j, :);
            if ~isnan(jnt(1))
                n = n + 1;
                joints(n, :) = [j jnt];
            end
        end
        % 将其转换为整数
        joints = int32(joints(1:n, :));
        % 将关节类型设置为以0为起始索引
        if zero_indexed_joint_ids
            joints(:,1) = joints(:,1) - 1;
        end
        % 将当前图片的所有关节保存到all_joints
        all_joints{k} = joints;
    end
    
    entry = struct;
    % 图片名称
    entry.image = filename;
    % 图片的维度
    entry.size = [channels, annolist(i).image_size];
    % 当前图片中包含的所有人的关节坐标
    entry.joints = all_joints;
    dataset(i) = entry;
end

out_filename = fullfile(saveDir, 'dataset.mat');
fprintf('Generated dataset definition file: %s\n', out_filename);
save(out_filename, 'dataset');

end

