import logging
import random as rand
from enum import Enum

import numpy as np
# arr是array的缩写
from numpy import array as arr
# cat是concatenate的缩写
from numpy import concatenate as cat

import scipy.io as sio
from scipy.misc import imread, imresize

# 一批数据的名字及其索引
class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    data_item = 5


def mirror_joints_map(all_joints, num_joints):
	# 对关节的类型进行mirror
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res

# crop是左上角和有效做的坐标
# crop包含四个元素左上角的(x,y)和右下角(x,y)
# crop_pad是指往外扩的大小，这里指定image_size是为了
# 防止扩的时候不超过图像大小
def extend_crop(crop, crop_pad, image_size):
	# 获取经过外扩的左上角和右下角的框框坐标
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


def data_to_input(data):
    # 将数据在axis=0这一维度扩一下，并且将数据类型转换为float
    # Insert a new axis, corresponding to a given position in the array shape.
    return np.expand_dims(data, axis=0).astype(float)


class DataItem:
    pass


# 所有数据集的基类
class PoseDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        # 加载数据的信息，此时并没有加载数据
        self.data = self.load_dataset()
        self.num_images = len(self.data)
        if self.cfg.mirror:
        	# cfg.all_joints是相互左右对称的关节组成的tuple
        	# cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
        	# cfg.all_joints_names = 
        	# ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
            self.symmetric_joints = mirror_joints_map(cfg.all_joints, cfg.num_joints)
        self.curr_img = 0
        # 设置是否数据进行洗牌
        self.set_shuffle(cfg.shuffle)

    
    def load_dataset(self):
    	# 读取数据的信息

        cfg = self.cfg
        # 数据集信息的名称
        file_name = cfg.dataset
        # Load Matlab file dataset annotation
        # 从mat文件读取数据集信息
        mlab = sio.loadmat(file_name)
        self.raw_data = mlab
        mlab = mlab['dataset']

        num_images = mlab.shape[1]
        data = []
        # has_gt=True表明是训练，如果has_gt是False表明是测试
        has_gt = True

        # 遍历每一个样本
        for i in range(num_images):
        	# 取出每一个样本
            sample = mlab[0, i]

            # 将数据存放到item
            item = DataItem()
            # 图片编号
            item.image_id = i
            # 图片路径
            item.im_path = sample[0][0]
            # 图片宽和高
            item.im_size = sample[1][0]
            if len(sample) >= 3:
            	# 关节坐标
                joints = sample[2][0][0]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert((joint_id < cfg.num_joints).any())
                joints[:, 0] = joint_id
                # 注意是加了个列表[joints]，表明是一个人
                item.joints = [joints]
            else:
                has_gt = False
            # 是否需要对图片进行crop，因为matlab代码中已经crop了这里应该是false
            if cfg.crop:
            	# 将框框的左上角和右下角坐标转换到以(0,0)为原点的
                crop = sample[3][0] - 1
                # 将框框的左上角和右下角往外外扩cfg.crop_pad大小，但是不能超过图像的大小item.im_size
                item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        self.has_gt = has_gt
        return data

    def set_test_mode(self, test_mode):
    	# 设置为测试模式
        self.has_gt = not test_mode# 表明是训练，如果has_gt是False表明是测试

    def set_shuffle(self, shuffle):
    	# 设置是否shuffle这个内部变量
        self.shuffle = shuffle
        if not shuffle:
        	# 也就是说如果没有shuffle，此时mirror也不能是false
        	# shuffle=False的时候mirror也必须是false
        	# 也就是测试的时候可以这样？不需要打乱，也不需要mirror？
            assert not self.cfg.mirror
            self.image_indices = np.arange(self.num_images)

    def mirror_joint_coords(self, joints, image_width):
    	# 将关节的坐标进行水平翻转
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints

    def mirror_joints(self, joints, symmetric_joints, image_width):
    	# 将关节坐标和关节类型都进行翻转！
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = self.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one 翻转过关节之后需要交换关节的类型
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res

    def shuffle_images(self):
    	# 对图像进行洗牌，并且对镜像的图片也进行洗牌
        num_images = self.num_images
        if self.cfg.mirror:
            image_indices = np.random.permutation(num_images * 2)
            # self.mirrored是一个True和False组成的ndarray
            self.mirrored = image_indices >= num_images
            image_indices[self.mirrored] = image_indices[self.mirrored] - num_images
            self.image_indices = image_indices
        else:
            self.image_indices = np.random.permutation(num_images)

    def num_training_samples(self):
    	# 获取训练样本的数量
        num = self.num_images
        if self.cfg.mirror:
            num *= 2
        return num

    def next_training_sample(self):
    	# 获得下一个训练样本的图片缩影以及是否为镜像
        if self.curr_img == 0 and self.shuffle:
            self.shuffle_images()

        curr_img = self.curr_img
        self.curr_img = (self.curr_img + 1) % self.num_training_samples()

        imidx = self.image_indices[curr_img]
        # 如果self.cfg.mirror为true则返回self.mirrored[curr_img]，否则返回self.cfg.mirror
        mirror = self.cfg.mirror and self.mirrored[curr_img]

        # imidx是当前图片的索引，mirror表示是否为镜像的样本
        return imidx, mirror

    def get_training_sample(self, imidx):
    	# 获得样本的信息
        return self.data[imidx]

    def get_scale(self):
    	# 获得经过经过尺度抖动的尺度
        cfg = self.cfg
        scale = cfg.global_scale
        # scale_jitter_lo和scale_jitter_up是指对scale进行抖动
        # lo是抖动下界，up是抖动上界
        # 判断是否开启尺度抖动来对数据进行扩增
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale

    def next_batch(self):
    	# 获得一批数据
        while True:
        	# 获得一个训练样本imidx(图像id)和mirror(是否为镜像的标志)
            imidx, mirror = self.next_training_sample()
            # 获得样本数据
            data_item = self.get_training_sample(imidx)
            # 获得尺度
            scale = self.get_scale()

            # 判断是否有效，即判断经过缩放之后的图像宽或者高必须有一个大于100
            # 此外图像的面积必须<max_input_size^2,否则忽略当前数据
            if not self.is_valid_size(data_item.im_size, scale):
                continue
            # 
            return self.make_batch(data_item, scale, mirror)

    def is_valid_size(self, image_size, scale):
    	# 判断是否有效，即判断经过缩放之后的图像宽或者高必须有一个大于100
        # 此外图像的面积必须<max_input_size^2,否则忽略当前数据
        im_width = image_size[2]
        im_height = image_size[1]

        # 确保图像的高度或者宽度有一个必须大于100
        max_input_size = 100
        if im_height < max_input_size or im_width < max_input_size:
            return False

        # 确保图像的面积必须小于max_input_size的平方
        if hasattr(self.cfg, 'max_input_size'):
            max_input_size = self.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True

    def make_batch(self, data_item, scale, mirror):
    	# 根据数据信息读取图片并且根据配置文件的配置
    	# 来决定是否对图片进行crop，此外还需要对关节的坐标进行crop
    	# 然后对图片进行缩放，对关节坐标进行缩放
    	# 最后生成关节的heatmap,关节heatmap对应的weight
    	# 以及精细化定位的heatmap,精细化定位的heatmap对应的mask
    	'''
		这四个如下：
    	Batch.part_score_targets: part_score_targets,  关节的heatmap
        Batch.part_score_weights: part_score_weights,  关节heatmap对应的weight
        Batch.locref_targets: locref_targets,  	       精细化定位的heatmap
        Batch.locref_mask: locref_mask				   精细化定位的heatmap对应的mask
    	'''

    	# 文件路径
        im_file = self.cfg.dataset_path + '/' + data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)
        image = imread(im_file, mode='RGB')

        # 是否是测试，self.has_gt=True表示为训练，否则为测试
        if self.has_gt:
        	# 这里要特别注意
        	# 假设data_item.joints = [array([1, 2, 3])]
        	# 那么经过np.copy(data_item.joints)之后变成了
        	# array([[1, 2, 3]])
        	# 这TMD太神奇了！！！！！
            joints = np.copy(data_item.joints)

        # 是否经过crop
        if self.cfg.crop:
        	# 如果crop了那么就对图像进行crop
            crop = data_item.crop
            image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
            if self.has_gt:
            	# 如果有关节坐标则也进行crop
            	# 这里只对1和2列进行了处理，因为第0列是关节类型
                joints[:, 1:3] -= crop[0:2].astype(joints.dtype)

        # 图像进行缩放
        img = imresize(image, scale) if scale != 1 else image
        # 缩放之后的图像大小，取img.shape[0]和img.shape[1]
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
        	# 如果是镜像的则对图像进行水平翻转
            img = np.fliplr(img)

       	# 将图片放入到batch字典中
        batch = {Batch.inputs: img}

        if self.has_gt:# 如果有关节坐标
            stride = self.cfg.stride

            if mirror:
            	# 翻转关节坐标及其类型
                joints = [self.mirror_joints(person_joints, self.symmetric_joints, image.shape[1]) for person_joints in
                          joints]

            # heatmap的大小是通过输入图像的大小动态进行计算的，每个heatmap的大小都不一样的
            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

            # 对每个人(这里其实只有一个人)每个关节坐标进行缩放,
            # 因为图片也进过缩放了，所以关节坐标也要缩放，这样才能对应上
            # 注意这里的1:3，实际上取的是1和2，取不到3！！！！！，注意区间，左闭右开
            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]

            # 获取每个人(这里其实只有一个人)每个关节的类型id
            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            # 计算heatmap，heatmap的weight，精细化定位的heatmap，以及精细化定位的heatmap的mask
            part_score_targets, part_score_weights, locref_targets, locref_mask = self.compute_target_part_scoremap(
                joint_id, scaled_joints, data_item, sm_size, scale)

            # 更新字典中的值
            # batch[Batch.part_score_targets] = part_score_targets
            # batch[part_score_weights] = part_score_weights
            # batch[locref_targets] = locref_targets
            # batch[locref_mask] = locref_mask
            batch.update({
                Batch.part_score_targets: part_score_targets,
                Batch.part_score_weights: part_score_weights,
                Batch.locref_targets: locref_targets,
                Batch.locref_mask: locref_mask
            })

        # 对batch内部的每个数据进行处理外扩一个维度
        # 为啥要外扩？
        # TODO:请搞清楚为啥要外扩一个axis
        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch

    def compute_target_part_scoremap(self, joint_id, coords, data_item, size, scale):
    	# stride是输入图像经过CNN得到的heatmap相比于原来的图像大小的比例
    	# 比如图像是64x128输入到网络的，而得到的heatmap变成了32x64
    	# 那么stride=2,因为64/32=2,此外128/64=2
        stride = self.cfg.stride
        # pos_dist_thresh是关节周围的大小,这里乘以尺度因为图像进行了缩放
        dist_thresh = self.cfg.pos_dist_thresh * scale
        # 关节的个数
        num_joints = self.cfg.num_joints
        half_stride = stride / 2
        # heatmap矩阵
        scmap = np.zeros(cat([size, arr([num_joints])]))
        # 精细化heatmap的形状
        locref_size = cat([size, arr([num_joints * 2])])
        # 精细化heatmap的mask
        locref_mask = np.zeros(locref_size)
        # 精细化heatmap矩阵
        locref_map = np.zeros(locref_size)

        # 精细化heatmap的尺度
        locref_scale = 1.0 / self.cfg.locref_stdev

        # 定义的距离关节的范围，在这个范围内才设置为1
        dist_thresh_sq = dist_thresh ** 2

        # heatmap的宽和高
        width = size[1]
        height = size[0]

        # 遍历每个人的坐标(实际上这里只有一个人)
        for person_id in range(len(coords)):
        	# 遍历每个关节的类型
        	# 注意这里使用了enumerate，k返回的是索引，j_id才是真正的值
        	# j_id即是关节的类型
            for k, j_id in enumerate(joint_id[person_id]):
            	# 获取关节的坐标
                joint_pt = coords[person_id][k, :]
                # joint_pt[0]和joint_pt[1]都是ndarray的，需要转换为scalar，即标量
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

                # don't loop over entire heatmap, but just relevant locations
                # 转换到heatmap上的关节坐标
                # TODO: 这里需要搞清楚为啥要减去half_stride
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)

                # 因为关节坐标周围用dist_thresh定义了一个正方形的框框
                # 所以这里是得到这个框框的左上角和右下角的坐标
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                # 只在这个框框内部进行
                for j in range(min_y, max_y + 1):  # range(height):
                	# 注意了这里的pt_y是转换回到原始图片大小的y坐标了
                    pt_y = j * stride + half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python

                        # 注意了这里的pt_x是转换回到原始图片大小的x坐标了
                        pt_x = i * stride + half_stride
                        # 与关节坐标之间的差值
                        dx = j_x - pt_x
                        dy = j_y - pt_y

                        # 与关节之间的距离的平方
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))

                        # 距离的平方小于所定义的阈值的平方值就认为在这个框框内
                        if dist <= dist_thresh_sq:
                        	# 简单地设置为1
                        	# 这里并没有像往常计算heatmap那样通过高斯公式去生成
                        	# 那样计算量很大
                            scmap[j, i, j_id] = 1
                            locref_mask[j, i, j_id * 2 + 0] = 1
                            locref_mask[j, i, j_id * 2 + 1] = 1
                            locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
                            locref_map[j, i, j_id * 2 + 1] = dy * locref_scale
    
       	# 计算heatmap的weight
        weights = self.compute_scmap_weights(scmap.shape, joint_id, data_item)

        return scmap, weights, locref_map, locref_mask

    def compute_scmap_weights(self, scmap_shape, joint_id, data_item):
        cfg = self.cfg
        # weigh_only_present_joints这个参数控制着是否只计算存在关节的loss
        # weigh_only_present_joints=True表明
        # 如果关节坐标不存在就不计算loss了
        # 也不会反传了这部分误差
        # weigh_only_present_joints=True表明
        # 统统都计算误差并且反传
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0# 存在坐标的那个关节的weight矩阵的所有元素都是1
        else:
            weights = np.ones(scmap_shape)
        return weights
