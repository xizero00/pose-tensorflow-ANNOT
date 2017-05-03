from easydict import EasyDict as edict
# 文件的默认配置
cfg = edict()

# 图片经过CNN缩小了多少倍
cfg.stride = 8.0

# 没用到的参数
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False

cfg.fg_fraction = 0.25
# 在计算loss的时候是否反传不存在关节坐标的loss
cfg.weigh_only_present_joints = False
# 图片3个channel的均值
cfg.mean_pixel = [123.68, 116.779, 103.939]
# 是否对数据进行洗牌
cfg.shuffle = True
# 保存的模型的前缀
cfg.snapshot_prefix = "snapshot"
# 日志文件的目录
cfg.log_dir = "log"
# 对图像进行缩放的全局参数
cfg.global_scale = 1.0
# 是否开启关节位置精细化定位
cfg.location_refinement = False
# 关节位置精细化定位用到的标准差
cfg.locref_stdev = 7.2801
# 关节位置精细化定位用到的weight矩阵
cfg.locref_loss_weight = 1.0
# 关节位置精细化定位的loss是否用huber_loss
cfg.locref_huber_loss = True
# 优化方法
cfg.optimizer = "sgd"
# 是否开启中间层监督
cfg.intermediate_supervision = False
# 中间监督的网络层是12
cfg.intermediate_supervision_layer = 12
# 是否启用正则化
cfg.regularize = False
# 是否启用weight decay,权重衰减
cfg.weight_decay = 0.0001
# 是否对图像进行水平翻转，也就是镜像
cfg.mirror = False
# 是否对图像进行抠图
cfg.crop = False
# 抠图的矩形框外扩多少像素
cfg.crop_pad = 0
# 测试的时候heatmap保存的目录
cfg.scoremap_dir = "test"
# 数据集的类型
cfg.dataset_type = "default"  # options: "default", "coco"
# 是否使用分割
cfg.use_gt_segm = False
# batch size的大小
cfg.batch_size = 1
# 是否使用视频
cfg.video = False
# video batch size
cfg.video_batch = False
