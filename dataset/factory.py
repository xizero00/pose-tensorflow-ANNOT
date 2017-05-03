from dataset.pose_dataset import PoseDataset

# 这是一个工厂类
def create(cfg):
	# 根据配置文件里面所指定的数据集类型创建不同的数据读取类的实例
    dataset_type = cfg.dataset_type
    if dataset_type == "mpii":
        from dataset.mpii import MPII
        data = MPII(cfg) # MPII继承自PoseDataset类
    elif dataset_type == "coco":
        from dataset.mscoco import MSCOCO
        data = MSCOCO(cfg)
    elif dataset_type == "penn_action":
        from dataset.penn_action import PennAction
        data = PennAction(cfg)
    elif dataset_type == "default":
        data = PoseDataset(cfg)# PoseDataset是所有数据集的基类
    else:
        raise Exception("Unsupported dataset_type: \"{}\"".format(dataset_type))
    return data
