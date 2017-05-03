from dataset.pose_dataset import PoseDataset


class MPII(PoseDataset):
    def __init__(self, cfg):
    	# 相邻关节对
        cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
        # 所有关节对的名字
        cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
        # 关节的名字
        cfg.num_joints = 14
        super().__init__(cfg)

    # 将所有的关节翻转
    # joints是关节的坐标是14x2的
    # 其中joints[:, 1]是所有关节的x坐标
    def mirror_joint_coords(self, joints, image_width):
        joints[:, 1] = image_width - joints[:, 1]
        return joints
