from nnet.pose_net import PoseNet

# 网络定义的工厂类
def pose_net(cfg):
    cls = PoseNet# 所有网络的基类
    return cls(cfg)
