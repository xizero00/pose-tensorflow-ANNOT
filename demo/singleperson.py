import os
import sys

# 将当前文件所在的目录的父目录加入系统路径中
sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

# 读取配置文件
cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
# 该函数返回session，输入算子，输出算子
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
# 读取图片
file_name = "demo/image.png"
image = imread(file_name, mode='RGB')

# 增加一个axis
image_batch = data_to_input(image)

# Compute prediction with the CNN
# 得到网络的输出
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
# 对网络输出进行处理
scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
# 获得关节坐标
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

# Visualise
# 显示出来
visualize.show_heatmaps(cfg, image, scmap, pose)
visualize.waitforbuttonpress()
