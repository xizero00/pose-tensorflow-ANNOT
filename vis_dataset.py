import logging

import numpy as np
from scipy.misc import imresize

import matplotlib.pyplot as plt

from config import load_config
from dataset.pose_dataset import Batch
from dataset.factory import create as dataset_create

# 这个文件用于可视化


def display_dataset():
    # 该文件可以显示训练之后的heatmap

    # 开启日志
    logging.basicConfig(level=logging.DEBUG)

    # 读取配置
    cfg = load_config()
    # 创建数据集读取类
    dataset = dataset_create(cfg)
    # 不需要对数据进行洗牌
    dataset.set_shuffle(False)

    while True:
        # 获得一批数据
        batch = dataset.next_batch()

        for frame_id in range(1):
            # 图像
            img = batch[Batch.inputs][frame_id,:,:,:]
            # 转换图像类型为uint8,8位无符号整型，范围在0-255之间
            # 正好为图像中像素值的区间范围
            img = np.squeeze(img).astype('uint8')

            # 获取heatmap
            scmap = batch[Batch.part_score_targets][frame_id,:,:,:]
            # 去掉多余的维度
            scmap = np.squeeze(scmap)

            # scmask = batch[Batch.part_score_weights]
            # if scmask.size > 1:
            #     scmask = np.squeeze(scmask).astype('uint8')
            # else:
            #     scmask = np.zeros(img.shape)

            # 画图，这里在一个窗口中画多个图所以要定义这个窗口有几行几列
            # 几行
            subplot_height = 4
            # 几列
            subplot_width = 5
            # 要画几次
            num_plots = subplot_width * subplot_height
            # 创建若干画图句柄
            f, axarr = plt.subplots(subplot_height, subplot_width)

            for j in range(num_plots):
                # 需要在第几行第几列画图
                plot_j = j // subplot_width
                plot_i = j % subplot_width

                # 获得当前的第plot_j行，第plot_i列对应的画图句柄
                curr_plot = axarr[plot_j, plot_i]
                # 将对应的坐标系的显示关闭
                curr_plot.axis('off')

                # 如果超过了需要显示的关节的个数则忽略
                if j >= cfg.num_joints:
                    continue

                # 将第j个关节的heatmap取出来
                scmap_part = scmap[:,:,j]
                # 缩放heatmap为8x8的，并且使用最近邻插值进行处理
                scmap_part = imresize(scmap_part, 8.0, interp='nearest')
                # 
                scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')

                # 当前画的标题是关节的编号
                curr_plot.set_title("{}".format(j+1))
                # 显示图片
                curr_plot.imshow(img)
                # hold(True)表示在上面显示出来的图片基础之上继续画
                curr_plot.hold(True)
                # 将关节的heatmap也画出来，设定透明度为0.5
                curr_plot.imshow(scmap_part, alpha=.5)

        # figure(0)
        # plt.imshow(np.sum(scmap, axis=2))
        # plt.figure(100)
        # plt.imshow(img)
        # plt.figure(2)
        # plt.imshow(scmask)

        # 显示窗口
        plt.show()
        # 等待按键
        plt.waitforbuttonpress()


if __name__ == '__main__':
    display_dataset()
