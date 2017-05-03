import math

import numpy as np
from scipy.misc import imresize
import matplotlib

# running Matplotlib with tkinter by zf
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    # 很牛逼的用法！
    # 小于半径的点的坐标都在index里面了。。。
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


def check_point(cur_x, cur_y, minx, miny, maxx, maxy):
    # 检查点是否合法
    return minx < cur_x < maxx and miny < cur_y < maxy


def visualize_joints(image, pose):
    # 表示关节圈圈的半径大小为8
    marker_size = 8
    # 因为关节圈圈也是要占用空间的，所以这里计算左上角和右下角的坐标
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = image.shape[1] - 2 * marker_size
    maxy = image.shape[0] - 2 * marker_size
    # 关节个数
    num_joints = pose.shape[0]

    # 复制一份，避免在原图上画
    visim = image.copy()
    # 各个关节的颜色
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for p_idx in range(num_joints):
        # 关节的位置
        cur_x = pose[p_idx, 0]
        cur_y = pose[p_idx, 1]
        # 检查关节位置是否合法
        if check_point(cur_x, cur_y, minx, miny, maxx, maxy):
            # 画个圈圈，参数是图像，坐标x,y，以及圈圈的大小和颜色，最后一个参数是透明度
            _npcircle(visim,
                      cur_x, cur_y,
                      marker_size,
                      colors[p_idx],
                      0.0)
    return visim


def show_heatmaps(cfg, img, scmap, pose, cmap="jet"):
    interp = "bilinear"
    '''
    # 相邻关节对
    cfg.all_joints = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12], [13]]
    # 所有关节对的名字
    cfg.all_joints_names = ['ankle', 'knee', 'hip', 'wrist', 'elbow', 'shoulder', 'chin', 'forehead']
    '''
    all_joints = cfg.all_joints
    all_joints_names = cfg.all_joints_names
    # 定义一行显示3个图片
    subplot_width = 3
    # 计算得到有几行
    subplot_height = math.ceil((len(all_joints) + 1) / subplot_width)
    # 创建若干画图句柄
    f, axarr = plt.subplots(subplot_height, subplot_width)
    for pidx, part in enumerate(all_joints):
         # 需要在第几行第几列画图
        plot_j = (pidx + 1) // subplot_width
        plot_i = (pidx + 1) % subplot_width


        scmap_part = np.sum(scmap[:, :, part], axis=2)
        scmap_part = imresize(scmap_part, 8.0, interp='bicubic')
        scmap_part = np.lib.pad(scmap_part, ((4, 0), (4, 0)), 'minimum')

        # 获得当前的第plot_j行，第plot_i列对应的画图句柄
        curr_plot = axarr[plot_j, plot_i]
        # 当前画的标题是关节的名称
        curr_plot.set_title(all_joints_names[pidx])
        # 将对应的坐标系的显示关闭
        curr_plot.axis('off')
        # 缩放heatmap为8x8的，并且使用双线性插值进行处理
        curr_plot.imshow(img, interpolation=interp)
        curr_plot.imshow(scmap_part, alpha=.5, cmap=cmap, interpolation=interp)

    # 获得第一个图片的句柄
    curr_plot = axarr[0, 0]
    # 设定第一个图片的标题
    curr_plot.set_title('Pose')
    # 将对应的坐标系的显示关闭
    curr_plot.axis('off')
    # 显示关节
    curr_plot.imshow(visualize_joints(img, pose))

    plt.show()


def waitforbuttonpress():
    plt.waitforbuttonpress(timeout=1)