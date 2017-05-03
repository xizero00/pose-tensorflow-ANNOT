import argparse
import logging
import os

import numpy as np
import scipy.io
import scipy.ndimage

from config import load_config
from dataset.factory import create as create_dataset
from dataset.pose_dataset import Batch
from nnet.predict import setup_pose_prediction, extract_cnn_output, argmax_pose_predict
from util import visualize


def test_net(visualise, cache_scoremaps):
    # 打开python的日志功能
    logging.basicConfig(level=logging.INFO)

    # 加载配置文件
    cfg = load_config()
    # 根据配置文件中的信息产生数据读取类的实例
    dataset = create_dataset(cfg)
    # 不用对数据进行洗牌
    dataset.set_shuffle(False)
    # 告诉数据读取类没有类标，即处于测试模式
    dataset.set_test_mode(True)

    # 该函数返回session，输入算子，输出算子
    sess, inputs, outputs = setup_pose_prediction(cfg)

    # 是否需要保存测试过程中的heatmap
    if cache_scoremaps:
        # 保存heatmap的目录
        out_dir = cfg.scoremap_dir
        # 目录不存在则创建
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # 图片个数
    num_images = dataset.num_images
    # 预测的关节坐标都保存在这里
    predictions = np.zeros((num_images,), dtype=np.object)

    for k in range(num_images):
        print('processing image {}/{}'.format(k, num_images-1))

        # 获得一批数据
        batch = dataset.next_batch()

        # 进行预测
        outputs_np = sess.run(outputs, feed_dict={inputs: batch[Batch.inputs]})

        # 得到heatmap和精细化的heatmap
        scmap, locref = extract_cnn_output(outputs_np, cfg)

        # 获得最终的关节坐标
        '''
        pose = [ [ pos_f8[::-1], [scmap[maxloc][joint_idx]] ] .... ..... ....   ]
        用我的话说就是下面的结构
        pose = [ [关节的坐标,  关节坐标的置信度] .... ..... ....  ]
        '''
        pose = argmax_pose_predict(scmap, locref, cfg.stride)

        pose_refscale = np.copy(pose)
        # 除以尺度，就能恢复到未经过缩放的图像的坐标系上去
        # 注意0:2是左开右闭的区间只取到了0和1
        pose_refscale[:, 0:2] /= cfg.global_scale
        predictions[k] = pose_refscale

        if visualise:
            # 获取图片
            img = np.squeeze(batch[Batch.inputs]).astype('uint8')
            # 显示heatmap
            visualize.show_heatmaps(cfg, img, scmap, pose)
            # 等待按键按下
            visualize.waitforbuttonpress()

        if cache_scoremaps:
            # 保存heatmap
            base = os.path.basename(batch[Batch.data_item].im_path)
            raw_name = os.path.splitext(base)[0]
            out_fn = os.path.join(out_dir, raw_name + '.mat')
            scipy.io.savemat(out_fn, mdict={'scoremaps': scmap.astype('float32')})

            # 保存精细化关节定位的heatmap
            out_fn = os.path.join(out_dir, raw_name + '_locreg' + '.mat')
            if cfg.location_refinement:
                scipy.io.savemat(out_fn, mdict={'locreg_pred': locref.astype('float32')})

    # 将最终预测的关节坐标保存起来
    scipy.io.savemat('predictions.mat', mdict={'joints': predictions})

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--novis', default=False, action='store_true')
    parser.add_argument('--cache', default=False, action='store_true')
    args, unparsed = parser.parse_known_args()

    test_net(not args.novis, args.cache)
