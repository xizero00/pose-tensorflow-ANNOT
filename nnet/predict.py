import numpy as np

import tensorflow as tf

from nnet.net_factory import pose_net

# 这个文件在test.py中会用到

def setup_pose_prediction(cfg):
	# 该函数返回session，输入算子，输出算子

	# 输入的placeholder
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size   , None, None, 3])

    # 获得测试的网络
    net_heads = pose_net(cfg).test(inputs)
    # 获得网络输出的heatmap
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
    	# 是否使用了关节位置精细化定位
        outputs.append(net_heads['locref'])

    # 定义一个恢复权重的算子
    restorer = tf.train.Saver()

    sess = tf.Session()

    # 初始化全局核局部变量
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    # 从cfg.init_weights位置恢复权重
    restorer.restore(sess, cfg.init_weights)

    # 返回session，输入算子，输出算子
    return sess, inputs, outputs


def extract_cnn_output(outputs_np, cfg):
	# 对网络的输出进行处理，包括heatmap和关节精细化定位的heatmap

    scmap = outputs_np[0]
    scmap = np.squeeze(scmap)
    locref = None
    if cfg.location_refinement:
    	# 如果开启了关节精细化定位，则对精细化定位的heatmap进行处理
        locref = np.squeeze(outputs_np[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        # TODO:为啥要乘以这玩意locref_stdev？
        locref *= cfg.locref_stdev
    return scmap, locref


def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []

    # 获得每个关节的位置
    for joint_idx in range(num_joints):
    	# numpy.unravel_index(indices, dims, order='C')
    	# Converts a flat index or array of flat indices into a tuple of coordinate arrays.
    	# 我操，python里面直接有这么一个函数可以获得关节的位置太简单了
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)

        offset = np.array(offmat[maxloc][joint_idx])[::-1]
        # offset直接就是用来加到关节的位置上去
        # np.array(maxloc).astype('float')是转换为ndarray然后为float
        # np.array(maxloc).astype('float') * stride是转换到输入图片坐标系上去了
        # TODO: 为什么要加上0.5 * stride？
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        # 返回的是关节的位置和这个位置的置信度scmap[maxloc][joint_idx
        '''
        [
        pos_f8[::-1],
        [scmap[maxloc][joint_idx]]
        ]
        '''
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return np.array(pose)
