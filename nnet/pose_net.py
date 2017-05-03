import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

from dataset.pose_dataset import Batch
from nnet import losses

# 两种不同的resnet网络的类型
net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101}

# 在测试的时候用的输出heatmap的层
def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred

# 获取batch_spec
# 根据关节个数和batch_size
# 返回的batch_spec是
# 输入的大小
# 关节heatmap的大小
# 关节weight的大小
# 精细化heatmap的大小
# 精细化mask的大小
def get_batch_spec(cfg):
    num_joints = cfg.num_joints
    batch_size = cfg.batch_size
    return {
        # 0:[batch_size, None, None, 3]
        Batch.inputs: [batch_size, None, None, 3],
        # 1 [batch_size, None, None, num_joints]
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        # 2 [batch_size, None, None, num_joints]
        Batch.part_score_weights: [batch_size, None, None, num_joints],
        # 3 [batch_size, None, None, num_joints * 2]
        Batch.locref_targets: [batch_size, None, None, num_joints * 2],
        # 4 [batch_size, None, None, num_joints * 2]
        Batch.locref_mask: [batch_size, None, None, num_joints * 2]
    }


class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, inputs):
    	# 该函数是定义resnet的网络结构

        # 根据配置选择对应的网络结构
        # resnet_v1_50 还是resnet_v1_101
        net_fun = net_funcs[self.cfg.net_type]

        # 均值是配置文件中提供的
        mean = tf.constant(self.cfg.mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        # 减去均值
        im_centered = inputs - mean

        # 获取resnet的参数scope，然后设定为当前的参数scope
        with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
            # 注意这里的output_stride=16!!与论文里面的含义
            # 一定要理解这里的东西
            # net是网络，end_points是啥？
            net, end_points = net_fun(im_centered,
                                      global_pool=False, output_stride=16)
        # net相当于提取的特征
        # end_points相当于提取的用于中间监督的特征
        return net, end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        # 用正则表达式获得字符串中的值
        # 比如字符串是"resnet_101"，那么获得是就是101
        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        # 定义网络层的名字，注意这里格式化字符串用的是{}来代表你的参数
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with tf.variable_scope('pose', reuse=reuse):
        	#　out['part_pred']是网络输出关节heatmap的operator
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            if cfg.location_refinement:
            	# 如果配置文件里面设定需要进行关节精细化定位则
            	# out['locref']表示进行关节精细化定位的heatmap
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision:
            	# 如果配置文件里面设定需要进行中间层的监督训练
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                # out['part_pred_interm']表示进行中间层训练的的heatmap
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)

        return out

    def get_net(self, inputs):
    	# 定义一个基于resnet_v1的预测关节位置网络
    	# 定义网络的除了输出节点的部分
        net, end_points = self.extract_features(inputs)
        # 定义网络的输出节点的部分
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
    	# 获得网络的输出节点
        heads = self.get_net(inputs)
        # heatmap再经过sigmoid才是最终的输出
        # 而精细化定位则不需要经过sigmoid!!
        prob = tf.sigmoid(heads['part_pred'])
        return {'part_prob': prob, 'locref': heads['locref']}

    def train(self, batch):
        cfg = self.cfg

        # 获取resnet_v1的网络结构
        heads = self.get_net(batch[Batch.inputs])

        # 从配置文件中获得是否需要weigh_part预测
        weigh_part_predictions = cfg.weigh_part_predictions
        # 如果则part_score_weights为读取到的，否则为1
        part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0

        # 定义损失的函数
        def add_part_loss(pred_layer):
            return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
                                                   heads[pred_layer],
                                                   part_score_weights)

        # 生成关节的loss
        loss = {}
        # 定义heatmap的损失函数
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if cfg.intermediate_supervision:
        	# 定义中间监督训练的损失函数
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']
        # 生成关节精细化的loss
        if cfg.location_refinement:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            # 将huber loss用于计算精细化定位的损失函数
            # 这里需要用到locref_weight（权重），locref_pred（网络的预测值），locref_targets（类标）
            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        # 生成总的loss
        loss['total_loss'] = total_loss
        return loss
