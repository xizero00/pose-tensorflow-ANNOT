import logging
import threading

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import load_config
from dataset.factory import create as create_dataset
from nnet.net_factory import pose_net
from nnet.pose_net import get_batch_spec
from util.logging import setup_logging

# 学习率生成类
class LearningRate(object):
    def __init__(self, cfg):
        self.steps = cfg.multi_step
        self.current_step = 0

    def get_lr(self, iteration):
        lr = self.steps[self.current_step][0]
        if iteration == self.steps[self.current_step][1]:
            self.current_step += 1

        return lr

# 根据batch_spec来生成placeholders并且
# 返回入队op和placeholder和batchop
def setup_preloading(batch_spec):
	# 根据batch_spec生成placeholder字典
	# 这个可以用来喂数据
    placeholders = {name: tf.placeholder(tf.float32, shape=spec) for (name, spec) in batch_spec.items()}
    # 取出所有的placeholder字典中对应的名字
    names = placeholders.keys()
    # 去除对应的placeholder字典中对应的placeholder实例
    placeholders_list = list(placeholders.values())

    QUEUE_SIZE = 20

    # batch_spec指定的这么多个tensor组成的tuple是一个输入，队列有20个这样的tuple
    q = tf.FIFOQueue(QUEUE_SIZE, [tf.float32]*len(batch_spec))
    # 将一个list入队，这个入队op可以用来喂数据
    enqueue_op = q.enqueue(placeholders_list)
    # 出队一个元素，是一个tuple包含了一个batch_spec指定的数据
    batch_list = q.dequeue()

    batch = {}
    # 把数据恢复成dict
    for idx, name in enumerate(names):
        batch[name] = batch_list[idx]
        batch[name].set_shape(batch_spec[name])
    return batch, enqueue_op, placeholders

# 去除一批数据并且生成dict，然后入队列
def load_and_enqueue(sess, enqueue_op, coord, dataset, placeholders):
    while not coord.should_stop():
    	# 读取一批数据
        batch_np = dataset.next_batch()
        # 根据placeholder放入到dict
        food = {pl: batch_np[name] for (name, pl) in placeholders.items()}
        # 入队列
        sess.run(enqueue_op, feed_dict=food)

# 开启一个线程不停地预处理数据并放入队列
def start_preloading(sess, enqueue_op, dataset, placeholders):
    coord = tf.train.Coordinator()

    t = threading.Thread(target=load_and_enqueue,
                         args=(sess, enqueue_op, coord, dataset, placeholders))
    t.start()

    return coord, t

# 获取train_op和学习率op
def get_optimizer(loss_op, cfg):
    learning_rate = tf.placeholder(tf.float32, shape=[])

    if cfg.optimizer == "sgd":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif cfg.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(cfg.adam_lr)
    else:
        raise ValueError('unknown optimizer {}'.format(cfg.optimizer))
    # 创建一个train_op，用于计算loss和更新梯度
    train_op = slim.learning.create_train_op(loss_op, optimizer)

    return learning_rate, train_op


def train():
	# 设置日志
    setup_logging()

    # 载入训练配置文件pose_cfg.yaml
    cfg = load_config()
    # 创建数据集类的实例
    dataset = create_dataset(cfg)

    # 获取batch_spec
    # 包含输入图片大小
	# 关节heatmap的大小
	# 关节weight的大小
	# 精细化heatmap的大小
	# 精细化mask的大小
    batch_spec = get_batch_spec(cfg)
    # 根据batch_spec产生入队操作、placeholder和batch数据
    batch, enqueue_op, placeholders = setup_preloading(batch_spec)

    # 生成网络结构并且产生losses op
    # 其中losses包括很多类型的loss
    losses = pose_net(cfg).train(batch)
    total_loss = losses['total_loss']

    # 把多个loss合并起来
    for k, t in losses.items():
    	# return a scalar Tensor of type string which contains a Summary protobuf.
        tf.summary.scalar(k, t)
    # returns a scalar Tensor of type string containing the serialized 
    # Summary protocol buffer resulting from the merging
    merged_summaries = tf.summary.merge_all()

    # 获取/resnet_v1下面的所有的变量
    variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
    # Create the saver which will be used to restore the variables.

    # 创建一个恢复resent_v1的权重的op
    restorer = tf.train.Saver(variables_to_restore)
    # 创建一个保存训练状态的op
    saver = tf.train.Saver(max_to_keep=5)

    sess = tf.Session()

    # 开启一个线程去读取数据并且装入到队列
    coord, thread = start_preloading(sess, enqueue_op, dataset, placeholders)

    # 打开一个训练的记录器
    train_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

    # 获取train_op和学习率op
    learning_rate, train_op = get_optimizer(total_loss, cfg)

    # 初始化全局和局部变量
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.从文件中读取权重到内存
    restorer.restore(sess, cfg.init_weights)

    # 从配置文件获取最大迭代次数
    max_iter = int(cfg.multi_step[-1][1])

    display_iters = cfg.display_iters
    cum_loss = 0.0

    # 生成一个学习率产生器的实例
    lr_gen = LearningRate(cfg)

    for it in range(max_iter+1):
        # 根据当前迭代的次数产生一个学习率
        current_lr = lr_gen.get_lr(it)
        # 进行训练
        [_, loss_val, summary] = sess.run([train_op, total_loss, merged_summaries],
                                          feed_dict={learning_rate: current_lr})
        # 累加loss
        cum_loss += loss_val
        # 将迭代次数保存起来
        train_writer.add_summary(summary, it)

        if it % display_iters == 0:# 每隔display_iters就显示一次 loss
            average_loss = cum_loss / display_iters# 平均loss
            cum_loss = 0.0
            logging.info("iteration: {} loss: {} lr: {}"
                         .format(it, "{0:.4f}".format(average_loss), current_lr))

        # Save snapshot
        # 每隔cfg.save_iters次就会保存
        if (it % cfg.save_iters == 0 and it != 0) or it == max_iter:
            # 获得模型的名称
            model_name = cfg.snapshot_prefix
            # 保存模型
            saver.save(sess, model_name, global_step=it)

    sess.close()
    # 请求数据读取线程停止
    coord.request_stop()
    # 等待数据读取线程结束
    coord.join([thread])


if __name__ == '__main__':
    train()
