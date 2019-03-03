# -*- coding: utf-8 -*-
"""
基于cnn模型的违规文本反垃圾功能
@author Jerry.Fang
"""
import os
import random
import sys
import re

import datetime
import numpy as np
import time
import subprocess


import tensorflow as tf
import tensorflow.contrib.keras as kr
#import jieba as jb
from tensorflow.contrib import learn



class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 199999  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    def __init__(self):
        # 输出相关路径
        self.base_path = '/tmp/tf_data/'
        self.save_mod = self.base_path + 'cnn_mod_v1'  # 训练模型结果文件前缀

        # 部署数据路径
        self.model_res_source = os.path.dirname(os.path.realpath(__file__)) + '/../data/tf_model_res.zip'

        # 类别数
        self.num_classes = len(CATEGORIES_ID_MAP)


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        self.logits = None
        self.y_pred_cls = None
        self.acc = None
        self.loss = None
        self.optim = None

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 分类信息
CATEGORY_NORMAL = 'normal'
CATEGORY_AD = 'ad'
CATEGORIES_ID_MAP = {
    CATEGORY_NORMAL: 0,
    CATEGORY_AD: 1,
}
FILE_NAMES = CATEGORIES_ID_MAP.keys()
WORD_DICT_NAME = 'word_dict.txt'


def get_cate_name_by_id(input_cate_id):
    """
    根据 cate id 获取名称
    :param input_cate_id: 
    :return: 
    """
    for cate_name, cate_id in CATEGORIES_ID_MAP.items():
        if cate_id == input_cate_id:
            return cate_name
    return ''


def convert_raw_str_to_ids(raw_text):
    """
    将文本转换为 ID编码的一维向量
    :param raw_text: 
    :return: 
    """
    # 确保unicode类型
    raw_text = raw_text.strip()
    if isinstance(raw_text, str):
        raw_text = raw_text.decode('utf-8')

    # 转为十进制编码
    global tf_words_dict
    d_ids = []
    for word in jb.cut(raw_text, cut_all=False):
        if word not in tf_words_dict:
            continue
        d_ids.append(tf_words_dict[word])

    return d_ids


def process_file(file_path, max_length=600):
    """
    读取训练（90%）\验证（10%）数据
    :param file_path: 
    :param max_length: 
    :return: 
    """

    # 获取文件数据
    tmp_data_list = []
    tmp_validate_data_list = []
    count = 0
    for file_name in FILE_NAMES:
        file_handler = open(file_path + '%s.txt' % file_name)
        for line in file_handler:

            # 每行文本转换为十进制编码一维数组
            d_ids = convert_raw_str_to_ids(line)
            if not d_ids:
                continue

            # 90% 作为训练数据，这种划分方法比较均匀
            count += 1
            if count % 10 > 0:
                tmp_data_list.append({
                    'd_ids': d_ids,
                    'c_ids': [CATEGORIES_ID_MAP[file_name]],
                })

            # 10% 作为验证数据
            else:
                tmp_validate_data_list.append({
                    'd_ids': d_ids,
                    'c_ids': [CATEGORIES_ID_MAP[file_name]],
                })

    # 随机打乱顺序
    random.shuffle(tmp_data_list)
    random.shuffle(tmp_validate_data_list)

    # 获取训练数据集合
    data_ids = []
    cate_ids = []
    for data_info in tmp_data_list:
        data_ids.append(data_info['d_ids'])
        cate_ids.append(data_info['c_ids'])

    # 获取验证数据集合
    valid_data_ids = []
    valid_cate_ids = []
    for data_info in tmp_validate_data_list:
        valid_data_ids.append(data_info['d_ids'])
        valid_cate_ids.append(data_info['c_ids'])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_ids, max_length)
    y_pad = kr.utils.to_categorical(cate_ids)  # 将标签转换为one-hot表示
    valid_x_pad = kr.preprocessing.sequence.pad_sequences(valid_data_ids, max_length)
    valid_y_pad = kr.utils.to_categorical(valid_cate_ids)  # 将标签转换为one-hot表示
    return x_pad, y_pad, valid_x_pad, valid_y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(model, x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, model, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def text_cnn_train():
    """
    进行训练并生成训练结果
    :return: 
    """
    # 初始化
    config, model = init_tf_cnn()

    # 展示数据writer
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config.base_path)

    # 生成训练模型保存文件夹
    saver = tf.train.Saver()

    # 创建字典集
    generate_words_dict_info(config.base_path)

    # 载入训练集与验证集
    print("Loading training and validation data...")
    start_time = int(time.time())
    x_train, y_train, x_val, y_val = process_file(config.base_path, config.seq_length)
    print("Time usage:", int(time.time()) - start_time)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = int(time.time())
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, model, x_val, y_val)

                if acc_val >= best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=config.save_mod)
                    improved_str = 'save'
                else:
                    improved_str = 'abandon'

                time_dif = int(time.time()) - start_time
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break
    pass


anti_text_tf_session = None


def load_tf_session():
    """
    加载模型
    :return: 
    """
    try:
        # 初始化
        config, _ = init_tf_cnn()

        # 已初始化则直接返回
        global anti_text_tf_session
        if anti_text_tf_session:
            return anti_text_tf_session

        # 初次加载字典集合
        global tf_words_dict
        dict_file = file(config.base_path + WORD_DICT_NAME)
        for line in dict_file:
            if isinstance(line, str):
                line = line.decode('utf-8')
            line = line.strip(u'\n')
            w_code, word = line.split(': ', 1)
            tf_words_dict[word] = int(w_code)

        # 初次加载训练模型
        start_time = int(time.time())
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=config.save_mod)
        anti_text_tf_session = session
        print('load time %ss' % (int(time.time()) - start_time))
        return session
    except Exception as e:
        print(str(e))
        return None


t_cnn_cfg = None
t_cnn_mod = None


def init_tf_cnn():
    """
    初始化cnn mode
    :return: 
    """
    global t_cnn_cfg
    global t_cnn_mod
    if t_cnn_cfg and t_cnn_mod:
        return t_cnn_cfg, t_cnn_mod

    # 生成配置文件
    t_cnn_cfg = TCNNConfig()
    print('create CNN config ok')

    # 生成CNN模型
    t_cnn_mod = TextCNN(t_cnn_cfg)
    print('create CNN model ok')

    # 部署相关数据
    if not os.path.exists(t_cnn_cfg.base_path):
        untar_handler = subprocess.Popen('mkdir %s' % t_cnn_cfg.base_path, shell=True, stdout=subprocess.PIPE)
        untar_handler.wait()
        untar_handler = subprocess.Popen('unzip %s -d %s' % (t_cnn_cfg.model_res_source, t_cnn_cfg.base_path), shell=True, stdout=subprocess.PIPE)
        untar_handler.wait()

    return t_cnn_cfg, t_cnn_mod


def tf_anti_text(raw_text):
    """
    使用生成的训练模型处理文本
    :return: 
    """

    # 加载模型
    session = load_tf_session()
    if not session:
        return CATEGORY_NORMAL

    # 转换输入向量
    d_ids = convert_raw_str_to_ids(raw_text)
    if not d_ids:
        return CATEGORY_NORMAL
    x_pad = kr.preprocessing.sequence.pad_sequences([d_ids], 600)

    # 输出预测结果
    y_pred_cls = np.zeros(shape=1, dtype=np.int32)
    feed_dict = {
        t_cnn_mod.input_x: x_pad,
        t_cnn_mod.keep_prob: 1.0
    }
    y_pred_cls[:1] = session.run(t_cnn_mod.y_pred_cls, feed_dict=feed_dict)
    cate_name = get_cate_name_by_id(y_pred_cls[0])
    return cate_name


v2_inited = False
v2_session = ''
v2_graph = ''
v2_vocab_processor = ''


def tf_anti_text_v2(raw_text_ori):
    """
    使用新模型的文本检测，暂时只支持广告
    输入假设为unicode
    :return:
    """
    global v2_inited
    global v2_session
    global v2_graph
    global v2_vocab_processor
    if not v2_inited:
        xcpath = os.path.realpath(__file__).split('xcspam/')[0] + 'xcspam'
        temp_xcpath = xcpath + '/bin'
        if os.path.exists(temp_xcpath):
            xcpath = temp_xcpath
        if not os.path.exists(xcpath):
            xcpath = os.path.abspath(__file__).split('rnn_anti_text.py')[0] + '..'
        cp_dir_path = xcpath + '/data/tf_anti_text_v2/checkpoints'
        vocab_path = os.path.join(cp_dir_path, "..", "vocab")
        v2_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        # checkpoint_file = tf.train.latest_checkpoint(cp_dir_path)
        # 先写死，流程正常化之后改回来tf
        checkpoint_file = cp_dir_path + '/model-9600'
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        v2_session = tf.Session(config=session_conf)
        v2_graph = v2_session.graph
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(v2_session, checkpoint_file)
        v2_inited = True

    if not raw_text_ori:
        return CATEGORY_NORMAL
    raw_text = ''
    for c in raw_text_ori:
        raw_text += c + ' '
    input_x = v2_graph.get_operation_by_name("input_x").outputs[0]
    dropout_keep_prob = v2_graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    predictions = v2_graph.get_operation_by_name("output/predictions").outputs[0]
    text = np.array(list(v2_vocab_processor.transform([raw_text])))

    predictions_res = v2_session.run(predictions, {input_x: text, dropout_keep_prob: 1.0})
    if not predictions_res:
        return CATEGORY_NORMAL

    if predictions_res[0] == 0:
        return CATEGORY_NORMAL
    else:
        return CATEGORY_AD


tf_words_dict = {}


def generate_words_dict_info(file_path):
    """
    根据指定目录下的训练数据，生成词典
    :param file_path: 
    :return: 
    """
    global tf_words_dict

    words_dict = {}
    for file_name in FILE_NAMES:
        file_handler = open(file_path + '%s.txt' % file_name)
        for line in file_handler:

            # 确保编码方式unicode
            if isinstance(line, str):
                line = line.decode('utf-8')
            line = line.strip()

            # 处理分词列表
            for word in jb.cut(line, cut_all=False):
                if word not in words_dict:
                    words_dict[word] = 0
                words_dict[word] += 1

    # 按词组出现频率进行倒排
    words_list = []
    for word, num in words_dict.items():
        words_list.append({word: num})
    words_list.sort(key=lambda x: -x.values()[0])
    words_list = words_list[:TCNNConfig.vocab_size - 1]

    # 输出字典结果
    dict_file = open(file_path + WORD_DICT_NAME, mode='w')
    for w_idx, word_info in enumerate(words_list):
        word = word_info.keys()[0]
        w_code = w_idx+1
        dict_file.write('%s: %s\n' % (w_idx, word_info.keys()[0].encode('utf-8')))
        # 构建字典集合
        tf_words_dict[word] = w_code
    dict_file.close()


class TextCNNV2(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        # 输入，None表示实际输入时决定输入条数
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # 输出
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 均匀分布随机初始化embedding矩阵，
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 输入的字符串编码矩阵[[1,2,3,4],[1,2,3,5]]编码转为embedding特征[[[0.12,0.11,...,-0.899]...,[0.22,...0,76]],[...]]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # [句,字,字embedding向量]之后再扩展一维，在论文http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
            # 中指出最后一维是仿照图片识别的channel维度，并且给出了两个channel。然而本实现对应的论文只定义了一个channel。
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # 卷积层，卷积核[区域高度，区域宽度，channel数，filter数]，
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 随机正太分布W
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # 对于每条广告语句的100*128*1个输入，filter==3时，conv会通过3*100*1*128个参数输出(100-3+1)*128个神经元
                # conv.shape = (?, 98, 1, 128)
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    # 步长均为1
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # 传说中的拯救线性wx+b的非线性，论文中写此处调各种非线性函数意义不大。这步处理之后h维度不变
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # 池化层，可以理解成把3、4、5卷积后的最显著的字组合给拎出来其他全部舍弃
                # pooled.shape = (?, 1, 1, 128)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def text_cnn_train_v2():
    """
    新版广告训练模型，准确率提升至90%
    论文：http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    :return: 
    """

    cp_dir_path = 'data/ad1/'



    # Data loading params
    tf.flags.DEFINE_float("dev_sample_percentage", .5, "Percentage of the training data to use for validation")
    # 广告文本路径
    tf.flags.DEFINE_string("positive_data_file", cp_dir_path + "ad.pos", "Data source for the positive data.")
    # 非广告文本路径
    tf.flags.DEFINE_string("negative_data_file", cp_dir_path + "ad.neg", "Data source for the negative data.")

    # Model Hyperparameters
    # 特征数，对于分词后的word2vec词向量是个固定的数组。而对于当前的基于字的cnn则一开始生成一个随机的特征向量，并且在之后也纳入训练过程。
    # 经过初步测试，词向量效果没有比字的更好，可能的原因的广告文本新词太多，训练词向量语料和jieba结合不能涵盖。
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    # 卷积核高度取值，对于nlp的cnn，卷积核的长度是固定，等于embedding层的维度，即上面的embedding_dim
    # 此外，尽管直观觉得汉语单字信息少于英语，但是经测试，3、4、5这种缺省的组合效果仍然较好
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    # 同高度卷积核filter的个数，128意味着对于embedding_dim为128的长度为100个字的字符串，输入为100字*128特征，卷积层处理后达到
    # (100-3+1)*128 + (100-4+1)*128 + (100-5+1)*128个神经元，卷积层有 3*128*128 + 4*128*128 + 5*128*128个参数
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    # 据说很重要，但是没调过
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    # 论文中写道L2 norm constraints没有什么用
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
    # 训练轮数，因为当前样本集比较小，实际测试远无需用到缺省的200轮，即使gpu加速后，200轮也是需要很长的时间
    tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    # 当没有gpu时，允许tf跑在cpu上
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    flags = tf.flags.FLAGS
    print("\nParameters:")
    for attr, value in sorted(flags.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    # x_text形如["我 是 广 告 文 本", "我 是 正 常 文 本"]，y形如[[0, 1], [1, 0]]，[0, 1]表示正常文本概率是0且广告文本概率是1
    x_text, y = load_data_and_labels(flags.positive_data_file, flags.negative_data_file, True)

    # Build vocabulary
    # 最长句子字符数，在load_data_and_labels已经限制为最长100
    max_document_length = max([len(x.split(" ")) for x in x_text])
    # 没用用分词，因为考虑广告文本很多是自己造的词或者命名实体，比如v信，佳美斯减肥茶。其实max_document_length在这一步也是多余的，因为前面
    # 已经截取过了。
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # fit_transform会按照空格过滤出所有单词，并且加以编号，["我 是 广 告 文 本", "我 是 正 常 文 本"]会被转化成[[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 5, 6]]
    # 对于没有达到max_document_length的句子，fit_transform会以0补齐，"我 是"补齐为[1, 2, 0, 0, 0, 0]
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    # 按同样的随机规则打乱文本和标签的顺序
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # 90% 作为训练数据, 10%作为验证数据
    dev_sample_index = -1 * int(flags.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            # 允许在指定了gpu然而没有的前提下tf自行选择cpu运行，比如在本地测试。gpu训练机器，ssh xc01@172.16.110.3，123456
            allow_soft_placement=flags.allow_soft_placement,
            log_device_placement=flags.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNNV2(
                # 训练语料最长句长
                sequence_length=x_train.shape[1],
                # 标签的个数，对于广告文本识别，只有两个标签，"是文本"和"不是文本"
                num_classes=y_train.shape[1],
                # 不同的字符数，即转化为数字后的字符的最大数
                vocab_size=len(vocab_processor.vocabulary_),
                # 单个字的特征数，对应到word2vec词向量更好理解
                embedding_size=flags.embedding_dim,
                # 和v1最大区别，才成其为cnn
                filter_sizes=list(map(int, flags.filter_sizes.split(","))),
                num_filters=flags.num_filters,
                l2_reg_lambda=flags.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=flags.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: flags.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter_v2(
                list(zip(x_train, y_train)), flags.batch_size, flags.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % flags.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % flags.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def batch_iter_v2(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def clean_chinese_str(s):
    out = ''
    #mystr = string.decode('utf-8')
    mystr = s
    mystr = mystr[:100]
    for letter in mystr:
        out += letter + ' '
    return out


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file, is_chinese=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    if is_chinese:
        # 内部转化为unicode，对比英文单词汉字用空格隔开，并且只取前一百个字符，对于社区的帖子、评论相对比较短的现状来说已经够了。过长的文本会导致训练时长大增
        x_text = [clean_chinese_str(sent) for sent in x_text]
    else:
        x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


if __name__ == '__main__':

    # 检测运行模式
    if len(sys.argv) != 2:
        raise ValueError("""usage: python run_cnn.py [train / anti_text ]""")

    # 进行训练
    if sys.argv[1] == 'train':
        text_cnn_train_v2()

    # 使用已生成模型进行文本分类
    elif sys.argv[1] == 'anti_text':

        test_text = '蜘蛛侠系列电影全有 需要的宝宝加我vx17856764520 秒回'
        print(tf_anti_text_v2(test_text))

        test_text = '有需要口红的小姐姐可以加我vx:xxl1113940471'
        print(tf_anti_text_v2(test_text))

        test_text = '想要这个微信wqzlpyouyi99谢谢啊'
        print(tf_anti_text_v2(test_text))
