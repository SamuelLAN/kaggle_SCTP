#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import lib.nn as base

'''
 长短途记忆模型 LSTM
'''


class BiLSTM(base.NN):
    MODEL_NAME = 'bi_lstm'  # 模型的名称

    ''' 参数 '''

    SHOW_PROGRESS_FREQUENCY = 100

    WEIGHT_MAJOR = 5.0
    WEIGHT_MINOR = 0.5

    EARLY_STOP_EPOCH = 20

    # 初始 学习率
    BASE_LEARNING_RATE = 0.01
    # 学习率 的 下降速率
    DECAY_RATE = 0.4

    # 梯度剪切的参数
    CLIP_NORM = 1.25

    # dropout 的 keep_prob
    KEEP_PROB = 0.5
    REG_BETA = 0.03

    NUM_STEPS = 200  # 序列数据一次输入 num_steps 个数据
    BATCH_SIZE = 25  # 随机梯度下降的 batch 大小
    EPOCH_TIMES = 200  # 迭代的 epoch 次数

    EMBEDDING_SIZE = 1  # 词向量的大小
    NUM_NODES = 64  # layer1 有多少个节点
    NUM_CLASSES = 2

    # 输入 data 的 shape
    SHAPE_DATA = [BATCH_SIZE, ]
    # cell 中与输入相乘的权重矩阵的 shape
    SHAPE_INPUT = [EMBEDDING_SIZE, NUM_NODES]
    # cell 中与上次输出相乘的权重矩阵的 shape
    SHAPE_OUTPUT = [NUM_NODES, NUM_NODES]

    # 正则化的 beta 参数
    # REGULAR_BETA = 0.01

    # 若校验集的 accuracy 连续超过 EARLY_STOP_CONDITION 次没有高于 best_accuracy, 则提前结束迭代
    EARLY_STOP_CONDITION = 100

    # 每 summary_frequency 次迭代输出信息
    SUMMARY_FREQUENCY = 100

    ''' 自定义 初始化变量 过程 '''

    def init(self):
        # 最后分类器的权重矩阵的 shape
        self.__shape_fc_w = [self.NUM_NODES * 2, self.NUM_CLASSES]

        # 输入 与 label
        self.__X = []
        for _ in range(self.NUM_STEPS):
            input_ph = tf.placeholder(tf.float32, shape=[None, 1], name='input')
            self.__X.append(input_ph)

        self.__y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    ''' 初始化 各种门需要的 权重矩阵 以及 偏置量 '''

    @staticmethod
    def __init_gate(name=''):
        # 与输入相乘的权重矩阵
        w_x = BiLSTM.init_weight(BiLSTM.SHAPE_INPUT, name=name + 'x')
        # 与上次输出相乘的权重矩阵
        w_m = BiLSTM.init_weight(BiLSTM.SHAPE_OUTPUT, name=name + 'm')
        # bias
        b = BiLSTM.init_bias(BiLSTM.SHAPE_INPUT, name=name + 'b')
        return w_x, w_m, b

    ''' 初始化权重矩阵 '''

    @staticmethod
    def init_weight(shape, name='weights'):
        return tf.Variable(tf.truncated_normal(shape, -0.1, 0.1), name=name)

    def lstm(self, inputs, name='lstm'):
        with tf.name_scope(name):
            # ********************** 初始化模型所需的变量 **********************

            # 输入门
            with tf.name_scope('input_gate'):
                w_input_x, w_input_m, b_input = self.__init_gate()

            # 忘记门
            with tf.name_scope('forget_gate'):
                w_forget_x, w_forget_m, b_forget = self.__init_gate()

            # 记忆单元
            with tf.name_scope('lstm_cell'):
                w_cell_x, w_cell_m, b_cell = self.__init_gate()

            # 输出门
            with tf.name_scope('output_gate'):
                w_output_x, w_output_m, b_output = self.__init_gate()

            # 在序列 num steps 之间保存状态的变量
            with tf.name_scope('saved_variable'):
                saved_output = tf.Variable(tf.zeros([self.BATCH_SIZE, self.NUM_NODES]),
                                           trainable=False, name='saved_output')
                saved_state = tf.Variable(tf.zeros([self.BATCH_SIZE, self.NUM_NODES]),
                                          trainable=False, name='saved_state')

            # ************************* 生成模型 ****************************

            outputs = list()
            output = saved_output
            state = saved_state

            for _input in inputs:
                output, state = self.__cell(_input, output, state,
                                            w_input_x, w_input_m, b_input,
                                            w_forget_x, w_forget_m, b_forget,
                                            w_cell_x, w_cell_m, b_cell,
                                            w_output_x, w_output_m, b_output)
                outputs.append(output)

        return w_input_x, w_input_m, b_input, \
               w_forget_x, w_forget_m, b_forget, \
               w_cell_x, w_cell_m, b_cell, \
               w_output_x, w_output_m, b_output, \
               saved_output, saved_state, \
               outputs, output, state

    ''' 模型 '''

    def __model(self):
        # 正向 lstm 模型
        self.__wIX, self.__wIM, self.__bI, \
        self.__wFX, self.__wFM, self.__bF, \
        self.__wCX, self.__wCM, self.__bC, \
        self.__wOX, self.__wOM, self.__bO, \
        self.__saved_output, self.__saved_state, \
        self.__outputs, output, state = self.lstm(self.__X, 'forward_lstm')

        # 将输入反向
        indices = list(range(len(self.__X)))
        indices.reverse()
        reverse_x = [self.__X[i] for i in indices]

        # 反向 lstm 模型
        self.__wIX_reverse, self.__wIM_reverse, self.__bI_reverse, \
        self.__wFX_reverse, self.__wFM_reverse, self.__bF_reverse, \
        self.__wCX_reverse, self.__wCM_reverse, self.__bC_reverse, \
        self.__wOX_reverse, self.__wOM_reverse, self.__bO_reverse, \
        self.__saved_output_reverse, self.__saved_state_reverse, \
        self.__outputs_reverse, output_reverse, state_reverse = self.lstm(reverse_x, 'reverse_lstm')

        # 将输出再次反向
        self.__outputs_reverse.reverse()

        # 全连接层，增加左右关联
        with tf.name_scope('fc'):
            self.__w = self.init_weight(self.__shape_fc_w, name='w')
            self.__b = self.init_bias(self.__shape_fc_w, name='b')

        # *********************** 计算 loss *****************************

        # 保证状态传递完毕
        with tf.control_dependencies([self.__saved_output.assign(output),
                                      self.__saved_state.assign(state),
                                      self.__saved_output_reverse.assign(output_reverse),
                                      self.__saved_state_reverse.assign(state_reverse)]):
            with tf.name_scope('softmax'):
                outputs = tf.concat(self.__outputs, axis=0, name='outputs_forward_concat')
                outputs_reverse = tf.concat(self.__outputs_reverse, axis=0, name='outputs_reverse_concat')
                outputs_concat = tf.concat([outputs, outputs_reverse], axis=-1, name='outputs_concat')

                # 分类器
                self.__logits = tf.nn.xw_plus_b(outputs_concat, self.__w, self.__b)
                self.__output_prob = tf.nn.softmax(self.__logits)

            # 计算 loss
            self.__get_loss()

    ''' 定义每个单元里的计算过程 '''

    def __cell(self, _input, _output, _state,
               w_input_x, w_input_m, b_input,
               w_forget_x, w_forget_m, b_forget,
               w_cell_x, w_cell_m, b_cell,
               w_output_x, w_output_m, b_output):
        with tf.name_scope('cell'):
            # _input = tf.nn.dropout(_input, self.keep_prob, name='input_dropout')

            with tf.name_scope('input_gate'):
                input_gate = tf.sigmoid(tf.matmul(_input, w_input_x) + tf.matmul(_output, w_input_m) + b_input,
                                        name='input_gate')

            with tf.name_scope('forget_gate'):
                forget_gate = tf.sigmoid(tf.matmul(_input, w_forget_x) + tf.matmul(_output, w_forget_m) + b_forget,
                                         name='forget_gate')

            update = tf.add(tf.matmul(_input, w_cell_x) + tf.matmul(_output, w_cell_m), b_cell,
                            name='update')

            _state = tf.add(forget_gate * _state, input_gate * tf.tanh(update), name='state')

            with tf.name_scope('output_gate'):
                output_gate = tf.sigmoid(tf.matmul(_input, w_output_x) + tf.matmul(_output, w_output_m) + b_output,
                                         name='output_gate')

            _output = tf.multiply(output_gate, tf.tanh(_state), name='output')

            # 添加 dropout
            _output = tf.nn.dropout(_output, self.keep_prob, name='output_drop')
            return _output, _state

    ''' 计算 loss '''

    def __get_loss(self):
        with tf.name_scope('loss'):
            # give the point that target == 1 bigger weight
            y_column_1 = tf.cast(self.__y[:, 1], tf.float32)
            weight = y_column_1 * self.WEIGHT_MAJOR + (-y_column_1 + 1) * self.WEIGHT_MINOR

            # self.__labels = tf.concat(self.__y, 0, name='labels')
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__logits, labels=self.__y) * weight,
                name='loss'
            )

            self.__loss = self.regularize(self.__loss, self.REG_BETA)

    ''' 获取 train_op '''

    def __get_train_op(self, loss, learning_rate, global_step):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.CLIP_NORM)
            self.__train_op = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    def __summary(self):
        ''' record some indicators to tensorboard '''
        with tf.name_scope('summary'):
            self.__mean_loss = tf.placeholder(tf.float32, name='loss')
            self.__mean_auc = tf.placeholder(tf.float32, name='auc')

            tf.summary.scalar('learning_rate', self.__learning_rate)
            tf.summary.scalar('mean_auc', self.__mean_auc)
            tf.summary.scalar('mean_loss', self.__mean_loss)

    ''' 初始化 校验集 或 测试集 预测所需的变量 '''

    def __predict(self):
        with tf.name_scope('sample_predict'):
            self.__sampleInput = tf.placeholder(tf.int32, name='sample_input')
            sample_embed = tf.nn.embedding_lookup(self.__embedding_matrix, self.__sampleInput,
                                                  name='sample_embed_input')

            self.__sampleSavedOutput = tf.Variable(tf.zeros([1, self.NUM_NODES]), name='sample_saved_output')
            self.__sampleSavedState = tf.Variable(tf.zeros([1, self.NUM_NODES]), name='sample_saved_state')

            self.__sampleResetState = tf.group(
                self.__sampleSavedOutput.assign(tf.zeros([1, self.NUM_NODES])),
                self.__sampleSavedState.assign(tf.zeros([1, self.NUM_NODES]))
            )

            self.__sampleOutput, self.__sampleState = self.__cell(sample_embed,
                                                                  self.__sampleSavedOutput,
                                                                  self.__sampleSavedState,
                                                                  self.__wIX, self.__wIM, self.__bI,
                                                                  self.__wFX, self.__wFM, self.__bF,
                                                                  self.__wCX, self.__wCM, self.__bC,
                                                                  self.__wOX, self.__wOM, self.__bO)

            with tf.control_dependencies([
                self.__sampleSavedOutput.assign(self.__sampleOutput),
                self.__sampleSavedState.assign(self.__sampleState)
            ]):
                self.__samplePrediction = tf.nn.softmax(tf.nn.xw_plus_b(self.__sampleOutput, self.__w, self.__b))

    def __before_train(self, steps):
        # with the iterations going, the learning rate will be decreased
        self.__learning_rate = self.get_learning_rate(
            self.BASE_LEARNING_RATE, self.global_step, steps, self.DECAY_RATE, staircase=False
        )

        # build model
        self.__model()

        # get some useful variables and add to tensorboard
        self.__get_loss()
        self.__summary()
        self.__train_op = self.get_train_op(self.__loss, self.__learning_rate, self.global_step)

        # init
        self.init_variables()
        self.merge_summary()

    def __measure(self, X, y, add_summary=True, epoch=None):
        batch_size = 100
        len_x = len(X)
        cur_index = 0

        mean_loss = 0
        all_batch_y = []
        all_batch_output = []
        times = 0

        while cur_index < len_x:
            batch_x = X[cur_index: cur_index + batch_size]
            batch_y = y[cur_index: cur_index + batch_size]
            cur_index += batch_size
            times += 1

            batch_x = self.transform_one(batch_x)

            # 给 feed_dict 赋值
            feed_dict = {self.__y: batch_y, self.keep_prob: 1.0, self.t_is_train: False}
            for i in range(self.NUM_STEPS):
                feed_dict[self.__X[i]] = batch_x[:, i:i+1]
            loss, _output = self.sess.run([self.__loss, self.__output_prob], feed_dict)

            mean_loss += loss
            all_batch_y.append(batch_y)
            all_batch_output.append(_output)

        mean_loss /= times
        mean_auc = roc_auc_score(np.vstack(all_batch_y), np.vstack(all_batch_output))

        if not add_summary:
            return mean_loss, mean_auc

        feed_dict = {self.__mean_loss: mean_loss, self.__mean_auc: mean_auc}
        self.add_summary_val(feed_dict, epoch)
        return mean_loss, mean_auc

    def train(self, data_object, train_x, train_y, val_x, val_y):
        # calculate train steps
        train_size = len(train_y)
        steps = train_size / self.BATCH_SIZE * self.EPOCH_TIMES
        iter_per_epoch = int((train_size + train_size % self.BATCH_SIZE) // self.BATCH_SIZE)

        # init model and variables
        self.__before_train(steps)

        # init some temporary variables
        all_batch_y = []
        all_batch_output = []
        mean_loss = 0
        decr_val_auc_times = 0

        _, best_val_auc = self.__measure(val_x, val_y, False)
        self.echo('best val auc: %f ' % best_val_auc)

        self.echo('\nStart training ... \n')

        for step in range(steps):
            # show the progress
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % iter_per_epoch / iter_per_epoch * 100.0
                step_progress = float(step) / steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, iter_per_epoch, epoch_progress,
                                                                        steps, step_progress), False)

                # 获取下一个 batch 的数据
            batch_x, batch_y = data_object.next_batch(self.BATCH_SIZE)
            batch_x = self.transform_one(batch_x)

            # 给 feed_dict 赋值
            feed_dict = {self.__y: batch_y, self.keep_prob: self.KEEP_PROB, self.t_is_train: True}
            for i in range(self.NUM_STEPS):
                feed_dict[self.__X[i]] = batch_x[:, i]

            # 运行 训练
            _, batch_loss, batch_output = self.sess.run([self.__train_op, self.__loss, self.__output_prob], feed_dict)

            # record the training result
            all_batch_y.append(batch_y)
            all_batch_output.append(batch_output)
            mean_loss += batch_loss

            # tag_accuracy, sentence_accuracy = self.evaluate(batch_x_train, batch_y_train, train_logits)

            # # 记录训练指标到平均值
            # mean_train_loss += train_loss
            # mean_train_accuracy += train_accuracy
            # mean_train_tag_accuracy += tag_accuracy
            # mean_train_sentence_accuracy += sentence_accuracy

            # after finish a epoch, evaluate the model
            if step % iter_per_epoch == 0 and step != 0:
                epoch = int(step // iter_per_epoch)

                # for calculating the mean training auc and loss
                all_batch_y = np.vstack(all_batch_y)
                all_batch_output = np.vstack(all_batch_output)

                mean_loss /= iter_per_epoch
                mean_auc = roc_auc_score(all_batch_y, all_batch_output)

                # for training tensorboard
                feed_dict[self.__mean_loss] = mean_loss
                feed_dict[self.__mean_auc] = mean_auc
                self.add_summary_train(feed_dict, epoch)

                # train_loss, train_auc = self.__measure(train_x, train_y, True, True, epoch)
                val_loss, val_auc = self.__measure(val_x, val_y, True, epoch)

                # for showing the result to console
                self.echo('\nepoch: %d, train_loss: %.6f, train_auc: %.6f, val_loss: %.6f, val_auc: %.6f ' %
                          (epoch, mean_loss, mean_auc, val_loss, val_auc))

                # reinitialize variables
                mean_loss = 0
                all_batch_y = []
                all_batch_output = []

                # decide whether to early stop the training
                if best_val_auc < val_auc:
                    # if best result
                    self.echo('best ')
                    best_val_auc = val_auc
                    decr_val_auc_times = 0

                    # save the best model
                    self.save_model()
                else:
                    decr_val_auc_times += 1
                    # if match the early stop conditions, then stop
                    if decr_val_auc_times > self.EARLY_STOP_EPOCH:
                        break

        self.echo('finish training         ')

        # close TensorBoard
        self.close_summary()

        # restore best model and init their variables
        self.restore_model()
        # self.__get_loss()
        # self.init_variables()

        self.echo('done')

    def transform_one(self, X):
        return X
        # return np.expand_dims(X, -1)


# o_nn = BiLSTM()
# o_nn.run()
