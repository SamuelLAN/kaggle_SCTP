#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.metrics import auc_using_histogram
from lib.nn import NN
from sklearn.metrics import roc_auc_score


class CNN_1D(NN):
    ''' Shallow neural networks '''
    MODEL_NAME = 'CNN_1D'

    # parameters that need to be tuned
    BATCH_SIZE = 25
    EPOCH_TIMES = 200
    BASE_LEARNING_RATE = 0.000005
    DECAY_RATE = 0.05
    EARLY_STOP_EPOCH = 20
    KEEP_PROB = 0.5
    REG_BETA = 0.06
    WEIGHT_MAJOR = 5.0
    WEIGHT_MINOR = 0.5
    USE_BN = True

    # fix params
    INPUT_DIM = 200
    NUM_CLASSES = 2

    SHOW_PROGRESS_FREQUENCY = 100

    MODEL = [
        {
            'name': 'conv_1_1',
            'type': 'conv',
            'shape': [1, 6],
            'k_size': [1, 5],
            'bn': USE_BN,
            'stride': [1, 1, 1, 1],
        },
        {
            'name': 'conv_1_2',
            'type': 'conv',
            'shape': [6, 12],
            'k_size': [1, 5],
            'bn': USE_BN,
            'stride': [1, 1, 2, 1],
        },
        # {
        #     'type': 'pool',
        #     'k_size': [1, 2],
        # },
        {
            'name': 'conv_2_1',
            'type': 'conv',
            'shape': [12, 12],
            'k_size': [1, 5],
            'bn': USE_BN,
            'stride': [1, 1, 1, 1],
        },
        {
            'name': 'conv_2_2',
            'type': 'conv',
            'shape': [12, 32],
            'k_size': [1, 5],
            'bn': USE_BN,
            'stride': [1, 1, 2, 1],
        },
        # {
        #     'type': 'pool',
        #     'k_size': [1, 2],
        # },
        {
            'name': 'fc_3',
            'type': 'fc',
            'filter_out': 32,
            'trainable': True,
            'BN': USE_BN,
        },
        {
            'name': 'dropout',
            'type': 'dropout',
        },
        {
            'name': 'softmax',
            'type': 'fc',
            'filter_out': NUM_CLASSES,
            'activate': False,
        },
    ]

    def init(self):
        ''' customize initialization '''
        # input and label
        self.__X = tf.placeholder(tf.float32, [None, 1, self.INPUT_DIM, 1], name='X')
        self.__y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')

        # for dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.__has_rebuild = False

    def __model(self):
        if self.start_from_model:
            self.restore_model_w_b(self.start_from_model)
            self.__rebuild_model()
        else:
            self.__output = self.parse_model(self.__X)

    def __rebuild_model(self):
        self.__output = self.parse_model_rebuild(self.__X)

    def __get_loss(self):
        with tf.name_scope('loss'):
            # give the point that target == 1 bigger weight
            y_column_1 = tf.cast(self.__y[:, 1], tf.float32)
            weight = y_column_1 * self.WEIGHT_MAJOR + (-y_column_1 + 1) * self.WEIGHT_MINOR

            # softmax with weight
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__output, labels=self.__y) * weight
            )

            self.__loss = self.regularize_trainable(self.__loss, self.REG_BETA)

    def __summary(self):
        with tf.name_scope('summary'):
            self.__mean_loss = tf.placeholder(tf.float32, name='mean_loss')
            self.__mean_auc = tf.placeholder(tf.float32, name='mean_auc')

            tf.summary.scalar('learning_rate', self.__learning_rate)
            tf.summary.scalar('mean_auc', self.__mean_auc)
            tf.summary.scalar('mean_loss', self.__mean_loss)

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

    def __measure(self, X, y, add_summary=True, is_train=True, epoch=None):
        feed_dict = {self.__X: X, self.__y: y, self.keep_prob: 1.0, self.t_is_train: False}
        loss, _output = self.sess.run([self.__loss, self.__output], feed_dict)
        auc = roc_auc_score(y[:, 1], _output[:, 1])

        if not add_summary:
            return loss, auc

        feed_dict[self.__mean_auc] = auc
        if is_train:
            self.add_summary_train(feed_dict, epoch)
        else:
            self.add_summary_val(feed_dict, epoch)

        return loss, auc

    def __measure_all(self, X, y, add_summary=True, epoch=None):
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

            feed_dict = {self.__X: batch_x, self.__y: batch_y, self.keep_prob: 1.0, self.t_is_train: False}
            loss, _output = self.sess.run([self.__loss, self.__output], feed_dict)

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

        # build model and prepare some useful variable
        self.__before_train(steps)

        # init some temporary variables
        all_batch_y = []
        all_batch_output = []
        mean_loss = 0
        mean_auc = 0

        _, best_val_auc = self.__measure_all(val_x, val_y, False)
        self.echo('best val auc: %f ' % best_val_auc)

        # best_val_auc = 0.0
        decr_val_auc_times = 0
        iter_per_epoch = int((train_size + train_size % self.BATCH_SIZE) // self.BATCH_SIZE)

        # moment = 0.975
        # self.__running_mean = None
        # self.__running_std = None

        for step in range(steps):
            if step % self.SHOW_PROGRESS_FREQUENCY == 0:
                epoch_progress = float(step) % iter_per_epoch / iter_per_epoch * 100.0
                step_progress = float(step) / steps * 100.0
                self.echo('\r step: %d (%d|%.2f%%) / %d|%.2f%% \t\t' % (step, iter_per_epoch, epoch_progress,
                                                                        steps, step_progress), False)

            batch_x, batch_y = data_object.next_batch(self.BATCH_SIZE)
            batch_x = np.expand_dims(batch_x, axis=-1)
            batch_x = np.expand_dims(batch_x, axis=1)

            # self batch normalize
            # reduce_axis = tuple(range(len(batch_x.shape) - 1))
            # _mean = np.mean(batch_x, axis=reduce_axis)
            # _std = np.std(batch_x, axis=reduce_axis)
            # self.__running_mean = moment * self.__running_mean + (1 - moment) * _mean if not isinstance(
            #     self.__running_mean, type(None)) else _mean
            # self.__running_std = moment * self.__running_std + (1 - moment) * _std if not isinstance(
            #     self.__running_std, type(None)) else _std
            # batch_x = (batch_x - _mean) / (_std + self.EPSILON)

            feed_dict = {self.__X: batch_x, self.__y: batch_y, self.keep_prob: self.KEEP_PROB, self.t_is_train: True}
            _, batch_loss, batch_output = self.sess.run([self.__train_op, self.__loss, self.__output], feed_dict)

            all_batch_y.append(batch_y)
            all_batch_output.append(batch_output)
            mean_loss += batch_loss

            if step % iter_per_epoch == 0 and step != 0:
                epoch = int(step // iter_per_epoch)

                all_batch_y = np.vstack(all_batch_y)
                all_batch_output = np.vstack(all_batch_output)

                mean_loss /= iter_per_epoch
                mean_auc = roc_auc_score(all_batch_y, all_batch_output)

                # self.mean_x = self.__running_mean
                # self.std_x = self.__running_std * (self.BATCH_SIZE / float(self.BATCH_SIZE - 1))

                # train_loss, train_auc = self.__measure(train_x, train_y, True, True, epoch)
                val_loss, val_auc = self.__measure_all(val_x, val_y, True, epoch)

                self.echo('\nepoch: %d, train_loss: %.6f, train_auc: %.6f, val_loss: %.6f, val_auc: %.6f ' %
                          (epoch, mean_loss, mean_auc, val_loss, val_auc))

                mean_loss = 0
                all_batch_y = []
                all_batch_output = []

                if best_val_auc < val_auc:
                    self.echo('best ')
                    best_val_auc = val_auc
                    decr_val_auc_times = 0

                    self.save_model_w_b()
                else:
                    decr_val_auc_times += 1
                    if decr_val_auc_times > self.EARLY_STOP_EPOCH:
                        break

        # close TensorBoard
        self.close_summary()

        # restore best model and init their variables
        self.restore_model_w_b()
        self.__get_loss()
        self.init_variables()

    def predict(self, X):
        return self.sess.run(self.__output, {self.__X: X, self.keep_prob: 1.0, self.t_is_train: False})[:, 1]

    def save(self):
        return

    def test_auc(self, test_x, test_y):
        output = self.sess.run(self.__output, {self.__X: test_x, self.keep_prob: 1.0, self.t_is_train: False})
        auc = roc_auc_score(test_y[:, 1], output[:, 1])
        print('test auc: %f' % auc)
        return auc