#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from models.eval_base import Eval
from models.shallow_neural_network import ShallowNN
from models.cnn_1d import CNN_1D


class EvalNN(Eval):
    MODEL_NAME = 'CNN_1D'
    DATA_CACHE_NAME = 'origin_min_max_scaling'
    PROCESSOR_LIST = []

    def init_model(self):
        # if we want to use a specific trained model, we can set the start_time
        start_time = '2019_03_23_03_56_50'
        # self.model = ShallowNN(False, start_time)
        self.model = CNN_1D(False, start_time)

        # add start_time to model_name, so that we can diff the model when saving results
        if not start_time:
            start_time = self.model.TIME
        self.model_name = '%s_%s_%s' % (self.MODEL_NAME, self.DATA_CACHE_NAME, start_time)

    def __transform_data(self):
        ''' transform labels to ont-hot format '''
        self.train_y = np.eye(2, dtype=np.float32)[self.train_y]
        self.val_y = np.eye(2, dtype=np.float32)[self.val_y]
        self.test_y = np.eye(2, dtype=np.float32)[self.test_y]

        self.train_x = np.expand_dims(self.train_x, axis=-1)
        self.train_x = np.expand_dims(self.train_x, axis=1)
        self.val_x = np.expand_dims(self.val_x, axis=-1)
        self.val_x = np.expand_dims(self.val_x, axis=1)
        self.test_x = np.expand_dims(self.test_x, axis=-1)
        self.test_x = np.expand_dims(self.test_x, axis=1)
        self.real_test_x = np.expand_dims(self.real_test_x, axis=-1)
        self.real_test_x = np.expand_dims(self.real_test_x, axis=1)

    def train(self):
        ''' train and save model '''
        self.__transform_data()
        self.model.train(self.data, self.train_x, self.train_y, self.val_x, self.val_y)
        self.model.test_auc(self.test_x, self.test_y)
        self.model.save()


o_model = EvalNN()
o_model.train()
o_model.predict()

print('\ndone')
