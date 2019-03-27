#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)
os.chdir(PATH_CUR)

from models.eval_base import Eval
from models.shallow_neural_network import ShallowNN
from models.cnn_1d import CNN1D
from models.lenet import LeNet
from models.bilstm import BiLSTM
from pre_processing.process import EnsembleWriteData
from pre_processing.processors import Processors


class EvalNN(Eval):
    MODEL_NAME = 'CNN_1D'
    DATA_CACHE_NAME = 'origin_7_min_max_scaling_aug_4.0_shuffle_dim_2.0_0.0'
    PROCESSOR_LIST = []

    def init_model(self):
        # if we want to use a specific trained model, we can set the start_time
        # start_time = '2019_03_23_03_56_50'    # for shallowNN
        start_time = '2019_03_24_20_16_21'  # for CNN1D
        # start_time = '2019_03_25_04_06_48'  # for LeNet
        # start_time = ''  # for BiLSTM
        # self.model = ShallowNN(False, start_time)
        self.model = CNN1D(False, start_time)
        # self.model = LeNet(False, start_time)
        # self.model = BiLSTM(False, start_time)

        # add start_time to model_name, so that we can diff the model when saving results
        if not start_time:
            start_time = self.model.TIME
        self.model_name = '%s_%s_%s' % (self.MODEL_NAME, self.DATA_CACHE_NAME, start_time)

    def __transform_data(self):
        ''' transform data to the appropriate format '''
        # transform labels to ont-hot format
        self.train_y = np.eye(2, dtype=np.float32)[np.cast['int32'](self.train_y)]
        self.val_y = np.eye(2, dtype=np.float32)[np.cast['int32'](self.val_y)]
        self.test_y = np.eye(2, dtype=np.float32)[np.cast['int32'](self.test_y)]

        # different model may need different format of the data, let the model transform the data into correct format
        self.train_x, self.val_x, self.test_x, self.real_test_x, self.train_y, self.val_y, self.test_y = \
            self.model.transform(self.train_x, self.val_x, self.test_x, self.real_test_x,
                                 self.train_y, self.val_y, self.test_y)

    def train(self):
        ''' train and save model '''
        self.__transform_data()
        self.model.train(self.data, self.train_x, self.train_y, self.val_x, self.val_y)
        self.model.test_auc(self.test_x, self.test_y)
        self.model.save()

    def gen_ensemble_data(self):
        print('\nStart generating ensemble data ...\ntransforming data ...')

        ensemble_train_x = self.model.transform_one(self.data.ensemble_train_x)
        ensemble_test_x = self.model.transform_one(self.data.ensemble_test_x)

        print('generating ensemble train data ... ')

        # predict train result and save it to the ensemble data
        train_prob = self.model.predict(ensemble_train_x)
        EnsembleWriteData.write(self.model_name, train_prob)

        print('generating ensemble test data ... ')

        # predict test result and save it to the ensemble data
        test_prob = self.model.predict(ensemble_test_x)
        EnsembleWriteData.write(self.model_name, test_prob, False)

        print('Finish generating ensemble data')


o_model = EvalNN()
o_model.train()
o_model.predict()
o_model.gen_ensemble_data()

print('\ndone')
