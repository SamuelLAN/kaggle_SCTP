#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_RESULT = os.path.join(PATH_CUR, 'results')
sys.path.append(PATH_PRJ)

from pre_processing.process import Data
from pre_processing.processors import Processors
from models.lgb import LGB
from models.xgb import XGB


class Eval:
    MODEL_NAME = 'lgb'
    DATA_CACHE_NAME = 'origin'

    def __init__(self):
        # Load train data
        self.__data = Data([], cache_name=self.DATA_CACHE_NAME, new_cache_name='')
        self.__train_x, self.__train_y = self.__data.train_data()
        self.__val_x, self.__val_y = self.__data.val_data()
        self.__test_x, self.__test_y = self.__data.test_data()
        self.__real_test_x, self.__real_test_ids = self.__data.real_test_data()

        # init model
        self.__model_name = '%s_%s' % (self.MODEL_NAME, self.DATA_CACHE_NAME)
        self.__model = LGB(self.__model_name)
        # self.__model = XGB()

    def train(self):
        self.__model.train(self.__train_x, self.__train_y, self.__val_x, self.__val_y)
        self.__model.test_auc(self.__test_x, self.__test_y)
        self.__model.save()

    def predict(self):
        print('\nStart predicting test data ...')
        prob = self.__model.predict(self.__real_test_x)
        print('Finish predicting')
        # save result to csv
        self.gen_csv(self.__model_name, self.__real_test_ids, prob)

    @staticmethod
    def gen_csv(name, ids, prob):
        print('\nStart generating %s.csv ...' % name)

        # integrate content
        content = 'ID_code,target\n'
        for i in range(len(ids)):
            content += '%s,%f\n' % (ids[i], prob[i])

        # write content to csv file
        with open(os.path.join(PATH_RESULT, name + '.csv'), 'wb') as f:
            f.write(content)

        print('Finish generating csv')


o_model = Eval()
o_model.train()
o_model.predict()
