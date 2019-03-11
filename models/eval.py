#!/usr/bin/Python
# -*- coding: utf-8 -*-
from pre_processing.process import Data
from pre_processing.processors import Processors
from models.lgb import LGB


class Eval:
    def __init__(self):
        self.__o_data = Data([], cache_name='origin_min_max_scaling', new_cache_name='')
        self.__train_x, self.__train_y = self.__o_data.train_data()
        self.__val_x, self.__val_y = self.__o_data.val_data()
        self.__test_x, self.__test_y = self.__o_data.test_data()

    def train(self):
        model = LGB()
        model.train(self.__train_x, self.__train_y, self.__test_x, self.__test_y)

    def predict(self):
        pass


o_model = Eval()
o_model.train()
