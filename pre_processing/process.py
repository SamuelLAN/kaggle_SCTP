#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd

PATH_CUR = os.path.split(__file__)[0]
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_TRAIN_DATA = os.path.join(PATH_PRJ, 'dataset', 'train.csv')


class Data:
    def __init__(self, train_size=0.8, val_size=0.1, test_size=0.1):
        print('Start loading data from %s ...' % PATH_TRAIN_DATA)
        data = np.asarray(pd.read_csv(PATH_TRAIN_DATA).iloc[:, :])
        data = self.__shuffle_data(data)

        # divide the data into trainset, valset, testset
        train_data, val_data, test_data = self.__divide_data(data, train_size, val_size, test_size)

        self.__train_x = train_data[:, 2:]
        self.__train_y = train_data[:, 1]
        self.__val_x = val_data[:, 2:]
        self.__val_y = val_data[:, 1]
        self.__test_x = test_data[:, 2:]
        self.__test_y = test_data[:, 1]

        # delete useless data
        del data, train_data, val_data, test_data
        print('Finish loading data')

    @staticmethod
    def __divide_data(data, train_size=0.8, val_size=0.1, test_size=0.1):
        ''' divide the data into trainset, valset, testset '''
        train_data = Data.__get_sub_data(data, 0.0, train_size)
        val_data = Data.__get_sub_data(data, train_size, train_size + val_size)
        test_data = Data.__get_sub_data(data, 1.0 - test_size, 1.0)
        return train_data, val_data, test_data

    @staticmethod
    def __get_sub_data(data, start_ratio, end_ratio):
        ''' According to the ratio, get part of the data '''
        len_data = len(data)
        start_index = len_data * start_ratio
        end_index = len_data * end_ratio
        return data[start_index: end_index]

    @staticmethod
    def __shuffle_data(data):
        ''' shuffle data '''
        # generate shuffle indices
        indices = range(len(data))
        random.shuffle(indices)

        # according to the shuffled indices, reorder the data
        new_data = []
        for i in indices:
            new_data.append(data[i, :])
        return np.asarray(new_data)
