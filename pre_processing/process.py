#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
import pandas as pd

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_TRAIN_DATA = os.path.join(PATH_PRJ, 'dataset', 'train.csv')


class Data:
    def __init__(self, processors=[], train_size=0.8, val_size=0.1, test_size=0.1):
        print('Start loading data from %s ...' % PATH_TRAIN_DATA)
        data = np.asarray(pd.read_csv(PATH_TRAIN_DATA).iloc[:, :])
        print('shuffling data ...')
        data = self.__shuffle_data(data)

        print('dividing data ...')
        # divide the data into trainset, valset, testset
        train_data, val_data, test_data = self.__divide_data(data, train_size, val_size, test_size)

        self.__train_x = np.cast[np.float32](train_data[:, 2:])
        self.__train_y = np.cast[np.int32](train_data[:, 1])
        self.__val_x = np.cast[np.float32](val_data[:, 2:]) if val_data.any() else []
        self.__val_y = np.cast[np.int32](val_data[:, 1]) if val_data.any() else []
        self.__test_x = np.cast[np.float32](test_data[:, 2:])
        self.__test_y = np.cast[np.int32](test_data[:, 1])

        # delete useless data
        del data, train_data, val_data, test_data
        print('Finish loading data\n\nStart pre-processing data ...')

        # pre-process data
        self.__process_data(processors)

        print('Finish pre-processing')

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
        start_index = int(len_data * start_ratio)
        end_index = int(len_data * end_ratio)
        return data[start_index: end_index]

    def __process_data(self, processors=[]):
        ''' pre-process data '''
        if not processors:
            return

        for i, processor in enumerate(processors):
            # processor must be function
            if not callable(processor):
                continue

            print('running pre-processor %d ... ' % i)
            self.__train_x, self.__val_x, self.__test_x = processor(self.__train_x, self.__val_x, self.__test_x)
            print('finish running pre-processor %d' % i)

    def train_data(self):
        return self.__train_x, self.__train_y

    def val_data(self):
        return self.__val_x, self.__val_y

    def test_data(self):
        return self.__test_x, self.__test_y

    def next_batch(self):
        ''' Get train data batch by batch '''
        pass


# from processors import Processors
#
# # o_data = Data([Processors.min_max_scaling])
# o_data = Data([Processors.standardization])
# train_x, train_y = o_data.train_data()
#
# print(train_x.shape)
