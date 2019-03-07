#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_TRAIN_DATA = os.path.join(PATH_PRJ, 'dataset', 'train.csv')
PATH_CACHE_DIR = os.path.join(PATH_CUR, 'cache')


class Data:
    def __init__(self, processors=[], train_size=0.8, val_size=0.1, test_size=0.1, cache_name=''):
        # get cache path
        cache_name = cache_name if cache_name else 'data'
        self.__cache_path = os.path.join(PATH_CACHE_DIR, cache_name + '.pkl')

        print('Start loading data from cache %s ...' % self.__cache_path)

        # if cache exist, use cache and return
        if self.__use_cache():
            print('Finish loading')
            return

        print('No cache\n\nStart loading data from %s ...' % PATH_TRAIN_DATA)

        # load data
        self.__load(train_size, val_size, train_size)

        print('Finish loading data\n\nStart pre-processing data ...')

        # pre-process data
        self.__process_data(processors)

        print('Finish pre-processing\n\nStart caching data ...')

        self.__cache()

        print('Finish caching')

    def __load(self, train_size, val_size, test_size):
        ''' load data '''
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
            self.__train_x, self.__val_x, self.__test_x, self.__train_y, self.__val_y, self.__test_y = \
                processor(self.__train_x, self.__val_x, self.__test_x, self.__train_y, self.__val_y, self.__test_y)
            print('finish running pre-processor %d' % i)

    def __cache(self):
        ''' cache data after pre-processing '''
        with open(self.__cache_path, 'wb') as f:
            pickle.dump((self.__train_x, self.__val_x, self.__test_x,
                         self.__train_y, self.__val_y, self.__test_y), f, pickle.HIGHEST_PROTOCOL)

    def __use_cache(self):
        ''' load from cache '''
        if not os.path.isfile(self.__cache_path):
            return False

        with open(self.__cache_path, 'rb') as f:
            data = pickle.load(f)
        self.__train_x, self.__val_x, self.__test_x, self.__train_y, self.__val_y, self.__test_y = data
        return True

    def train_data(self):
        return self.__train_x, self.__train_y

    def val_data(self):
        return self.__val_x, self.__val_y

    def test_data(self):
        return self.__test_x, self.__test_y

    def next_batch(self):
        ''' Get train data batch by batch '''
        pass


from processors import Processors

o_data = Data([Processors.smote])
train_x, train_y = o_data.train_data()
val_x, val_y = o_data.val_data()

print(train_x.shape)
print(train_y.shape)

print(val_x.shape)
print(val_y.shape)

print(train_x)
