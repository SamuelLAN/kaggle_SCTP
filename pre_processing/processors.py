#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import random
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from lib.ml import Norm


class Processors:
    '''
    Processors for pre-processing data.
    Every func must have the param (train_x, val_x, test_x, train_y, val_y, test_y).
                and return (train_x, val_x, test_x, train_y, val_y, test_y).
    '''

    def __init__(self):
        pass

    @staticmethod
    def standardization(train_x, val_x, test_x, train_y, val_y, test_y):
        ''' Norm formula: (x - mean) / std '''
        train_data, means, stds = Norm.standardization(train_x, 0)
        # val_data may be null
        if val_x.any():
            val_data, _, _ = Norm.standardization(val_x, 0, means, stds)
        test_x, _, _ = Norm.standardization(test_x, 0, means, stds)
        del means, stds
        return train_x, val_x, test_x, train_y, val_y, test_y

    @staticmethod
    def min_max_scaling(train_x, val_x, test_x, train_y, val_y, test_y):
        ''' Norm formula: (x - min) / (max - min) '''
        train_x, minimums, maximums = Norm.min_max_scaling(train_x, 0)
        # val_data may be null
        if val_x.any():
            val_x, _, _ = Norm.min_max_scaling(val_x, 0, minimums, maximums)
        test_x, _, _ = Norm.min_max_scaling(test_x, 0, minimums, maximums)
        del minimums, maximums
        return train_x, val_x, test_x, train_y, val_y, test_y

    @staticmethod
    def under_sample(train_x, val_x, test_x, train_y, val_y, test_y):
        ''' under sample the majority class '''
        ratio_1_by_0 = 2.0
        num_equal_1 = np.sum(train_y)
        num_equal_0_sample = int(num_equal_1 * ratio_1_by_0)

        # store the data after under sampling
        new_train_x = []
        new_train_y = []

        # record the number of data which has sampled
        num_has_sample_0 = 0
        num_has_sample_1 = 0

        # start sample data
        for i in range(len(train_y)):
            # if complete sampling, break
            if num_has_sample_1 >= num_equal_1 and num_has_sample_0 >= num_equal_0_sample:
                break

            # record the number of data sampled
            if train_y[i] == 0:
                if num_has_sample_0 >= num_equal_0_sample:
                    continue
                num_has_sample_0 += 1
            else:
                num_has_sample_1 += 1

            # save the sample data
            new_train_x.append(train_x[i])
            new_train_y.append(train_y[i])

        # shuffle data
        new_train_x, new_train_y = Processors.__shuffle(new_train_x, new_train_y)

        return new_train_x, val_x, test_x, new_train_y, val_y, test_y

    @staticmethod
    def lda(train_x, val_x, test_x, train_y, val_y, test_y):
        ''' LDA reduce the dimensions of the features '''
        _lda = LDA()
        train_x = _lda.fit_transform(train_x, train_y)
        val_x = _lda.transform(val_x)
        test_x = _lda.transform(test_x)
        return train_x, val_x, test_x, train_y, val_y, test_y

    @staticmethod
    def __shuffle(X, y):
        ''' shuffle data '''
        # generate shuffled indices
        shuffle_indices = range(len(y))
        random.shuffle(shuffle_indices)

        # according to the shuffled indices, generate new data
        new_x = []
        new_y = []
        for i in shuffle_indices:
            new_x.append(X[i])
            new_y.append(y[i])

        return np.asarray(new_x), np.asarray(new_y)
