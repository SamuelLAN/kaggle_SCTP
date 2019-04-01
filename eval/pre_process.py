#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np


def process(aug_processors, norm_processors, feat_processors, x, y, trn_idx, val_idx, x_test):
    """ pre-process data """
    print('Start pre-processing data ...')

    train_x, train_y, val_x, val_y, test_x = __split_and_copy(x, y, trn_idx, val_idx, x_test)

    print('augmenting ...')
    # data augmentation
    train_x, train_y = __augment(train_x, train_y, aug_processors)

    print('normalizing ...')
    # normalization
    train_x, val_x, test_x = __normalize(train_x, val_x, test_x, norm_processors)

    print('features changing ...')
    # features change
    train_x, val_x, test_x = __feature_change(train_x, train_y, val_x, test_x, feat_processors)

    print('Finish pre-processing')

    return train_x, train_y, val_x, val_y, test_x


def __split_and_copy(x, y, trn_idx, val_idx, x_test):
    """ according to the k-fold indices, get the train data, val data """
    return np.cast['float32'](np.copy(x[trn_idx])), np.cast['int32'](np.copy(y[trn_idx])), \
           np.cast['float32'](np.copy(x[val_idx])), np.cast['int32'](np.copy(y[val_idx])), \
           np.cast['float32'](np.copy(x_test))


def __augment(train_x, train_y, processors):
    """ data augmentation """
    if processors:
        for processor in processors:
            X, y = processor(train_x, train_y)
    return train_x, train_y


def __normalize(train_x, val_x, test_x, processors):
    """ data normalization """
    if processors:
        for processor in processors:
            train_x, val_x, test_x = processor(train_x, val_x, test_x)
    return train_x, val_x, test_x


def __feature_change(train_x, train_y, val_x, test_x, processors):
    """ change (add or subtract) some features """
    if processors:
        for processor in processors:
            train_x, val_x, test_x = processor(train_x, train_y, val_x, test_x)
    return train_x, val_x, test_x
