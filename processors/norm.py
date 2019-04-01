#!/usr/bin/Python
# -*- coding: utf-8 -*-
from lib.ml import Norm


def standardization(train_x, val_x, test_x):
    ''' Norm formula: (x - mean) / std '''
    train_x, means, stds = Norm.standardization(train_x, 0)
    val_x, _, _ = Norm.standardization(val_x, 0, means, stds)
    test_x, _, _ = Norm.standardization(test_x, 0, means, stds)
    return train_x, val_x, test_x


def min_max_scaling(train_x, val_x, test_x):
    ''' Norm formula: (x - min) / (max - min) '''
    train_x, minimums, maximums = Norm.min_max_scaling(train_x, 0)
    val_x, _, _ = Norm.min_max_scaling(val_x, 0, minimums, maximums)
    test_x, _, _ = Norm.min_max_scaling(test_x, 0, minimums, maximums)
    return train_x, val_x, test_x
