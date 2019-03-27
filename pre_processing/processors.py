#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

from lib.ml import Norm, Sampling


class Processors:
    '''
    Processors for pre-processing data.
    Every func must have the param (train_x, val_x, test_x, train_y, val_y, test_y).
                and return (train_x, val_x, test_x, train_y, val_y, test_y).
    '''

    def __init__(self):
        pass

    @staticmethod
    def standardization(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' Norm formula: (x - mean) / std '''
        train_x, means, stds = Norm.standardization(train_x, 0)
        # val_data may be null
        if val_x.any():
            val_x, _, _ = Norm.standardization(val_x, 0, means, stds)
        test_x, _, _ = Norm.standardization(test_x, 0, means, stds)
        real_test_x, _, _ = Norm.standardization(real_test_x, 0, means, stds)
        del means, stds
        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def min_max_scaling(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' Norm formula: (x - min) / (max - min) '''
        train_x, minimums, maximums = Norm.min_max_scaling(train_x, 0)
        # val_data may be null
        if val_x.any():
            val_x, _, _ = Norm.min_max_scaling(val_x, 0, minimums, maximums)
        test_x, _, _ = Norm.min_max_scaling(test_x, 0, minimums, maximums)
        real_test_x, _, _ = Norm.min_max_scaling(real_test_x, 0, minimums, maximums)
        del minimums, maximums
        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def duplicate(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' Duplicate the minority several times '''
        # the ratio of sampling minority
        ratio_duplicate = 4.5

        print('Start duplication ... ')

        ar_augment_x = Sampling.duplicate(train_x, train_y, 1, ratio_duplicate)
        ar_augment_y = np.ones([len(ar_augment_x), ])

        print('Finish duplication')

        # stack train data and shuffle it
        train_x, train_y = Processors.__stack_and_shuffle([train_x, ar_augment_x], [train_y, ar_augment_y])

        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def shuffle_same_dim(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ratio_aug_minority = 2.0
        ratio_aug_majority = 1.0

        print('Start shuffling same dimension and augmentation ... ')

        ar_augment_minor_x = Sampling.shuffle_same_dim(train_x, train_y, 1, ratio_aug_minority)
        ar_augment_minor_y = np.ones([len(ar_augment_minor_x), ])

        stack_list_x = [train_x, ar_augment_minor_x]
        stack_list_y = [train_y, ar_augment_minor_y]

        if ratio_aug_majority > 0:
            ar_augment_major_x = Sampling.shuffle_same_dim(train_x, train_y, 0, ratio_aug_majority)
            ar_augment_major_y = np.ones([len(ar_augment_major_x), ])

            stack_list_x.append(ar_augment_major_x)
            stack_list_y.append(ar_augment_major_y)

        print('Finish shuffling and augmentation')

        # stack train data and shuffle it
        train_x, train_y = Processors.__stack_and_shuffle(stack_list_x, stack_list_y)

        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def duplicate_shuffle_same_dim(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        # the ratio of sampling minority
        ratio_duplicate = 2.0
        ratio_aug_minority = 2.0
        ratio_aug_majority = 0.5

        print('Start duplication ... ')

        ar_duplication_x = Sampling.duplicate(train_x, train_y, 1, ratio_duplicate)
        ar_duplication_y = np.ones([len(ar_duplication_x), ])

        print('Finish duplication\nStart shuffling same dimension and augmentation ... ')

        ar_augment_minor_x = Sampling.shuffle_same_dim(train_x, train_y, 1, ratio_aug_minority)
        ar_augment_minor_y = np.ones([len(ar_augment_minor_x), ])

        stack_list_x = [train_x, ar_duplication_x, ar_augment_minor_x]
        stack_list_y = [train_y, ar_duplication_y, ar_augment_minor_y]

        if ratio_aug_majority > 0:
            ar_augment_major_x = Sampling.shuffle_same_dim(train_x, train_y, 0, ratio_aug_majority)
            ar_augment_major_y = np.ones([len(ar_augment_major_x), ])

            stack_list_x.append(ar_augment_major_x)
            stack_list_y.append(ar_augment_major_y)

        print('Finish shuffling and augmentation')

        # stack train data and shuffle it
        train_x, train_y = Processors.__stack_and_shuffle(stack_list_x, stack_list_y)

        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def smote(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        # the ratio of sampling minority
        ratio_over_sample = 20.0

        print('Start smote ... ')

        # over-sample by smote
        synthetic_train_x = Sampling.smote(train_x, train_y, 1, 5, float(ratio_over_sample / 100.0))
        synthetic_train_y = np.ones([len(synthetic_train_x), ])

        print('Finish smote')

        # stack train data and shuffle it
        train_x, train_y = Processors.__stack_and_shuffle([train_x, synthetic_train_x], [train_y, synthetic_train_y])

        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def under_sample(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' under sample the majority class '''
        ratio_major_by_minor = 0.8
        train_x, train_y = Sampling.under_sample(train_x, train_y, 0, ratio_major_by_minor)
        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def lda(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' LDA reduce the dimensions of the features '''
        _lda = LDA()
        train_x = _lda.fit_transform(train_x, train_y)
        if val_x.any():
            val_x = _lda.transform(val_x)
        test_x = _lda.transform(test_x)
        real_test_x = _lda.transform(real_test_x)
        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def add_lda(train_x, val_x, test_x, train_y, val_y, test_y, real_test_x):
        ''' LDA reduce the dimensions of the features; and add this lda feature to the origin features '''
        _lda = LDA()
        train_lda = _lda.fit_transform(train_x, train_y)
        if val_x.any():
            val_lda = _lda.transform(val_x)
            val_x = np.hstack([val_x, val_lda])
        test_lda = _lda.transform(test_x)
        real_test_lda = _lda.transform(real_test_x)

        train_x = np.hstack([train_x, train_lda])
        test_x = np.hstack([test_x, test_lda])
        real_test_x = np.hstack([real_test_x, real_test_lda])
        return train_x, val_x, test_x, train_y, val_y, test_y, real_test_x

    @staticmethod
    def __stack_and_shuffle(stack_list_x, stack_list_y):
        ''' stack train data and shuffle it '''
        # add the over-sampling data to train data
        train_x = np.vstack(stack_list_x)
        train_y = np.hstack(stack_list_y)

        # shuffle train data
        return Sampling.shuffle(train_x, train_y)
