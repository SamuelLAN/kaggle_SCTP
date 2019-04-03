#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def lda(train_x, train_y, val_x, test_x):
    ''' LDA reduce the dimensions of the features '''
    _lda = LDA()
    train_x = _lda.fit_transform(train_x, train_y)
    val_x = _lda.transform(val_x)
    test_x = _lda.transform(test_x)
    return train_x, val_x, test_x


def add_lda(train_x, train_y, val_x, test_x):
    ''' LDA reduce the dimensions of the features; and add this lda feature to the origin features '''
    _lda = LDA()
    train_lda = _lda.fit_transform(train_x, train_y)
    val_lda = _lda.transform(val_x)
    test_lda = _lda.transform(test_x)

    train_x = np.hstack([train_x, train_lda])
    val_x = np.hstack([val_x, val_lda])
    test_x = np.hstack([test_x, test_lda])
    return train_x, val_x, test_x
