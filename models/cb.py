#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import catboost as cb
from sklearn.metrics import roc_auc_score

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_MODEL = os.path.join(PATH_CUR, 'saved_models')


class CB:
    '''
    Catboost
    '''

    def __init__(self, model_name=None):
        self.__has_train = False
        self.__model_path = os.path.join(PATH_MODEL, model_name + '.cbm' if model_name else 'lgb.cbm')
        self.__model = cb.CatBoostClassifier(iterations=999999,
                                             max_depth=2,
                                             learning_rate=0.02,
                                             colsample_bylevel=0.03,
                                             objective="Logloss")

        if model_name and os.path.isfile(self.__model_path):
            self.__model.load_model(self.__model_path)
            self.__has_train = True

    def train(self, train_x, train_y, val_x, val_y):
        if self.__has_train:
            return
        self.__model.fit(train_x, train_y,
                         eval_set=[(val_x, val_y)],
                         verbose=True,
                         early_stopping_rounds=1000)

    def test_auc(self, test_x, test_y):
        test_output = self.predict(test_x)
        auc = roc_auc_score(test_y, test_output)
        print('test auc: %f' % auc)
        return auc

    def predict(self, X):
        return self.__model.predict_proba(X)[:, 1]

    def save(self):
        if self.__has_train:
            return
        self.__model.save_model(self.__model_path)
        print('Finish saving model to %s' % self.__model_path)
