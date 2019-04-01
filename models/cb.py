#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import catboost as cb
from config.path import PATH_MODEL_CB, mkdir_time


class CB:
    '''
    Catboost
    '''

    params = {
        'iterations': 999999,
        'max_depth': 2,
        'learning_rate': 0.02,
        'colsample_bylevel': 0.03,
        'objective': 'Logloss'
    }

    def __init__(self, model_dir, model_name=None):
        dir_path = mkdir_time(PATH_MODEL_CB, model_dir)
        self.__model_path = os.path.join(dir_path, model_name + '.cbm' if model_name else 'cb.cbm')
        self.__model = cb.CatBoostClassifier(**self.params)
        self.__has_train = False

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

    def predict(self, X):
        return self.__model.predict_proba(X)[:, 1]

    def save(self):
        if self.__has_train:
            return
        self.__model.save_model(self.__model_path)
        print('Finish saving model to %s' % self.__model_path)
