#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PATH_CUR = os.path.abspath(os.path.split(__file__)[0])
PATH_PRJ = os.path.split(PATH_CUR)[0]
sys.path.append(PATH_PRJ)

import load
from eval.pre_process import process
from result_save import gen_csv, log_results, save_for_ensemble
from models.lgb import LGB
from processors import augment, norm, feat
from config.param import K_FOLD
from config.param import RANDOM_STATE


class Integration:
    ''' Integrate the whole process '''

    # the class of the model
    #  model class must provide functions of "train", "predict", "save", "params"
    #  and the __init__ func of the class should provide param "model_name"
    MODEL_CLASS = LGB

    # names of the model and result
    MODEL_DIR = time.strftime('%Y_%m_%d_%H_%M_%S')
    MODEL_NAME = 'lgb'
    RESULT_NAME = MODEL_NAME + '_' + MODEL_DIR

    # list of functions for data augmentation
    AUG_PROCESSORS = [augment.duplicate_shuffle_same_dim]
    # list of functions for data normalization
    NORM_PROCESSORS = [norm.min_max_scaling, norm.standardization]
    # list of functions for changing features
    FEAT_PROCESSORS = []

    # params
    K_FOLD = K_FOLD
    RANDOM_STATE = RANDOM_STATE

    def __init__(self):
        self.__ids, self.__X, self.__y = load.train_data()
        self.__ids_test, self.__X_test = load.test_data()

    def run(self):
        self.__k_fold_eval()

    def __k_fold_eval(self):
        """ eval the k-fold cross-validation """
        # initialize the StratifiedKFold object
        skf = StratifiedKFold(n_splits=self.K_FOLD, shuffle=True, random_state=self.RANDOM_STATE)

        # variable for saving the results
        record_val_auc = []
        record_val_pred = np.zeros((len(self.__y),), np.float32)
        record_test_pred = np.zeros((len(self.__X_test),), np.float32)

        # variables for saving the best fold
        best_val_auc = 0.5
        best_test_pred = np.zeros((len(self.__X_test),), np.float32)

        print('Start %d-fold eval ...' % self.K_FOLD)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(self.__X, self.__y)):
            print('\nStart the %d fold eval ...' % fold)

            # pre-process data
            train_x, train_y, val_x, val_y, test_x = process(self.AUG_PROCESSORS,
                                                             self.NORM_PROCESSORS,
                                                             self.FEAT_PROCESSORS,
                                                             self.__X, self.__y, trn_idx, val_idx,
                                                             self.__X_test)

            # model train and predict
            val_auc, val_pred, test_pred = self.__train_and_predict(fold, train_x, train_y, val_x, val_y, test_x)

            print('Finish the %d fold (auc: %f)' % (fold, val_auc))

            # record results
            record_val_auc.append(val_auc)
            record_val_pred[val_idx] = val_pred
            record_test_pred += test_pred

            # record the best fold
            if best_val_auc < val_auc:
                best_val_auc = val_auc
                best_test_pred = test_pred

        print('\nFinish %d-fold eval' % self.K_FOLD)

        # show results
        log_results(self.MODEL_NAME, self.RESULT_NAME, record_val_auc, record_val_pred, self.__y, self.__model.params)

        # get the mean test prediction
        record_test_pred /= self.K_FOLD

        # gen test result csv
        gen_csv(self.RESULT_NAME, self.__ids_test, record_test_pred)
        gen_csv(self.RESULT_NAME, self.__ids_test, best_test_pred, True)

        # save the results for model fusion
        save_for_ensemble(self.RESULT_NAME, record_val_pred, record_test_pred)

        print('\ndone')

    def __train_and_predict(self, fold_no, train_x, train_y, val_x, val_y, test_x):
        """ train the model and predict """
        model_name = '%s_fold_%d' % (self.MODEL_NAME, fold_no)

        print('\nStart training model %s/%s ...' % (self.MODEL_DIR, model_name))

        self.__model = self.MODEL_CLASS(self.MODEL_DIR, model_name)
        self.__model.train(train_x, train_y, val_x, val_y)

        print('Finish training\n\nStart predicting validation ...')

        # get validation prediction and auc
        val_pred = self.__model.predict(val_x)
        val_auc = roc_auc_score(val_y, val_pred)

        print('Finish predicting validation\n\nStart predicting test ...')

        # get the test prediction
        test_pred = self.__model.predict(test_x)

        print('Finish predicting test')

        self.__model.save()

        return val_auc, val_pred, test_pred


o_integration = Integration()
o_integration.run()
