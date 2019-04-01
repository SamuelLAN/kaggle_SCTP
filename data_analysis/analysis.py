#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import pandas as pd
from matplotlib import pylab
from matplotlib import pyplot as plt

from old.lib.ml import Cluster, ReduceDim

PATH_CUR = os.path.split(__file__)[0]
PATH_PRJ = os.path.split(PATH_CUR)[0]
PATH_TRAIN_DATA = os.path.join(PATH_PRJ, 'dataset', 'train.csv')


class Analysis:
    '''
    Data Analysis
    '''

    def __init__(self):
        print('Start loading data from %s ...' % PATH_TRAIN_DATA)
        data = np.asarray(pd.read_csv(PATH_TRAIN_DATA).iloc[:, :])

        # generate shuffle indices
        indices = range(len(data))
        random.shuffle(indices)

        # according to the shuffled indices, reorder the data
        new_data = []
        for i in indices:
            new_data.append(data[i, :])
        new_data = np.asarray(new_data)

        self.__targetList = new_data[:, 1]
        self.__idList = new_data[:, 0]
        self.__features = new_data[:, 2:]

        # delete useless data
        del data, new_data
        print('Finish loading data')

    # For original data, Visualize data
    def plot(self, data):
        print('\nStart plotting data ... ')

        pylab.figure(figsize=(8, 8))
        for i in range(len(data)):
            color = '#00CED1' if self.__targetList[i] == 0 else '#DC143C'
            x, y = data[i, :]

            pylab.scatter(x, y, c=color)

        pylab.show()

        print('Finish plotting')

    # For clustering, visualize data
    def plot_for_cluster(self, data, label_list, color_list):
        print('\nStart plotting data ... ')

        pylab.figure(figsize=(8, 8))
        for i in range(len(data)):
            x, y = data[i, :]
            label = label_list[i]
            color = color_list[label]
            string = str(self.__targetList[i])

            pylab.scatter(x, y, c=color)
            pylab.annotate(string, xy=(x, y), xytext=(5, 2),
                           textcoords='offset points', ha='right', va='bottom')

        pylab.show()

        print('Finish plotting')

    # For plotting range of every feature as line chart
    @staticmethod
    def plot_line_chart(data):
        plt.figure(figsize=(6, 3))
        x = np.array(range(len(data)))
        plt.plot(x, data[:, 0], c='red')
        plt.plot(x, data[:, 1], c='blue')
        plt.grid()
        plt.show()

    # For plotting the distribution of every feature
    @staticmethod
    def plot_hist(data, bins=100):
        plt.hist(data, bins=bins)
        plt.show()

    def run(self):
        sample_number = 500
        features = self.__features[:sample_number]
        targets = np.cast[np.int32](self.__targetList[:sample_number])

        print('\nStart T-SNE ...')

        # Use T-SNE to reduce the dimension and then visualize the data.
        data_with_low_dims = ReduceDim.tsne(np.asarray(features))
        self.plot(data_with_low_dims)

        print('Finish T-SNE\nStart K-means (n_clusters=7) ...')

        # Use K-means to cluster. After that, use T-sne and then visualize data again.
        kmeans_label_list = Cluster.kmeans(features, n_clusters=7, n_init=8, max_iter=300)
        self.plot_for_cluster(data_with_low_dims, kmeans_label_list,
                              ['#000000', '#0000FF', '#00FF00', '#FF0000', '#FFFF00', '#00FFFF', '#111111'])

        print('Finish K-means\nStart Spectral clustering (n_clusters=2) ...')

        # Use Spectral Clustering to cluster.
        # After that, use T-sne and then visualize data again.
        sc_label_list = Cluster.spectral_clustering(features, 2)
        self.plot_for_cluster(data_with_low_dims, sc_label_list, ['#00CED1', '#DC143C'])

        print('Finish Spectral clustering\nStart PCA ...')

        # Use PCA to reduce dimension, then again use T-sne and visualize data.
        pca_reduced_data = ReduceDim.pca(features, n_components=2)
        # pca_reduced_data = ReduceDim.tsne(pca_reduced_data)
        self.plot(pca_reduced_data)

        print('Finish PCA\nStart LDA ...')

        # Use LDA to reduce dimension, then again use T-sne and visualize data.
        lda_reduced_data = ReduceDim.lda(features, targets)
        data_after_lda_tsne = ReduceDim.tsne(lda_reduced_data)
        self.plot(data_after_lda_tsne)

        print('Dimension after LDA: %d' % np.asarray(lda_reduced_data).shape[1])
        print('Finish LDA\n')

        # ------------------------- analyse other aspects ---------------------------------

        number_total = len(self.__idList)
        number_target_equal_1 = np.sum(self.__targetList)

        print('total number: %d' % number_total)
        print('target == 1 number: %d' % number_target_equal_1)

        # calculate range of every features
        range_list_of_every_feature = []
        for i in range(200):
            column_i_features = self.__features[:, i]
            range_list_of_every_feature.append([min(column_i_features), max(column_i_features)])
        range_list_of_every_feature = np.asarray(range_list_of_every_feature)

        print('\n------------------ range of every feature ----------------------')
        for i in range(200):
            print(range_list_of_every_feature[i])

        self.plot_line_chart(range_list_of_every_feature)

        print('\nStart random sample some features and plot their histogram ... ')

        feature_indices = range(200)
        random.shuffle(feature_indices)
        for i in range(10):
            self.plot_hist(self.__features[:, feature_indices[i]])

        print('Done')


o_analysis = Analysis()
o_analysis.run()
