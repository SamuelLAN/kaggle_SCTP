#!/usr/bin/Python
# -*- coding: utf-8 -*-
import random
import numpy as np
from numpy import linalg as LA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import echo, k_neighbors


class Cluster:
    def __init__(self):
        pass

    @staticmethod
    def kmeans(X, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, n_jobs=4):
        model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, tol=1e-4, n_jobs=4)
        model.fit(X)
        return model.labels_

    @staticmethod
    def __similarity(points):
        ''' use rbf_kernel to calculate the similarity '''
        res = rbf_kernel(points)
        for i in range(len(res)):
            res[i, i] = 0
        return res

    @staticmethod
    def spectral_clustering(points, k):
        """
        Spectral clustering
        :param points: 样本点
        :param k: 聚类个数
        :return: 聚类结果
        """
        W = Cluster.__similarity(points)
        # 度矩阵D可以从相似度矩阵W得到，这里计算的是D^(-1/2)
        # D = np.diag(np.sum(W, axis=1))
        # Dn = np.sqrt(LA.inv(D))
        Dn = np.diag(np.power(np.sum(W, axis=1), -0.5))
        # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn
        L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
        eigvals, eigvecs = LA.eig(L)
        # 前k小的特征值对应的索引，argsort函数
        indices = np.argsort(eigvals)[:k]
        # 取出前k小的特征值对应的特征向量，并进行正则化
        k_smallest_eigenvectors = normalize(eigvecs[:, indices])
        # 利用KMeans进行聚类
        return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)


class ReduceDim:
    def __init__(self):
        pass

    @staticmethod
    def tsne(data):
        tsne = TSNE()
        return tsne.fit_transform(data)

    @staticmethod
    def pca(data, n_components=None):
        _pca = PCA(n_components=n_components)
        return _pca.fit_transform(data)

    @staticmethod
    def lda(data, labels):
        _lda = LDA()
        return _lda.fit_transform(data, labels)


class Norm:
    # in case for the zero divisor
    Epsilon = 0.0001

    def __init__(self):
        pass

    @staticmethod
    def standardization(data, axis=None,
                        # For the prediction
                        means=None, stds=None):
        # if not for the prediction
        if means is None:
            means = np.mean(data, axis=axis)
            stds = np.std(data, axis=axis)

        # The formula for normalization
        norm_data = (data - means) / (stds + Norm.Epsilon)
        del data
        return norm_data, means, stds

    @staticmethod
    def min_max_scaling(data, axis=None,
                        # For the prediction
                        minimums=None, maximums=None):
        # if not for the prediction
        if minimums is None:
            minimums = np.min(data, axis=axis)
            maximums = np.max(data, axis=axis)

        # THe formula for normalization
        norm_data = (data - minimums) / (maximums - minimums + Norm.Epsilon)
        del data
        return norm_data, minimums, maximums


class Sampling:
    def __init__(self):
        pass

    @staticmethod
    def smote(X, y, minority_value, n_neighbors, synthetic_num_per_point):
        ''' Over-sampling data '''
        # store the synthetic minority data
        ar_synthetic = []

        # find out all the data of the minor class
        ar_minority = X[np.argwhere(y == minority_value)[:, 0]]
        len_minority = len(ar_minority)

        # TODO delete it
        import time
        print(time.localtime())

        # calculate the k nearest neighbors
        _k_neighbors = k_neighbors(ar_minority, n_neighbors)

        # TODO delete it
        print(time.localtime())

        # traverse the minority data
        for i in range(len_minority):
            # show the progress
            if i % 10 == 0:
                progress = float(i + 1) / len_minority * 100.0
                echo('Synthetic progress: %.2f   \r' % progress, False)

            # generate synthetic_num_per_point times minority data
            for j in range(synthetic_num_per_point):
                rand_index = random.randint(1, n_neighbors)

                # calculate the difference between this point to the nearest point
                diff = ar_minority[_k_neighbors[i][rand_index][0]] - ar_minority[i]
                gap = random.random()

                # add the new synthetic data to ar_synthetic
                ar_synthetic.append(ar_minority[i] + gap * diff)

        return ar_synthetic

    @staticmethod
    def under_sample(X, y, majority_value, ratio_major_by_minor=1.5):
        # calculate the number of majority that need to be sampled
        len_major = np.sum(y == majority_value)
        len_minor = len(y) - len_major
        len_sample_major = int(len_minor * ratio_major_by_minor)

        # store the data after under sampling
        new_x = []
        new_y = []

        # record the number of data which has sampled
        num_has_sample_major = 0
        num_has_sample_minor = 0

        # start sample data
        for i in range(len(y)):
            # if complete sampling, break
            if num_has_sample_minor >= len_minor and num_has_sample_major >= len_sample_major:
                break

            # record the number of data sampled
            if y[i] == majority_value:
                if num_has_sample_major >= len_sample_major:
                    continue
                num_has_sample_major += 1
            else:
                num_has_sample_minor += 1

            # save the sample data
            new_x.append(X[i])
            new_y.append(y[i])

        return Sampling.shuffle(new_x, new_y)

    @staticmethod
    def shuffle(X, y):
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
