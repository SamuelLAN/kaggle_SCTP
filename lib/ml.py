#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import Normalizer


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
        means = np.mean(data, axis=axis) if means is None else means
        stds = np.std(data, axis=axis) if stds is None else stds

        # The formula for normalization
        norm_data = (data - means) / (stds + Norm.Epsilon)
        return norm_data, means, stds

    @staticmethod
    def min_max_scaling(data, axis=None,
                        # For the prediction
                        minimum=None, maximum=None):
        # if not for the prediction
        minimum = np.min(data, axis=axis) if minimum is None else minimum
        maximum = np.max(data, axis=axis) if maximum is None else maximum

        # THe formula for normalization
        norm_data = (data - minimum) / (maximum - minimum + Norm.Epsilon)
        return norm_data, minimum, maximum
