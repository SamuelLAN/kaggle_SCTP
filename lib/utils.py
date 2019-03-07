#!/usr/bin/Python
# -*- coding: utf-8 -*-
import sys
import numpy as np


def k_neighbors(X, k):
    ''' Calculate the K nearest neighbors '''
    results = []  # save results and return it after the func finishes
    len_x = len(X)

    for i in range(len_x):
        # show the progress of knn
        if i % 10 == 0:
            progress = float(i + 1) / len_x * 100.0
            echo('k_neighbors progress: %.2f (%d | %d)   \r' % (progress, i + 1, len_x), False)

        cur = X[i]
        _k_neighbors = []  # save the K nearest neighbor

        for j in range(len_x):
            dist = np.sum(np.power(cur - X[j], 2))

            if len(_k_neighbors) <= k:
                _k_neighbors.append([j, dist])
                _k_neighbors.sort(key=lambda x: x[1])
            else:
                if dist < _k_neighbors[-1][1]:
                    _k_neighbors[-1] = [j, dist]
                    _k_neighbors.sort(key=lambda x: x[1])

        # append the nearest neighbor to results
        results.append(_k_neighbors)

    return results


def echo(msg, crlf=True):
    if crlf:
        print(msg)
    else:
        try:
            sys.stdout.write(msg)
            sys.stdout.flush()
        except:
            print(msg)
