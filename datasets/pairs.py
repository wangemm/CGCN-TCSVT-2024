import os
import numpy as np
from random import randint
from sklearn.neighbors import NearestNeighbors
import scipy.io as scio


def pair(data='', label='', number_n = 10, metrix='minkowski'):

    x_train, y_train = data, label

    # KNN group
    n_train = len(x_train)
    knn = n_train
    # print('computing k={} nearest neighbors...'.format(knn))

    x_train_flat = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))[:n_train]
    # print('x_train_flat', x_train_flat.shape)

    train_neighbors = NearestNeighbors(n_neighbors=number_n, metric=metrix).fit(x_train_flat)
    _, idx = train_neighbors.kneighbors(x_train_flat)

    # print('idx shape before', idx.shape)

    new_idx = np.empty((idx.shape[0], idx.shape[1] - 1))
    assert (idx >= 0).all()
    for i in range(idx.shape[0]):
        try:
            new_idx[i] = idx[i, idx[i] != i][:idx.shape[1] - 1]
        except Exception as e:
            print(idx[i, ...], new_idx.shape, idx.shape)
            raise e
    idx = new_idx.astype(np.int)

    counter = 0

    for i, v in enumerate(idx):
        for vv in v:
            if y_train[i] != y_train[vv]:
                counter += 1
    error = counter / (y_train.shape[0] * number_n)
    # print('error rate: {}'.format(error))
    # graph = np.empty(shape=[0, 2], dtype=int)
    # for i, m in enumerate(idx):
    #     for mm in m:
    #         # print(i, mm)
    #         graph = np.append(graph, [[i, mm]], axis=0)
    neighbors = NearestNeighbors(n_neighbors=100, metric=metrix).fit(x_train_flat)
    _, id = neighbors.kneighbors(x_train_flat)
    return id, error

