from __future__ import absolute_import
import scipy.io as scio
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pairs import pair
import argparse
import time


def normalize(x):
    x = (x - np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0) - np.min(x, axis=0)),
                                                                    (x.shape[0], 1))
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='1', type=int,
                        help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST-30000, 4-BDGP, 5-Animal')
    parser.add_argument('--nn', default='100', type=int,
                        help='Number of nearest neighbors from [1,100]')
    parser.add_argument('--metrix', type=str, default=None)
    args = parser.parse_args()

    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000', 'BDGP', 'Animal']
    dataset = data_name[args.data]

    mat = scio.loadmat('../datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        data[0] = data[0][:9084]
        data[1] = data[1][:9084]
        label = np.squeeze(mat['Y'])[:9084]
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'NoisyMNIST-30000':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])
    elif dataset == 'BDGP':
        data = mat['X']
        label = mat['Y']
    elif dataset == 'Animal':
        data = mat['X']
        label = mat['Y']

    label = label.reshape(-1)
    label = np.array(label, 'float64')
    args.n_clusters = len(np.unique(label))
    if dataset in ['Reuters_dim10', 'NoisyMNIST-30000']:
        X = np.array(data).squeeze()
    else:
        X = data.T.squeeze()

    args.n_views = X.shape[0]
    for i in range(X.shape[0]):
        X[i] = X[i].reshape((X[i].shape[0], -1))
    idx_dict = {}
    err = []

    for i in range(len(X)):
        if args.metrix is None:
            me = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                  'manhattan']
        else:
            me = args.metrix
        idx_all = []
        e_all = []
        for met in me:
            start_time = time.time()
            id, e = pair(X[i], label, args.nn + 1, metrix=met)
            idx_all.append(id)
            e_all.append(e)

        err.append(min(e_all))
        idx_dict[i] = idx_all[e_all.index(min(e_all))]
        print('View-{}: Best metrix-{}: {}-nn ac:{}'.format(i + 1, me[e_all.index(min(e_all))], args.nn,
                                                            (1. - min(e_all))))

    save_all_file = str(data_name[args.data]) + '_disMat.npy'
    np.save(save_all_file, idx_dict)

    print(time.time() - start_time)
