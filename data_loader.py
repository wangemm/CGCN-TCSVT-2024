import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from idecutils import normalize, aligned_data_split
from sklearn.preprocessing import StandardScaler
import torch
import random
from numpy.random import permutation


def load_data(dataset):
    label = []

    mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        data[0] = data[0][:9083]
        data[1] = data[1][:9083]
        label = np.squeeze(mat['Y'])[:9083]
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
    elif dataset == 'bbcsport':
        data = mat['X']
        label = mat['truth']
    elif dataset == '100Leaves':
        data = mat['data']
        label = mat['truelabel'][0][0]

    label = label.reshape(-1)
    label = np.array(label, 'float64')
    y = label
    if dataset in ['Reuters_dim10', 'NoisyMNIST-30000']:
        x = np.array(data).squeeze()
    elif dataset in ['bbcsport', '100Leaves']:
        x = data.T.squeeze()
        for i in range(x.shape[0]):
            x[i] = x[i].T
    else:
        x = data.T.squeeze()
    n_view = x.shape[0]
    nn_graph = np.load('./datasets/' + dataset + '_disMat.npy', allow_pickle=True).item()
    if n_view == 2:
        g = np.stack((nn_graph[0], nn_graph[1]), axis=0)
    elif n_view == 3:
        g = np.stack((nn_graph[0], nn_graph[1], nn_graph[2]), axis=0)
    return x, y, g


def data_process(data, label, graph, seed, args):
    aligned_sample_index, unaligned_sample_index = aligned_data_split(len(label), args.aligned_p, seed)
    shuffle_index = permutation(unaligned_sample_index)
    auxiliary_view = abs(args.main_view - 1)
    data[auxiliary_view][unaligned_sample_index] = data[auxiliary_view][shuffle_index]

    aligned_graph = np.empty(shape=(graph.shape[0], graph.shape[1], args.n_p), dtype=int)

    for i in range(len(graph)):
        for ii in range(graph[i].shape[0]):
            _, index, _ = np.intersect1d(graph[i][ii], aligned_sample_index, return_indices=True)
            paired = graph[i][ii][np.sort(index)][:args.n_p]
            aligned_graph[i][ii] = paired

    aligned_graph[auxiliary_view][unaligned_sample_index] = aligned_graph[auxiliary_view][shuffle_index]

    y = np.empty(shape=(2, len(label)))
    y[0] = y[1] = label
    y[auxiliary_view][unaligned_sample_index] = y[auxiliary_view][shuffle_index]

    data0 = np.array(data[0], 'float64')
    data1 = np.array(data[1], 'float64')

    data0 = StandardScaler().fit_transform(data0)
    data1 = StandardScaler().fit_transform(data1)

    graphdata0 = np.empty(shape=(graph.shape[0], data0.shape[0], args.n_p, data0.shape[1]))
    graphdata1 = np.empty(shape=(graph.shape[0], data1.shape[0], args.n_p, data1.shape[1]))

    for v in range(len(graph)):
        for n in range(len(data0)):
            for p in range(args.n_p):
                graphdata0[v][n][p] = data0[aligned_graph[v][n][p]]
                graphdata1[v][n][p] = data1[aligned_graph[v][n][p]]

    data0 = torch.Tensor(data0).to(args.device)
    data1 = torch.Tensor(data1).to(args.device)

    graphdata0 = torch.Tensor(graphdata0).to(args.device)
    graphdata1 = torch.Tensor(graphdata1).to(args.device)

    return data0, data1, graphdata0, graphdata1, aligned_graph, aligned_sample_index, unaligned_sample_index, y

def data_process_view3(data, label, graph, seed, args):
    aligned_sample_index, unaligned_sample_index = aligned_data_split(len(label), args.aligned_p, seed)
    shuffle_index = np.empty(shape=(2, unaligned_sample_index.shape[0]), dtype=int)
    for i in range(shuffle_index.shape[0]):
        shuffle_index[i] = permutation(unaligned_sample_index)
    auxiliary_view = np.arange(3)
    auxiliary_view = np.delete(auxiliary_view, args.main_view)
    temp = 0
    for i in auxiliary_view:
        data[i][unaligned_sample_index] = data[i][shuffle_index[temp]]
        temp += 1
    aligned_graph = np.empty(shape=(graph.shape[0], graph.shape[1], args.n_p), dtype=int)

    for i in range(len(graph)):
        for ii in range(graph[i].shape[0]):
            _, index, _ = np.intersect1d(graph[i][ii], aligned_sample_index, return_indices=True)
            paired = graph[i][ii][np.sort(index)][:args.n_p]
            aligned_graph[i][ii] = paired

    temp = 0
    for i in auxiliary_view:
        aligned_graph[i][unaligned_sample_index] = aligned_graph[i][shuffle_index[temp]]
        temp += 1

    y = np.empty(shape=(3, len(label)))
    y[0] = y[1] = y[2] = label

    temp = 0
    for i in auxiliary_view:
        y[i][unaligned_sample_index] = y[i][shuffle_index[temp]]
        temp += 1

    data0 = np.array(data[0], 'float64')
    data1 = np.array(data[1], 'float64')
    data2 = np.array(data[2], 'float64')

    data0 = StandardScaler().fit_transform(data0)
    data1 = StandardScaler().fit_transform(data1)
    data2 = StandardScaler().fit_transform(data2)

    graphdata0 = np.empty(shape=(graph.shape[0], data0.shape[0], args.n_p, data0.shape[1]))
    graphdata1 = np.empty(shape=(graph.shape[0], data1.shape[0], args.n_p, data1.shape[1]))
    graphdata2 = np.empty(shape=(graph.shape[0], data2.shape[0], args.n_p, data2.shape[1]))

    for v in range(len(graph)):
        for n in range(len(data0)):
            for p in range(args.n_p):
                graphdata0[v][n][p] = data0[aligned_graph[v][n][p]]
                graphdata1[v][n][p] = data1[aligned_graph[v][n][p]]
                graphdata2[v][n][p] = data2[aligned_graph[v][n][p]]

    data0 = torch.Tensor(data0).to(args.device)
    data1 = torch.Tensor(data1).to(args.device)
    data2 = torch.Tensor(data2).to(args.device)

    graphdata0 = torch.Tensor(graphdata0).to(args.device)
    graphdata1 = torch.Tensor(graphdata1).to(args.device)
    graphdata2 = torch.Tensor(graphdata2).to(args.device)

    return data0, data1, data2, graphdata0, graphdata1, graphdata2, aligned_graph, aligned_sample_index, unaligned_sample_index, y

