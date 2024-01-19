from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.optim import Adam
from idecutils import cluster_acc
from queue import Queue
from losses import InstanceLoss
from models import AE_3views as AE
import os
from data_loader import load_data, data_process_view3
import time
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
manual_seed = 0
os.environ['PYTHONHASHSEED'] = str(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()

parser.add_argument('--n_z', default=32, type=int, help='choose from [32, 64]')
parser.add_argument('--lr_train', default=0.001, type=float, help='choose from [0.0001~0.001]')
parser.add_argument('--lambda1', default=0.01, type=float, help='choose from [0.001~0.01]')
parser.add_argument('--train_epoch', default=500, type=int, help='set according to the learning rate from [500, 1000]')
parser.add_argument('--batch_size', default=1600, type=int, help='choose from [512, 1024, 2048]')  # fix
parser.add_argument('--n_p', default=3, type=int, help='number of positive pairs for each sample')
parser.add_argument('--tol', default=1e-7, type=float)
parser.add_argument('--CL_temperature', default=1, type=float)
# Data
parser.add_argument('--data', default='7', type=int,
                    help='choose dataset from 0-Scene15, 1-Caltech101, 2-Reuters, 4-BDGP, 5-Animal, 6-bbcsport, 7-100Leaves')
parser.add_argument('--aligned_p', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('--main_view', default=2, type=int,
                    help='main view to obtain the final clustering assignments, from[0, 1]')
# Train
parser.add_argument('--t', default=10, type=int)
parser.add_argument('--train_flag', default=1, type=int)


def set_weight(num, all):
    p = (num + 1) ** (-1)
    q = math.log(all) + 0.5772156649
    return p / q


def mse_loss(input, target):
    ret = (target - input) ** 2
    ret = torch.mean(ret)
    return ret


class MFC(nn.Module):

    def __init__(self,
                 n_stacks,
                 n_input,
                 n_clusters,
                 n_z):
        super(MFC, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z)

        for m in self.ae.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def train_model(self, args, x0, x1, x2, g0data, g1data, g2data, graphAli, label):
        if args.train_flag == 1:
            print('Start train...')
            train_ae(self.ae, args, x0, x1, x2, g0data, g1data, g2data, graphAli, label)
            print('trained ae finished')
            args.train_flag = 0
        else:
            self.ae.load_state_dict(torch.load(args.train_path))
            print('load trained ae model from', args.train_path)

    def forward(self, xv0, xv1, xv2):
        _, _, _, zv0, zv1, zv2 = self.ae(xv0, xv1, xv2)
        return zv0, zv1, zv2


def train_ae(model, args, x0, x1, x2, gdata0, gdata1, gdata2, graphA, label):
    criterion_instance = InstanceLoss(args.batch_size, args.CL_temperature, 0).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr_train)
    index_array = np.arange(x0.shape[0])
    np.random.shuffle(index_array)
    loss_q = Queue(maxsize=50)
    g0data0 = gdata0[0]
    g1data0 = gdata0[1]
    g2data0 = gdata0[2]
    g0data1 = gdata1[0]
    g1data1 = gdata1[1]
    g2data1 = gdata1[2]
    g0data2 = gdata2[0]
    g1data2 = gdata2[1]
    g2data2 = gdata2[2]
    train_time = 0
    for epoch in range(args.train_epoch):
        total_loss = 0.
        time0 = time.time()
        for batch_idx in range(np.int_(np.ceil(x0.shape[0] / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, x0.shape[0])]
            x0b = x0[idx].to(args.device)
            x1b = x1[idx].to(args.device)
            x2b = x2[idx].to(args.device)

            g0data0b = g0data0[idx].to(args.device)
            g1data0b = g1data0[idx].to(args.device)
            g2data0b = g2data0[idx].to(args.device)
            g0data1b = g0data1[idx].to(args.device)
            g1data1b = g1data1[idx].to(args.device)
            g2data1b = g2data1[idx].to(args.device)
            g0data2b = g0data2[idx].to(args.device)
            g1data2b = g1data2[idx].to(args.device)
            g2data2b = g2data2[idx].to(args.device)

            g0data0br = g0data0b.reshape(g0data0b.shape[0] * g0data0b.shape[1], g0data0b.shape[2])
            g0data1br = g0data1b.reshape(g0data1b.shape[0] * g0data1b.shape[1], g0data1b.shape[2])
            g0data2br = g0data2b.reshape(g0data2b.shape[0] * g0data2b.shape[1], g0data2b.shape[2])
            g1data0br = g1data0b.reshape(g1data0b.shape[0] * g1data0b.shape[1], g1data0b.shape[2])
            g1data1br = g1data1b.reshape(g1data1b.shape[0] * g1data1b.shape[1], g1data1b.shape[2])
            g1data2br = g1data2b.reshape(g1data2b.shape[0] * g1data2b.shape[1], g1data2b.shape[2])
            g2data0br = g2data0b.reshape(g2data0b.shape[0] * g2data0b.shape[1], g2data0b.shape[2])
            g2data1br = g2data1b.reshape(g2data1b.shape[0] * g2data1b.shape[1], g2data1b.shape[2])
            g2data2br = g2data2b.reshape(g2data2b.shape[0] * g2data2b.shape[1], g2data2b.shape[2])

            optimizer.zero_grad()
            _, _, _, kg0z0, kg0z1, kg0z2 = model(g0data0br, g0data1br, g0data2br)
            _, _, _, kg1z0, kg1z1, kg1z2 = model(g1data0br, g1data1br, g1data2br)
            _, _, _, kg2z0, kg2z1, kg2z2 = model(g2data0br, g2data1br, g2data2br)

            vx0, vx1, vx2, vz0, vz1, vz2 = model(x0b, x1b, x2b)

            # cross-view contrastive loss
            cl_loss_cross = torch.zeros(1).to(args.device)
            for i in range(args.n_p):
                num_i = np.arange(len(idx)) * args.n_p + i
                cl_loss_cross += (criterion_instance(kg0z0[num_i], vz0) + criterion_instance(kg0z1[num_i],
                                                                                             vz0) + criterion_instance(
                    kg0z2[num_i], vz0) + criterion_instance(kg1z0[num_i], vz1) + criterion_instance(kg1z1[num_i],
                                                                                                    vz1) + criterion_instance(
                    kg1z2[num_i], vz1) + criterion_instance(kg2z0[num_i], vz2) + criterion_instance(kg2z1[num_i],
                                                                                                    vz2) + criterion_instance(
                    kg2z2[num_i], vz2)) * set_weight(i, args.n_p) / args.n_p

            rec_loss = mse_loss(x0b, vx0) + mse_loss(x1b, vx1) + mse_loss(x2b, vx2)

            fusion_loss = rec_loss + args.lambda1 * cl_loss_cross
            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()
        epoch_time = time.time() - time0
        train_time += epoch_time
        loss_q.put(total_loss)
        if loss_q.full():
            loss_q.get()
        mean_loss = np.mean(list(loss_q.queue))
        if np.abs(mean_loss - total_loss) <= 0.0001 and epoch >= (args.train_epoch * 0.5):
            print('Training stopped: epoch=%d, loss=%.4f, loss=%.4f' % (
                epoch, total_loss / (batch_idx + 1), mean_loss / (batch_idx + 1)))
            break
        # if (epoch + 1) % args.t == 0:
        #     _, _, _, z0, z1, z2 = model(x0, x1, x2)
        #     mapping = graphA[args.main_view][:, 0]
        #     z0n = z0[mapping]
        #     z1n = z1[mapping]
        #     z = torch.cat((z2, z0n, z1n), axis=1)
        #     kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
        #     y_pred0 = kmeans.fit_predict(z0.data.cpu().numpy())
        #     y_pred1 = kmeans.fit_predict(z1.data.cpu().numpy())
        #     y_pred2 = kmeans.fit_predict(z2.data.cpu().numpy())
        #     y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        #     acc = np.zeros(shape=(4,))
        #     nmi = np.zeros(shape=(4,))
        #     ari = np.zeros(shape=(4,))
        #     f1 = np.zeros(shape=(4,))
        #     acc[0], nmi[0], ari[0], f1[0], _, _ = cluster_acc(label[0], y_pred0)
        #     acc[1], nmi[1], ari[1], f1[1], _, _ = cluster_acc(label[1], y_pred1)
        #     acc[2], nmi[2], ari[2], f1[2], _, _ = cluster_acc(label[2], y_pred2)
        #     acc[3], nmi[3], ari[3], f1[3], _, _ = cluster_acc(label[args.main_view], y_pred)
        print("ae_epoch {} loss={:.4f} mean_loss={:.4f} epoch_time={:.2f}".format(epoch,
                                                                                  total_loss / (batch_idx + 1),
                                                                                  mean_loss / (batch_idx + 1),
                                                                                  round(epoch_time, 2)))
        torch.save(model.state_dict(), args.train_path)
    print("model saved to {}.".format(args.train_path))
    print('******** Training End, training time = {} s ********'.format(round(train_time, 2)))


def main():
    args = parser.parse_args()
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000', 'BDGP', 'Animal', 'bbcsport',
                 '100Leaves']

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.train_path = './save_weight/' + str(data_name[args.data]) + '/train/' + 'CGCN_manualSeed_' + str(
        manual_seed) + '_trainLr_' + str(args.lr_train) + '_lambda1_' + str(args.lambda1) + '_n_z_' + str(
        args.n_z) + '_n_p_' + str(args.n_p) + '_batchSize_' + str(args.batch_size) + '_trainEp_' + str(
        args.train_epoch) + '.pkl'

    ####################################################################
    # Load data, process
    ####################################################################
    data, label, graph = load_data(data_name[args.data])

    dataV0, dataV1, dataV2, gdatall0, gdatall1, gdatall2, graphAlign, indexAlign, _, label = data_process_view3(data,
                                                                                                                label,
                                                                                                                graph,
                                                                                                                manual_seed,
                                                                                                                args)

    del data, graph, data_name

    args.n_clusters = len(np.unique(label[args.main_view]))
    args.n_input = [dataV0.shape[1], dataV1.shape[1], dataV2.shape[1]]

    model = MFC(
        n_stacks=4,
        n_input=args.n_input,
        n_clusters=args.n_clusters,
        n_z=args.n_z).to(args.device)

    model.train_model(args, dataV0, dataV1, dataV2, gdatall0, gdatall1, gdatall2, graphAlign, label)

    del gdatall0, gdatall1, gdatall2
    print('Clustering using trained representation.')
    start_time = time.time()
    z0, z1, z2 = model(dataV0, dataV1, dataV2)
    print(time.time()-start_time)
    mapping = graphAlign[args.main_view][:, 0]
    z0n = z0[mapping]
    z1n = z1[mapping]
    z = torch.cat((z2, z0n, z1n), axis=1)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
    y_pred0 = kmeans.fit_predict(z0.data.cpu().numpy())
    y_pred1 = kmeans.fit_predict(z1.data.cpu().numpy())
    y_pred2 = kmeans.fit_predict(z2.data.cpu().numpy())
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    acc = np.zeros(shape=(4,))
    nmi = np.zeros(shape=(4,))
    ari = np.zeros(shape=(4,))
    f1 = np.zeros(shape=(4,))
    acc[0], nmi[0], ari[0], f1[0], _, _ = cluster_acc(label[0], y_pred0)
    acc[1], nmi[1], ari[1], f1[1], _, _ = cluster_acc(label[1], y_pred1)
    acc[2], nmi[2], ari[2], f1[2], _, _ = cluster_acc(label[2], y_pred2)
    acc[3], nmi[3], ari[3], f1[3], _, _ = cluster_acc(label[args.main_view], y_pred)

    print('Results of the view-0 ACC:{:.4f} NMI:{:.4f} ARI:{:.4f} F1:{:.4f}'.format(acc[0], nmi[0], ari[0], f1[0]))
    print('Results of the view-1 ACC:{:.4f} NMI:{:.4f} ARI:{:.4f} F1:{:.4f}'.format(acc[1], nmi[1], ari[1], f1[1]))
    print('Results of the view-2 ACC:{:.4f} NMI:{:.4f} ARI:{:.4f} F1:{:.4f}'.format(acc[2], nmi[2], ari[2], f1[2]))
    print('Results of matching Z ACC:{:.4f} NMI:{:.4f} ARI:{:.4f} F1:{:.4f}'.format(acc[3], nmi[3], ari[3], f1[3]))


if __name__ == '__main__':
    main()

