#from model.utils import load_dataset, train_test_split, load_npz  #, score_link_prediction, score_node_classification
import numpy as np
from scipy import linalg, sparse
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.utils import get_laplacian, degree, remove_self_loops, to_networkx
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
#from logger import Logger
import networkx as nx
import math
import argparse
import os.path as osp
import math
import pandas as pd
import time
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
#%%
# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)

def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c
#### both multiscales simple lambda is used for the noised version of Framelet#####
def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    #assert mode in ('soft', 'hard'), 'shrinkage type is invalid'
    if mode == None:
        return x
    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x

def multiScales(x, r, Lev, num_nodes):
    """
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range((r-1) * Lev + 1):
        if block_idx == 0:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev + 1, 'something wrong in multiScales'
    return specEnergy

def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr

# function for pre-processing
# This is the original Dong Bin or Xuebin Zheng's version.  I think it does not match the paper as it is in reverse way for level
# Please refer to corrected version get_operator2
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d

# function for pre-processing
def get_operator2(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(Lev):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J - l) / a) * L) @ T0F - T0F
            d[j, l] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J - l)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l] += c[j][k] * TkF
        FD1 = d[0, l]

    return d

# function for pre-processing For no Chebyshev approximation: Slow
def get_operator1(L, DFilters, lambdas, eigenvecs, s, Lev):
    num_nodes = L.shape[0]
    lambdas[lambdas <= 0.0] = 0.0
    lambdas[lambdas > 2.0] = 2.0
    r = len(DFilters)   # DFiliters is a list of g functions
    J =  np.log(lambdas[-1] / np.pi) / np.log(s)  # dilation level to start the decomposition
    #a = np.pi / 2  # consider the domain of masks as [0, pi]
    d = dict()
    FD1 = 1.0
    for l in range(Lev):
        for j in range(r):
            d[j, l] = FD1 * DFilters[j](s**(- J - l) * lambdas)
        FD1 = d[0, l]

    d_list = list()
    for i in range(r):
        for l in range(Lev):
            print('Calculating Matrix...{0:2d}, {1:2d}'.format(i, l))
            d_list.append(np.matmul(eigenvecs, np.diag(d[i, l]) @ eigenvecs.T))

    print(FD1)
    return d_list

class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # perform wavelet shrinkage (optional)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = torch.mul(torch.sign(x), (((torch.abs(x) - self.threshold) + torch.abs(torch.abs(x) - self.threshold)) / 2))
            elif self.shrinkage == 'hard':
                x = torch.mul(x, (torch.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
        x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])

        if self.bias is not None:
            x += self.bias
        return x


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, activation = F.relu, dropout_prob=0.7):
        super(Net, self).__init__()
        self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.drop1 = nn.Dropout(dropout_prob)
        self.act = activation

    def forward(self, data, d_list):
        x = data.x  # x has shape [num_nodes, num_input_features]

        x = self.act(self.GConv1(x, d_list))
        x = self.drop1(x)
        x = self.GConv2(x, d_list)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Pubmed',
                        help='name of dataset (default: Cora): Cora, Citeseer, Pubmed')
    parser.add_argument('--reps', type=int, default=12,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.015,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=0.005, #0.015 best for cora;0.05 best for cit,0.005 for pub
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=64, #
                        help='number of hidden units (default: 16)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--scale', type=float, default=3, #1.5 best for pub
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Linear): Haar, Linear, Quadratic, Sigmoid, Entropy')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout probability (default: 0.7)')
    parser.add_argument('--activation', type=str, default= 'relu',
                        help='activation function (default: relu): None, elu, sigmoid, relu, tanh')
    parser.add_argument('--shrinkage', type=str, default='hard',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='threshold value (default: 1e-4)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='alpha value in Framelet function (default: 0.5 for Entropy; 20.0 for Sigmoid)')   #newly added parameter
    parser.add_argument('--CurvatureType', type=str, default='Ollivier',
                        help='Ricci curvature type: Ollivier (default) or Forman or None')
    parser.add_argument('--Ollivier_alpha', type = float, default = 0.7,
                        help='Ollivier_alpha value in Ollivier-Ricci Curvature (default: 0.7)')
    parser.add_argument('--noiseLev', type=float, default=0.0,
                        help='Added noise level (default: 10.0)')           #newly added parameter
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1000)')
    parser.add_argument('--ExpNum', type=int, default='1',
                        help='The Experiment Number (default: 1)')
    parser.add_argument('--Chebyshev', default=True, action='store_false',
                        help='Whether to use Chebyshev approximation (default: True)')
    parser.add_argument('--FrequencyNum', type=int, default=100,
                        help='The number of (noise) high frequency components (default: 100)')

    args = parser.parse_args()
    print(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
#%%

    start_time = time.time()
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), 'data', dataname)
    dataset = Planetoid(root=rootname, name=dataname)
    
    #dataset = PygNodePropPredDataset(name = "ogbn-arxiv")  
    data = dataset[0]
#%%
    G = to_networkx(data, to_undirected=True)
    
    # get the graph Laplacian L and save it as scipy sparse tensor
    # num_nodes = data.adj_t.storage.sparse_sizes()[0]
    # load dataset and prepare noise through high frequence of Laplacian
    

    num_nodes = dataset[0].x.shape[0]
    nfeatures = dataset[0].x.shape[1]

    # Prepare Laplacian
    if args.CurvatureType == None:
        L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
    else:
        G = to_networkx(data, to_undirected=True)
        if args.CurvatureType == 'Ollivier':
            orc = OllivierRicci(G, alpha=args.Ollivier_alpha, verbose="INFO")
        else:
            orc = FormanRicci(G)
        orc.compute_ricci_curvature()
        G_orc = orc.G.copy()
        rc = np.array(list(nx.get_edge_attributes(G_orc, 'ricciCurvature').values()))
        min_rc = abs(min(rc))
        A = np.zeros((num_nodes, num_nodes))
        for n1, n2 in G.edges()
            #A[n1, n2] = 1-(orc.G[n1][n2]["ricciCurvature"]) 
            A[n2, n1] = A[n1, n2]
        edge_index = A.nonzero()
        weight = torch.tensor(A[edge_index])
        edge_index = torch.tensor(np.array(edge_index))
        L = get_laplacian(edge_index, edge_weight = weight, num_nodes=num_nodes, normalization='sym')

    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]

    data = data.to(device)
    data.x = data.x.to(torch.float64)

    # Defining the Framelet. We have mirrored all the framelet function for SVD purpose  by x' = \pi - x
    FrameType = args.FrameType
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Sigmoid':
        alpha = args.alpha    # make sure default value = 20.0
        D1 = lambda x: np.sqrt(1.0 - 1.0 / (1.0+np.exp(-alpha*(x/np.pi-0.5))))
        D2 = lambda x: np.sqrt(1.0 / (1.0+np.exp(-alpha*(x/np.pi-0.5))))
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Entropy':
        alpha = args.alpha   # with a default value = 0.5  (can be made a tunable parameter)
        D1 = lambda x: np.sqrt((1 - alpha*4*(x/np.pi) + alpha*4*(x/np.pi)*(x/np.pi))*((x/np.pi)<=0.5))
        D2 = lambda x: np.sqrt(alpha*4*(x/np.pi) - alpha*4*(x/np.pi)*(x/np.pi))
        D3 = lambda x: np.sqrt((1 - alpha*4*(x/np.pi) + alpha*4*(x/np.pi)*(x/np.pi))*(x/np.pi>0.5))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    # Preparing SVD-Framelet Matrix
    Lev = args.Lev  # level of transform
    scale = args.scale  # dilation scale
    n = args.n
    r = len(DFilters)

    # get matrix operators
    if args.Chebyshev:
        if (FrameType == 'Entropy' or FrameType == 'Sigmoid'):
            J = np.log(lambda_max / np.pi) / np.log(scale)
            d = get_operator2(L, DFilters, n, scale, J, Lev)
        else:
            J = np.log(lambda_max / np.pi) / np.log(scale) + Lev - 1
            d = get_operator(L, DFilters, n, scale, J, Lev)
        # enhance sparseness of the matrix operators (optional)
        # d[np.abs(d) < 0.001] = 0.0
        d_list = list()
        for i in range(r):
            for l in range(Lev):
                d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
    else:
        lambdas, eigenvecs = np.linalg.eigh(L.todense())
        # lambda_max = lambdas[-1]
        d_list = get_operator1(L, DFilters, lambdas, eigenvecs, scale, Lev)
        d_list = [torch.tensor(x).to(device) for x in d_list]
        
        
  #%%      
        
        
        
        
        
        
'''
    Training Scheme
    '''
for i in range(2):

    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    if args.activation == 'None':
        activation = None
    else:  
        activation = eval('F.'+ args.activation)   # make the string into a function
    dropout_prb = args.dropout

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))


    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    SaveResultFilename = args.dataset + 'Exp{0:03d}'.format(args.ExpNum)
    ResultCSV = args.dataset + 'exp_final.csv'

    # Data to torch
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        record_test_acc = 0.0

        # initialize the model: setting cutoff to True makes the first layer as hard high-frequency cut-off
        model = Net(dataset.num_node_features, nhid, dataset.num_classes, r, Lev, num_nodes,
                    shrinkage=args.shrinkage, threshold=args.threshold, activation = activation, dropout_prob=args.dropout).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, d_list)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data, d_list)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            if (epoch + 1) % 2 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model   We dont need this on HPC
            if epoch > 10:
               if epoch_acc['val_mask'][rep, epoch] > max_acc:
                   #torch.save(model.state_dict(), SaveResultFilename + '.pth')
                   # print('Epoch: {:3d}'.format(epoch + 1),
                   #      'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   #      'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   #      'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   #      'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   #      'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   #      'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))
                   print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                   max_acc = epoch_acc['val_mask'][rep, epoch]
                   record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    if osp.isfile(ResultCSV):
        df = pd.read_csv(ResultCSV)
    else:
        outputs_names = {name: type(value).__name__ for (name, value) in args._get_kwargs()}
        outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1,num_reps+1)})
        outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    new_row = {name: value for (name, value) in args._get_kwargs()}
    new_row.update({'Replicate{0:2d}'.format(ii): saved_model_test_acc[ii-1] for ii in range(1,num_reps+1)})
    new_row.update({'Ave_Test_Acc': np.mean(saved_model_test_acc), 'Test_Acc_std': np.std(saved_model_test_acc)})
    df = df.append(new_row, ignore_index=True)
    df.to_csv(ResultCSV, index=False)

    np.savez(SaveResultFilename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)

    print("--- %s seconds ---" % (time.time() - start_time))
