### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362
### TODO: This algorithm doesn't work on graph with multiple connected components

import numpy as np
from numpy import linalg
from scipy import sparse
from scipy.sparse.linalg import svds
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

DIRECTORY = "datasets/"

DATASET_NAME = "sp_school_day_1"

## Read input from file
train_i = []
train_j = []
train_val = []
with open(DIRECTORY + DATASET_NAME + ".edges", "r") as file:
    num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
    for line in file:
        i, j = line.split()
        train_i.append(int(i))
        train_j.append(int(j))
        train_val.append(1)
    keys = set(train_i + train_j)
    coding = {k: v for k, v in zip(keys, range(len(keys)))}
    train_i = [coding[v] for v in train_i]
    train_j = [coding[v] for v in train_j]

# Read ground truth labels if a label file exists
try:
    with open(DIRECTORY + DATASET_NAME + ".labels", "r") as file:
        true_comm = [None for _ in range(num_nodes)]
        for line in file:
            node_id, label = line.split()
            true_comm[int(node_id)] = int(label)
except:
    true_comm = None

def construct_matrix(i, j, val, dropout_rate=0):
    num_edges = len(i)
    mask = np.random.rand(num_edges) > dropout_rate
    val = [v for v, m in zip(val, mask) if m]
    i = [v for v, m in zip(i, mask) if m]
    j = [v for v, m in zip(j, mask) if m]
    edges = [(int(x), int(y)) for x, y in zip(i, j)]
    return edges, sparse.coo_matrix((val + val, (i + j, j + i)), shape=(num_nodes, num_nodes)).tocsr()

def run_experiment(true_comm, dropout_rate):
    ## Construct dropped-out matrix for experiment
    orig_edges, orig_M = construct_matrix(train_i, train_j, train_val, dropout_rate=0)
    edges, M = construct_matrix(train_i, train_j, train_val, dropout_rate=dropout_rate)

    """
    Robust nonnegative matrix factorization using L21-norm
    https://dl.acm.org/doi/pdf/10.1145/2063576.2063676
    Input:
        The adjacency matrix A;
        The set of nodes V;
        The hyper-parameter λ;
        The number of communities k;
    Output: the set of communities S = {s1, s2, · · · , sk};
    """

    ## Initialize hyperparameters
    n = num_nodes
    k = num_comms
    num_iter = 2000
    epsilon = 1e-16
    convergence_cri = 1e-5 ## 1e-7
    X = M.toarray()

    ## Initialize trainable parameters
    F_basic = np.random.rand(n, k)
    G_basic = np.random.rand(k, n)
    F = F_basic.copy()
    G = G_basic.copy()

    basic_losses = []
    robust_losses = []

    if not true_comm:
        ## Standard NMF on original graph
        X_o = orig_M.toarray()
        F_true = np.random.rand(n, k)
        G_true = np.random.rand(k, n)
        
        for i in range(num_iter):
            F_true = F_true * (X_o @ G_true.T) / (F_true @ G_true @ G_true.T + epsilon)
            G_true = G_true * (F_true.T @ X_o) / (F_true.T @ F_true @ G_true + epsilon)
        
        true_comm = []
        for i in range(n):
            true_comm.append(np.argmax(F_true[i, :]))

    # for i in tqdm(range(num_iter)):
    for i in range(num_iter):
        ## Standard NMF
        F_basic = F_basic * (X @ G_basic.T) / (F_basic @ G_basic @ G_basic.T + epsilon)
        G_basic = G_basic * (F_basic.T @ X) / (F_basic.T @ F_basic @ G_basic + epsilon)

        l2_loss = linalg.norm(X - F_basic @ G_basic, 'fro')
        basic_losses.append(l2_loss)
        
        if i > 0:
            prev_loss = basic_losses[-2]
            converge = ((prev_loss - l2_loss) / prev_loss ) < convergence_cri
            if converge:
                break

    # for i in tqdm(range(num_iter)):
    for i in range(num_iter):
        ## Robust NMF
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
        F = F * (X @ D @ G.T) / (F @ G @ D @ G.T + epsilon)
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
        G = G * (F.T @ X @ D) / (F.T @ F @ G @ D + epsilon)

        l21_loss = 0
        for row in X - F @ G:
            l21_loss += linalg.norm(row)
        robust_losses.append(l21_loss)

        if i > 0:
            prev_loss = robust_losses[-2]
            converge = ((prev_loss - l21_loss) / prev_loss) < convergence_cri
            if converge:
                break

    ## Inference
    basic_comm = []
    robust_comm = []
    for i in range(n):
        basic_comm.append(np.argmax(F_basic[i]))
        robust_comm.append(np.argmax(F[i]))
    
    ## Evaluation: Clutering Accuracy
    def cluster_acc(y_true, y_pred):
        num_comms = len(set(y_true + y_pred))
        w = np.zeros((num_comms, num_comms))
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(y_pred)

    basic_acc = cluster_acc(true_comm, basic_comm)
    robust_acc = cluster_acc(true_comm, robust_comm)

    ## Evaluation: Rand index
    # basic_acc = metrics.rand_score(true_comm, basic_comm)
    # robust_acc = metrics.rand_score(true_comm, robust_comm)

    ## Evaluation: Conductance
    def conductance(edges, y_pred):
        conductances = []
        for s in set(y_pred):
            cut_size, s_vol, s_bar_vol = 0, 0, 0
            for u, v in edges:
                if y_pred[u] == s and y_pred[v] == s:
                    s_vol += 2
                elif y_pred[u] == s or y_pred[v] == s:
                    cut_size += 1
                    s_vol += 1
                    s_bar_vol += 1
                else:
                    s_bar_vol += 2
            s_cond = cut_size / min(s_vol, s_bar_vol)
            conductances.append(s_cond)
        return min(conductances)
    
    basic_cond = conductance(orig_edges, basic_comm)
    robust_cond = conductance(orig_edges, robust_comm)

    return basic_acc, robust_acc, basic_cond, robust_cond, basic_losses, robust_losses

##### Experiment Configuration #####
DROPOUT_RATES = [0.05, 0.10, 0.20, 0.40]
NUM_EXPERIMENT = 100
for rate in DROPOUT_RATES:
    basic_acc = []
    robust_acc = []
    basic_cond = []
    robust_cond = []
    basic_iter = []
    robust_iter = []

    for _ in tqdm(range(NUM_EXPERIMENT)):
    # for _ in range(NUM_EXPERIMENT):
        b_acc, r_acc, b_cond, r_cond, b_losses, r_losses= run_experiment(true_comm, rate)
        basic_acc.append(b_acc)
        robust_acc.append(r_acc)
        basic_cond.append(b_cond)
        robust_cond.append(r_cond)
        basic_iter.append(len(b_losses))
        robust_iter.append(len(r_losses))

    print("##### Experiment result (dropout_rate = {:.2f}) #####".format(rate))
    print("Basic NMF accuracy: {:.4f}".format(np.mean(np.asarray(basic_acc))))
    print("Robust NMF accuracy: {:.4f}".format(np.mean(np.asarray(robust_acc))))
    print()

    print("Basic NMF conductance: {:.4f}".format(np.mean(np.asarray(basic_cond))))
    print("Robust NMF conductance: {:.4f}".format(np.mean(np.asarray(robust_cond))))
    print()

    print("Basic NMF #iterations before convergence: {:.4f}".format(np.mean(np.asarray(basic_iter))))
    print("Robust NMF #iterations before convergence: {:.4f}".format(np.mean(np.asarray(robust_iter))))
    print()