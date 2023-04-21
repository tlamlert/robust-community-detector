### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362
### TODO: This algorithm doesn't work on graph with multiple connected components

import numpy as np
from numpy import linalg
from scipy import sparse
import networkx as nx
from matplotlib import pyplot as plt

## Read input from file
DIRECTORY = "homogenous_dataset/"
# FILEPATH = "zacharys_karate_club"
FILEPATH = "dolphins_social_network"
# FILEPATH = "les_miserables"
train_i = []
train_j = []
train_val = []

init_edges = []
with open(DIRECTORY + FILEPATH, "r") as file:
    num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
    for line in file:
        i, j = line.split()
        train_i.append(int(i))
        train_j.append(int(j))
        init_edges.append((int(i), int(j)))
        train_val.append(1)

def construct_matrix(i, j, val, dropout_rate=0):
    num_edges = len(i)
    mask = np.random.rand(num_edges) > dropout_rate
    val = [val[idx] for idx in range(len(mask)) if mask[idx]]
    i = [i[idx] for idx in range(len(mask)) if mask[idx]]
    j = [j[idx] for idx in range(len(mask)) if mask[idx]]
    edges = [(int(x), int(y)) for x, y in zip(i, j)]
    return edges, sparse.coo_matrix((val + val, (i + j, j + i)), shape=(num_nodes, num_nodes)).tocsr()

edges, M_original = construct_matrix(train_i, train_j, train_val, dropout_rate=0)
edges, M = construct_matrix(train_i, train_j, train_val, dropout_rate=0.10)
edges = init_edges

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
num_iter = 5000
epsilon = 1e-16
X = M.toarray()
X_o = M_original.toarray()

## Initialize trainable parameters
# np.random.seed(42)
F_true = np.random.rand(n, k)
G_true = np.random.rand(k, n)
F_basic = np.random.rand(n, k)
G_basic = np.random.rand(k, n)
F = np.random.rand(n, k)
G = np.random.rand(k, n)

for i in range(num_iter):
    ## Basic NMF on original graph
    F_true = F_true * (X_o @ G_true.T) / (F_true @ G_true @ G_true.T + epsilon)
    G_true = G_true * (F_true.T @ X_o) / (F_true.T @ F_true @ G_true + epsilon)

    ## Basic NMF
    F_basic = F_basic * (X @ G_basic.T) / (F_basic @ G_basic @ G_basic.T + epsilon)
    G_basic = G_basic * (F_basic.T @ X) / (F_basic.T @ F_basic @ G_basic + epsilon)

    ## Robust NMF
    D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
    F = F * (X @ D @ G.T) / (F @ G @ D @ G.T + epsilon)
    D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
    G = G * (F.T @ X @ D) / (F.T @ F @ G @ D + epsilon)

    if (i + 1) % 1000 == 0:
        loss = linalg.norm(X - F @ G) ** 2
        print("training loss after {} iterations: {}".format(i + 1, loss))

## Inference
true_community = [[] for _ in range(num_comms)]
basic_community = [[] for _ in range(num_comms)]
robust_community = [[] for _ in range(num_comms)]
for i in range(n):
    true_community[np.argmax(F_true[i, :])].append(i)
    basic_community[np.argmax(F_basic[i, :])].append(i)
    robust_community[np.argmax(F[i, :])].append(i)

def visualize_community(community, filename):
    ## Visualization
    rng : np.random.Generator = np.random.default_rng(seed=4237502)

    G = nx.Graph(edges)
    springpos = nx.spring_layout(nx.Graph(init_edges), seed=3113794652)
    colorlist = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple", "tab:gray", "tab:brown", "tab:olive", "tab:cyan"]
    pos = {}
    for i in range(num_comms):
        for node in community[i]:
            pos[node] = (100 * np.math.cos(2*i*np.math.pi / num_comms) + 20*rng.random(), 100 * np.math.sin(2*i*np.math.pi / num_comms) + 20*rng.random())
        nx.draw_networkx_nodes(G, springpos, node_size=25, nodelist=community[i], node_color=colorlist[i])

    nx.draw_networkx_edges(G, springpos, width=0.2, alpha=0.2)
    plt.tight_layout()
    plt.axis("off")
    # plt.show()
    plt.savefig(filename)

visualize_community(true_community, "graphics/" + FILEPATH + "_true_community")
visualize_community(basic_community, "graphics/" + FILEPATH + "_basic_community")
visualize_community(robust_community, "graphics/" + FILEPATH + "_robust_community")