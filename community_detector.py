### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362
### TODO: This algorithm doesn't work on graph with multiple connected components

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import time

## Read input from file
# FILEPATH = "zacharys_karate_club"
# FILEPATH = "dolphins_social_network"
# FILEPATH = "les_miserables"
FILEPATH = "example"
# FILEPATH = "example_broken"
train_i = []
train_j = []
train_val = []

with open(FILEPATH, "r") as file:
    num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
    for line in file:
        i, j = line.split()
        train_i.append(int(i))
        train_j.append(int(j))
        train_val.append(1)

M = sparse.coo_matrix((train_val + train_val, (train_i + train_j, train_j + train_i)), shape=(num_nodes, num_nodes)).tocsr()

"""
Optimization algorithm of EA2NMF
Input:
    The adjacency matrix A;
    The set of nodes V;
    The hyper-parameter λ;
    The number of communities k;
    The numbers of iterations initer, outiter;
Output: the set of communities S = {s1, s2, · · · , sk};
"""

## Initialize hyperparameters
n = num_nodes
k = num_comms
_lambda = 2                 # value used in experiment
initer, outiter = 500, 50   # value used in experiment
initer, outiter = 100, 10   # value used in experiment

## Initialize trainable parameters
np.random.seed(0)
D = np.random.rand(n, k,)
C = np.random.rand(k, n)
X = A = M.toarray()

## Initialize D and C
for _ in range(1):
    D, C = D * (X @ C.T) / (D @ C @ C.T), C * (D.T @ X) / (D.T @ D @ C)

for _ in range(outiter):
    ## Maximize perturbation
    A_bar = D @ C
    P = np.maximum(A - A_bar / (_lambda - 1), -A)
    X = A + P
    for _ in range(initer):
        ## Minimize objective function
        D, C = D * (X @ C.T) / (D @ C @ C.T), C * (D.T @ X) / (D.T @ D @ C)

## Inference
community = []
for i in range(n):
    comm = np.argmax(C[:,i])
    community.append(comm)

for i in range(num_comms):
    print("Community {} ".format(i), np.arange(num_nodes)[np.array(community) == i])