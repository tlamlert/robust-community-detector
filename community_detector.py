### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362
### TODO: This algorithm doesn't work on graph with multiple connected components

import numpy as np
from numpy import linalg
from scipy import sparse
from scipy.sparse.linalg import svds

## Read input from file
DIRECTORY = "homogenous_dataset/"
# FILEPATH = "zacharys_karate_club"
FILEPATH = "dolphins_social_network"
# FILEPATH = "les_miserables"
train_i = []
train_j = []
train_val = []

with open(DIRECTORY + FILEPATH, "r") as file:
    num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
    for line in file:
        i, j = line.split()
        train_i.append(int(i))
        train_j.append(int(j))
        train_val.append(1)

def construct_matrix(i, j, val, dropout_rate=0):
    num_edges = len(i)
    mask = np.random.rand(num_edges) > dropout_rate
    val = [val[idx] for idx in range(len(mask)) if mask[idx]]
    i = [i[idx] for idx in range(len(mask)) if mask[idx]]
    j = [j[idx] for idx in range(len(mask)) if mask[idx]]
    return sparse.coo_matrix((val + val, (i + j, j + i)), shape=(num_nodes, num_nodes)).tocsr()

M = construct_matrix(train_i, train_j, train_val, dropout_rate=0)

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
initer, outiter = 500, 50   # value used in experiment'
epsilon = 1e-16

## Initialize trainable parameters
np.random.seed(42)
D = np.random.rand(n, k)
C = np.random.rand(k, n)
X = A = M.toarray()

# Initialize D and C
for _ in range(10):
    D = D * (X @ C.T) / (D @ C @ C.T + epsilon)
    C = C * (D.T @ X) / (D.T @ D @ C + epsilon)

for _ in range(outiter):
    ## Maximize perturbation
    A_bar = D @ C
    P = np.maximum((A - A_bar) / (_lambda - 1), -A)
    X = A + P
    for _ in range(initer):
        ## Minimize objective function
        D = D * (X @ C.T) / (D @ C @ C.T + epsilon)
        C = C * (D.T @ X) / (D.T @ D @ C + epsilon)

print("P norm: ", linalg.norm(P) ** 2)
print("X norm: ", linalg.norm(X) ** 2)
print("Loss: ", linalg.norm(X - D @ C) ** 2 - linalg.norm(P) ** 2)

## Inference
community = []
for i in range(n):
    comm = np.argmax(C[:,i])
    community.append(comm)

for i in range(num_comms):
    print("Community {} ".format(i), np.arange(num_nodes)[np.array(community) == i])