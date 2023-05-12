### Community Detection Algorithm
### Implementation based on "Towards Robust Community Detection via Extreme Adversarial Attacks"
### https://ieeexplore.ieee.org/abstract/document/9956362

import numpy as np
from numpy import linalg
from scipy import sparse
from scipy.sparse.linalg import svds
from tqdm import tqdm
import itertools

## Read input from file
DIRECTORY_PATH = "multiview_data_20130124/politicsie/"
ID_FILEPATH = "politicsie.ids"
LINK_FILEPATH = ["politicsie-follows.mtx", "politicsie-mentions.mtx", "politicsie-retweets.mtx"]
CONTENT_FILEPATH = ["politicsie-lists500.mtx", "politicsie-tweets500.mtx"]
LEBEL_FILEPATH = "politicsie.communities"
OUTPUT_FILEPATH = "politicsie_output"
num_comms = 7

user_ids = []
_index = {}
A = []
X = []

with open(DIRECTORY_PATH + ID_FILEPATH) as file:
    for i, line in enumerate(file):
        user_id = int(line.split()[0])
        user_ids.append(user_id)
        _index[user_id] = i

for path in LINK_FILEPATH:
    train_i = []
    train_j = []
    train_val = []
    with open(DIRECTORY_PATH + path, "r") as file:
        num_users, num_users, _ = [int(val) for val in next(file).split()]
        for line in file:
            i, j, k = line.split()
            train_i.append(_index[int(i)])
            train_j.append(_index[int(j)])
            train_val.append(float(k))
        A.append(sparse.coo_matrix((train_val + train_val, (train_i + train_j, train_j + train_i)), shape=(num_users, num_users)).tocsr().toarray())

for path in CONTENT_FILEPATH:
    train_i = []
    train_j = []
    train_val = []
    with open(DIRECTORY_PATH + path, "r") as file:
        num_features, num_users, _ = [int(val) for val in next(file).split()]
        # print(num_features)
        for line in file:
            i, j, k = line.split()
            train_i.append(_index[int(j)])
            train_j.append(int(i))
            train_val.append(float(k))
        X.append(sparse.coo_matrix((train_val, (train_i, train_j)), shape=(num_users, num_features)).tocsr().toarray())

label_comms = {}
with open(DIRECTORY_PATH + LEBEL_FILEPATH, "r") as file:
    for line in file:
        comm_name, users = line.split(" ")
        label_comms[comm_name] = [int(user) for user in users.split(",")]

"""
RJNMF community detection.
Input:
    Adjacency matrices          A_1, ..., A_p;
    User-content matrices       X_1, ..., X_q;
    The number of communities               k;
    Parameters             a_i, b_i, c_i, d_i;
    The number of iterations   initer, outier; (???)
"""

## Initialize hyperparameters
p, q = len(A), len(X)
n = num_users
k = num_comms
# a = c = [1 for _ in range(p)]
# b = d = [1 for _ in range(q)] # default values when relative importance is unknown
a = [6, 3, 3]
b = [2, 1]
c = [4, 2, 2]
d = [2, 1]                      # value used in experiment
epsilon = 1e-16

## Initialize trainable parameters
# np.random.seed(42)
U = [np.random.rand(n, k) for _ in A]
W = [np.random.rand(n, k) for _ in X]
H = [np.random.rand(matrix.shape[1], k) for matrix in X]
S = np.random.rand(n, k)

## Minimize objective function
## https://www.hindawi.com/journals/mpe/2016/5750645/
for i in tqdm(range(2000)):
    for t in range(p):
        # Update U
        numerator = a[t] * A[t] @ U[t] + c[t] * U[t] @ S.T @ S
        denominator = a[t] * U[t] @ U[t].T @ U[t] + c[t] * U[t] @ U[t].T @ U[t]
        U[t] = U[t] * ((numerator / (denominator + epsilon)) ** 0.5)
    for t in range(q):
        # Update W
        W_numerator = b[t] * X[t] @ H[t] + 2 * d[t] * W[t] @ S.T @ S
        W_denominator = b[t] * W[t] @ H[t].T @ H[t] + 2 * d[t] * W[t] @ W[t].T @ W[t]
        W[t] = W[t] * ((W_numerator / (W_denominator + epsilon)) ** 0.5)
        # Update H
        H_numerator = X[t].T @ W[t]
        H_denominator = H[t] @ W[t].T @ W[t]
        H[t] = H[t] * ((H_numerator / (H_denominator + epsilon)) ** 0.5)
    # Update S
    numerator = sum(c[t] * S @ U[t].T @ U[t] for t in range(p)) + sum(d[t] * S @ W[t].T @ W[t] for t in range(q))
    denominator = (sum(c) + sum(d)) * S @ S.T @ S
    S = S * ((numerator / (denominator + epsilon)) ** 0.5)

    if (i + 1) % 100 == 0:
        loss = (sum(a[t] * linalg.norm(A[t] - U[t] @ U[t].T) ** 2 for t in range(p))
                + sum(b[t] * linalg.norm(X[t] - W[t] @ H[t].T) ** 2 for t in range(q))
                + sum(c[t] * linalg.norm(U[t].T @ U[t] - S.T @ S) ** 2 for t in range(p))
                + sum(d[t] * linalg.norm(W[t].T @ W[t] - S.T @ S) ** 2 for t in range(q)))
        print("training loss after {} iterations: {}".format(i + 1, loss))
        print(sum(a[t] * linalg.norm(A[t] - U[t] @ U[t].T) ** 2 for t in range(p)))
        print(sum(b[t] * linalg.norm(X[t] - W[t] @ H[t].T) ** 2 for t in range(q)))

## Inference
community = [[] for _ in range(num_comms)]
for i in range(n):
    comm = np.argmax(S[i, :])
    community[comm].append(user_ids[i])

accuracy = 0
for perm in (itertools.permutations(range(num_comms))):
    num_correct = 0
    for i, comm in enumerate(label_comms):
        num_correct += len(set(community[perm[i]]) & set(label_comms[comm]))
    accuracy = max(accuracy, num_correct / num_users)
print("Model accuracy: {}".format(accuracy))

with open(OUTPUT_FILEPATH, "w") as file:
    for i in range(num_comms):
        community[i].sort()
        file.write("Community {}: {}\n".format(i, ", ".join(str(ele) for ele in community[i])))
