import numpy as np
from numpy import linalg

# Default values
DEFAULT_NUM_ITER = 2000
DEFAULT_CONV_CRIT = 1e-7
epsilon = 1e-16

def l2_NMF(X, k, num_iter=DEFAULT_NUM_ITER, conv_crit=DEFAULT_CONV_CRIT):
    # init
    n, _ = X.shape
    F = np.random.rand(n, k)
    G = np.random.rand(k, n)
    losses = []

    for _ in range(num_iter):
        F = F * (X @ G.T) / (F @ G @ G.T + epsilon)
        G = G * (F.T @ X) / (F.T @ F @ G + epsilon)

        l2_loss = linalg.norm(X - F @ G, 'fro')
        losses.append(l2_loss)

        if len(losses) > 1:
            prev_loss = losses[-2]
            if ((prev_loss - l2_loss) / prev_loss) < conv_crit:
                break
    
    ## Inference
    y_pred = np.argmax(F, axis=1)
    return y_pred, losses

def l21_NMF(X, k, num_iter=DEFAULT_NUM_ITER, conv_crit=DEFAULT_CONV_CRIT):
    """
    Robust nonnegative matrix factorization using L21-norm
    https://dl.acm.org/doi/pdf/10.1145/2063576.2063676
    Input:
        The adjacency matrix X;
        The number of communities k;
    Output: the set of communities S = {s1, s2, · · · , sk};
    """
    # init
    n, _ = X.shape
    F = np.random.rand(n, k)
    G = np.random.rand(k, n)
    losses = []

    for _ in range(num_iter):
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
        F = F * (X @ D @ G.T) / (F @ G @ D @ G.T + epsilon)
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=1) ** 0.5)
        G = G * (F.T @ X @ D) / (F.T @ F @ G @ D + epsilon)

        l21_loss = sum(linalg.norm(row) for row in (X - F @ G))
        losses.append(l21_loss)

        if len(losses) > 1:
            prev_loss = losses[-2]
            if ((prev_loss - l21_loss) / prev_loss) < conv_crit:
                break
    
    ## Inference
    y_pred = np.argmax(F, axis=1)
    return y_pred, losses