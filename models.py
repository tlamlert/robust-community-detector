import numpy as np
from numpy import linalg
import time

# Default values
DEFAULT_MAX_ITER = 2000
DEFAULT_CONV_CRIT = 1e-3
epsilon = 1e-16

def l2_NMF(X, k, max_iter=DEFAULT_MAX_ITER, conv_crit=DEFAULT_CONV_CRIT, return_losses=False):
    # init
    n, m = X.shape
    F = np.random.rand(n, k)
    G = np.random.rand(k, m)
    losses = []

    start_time = time.time()
    for _ in range(max_iter):
        F = F * (X @ G.T) / (F @ G @ G.T + epsilon)
        G = G * (F.T @ X) / (F.T @ F @ G + epsilon)

        l2_loss = linalg.norm(X - F @ G, 'fro')
        losses.append(l2_loss)

        if len(losses) > 1:
            prev_loss = losses[-2]
            if ((prev_loss - l2_loss) / prev_loss) < conv_crit:
                break
    time_used = time.time() - start_time

    ## Inference
    y_pred = np.argmax(F, axis=1)
    metrics = {
        "num_iter": len(losses),
        "time_used": time_used
    }
    if return_losses:
        metrics["losses"] = losses
    
    return y_pred, metrics

def l21_NMF(X, k, max_iter=DEFAULT_MAX_ITER, conv_crit=DEFAULT_CONV_CRIT, return_losses=False):
    """
    Robust nonnegative matrix factorization using L21-norm
    https://dl.acm.org/doi/pdf/10.1145/2063576.2063676
    Input:
        The adjacency matrix X;
        The number of communities k;
    Output: the set of communities S = {s1, s2, · · · , sk};
    """
    # init
    n, m = X.shape
    F = np.random.rand(n, k)
    G = np.random.rand(k, m)
    losses = []

    start_time = time.time()
    for _ in range(max_iter):
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=0) ** 0.5)
        F = F * (X @ D @ G.T) / (F @ G @ D @ G.T + epsilon)
        D = np.diag(1 / ((X - F @ G) ** 2 + epsilon).sum(axis=0) ** 0.5)
        G = G * (F.T @ X @ D) / (F.T @ F @ G @ D + epsilon)

        l21_loss = sum(linalg.norm(col) for col in (X - F @ G).T)
        losses.append(l21_loss)

        if len(losses) > 1:
            prev_loss = losses[-2]
            if ((prev_loss - l21_loss) / prev_loss) < conv_crit:
                break
    time_used = time.time() - start_time

    ## Inference
    y_pred = np.argmax(F, axis=1)
    metrics = {
        "num_iter": len(losses),
        "time_used": time_used
    }
    if return_losses:
        metrics["losses"] = losses

    return y_pred, metrics