import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# https://datascience.stackexchange.com/questions/17461/how-to-test-accuracy-of-an-unsupervised-clustering-model-output
def cluster_acc(y_true, y_pred, num_comms):
    w = np.zeros((num_comms, num_comms))
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(y_pred), {k: v for k, v in zip(col_ind, row_ind)}

# https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
# https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def conductance(edges, y_pred):
    num_comms = len(set(y_pred))
    w = np.zeros((num_comms, num_comms))
    for u, v in edges:
        w[y_pred[u], y_pred[v]] += 1
        w[y_pred[v], y_pred[u]] += 1

    conductances = []
    for s in range(num_comms):
        cut_size = np.sum(w[s, :]) - w[s, s]
        s_vol = np.sum(w[s, :])
        s_bar_vol = np.sum(w) - s_vol
        s_cond = cut_size / min(s_vol, s_bar_vol)
        conductances.append(s_cond)
    return min(conductances)

def evaluate_clusters(y_true, y_pred, edges, k):
    cluster_accuracy, mapping = cluster_acc(y_true, y_pred, k)
    metrics = {
        "cluster_accuracy": cluster_accuracy,
        "purity": purity_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "conductance": conductance(edges, y_pred),
    }
    return metrics, mapping