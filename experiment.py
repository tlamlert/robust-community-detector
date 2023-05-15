import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import networkx as nx
from matplotlib import pyplot as plt

from models import l2_NMF, l21_NMF
from metrics import evaluate_clusters

def read_input(file_path):
    ## Read edges and metadata from file
    with open(file_path + ".edges", "r") as file:
        num_nodes, num_edges, num_comms = [int(val) for val in next(file).split()]
        edges = []
        for line in file:
            i, j = line.split()
            edges.append((int(i), int(j)))
    
    ## Read ground truth labels from file
    with open(file_path + ".labels", "r") as file:
        true_comm = [None for _ in range(num_nodes)]
        for line in file:
            node_id, label = line.split()
            true_comm[int(node_id)] = int(label)
    
    return num_nodes, num_edges, num_comms, edges, true_comm

def construct_matrix(num_nodes, edges, dropout_rate=0):
    ## Construct a sparse matrix from input edges
    mask = np.random.rand(len(edges)) > dropout_rate
    remaining = [v for v, m in zip(edges, mask) if m]
    i, j = [v[0] for v in remaining], [v[1] for v in remaining]
    M = sparse.coo_matrix(([1] * 2 * len(remaining), (i + j, j + i)), shape=(num_nodes, num_nodes)).tocsr()
    return remaining, M

def visualize_community(edges, label_list, filename):
    ## Preprocess
    to_community = lambda labels: [np.argwhere(labels == comm).flatten() for comm in set(labels)]
    label_list = [to_community(label) for label in label_list]

    ## Visualization
    colorlist = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:pink", "tab:purple", "tab:gray", "tab:brown", "tab:olive", "tab:cyan"]
    rng : np.random.Generator = np.random.default_rng(seed=42)

    def visualize_graph(edges, community):
        G = nx.Graph(edges)
        springpos = nx.spring_layout(nx.Graph(edges), seed=42)
        for i, comm in enumerate(community):
            nx.draw_networkx_nodes(G, springpos, node_size=25, nodelist=comm, node_color=colorlist[i])
        nx.draw_networkx_edges(G, springpos, width=0.2, alpha=0.2)

    num_graphs = len(label_list)
    for i, labels in enumerate(label_list):
        plt.subplot(100 + num_graphs * 10 + i + 1)
        visualize_graph(edges, labels)

    plt.savefig(filename)
    plt.close()

def run_experiment(num_nodes, num_comms, edges, true_comm, dropout_rate, return_losses=False, filename=None):
    ## Construct prunned matrix for experiment
    true_comm = np.asarray(true_comm)
    _, M = construct_matrix(num_nodes, edges, dropout_rate=dropout_rate)
    X = M.toarray()

    ## Run standard and robust NMF on the prunned input graph
    basic_pred, b_metrics = l2_NMF(X, num_comms, return_losses=return_losses)
    robust_pred, r_metrics = l21_NMF(X, num_comms, return_losses=return_losses)

    ## Evaluate predictions
    basic_metrics, basic_mapping = evaluate_clusters(true_comm, basic_pred, edges, num_comms)
    robust_metrics, robust_mapping = evaluate_clusters(true_comm, robust_pred, edges, num_comms)
    basic_metrics.update(b_metrics)
    robust_metrics.update(r_metrics)

    ## Visualization
    if filename:
        ## Align y_pred to y_true
        basic_pred = [basic_mapping[v] for v in basic_pred]
        robust_pred = [robust_mapping[v] for v in robust_pred]
        label_list = [true_comm, basic_pred, robust_pred]
        visualize_community(edges, label_list, filename + "_{}".format(int(dropout_rate * 100)))
    
    return basic_metrics, robust_metrics

def run_experiment_on_dataset(dataset, visualization_on=False, print_to_terminal=True, save_to_file=False):
    # Filepath config
    DATA_DIRECTORY = "datasets/"
    GRAPHICS_DIRECTORY = "graphics/"
    OUTPUT_DIRECTORY = "experiment_results/"

    # Read input for one dataset
    num_nodes, num_edges, num_comms, edges, true_comm = read_input(DATA_DIRECTORY + dataset)

    for rate in DROPOUT_RATES:
        basic_metrics, robust_metrics = [], []
        for _ in tqdm(range(NUM_EXPERIMENT), leave=False):
            b_metrics, r_metrics = run_experiment(num_nodes, num_comms, edges, true_comm, rate)
            basic_metrics.append(b_metrics)
            robust_metrics.append(r_metrics)
        
        # Visualization
        if visualization_on:
            _= run_experiment(num_nodes, num_comms, edges, true_comm, rate, filename=(GRAPHICS_DIRECTORY + dataset))

        # Construct df
        average = lambda dict_list: {key: sum(d[key] for d in dict_list) / len(dict_list) for key in dict_list[0]}
        basic_metrics = average(basic_metrics)
        robust_metrics = average(robust_metrics)
        model_names = ["L21_NMF", "L2_NMF"]
        metrics_pd = pd.DataFrame.from_dict([basic_metrics, robust_metrics]).T.rename(columns=lambda x: model_names[x])

        # Print to terminal
        if print_to_terminal:
            print("\nExperiment result (dropout_rate = {:.2f})".format(rate))
            print(metrics_pd)

        # Save to files
        if save_to_file:
            metrics_pd.to_csv(OUTPUT_DIRECTORY + dataset + "_{}".format(int(rate * 100)))

##### Experiment Configuration #####
DROPOUT_RATES = [0.10, 0.20, 0.40]
NUM_EXPERIMENT = 100

# ## Option 1: run experiment on all datasets
# SMALL_DATASETS = ["dolphins", "football", "karate", "polbooks", "sp_school_day_1", "sp_school_day_2"]
LARGE_DATASETS = ["cora", "eu-core", "eurosis", "polblogs"]
# ALL_DATASETS = ["cora", "dolphins", "eu-core", "eurosis", "football", "karate", "polblogs", "polbooks", "sp_school_day_1", "sp_school_day_2"]
for dataset in ["cora", "eurosis", "polblogs"]:
    print("Running experiment on {}...".format(dataset))
    run_experiment_on_dataset(dataset, visualization_on=False, print_to_terminal=True, save_to_file=True)

# ## Option 2: run experiment on one dataset
# DATASET_NAME = "football"
# VISUALIZATION_ON = False
# run_experiment_on_dataset(DATASET_NAME, VISUALIZATION_ON)