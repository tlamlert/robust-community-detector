## Convert gml to txt file
import networkx as nx

## Hyperparameters
FILEPATH = "lesmis.gml"
OUT_PATH = "les_miserables"
NUM_COMM = 2

G = nx.read_gml(FILEPATH, label='id')
name_to_id = dict(zip(G.nodes, range(G.number_of_nodes())))
with open(OUT_PATH, "w") as file:
    file.write("{} {} {}\n".format(G.number_of_nodes(), G.number_of_edges(), NUM_COMM))
    for s, t in G.edges():
        file.write("{} {}\n".format(name_to_id[s], name_to_id[t]))