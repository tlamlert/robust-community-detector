## Convert gml to txt file
import networkx as nx

def preprocess(FILENAME):
    IN_DIRECTORY = "gml_graphs/"
    OUT_DIRECTORY = "datasets/"
    IN_PATH = IN_DIRECTORY + FILENAME + ".gml"
    OUT_PATH = OUT_DIRECTORY + FILENAME

    G = nx.read_gml(IN_PATH, label='id')
    gts = set(G.nodes[i]["gt"] for i in G.nodes)
    NUM_COMM = len(gts)
    name_to_comm = dict(zip(gts, range(NUM_COMM)))
    name_to_id = dict(zip(G.nodes, range(G.number_of_nodes())))

    with open(OUT_PATH + ".labels", "w") as file:
        for i in G.nodes:
            file.write("{} {}\n".format(name_to_id[i], name_to_comm[G.nodes[i]["gt"]]))

    with open(OUT_PATH + ".edges", "w") as file:
        file.write("{} {} {}\n".format(G.number_of_nodes(), G.number_of_edges(), NUM_COMM))
        for s, t in G.edges():
            file.write("{} {}\n".format(name_to_id[s], name_to_id[t]))

if __name__ == "__main__":
    # "citeseer", 
    FILENAMES = ["cora", "dolphins", "eu-core", "eurosis", "football", "karate", 
                 "polblogs", "polbooks", "sp_school_day_1", "sp_school_day_2"]
    for file in FILENAMES:
        preprocess(file)