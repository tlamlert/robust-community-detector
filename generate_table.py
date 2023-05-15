import pandas as pd

DIRECTORY = "experiment_results/"
# SMALL_DATASETS = ["dolphins", "football", "karate", "polbooks", "sp_school_day_1", "sp_school_day_2"]
LARGE_DATASETS = ["cora", "eu-core", "eurosis", "polblogs"]
# ALL_DATASETS = ["cora", "dolphins", "eu-core", "eurosis", "football", "karate", "polblogs", "polbooks", "sp_school_day_1", "sp_school_day_2"]

DROPOUT_RATES = [0.10, 0.20, 0.40]
MODELS = ["L21_NMF", "L2_NMF"]
METRICS = ["cluster_accuracy", "purity", "NMI", "conductance"]

for dataset in ["eu-core"]:
    string = {k: {model: "" for model in MODELS} for k in METRICS}
    for rate in DROPOUT_RATES:
        filename = DIRECTORY + dataset + "_{}".format(int(rate * 100))
        df = pd.read_csv(filename)
        for model in MODELS:
            for i, key in enumerate(METRICS):
                string[key][model] += " {:.4f}".format(df[model][i]) + "&"
    
    for k in string:
        print(("&" + string[k]["L21_NMF"] + string[k]["L2_NMF"])[:-1] + " \\\\")
    print()