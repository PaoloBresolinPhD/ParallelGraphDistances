import sys
import json
import copy
import numpy as np
from tumor_dataset import TumorDataset
from utils import Utils
from graph_distances import GraphDistances

if __name__ == '__main__':

    # load the input file with configuration parameters
    with open(sys.argv[1], 'r') as file:
        settings = json.load(file)
    
    # load training data and create a list of TumorGraph objects
    dataset = TumorDataset(settings['dataset_path'])

    # dataset.sample_one_graph_per_patient(rd_seed=27)                          # USED ONLY FOR DEBUGGING

    dataset  = Utils.flatten_list_of_lists(dataset.to_dataset_DiGraphs())
    
    # compute and print the number of graphs in the dataset
    n_graphs = len(dataset)
    print(f"The number of graphs in the dataset is {n_graphs}")

    # initialize the array that will contain the distances between all pairs of graphs
    distances = np.zeros((n_graphs, n_graphs), dtype=np.int32)

    # compute the ancestry set of each graph
    ancestry_sets = np.empty((n_graphs), dtype=object)
    for i, graph in enumerate(dataset):
        ancestry_sets[i] = GraphDistances.ancestry_set(graph)

    # compute the distances between all pairs of graphs in the dataset
    current_distances = GraphDistances.compute_distances(ancestry_sets, ancestry_sets)

    # np.set_printoptions(threshold=sys.maxsize)
    print(current_distances)
        