import sys
import json
import torch
from tumor_dataset import TumorDataset
from utils import Utils
from graph_distances import GraphDistances

if __name__ == '__main__':
    
    # load the input file with configuration parameters
    with open(sys.argv[1], 'r') as file:
        settings = json.load(file)
    
    # set the device to use for tensor operations
    device = Utils.get_device(settings['device'])
    print(f"Using device: {device}")

    # limit the cores used by torch
    torch.set_num_threads(settings['max_n_cores'])
    torch.set_num_interop_threads(settings['max_n_cores'])

    # load training data and create a TumorDataset object
    train_data = TumorDataset(settings['train_dataset'])

    # compute the set of labels to be considered, based on the number of occurrences in the training set
    if settings['min_label_occurrences'] > 0:
        train_data.remove_infreq_labels(settings['min_label_occurrences'])
    
    # sample one graph per patient
    train_data.sample_one_graph_per_patient(rd_seed=settings['random_seed'])

    # compute the tensor with the distances between all pairs of graphs in the training dataset
    train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)
    