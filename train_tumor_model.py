import sys
import os
import json
import torch
from tumor_dataset import TumorDataset
from torch_dataset import TorchTumorDataset
from tumor_model import TumorGraphGNN
from trainer import Trainer
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

    # convert the dataset into a TorchTumorDataset object
    train_torch_data = TorchTumorDataset(train_data, node_encoding_type=settings['node_encoding_type'])    

    # compute the tensor with the distances between all pairs of graphs in the training dataset
    train_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)
    
    # if provided, load validation data, create a TorchTensorDataset and compute the distances between all pairs of graphs in the validation dataset
    val_torch_data = None
    val_distances = None
    if settings['val_dataset'] is not None:
        val_data = TumorDataset(settings['val_dataset'])
        if settings['min_label_occurrences'] > 0:
            val_data.replace_label_set(train_data.node_labels(), replace_with='empty') # replace labels not present in the frequent set computed on the training set with 'empty'
        val_data.sample_one_graph_per_patient(rd_seed=settings['random_seed'])
        val_torch_data = TorchTumorDataset(val_data, node_encoding_type=settings['node_encoding_type'], known_labels_mapping=train_torch_data.node_labels_mapping) # use the labels mapping computed for the tarining set as mapping of known labels for the val data
        val_distances = GraphDistances.compute_distances(Utils.flatten_list_of_lists(val_data.to_dataset_DiGraphs()), GraphDistances.ancestor_descendant_dist).to(device)

    # create a TumorGraphGNN instance with input size based on the labels in the training set
    model = TumorGraphGNN(
        n_node_labels=len(train_data.node_labels()),
        h_1_dim=settings['h_1'],
        h_2_dim=settings['h_2'],
        output_dim=settings['embedding_dim'],
        dropout_probs=settings['dropout_probs'],
        batch_normalization=settings['batch_normalization'],
        device=device
    )

    # create all intermediate folders in the paths where to save model and plots, if they do not exist
    os.makedirs(os.path.dirname(settings['model_save']), exist_ok=True)
    os.makedirs(os.path.dirname(settings['train_plot_save']), exist_ok=True)

    # create a Trainer instance
    trainer = Trainer(model=model, device=device)

    # train the model on the training set with the input parameters and validate it in the validation set, if provided
    trainer.train(
        train_data=train_torch_data,
        train_graph_distances=train_distances,
        loss_fn=Utils.select_loss(settings['loss_fn']),
        batch_size=settings['batch_size'],
        val_data=val_torch_data,
        val_graph_distances=val_distances,
        optimizer=Utils.select_optimizer(settings['optimizer']),
        plot_save=settings['train_plot_save'],
        verbose=settings['verbose'],
        epochs=settings['epochs'],
        lr=settings['learning_rate'],
        lr_gamma=settings['lr_gamma'],
        lr_milestones=settings['lr_milestones'],
        early_stopping_tolerance=settings['early_stopping_tolerance']
    )

    # save the trained model
    trainer.save_model(settings['model_save'])