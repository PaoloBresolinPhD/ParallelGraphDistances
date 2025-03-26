import os
import random as rd
import networkx as nx
from tumor_node import TumorNode
from tumor_graph import TumorGraph

class TumorDataset:
    """
    Class to manage a dataset of TumorGraph objects organized into patients.
    """

    def __init__(self, data):
        """
        Constructor.

        Parameters:
        - data: input data. Can be either the path to a .txt file with tumor data or a list of patients, where a patient is a list of TumorGraphs.
        """
        
        # initialize the dataset with the input data
        if isinstance(data, (str, os.PathLike)):
            self.dataset = self.read_txt(data)
        else:
            self.dataset = data

    def read_txt(self, dataset_path):
        """
        Loads data from a .txt file with proper format.

        Parameters:
        - dataset_path: path to the .txt file containing data to load.

        Returns:
        - data: list of patients. Each patient is a list of TumorGraph objects
        """

        # initialize the list that will contain the data to load
        data = []

        # open the file in read mode
        with open(dataset_path, "r") as file:

            # the first line contains the number of patients
            n_patients = int(file.readline().split()[0])

            # iterate through patients
            for i in range(n_patients):

                # append a list related to a new patient
                data.append([])

                # the first line related to a patient has the number of graphs it contains
                n_graphs = int(file.readline().split()[0])

                # iterate through the graphs for the current patient
                for j in range(n_graphs):
                    
                    # initialize the current TumorGraph for the current patient
                    data[i].append(TumorGraph())

                    # the first line related to a graph has the number of nodes in the graph
                    n_nodes = int(file.readline().split()[0])

                    # add all nodes, represented as TumorNodes to the current TumorGraph
                    for k in range(n_nodes):
                        node = file.readline().split()
                        data[i][j].add_node(TumorNode(node_id=int(node[0]), node_label=node[1]))
                    
                    # the next line contains the number of edges in the current graph
                    n_edges = int(file.readline().split()[0])

                    # add all edges, represented as tuple with the ids of the TumorNodes it links
                    for l in range(n_edges):
                        edge = file.readline().split()
                        data[i][j].add_edge((int(edge[0]), int(edge[1])))
        
        return data
    
    def n_patients(self):
        """
        Computes the number of patients in the dataset.

        Returns:
        - len(self.dataset): number of patients in the dataset.
        """

        return len(self.dataset)
    
    def n_graphs(self):
        """
        Computes the overall number of graphs in the dataset.

        Returns:
        - n_graphs: overall number of graphs in the dataset.
        """

        # iterate through patients to compute the overall number of trees in the dataset
        n_graphs = 0
        for patient in self.dataset:
            n_graphs += len(patient)
        
        return n_graphs

    def n_graphs_patient(self, i):
        """
        Computes the number of graphs for the patient in position i in the dataset.

        Parameters:
        - i: index of the patient in the dataset.

        Returns:
        - len(self.dataset[i]): number of graphs for patient i in the dataset.
        """

        return len(self.dataset[i])

    def node_labels(self):
        """
        Returns all node labels appearing in self.dataset.
        Node labels "root", "unknown" and "empty" are added to the set even in case they do not appear in the self.dataset.

        Returns:
        - labels: set with all labels appearing in self.dataset plus "root", "unknown" and "empty".
        """

        # initialize the set of labels with "root", "unknown" and "empty"
        labels = {"root", "unknown", "empty"}

        # iterate through all nodes in the dataset and add not already found labels
        for patient in self.dataset:
            for graph in patient:
                labels.update(graph.get_unique_labels())
        
        return labels

    def labels_counts(self):
        """
        Computes the number of patients in which each node label present in self.dataset appears.

        Returns:
        - labels_counts_dic: dictionary with all node labels in self.dataset as keys and the corresponding number of patients in which they occur as values.
        """

        # compute the set of all node labels appearing in the dataset
        labels = self.node_labels()

        # initialize the dictionary
        labels_counts_dic = {}

        # iterate through all labels in the dataset
        for label in labels:

            # initialiaze the number of patients that have the current label
            labels_counts_dic[label] = 0

            # compute the number of patients with the current label
            for patient in self.dataset:
                for graph in patient:
                    if label in graph.get_unique_labels():
                        labels_counts_dic[label] += 1
                        break
    
        return labels_counts_dic

    def node_labels_freq(self, n=2):
        """
        Returns all node labels appearing at least in n patients in self.dataset.
        A lable is considered to appear in a patient if it is present in a node of at least one of the graphs for the considered patient.
        
        Parameters:
        - n: minimum number of patients with at least one node in one graph with a given label for it to be included in the set of labels.

        Returns:
        - labels: set with all labels appearing in at least n patients in self.dataset.
        """

        # compute the number of patients in which each label appears
        labels_counts_dic = self.labels_counts()

        # return only the labels appearing in at least n patients
        return set(key for key in labels_counts_dic.keys() if labels_counts_dic[key] >= n)
    
    def remove_infreq_labels(self, threshold):
        """
        Removes all node labels appearing in less than threshold patients in self.dataset.
        When a label is removed from a node, the node is assigned label "empty", representing a node with no label.
        Also the node label "unknown", regrdless of the number of times it appears in the dataset, is replaced with "empty".

        Parameters:
        - threshold: minimum number of patients in which a node label must appear not to be removed from self.dataset.
        """

        # compute the number of patients in which each label appears
        labels_counts_dic = self.labels_counts()

        # set the count of "unknown" to 0, so not to keep it
        labels_counts_dic["unknown"] = 0

        # replace the label of each TumorNode with infrequent label by "empty"
        for i in range(self.n_patients()):
            for j in range(self.n_graphs_patient(i)):
                for k, node in enumerate(self.dataset[i][j].nodes):
                    if labels_counts_dic[node.label] < threshold:
                        self.dataset[i][j].nodes[k].label = "empty"
        
    def remove_large_graphs(self, max_n_edges=10):
        """
        Removes the graphs with more than max_n_edges edges from self.dataset.
        If a patient becomes empty due to the removal of all its graphs, then it is removed from self.dataset.

        Parameters:
        - max_n_edges: maximum number of edges that a graph can have so to be kept in self.dataset
        """

        # version of the self.dataset that will not have large graphs
        new_dataset = []

        # iterate through all graphs and keep only those smaller than max_n_edges
        for patient in self.dataset:
            new_patient = []
            for graph in patient:
                if graph.n_edges() <= max_n_edges:
                    new_patient.append(graph)
            
            # append new_patient only if it is not empty
            if len(new_patient) > 0:
                new_dataset.append(new_patient)

        # update self.dataset with new_dataset
        self.dataset = new_dataset

    def remove_uncertain_patients(self, max_n_graphs=20):
        """
        Removes uncertain patients from self.dataset.
        A patient is considered uncertain if it has a number of graphs larger than max_n_graphs.
        
        Parameters:
        - max_n_graphs: maximum number of graphs that a patient can have so to be kept in self.dataset.
        """

        # new version of the dataset without uncertain patients
        new_dataset = []

        # iterate through patients and keep only those with less than max_n_graphs graphs
        for i in range(self.n_patients()):
            if self.n_graphs_patient(i) < max_n_graphs:
                new_dataset.append(self.dataset[i])
        
        # update self.dataset with the version with no uncertain patient
        self.dataset = new_dataset

    def sample_one_graph_per_patient(self, rd_seed=None):
        """
        Samples uniformly at random one graph per patient from self.dataset.
        The function updates self.dataset such dataset each patient will be left with just one graph chosen uniformly at random among those previously contained by the patient.
        Notice that each patient will still be a list of graphs, but with only one element, that is, the sampled graph.

        Parameters:
        - rd_seed: random seed for sampling reproducibility.
        """

        # set the random seed, if required
        if rd_seed is not None:
            rd.seed(rd_seed)

        # new version of the dataset with only one graph per patient
        new_dataset = []

        # iterate through patients and sample just one graph uniformly at random
        for patient in self.dataset:
            new_dataset.append([patient[rd.randint(0, len(patient) - 1)]])
        
        # update self.dataset with the new version of the dataset
        self.dataset = new_dataset

    def replace_label_set(self, known_labels, replace_with="unknown"):
        """
        Replaces all node labels in all graphs for all patients in self.dataset that are not in known_labels with the label replace_with.
        Labels "root", "unknown" and "empty" are added to known_labels and not replaced when found.

        Parameters:
        - known_labels: set with known node labels. Node labels in this set will not be replaced with label replace_with.
                        Also "root", "empty" and "unknown" will not br replaced.
        """

        # add to the set of labels to keep also "root", "empty", "unknown" and replace_with, if not already present
        labels_to_keep = known_labels.copy()
        labels_to_keep.update({"root", "empty", "unknown", replace_with})

        # iterate through all nodes in self.dataset and replace the node labels not in knwon_labels with label replace_with
        for i in range(self.n_patients()):
            for j in range(self.n_graphs_patient(i)):
                for k, node in enumerate(self.dataset[i][j].nodes):
                    if node.label not in labels_to_keep:
                        self.dataset[i][j].nodes[k].label = replace_with
    
    def to_dataset_DiGraphs(self):
        """
        Returns a list of patients, where each patient is a list of networkx DiGraph objects.
        Each TumorGraph in the TumorDataset is converted into a networkx DiGraph.

        Returns:
        - dataset_nx: list of patients, where each patient is a list of networkx DiGraph objects.
        """

        # list of patients with networkx DiGraph objects
        dataset_nx = []

        # iterate through all patients and convert each TumorGraph into a networkx DiGraph
        for patient in self.dataset:
            patient_nx = []
            for graph in patient:
                patient_nx.append(graph.to_DiGraph())
            dataset_nx.append(patient_nx)
        
        return dataset_nx
