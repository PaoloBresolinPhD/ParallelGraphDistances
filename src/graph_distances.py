import networkx as nx
import numpy as np

class GraphDistances:
    """
    Class with static methods to compute distance between graphs.
    """

    @staticmethod
    def ancestry_set(graph):
        """
        Computes the set of all ancestor-descendant pairs of node labels in the input graph.
        
        Parameters:
        - graph: NetworkX DiGraph representing a directed graph.

        Returns:
        - AD_pairs: set with ancestor-descendant pairs of node labels in the input graph.
        """

        # AD pairs in the input graph
        AD_pairs = set()

        # iterate over nodes in the graph
        for node in graph.nodes:
            ancestors = nx.ancestors(graph, node)
            for anc in ancestors:
                AD_pairs.add((graph.nodes[anc]['label'], graph.nodes[node]['label']))
        
        return AD_pairs

    @staticmethod
    def compute_distances(ancestry_sets_1, ancestry_sets_2):
        """
        Computes the distances between all pairs of graphs in two sets of ancestry sets.
        The distance between two graphs is the symmetric difference between the corresponding ancestry sets.

        Parameters:
        - ancestry_sets_1: array of sets with ancestor-descendant pairs of node labels for each graph in the first set.
        - ancestry_sets_2: array of sets with ancestor-descendant pairs of node labels for each graph in the second set.

        Returns:
        - distances: numpy array of shape (len(ancestry_sets_1), len(ancestry_sets_2)) with the distances between all pairs of graphs in the two sets.
        """

        # number of graphs in the two sets
        n_graphs_1 = len(ancestry_sets_1)
        n_graphs_2 = len(ancestry_sets_2)

        # initialize the numpy array to store the distances between all pairs of graphs in the two sets
        distances = np.zeros((n_graphs_1, n_graphs_2), dtype=np.int32)

        # iterate over all pairs of graphs to compute distances
        for i in range(n_graphs_1):
            for j in range(n_graphs_2):
                dist = len(ancestry_sets_1[i].symmetric_difference(ancestry_sets_2[j]))
                distances[i, j] = dist

        return distances
