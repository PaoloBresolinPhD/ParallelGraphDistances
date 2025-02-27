import networkx as nx
import torch
from tqdm import tqdm

class GraphDistances:
    """
    Class with static methods to compute distance between graphs.
    """

    @staticmethod
    def compute_ancestry_set(graph):
        """
        Computes the set of all ancestor-descendant pairs of node labels in the input graph.
        
        Parameters:
        - graph: NetworkX DiGraph representing a directed graph.

        Returns:
        - AD_pairs: ancestor-descendant pairs of node labels in the input graph.
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
    def ancestor_descendant_dist(graph_1, graph_2):
        """
        Computes the ancestor-descendant distance between the two input graphs.

        Parameters:
        - graph_1: networkx DiGraph.
        - graph_2: networkx DiGraph.
            
        Returns:
        - len(symmetric_diff): AD distance between the two input graphs.
        """

        # ancestry sets of the two graphs, i.e., the sets with all ancestor-descendant pairs in each graph
        A_1 = GraphDistances.compute_ancestry_set(graph_1)
        A_2 = GraphDistances.compute_ancestry_set(graph_2)

        # compute the symmetric difference between A_1 and A_2
        symmetric_diff = A_1.symmetric_difference(A_2)

        # return the number of ancestor-descendant pairs in the symmetric_diff set
        return len(symmetric_diff)

    @staticmethod
    def compute_distances(graphs, graph_dist_fn):
        """
        Computes all distances between graphs in the input dataset, using the input graph distance function.

        Parameters:
        - graphs: list of graphs. Each graph is a list of edges. Each edge is a list with the two
                mutations it contains.
        - graph_dist_fn: graph distance function to be used to compute the distance between two graphs.

        Returns:
        - distances: torch tensor of shape (len(graphs), len(graphs)) with the distances between all pairs of graphs.
        """

        # number of graphs in the input dataset
        n_graphs = len(graphs)

        # initialize the tensor to store the distances between all pairs of graphs
        distances = torch.zeros((n_graphs, n_graphs))

        # iterate over all pairs of graphs to compute distances
        for i in tqdm(range(n_graphs), desc='Computing distances', unit='graphs'):
            for j in range(i, n_graphs):
                dist = graph_dist_fn(graphs[i], graphs[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances
