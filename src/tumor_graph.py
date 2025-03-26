import networkx as nx
from tumor_node import TumorNode

class TumorGraph:
    """
    Class to store and use graphs representing cancer clonal evolution.
    """

    def __init__(self, nodes=None, edges=None):
        """
        Constructor.

        Parameters:
        - nodes: list of TumorNode objects. Can be None, in which case an empty TumorGraph will be created.
        - edges: list of edges. An edge is a tuple with the ids of the two TumorNode objects it links.
                 Since edges are directed, order matters.
                 Can be None, in which case, the TumorGraph will not have any edge.
        """

        # nodes
        self.nodes = []
        if nodes is not None:
            self.add_node_list(nodes)

        # edges
        self.edges = []
        if edges is not None:
            self.add_edge_list(edges)

    def n_nodes(self):
        """
        Computes the number of TumorNodes in the TumorGraph.

        Returns:
        - n_nodes: number of TumorNodes in the TumorGraph.
        """

        return len(self.nodes)

    def n_edges(self):
        """
        Computes the number of edges in the TumorGraph.

        Returns:
        - n_edges: number of edges in the TumorGraph.
        """

        return len(self.edges)

    def remove_node(self, id):
        """
        Removes the TumorNode with the input id from the TumorGraph and all edges in which it appears.

        Parameters:
        - id: id of the TumorNode to be removed from the graph.

        Returns:
        - 0 if the TumorNode was successfully removed, -1 if no TumorNode with the input id is in the TumorGraph.
        """

        # if there is no TumorNode with the input id, then return -1
        if id not in self.get_node_ids():
            return -1

        # initialize the new lists of TumorNodes and edges
        new_nodes = []
        new_edges = []

        # add all TumorNodes with id different from the input one to new_nodes
        for node in self.nodes:
            if node.id != id:
                new_nodes.append(node)
        
        # add all edges in which the input id does not appear to new_edges
        for edge in self.edges:
            if edge[0] != id and edge[1] != id:
                new_edges.append(edge)
        
        # update the list of TumorNodes and edges of the TumorGraph
        self.nodes = new_nodes
        self.edges = new_edges

        return 0

    def remove_edge(self, edge):
        """
        Removes the input edge from the TumorGraph.
        If the input edge is not in the TumorGraph, then nothing happens.

        Parameters:
        - edge: tuple with the ids of the TumorNodes it links. Edge to be removed from the TumorGraph.

        Returns:
        - 0 if the edge was successfully removed, -1 if the edge not present in the TumorGraph.
        """

        # if the edge not present in the TumorGraph, then return -1
        if edge not in self.edges:
            return -1

        # otherwise, remove it from the TumorGraph
        self.edges.remove(edge)

        return 0

    def add_node(self, node):
        """
        Adds the input TumorNode to the TumorGraph.
        If a TumorNode with the same id already exists in the TumorGraph, then it is replaced by the input one.

        Parameters:
        - node: TumorNode to be added to the TumorGraph.
        """

        # if a TumorNode with the same id of the one to be added is already in the TumorGraph, then replace it by the input one
        for i, curr_node in enumerate(self.nodes):
            if curr_node.id == node.id:
                self.nodes[i] = node
                return
        
        # otherwise, add the TumorNode to the TumorGraph
        self.nodes.append(node)

    def add_edge(self, edge):
        """
        Adds the input edge to the TumorGraph.
        If the edge contains ids of TumorNodes not in the TumorGraph, then an empty TumorNode is created and added to the TumorGraph for each id not present.
        If the edge is already present in the TumorGraph, then nothing happens.

        Parameters:
        - edge: tuple with the ids of the two TumorNodes it links. Order matters since edges are directed.
        """

        # if the edge is already present in the TumorGraph, then simply return
        if edge in self.edges:
            return

        # create a TumorNode for each id in the edge not present in the TumorGraph
        ids = self.get_node_ids()
        for node_id in edge:
            if node_id not in ids:
                self.nodes.append(TumorNode(node_id))
        
        # add the edge to the TumorGraph
        self.edges.append(edge)

    def add_node_list(self, nodes):
        """
        Adds the input list of TumorNodes to the TumorGraph object.
        TumorNodes already present in the TumorGraph will be replaced.

        Parameters:
        - nodes: list of TumorNode objects to be added.
        """

        # add to the TumorGraph each TumorNode in nodes
        for node in nodes:
            self.add_node(node)
    
    def add_edge_list(self, edges):
        """
        Adds the input list of edges to the TumorGraph.
        Edges already present will not be added.

        Parameters:
        - edges: list of edges to be added to the TumorGraph. An edge is a tuple with the ids of the two TumorNodes it connects.
        """

        # add each edge in edges to the TumorGraph
        for edge in edges:
            self.add_edge(edge)
    
    def get_node_ids(self, sort=False):
        """
        Returns all the ids of the TumorNode objects in the TumorGraph.

        Parameters:
        sort: boolean indicating whether the ids must be returned in ascending order or not.

        Returns:
        - ids: list with the ids of the nodes in the TumorGraph.
        """

        # initialize the list of ids
        ids = []

        # fill the list with all the ids of the nodes in the TumorGraph
        for node in self.nodes:
            ids.append(node.id)
        
        # return the ids sorted, if required
        if sort:
            return sorted(ids)
        return ids
    
    def get_unique_labels(self):
        """
        Returns the set of labels appearing in the TumorNodes of the TumorGraph.
        Each label is reported once.

        Returns:
        labels: set with all node labels appearing in the TumorGraph.
        """

        # initialize the set of labels appearing in the TumorNodes of the TumorGraph
        labels = set()

        # iterate through the TumorNodes in the graph so to add new labels in the set
        for node in self.nodes:
            labels.add(node.label)
        
        return labels
    
    def to_DiGraph(self):
        """
        Returns the TumorGraph as a networkx DiGraph object.

        Returns:
        - graph_nx: networkx DiGraph object representing the TumorGraph.
        """

        # create a directed graph
        graph_nx = nx.DiGraph()

        # add nodes to the graph
        for node in self.nodes:
            graph_nx.add_node(node.id, label=node.label)
        
        # add edges to the graph
        graph_nx.add_edges_from(self.edges)

        return graph_nx