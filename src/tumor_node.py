class TumorNode:
    """
    Class to store and use a node of a TumorGraph object.
    """

    def __init__(self, node_id, node_label="empty"):
        """
        Constructor.

        Parameters:
        - node_id: integer id that univoquely identifies the node in the graph it belongs to.
                   It must be an integer in [0, n_nodes - 1], where n_nodes is the number of nodes in the graph it belongs to.
                   If the current node is the root, i.e., if it represents the germline subpopulation of cells, then its id must be 0.
        - node_label: string with the label to be assigned to the node.
                      If the node represents the germline subpopulation of cells, then its label must be "root".
                      If the node does not have any label, then its label must be set to "empty".
                      If the node has a label that is not known, then its label must be set to "unknown".
        """

        # node id
        self.id = node_id

        # node label
        self.label = node_label