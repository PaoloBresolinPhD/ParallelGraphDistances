import sys
import json
import numpy as np
from tumor_graph import TumorGraph
from tumor_dataset import TumorDataset
from utils import Utils
from graph_distances import GraphDistances
import mpi4py
import pickle

if __name__ == '__main__':
    
    ROOT_PROCESS = 0
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()
    n_graphs = 0
    if(rank == ROOT_PROCESS):
        # load the input file with configuration parameters
        with open(sys.argv[1], 'r') as file:
            settings = json.load(file)
        
        # load training data and create a TumorDataset object
        train_data = TumorDataset(settings['dataset_path'])
        
        train_data  = Utils.flatten_list_of_lists(train_data.to_dataset_DiGraphs())
        
        n_graphs = len(train_data)

        ancestry_sets = np.empty(shape = n_graphs, dtype = set)

        # Create a serialized object 

        pickled_graphs = [pickle.dumps(graph) for graph in train_data]

        #Convert each graph into an array of bytes 
        byte_graphs = [np.frombuffer(pickled_graph, dtype = np.uint8 ) for pickled_graph in pickled_graphs]

        # Sizes and Displacements
        sizes = np.array([ arr.size for arr in byte_graphs], dtype= np.int)
        displacements = np.insert(np.cumsum(sizes),0,0)[:-1]

        # COncatenate all bytes into one array
        send_buffer = np.concatenate(byte_graphs)
    else:
        send_buffer = None
        sizes = np.empty(p, dtype = np.int)
        displacements = None 
    
    #Broadcast sizes to all processes
    comm.Bcast(sizes, root = ROOT_PROCESS)

    #Allocate receive buffers
    recv_buffer = np.empty(sizes[rank], dtype = np.uint8)

    #Scatter serialized data 
    comm.Scatterv([send_buffer, sizes, displacements, MPI.BYTE], recv_buffer, root = ROOT_PROCESS)

    #Deserialize Received objs
    received_objects = pickle.loads(recv_buffer.tobytes())

    print(f"Rank {rank} received {len(received_objects)}")


    # compute the tensor with the distances between all pairs of graphs in the training dataset
    #train_distances = GraphDistances.compute_distances( , GraphDistances.ancestor_descendant_dist)
    