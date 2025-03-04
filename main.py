import sys
import json
import numpy as np
from tumor_dataset import TumorDataset
from utils import Utils
#from graph_distances import GraphDistances
from mpi4py import MPI as MPI
import pickle

if __name__ == '__main__':
    
    # initialize MPI variables
    ROOT_PROCESS = 0
    comm = MPI.COMM_WORLD
    n_processes = comm.Get_size()
    rank = comm.Get_rank()

    # print the number of processes
    print(f"The number of processes is: {n_processes}")

    # root node
    if rank == ROOT_PROCESS:

        # load the input file with configuration parameters
        with open(sys.argv[1], 'r') as file:
            settings = json.load(file)
        
        # load training data and create a list of TumorGraph objects
        dataset = TumorDataset(settings['dataset_path'])
        dataset  = Utils.flatten_list_of_lists(dataset.to_dataset_DiGraphs())
        
        # compute and print the number of graphs in the dataset
        n_graphs = len(dataset)
        print(f"The number of graphs in the dataset is {n_graphs}")

        # initialize the array that will contain the ancestry set of each graph
        ancestry_sets = np.empty(shape = n_graphs, dtype = set)

        # since we have an array of objects, we need to convert it into an array of bytes so that we can scatter it
        # first, serialize each graph
        pickled_graphs = [pickle.dumps(graph) for graph in dataset]

        # convert each graph into an array of bytes 
        byte_graphs = [np.frombuffer(pickled_graph, dtype=np.uint8) for pickled_graph in pickled_graphs]

        # compute an array with the size of each graph in terms of number of bytes
        graph_sizes_in_bytes = np.array([arr.size for arr in byte_graphs], dtype=int)

        # compute the cumulative sum of the sizes of the graphs
        cumulative_sum = np.cumsum(graph_sizes_in_bytes)

        # create an array with the number of bytes to be sent to each process
        sizes = np.empty(n_processes, dtype=int)
        sizes[0] = cumulative_sum[n_graphs // n_processes - 1]
        for i in range(1, n_processes):
            sizes[i] = cumulative_sum[(i + 1) * (n_graphs // n_processes) - 1] - sizes[i-1]
        
        # compute the displacements
        displacements = np.insert((sizes), 0, 0)[:-1] 
        
        # concatenate all bytes into one array
        send_buffer = np.concatenate(byte_graphs)
    
    # all non-root processes
    else:

        # no buffer must be sent
        send_buffer = None

        # initialize the array that will contain the sizes
        sizes = np.empty(n_processes, dtype=int)

        # no displacements must be computed
        displacements = None 
    
    # broadcast sizes to all processes
    comm.Bcast(sizes, root=ROOT_PROCESS)

    # allocate receive buffers
    recv_buffer = np.empty(sizes[rank], dtype = np.uint8)

    # scatter the serialized data 
    comm.Scatterv([send_buffer, sizes, displacements, MPI.BYTE], recv_buffer, root = ROOT_PROCESS)

    # deserialize the received objects
    received_objects = pickle.loads(recv_buffer.tobytes()) # TODO: fix here

    # print the number of received objects
    print(f"Rank {rank} received {len(received_objects)} graphs")


    # compute the tensor with the distances between all pairs of graphs in the training dataset
    #train_distances = GraphDistances.compute_distances( , GraphDistances.ancestor_descendant_dist)
    