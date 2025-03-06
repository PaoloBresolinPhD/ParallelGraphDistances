import sys
import json
import copy
import numpy as np
from tumor_dataset import TumorDataset
from utils import Utils
from graph_distances import GraphDistances
from mpi4py import MPI as MPI

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

        #dataset.sample_one_graph_per_patient(rd_seed=27)                          # USED ONLY FOR DEBUGGING

        dataset  = Utils.flatten_list_of_lists(dataset.to_dataset_DiGraphs())
        
        # compute and print the number of graphs in the dataset
        n_graphs = len(dataset)
        print(f"The number of graphs in the dataset is {n_graphs}")

        # initialize the array that will contain the distances between all pairs of graphs
        distances = np.zeros((n_graphs, n_graphs), dtype=np.int32)

        # initialize the array that will contain an array of ancestry sets for each process
        ancestry_sets = np.empty(n_processes, dtype=object)

        # split the graphs across processes
        n_graphs_per_process = n_graphs // n_processes
        dataset_size_per_process = [n_graphs_per_process for process in range(n_processes)]
        dataset_size_per_process[n_processes - 1] += n_graphs % n_processes                  # assign the remaining graphs to the last process so to preserve initial graph ordering
        curr_pos = 0
        process_graphs = []
        for process in range(n_processes):
            process_graphs.append(dataset[curr_pos:curr_pos + dataset_size_per_process[process]])
            curr_pos += dataset_size_per_process[process]

        # print the number of graphs assigned to each process
        for i in range(n_processes):
            print(f"Process with rank {i} is assigned {len(process_graphs[i])} graphs")

        # initialize the array that will contain the distances between all pairs of graphs in each process
        current_distances_ring = np.empty(n_processes, dtype=object)

    # all non-root processes
    else:

        # initialize the list that will contain the graphs assigned to the current process
        process_graphs = []

    # scatter the graphs across processes
    process_graphs = comm.scatter(process_graphs, root=ROOT_PROCESS)

    # print the number of graphs assigned to the current process
    print(f"Process with rank {rank} received {len(process_graphs)} graphs")

    # compute the ancestry set of each graph assigned to the current process
    process_ancestry_sets = np.empty(len(process_graphs), dtype=set)
    for i, graph in enumerate(process_graphs):
        process_ancestry_sets[i] = GraphDistances.ancestry_set(graph)
    
    # print the number of ancestry sets computed by the current process
    print(f"Process with rank {rank} computed {len(process_ancestry_sets)} ancestry sets")

    # gather the ancestry sets computed by all processes
    ancestry_sets = comm.gather(process_ancestry_sets, root=ROOT_PROCESS)

    # compute the distances between all pairs of graphs in the current process
    current_distances = GraphDistances.compute_distances(process_ancestry_sets, process_ancestry_sets)

    # gather the distances computed by all processes
    current_distances_ring = comm.gather(current_distances, root=ROOT_PROCESS)

    # root process
    if rank == ROOT_PROCESS:

        # print the number of ancestry sets gathered by the root process from each process
        for i in range(n_processes):
            print(f"The root process gathered {len(ancestry_sets[i])} ancestry sets from the process with rank {i}")
        
        # print the shapes of the arrays with distances computed by each process
        for i in range(n_processes):
            print(f"The array with distances computed by the process with rank {i} has shape {current_distances_ring[i].shape}")

        # fill the array that will contain the final distances with the distances computed by each process with the ancestry sets they computed
        origin_i = 0
        origin_j = 0
        for process in range(n_processes):
            for i in range(current_distances_ring[process].shape[0]):
                for j in range(current_distances_ring[process].shape[1]):
                    distances[origin_i + i, origin_j + j] = current_distances_ring[process][i, j]
            origin_i += current_distances_ring[process].shape[0]
            origin_j += current_distances_ring[process].shape[1]


    # compute all remaining distances in parallel by moving the ancestry sets in a ring fashion and pass them to processes
    n_rolls = n_processes // 2
    if n_processes % 2 == 1:
        n_rolls += 1
    
    rolled_ancestry_sets = copy.deepcopy(ancestry_sets)
    for roll in range(1, n_rolls):

        # root process
        if rank == ROOT_PROCESS:

            # move the ancestry sets in a ring fashion
            temp = rolled_ancestry_sets[0]
            for i in range(n_processes):
                rolled_ancestry_sets[i] = rolled_ancestry_sets[(i + 1) % n_processes]
            rolled_ancestry_sets[-1] = temp
        
        # other non-root processes
        else:

            # initialize the list that will contain the ancestry sets received from the root process
            rolled_ancestry_sets = []
            
        # scatter the rolled ancestry sets across processes
        received_ancestry_sets = comm.scatter(rolled_ancestry_sets, root=ROOT_PROCESS)

        # compute the distances between all pairs of graphs in the current process
        current_distances = GraphDistances.compute_distances(process_ancestry_sets, received_ancestry_sets)

        # gather the distances computed by all processes
        current_distances_ring = comm.gather(current_distances, root=ROOT_PROCESS)

        # the root process fills the result matrix with the computations gathered from the other processes
        if rank == ROOT_PROCESS:
            
            # compute the coordinates in the final distance matrix of the point where to start filling it with the gathered distances in the current roll iteration
            origin_r = 0
            origin_c = 0
            for k in range(roll):
                origin_c += current_distances_ring[k].shape[0]
            
            # fill the matrix with the computed distances
            transpose = False
            for process in range(n_processes):
                if origin_c == distances.shape[1]:
                    origin_r = 0
                    origin_c = distances.shape[1]
                    for k in range(roll):
                        origin_c -= current_distances_ring[-1 - k].shape[0]
                    transpose = True
                block_to_insert = copy.deepcopy(current_distances_ring[process])
                if transpose:
                    block_to_insert = np.transpose(block_to_insert)
                print(f"n_rows_block: {block_to_insert.shape[0]}")
                print(f"n_cols_block: {block_to_insert.shape[1]}")
                for r in range(block_to_insert.shape[0]):
                    for c in range(block_to_insert.shape[1]):
                        distances[origin_r + r, origin_c + c] = block_to_insert[r, c]
                origin_r += block_to_insert.shape[0]
                origin_c += block_to_insert.shape[1]

    # if the number of processes is even, then just half or a roll is needed
    if n_processes % 2 == 0:

        # perform half of a roll
        
        # root process
        if rank == ROOT_PROCESS:

            # move the ancestry sets in a ring fashion
            temp = rolled_ancestry_sets[0]
            for i in range(n_processes):
                rolled_ancestry_sets[i] = rolled_ancestry_sets[(i + 1) % n_processes]
            rolled_ancestry_sets[-1] = temp

        # other non-root processes
        else:

            # initialize the list that will contain the ancestry sets received from the root process
            rolled_ancestry_sets = []
            
        # scatter the rolled ancestry sets across processes
        received_ancestry_sets = comm.scatter(rolled_ancestry_sets, root=ROOT_PROCESS)

        # only half of the processes have to compute the distances, the others would do exactly the same computations
        if rank < n_processes // 2:
            current_distances = GraphDistances.compute_distances(process_ancestry_sets, received_ancestry_sets)
        else:
            current_distances = None

        # gather the distances computed by all processes
        current_distances_ring = comm.gather(current_distances, root=ROOT_PROCESS)
        
        # the root process fills the result matrix with the computations gathered from the other processes
        if rank == ROOT_PROCESS:
            
            # compute the coordinates in the final distance matrix of the point where to start filling it with the gathered distances in the current roll iteration
            origin_r = 0
            origin_c = 0
            for k in range(n_rolls):
                origin_c += current_distances_ring[k].shape[0]
            
            # fill the matrix with the computed distances
            for process in range(n_processes):
                print(f"Roll: {n_rolls}, process: {process}, origin_r: {origin_r}, origin_c: {origin_c}")
                if origin_c == distances.shape[1]:
                    break                                                       # we completed half of a roll, so the final matrix is fulfilled
                for r in range(current_distances_ring[process].shape[0]):
                    for c in range(current_distances_ring[process].shape[1]):
                        distances[origin_r + r, origin_c + c] = current_distances_ring[process][r, c]
                origin_r += current_distances_ring[process].shape[0]
                origin_c += current_distances_ring[process].shape[1]

    
    #if rank == ROOT_PROCESS:
    #    np.set_printoptions(threshold=sys.maxsize)
    #    print(distances)
        