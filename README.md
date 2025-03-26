# ParallelGraphDistances
A simple mpi4py implementation to efficiently compute All-to-All distances in large phylogenetic tree datasets.

## Motivation
The computation of distances between these trees can be very complex and time-consuming, hence this simple python script.

## Requirements 
Only an MPI implementation and mpi4py, beyond the needed python packages.
One can also launch the code through a singularity container whose definition is provided.

## Versatility
Replacing the dataset and the distance function can make the algorithm suitable for other kinds of data as well.