#!/bin/bash
#SBATCH --job-name graph_NO_SPACK
#SBATCH --error %j.err
#SBATCH --output %j.out
#SBATCH --mail-user you@yourdom
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 8
#SBATCH --mem 20G
#SBATCH --time 02:00:00

singularity exec --pwd ${SINGULARITY_ROOTFS}/PGD/ParallelGraphDistances compatible_image.sif mpirun -np 8 python3 -m mpi4py main.py config.json