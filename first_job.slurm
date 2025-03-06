#!/bin/sh
#SBATCH --job-name graph_dist_test_4_cores
#SBATCH --error %j.err
#SBATCH --output %j.out
#SBATCH --mail-user yourname@yourdomain.boh
#SBATCH --mail-type END,FAIL
#SBATCH --partition allgroups
#SBATCH --ntasks 1
#SBATCH --mem 1G
#SBATCH --time 00:02

spack load intel-parallel-studio@professional.2019.4

#srun singularity exec --pwd ${SINGULARITY_ROOTFS}/PGD/ParallelGraphDistances PGD_image.sif mpiexec -np 4 python3 -m mpi4py main.py config.json
