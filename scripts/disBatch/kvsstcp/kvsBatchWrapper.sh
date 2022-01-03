#!/bin/bash
# Should get an allocation that will give us 6 groups of 4 cores each,
# and we will have exclusive access to the nodes used in the
# allocation.

prog=$1
date
echo "Running $prog on $(hostname) (${SLURM_NTASKS})"
./kvsstcp.py -a kvss_addr_${SLURM_JOB_ID}.txt --logfile kvss_${SLURM_JOB_ID}.log --execcmd 'srun -e %j_%t_err.txt -o %j_%t_out.txt '"$*"
date
