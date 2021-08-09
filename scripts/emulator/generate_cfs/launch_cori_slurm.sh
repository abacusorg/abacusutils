#!/usr/bin/env bash

# Launch an array of Slurm jobs on Cori to evaluate the 2PCF on multiple sims.

set -e  # exit on fail

# Set the following parameters based on the problem

time="15"  # mins
cpus_per_task=10  # tune this based on memory
nnode=2
outdir="${SCRATCH}/emulator_cfs/run1"  # usually will want to change this for new runs
invocation="generate_cf.py --ndens 1e-4 --outdir=${outdir} --nthread=${cpus_per_task}"
suite="${CFS}/desi/cosmosim/Abacus"
cats="$(echo ${suite}/AbacusSummit_base_c{000,{100..126}}_ph000/halos/z0.500)"

# Everything below here usually does not need to change

disbatch=$(realpath "../../disBatch/disBatch")
taskfile="${outdir}/log/tasks.disbatch"
logdir=${outdir}/log

mkdir ${outdir}  # intentionally fail if dir already exists to prevent overwrites
mkdir ${logdir}

# Write the disbatch taskfile
echo "#DISBATCH PREFIX $(realpath .)/${invocation} " >> ${taskfile}
echo '#DISBATCH SUFFIX &> task_${DISBATCH_TASKID}.log' >> ${taskfile}
echo "${cats}" | xargs -n1 >> ${taskfile}

cd ${logdir}

sbatch -C haswell -q regular -t ${time} -N ${nnode} -c ${cpus_per_task} \
    --wrap "${disbatch} -p ./disbatch ${taskfile}"
