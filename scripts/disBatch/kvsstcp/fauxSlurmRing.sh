#!/bin/bash

size=3
[[ $# == 0 ]] || size=$1

x=0
while [[ $x < ${size} ]]
do 
   SLURM_NTASKS=${size} SLURM_PROCID=$x python kvsRing.py &
   x=$(( x + 1 ))
done

wait
