#!/bin/bash
# usage: runq.sh <jobs file> <parallel>   each line: arguments for sim_lib.py
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
xargs -P "$2" -L 1 ../../.venv/bin/python sim_lib.py < "$1" >> logs/runq.out 2>&1
echo "QUEUE DONE $1" >> logs/runq.out
