#!/bin/bash
# usage: runq.sh <jobs file> <parallel>   each line: args for outer_fold.py
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
xargs -P "$2" -L 1 ../../.venv/bin/python outer_fold.py < "$1" >> logs/runq.out 2>&1
echo "QUEUE DONE $1" >> logs/runq.out
