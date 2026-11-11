#!/bin/bash
# usage: runq.sh <jobs file> <parallel>   each line: arguments for zoo.py
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
xargs -P "$2" -L 1 ../../.venv/bin/python zoo.py < "$1" >> logs/runq.out 2>&1
echo "QUEUE DONE $1" >> logs/runq.out
