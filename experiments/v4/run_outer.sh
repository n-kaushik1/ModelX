#!/bin/bash
# usage: run_outer.sh <snapshot.py> <tag> <n_jobs> <problems...>
PY="${PY:-python}"   # set PY to the Python 3.11 interpreter with the leaderboard versions
cd "$(dirname "$0")"
snap=$1; tag=$2; nj=$3; shift 3
for k in "$@"; do
  "$PY" eval_outer.py "$snap" "$tag" "$k" "$nj" 0 > "logs/run_outer_${tag}_${k}.out" 2>&1
done
echo "stream $tag finished"
