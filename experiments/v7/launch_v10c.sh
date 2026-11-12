#!/bin/bash
for p in $(pgrep -f "runq.sh jobs_sim7"); do pkill -P $p; kill $p; done
sleep 1
for p in $(pgrep -f "^xargs -P"); do pkill -P $p; kill $p; done
sleep 1
pgrep -f "^../../.venv/bin/python (outer_fold|sim_run)" | xargs -r kill
sleep 2
cd ~/modelx/experiments/v5 && (setsid nohup ./runq.sh jobs_round9.txt 6 > /dev/null 2>&1 < /dev/null &)
cd ~/modelx/experiments/v6 && (setsid nohup ./runq.sh jobs_sim8.txt 2 > /dev/null 2>&1 < /dev/null &)
sleep 3
pgrep -fc "^../../.venv/bin/python (outer_fold|sim_run)"
