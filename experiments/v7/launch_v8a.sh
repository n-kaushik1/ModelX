#!/bin/bash
cd ~/modelx/experiments/v5 && (setsid nohup ./runq.sh jobs_round6.txt 6 > /dev/null 2>&1 < /dev/null &)
cd ~/modelx/experiments/v6 && (setsid nohup ./runq.sh jobs_sim5.txt 2 > /dev/null 2>&1 < /dev/null &)
sleep 3
pgrep -fc "^../../.venv/bin/python (outer_fold|sim_run)"
