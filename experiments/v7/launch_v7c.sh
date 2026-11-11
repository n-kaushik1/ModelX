#!/bin/bash
for p in $(pgrep -f "runq.sh jobs_round5|runq.sh jobs_sim4"); do pkill -P $p; kill $p; done
sleep 1
for p in $(pgrep -f "^xargs -P"); do pkill -P $p; kill $p; done
sleep 1
pgrep -f "^../../.venv/bin/python (outer_fold|sim_run)" | xargs -r kill
sleep 2
cd ~/modelx/experiments
rm -f v5/logs/outer/v7b_* v6/logs/sim/v7b_*
grep -v "^v7b" v5/logs/outer.log > v5/logs/outer.tmp; mv v5/logs/outer.tmp v5/logs/outer.log
grep -v "^v7b" v6/logs/sim.log > v6/logs/sim.tmp; mv v6/logs/sim.tmp v6/logs/sim.log
cd v5 && (setsid nohup ./runq.sh jobs_round5.txt 6 > /dev/null 2>&1 < /dev/null &)
cd ../v6 && (setsid nohup ./runq.sh jobs_sim4.txt 2 > /dev/null 2>&1 < /dev/null &)
sleep 3
pgrep -fc "^../../.venv/bin/python (outer_fold|sim_run)"
