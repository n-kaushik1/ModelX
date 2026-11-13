#!/bin/bash
cd "$(dirname "$0")"
until grep -q "QUEUE DONE jobs_v13_suite.txt" logs/runq.out; do sleep 60; done
caffeinate -i ./runq_lib.sh jobs_v13_lib.txt 4
caffeinate -i ./runq.sh jobs_v13_new.txt 4
caffeinate -i ./runq.sh jobs_v13_wide.txt 4
