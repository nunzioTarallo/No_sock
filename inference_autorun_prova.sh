#!/bin/bash

start=$(date +%s.%N)

./inference_autorun_prova1.sh &

./inference_autorun_prova2.sh &

./inference_autorun_prova3.sh &

wait

duration=$(echo "$(date +%s.%N) - $start " | bc)
LC_NUMERIC="en_US.UTF-8" execution_time= printf "%.2f\n" $duration >>tempi.txt
