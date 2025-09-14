#!/bin/bash

# Create the experiments folder if it doesn't exist
mkdir -p experiments
Nw=3
# Loop through the modes
for mode in "single_chain" "multi_chain" "multi_couplings"; do

echo "Working with mode = $mode ..."

# Create the config.yaml file for each mode
cat <<EOL > experiments/config_${mode}.yaml
N : 2000
N_samples : 500
seed : 1234
p : 20
dt_samples : 1
T : 1.0
mode: ${mode}
N_walkers: $([[ "$mode" == "single_chain" ]] && echo 1 || echo $Nw)
EOL

kernprof -l -o logs/profile.lprof -v scripts/run_profile.py -- -c experiments/config_$mode > logs/new_profiling_mode_$mode.txt

done

# Remove unnecesary file
rm logs/*.lprof