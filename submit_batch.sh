#!/bin/bash

# calc_array=("adam" "preAdam" "preSGD")
calc_array=("adam")
weight_array=("CBM" "CB" "CBVBM" "CBVB")

for calc in "${calc_array[@]}"; do
  for weight in "${weight_array[@]}"; do
    for ((i=0; i<=3; i++)); do
      echo "Submitting: r4_inputs_heavyG_64kpts_${calc}_${weight}_${i}/"
      
      sed -i "s/adam_CB_0/${calc}_${weight}_${i}/g" submit_CsPbI3_manualReorder_64kpts.in
      
      sbatch submit_CsPbI3_manualReorder_64kpts.in
      
      sed -i "s/${calc}_${weight}_${i}/adam_CB_0/g" submit_CsPbI3_manualReorder_64kpts.in
    done
  done
done
