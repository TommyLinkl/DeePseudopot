#!/bin/bash

# calc_array=("adam" "preAdam" "preSGD")
calc_array=("adam")
weight_array=("CBM" "CB" "CBVBM" "CBVB")

for calc in "${calc_array[@]}"; do
  for weight in "${weight_array[@]}"; do
    for i in {2..4}; do # for i in {0..4}; do
      echo "Submitting: r4_inputs_${calc}_64kpts_${weight}_${i}/"
      
      sed -i "s/adam_64kpts_CB_2/${calc}_64kpts_${weight}_${i}/g" submit_CsPbI3_manualReorder_64kpts.in
      
      sbatch submit_CsPbI3_manualReorder_64kpts.in
      
      sed -i "s/${calc}_64kpts_${weight}_${i}/adam_64kpts_CB_2/g" submit_CsPbI3_manualReorder_64kpts.in
    done
  done
done
