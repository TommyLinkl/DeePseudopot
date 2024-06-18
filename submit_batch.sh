#!/bin/bash

calc_array=("adam" "preAdam" "preSGD")
weight_array=("CBM" "CB" "CBVBM" "CBVB")

for calc in "${calc_array[@]}"; do
  for weight in "${weight_array[@]}"; do
    for i in {0..4}; do
      echo "Submitting: r4_inputs_${calc}_64kpts_${weight}_${i}/"
      
      sed -i "s/preSGD_64kpts_CBVB_4/${calc}_64kpts_${weight}_${i}/g" submit_CsPbI3_manualReorder_64kpts.in
      
      sbatch submit_CsPbI3_manualReorder_64kpts.in
      
      sed -i "s/${calc}_64kpts_${weight}_${i}/preSGD_64kpts_CBVB_4/g" submit_CsPbI3_manualReorder_64kpts.in
    done
  done
done
