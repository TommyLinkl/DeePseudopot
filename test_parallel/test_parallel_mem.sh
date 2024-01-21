#!/bin/bash

testCases_list=('noSOC_noPara' 'noSOC_para1' 'noSOC_para3' 'SOC_noPara' 'SOC_para1' 'SOC_para3')

for testCase in "${testCases_list[@]}"; do
    output_file_path="mprofile_output_${testCase}.dat"

    mprof run --output "${output_file_path}" test_parallel/test_parallel_mem.py main "test_parallel/inputs_${testCase}/" "test_parallel/results_${testCase}/" "${testCase}"

    mprof plot -o "test_parallel/mprofile_${testCase}.png"
done
