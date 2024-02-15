#!/bin/bash

testCases_list=('SOC_para8' 'SOC_para4' 'SOC_para2' 'SOC_para1' 'SOC_noPara') 

for testCase in "${testCases_list[@]}"; do
    output_file="mprofile_output_${testCase}.dat"

    mprof run --output "${output_file}" test_parallel_mem.py "${testCase}"

    mprof plot -o "mprofile_${testCase}.png" "${output_file}"
done
