#!/bin/bash

testCases_list=('noSOC_noPara' 'noSOC_para1' 'noSOC_para3' 'SOC_noPara' 'SOC_para1' 'SOC_para3')

for testCase in "${testCases_list[@]}"; do
    output_file="mprofile_output_${testCase}.dat"

    mprof run --output "${output_file}" test_parallel_mem.py "${testCase}"

    mprof plot -o "mprofile_${testCase}.png" "${output_file}"
done
