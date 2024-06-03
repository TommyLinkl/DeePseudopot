#!/bin/bash

testCases_list=('checkpoint' 'separateKpts' 'vanilla' 'sepKptsAndChkpnt')

for testCase in "${testCases_list[@]}"; do
    output_file="mprofile_output_${testCase}.dat"

    mprof run --output "${output_file}" test_memory.py "${testCase}"

    mprof plot -o "mprofile_${testCase}.png" "${output_file}"
done
