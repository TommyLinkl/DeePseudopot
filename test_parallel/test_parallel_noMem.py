import sys
from main import main

testCases_list = ['noSOC_noPara', 'noSOC_para1', 'noSOC_para3', 'SOC_noPara', 'SOC_para1', 'SOC_para3']
# testCases_list = ['noSOC_noPara', 'SOC_noPara']
# testCases_list = ['noSOC_para3', 'SOC_para3']

for testCase in testCases_list: 
    print(f"Testing case: {testCase}")
    output_file_path = f"test_parallel/run_{testCase}.dat"
    with open(output_file_path, "w") as output_file:
        sys.stdout = output_file
        main(f"test_parallel/inputs_{testCase}/", f"test_parallel/results_{testCase}/")
    sys.stdout = sys.__stdout__
