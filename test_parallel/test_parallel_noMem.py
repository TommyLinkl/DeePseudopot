import sys
from main import main

if __name__ == "__main__":
    testCases_list = ['SOC_noPara', 'SOC_para1', 'SOC_para2', 'SOC_para4', 'SOC_para8']

    for testCase in testCases_list: 
        print(f"Testing case: {testCase}")
        output_file_path = f"test_parallel/run_{testCase}.dat"
        '''
        with open(output_file_path, "w") as output_file:
            sys.stdout = output_file
            main(f"test_parallel/inputs_{testCase}/", f"test_parallel/results_{testCase}/")
        sys.stdout = sys.__stdout__
        '''
        main(f"test_parallel/inputs_{testCase}/", f"test_parallel/results_{testCase}/")
        print("\n"*10)