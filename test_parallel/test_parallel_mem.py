import sys
sys.path.append('..')
from main import main
from memory_profiler import profile

@profile
def run_main_memProfile(input_path, output_path):
    main(input_path, output_path)

if len(sys.argv) != 2:
    print("Usage: python test_parallel_mem.py <testCase>")
    sys.exit(1)

testCase = sys.argv[1]
print("Input argument:", testCase)

input_path = f"inputs_{testCase}/"
output_path = f"results_{testCase}/"
run_main_memProfile(input_path, output_path)