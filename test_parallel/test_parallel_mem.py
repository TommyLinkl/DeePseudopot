import os
import sys
from main import main
from memory_profiler import profile

main = profile(main)

def wrapped_main(test_case):
    inputs_folder = 'inputs/'
    results_folder = 'results/'

    # Call the original main function with the provided arguments
    main(inputs_folder, results_folder, test_case)

if __name__ == "__main__":
    # Access the command-line argument (test case)
    test_case = sys.argv[1]

    # Call the wrapped main function with the test case argument
    wrapped_main(test_case)