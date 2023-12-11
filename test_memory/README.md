Run these tests from the root directory of the repo,
using the command:

python -m test_memory.test_vanilla > test_memory/vanilla_run.dat
python -m test_memory.test_separateKpts > test_memory/separateKpts_run.dat
python -m test_memory.test_checkpoint > test_memory/checkpoint_run.dat

(note 1) the '.' rather than '/', 2) no .py at the end)

Please see xxx_results/memoryUsage.png for memory usage figure, and xxx_run.dat for run time. 