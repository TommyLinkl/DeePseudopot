import psutil
import matplotlib.pyplot as plt 

memory_usage_data = []
DEBUG_MEMORY_FLAG = True

def set_debug_memory_flag(value):
    global DEBUG_MEMORY_FLAG
    DEBUG_MEMORY_FLAG = value

def print_memory_usage():
    if DEBUG_MEMORY_FLAG: 
        process = psutil.Process()
        current_memory_usage = process.memory_info().rss / (1024 ** 3)
        memory_usage_data.append(current_memory_usage)
        print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")
    return

def plot_memory_usage(resultsFolder): 
    if DEBUG_MEMORY_FLAG: 
        fig, axs = plt.subplots(figsize=(8,8))
        axs.plot(range(len(memory_usage_data)), memory_usage_data, 'o-')
        axs.set(xlabel='Call Number', ylabel='Memory Usage (GB)', title='Memory Usage Over Calls')
        fig.tight_layout()
        fig.savefig(resultsFolder+"memoryUsage.png") 
    return