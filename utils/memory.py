import psutil
import matplotlib as mpl
import matplotlib.pyplot as plt 

memory_usage_data = []

def print_memory_usage():
    process = psutil.Process()
    current_memory_usage = process.memory_info().rss / (1024 ** 3)
    memory_usage_data.append(current_memory_usage)
    print(f"Memory Usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")
    return

def plot_memory_usage(list): 
    fig, axs = plt.subplots(figsize=(10,10))
    axs.plot(list, 'o-')
    axs.set(xlabel='Call Number', ylabel='Memory Usage (GB)', title='Memory Usage Over Calls')
    fig.tight_layout()
    return fig
