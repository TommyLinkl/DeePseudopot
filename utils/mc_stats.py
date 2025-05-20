import numpy as np

calc_list = []

for i in range(1, 21):
    calc_list.append(f"CALCS_largeNLSOC/gap_heat_{i}_results/")

for calc in calc_list: 
    try:
        mc_data = np.loadtxt(f"{calc}final_mc_cost.dat")
        ratio = np.sum(mc_data[:, 2]) / len(mc_data[:, 2])
        max_loss = np.max(mc_data[:, 1])
        max_acc_loss = np.max(mc_data[:, 4])
        best_loss = mc_data[-1, 3]
        print(f"{calc.split('/')[-2]}, max_loss, mac_acc_loss, acc_ratio, best_loss = {max_loss}, {max_acc_loss}, {ratio*100:.1f}%, {best_loss}")
    except FileNotFoundError:
        print(f"File not found: {calc}final_mc_cost.dat")
    except Exception as e:
        print(f"An error occurred while processing {calc}: {e}")