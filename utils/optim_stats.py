import numpy as np

calc_list = []

for i in range(1, 21):
    calc_list.append(f"CALCS/CsPbI3_gap_plus_4/h_optim_{i}_results/")

for calc in calc_list: 
    try:
        mc_data = np.loadtxt(f"{calc}final_training_cost.dat")
        best_loss = mc_data[-1, 1]
        print(f"{calc.split('/')[-2]}, best_loss = {best_loss:.4f}")
    except FileNotFoundError:
        print(f"File not found: {calc}final_training_cost.dat")
    except Exception as e:
        print(f"An error occurred while processing {calc}: {e}")