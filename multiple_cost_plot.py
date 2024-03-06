from utils.pp_func import plot_multiple_train_cost


fig = plot_multiple_train_cost('CALCS/CsPbI3_16kpts/results_1/final_training_cost.dat', 'CALCS/CsPbI3_16kpts/results_2/final_training_cost.dat', 'CALCS/CsPbI3_16kpts/results_3/final_training_cost.dat', 'CALCS/CsPbI3_16kpts/results_4/final_training_cost.dat')
fig.savefig('CALCS/CsPbI3_16kpts/all_train_cost.png')