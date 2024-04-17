from utils.pp_func import plot_multiple_train_cost

filePrefix = 'CALCS/CsPbI3_16kpts/'

'''
fig = plot_multiple_train_cost(
    [f'{filePrefix}results_1/final_training_cost.dat'], 
    [f'{filePrefix}results_smallerNN/final_training_cost.dat'], 
    [f'{filePrefix}results_smallestNN/final_training_cost.dat'], 
    [f'{filePrefix}results_shallowerNN/final_training_cost.dat'], 
    [f'{filePrefix}results_deeperNN/final_training_cost.dat', f'{filePrefix}results_deeperNN_2/final_training_cost.dat'], 
    [f'{filePrefix}results_largerNN/final_training_cost.dat', f'{filePrefix}results_largerNN_2/final_training_cost.dat'], 
    labels=['32x32x32', '16x16x16', '8x8x8', '16x16', '32x32x32x32', '48x64x48'], ymin=200, ymax=600)
fig.savefig(f'{filePrefix}NNsize_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_1/final_training_cost.dat', f'{filePrefix}results_2/final_training_cost.dat', f'{filePrefix}results_3/final_training_cost.dat', f'{filePrefix}results_4/final_training_cost.dat'], 
    labels=['results_1,2,3,4'], ymin=300, ymax=500)
fig.savefig(f'{filePrefix}results_1234_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_HeInit_initGradTest/final_training_cost.dat', f'{filePrefix}results_HeInit_2/final_training_cost.dat', f'{filePrefix}results_HeInit_3/final_training_cost.dat'], 
    [f'{filePrefix}results_HeInit_largelr_1/final_training_cost.dat', f'{filePrefix}results_HeInit_largelr_2/final_training_cost.dat'], 
    labels=['HeInit, lr=0.0001, 0.0005', 'HeInit, lr=0.01'], ymin=300, ymax=500)
fig.savefig(f'{filePrefix}results_HeInit_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_4_repeat_test/final_training_cost.dat'], 
    [f'{filePrefix}results_adadelta/final_training_cost.dat'], 
    [f'{filePrefix}results_adagrad/final_training_cost.dat'], 
    [f'{filePrefix}results_adamw/final_training_cost.dat'], 
    [f'{filePrefix}results_adamax/final_training_cost.dat'], 
    [f'{filePrefix}results_nadam/final_training_cost.dat'], 
    [f'{filePrefix}results_rmsprop/final_training_cost.dat'], 
    labels=['adam (results_4)', 'adadelta', 'adagrad', 'adamw', 'adamax', 'nadam', 'rmsprop'], ymin=310.5, ymax=315, ylogBoolean=True)
fig.savefig(f'{filePrefix}adamVarieties_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_bandEdge_1/final_training_cost.dat', f'{filePrefix}results_bandEdge_2/final_training_cost.dat'], 
    labels=['bandEdge_12'], ymin=140, ymax=250)
fig.savefig(f'{filePrefix}results_bandEdge12_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_sgd/final_training_cost.dat'], 
    [f'{filePrefix}results_sgd_largeMom/final_training_cost.dat'], 
    [f'{filePrefix}results_sgd_medMom/final_training_cost.dat'], 
    [f'{filePrefix}results_sgd_smallMom/final_training_cost.dat'], 
    [f'{filePrefix}results_asgd/final_training_cost.dat'],  
    labels=['sgd', 'sgd, mom=0.9', 'sgd, mom=0.5', 'sgd, mom=0.1', 'asgd'], ylogBoolean=True)
fig.savefig(f'{filePrefix}sgdVarieties_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_4_repeat_test/final_training_cost.dat'], 
    [f'{filePrefix}results_adam_lessMom/final_training_cost.dat'], 
    [f'{filePrefix}results_adam_moreMom/final_training_cost.dat'],  
    labels=['default adam', 'adam, b1=0.8, b2=0.8', 'adam, b1=0.99, b2=0.9999'], ylogBoolean=True)
fig.savefig(f'{filePrefix}adamMom_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_1/final_training_cost.dat'], 
    [f'{filePrefix}results_deeperNN/final_training_cost.dat', f'{filePrefix}results_deeperNN_2/final_training_cost.dat'], 
    [f'{filePrefix}results_largerNN/final_training_cost.dat', f'{filePrefix}results_largerNN_2/final_training_cost.dat'], 
    [f'{filePrefix}results_64x64/final_training_cost.dat', f'{filePrefix}results_64x64_2/final_training_cost.dat'], 
    [f'{filePrefix}results_128/final_training_cost.dat', f'{filePrefix}results_128_2/final_training_cost.dat'], 
    labels=['32x32x32', '32x32x32x32', '48x64x48', '64x64', '128'], ymin=200, ymax=600)
fig.savefig(f'{filePrefix}NNsize_largerDeeper_train_cost.png')

fig = plot_multiple_train_cost(
    [f'{filePrefix}results_1_wAdam/final_training_cost.dat', f'{filePrefix}results_1_wAdam_2/final_training_cost.dat'], 
    labels=['wAdam, lr=0.05'], ymin=250, ymax=400)
fig.savefig(f'{filePrefix}results_wAdam_12_train_cost.png')
'''


fig = plot_multiple_train_cost(
    [f'{filePrefix}results_1/final_training_cost.dat', f'{filePrefix}results_2/final_training_cost.dat', f'{filePrefix}results_3/final_training_cost.dat', f'{filePrefix}results_4/final_training_cost.dat'], 
    [f'{filePrefix}results_sigmoidXavier/final_training_cost.dat'], 
    [f'{filePrefix}results_celuHeInit/final_training_cost.dat', f'{filePrefix}results_celuHeInit_2/final_training_cost.dat'], 
    labels=['relu + Xavier', 'sigmoid + Xavier', 'celu + HeInit'], ymin=300, ymax=600)
fig.savefig(f'{filePrefix}activationFunc_train_cost.png')