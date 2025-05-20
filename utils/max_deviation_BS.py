import numpy as np

def calc_max_deviation_BS(calcBS_filename, refBS_filename, bandWeights_filename): 
    try: 
        calcBS = np.loadtxt(calcBS_filename)
        calcBS = calcBS[np.newaxis, :] if calcBS.ndim == 1 else calcBS
        calcBS = calcBS[:, 1:]

        refBS = np.loadtxt(refBS_filename)
        refBS = refBS[np.newaxis, :] if refBS.ndim == 1 else refBS
        refBS = refBS[:, 1:]
        bandWeights = np.loadtxt(bandWeights_filename)

        remove_bandIdx = [bandIdx for bandIdx, bandWeight in enumerate(bandWeights) if bandWeight == 0.0]
        calcBS = np.delete(calcBS, remove_bandIdx, axis=1)
        refBS = np.delete(refBS, remove_bandIdx, axis=1)

        nBands = calcBS.shape[1] // 2
        calcBS_noSpin = calcBS.reshape(calcBS.shape[0], nBands, 2).mean(axis=2)
        refBS_noSpin = refBS.reshape(refBS.shape[0], nBands, 2).mean(axis=2)

        max_deviation = np.max(calcBS_noSpin-refBS_noSpin)
    except FileNotFoundError: 
        print("Files not found. ")
        max_deviation, calcBS_noSpin, refBS_noSpin = None, None, None

    return max_deviation, calcBS_noSpin, refBS_noSpin
    
    


if __name__=="__main__": 
    '''
    # Relative 
    calc_list = []
    for i in range(1, 6):
        calc_list.append(f"CALCS/CsPbI3_relative_gap_plus_2/optim_{i}")

    for calcDir in calc_list: 
        totalEpochs = 200

        max_deviation, _, _ = calc_max_deviation_BS(f"{calcDir}_results/epoch_{totalEpochs}_BS_sys0_relative.dat", 
                                                    f"{calcDir}_inputs/expBandStruct_0.par", 
                                                    f"{calcDir}_inputs/bandWeights_0.par")

        print(max_deviation)
    '''

    # Absolute
    calc_list = []
    for i in range(1, 5):
        calc_list.append(f"CALCS_largeNLSOC/gap_optim_{i}")

    for calcDir in calc_list: 
        totalEpochs = 500

        max_deviation, _, _ = calc_max_deviation_BS(f"{calcDir}_results/epoch_{totalEpochs}_BS_sys0.dat", 
                                                    f"{calcDir}_inputs/expBandStruct_0.par", 
                                                    f"{calcDir}_inputs/bandWeights_0.par")

        print(max_deviation)