import numpy
import shutil
import os 
import subprocess

selected_round2_calcs = ['clustered_anneal65', 'clustered_anneal6', 'clustered_anneal75', 'clustered_anneal66', 'clustered_anneal32', 'clustered_anneal14', 'clustered_anneal58', 'clustered_anneal44', 'clustered_anneal67', 'clustered_anneal1', 'clustered_anneal64', 'clustered_anneal90', 'clustered_anneal28', 'clustered_anneal21', 'clustered_anneal62', 'clustered_anneal72', 'clustered_anneal52', 'clustered_anneal96', 'clustered_anneal83', 'clustered_anneal70', 'clustered_anneal85', 'clustered_anneal59', 'clustered_anneal82', 'clustered_anneal53', 'clustered_anneal33', 'clustered_anneal77', 'clustered_anneal57', 'clustered_anneal50', 'clustered_anneal63', 'clustered_anneal54']
heat_perturb = [0.15, 0.1, 0.05, 0.04]
heat_beta = [0.5, 0.8, 1.0, 3.0]

i = 1
for calc in selected_round2_calcs: 
    for numHeat in range(len(heat_perturb)): 
        os.makedirs(f"CALCS/CsPbI3_ultraSmall_round4/heat{i}_inputs", exist_ok=True)

        src_dir = f"CALCS/CsPbI3_ultraSmall_round3/{calc}_inputs/"

        # Copy input files
        for file_name in ["bandWeights_0.par", "expBandStruct_0.par", "input_0.par", "kpoints_0.par", "NN_config.par", "system_0.par"]:
            src_file = f"{src_dir}{file_name}"
            dest_file = f"CALCS/CsPbI3_ultraSmall_round4/heat{i}_inputs/{file_name}"
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
            else:
                print(f"Source file {src_file} does not exist.")

        for file_name, new_name in zip(["best_CsParams.dat", "best_IParams.dat", "best_PbParams.dat", "best_PPmodel.pth"], ["init_CsParams.par", "init_IParams.par", "init_PbParams.par", "init_PPmodel.pth"]):
            src_file = f"CALCS/CsPbI3_ultraSmall_round3/{calc}_results/{file_name}"
            dest_file = f"CALCS/CsPbI3_ultraSmall_round4/heat{i}_inputs/{new_name}"
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)
            else:
                print(f"Source file {src_file} does not exist.")

        filename = f"CALCS/CsPbI3_ultraSmall_round4/heat{i}_inputs/NN_config.par"
        mc_percentage_value = heat_perturb[numHeat]
        mc_beta_value = heat_beta[numHeat]
        mc_percentage_cmd = f"sed -i 's/mc_percentage = [0-9.]\+/mc_percentage = {mc_percentage_value}/' {filename}"
        # print(f"Executing: {mc_percentage_cmd}")
        subprocess.run(mc_percentage_cmd, shell=True, check=True)
        
        mc_beta_cmd = f"sed -i 's/mc_beta = [0-9.]\+/mc_beta = {mc_beta_value}/' {filename}"
        # print(f"Executing: {mc_beta_cmd}")
        subprocess.run(mc_beta_cmd, shell=True, check=True)

        i += 1
