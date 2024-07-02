import numpy as np

inflate_factor = 1.0/1.833

original_kpts = np.loadtxt("CALCS/CsPbI3_eval_fullBand/inputs/kpoints_0.par")
print(original_kpts[0])

new_kpts = np.copy(original_kpts)
new_kpts[:, :3] *= inflate_factor
print(new_kpts[0])

np.savetxt("inflated_kpoints_0.par", new_kpts, fmt="%f")