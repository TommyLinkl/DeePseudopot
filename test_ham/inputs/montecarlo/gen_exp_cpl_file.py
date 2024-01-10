
cpl_dict = {}
atoms = ["Pb", "I", "I", "I", "Cs"]
cpl_dict[(0,0,0,"vb")] = 0.028
cpl_dict[(0,1,0,"vb")] = 0.120
cpl_dict[(0,2,0,"vb")] = 0.009
cpl_dict[(1,0,0,"vb")] = 0.0
cpl_dict[(1,1,0,"vb")] = 0.0
cpl_dict[(1,2,0,"vb")] = 0.0
cpl_dict[(2,0,0,"vb")] = 0.0
cpl_dict[(2,1,0,"vb")] = 0.0
cpl_dict[(2,2,0,"vb")] = 0.0
cpl_dict[(3,0,0,"vb")] = 0.0
cpl_dict[(3,1,0,"vb")] = 0.0
cpl_dict[(3,2,0,"vb")] = 0.0

cpl_dict[(0,0,1,"vb")] = 0.69
cpl_dict[(0,1,1,"vb")] = 0.69
cpl_dict[(0,2,1,"vb")] = 0.69
cpl_dict[(1,0,1,"vb")] = 0.182
cpl_dict[(1,1,1,"vb")] = 0.182
cpl_dict[(1,2,1,"vb")] = 0.182
cpl_dict[(2,0,1,"vb")] = 1/3
cpl_dict[(2,1,1,"vb")] = 1/3
cpl_dict[(2,2,1,"vb")] = 1/3
cpl_dict[(3,0,1,"vb")] = 1/3
cpl_dict[(3,1,1,"vb")] = 1/3
cpl_dict[(3,2,1,"vb")] = 1/3

cpl_dict[(0,0,0,"cb")] = 0.028
cpl_dict[(0,1,0,"cb")] = 0.120
cpl_dict[(0,2,0,"cb")] = 0.009
cpl_dict[(1,0,0,"cb")] = 0.0
cpl_dict[(1,1,0,"cb")] = 0.0
cpl_dict[(1,2,0,"cb")] = 0.0
cpl_dict[(2,0,0,"cb")] = 0.0
cpl_dict[(2,1,0,"cb")] = 0.0
cpl_dict[(2,2,0,"cb")] = 0.0
cpl_dict[(3,0,0,"cb")] = 0.0
cpl_dict[(3,1,0,"cb")] = 0.0
cpl_dict[(3,2,0,"cb")] = 0.0

cpl_dict[(0,0,1,"cb")] = 1.25 
cpl_dict[(0,1,1,"cb")] = 0.7 
cpl_dict[(0,2,1,"cb")] = 0.69 
cpl_dict[(1,0,1,"cb")] = 0.4
cpl_dict[(1,1,1,"cb")] = 0.4
cpl_dict[(1,2,1,"cb")] = 0.4
cpl_dict[(2,0,1,"cb")] = 1/6   
cpl_dict[(2,1,1,"cb")] = 1/6
cpl_dict[(2,2,1,"cb")] = 1/6
cpl_dict[(3,0,1,"cb")] = 1/3
cpl_dict[(3,1,1,"cb")] = 1/3
cpl_dict[(3,2,1,"cb")] = 1/3


for i in range(5):
    print(f"Atom idx = {i}   atom = {atoms[i]}   position = XXX")
    for band in ["vb", "cb"]:
                    print(f"{band}-{band} coupling elements. ", end="")
                    for gamma in range(3):
                        if gamma == 0:
                            print("polarization of derivative = x")
                        elif gamma == 1:
                            print("polarization of derivative = y")
                        else:
                            print("polarization of derivative = z")
                        
                        for qidx in range(2):
                            if i == 4:
                                print(f"0.00000   ", end="")
                            else:
                                print(f"{cpl_dict[(i, gamma, qidx, band)]:.6g}   ", end="")
                        print("\n", end="")
                    print("\n", end="")

