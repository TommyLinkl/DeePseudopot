import numpy as np
np.set_printoptions(formatter={'float': '{:0.10f}'.format})

def map_fractional_positions(old_lattice_vectors, old_frac_positions, new_lattice_vectors):
    """
    Convert fractional positions from one lattice basis to another.

    Parameters
    ----------
    old_lattice_vectors : array_like, shape (3,3)
        Old lattice vectors, each row is a lattice vector.
    old_frac_positions : array_like, shape (N,3)
        Fractional positions with respect to the old lattice.
    new_lattice_vectors : array_like, shape (3,3)
        New lattice vectors, each row is a lattice vector.

    Returns
    -------
    new_frac_positions : ndarray, shape (N,3)
        Fractional positions in the new lattice, wrapped into [0,1).
    """
    old_lat = np.array(old_lattice_vectors, dtype=float)
    old_frac = np.array(old_frac_positions, dtype=float)
    new_lat = np.array(new_lattice_vectors, dtype=float)

    # Compute Cartesian coordinates from old fractional positions
    cart_pos = old_frac.dot(old_lat)

    # Solve for new fractional coordinates
    inv_new_lat = np.linalg.inv(new_lat)
    new_frac = cart_pos.dot(inv_new_lat)

    # Wrap into [0,1)
    new_frac_wrapped = np.mod(new_frac, 1.0)

    return new_frac_wrapped

if __name__ == "__main__":
    old_lattice_vectors = [[6.650888580000, 0.000000000000, 0.000000000000],
                           [0.000000000000, 6.650888580000, 0.000000000000],
                           [0.000000000000, 0.000000000000, 3.844771320000]]
    
    old_frac_positions = [[0.1788010600000000, 0.1788010600000000, 0.0000000000000000],
                          [0.3211989400000000, 0.3211989400000000, 0.5000000000000000],
                          [0.1788010600000000, 0.8211989400000000, 0.0000000000000000],
                          [0.3211989400000000, 0.6788010600000000, 0.5000000000000000],
                          [0.6788010600000000, 0.6788010600000000, 0.5000000000000000],
                          [0.8211989400000000, 0.8211989400000000, 0.0000000000000000],
                          [0.6788010600000000, 0.3211989400000000, 0.5000000000000000],
                          [0.8211989400000000, 0.1788010600000000, 0.0000000000000000]]

    new_lattice_vectors = [[-6.650888580000/2,   6.650888580000/2,   3.844771320000/2],
                           [6.650888580000/2,   -6.650888580000/2,   3.844771320000/2],
                           [6.650888580000/2,   6.650888580000/2,   -3.844771320000/2]]
    
    new_frac_positions = map_fractional_positions(old_lattice_vectors,
                                                  old_frac_positions,
                                                  new_lattice_vectors)
    
    print("New lattice vectors:")
    print(np.array(new_lattice_vectors))

    # print("New fractional positions:")
    # print(new_frac_positions)   

    rounded = np.round(new_frac_positions, 5)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_atoms = new_frac_positions[sorted(unique_indices)]

    print("New unique fractional positions:")
    print(unique_atoms)