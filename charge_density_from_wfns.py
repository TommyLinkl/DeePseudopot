"""
Total charge density from plane-wave eigenvectors across k-points.
Config-driven version (no command-line args). Edit config.py to control behavior.

Expected files:
- system_0.par : 
    scale = <Bohr>
    cell
    a11 a12 a13
    a21 a22 a23
    a31 a32 a33
    atoms
    Symb fx fy fz
    ... (fractional atom positions)
- input_0.par :
    nBands = <int>
    maxKE  = <Hartree>   # atomic units
- kpoints_0.par :
    kx ky kz weight      # k in Cartesian units of (2π/alat)
- eigVec_k{idx}.npz :
    arr_0, arr_1, ..., arr_(nbands-1)  (each 1D complex PW coefficient vector)

ρ(r) = spin_deg * Σ_k w_k Σ_{n≤VBM} |ψ_{n,k}(r)|², ψ_{n,k}(r)= (1/√Ω) Σ_G C_{n,k}(G) e^{i(k+G)·r}
"""

import os, sys, math, json
from pathlib import Path
import numpy as np

# ---- Load user config (edit config.py) ----
# The script expects a config.py in the same directory with a dict named `config`.
# Example:
# config = {
#     "system_file": "system_0.par",
#     "input_file":  "input_0.par",
#     "kpoints_file":"kpoints_0.par",
#     "eigvec_pattern": "eigVec_k{idx}.npz",
#     "nk": 47,
#     "vbm_index": 18,
#     "ngrid": [96,96,96],
#     "spin_deg": 2.0,
#     "output_cube": "total_density.cube",
#     # Optional overrides:
#     # "ecut_ha": 8.0,      # override maxKE from input_0.par
#     # "npz_key": null      # use if NPZ stores a single 2D array under this name
# }
try:
    import charge_density_from_wfns_config as user_config
    CONFIG = user_config.config
except Exception as e:
    raise SystemExit(f"Failed to import charge_density_from_wfns_config.py with dict `config`. Error: {e}")

# ---- Constants ----
BOHR_TO_ANG = 0.529177210903
SYMBOL_TO_Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20,
    "Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,
    "Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,
    "Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
    "Sb":51,"Te":52,"I":53,"Xe":54
}

# ---- Parsers ----
def parse_system_par(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    sc_line = next((ln for ln in lines if ln.lower().startswith("scale")), None)
    if sc_line is None or "=" not in sc_line:
        raise ValueError(f"scale not found in {path}")
    scale_bohr = float(sc_line.split("=")[1].strip())
    alat_ang = scale_bohr * BOHR_TO_ANG

    try:
        i_cell = lines.index("cell")
    except ValueError:
        raise ValueError(f"'cell' section not found in {path}")
    cell_rows = []
    for j in range(1,4):
        parts = lines[i_cell+j].split()
        if len(parts) != 3:
            raise ValueError(f"cell row {j} malformed in {path}: {lines[i_cell+j]}")
        cell_rows.append([float(x) for x in parts])
    cell_frac = np.array(cell_rows, dtype=float)  # rows

    try:
        i_atoms = lines.index("atoms")
    except ValueError:
        raise ValueError(f"'atoms' section not found in {path}")
    atom_entries = []
    for ln in lines[i_atoms+1:]:
        parts = ln.split()
        if len(parts) != 4: continue
        sym = parts[0]
        fx, fy, fz = map(float, parts[1:4])
        Z = SYMBOL_TO_Z.get(sym)
        if Z is None:
            raise ValueError(f"Unknown element symbol '{sym}' in {path}")
        atom_entries.append((Z, np.array([fx,fy,fz], float)))

    lattice_ang = (scale_bohr * BOHR_TO_ANG) * cell_frac  # rows
    a1, a2, a3 = lattice_ang
    atoms = []
    for Z, fxyz in atom_entries:
        r = fxyz[0]*a1 + fxyz[1]*a2 + fxyz[2]*a3
        atoms.append({"Z": int(Z), "pos": r.tolist()})
    return lattice_ang, atoms, alat_ang

def parse_input_par(path):
    maxKE = None
    nBands = None
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or "=" not in ln: continue
            key, val = [x.strip() for x in ln.split("=",1)]
            if key == "maxKE":
                maxKE = float(val)
            elif key == "nBands":
                try: nBands = int(val)
                except: pass
    if maxKE is None:
        raise ValueError(f"maxKE not found in {path}")
    return maxKE, nBands

def load_kpoints(path, alat_ang):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) < 4: continue
            kx, ky, kz, w = map(float, parts[:4])
            scale = 2.0 * math.pi / float(alat_ang)  # (2π/alat) → 1/Å
            data.append((kx*scale, ky*scale, kz*scale, w))
    if not data:
        raise ValueError(f"No k-points parsed from {path}")
    K = np.array([d[:3] for d in data], float)
    W = np.array([d[3] for d in data], float)
    # Normalize weights if not already
    s = W.sum()
    if abs(s-1.0) > 1e-10:
        W = W / s
        print(f"[info] normalized k-point weights to 1.0 (was {s})")
    return K, W

# --- NEW: get G from your own BulkSystem path (self.basis) ---
def get_g_from_self_basis(inputs_folder, results_folder, *, n_system=1):
    """
    Use the user's read.py -> setAllBulkSystems(...) to construct BulkSystem(s)
    and return the G-basis from systemsList[0].basis() in the *exact* internal order.
    """
    try:
        from utils.read import BulkSystem, read_NNConfigFile
    except Exception as e:
        raise ImportError(f"Could not import from BulkSystem, read_NNConfigFile from utils.read: {e}")

    system = BulkSystem()
    system.setSystem(f"{inputs_folder}system_0.par")
    system.setInputs(f"{inputs_folder}input_0.par")

    G_t = system.basis()  # torch tensor (nPW,3)
    Gcart = G_t.detach().cpu().numpy().astype(float, copy=False)
    print(f"[info] The reciprocal basis shape: {Gcart.shape}")
    return Gcart

def load_basis_from_file(path):
    """
    Load G-vectors from a text file produced by BulkSystem.print_basisStates().
    Expected columns per line:
        idx  Gx  Gy  Gz  |G|
    Returns
    -------
    numpy.ndarray of shape (nPW, 3) with columns [Gx, Gy, Gz] in 1/Å.
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"{path} must have at least 4 columns (idx, Gx, Gy, Gz[, |G|])")
    G = arr[:, 1:4].astype(float, copy=False)
    return G

def _sanity_check_coeff_length_against_basis(example_npz, nPW, npz_key=None):
    """
    Quick check to avoid silent garbage if basis length/order is wrong.
    """
    try:
        data = np.load(example_npz)
        if npz_key is None:
            # default: arrays arr_0, arr_1, ...
            keys = sorted(data.files, key=lambda s: int(s.split('_')[1]))
            coeff = data[keys[0]]
        else:
            coeff = data[npz_key][0]
        data.close()
    except Exception as e:
        print(f"[warn] Could not open {example_npz} for sanity check ({e}). Skipping.")
        return

    L = coeff.shape[0]
    if L == nPW:
        print(f"[ok] eigvec length matches nPW ({L}).")
    elif (L % nPW) == 0:
        ncomp = L // nPW
        print(f"[ok] eigvec length = {L} = {ncomp} * nPW (spinor/components). Will split per component.")
    else:
        raise ValueError(
            f"Eigenvector length {L} is incompatible with basis nPW={nPW}. "
            f"This usually means the G-basis ordering/selection does not match the eigenvector basis."
        )

def load_npz_eigvecs(filename):
    """
    Load eigenvectors from a .npz file with arrays arr_0, arr_1, ..., arr_(nbands-1).
    Returns a list of 1D complex arrays [coeffs_G for band].
    """
    data = np.load(filename)
    keys = sorted(data.files, key=lambda s: int(s.split('_')[1]))
    arrs = [data[k] for k in keys]
    data.close()
    return arrs

# ---- Basis construction ----
def build_reciprocal(lattice_ang):
    A = np.array(lattice_ang, float).T  # columns a1,a2,a3
    vol = np.dot(A[:,0], np.cross(A[:,1], A[:,2]))
    b1 = 2*np.pi * np.cross(A[:,1], A[:,2]) / vol
    b2 = 2*np.pi * np.cross(A[:,2], A[:,0]) / vol
    b3 = 2*np.pi * np.cross(A[:,0], A[:,1]) / vol
    return np.stack([b1,b2,b3], axis=1)

def build_plane_wave_basis(max_ke_ha, lattice_ang, module_name_options=("utils.read", "read")):
    """
    Build G-vectors (1/Å) using your torch-based BulkSystem.basis() from read.py.
    - max_ke_ha: float, kinetic energy cutoff in Hartree (atomic units).
    - lattice_ang: (3,3) array-like in Angstrom (rows = a1,a2,a3).
    - module_name_options: tuple of module names to try importing BulkSystem from.

    Returns
    -------
    numpy.ndarray, shape (nPW, 3), dtype=float
        Plane-wave G vectors in Cartesian 1/Å, in the exact ordering defined by your basis().
    """
    import importlib, sys, numpy as np
    import torch

    last_err = None
    readmod = None
    for modname in module_name_options:
        try:
            readmod = importlib.import_module(modname)
            break
        except Exception as e:
            last_err = e
    if readmod is None:
        raise ImportError(f"Could not import BulkSystem from any of {module_name_options}: {last_err}")

    # Ensure torch tensors (float64) for everything BulkSystem touches
    lattice_t = torch.tensor(np.asarray(lattice_ang, dtype=float), dtype=torch.float64)
    atom_types_np = np.array([], dtype=int)  # fine as numpy; not used in basis construction
    atom_pos_t = torch.zeros((0, 3), dtype=torch.float64)
    kpts_t = torch.zeros((0, 3), dtype=torch.float64)
    exp_bs_t = torch.zeros((0,), dtype=torch.float64)

    try:
        BulkSystem = getattr(readmod, "BulkSystem")
    except AttributeError as e:
        raise AttributeError(f"Module {readmod.__name__} has no 'BulkSystem'") from e

    try:
        # Some versions expect these exact argument names; adjust if yours differ
        bs = BulkSystem(
            unitCellVectors_unscaled=lattice_t,  # torch tensor
            scale=1.0,
            atomTypes=atom_types_np,
            atomPos_unscaled=atom_pos_t,        # torch tensor
            kpts_recipLatVec=kpts_t,            # torch tensor
            expBandStruct=exp_bs_t              # torch tensor
        )
        # Make sure attributes are tensors where the code expects torch ops
        bs.unitCellVectors = lattice_t
        bs.maxKE = float(max_ke_ha)  # scalar (your code compares with torch expressions internally)
        G_t = bs.basis()             # should be a torch tensor of shape (nPW, 3)
        if not isinstance(G_t, torch.Tensor):
            raise TypeError(f"BulkSystem.basis() returned {type(G_t)}, expected torch.Tensor")
        Gcart = G_t.detach().cpu().numpy().astype(float, copy=False)
        return Gcart
    except Exception as e:
        # STRICT: do not fall back—surface the real issue
        raise RuntimeError(
            f"Failed to construct PW basis via BulkSystem.basis() "
            f"(ensure inputs are torch.float64 tensors and import path is correct)."
        ) from e

def split_spinor_components(coeffs, nPW, ncomp=2, layout="blocked"):
    coeffs = np.asarray(coeffs)
    if coeffs.size != nPW * ncomp:
        raise ValueError(f"coeffs.size={coeffs.size} != nPW*ncomp={nPW*ncomp}")
    if layout == "blocked":
        return [coeffs[i*nPW:(i+1)*nPW] for i in range(ncomp)]
    elif layout == "interleaved":
        return [coeffs[i::ncomp][:nPW] for i in range(ncomp)]
    else:
        raise ValueError("layout must be 'blocked' or 'interleaved'")

# ---- Real-space projection ----
def plane_wave_sum_on_grid(coeffs_G, k_cart, Gcart, grid, origin, axes):
    nx, ny, nz = grid
    a1, a2, a3 = axes
    # Build grid in fractions of lattice vectors in [0,1)
    x = np.arange(nx)/nx
    y = np.arange(ny)/ny
    z = np.arange(nz)/nz
    rx = origin[0] + x[:,None,None]*a1[0] + y[None,:,None]*a2[0] + z[None,None,:]*a3[0]
    ry = origin[1] + x[:,None,None]*a1[1] + y[None,:,None]*a2[1] + z[None,None,:]*a3[1]
    rz = origin[2] + x[:,None,None]*a1[2] + y[None,:,None]*a2[2] + z[None,None,:]*a3[2]
    r_flat = np.stack([rx.ravel(), ry.ravel(), rz.ravel()], axis=1)  # (Ngrid,3)
    
    kplusG = Gcart # + k_cart[None,:]
    phase = r_flat @ kplusG.T
    vol = abs(np.dot(a1, np.cross(a2, a3)))
    psi_flat = (np.exp(-1*1j*phase) @ coeffs_G) / np.sqrt(vol)
    psi = psi_flat.reshape(nx,ny,nz)
    return psi

# ---- Cube writer ----
def write_cube(filename, density, lattice_ang, atoms, origin=None):
    nx, ny, nz = density.shape
    a1, a2, a3 = [np.array(v, float) for v in lattice_ang]
    if origin is None: origin = np.array([0.0,0.0,0.0], float)
    with open(filename, "w") as f:
        f.write("Total charge density from PW wavefunctions\n")
        f.write("Generated by charge_density_from_wfns.py (config-driven)\n")
        f.write(f"{-len(atoms):4d} {origin[0]:13.6f} {origin[1]:13.6f} {origin[2]:13.6f}\n")
        f.write(f"{nx:4d} {a1[0]/nx:13.6f} {a1[1]/nx:13.6f} {a1[2]/nx:13.6f}\n")
        f.write(f"{ny:4d} {a2[0]/ny:13.6f} {a2[1]/ny:13.6f} {a2[2]/ny:13.6f}\n")
        f.write(f"{nz:4d} {a3[0]/nz:13.6f} {a3[1]/nz:13.6f} {a3[2]/nz:13.6f}\n")
        for atom in atoms:
            Z = int(atom["Z"]); x,y,z = atom["pos"]
            f.write(f"{Z:4d} {float(Z):13.6f} {x:13.6f} {y:13.6f} {z:13.6f}\n")
        vals = density.ravel(order="F")
        for i in range(0, vals.size, 6):
            chunk = vals[i:i+6]
            f.write("".join(f"{v:13.5E}" for v in chunk) + "\n")

BOHR_PER_ANG = 1.0 / 0.529177210903  # Å -> Bohr

def write_cube_triclinic(
    filename,
    origin_A,         # (3,) in Å
    a1_A, a2_A, a3_A, # (3,) lattice vectors in Å
    field,            # (Nx,Ny,Nz) float or complex (we'll |.|^2 if complex)
    atoms_A=None,     # list of (Z, charge, xÅ, yÅ, zÅ); charge can be 0.0
    title="generated by python",
    comment="volumetric data",
    assume_field_is_density=False,  # if False and field is complex, we write |field|^2
):
    """
    Writes a Gaussian CUBE supporting non-orthogonal (triclinic) cells.

    The three axis lines are per-voxel step vectors: a1/Nx, a2/Ny, a3/Nz (in Bohr).
    Data order is x-fastest, then y, then z (standard CUBE).
    """
    field = np.asarray(field)
    if np.iscomplexobj(field) and not assume_field_is_density:
        # if a complex wavefunction grid was passed, write density
        field = (field.real**2 + field.imag**2)

    if field.ndim != 3:
        raise ValueError(f"field must be (Nx,Ny,Nz); got shape {field.shape}")

    # Shapes and unit conversion
    Nx, Ny, Nz = map(int, field.shape)
    origin_B = np.asarray(origin_A, dtype=float) * BOHR_PER_ANG
    a1_B = np.asarray(a1_A, dtype=float) * BOHR_PER_ANG
    a2_B = np.asarray(a2_A, dtype=float) * BOHR_PER_ANG
    a3_B = np.asarray(a3_A, dtype=float) * BOHR_PER_ANG
    step1_B, step2_B, step3_B = a1_B / Nx, a2_B / Ny, a3_B / Nz

    # Right-handed check (helps some viewers)
    vol = float(np.dot(a1_B, np.cross(a2_B, a3_B)))
    if vol < 0:
        # swap a2<->a3 and Ny<->Nz and transpose field accordingly
        a2_B, a3_B = a3_B, a2_B
        step2_B, step3_B = step3_B, step2_B
        Ny, Nz = Nz, Ny
        field = np.transpose(field, (0, 2, 1))  # (Nx,Ny,Nz) -> (Nx,Nz,Ny)

    # atoms
    atoms_B = []
    if atoms_A:
        for Z, q, xA, yA, zA in atoms_A:
            atoms_B.append((int(Z), float(q), xA*BOHR_PER_ANG, yA*BOHR_PER_ANG, zA*BOHR_PER_ANG))
    n_atoms = len(atoms_B)

    # write
    with open(filename, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{comment}\n")
        f.write(f"{n_atoms:5d} {origin_B[0]:13.6f} {origin_B[1]:13.6f} {origin_B[2]:13.6f}\n")
        f.write(f"{Nx:5d} {step1_B[0]:13.6f} {step1_B[1]:13.6f} {step1_B[2]:13.6f}\n")
        f.write(f"{Ny:5d} {step2_B[0]:13.6f} {step2_B[1]:13.6f} {step2_B[2]:13.6f}\n")
        f.write(f"{Nz:5d} {step3_B[0]:13.6f} {step3_B[1]:13.6f} {step3_B[2]:13.6f}\n")
        for Z, q, xB, yB, zB in atoms_B:
            f.write(f"{Z:5d} {q:13.6f} {xB:13.6f} {yB:13.6f} {zB:13.6f}\n")

        # volumetric data (x-fastest; 6 per line)
        # ensure float (CUBE expects scalar)
        data = np.asarray(field, dtype=np.float32, order="C")
        cnt = 0
        for k in range(Nz):
            for j in range(Ny):
                row = data[:, j, k]  # x-fastest slice
                for val in row:
                    f.write(f"{val:13.5e} ")
                    cnt += 1
                    if cnt % 6 == 0:
                        f.write("\n")
        if cnt % 6 != 0:
            f.write("\n")


# ---- Main driver (config only) ----
def main():
    cfg = CONFIG
    # Required
    system_file   = cfg["system_file"]
    input_file    = cfg["input_file"]
    kpoints_file  = cfg["kpoints_file"]
    eigpat        = cfg["eigvec_pattern"]
    nk            = int(cfg["nk"])
    vbm_index     = int(cfg["vbm_index"])
    ngrid         = tuple(cfg["ngrid"])
    spin_deg      = float(cfg.get("spin_deg", 2.0))
    outcube       = cfg.get("output_cube", "total_density.cube")
    npz_key       = cfg.get("npz_key", None)

    # Geometry and cutoff
    lattice, atoms, alat_ang = parse_system_par(system_file)
    maxKE_in, nBands = parse_input_par(input_file)
    ecut = float(cfg.get("ecut_ha", maxKE_in))

    # K-points
    K, W = load_kpoints(kpoints_file, alat_ang)
    if nk != K.shape[0]:
        print(f"[warn] config nk={nk} but parsed {K.shape[0]} k-points; proceeding with parsed count.")
        nk = K.shape[0]

        # ---- Basis (prefer user's self.basis to avoid ordering mismatches) ----
    basis_mode = cfg.get("basis_mode", "self")  # "self" | "file" | "construct"
    basis_file = cfg.get("basis_file", None)
    inputs_folder  = cfg.get("inputs_folder", "./")    # folder containing system_0.par, input_0.par, kpoints_0.par
    results_folder = cfg.get("results_folder", "./")   # where basisStates_0.dat would be written (unused unless you still need it)

    if basis_mode == "self":
        try:
            Gcart = get_g_from_self_basis(inputs_folder, results_folder)
            print(f"[info] G-basis from self.basis() (nPW={Gcart.shape[0]})")
        except Exception as e:
            print(f"[warn] self.basis() path failed ({e}). Falling back to 'file'/'construct'.")
            basis_mode = "file" if basis_file else "construct"

    if basis_mode == "file" and basis_file:
        Gcart = load_basis_from_file(basis_file)
        print(f"[info] G-basis from file: {basis_file} (nPW={Gcart.shape[0]})")
    elif basis_mode == "construct" or (basis_mode == "file" and not basis_file):
        # Last resort: construct via minimal BulkSystem (uses maxKE and lattice)
        Gcart = build_plane_wave_basis(ecut, lattice)
        print(f"[info] G-basis constructed from (lattice, ecut) (nPW={Gcart.shape[0]})")

    nPW = Gcart.shape[0]
    print(f"[info] plane-wave count nPW={nPW}")

    # Quick sanity check against a sample eigvec file
    sample_eig = eigpat.format(idx=0)
    _sanity_check_coeff_length_against_basis(sample_eig, nPW, npz_key=npz_key)


    nx, ny, nz = ngrid
    a1, a2, a3 = lattice
    origin = np.array([0.0,0.0,0.0], float)

    rho = np.zeros((nx,ny,nz), dtype=np.float64)

    # Loop over k-points
    print("[info] summing valence charge density over k-points and bands...")
    for ik in range(nk):
        print("[info] summing k-point {}/{}...".format(ik+1, nk))
        k_cart = K[ik]
        w_k = W[ik]
        npz_path = eigpat.format(idx=ik)
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"Missing eigenvector file: {npz_path}")
        bands = load_npz_eigvecs(npz_path) if npz_key is None else [np.load(npz_path)[npz_key][i] for i in range(np.load(npz_path)[npz_key].shape[0])]
        nbands = len(bands)
        if vbm_index >= nbands:
            raise ValueError(f"vbm_index={vbm_index} >= nbands={nbands} in {npz_path}")
        # Validate nPW (support scalar and spinor/multi-component)
        # Allow: len == nPW (scalar) OR len == ncomp*nPW (spinor)
        lengths = [bands[b].shape[0] for b in range(nbands)]
        for b, L in enumerate(lengths):
            if (L == nPW) or (L % nPW == 0):
                continue
            else:
                raise ValueError(f"{npz_path}: band {b} has length {L} which is neither nPW={nPW} nor a multiple of it (spinor?).")
        # Determine layout preference
        spinor_layout = cfg.get("spinor_layout", "blocked")
        # Sum valence density (handle scalar or spinor coeffs)
        for b in range(vbm_index+1):
            coeffs = bands[b]
            if coeffs.shape[0] == nPW:
                psi = plane_wave_sum_on_grid(coeffs, k_cart, Gcart, (nx,ny,nz), origin, (a1,a2,a3))
                rho += w_k * (np.abs(psi)**2).real
            else:
                ncomp = coeffs.shape[0] // nPW
                comps = split_spinor_components(coeffs, nPW, ncomp=ncomp, layout=spinor_layout)
                comp_rho = None
                for c in comps:
                    psi_c = plane_wave_sum_on_grid(c, k_cart, Gcart, (nx,ny,nz), origin, (a1,a2,a3))
                    dens_c = (np.abs(psi_c)**2).real
                    comp_rho = dens_c if comp_rho is None else (comp_rho + dens_c)
                rho += w_k * comp_rho
        # TODO: Write intermediate cube files for each kpoint. 

    rho *= float(spin_deg)

    write_cube(outcube, rho, lattice, atoms, origin=origin)
    print(f"[done] Wrote cube: {outcube}")

if __name__ == "__main__":
    main()
