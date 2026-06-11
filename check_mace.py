#!/usr/bin/env python3
"""
check_mace.py -- Phase 0 smoke test for the MACE descriptor backend.

Confirms that `mace-torch` is installed, that the MACE-MP-0 foundation model
loads, and that we can extract per-atom invariant descriptors for one of your
configurations. Prints the descriptor dimension D and how it splits per element
-- the numbers needed to design utils/mace_descriptors.py.

Run on a node with the project env (after `pip install mace-torch`):
    python check_mace.py [path/to/POSCAR]    (default: results_000_g_1/0.POSCAR)
"""
import sys, re
import numpy as np

POSCAR = sys.argv[1] if len(sys.argv) > 1 else "results_000_g_1/0.POSCAR"


def read_poscar_atoms(path):
    """Build an ASE Atoms from a POSCAR whose element line uses per-atom labels
    (Cs0, I11, ...). Strips trailing digits to real element symbols."""
    from ase import Atoms
    lines = [ln.rstrip("\n") for ln in open(path)]
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
    names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode = lines[7].strip().lower()
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n)])
    syms = []
    for tok, c in zip(names, counts):
        syms += [re.match(r"[A-Za-z]+", tok).group(0)] * c
    if mode.startswith("d"):
        coords = coords @ cell
    return Atoms(symbols=syms, positions=coords, cell=cell, pbc=True)


def main():
    try:
        from mace.calculators import mace_mp
    except Exception as e:
        print("IMPORT FAILED:", repr(e))
        print("Install with:  pip install mace-torch ase   (pulls e3nn)")
        return

    print(f"Reading {POSCAR}")
    atoms = read_poscar_atoms(POSCAR)
    print(f"  {len(atoms)} atoms; symbols: {atoms.get_chemical_symbols()}")
    print(f"  cell (A):\n{np.array(atoms.cell)}")

    print("\nLoading MACE-MP-0 (medium, cpu, float64) -- may download on first run...")
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

    # get_descriptors signature varies across mace versions; try the common forms.
    desc = None
    for kwargs in ({"invariants_only": True}, {}):
        try:
            desc = calc.get_descriptors(atoms, **kwargs)
            print(f"\nget_descriptors(atoms, {kwargs}) -> ok")
            break
        except TypeError as e:
            print(f"  get_descriptors(atoms, {kwargs}) raised TypeError: {e}")
        except AttributeError as e:
            print(f"  calc has no get_descriptors: {e}")
            print(f"  available calc methods: {[m for m in dir(calc) if not m.startswith('_')]}")
            return

    if desc is None:
        print("Could not call get_descriptors with the tried signatures; "
              "inspect the methods above and tell me the right one.")
        return

    desc = np.asarray(desc)
    print(f"\nDescriptor array shape: {desc.shape}  (expect [n_atoms, D])")
    print(f"  dtype: {desc.dtype}")
    syms = np.array(atoms.get_chemical_symbols())
    for el in ["Cs", "I", "Pb"]:
        m = syms == el
        if m.any():
            d = desc[m]
            print(f"  {el}: {m.sum()} atoms, D={d.shape[1]}, "
                  f"per-dim std range [{d.std(0).min():.3g}, {d.std(0).max():.3g}], "
                  f"#near-constant dims (std<1e-6)={int((d.std(0) < 1e-6).sum())}")

    print("\nPhase 0 OK. Report D and the per-element std ranges back and I'll "
          "build utils/mace_descriptors.py.")


if __name__ == "__main__":
    main()
