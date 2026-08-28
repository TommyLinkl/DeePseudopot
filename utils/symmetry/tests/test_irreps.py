"""Self-tests for utils/symmetry/irreps.py."""
import numpy as np
from utils.symmetry.group import build_Oh
from utils.symmetry.irreps import DoubleLittleGroup


CASES = {
    'Gamma': ([0, 0, 0], 96, {2: 4, 4: 2}),      # O_h double: Gamma6+/-,Gamma7+/-(2), Gamma8+/-(4)
    'R':     ([0.5, 0.5, 0.5], 96, {2: 4, 4: 2}),
    'X':     ([0.5, 0.0, 0.0], 32, {2: 4}),      # D_4h double: 4 doublets
    'M':     ([0.5, 0.5, 0.0], 32, {2: 4}),
    'T':     ([0.5, 0.5, 0.3], 16, {2: 2}),      # C_4v double: E1/2, E3/2
    'Lambda':([0.3, 0.3, 0.3], 12, {1: 2, 2: 1}),# C_3v double: E1/2(2) + two 1-dim
    'Sigma': ([0.3, 0.3, 0.0], 8, {2: 1}),       # C_2v double: one doublet
}


def run():
    oh = build_Oh()
    for name, (k, ndg, spinor_dim_counts) in CASES.items():
        dlg = DoubleLittleGroup(oh, k, name=name)
        assert dlg.n == ndg, f"{name}: |double group|={dlg.n} expected {ndg}"
        chars, dims = dlg.character_table()
        assert dlg.nclass == len(dims), f"{name}: nclass {dlg.nclass} != nirr {len(dims)}"
        spin = dlg.spinor_irreps()
        # tally spinor dims
        got = {}
        for r in spin:
            got[r['dim']] = got.get(r['dim'], 0) + 1
        assert got == spinor_dim_counts, f"{name}: spinor dims {got} expected {spinor_dim_counts}"
        # sum of spinor dim^2 == |double|/2
        s2 = sum(r['dim'] ** 2 for r in spin)
        assert abs(s2 - dlg.n / 2) < 1e-9, f"{name}: sum spinor dim^2={s2} != {dlg.n/2}"
        labels = [f"{r['label']}(d{r['dim']})" for r in spin]
        print(f"  [ok] {name:7s} |G~|={dlg.n:3d}  #irr={len(dims):2d}  "
              f"spinor: {labels}")


if __name__ == "__main__":
    print("irreps.py self-tests:")
    run()
    print("ALL irreps.py TESTS PASSED")
