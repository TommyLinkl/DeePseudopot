"""
One-time QE-anchored irrep label report (no training needed).

Builds the real DeePseudopot system + Hamiltonian from an inputs folder (using
the SAME setup as the validation suite), then prints -- at every high-symmetry
POINT -- the reference irrep labels taken from a cubic QE bands.x run
(lsym=.true.) beside the model's own energy-ordered labels, flagging any
disagreement (e.g. a swapped Gamma6/Gamma7 = wrong initial basin).

Usage:
  PYTHONPATH=<DeePseudopot> python -m utils.symmetry.report_qe_labels \
      <inputs_folder> [path/to/bands_post.out]

If the QE path is omitted, it is read from NN_config.par's
`irrep_reference_labels`. The model used is a fresh init from the inputs folder
(init_PPmodel.pth if present via setNN), matching what freeze sees at setup.
"""
import os
import sys

import torch

from utils.constants import AUTOEV  # noqa: F401 (kept for parity with validate)
from .validate import _setup
from .train_hooks import IrrepContext


def main(inputs, qe_path=None):
    system, ham, NNConfig = _setup(inputs)
    # _setup builds a RANDOM-init model; load the real initial potential so the
    # QE-vs-model comparison reflects the basin the fit actually starts from.
    init_pth = inputs + 'init_PPmodel.pth'
    if os.path.exists(init_pth):
        ham.model.load_state_dict(torch.load(init_pth))
        print(f"[report] loaded initial potential {init_pth}")
    else:
        print(f"[report] WARNING: {init_pth} not found -- comparing QE labels to a "
              f"RANDOM-init model (labels themselves are still QE-anchored).")
    if qe_path is not None:
        NNConfig['irrep_reference_labels'] = qe_path
    if not NNConfig.get('irrep_reference_labels'):
        raise SystemExit("No QE bands.x path: pass it as arg 2 or set "
                         "irrep_reference_labels in NN_config.par")

    # cachedMats_info lives on the ham after initAndCacheHams; _setup attaches it
    # implicitly through the ham, but IrrepContext needs it explicitly. Rebuild a
    # minimal handle the same way NN_train does: buildHtot_cached reads the ham's
    # own cache, so pass None and let the ham use its attached cache.
    ctx = IrrepContext(system, ham, NNConfig, cachedMats_info=None)
    print(f"\n=== QE-anchored irrep label report for {inputs} ===")
    print(f"    QE file: {NNConfig['irrep_reference_labels']}")
    print(f"    high-sym POINTS detected: "
          f"{[ctx.sak[k].name for k in ctx.point_kidx]}")
    report = ctx.qe_label_report()

    print("\n=== summary ===")
    any_bad = False
    for name, r in report.items():
        tag = "OK " if not r['disagreements'] else "!! "
        any_bad = any_bad or bool(r['disagreements'])
        print(f"  {tag}{name}: {r['agree']}/{r['total']} multiplets agree with the "
              f"initial model")
    if any_bad:
        print("\n  At least one point disagrees: the initial model's Gamma6/Gamma7 "
              "(or parity) ordering differs from QE at those multiplets. With "
              "irrep_reference_labels set, the sector loss is anchored to the QE "
              "labels above, so the fit will be driven toward the QE symmetry "
              "assignment (and assert_sector_counts will flag any window-edge "
              "count mismatch).")
    else:
        print("\n  The initial model already agrees with QE at every labeled "
              "multiplet; QE anchoring changes nothing here but documents it.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    torch.set_default_dtype(torch.float64)
    main(sys.argv[1].rstrip('/') + '/', sys.argv[2] if len(sys.argv) > 2 else None)
