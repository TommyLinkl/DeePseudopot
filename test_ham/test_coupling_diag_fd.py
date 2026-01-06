import os
import tempfile
import pathlib
import numpy as np
import torch

from utils.read import read_NNConfigFile, setAllBulkSystems, setNN, BulkSystem
from utils.init_NN_train import init_ZungerPP
from utils.ham import Hamiltonian

# just test on cpu
device = torch.device("cpu")
torch.set_printoptions(precision=8)

pwd = pathlib.Path(__file__).parent.resolve()
inputs_dir = pwd / "eph_diag_fd_test_inputs"
expected_dir = pwd / "eph_diag_fd_test_results"


def load_expected_couplings(filepath):
    tmp_sys = BulkSystem()
    tmp_sys.setExpCouplings(str(filepath))
    return tmp_sys.expCouplingBands


def build_hamiltonians(systems, atomPPOrder, PPparams, NNConfig, model):
    hams = []
    for iSys, sys in enumerate(systems):
        ham = Hamiltonian(
            sys,
            PPparams,
            atomPPOrder,
            device,
            NNConfig=NNConfig,
            iSystem=iSys,
            SObool=NNConfig["SObool"],
            coupling=True,
        )
        ham.NN_locbool = True
        ham.set_NNmodel(model)
        hams.append(ham)
    return hams


def compare_couplings(expected, actual_by_delta, rtol=1e-4, atol=1e-6, zero_thresh=1e-9):
    def normalize(val):
        return 0.0 if abs(val) < zero_thresh else val

    def format_val(val):
        return f"{normalize(val):.5e}"

    deltas = sorted(actual_by_delta.keys())
    header = ["Coupling", "Expected"] + [f"{d:.0e}" for d in deltas]
    col_widths = [max(len(h), 12) for h in header]

    def fmt_row(values):
        return "  ".join(f"{val:<{col_widths[i]}}" for i, val in enumerate(values))

    print(fmt_row(header))
    print(fmt_row(["-" * w for w in col_widths]))

    for key in sorted(expected.keys()):
        row = [str(key), format_val(float(expected[key]))]
        for d in deltas:
            row.append(format_val(float(actual_by_delta[d].get(key, 0.0))))
        print(fmt_row(row))

    ok = True
    worst_key = None
    worst_diff = -1.0
    worst_exp = None
    worst_act = None

    for d, actual in actual_by_delta.items():
        for key, exp_val in expected.items():
            if key not in actual:
                ok = False
                print(f"Missing key in computed couplings: {key} for delta={d:.0e}")
                continue
            exp_val = normalize(float(exp_val))
            act_val = normalize(float(actual[key]))
            diff = abs(exp_val - act_val)
            if diff > worst_diff:
                worst_diff = diff
                worst_key = key
                worst_exp = exp_val
                worst_act = act_val
            if not np.isclose(exp_val, act_val, rtol=rtol, atol=atol):
                ok = False

        extra_keys = set(actual.keys()) - set(expected.keys())
        if extra_keys:
            ok = False
            print(f"Found unexpected keys in computed couplings for delta={d:.0e}: {sorted(extra_keys)}")

    print(f"All couplings close: {ok}")
    print(f"Worst diff {worst_diff} at {worst_key} (expected={worst_exp}, actual={worst_act})")
    if not ok:
        raise AssertionError("Coupling values do not match expected results.")


def main():
    inputs_folder = str(inputs_dir) + os.sep
    expected_file = expected_dir / "couplingBands_0.dat"

    NNConfig = read_NNConfigFile(inputs_folder + "NN_config.par")
    with tempfile.TemporaryDirectory(dir=str(pwd)) as tmpdir:
        results_folder = str(pathlib.Path(tmpdir)) + os.sep
        systems, atomPPOrder, nPseudopot, PPparams, totalParams, localPotParams = setAllBulkSystems(
            NNConfig["nSystem"],
            inputs_folder,
            results_folder,
        )
        for iSys, sys in enumerate(systems):
            sys.setQPointsAndWeights(inputs_folder + f"qpoints_{iSys}.par")

        model = setNN(NNConfig, nPseudopot)
        model, _ = init_ZungerPP(
            inputs_folder,
            model,
            atomPPOrder,
            localPotParams,
            nPseudopot,
            NNConfig,
            device,
            results_folder,
            force_retrain=False,
        )
        model.eval()

        hams = build_hamiltonians(systems, atomPPOrder, PPparams, NNConfig, model)
        expected = load_expected_couplings(expected_file)
        for iSys, ham in enumerate(hams):
            _ = ham.calcBandStruct()
            deltas = [1e-3, 1e-4, 1e-5]
            actual_by_delta = {d: ham.calcCouplings_diag_fd(delta=d) for d in deltas}
            compare_couplings(expected, actual_by_delta)


if __name__ == "__main__":
    main()
