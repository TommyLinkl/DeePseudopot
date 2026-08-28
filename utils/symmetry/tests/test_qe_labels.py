"""Self-tests for utils/symmetry/qe_labels.py (no Hamiltonian needed).

Checks the QE->model label convention map, the bands.x line parser, and the
reference<->QE alignment (including the valence/conduction scissor and a top
multiplet truncated by the fixed nBands window).
"""
import numpy as np

from utils.symmetry.qe_labels import (_qe_to_model_label, parse_bands_post,
                                      align_reference_to_qe, compare_labels)


def test_label_map():
    # O_h keeps Gamma6/7/8; D_4h subgroup renames the 2-dims to E1/2 (=G_6) and
    # E3/2 (=G_7); parity carried through.
    assert _qe_to_model_label('6', '+', 'O_h') == 'Gamma6+'
    assert _qe_to_model_label('7', '-', 'O_h') == 'Gamma7-'
    assert _qe_to_model_label('8', '+', 'O_h') == 'Gamma8+'
    assert _qe_to_model_label('6', '+', 'D_4h(4/mmm)') == 'E1/2+'
    assert _qe_to_model_label('7', '-', 'D_4h(4/mmm)') == 'E3/2-'
    print("  [ok] label map O_h/D_4h")


# a tiny synthetic bands.x snippet: one O_h block with a 2,2,4 pattern
_SNIPPET = """
                    xk=(   0.50000,   0.50000,   0.50000  )

     double point group O_h (m-3m)
     Band symmetry, O_h (m-3m)  double point group:

     e(  1 -  2) =    -6.00000  eV     2   --> G_6+
     e(  3 -  4) =    -5.00000  eV     2   --> G_7+
     e(  5 -  8) =    -4.00000  eV     4   --> G_8+
     e(  9 - 10) =    -1.00000  eV     2   --> G_6-
     e( 11 - 14) =     2.00000  eV     4   --> G_8-
"""


def test_parse_and_align(tmpwrite):
    path = tmpwrite("snippet_bands.out", _SNIPPET)
    blocks = parse_bands_post(path)
    assert len(blocks) == 1, blocks
    b = blocks[0]
    assert b['point_group'].startswith('O_h')
    labs = [m['label'] for m in b['multiplets']]
    assert labs == ['Gamma6+', 'Gamma7+', 'Gamma8+', 'Gamma6-', 'Gamma8-'], labs
    degs = [m['degen'] for m in b['multiplets']]
    assert degs == [2, 2, 4, 2, 4], degs
    print("  [ok] parse")

    # reference = QE shifted by -10 eV on VB, -9.7 on CB (a 0.3 eV scissor above
    # the gap between multiplet 3 and 4), deep bottom NOT dropped here, and the
    # top G_8- 4-fold truncated to 2 members by the window.
    #   VB (first 4 multiplets): -6,-5,-4,-1  -> -16,-15,-14,-11
    #   CB (last multiplet, partial): 2.0 -> -7.7, only 2 of 4 members
    E_ref = np.array([-16, -16, -15, -15, -14, -14, -14, -14, -11, -11, -7.7, -7.7])
    frozen, diag = align_reference_to_qe(E_ref, b['multiplets'], warn=lambda *a: None)
    assert diag['offset'] == 0, diag
    assert diag['last_partial'] is True, diag
    assert diag['spread'] < 1e-6, diag        # scissor absorbed by the one break
    got = [m['label'] for m in frozen]
    assert got == ['Gamma6+', 'Gamma7+', 'Gamma8+', 'Gamma6-'], got   # partial dropped
    assert [m['dim'] for m in frozen] == [2, 2, 4, 2]
    print("  [ok] align (scissor + truncated top)")

    # compare_labels: identical vs one swap
    model_seq = [(0, lab, 2) for lab in got]
    n_ok, n_tot, bad = compare_labels(frozen, model_seq)
    assert bad == [] and n_ok == n_tot == 4
    model_swapped = list(model_seq)
    model_swapped[1] = (0, 'Gamma6+', 2)      # pretend the model swapped G7->G6
    _, _, bad2 = compare_labels(frozen, model_swapped)
    assert bad2 == [(1, 'Gamma7+', 'Gamma6+')], bad2
    print("  [ok] compare_labels (agree + disagreement)")


def run():
    import os
    import tempfile
    d = tempfile.mkdtemp(prefix="qe_labels_test_")

    def tmpwrite(name, text):
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    test_label_map()
    test_parse_and_align(tmpwrite)


if __name__ == "__main__":
    print("qe_labels.py self-tests:")
    run()
    print("ALL qe_labels.py TESTS PASSED")
