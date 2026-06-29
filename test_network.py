"""
test_network.py - regression tests for the IEEE 33-bus BFS power flow solver.

Validates the base case against published Baran & Wu (1989) reference values.
Unlike a demo script, these tests FAIL loudly if the solver regresses.

Run from the repo root:
    pytest test_network.py -v
"""

import sys
import numpy as np

sys.path.insert(0, "src")
from network import load_array_kw_kvar, backward_forward_sweep, N_BUS

# Published reference values (Baran & Wu, 1989)
REF_VMIN = 0.9131      # pu, occurs at bus 18
REF_LOSSES = 202.68    # kW
TOL_V = 1e-3
TOL_LOSS = 0.5


def _base_case():
    P, Q = load_array_kw_kvar()
    res = backward_forward_sweep(P, Q)          # base case: no DER
    vmag = np.abs(res["V"][1:])                 # buses 1..33
    return res, vmag


def test_solver_converges():
    res, _ = _base_case()
    assert res["converged"], "BFS did not converge on the base case"


def test_slack_bus_is_unity():
    res, _ = _base_case()
    assert abs(res["V"][1] - 1.0) < 1e-9, f"slack bus = {res['V'][1]}, expected 1.0 pu"


def test_min_voltage_matches_reference():
    _, vmag = _base_case()
    assert abs(vmag.min() - REF_VMIN) < TOL_V, (
        f"V_min {vmag.min():.4f} pu != ref {REF_VMIN} pu"
    )


def test_min_voltage_located_at_bus_18():
    _, vmag = _base_case()
    bus = int(np.argmin(vmag)) + 1              # vmag[0] is bus 1
    assert bus == 18, f"expected V_min at bus 18, found at bus {bus}"


def test_losses_match_reference():
    res, _ = _base_case()
    assert abs(res["loss_kw"] - REF_LOSSES) < TOL_LOSS, (
        f"losses {res['loss_kw']:.2f} kW != ref {REF_LOSSES} kW"
    )


def test_voltages_in_physical_range():
    _, vmag = _base_case()
    assert np.all(vmag > 0.80), "a base-case voltage fell below 0.80 pu"
    assert np.all(vmag <= 1.0 + 1e-9), "a base-case voltage exceeded 1.0 pu (no DER -> none expected)"


def test_total_load_matches_literature():
    P, Q = load_array_kw_kvar()
    assert abs(P.sum() - 3715) < 1e-6, f"total P {P.sum()} kW != 3715 kW"
    assert abs(Q.sum() - 2300) < 1e-6, f"total Q {Q.sum()} kVAr != 2300 kVAr"


def test_solver_is_deterministic():
    r1, v1 = _base_case()
    r2, v2 = _base_case()
    assert np.allclose(v1, v2)
    assert abs(r1["loss_kw"] - r2["loss_kw"]) < 1e-9
