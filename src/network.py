"""
IEEE 33-Bus Radial Distribution Test System
=============================================
Source: Baran, M.E. & Wu, F.F. (1989), "Network reconfiguration in distribution
systems for loss reduction and load balancing", IEEE Trans. Power Delivery.
Branch/load data cross-validated against multiple secondary sources; total
system load sums to 3715 kW / 2300 kVAr, matching published literature values.

Base values: V_base = 12.66 kV (line-to-line), S_base = 10 MVA
Bus 1 is the substation / slack bus (V = 1.0 pu, angle = 0).
"""

import numpy as np

V_BASE_KV = 12.66
S_BASE_MVA = 10.0
Z_BASE_OHM = (V_BASE_KV ** 2) / S_BASE_MVA  # ~16.0276 ohm

N_BUS = 33

# Branch data: (from_bus, to_bus, R_ohm, X_ohm)  -- 1-indexed bus numbers
BRANCH_DATA = [
    (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864),
    (4, 5, 0.3811, 0.1941), (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188),
    (7, 8, 0.7114, 0.2351), (8, 9, 1.0300, 0.7400), (9, 10, 1.0440, 0.7400),
    (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1298), (12, 13, 1.4680, 1.1550),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
    (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565),
    (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
    (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
    (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
    (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
    (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
]

# Nominal load at receiving (to_bus) bus: (bus, P_kW, Q_kVAr)
LOAD_DATA = {
    2: (100, 60), 3: (90, 40), 4: (120, 80), 5: (60, 30), 6: (60, 20),
    7: (200, 100), 8: (200, 100), 9: (60, 20), 10: (60, 20), 11: (45, 30),
    12: (60, 35), 13: (60, 35), 14: (120, 80), 15: (60, 10), 16: (60, 20),
    17: (60, 20), 18: (90, 40), 19: (90, 40), 20: (90, 40), 21: (90, 40),
    22: (90, 40), 23: (90, 50), 24: (420, 200), 25: (420, 200), 26: (60, 25),
    27: (60, 25), 28: (60, 20), 29: (120, 70), 30: (200, 600), 31: (150, 70),
    32: (210, 100), 33: (60, 40),
}

assert sum(p for p, q in LOAD_DATA.values()) == 3715, "Total P load mismatch with literature"
assert sum(q for p, q in LOAD_DATA.values()) == 2300, "Total Q load mismatch with literature"


def build_topology():
    """Build parent/children structure for the radial tree (rooted at bus 1)."""
    children = {b: [] for b in range(1, N_BUS + 1)}
    parent = {1: None}
    branch_of_child = {}
    for idx, (f, t, r, x) in enumerate(BRANCH_DATA):
        children[f].append(t)
        parent[t] = f
        branch_of_child[t] = idx
    # topological order (BFS from root) -> guarantees parents processed before children
    order = [1]
    i = 0
    while i < len(order):
        b = order[i]
        order.extend(children[b])
        i += 1
    return children, parent, branch_of_child, order


CHILDREN, PARENT, BRANCH_OF_CHILD, TOPO_ORDER = build_topology()


def load_array_kw_kvar():
    P = np.zeros(N_BUS + 1)
    Q = np.zeros(N_BUS + 1)
    for b, (p, q) in LOAD_DATA.items():
        P[b] = p
        Q[b] = q
    return P, Q


def backward_forward_sweep(P_kw, Q_kvar, der_p_kw=None, tol=1e-8, max_iter=200):
    """
    Backward/Forward Sweep (BFS) power flow for a radial distribution network.

    P_kw, Q_kvar : arrays of length N_BUS+1 (index 0 unused), nodal NET LOAD
                   (i.e. consumption; positive = load) in kW / kVAr.
    der_p_kw     : optional array of length N_BUS+1, DER active power injection
                   (generation) in kW at each bus, subtracted from net load.
                   Net nodal injection S_i = -(P_load_i - P_der_i) - jQ_load_i
                   (negative = power flowing OUT of the bus into the network).

    Returns dict with: V (complex pu, len N_BUS+1), Iline (complex pu per branch),
    Sline (complex pu per branch, flowing from->to), losses_kw, iterations.
    """
    if der_p_kw is None:
        der_p_kw = np.zeros(N_BUS + 1)

    # net per-unit complex power consumed at each bus (load - DER generation)
    Pnet = (P_kw - der_p_kw) / (S_BASE_MVA * 1000)
    Qnet = Q_kvar / (S_BASE_MVA * 1000)
    Snet = Pnet + 1j * Qnet  # pu, consumption convention (positive = drawn from network)

    nb = N_BUS
    Z = {}
    for idx, (f, t, r, x) in enumerate(BRANCH_DATA):
        Z[t] = complex(r, x) / Z_BASE_OHM  # impedance of branch feeding bus t

    V = np.ones(nb + 1, dtype=complex)  # flat start, V[1] = slack = 1.0∠0
    Iline = {b: 0j for b in range(2, nb + 1)}  # current in branch feeding bus b

    for it in range(1, max_iter + 1):
        V_prev = V.copy()

        # --- Backward sweep: bus current injections, then accumulate branch currents ---
        Ibus = np.zeros(nb + 1, dtype=complex)
        for b in range(2, nb + 1):
            Ibus[b] = np.conj(Snet[b] / V[b]) if abs(V[b]) > 1e-9 else 0j

        # accumulate from leaves to root: branch current = own bus current + sum of children branch currents
        Iline = {b: 0j for b in range(2, nb + 1)}
        for b in reversed(TOPO_ORDER):
            if b == 1:
                continue
            total = Ibus[b]
            for c in CHILDREN[b]:
                total += Iline[c]
            Iline[b] = total

        # --- Forward sweep: update voltages from root to leaves ---
        for b in TOPO_ORDER:
            if b == 1:
                continue
            p = PARENT[b]
            V[b] = V[p] - Iline[b] * Z[b]

        if np.max(np.abs(V - V_prev)) < tol:
            break

    # line flows (from parent -> bus), in pu, sending-end convention
    Sline = {}
    for b in range(2, nb + 1):
        Sline[b] = V[PARENT[b]] * np.conj(Iline[b])

    total_loss_pu = sum(abs(Iline[b]) ** 2 * Z[b].real for b in range(2, nb + 1))

    return {
        "V": V,
        "Iline": Iline,
        "Sline": Sline,
        "loss_kw": total_loss_pu * S_BASE_MVA * 1000,
        "iterations": it,
        "converged": it < max_iter,
    }


if __name__ == "__main__":
    P, Q = load_array_kw_kvar()
    res = backward_forward_sweep(P, Q)
    Vmag = np.abs(res["V"][1:])
    print(f"Converged in {res['iterations']} iterations")
    print(f"Min voltage: {Vmag.min():.4f} pu at bus {np.argmin(Vmag) + 1}")
    print(f"Total real power loss: {res['loss_kw']:.2f} kW")
