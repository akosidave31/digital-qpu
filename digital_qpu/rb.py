"""Randomized benchmarking (RB): the standard lab experiment for measuring error per gate.
Run random sequences of m single-qubit Clifford operations followed by the one Clifford that undoes
them. Ideally the qubit returns to |0>; with errors, P(0) decays as A * p^m + B.
Error per Clifford r = (1 - p) / 2; error per physical gate ~ r / (average gates per Clifford)."""
import numpy as np
from digital_qubit import H, S, I2
from .qasm import Program, Op
from .device import Device
from .executor import probabilities


def _same(A, B):
    return abs(abs(np.trace(A.conj().T @ B)) - 2) < 1e-9


def clifford_group():
    """The 24 single-qubit Cliffords as (matrix, gate list), generated from h and s."""
    found, frontier = [(I2, [])], [(I2, [])]
    while frontier:
        nxt = []
        for U, seq in frontier:
            for name, G in (("h", H), ("s", S)):
                V = G @ U
                if not any(_same(V, W) for W, _ in found):
                    found.append((V, seq + [name]))
                    nxt.append((V, seq + [name]))
        frontier = nxt
    return found


CLIFFORDS = clifford_group()
AVG_GATES = float(np.mean([len(seq) for _, seq in CLIFFORDS]))


def _inverse_seq(U):
    for V, seq in CLIFFORDS:
        if _same(V, U.conj().T):
            return seq
    raise RuntimeError("inverse Clifford not found")


def single_qubit_device(device, q):
    """The part of a device that matters for single-qubit RB on qubit q."""
    pick = lambda lst: None if lst is None else [lst[q]]
    return Device(f"{device.name}-q{q}", 1, T1=pick(device.T1), T_phi=pick(device.T_phi),
                  gate_time_1q=device.gate_time_1q, readout_error=pick(device.readout_error),
                  gate_error_1q=pick(device.gate_error_1q))


def fit_decay(ms, ys):
    """Least-squares fit of y = A p^m + B: grid over p, exact A and B for each p (vectorized)."""
    ms, ys = np.asarray(ms, float), np.asarray(ys, float)
    ps = np.linspace(0.9, 0.999999, 20000)
    X = ps[:, None] ** ms[None, :]
    n = len(ms)
    sx, sy, sxx, sxy = X.sum(1), ys.sum(), (X * X).sum(1), (X * ys).sum(1)
    den = n * sxx - sx ** 2
    A = (n * sxy - sx * sy) / den
    B = (sy - A * sx) / n
    err = ((A[:, None] * X + B[:, None] - ys) ** 2).sum(1)
    k = int(np.argmin(err))
    return float(ps[k]), float(A[k]), float(B[k])


def predicted_epg(device, q):
    """First-order expectation: gate error (p/2) + decoherence during the gate."""
    t = device.gate_time_1q
    rate = (1 / device.T1[q] if device.T1 is not None else 0.0) + (1 / device.T_phi[q] if device.T_phi is not None else 0.0)
    return device.error_1q(q) / 2 + t * rate / 3


def randomized_benchmarking(device, qubit=0, lengths=(1, 20, 60, 120, 200), n_seq=10, seed=0):
    dev1 = single_qubit_device(device, qubit)
    rng = np.random.default_rng(seed)
    survival = []
    for m in lengths:
        vals = []
        for _ in range(n_seq):
            U, names = I2, []
            for k in rng.integers(0, len(CLIFFORDS), m):
                V, seq = CLIFFORDS[k]
                U = V @ U
                names += seq
            names += _inverse_seq(U)
            prog = Program(1, 1, [Op(g, (0,)) for g in names], {0: 0})
            vals.append(probabilities(prog, dev1)["0"])
        survival.append(float(np.mean(vals)))
    p, A, B = fit_decay(lengths, survival)
    epc = (1 - p) / 2
    return {"qubit": qubit, "lengths": list(lengths), "survival": survival, "p": float(p), "A": float(A),
            "B": float(B), "epc": float(epc), "epg": float(epc / AVG_GATES),
            "predicted_epg": float(predicted_epg(device, qubit)), "avg_gates_per_clifford": AVG_GATES}
