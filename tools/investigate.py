"""Investigation (v0.5.1): no library changes, measurements only.
Run from the repo root:  python tools/investigate.py
Part A: are bad days (TLS) independent across qubits?
Part B: where does the +10-20% randomized-benchmarking offset come from?"""
import os, sys, zlib
sys.path.insert(0, os.getcwd())
import numpy as np
from math import comb
import digital_qpu as dq
from digital_qpu.calibration import TLS_PROB
from digital_qpu.rb import randomized_benchmarking, single_qubit_device, CLIFFORDS, AVG_GATES, predicted_epg
from digital_qpu.executor import _apply, _gate_error
from digital_qpu.qasm import Op
from digital_qpu.device import Device
from digital_qubit import NDensity, PairNoise, gate_matrix

DQ5 = dq.DEVICES["dq-5"]

# ======================= Part A =======================
print("PART A: are bad days independent across qubits? (20000 simulated days)")
D, nq = 20000, DQ5.n_qubits
seed = zlib.crc32(DQ5.name.encode())
bad = np.random.default_rng([seed, 2]).random((D, 2 * nq))[:, :nq] < TLS_PROB   # same draws as calibrate()
print("  bad-day rate per qubit:", " ".join(f"q{q} {bad[:, q].mean() * 100:.2f}%" for q in range(nq)),
      f"(built in {TLS_PROB * 100:.0f}%)")
exp_pair = TLS_PROB ** 2 * D
worst = 0.0
for i in range(nq):
    for j in range(i + 1, nq):
        obs = int(np.sum(bad[:, i] & bad[:, j]))
        worst = max(worst, abs(obs - exp_pair) / np.sqrt(exp_pair))
print(f"  pairs bad on the same day: expected {exp_pair:.0f} each by chance; "
      f"largest deviation {worst:.1f} standard deviations")
k = bad.sum(1)
print(f"  {'#bad qubits':<12}{'observed':>10}{'expected (independent)':>25}")
for m in range(4):
    e = D * comb(nq, m) * TLS_PROB ** m * (1 - TLS_PROB) ** (nq - m)
    print(f"  {m:<12}{int(np.sum(k == m)):>10}{e:>25.1f}")
e3 = D * sum(comb(nq, m) * TLS_PROB ** m * (1 - TLS_PROB) ** (nq - m) for m in range(3, nq + 1))
print(f"  {'3 or more':<12}{int(np.sum(k >= 3)):>10}{e3:>25.1f}")
print(f"  day 1 bad qubits: {[int(q) for q in np.where(bad[1])[0]]}")
indep = worst < 3.5 and abs(np.sum(k >= 3) - e3) < 4 * np.sqrt(e3)
print("  VERDICT A:", "independent - day 1 was a rare coincidence" if indep else "NOT independent - investigate")

# ======================= Part B =======================
print("\nPART B: where does the RB offset come from? (qubit 0, nominal dq-5)")
S6 = [np.array(v, dtype=complex) / np.linalg.norm(v) for v in
      ([1, 0], [0, 1], [1, 1], [1, -1], [1, 1j], [1, -1j])]          # 6 states: exact average over all states


def noisy_gate(rho, name, dev):
    reg = _gate_error(_apply(NDensity(1, rho), Op(name, (0,))), Op(name, (0,)), dev)
    if dev.has_decoherence:
        T1 = dev.T1[0] if dev.T1 is not None else float("inf")
        Tp = dev.T_phi[0] if dev.T_phi is not None else None
        reg = reg.channel1(PairNoise().kraus(dev.gate_time_1q, T1, Tp), 0)
    return reg.rho


def exact_epg(dev):
    """Average gate infidelity of h and s, weighted by how often RB's Cliffords use them."""
    counts = {"h": 0, "s": 0}
    for _, seq in CLIFFORDS:
        for g in seq:
            counts[g] += 1
    r = {}
    for g in counts:
        U = gate_matrix(g)
        F = np.mean([np.real(np.vdot(U @ v, noisy_gate(np.outer(v, v.conj()), g, dev) @ (U @ v))) for v in S6])
        r[g] = 1 - F
    return sum(counts[g] * r[g] for g in counts) / sum(counts.values()), r


def fit_fixed_B(ms, ys, B):
    ms, ys = np.asarray(ms, float), np.asarray(ys, float)
    best = None
    for p in np.linspace(0.9, 0.999999, 20000):
        x = p ** ms
        A = np.dot(x, ys - B) / np.dot(x, x)
        err = np.sum((A * x + B - ys) ** 2)
        if best is None or err < best[0]:
            best = (err, p)
    return best[1]


dev1 = single_qubit_device(DQ5, 0)
ex, parts = exact_epg(dev1)
print(f"  exact error per gate (simulator, no approximation): {ex:.3e}   [h {parts['h']:.3e}, s {parts['s']:.3e}]")
print(f"  formula ('built-in expectation'):                   {predicted_epg(DQ5, 0):.3e}"
      f"   -> formula / exact = {predicted_epg(DQ5, 0) / ex:.3f}")
p01, p10 = DQ5.readout_error[0]
B0 = 0.5 * (1 - p01) + 0.5 * p10


def rb_row(label, dev, lengths=(1, 20, 60, 120, 200), n_seq=10):
    res = randomized_benchmarking(dev, 0, lengths=lengths, n_seq=n_seq, seed=0)
    ex_d = exact_epg(single_qubit_device(dev, 0))[0]
    ro = dev.readout_error[0] if dev.readout_error else (0.0, 0.0)
    pB = fit_fixed_B(res["lengths"], res["survival"], 0.5 * (1 - ro[0]) + 0.5 * ro[1])
    epg_B = (1 - pB) / 2 / AVG_GATES
    print(f"  {label:<34}{res['epg'] / ex_d:>9.3f}{epg_B / ex_d:>11.3f}   (free-fit B = {res['B']:.3f})")


print(f"\n  RB measured / exact:              {'free fit':>9}{'fixed B':>11}")
rb_row("standard (as in v0.2)", DQ5)
rb_row("3x more sequences", DQ5, n_seq=30)
rb_row("2x longer sequences", DQ5, lengths=(1, 40, 120, 240, 400))
base = dict(n_qubits=5, gate_time_1q=DQ5.gate_time_1q)
rb_row("gate error only", Device("g", gate_error_1q=DQ5.gate_error_1q, **base))
rb_row("T1 only", Device("t1", T1=DQ5.T1, **base))
rb_row("T_phi only", Device("tp", T_phi=DQ5.T_phi, **base))
print("\n  How to read: a column near 1.000 means that estimate is right.")
print("  If 'free fit' is off but 'fixed B' is ~1, the fit (not the physics) causes the offset.")
print("  If only 'T1 only' is off, energy loss (non-unital noise) is what biases standard RB.")
