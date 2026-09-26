"""Quantum-validation suite for Digital-QPU.

WHAT THIS CHECKS: whether Digital-QPU reproduces the MATHEMATICS of the ideal quantum-circuit model
(state vectors, amplitudes, phases, interference, entanglement, measurement statistics, and the
outputs of textbook algorithms).

WHAT THIS DOES NOT CHECK OR CLAIM: Digital-QPU is a classical program. It stores all 2^n complex
amplitudes in ordinary memory and multiplies them with ordinary arithmetic. Nothing here involves
physical qubits, physical superposition or entanglement, or quantum speedup; its cost grows
exponentially with the number of qubits. A passing test means "the classical simulation computes the
same numbers the quantum-circuit model predicts" - nothing more.

THREE SOURCES FOR EVERY EXACT TEST
  expected   : a hand-written analytic result (or numpy.fft for the QFT)
  reference  : an independent state-vector simulator in this file (own gate matrices, own QASM reader,
               full 2^n x 2^n matrices built with Kronecker products, little-endian index order). It
               imports nothing from digital_qpu or digital_qubit.
  Digital-QPU: the real code path (digital_qpu.parse -> final_state / probabilities / QPU.run).
Limits of that independence: all three are ordinary NumPy code, and the reference was written by the
same author as this suite. The Qiskit cross-check in tests/test_qiskit.py (run in CI) is the most
independent comparison in the project.

TOLERANCES (fixed before the first run, never loosened afterwards)
  exact tests      : |amplitude error| < 1e-10 and |probability error| < 1e-10
  sampling tests   : each outcome within 5 sigma of its exact probability (binomial); an outcome
                     with exact probability 0 must never appear
  fixed seed       : 20260926, 20000 shots

HISTORY (kept on purpose): the first run gave 94/96. Both failures were errors in this file's hand-written
expectations, not in Digital-QPU: SX|+> was written as e^{i pi/4}|+> (correct: |+>, eigenvalue 1) and the
identity X.Y = iZ was coded with phase -i (correct: +i). Digital-QPU and the reference agreed exactly in
both cases. The expectations were corrected; no tolerance was changed.

Run:  python validation/quantum_validation.py            (full report)
      python validation/quantum_validation.py --brief    (one line per test)
Exit code 0 only if every test passes."""
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root

AMP_TOL = 1e-10
PROB_TOL = 1e-10
SIGMAS = 5.0
SEED = 20260926
SHOTS = 20000
R = 1 / math.sqrt(2)

EXACT_AMPS = "mathematical equivalence: exact amplitudes and probabilities (classical simulation)"
EXACT_PROBS = "mathematical equivalence: exact probabilities; amplitudes vs reference (classical simulation)"
STATISTICAL = "statistical: sampled counts consistent with exact probabilities (classical pseudo-random sampling)"
COMPUTED = "computed quantity of the simulated state (not a physical experiment)"

# =====================================================================================================
# Independent reference simulator (no digital_qpu / digital_qubit imports)
# =====================================================================================================
_FIXED = {
    "id": [[1, 0], [0, 1]],
    "x": [[0, 1], [1, 0]],
    "y": [[0, -1j], [1j, 0]],
    "z": [[1, 0], [0, -1]],
    "h": [[R, R], [R, -R]],
    "s": [[1, 0], [0, 1j]],
    "sdg": [[1, 0], [0, -1j]],
    "t": [[1, 0], [0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]],
    "tdg": [[1, 0], [0, complex(math.cos(math.pi / 4), -math.sin(math.pi / 4))]],
    "sx": [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
}


def _ref_matrix(name, theta=None):
    if name in _FIXED:
        return np.array(_FIXED[name], dtype=complex)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[complex(c, -s), 0], [0, complex(c, s)]])
    if name in ("p", "u1"):
        return np.array([[1, 0], [0, complex(math.cos(theta), math.sin(theta))]])
    raise ValueError(f"reference: unknown gate {name}")


def _ref_full_1q(U, q, n):
    """Kronecker product; the first factor is the most significant bit = qubit n-1."""
    full = np.array([[1.0 + 0j]])
    for k in range(n - 1, -1, -1):
        full = np.kron(full, U if k == q else np.eye(2))
    return full


def _ref_full_2q(name, a, b, n):
    N = 2 ** n
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        ba, bb = (i >> a) & 1, (i >> b) & 1
        if name == "cx":
            M[i ^ (1 << b) if ba else i, i] = 1
        elif name == "cz":
            M[i, i] = -1 if (ba and bb) else 1
        elif name == "swap":
            M[i ^ (1 << a) ^ (1 << b) if ba != bb else i, i] = 1
        else:
            raise ValueError(f"reference: unknown gate {name}")
    return M


def ref_parse(text):
    """Minimal OpenQASM 2.0 reader, written independently of digital_qpu.qasm."""
    text = re.sub(r"//[^\n]*", "", text)
    n = ncl = None
    ops, meas = [], {}
    for stmt in text.split(";"):
        s = " ".join(stmt.split())
        if not s or s.startswith("OPENQASM") or s.startswith("include"):
            continue
        m = re.fullmatch(r"qreg q\[(\d+)\]", s)
        if m:
            n = int(m.group(1))
            continue
        m = re.fullmatch(r"creg c\[(\d+)\]", s)
        if m:
            ncl = int(m.group(1))
            continue
        if s == "measure q -> c":
            meas.update({i: i for i in range(n)})
            continue
        m = re.fullmatch(r"measure q\[(\d+)\] -> c\[(\d+)\]", s)
        if m:
            meas[int(m.group(1))] = int(m.group(2))
            continue
        m = re.fullmatch(r"([a-z0-9]+)(?:\(([^)]*)\))? (.+)", s)
        if not m:
            raise ValueError(f"reference: cannot read {s!r}")
        name, par, args = m.group(1), m.group(2), [a.strip() for a in m.group(3).split(",")]
        theta = None if par is None else float(eval(par, {"__builtins__": {}}, {"pi": math.pi}))
        if len(args) == 1 and args[0] == "q":                      # broadcast: h q;
            ops += [(name, theta, (i,)) for i in range(n)]
        else:
            ops.append((name, theta, tuple(int(re.fullmatch(r"q\[(\d+)\]", a).group(1)) for a in args)))
    return n, ncl, ops, meas


def ref_state(text):
    """Little-endian state vector: bit k of the index is qubit k."""
    n, ncl, ops, meas = ref_parse(text)
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1
    for name, theta, qs in ops:
        if len(qs) == 1:
            psi = _ref_full_1q(_ref_matrix(name, theta), qs[0], n) @ psi
        else:
            psi = _ref_full_2q(name, qs[0], qs[1], n) @ psi
    return psi, n, ncl, meas


def probs_from_vector(psi, n, ncl, meas):
    """Measured-outcome probabilities, Qiskit keys (classical bit 0 = rightmost character)."""
    out = {}
    for i, a in enumerate(psi):
        p = abs(a) ** 2
        if p < 1e-15:
            continue
        key = ["0"] * ncl
        for q, c in meas.items():
            key[ncl - 1 - c] = str((i >> q) & 1)
        k = "".join(key)
        out[k] = out.get(k, 0.0) + p
    return out


def ref_probs(text):
    return probs_from_vector(*ref_state(text))


# =====================================================================================================
# Digital-QPU side (the code under test)
# =====================================================================================================
from digital_qpu import parse, probabilities, DEVICES, QPU          # noqa: E402
from digital_qpu.executor import final_state                         # noqa: E402

IDEAL = DEVICES["ideal"]


def dq_state(text):
    """Digital-QPU amplitudes, converted to the reference's little-endian order."""
    prog = parse(text)
    n = prog.n_qubits
    T = np.asarray(final_state(prog, IDEAL).psi).reshape((2,) * n)     # axis q = qubit q
    return T.transpose(tuple(range(n - 1, -1, -1))).reshape(-1)


def dq_probs(text):
    return {k: float(v) for k, v in probabilities(parse(text), IDEAL).items() if v > 1e-15}


def dq_counts(text, shots=SHOTS, seed=SEED):
    return QPU("ideal").run(text, shots=shots, seed=seed).result()["counts"]


# =====================================================================================================
# Circuit builders (QASM text; fed to BOTH simulators)
# =====================================================================================================
def prog(n, body, measures=None, ncl=None):
    ncl = n if ncl is None else ncl
    meas = measures if measures is not None else [(i, i) for i in range(n)]
    return "\n".join(["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{ncl}];"] + body +
                     [f"measure q[{q}] -> c[{c}];" for q, c in meas]) + "\n"


def cp(theta, a, b):
    """Controlled phase diag(1,1,1,e^{i theta}) from p and cx (exact, no global phase)."""
    return [f"p({theta / 2!r}) q[{a}];", f"cx q[{a}],q[{b}];", f"p({-theta / 2!r}) q[{b}];",
            f"cx q[{a}],q[{b}];", f"p({theta / 2!r}) q[{b}];"]


def mcz(qs):
    """Multi-controlled Z on qubits qs (phase -1 on |1...1>), exactly, from parity phases:
    x1*...*xk = 2^-(k-1) * sum over non-empty subsets S of (-1)^(|S|-1) * parity(S)."""
    k = len(qs)
    if k == 1:
        return [f"z q[{qs[0]}];"]
    body = []
    for mask in range(1, 2 ** k):
        S = [qs[i] for i in range(k) if mask >> i & 1]
        alpha = math.pi * (-1) ** (len(S) - 1) / 2 ** (k - 1)
        tgt = S[-1]
        body += [f"cx q[{s}],q[{tgt}];" for s in S[:-1]]
        body.append(f"p({alpha!r}) q[{tgt}];")
        body += [f"cx q[{s}],q[{tgt}];" for s in reversed(S[:-1])]
    return body


def grover(n, marked, iterations):
    qs = list(range(n))
    body = [f"h q[{q}];" for q in qs]
    flips = [f"x q[{q}];" for q in qs if not marked >> q & 1]
    for _ in range(iterations):
        body += flips + mcz(qs) + flips                                   # oracle
        body += [f"h q[{q}];" for q in qs] + [f"x q[{q}];" for q in qs]   # diffusion
        body += mcz(qs) + [f"x q[{q}];" for q in qs] + [f"h q[{q}];" for q in qs]
    return prog(n, body)


def qft_body(n):
    """Textbook QFT, little-endian: |x> -> N^-1/2 sum_y exp(2 pi i x y / N) |y>."""
    body = []
    for j in reversed(range(n)):
        body.append(f"h q[{j}];")
        for k in reversed(range(j)):
            body += cp(math.pi / 2 ** (j - k), k, j)
    for i in range(n // 2):
        body.append(f"swap q[{i}],q[{n - 1 - i}];")
    return body


# =====================================================================================================
# Checks and records
# =====================================================================================================
RESULTS = []


def ket(i, n):
    return format(i, f"0{n}b")


def fmt_amps(v, n, top=4):
    idx = [i for i in np.argsort(-np.abs(v)) if abs(v[i]) > 1e-9][:top]
    parts = []
    for i in sorted(idx):
        a = v[i]
        if abs(a.imag) < 1e-9:
            c = f"{a.real:+.4f}"
        elif abs(a.real) < 1e-9:
            c = f"{a.imag:+.4f}i"
        else:
            c = f"({a.real:+.4f}{a.imag:+.4f}i)"
        parts.append(f"{c}|{ket(i, n)}>")
    more = sum(abs(x) > 1e-9 for x in v) - len(idx)
    return " ".join(parts) + (f" (+{more} more)" if more > 0 else "")


def fmt_probs(p, top=4):
    items = sorted(p.items(), key=lambda kv: -kv[1])[:top]
    more = len([v for v in p.values() if v > 1e-12]) - len(items)
    return " ".join(f"P({k})={v:.4f}" for k, v in sorted(items)) + (f" (+{more} more)" if more > 0 else "")


def record(section, name, circuit, expected, dq, ref, amp_err, prob_err, passed, kind, note=""):
    RESULTS.append(dict(section=section, name=name, circuit=circuit, expected=expected, dq=dq, ref=ref,
                        amp_err=amp_err, prob_err=prob_err, passed=bool(passed), kind=kind, note=note))


def max_prob_diff(a, b):
    return max([abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)] + [0.0])


def check_exact(section, name, circuit, text, expected_amps=None, expected_probs=None, expected_txt="",
                note=""):
    """Exact comparison: Digital-QPU vs independent reference vs analytic expectation."""
    dq = dq_state(text)
    rv, n, ncl, meas = ref_state(text)
    amp_err = float(np.max(np.abs(dq - rv)))
    if expected_amps is not None:
        e = np.asarray(expected_amps, dtype=complex)
        amp_err = max(amp_err, float(np.max(np.abs(dq - e))), float(np.max(np.abs(rv - e))))
        expected_probs = probs_from_vector(e, n, ncl, meas)
    norm_err = abs(float(np.linalg.norm(dq)) - 1.0)
    dqp, rp = dq_probs(text), probs_from_vector(rv, n, ncl, meas)
    prob_err = max(max_prob_diff(dqp, rp), max_prob_diff(dqp, expected_probs), max_prob_diff(rp, expected_probs),
                   abs(sum(dqp.values()) - 1.0), norm_err)
    passed = amp_err < AMP_TOL and prob_err < PROB_TOL
    kind = EXACT_AMPS if expected_amps is not None else EXACT_PROBS
    dq_txt = fmt_amps(dq, n) if expected_amps is not None else fmt_probs(dqp)
    ref_txt = fmt_amps(rv, n) if expected_amps is not None else fmt_probs(rp)
    if not expected_txt:
        expected_txt = fmt_amps(np.asarray(expected_amps), n) if expected_amps is not None else fmt_probs(expected_probs)
    record(section, name, circuit, expected_txt, dq_txt, ref_txt, amp_err, prob_err, passed, kind, note)
    return dq, rv, dqp


def check_sampling(section, name, circuit, text, expected_probs, shots=SHOTS, seed=SEED, note=""):
    counts = dq_counts(text, shots, seed)
    rp = ref_probs(text)
    worst_sigma, prob_err, ok = 0.0, 0.0, True
    for k in set(counts) | set(expected_probs):
        p, phat = expected_probs.get(k, 0.0), counts.get(k, 0) / shots
        prob_err = max(prob_err, abs(phat - p))
        if p < 1e-12:
            ok &= counts.get(k, 0) == 0
        else:
            sig = math.sqrt(p * (1 - p) / shots)
            worst_sigma = max(worst_sigma, abs(phat - p) / sig if sig > 0 else 0.0)
            ok &= abs(phat - p) <= SIGMAS * sig + 1e-12
    ok &= max_prob_diff(rp, expected_probs) < PROB_TOL
    measured = {k: v / shots for k, v in counts.items()}
    record(section, name, circuit, fmt_probs(expected_probs), fmt_probs(measured) + f"  [{shots} shots, seed {seed}]",
           fmt_probs(rp), None, prob_err, ok, STATISTICAL,
           (note + "; " if note else "") + f"worst deviation {worst_sigma:.2f} sigma (limit {SIGMAS:g})")
    return counts


# =====================================================================================================
# 1. Single-qubit validation
# =====================================================================================================
def section_single_qubit():
    S = "1 single-qubit"
    e4 = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    cases = {   # gate: (on |0>, on |1>, on |+>) - hand-written textbook results, little-endian [a0, a1]
        "x": ([0, 1], [1, 0], [R, R]),
        "y": ([0, 1j], [-1j, 0], [-1j * R, 1j * R]),
        "z": ([1, 0], [0, -1], [R, -R]),
        "h": ([R, R], [R, -R], [1, 0]),
        "s": ([1, 0], [0, 1j], [R, 1j * R]),
        "t": ([1, 0], [0, e4], [R, e4 * R]),
        "sdg": ([1, 0], [0, -1j], [R, -1j * R]),
        "tdg": ([1, 0], [0, e4.conjugate()], [R, e4.conjugate() * R]),
        "sx": ([0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j], [R, R]),   # |+> is an eigenvector (eigenvalue 1)
    }
    for g, outs in cases.items():
        for prep, label, e in (("", "|0>", outs[0]), ("x q[0];", "|1>", outs[1]), ("h q[0];", "|+>", outs[2])):
            check_exact(S, f"{g.upper()} on {label}", f"{prep} {g} q[0]".strip(), prog(1, [prep, f"{g} q[0];"] if prep else [f"{g} q[0];"]),
                        expected_amps=e, note="phase-only differences are visible in amplitudes, not in probabilities"
                        if g in ("z", "s", "t", "sdg", "tdg") and label != "|+>" else "")
    th = 0.7
    c, s = math.cos(th / 2), math.sin(th / 2)
    rot = {"rx": [c, -1j * s], "ry": [c, s], "rz": [complex(c, -s), 0]}
    for g, e in rot.items():
        check_exact(S, f"{g.upper()}({th}) on |0>", f"{g}({th}) q[0]", prog(1, [f"{g}({th}) q[0];"]), expected_amps=e,
                    note="RZ convention exp(-i theta Z/2) (Qiskit); global phase included" if g == "rz" else "")
    check_exact(S, f"P({th}) on |+>", f"h q[0]; p({th}) q[0]", prog(1, ["h q[0];", f"p({th}) q[0];"]),
                expected_amps=[R, R * complex(math.cos(th), math.sin(th))])

    # normalization and probability conservation on longer random circuits (fixed seed)
    rng = np.random.default_rng(SEED)
    for n, depth in ((1, 60), (3, 80), (5, 120)):
        body = []
        for _ in range(depth):
            r = rng.random()
            if n > 1 and r < 0.3:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False))
                body.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
            elif r < 0.6:
                body.append(f"{['rx', 'ry', 'rz', 'p'][int(rng.integers(4))]}({rng.uniform(-3, 3)!r}) q[{int(rng.integers(n))}];")
            else:
                body.append(f"{['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'sx'][int(rng.integers(9))]} q[{int(rng.integers(n))}];")
        text = prog(n, body)
        dq = dq_state(text)
        rv = ref_state(text)[0]
        norm_err = abs(float(np.linalg.norm(dq)) - 1)
        tot_err = abs(sum(dq_probs(text).values()) - 1)
        amp_err = float(np.max(np.abs(dq - rv)))
        record(S, f"normalization + probability conservation ({n} qubits, {depth} random gates)",
               f"random circuit, seed {SEED}", "||psi|| = 1, sum of probabilities = 1, same state as reference",
               f"| ||psi||-1 | = {norm_err:.1e}, |sum P - 1| = {tot_err:.1e}", fmt_amps(rv, n, 2),
               amp_err, max(norm_err, tot_err), max(norm_err, tot_err, amp_err) < AMP_TOL, EXACT_AMPS)

    # gate identities on a generic input state (so no identity can pass by accident on |0>)
    prep = ["ry(0.7) q[0];", "rz(1.3) q[0];"]
    identities = [   # (name, sequence A in time order, sequence B, phase such that A = phase * B)
        ("H.H = I", ["h", "h"], [], 1), ("X.X = I", ["x", "x"], [], 1), ("Y.Y = I", ["y", "y"], [], 1),
        ("Z.Z = I", ["z", "z"], [], 1), ("S.S = Z", ["s", "s"], ["z"], 1), ("T.T = S", ["t", "t"], ["s"], 1),
        ("S.Sdg = I", ["s", "sdg"], [], 1), ("T.Tdg = I", ["t", "tdg"], [], 1), ("SX.SX = X", ["sx", "sx"], ["x"], 1),
        ("H.Z.H = X", ["h", "z", "h"], ["x"], 1), ("H.X.H = Z", ["h", "x", "h"], ["z"], 1),
        ("H.Y.H = -Y", ["h", "y", "h"], ["y"], -1), ("X.Y = iZ  (Y first, then X)", ["y", "x"], ["z"], 1j),
    ]
    for name, A, B, ph in identities:
        tA = prog(1, prep + [f"{g} q[0];" for g in A])
        tB = prog(1, prep + [f"{g} q[0];" for g in B])
        dA, dB = dq_state(tA), dq_state(tB)
        rA, rB = ref_state(tA)[0], ref_state(tB)[0]
        err = max(float(np.max(np.abs(dA - ph * dB))), float(np.max(np.abs(dA - rA))),
                  float(np.max(np.abs(dB - rB))), float(np.max(np.abs(rA - ph * rB))))
        perr = max_prob_diff(dq_probs(tA), dq_probs(tB))
        record(S, f"identity {name}", "ry(0.7) rz(1.3) then " + " ".join(A) + "  vs  " + (" ".join(B) or "nothing"),
               f"state A = {ph} x state B (exactly, including phase)", fmt_amps(dA, 1), fmt_amps(rA, 1),
               err, perr, err < AMP_TOL and perr < PROB_TOL, EXACT_AMPS)


# =====================================================================================================
# 2. Interference validation
# =====================================================================================================
def section_interference():
    S = "2 interference"
    cases = [   # (name, preparation, gates, expected amps) - hand-derived
        ("H->H = I on |0>", "", "h h", [1, 0]),
        ("H->X->H = Z on |0>", "", "h x h", [1, 0]),
        ("H->X->H = Z on |1>", "x", "h x h", [0, -1]),
        ("H->Z->H = X on |0>", "", "h z h", [0, 1]),
        ("H->Z->H = X on |1>", "x", "h z h", [1, 0]),
        ("H->Z->H->Z on |0>", "", "h z h z", [0, -1]),
        ("H->Z->H->Z on |1>", "x", "h z h z", [1, 0]),
        ("H->S->S->H = X on |0>", "", "h s s h", [0, 1]),
        ("H->T->T->T->T->H = X on |0>", "", "h t t t t h", [0, 1]),
        ("H->S->H on |0>", "", "h s h", [0.5 + 0.5j, 0.5 - 0.5j]),
    ]
    for name, prep, gates, e in cases:
        body = ([f"{prep} q[0];"] if prep else []) + [f"{g} q[0];" for g in gates.split()]
        check_exact(S, name, (prep + " " if prep else "") + gates, prog(1, body), expected_amps=e,
                    note="destructive interference: the other outcome's amplitude cancels to 0" if 0 in e else "")
    # Mach-Zehnder phase sweep: H P(phi) H |0> = ((1+e^{i phi})|0> + (1-e^{i phi})|1>)/2, P(0) = cos^2(phi/2)
    worst_a, worst_p, rows = 0.0, 0.0, []
    for k in range(16):
        phi = 2 * math.pi * k / 16
        e = np.array([(1 + np.exp(1j * phi)) / 2, (1 - np.exp(1j * phi)) / 2])
        text = prog(1, ["h q[0];", f"p({phi!r}) q[0];", "h q[0];"])
        dq, rv = dq_state(text), ref_state(text)[0]
        worst_a = max(worst_a, float(np.max(np.abs(dq - e))), float(np.max(np.abs(rv - e))))
        worst_p = max(worst_p, abs(dq_probs(text).get("0", 0.0) - math.cos(phi / 2) ** 2))
        rows.append(abs(dq[0]) ** 2)
    record(S, "phase sweep H->P(phi)->H, 16 angles 0..2pi", "h p(phi) h, phi = 2 pi k/16",
           "P(0) = cos^2(phi/2) for every phi; amplitudes (1 +/- e^{i phi})/2",
           "P(0) at phi=0, pi/2, pi, 3pi/2: " + ", ".join(f"{rows[i]:.4f}" for i in (0, 4, 8, 12)),
           "same formula (reference simulator)", worst_a, worst_p, worst_a < AMP_TOL and worst_p < PROB_TOL, EXACT_AMPS)
    check_sampling(S, "H->H measured 20000 times (interference is deterministic)", "h h",
                   prog(1, ["h q[0];", "h q[0];"]), {"0": 1.0}, note="a classical coin flip twice would give 50/50")


# =====================================================================================================
# 3. Bell state
# =====================================================================================================
def section_bell():
    S = "3 Bell state"
    bell = prog(2, ["h q[0];", "cx q[0],q[1];"])
    check_exact(S, "H(q0) then CNOT(q0,q1) on |00>", "h q0; cx q0,q1", bell, expected_amps=[R, 0, 0, R],
                expected_txt="(|00> + |11>)/sqrt(2)")
    counts = check_sampling(S, "Bell measurement statistics", "h q0; cx q0,q1; measure",
                            bell, {"00": 0.5, "11": 0.5})
    same = sum(v for k, v in counts.items() if k[0] == k[1]) / sum(counts.values())
    record(S, "perfect correlation in the Z basis (sampled)", "Bell, 20000 shots",
           "bit0 == bit1 in every shot (fraction 1.0)", f"fraction equal = {same:.6f}", "exact: 1.0",
           None, abs(same - 1), same == 1.0, STATISTICAL)
    xb = prog(2, ["h q[0];", "cx q[0],q[1];", "h q[0];", "h q[1];"])
    rho_mix = 0.5 * (np.outer([1, 0, 0, 0], [1, 0, 0, 0]) + np.outer([0, 0, 0, 1], [0, 0, 0, 1]))
    HH = np.kron(_ref_matrix("h"), _ref_matrix("h"))
    mix_p = np.real(np.diag(HH @ rho_mix @ HH.conj().T))
    check_exact(S, "correlation in the X basis (entanglement signature)", "Bell, then H on both, measure", xb,
                expected_probs={"00": 0.5, "11": 0.5},
                note=f"a classical 50/50 mixture of |00> and |11> would give P = {', '.join(f'{p:.2f}' for p in mix_p)} "
                     "here (uncorrelated); the superposition keeps 00/11 only")
    # CHSH value of the simulated state from exact probabilities (Digital-QPU and reference)
    def E(a, b, f):
        t = prog(2, ["h q[0];", "cx q[0],q[1];", f"ry({-a!r}) q[0];", f"ry({-b!r}) q[1];"])
        P = f(t)
        return P.get("00", 0) + P.get("11", 0) - P.get("01", 0) - P.get("10", 0)
    a, a2, b, b2 = 0.0, math.pi / 2, math.pi / 4, -math.pi / 4
    chsh = lambda f: E(a, b, f) + E(a, b2, f) + E(a2, b, f) - E(a2, b2, f)
    sd, sr = chsh(dq_probs), chsh(ref_probs)
    record(S, "CHSH value of the simulated Bell state", "E(a,b) from exact probabilities, a=0, pi/2; b=+/-pi/4",
           "2*sqrt(2) = 2.828427 (quantum prediction; local-hidden-variable bound is 2)", f"{sd:.6f}", f"{sr:.6f}",
           None, abs(sd - 2 * math.sqrt(2)), abs(sd - 2 * math.sqrt(2)) < PROB_TOL and abs(sr - sd) < PROB_TOL, COMPUTED,
           "this reproduces the quantum PREDICTION; a classical computer cannot perform a physical Bell test")


# =====================================================================================================
# 4. Bernstein-Vazirani and 5. Deutsch-Jozsa
# =====================================================================================================
def bv_program(secret, n):
    body = [f"x q[{n}];"] + [f"h q[{i}];" for i in range(n + 1)]
    body += [f"cx q[{i}],q[{n}];" for i in range(n) if secret >> i & 1]
    body += [f"h q[{i}];" for i in range(n)]
    return prog(n + 1, body, [(i, i) for i in range(n)], ncl=n)


def section_bv():
    S = "4 Bernstein-Vazirani"
    for secret, n in ((0b000, 3), (0b101, 3), (0b011, 3), (0b111, 3), (0b1011, 4), (0b10110, 5)):
        e = np.zeros(2 ** (n + 1), dtype=complex)
        e[secret], e[secret | 1 << n] = R, -R                    # |secret> (x) |->  on the ancilla
        check_exact(S, f"hidden string {ket(secret, n)} (n={n})", f"BV oracle for s={ket(secret, n)}, one query",
                    bv_program(secret, n), expected_amps=e,
                    expected_txt=f"|{ket(secret, n)}> (x) |-> ; P({ket(secret, n)}) = 1 from ONE oracle query")
    from digital_qpu import bernstein_vazirani
    alg = bernstein_vazirani()
    check_exact(S, "existing implementation digital_qpu.bernstein_vazirani()", "project's own BV circuit, s=101",
                alg["qasm"], expected_probs={"101": 1.0})
    check_sampling(S, "BV s=101 sampled", "own BV circuit", bv_program(0b101, 3), {"101": 1.0})


def dj_program(kind, n=3):
    body = [f"x q[{n}];"] + [f"h q[{i}];" for i in range(n + 1)]
    if kind == "constant 1":
        body.append(f"x q[{n}];")
    elif kind.startswith("balanced"):
        s = int(kind.split("s=")[1], 2)
        body += [f"cx q[{i}],q[{n}];" for i in range(n) if s >> i & 1]
    body += [f"h q[{i}];" for i in range(n)]
    return prog(n + 1, body, [(i, i) for i in range(n)], ncl=n)


def section_dj():
    S = "5 Deutsch-Jozsa"
    for kind in ("constant 0", "constant 1", "balanced s=001", "balanced s=110", "balanced s=111"):
        if kind.startswith("constant"):
            exp = {"000": 1.0}
            txt = "P(000) = 1 (constant)"
        else:
            s = kind.split("s=")[1]
            exp = {s: 1.0}
            txt = f"P(000) = 0 (balanced); for f(x) = s.x the output is exactly |{s}>"
        check_exact(S, kind, f"DJ oracle: {kind}", dj_program(kind), expected_probs=exp, expected_txt=txt)
    from digital_qpu import deutsch_jozsa
    for kind in ("balanced", "constant"):
        text = deutsch_jozsa(kind)["qasm"]
        dqp, rp = dq_probs(text), ref_probs(text)
        p0 = dqp.get("000", 0.0)
        target = 1.0 if kind == "constant" else 0.0
        err = max(abs(p0 - target), max_prob_diff(dqp, rp), abs(sum(dqp.values()) - 1))
        record(S, f"existing implementation deutsch_jozsa('{kind}')", "project's own DJ circuit",
               f"P(000) = {target:g}", fmt_probs(dqp), fmt_probs(rp), None, err, err < PROB_TOL, EXACT_PROBS)
    check_sampling(S, "DJ constant sampled", "constant 0", dj_program("constant 0"), {"000": 1.0})


# =====================================================================================================
# 6. Grover search
# =====================================================================================================
def section_grover():
    S = "6 Grover"
    for n, marked, k in ((2, 0b10, 1), (3, 0b101, 2), (4, 0b1011, 3)):
        N = 2 ** n
        th = math.asin(1 / math.sqrt(N))
        pm = math.sin((2 * k + 1) * th) ** 2
        exp = {ket(i, n): (pm if i == marked else (1 - pm) / (N - 1)) for i in range(N)}
        check_exact(S, f"{n} qubits, marked {ket(marked, n)}, {k} iteration(s)", f"Grover n={n}, k={k}",
                    grover(n, marked, k), expected_probs=exp,
                    expected_txt=f"P({ket(marked, n)}) = sin^2({2 * k + 1} asin(1/sqrt({N}))) = {pm:.6f} "
                                 f"(uniform would be {1 / N:.4f})")
    from digital_qpu import grover3
    text = grover3()["qasm"]
    pm = math.sin(5 * math.asin(1 / math.sqrt(8))) ** 2
    exp = {ket(i, 3): (pm if i == 0b101 else (1 - pm) / 7) for i in range(8)}
    check_exact(S, "existing implementation digital_qpu.grover3()", "project's own Grover circuit (Toffoli-based)",
                text, expected_probs=exp)
    check_sampling(S, "Grover 3 qubits sampled", "Grover n=3, k=2", grover(3, 0b101, 2),
                   {ket(i, 3): (pm if i == 0b101 else (1 - pm) / 7) for i in range(8)})


# =====================================================================================================
# 7. Quantum Fourier Transform
# =====================================================================================================
def section_qft():
    S = "7 QFT"
    for n, x in ((2, 1), (2, 2), (2, 3), (3, 5), (4, 11)):
        N = 2 ** n
        e = np.array([np.exp(2j * math.pi * x * y / N) / math.sqrt(N) for y in range(N)])
        body = [f"x q[{i}];" for i in range(n) if x >> i & 1] + qft_body(n)
        check_exact(S, f"QFT|{x}> on {n} qubits", f"prepare |{ket(x, n)}>, QFT", prog(n, body), expected_amps=e,
                    expected_txt=f"N^-1/2 sum_y exp(2 pi i {x} y / {N}) |y>")
    n, N = 3, 8
    angles = [0.4, 1.1, 2.3]
    single = [np.array([math.cos(a / 2), math.sin(a / 2)]) for a in angles]
    inp = np.array([1.0 + 0j])
    for q in range(n - 1, -1, -1):
        inp = np.kron(inp, single[q])
    e = np.fft.ifft(inp) * math.sqrt(N)                  # numpy's FFT as an independent reference
    body = [f"ry({a!r}) q[{q}];" for q, a in enumerate(angles)] + qft_body(n)
    check_exact(S, "QFT of a superposition (3 qubits)", "ry(0.4), ry(1.1), ry(2.3), then QFT", prog(n, body),
                expected_amps=e, expected_txt="sqrt(N) * numpy.fft.ifft(input amplitudes)")
    from digital_qpu import phase_estimation
    check_exact(S, "phase estimation with the project's inverse QFT", "existing phase_estimation(k=5, n=3)",
                phase_estimation()["qasm"], expected_probs={"101": 1.0},
                expected_txt="phase 5/8 read exactly as 101: P(101) = 1")


# =====================================================================================================
# 8. Shor (N = 15, a = 7)
# =====================================================================================================
def classical_period_pipeline(y, N=15, a=7, t=4):
    """Written independently of digital_qpu.algorithms.shor_factors: continued fractions -> r -> gcd."""
    if y == 0:
        return None, None
    r = Fraction(y, 2 ** t).limit_denominator(N).denominator
    if r % 2 or pow(a, r, N) != 1:
        return r, None
    f = math.gcd(pow(a, r // 2) - 1, N)
    return r, (sorted((f, N // f)) if 1 < f < N else None)


def section_shor():
    S = "8 Shor N=15"
    from digital_qpu import shor15, shor_factors
    text = shor15()["qasm"]
    exp = {"0000": 0.25, "0100": 0.25, "1000": 0.25, "1100": 0.25}
    _, _, dqp = check_exact(S, "period-finding register, a=7 (8 qubits)", "project's shor15() circuit",
                            text, expected_probs=exp,
                            expected_txt="period r=4 -> y in {0,4,8,12}, each 1/4",
                            note="the circuit is 'compiled' for this case: it uses 7^4 = 1 mod 15 to omit gates, "
                                 "i.e. knowledge of the answer is built in (a known caveat of small Shor demos)")
    succ, rows = 0.0, []
    for k, p in sorted(dqp.items()):
        r, f = classical_period_pipeline(int(k, 2))
        rows.append(f"y={int(k, 2)}: r={r}, factors={f}")
        if f == [3, 5]:
            succ += p
    record(S, "classical post-processing of the exact distribution", "continued fractions + gcd on each y",
           "y=4,12 -> r=4 -> 15 = 3 x 5; y=0,8 give no factors; success probability 0.5",
           "; ".join(rows) + f"; success = {succ:.4f}", "exact: 0.5", None, abs(succ - 0.5),
           abs(succ - 0.5) < PROB_TOL, EXACT_PROBS)
    counts = check_sampling(S, "Shor sampled", "shor15(), 20000 shots", text, exp)
    r, f = shor_factors(counts)
    r2 = [classical_period_pipeline(int(k, 2)) for k in sorted(counts, key=counts.get, reverse=True)]
    mine = next((x for x in r2 if x[1]), (None, None))
    record(S, "end-to-end pipeline on sampled counts", "project's shor_factors() vs this suite's pipeline",
           "both find r=4 and 15 = 3 x 5", f"project: r={r}, factors={f}", f"suite: r={mine[0]}, factors={mine[1]}",
           None, None, f == [3, 5] and mine[1] == [3, 5] and r == 4, EXACT_PROBS,
           "small-N factoring is a correctness demo, NOT quantum advantage: 15 is factored classically in microseconds")


# =====================================================================================================
# Report
# =====================================================================================================
SECTIONS = [section_single_qubit, section_interference, section_bell, section_bv, section_dj, section_grover,
            section_qft, section_shor]


def run_all():
    RESULTS.clear()
    for f in SECTIONS:
        f()
    return RESULTS


def _e(x):
    return "n/a" if x is None else f"{x:.1e}"


def report(results, brief=False, out=sys.stdout):
    w = lambda s="": print(s, file=out)
    w("Digital-QPU quantum-validation suite")
    w("Classical simulation on classical hardware: this checks MATHEMATICAL agreement with the ideal")
    w("quantum-circuit model. It does not demonstrate physical quantum behaviour or any speedup.")
    w(f"Tolerances: exact < {AMP_TOL:g} (amplitudes) / < {PROB_TOL:g} (probabilities); sampling within "
      f"{SIGMAS:g} sigma; seed {SEED}; {SHOTS} shots")
    sec = None
    for i, r in enumerate(results, 1):
        if r["section"] != sec:
            sec = r["section"]
            w()
            w(f"== {sec} ==")
        mark = "PASS" if r["passed"] else "FAIL"
        if brief:
            w(f"[{mark}] {r['name']}  (amp err {_e(r['amp_err'])}, prob err {_e(r['prob_err'])})")
            continue
        w(f"[{mark}] {r['name']}")
        w(f"   circuit/input : {r['circuit']}")
        w(f"   expected      : {r['expected']}")
        w(f"   Digital-QPU   : {r['dq']}")
        w(f"   reference     : {r['ref']}")
        w(f"   abs error {_e(r['amp_err'])} | prob error {_e(r['prob_err'])}")
        w(f"   demonstrates  : {r['kind']}")
        if r["note"]:
            w(f"   note          : {r['note']}")
    w()
    w("== summary ==")
    by = {}
    for r in results:
        by.setdefault(r["section"], []).append(r["passed"])
    for s, v in by.items():
        w(f"  {s:<24} {sum(v):>3}/{len(v):<3} passed")
    total, passed = len(results), sum(r["passed"] for r in results)
    w(f"  {'TOTAL':<24} {passed:>3}/{total:<3} passed" + ("" if passed == total else "   <-- FAILURES ABOVE"))
    w()
    w("Verified here (as mathematics, by classical simulation, against an independent reference):")
    for line in VERIFIED:
        w(f"  + {line}")
    w("NOT verified or demonstrated:")
    for line in NOT_VERIFIED:
        w(f"  - {line}")


VERIFIED = [
    "gate action of X, Y, Z, H, S, T, Sdg, Tdg, SX, RX, RY, RZ, P on |0>, |1>, |+>, including phases",
    "state normalization and probability conservation (random circuits up to 5 qubits)",
    "gate identities including relative/global phase (HH=I, SS=Z, TT=S, HZH=X, HYH=-Y, XY=iZ, ...)",
    "single-qubit interference: cancellation of amplitudes, P(0) = cos^2(phi/2) phase sweep",
    "Bell state amplitudes; perfect Z-basis correlation in sampling; X-basis correlation that a classical "
    "mixture would not show; CHSH value 2*sqrt(2) of the simulated state",
    "Bernstein-Vazirani (n = 3, 4, 5) and Deutsch-Jozsa (constant/balanced) outputs, one oracle query",
    "Grover amplification on 2, 3, 4 qubits matching sin^2((2k+1) theta)",
    "QFT amplitudes on basis and superposition inputs (vs analytic formula and numpy FFT); phase estimation",
    "Shor N=15, a=7: exact period-finding distribution, classical post-processing, sampled end-to-end run",
    "sampling: measured frequencies consistent with exact probabilities; impossible outcomes never sampled",
]
NOT_VERIFIED = [
    "any PHYSICAL quantum behaviour: amplitudes are numbers in classical memory, not qubits",
    "quantum speedup or advantage: simulation cost grows as 2^n; nothing here is faster than classical",
    "Bell nonlocality: a computed CHSH value is a prediction, not a physical Bell-inequality experiment",
    "true randomness: measurement outcomes come from a seeded pseudo-random generator",
    "realism of the noisy chips (dq-5, dq-12) against real hardware data - this suite uses the ideal device",
    "Shor beyond N=15, a=7; the circuit uses knowledge of the period (compiled demo), and small-N factoring "
    "is not evidence of advantage",
    "behaviour beyond ~20 qubits (ideal) / 16 noisy qubits, mid-circuit measurement, classical control",
    "full independence of the reference: same author, same NumPy; the Qiskit cross-check (tests/test_qiskit.py) "
    "is the most independent comparison",
]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results = run_all()
    report(results, brief="--brief" in argv)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
