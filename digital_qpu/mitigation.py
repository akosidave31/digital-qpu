"""Error mitigation and benchmarking.
Readout mitigation (standard lab technique): each measured qubit's readout error is a known 2x2
confusion matrix from the day's calibration. Applying the inverse of every qubit's matrix undoes the
readout error. Inverting can create small negative "probabilities" from shot noise; these are clipped
to zero and the rest renormalized. Gate errors, decoherence and crosstalk are NOT undone by this."""
import numpy as np
from .qasm import parse
from .device import get_device, DEVICES
from .executor import probabilities


def distribution(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def tvd(p, q):
    """Total variation distance: 0 = identical, 1 = completely different."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def readout_mitigate(dist, device, clbit_qubits, n_clbits):
    """dist: {bitstring: probability} (Qiskit order); clbit_qubits: {clbit: physical qubit}."""
    if device.readout_error is None or not clbit_qubits:
        return dict(dist)
    cl = sorted(clbit_qubits)
    k = len(cl)
    P = np.zeros((2,) * k)
    for key, p in dist.items():
        P[tuple(int(key[n_clbits - 1 - c]) for c in cl)] += p
    for axis, c in enumerate(cl):
        p01, p10 = device.readout_error[clbit_qubits[c]]
        Minv = np.linalg.inv(np.array([[1 - p01, p10], [p01, 1 - p10]]))
        P = np.moveaxis(np.tensordot(Minv, P, axes=([1], [axis])), 0, axis)
    P = np.clip(P, 0, None)
    P = P / P.sum()
    out = {}
    for idx in np.ndindex(*P.shape):
        if P[idx] > 0:
            key = ["0"] * n_clbits
            for b, c in zip(idx, cl):
                key[n_clbits - 1 - c] = str(b)
            out["".join(key)] = float(P[idx])
    return out


def _random_circuit(rng, n, depth):
    L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
    for _ in range(depth):
        if rng.random() < 0.35:
            a = int(rng.integers(n - 1))
            L.append(f"cx q[{a}], q[{a + 1}];")
        else:
            g = ["h", "x", "s", "t", "sx"][int(rng.integers(5))] if rng.random() < 0.5 else \
                f"{['rx', 'ry', 'rz'][int(rng.integers(3))]}({rng.uniform(-3, 3):.4f})"
            L.append(f"{g} q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return " ".join(L)


def benchmark_suite(seed=0):
    """Circuits with exactly known answers: named algorithms plus seeded random circuits."""
    suite = [
        ("bell", "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;"),
        ("ghz3", "OPENQASM 2.0; qreg q[3]; creg c[3]; h q[0]; cx q[0], q[1]; cx q[1], q[2]; measure q -> c;"),
        ("ghz5", "OPENQASM 2.0; qreg q[5]; creg c[5]; h q[0]; cx q[0], q[1]; cx q[1], q[2]; "
                 "cx q[2], q[3]; cx q[3], q[4]; measure q -> c;"),
        ("grover2", "OPENQASM 2.0; qreg q[2]; creg c[2]; h q; cz q[0], q[1]; h q; x q; cz q[0], q[1]; "
                    "x q; h q; measure q -> c;"),
        ("far_bell", "OPENQASM 2.0; qreg q[5]; creg c[5]; h q[0]; cx q[0], q[4]; measure q -> c;"),
    ]
    rng = np.random.default_rng(seed)
    for i in range(5):
        suite.append((f"random{i}", _random_circuit(rng, int(rng.integers(3, 5)), 14)))
    return suite


def shot_noise_floor(ideal, shots, rng, repeats=5):
    """Distance a PERFECT machine would still show, only because of finite shots."""
    keys = list(ideal)
    p = np.array([ideal[k] for k in keys])
    vals = []
    for _ in range(repeats):
        c = rng.multinomial(shots, p / p.sum())
        vals.append(tvd({k: v / shots for k, v in zip(keys, c)}, ideal))
    return float(np.mean(vals))


def evaluate(device="dq-5", day=None, shots=4000, seed=0, learned=True, n_train=120, mitigators=None):
    """Distance to the exact answer for every benchmark circuit: raw, readout-mitigated and
    (optionally) learned mitigation (linear and MLP), plus the shot-noise floor."""
    from .qpu import QPU
    from .compiler import transpile
    qpu = QPU(device, day=day)
    if learned and mitigators is None:
        from .learned import training_data, LearnedMitigator
        X, y = training_data(qpu.device, n_train)
        mitigators = {k: LearnedMitigator(k).fit(X, y) for k in ("linear", "mlp")}
    rng = np.random.default_rng(seed + 12345)
    rows = []
    for i, (name, qasm) in enumerate(benchmark_suite()):
        ideal = probabilities(parse(qasm), DEVICES["ideal"])
        ideal = {k: v for k, v in ideal.items() if v > 1e-12}
        r = qpu.run(qasm, shots=shots, seed=seed + i).result()
        raw = distribution(r["counts"])
        mit = readout_mitigate(raw, qpu.device, r["clbit_qubits"], r["n_clbits"])
        row = {"circuit": name, "raw_tvd": tvd(raw, ideal), "readout_tvd": tvd(mit, ideal),
               "floor": shot_noise_floor(ideal, shots, rng)}
        if learned:
            native, _ = transpile(parse(qasm), qpu.device)
            for k, m in mitigators.items():
                row[f"{k}_tvd"] = tvd(m.mitigate(mit, native, qpu.device, r["clbit_qubits"], r["n_clbits"]), ideal)
        rows.append(row)
    return rows


def evaluate_many(device="dq-5", days=(0, 1, 2, 3, 4), shots=4000, n_train=120):
    """Repeat the benchmark on several calibration days; models are retrained on each day's chip
    and each run uses different shot randomness."""
    return [evaluate(device, day=d, shots=shots, seed=100 * i, n_train=n_train) for i, d in enumerate(days)]


def summarize(runs, a="linear_tvd", b="mlp_tvd", harm_tol=0.002):
    """Means with standard errors over runs, a paired comparison of a vs b, and harm rates
    (how often each learned model is worse than readout mitigation alone, beyond harm_tol)."""
    keys = ["raw_tvd", "readout_tvd", "linear_tvd", "mlp_tvd", "floor"]
    per_run = {k: np.array([np.mean([r[k] for r in rows]) for rows in runs]) for k in keys}
    n = len(runs)
    se = lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    out = {"n_runs": n, "mean": {k: float(v.mean()) for k, v in per_run.items()},
           "se": {k: se(v) for k, v in per_run.items()}}
    d = per_run[a] - per_run[b]
    out["paired"] = {"a": a, "b": b, "mean_diff": float(d.mean()), "se": se(d),
                     "b_better_by_2se": bool(n > 1 and d.mean() > 2 * se(d))}
    out["harm"] = {}
    for k in ("linear_tvd", "mlp_tvd"):
        excess = np.array([r[k] - r["readout_tvd"] for rows in runs for r in rows])
        worse = excess > harm_tol
        out["harm"][k] = {"rate": float(worse.mean()),
                          "mean_excess": float(excess[worse].mean()) if worse.any() else 0.0,
                          "worst": float(excess.max())}
    return out
