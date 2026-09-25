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


def evaluate(device="dq-5", day=None, shots=4000, seed=0):
    """Distance to the exact answer, raw vs readout-mitigated, for every benchmark circuit."""
    from .qpu import QPU
    qpu = QPU(device, day=day)
    rows = []
    for i, (name, qasm) in enumerate(benchmark_suite()):
        ideal = probabilities(parse(qasm), DEVICES["ideal"])
        ideal = {k: v for k, v in ideal.items() if v > 1e-12}
        r = qpu.run(qasm, shots=shots, seed=seed + i).result()
        raw = distribution(r["counts"])
        mit = readout_mitigate(raw, qpu.device, r["clbit_qubits"], r["n_clbits"])
        rows.append({"circuit": name, "raw_tvd": tvd(raw, ideal), "readout_tvd": tvd(mit, ideal)})
    return rows
