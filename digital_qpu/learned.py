"""Learned error mitigation (stage 2).
After readout mitigation, the remaining noise mostly blurs a result toward the uniform distribution:
    p_noisy ~ f * p_ideal + (1 - f) * uniform
where f (the surviving signal) depends on the circuit. A model predicts f from the circuit's error
budget (2-qubit and 1-qubit gate errors, decoherence exposure, ZZ exposure, measured qubits), taken
from the compiled program and the day's calibration; the blur is then undone.
Two models: "linear" (-log f is linear in the budget; interpretable) and "mlp" (digital_qubit's MLP).
Training circuits use a different random seed family than the benchmark suite."""
from itertools import product
import numpy as np
from digital_qubit import MLP, train_mlp
from .qasm import parse
from .device import DEVICES
from .executor import schedule, layer_duration, probabilities
from .compiler import transpile
from .mitigation import readout_mitigate

F_MIN = 0.05
FEATURES = ["2q gate error", "1q gate error", "decoherence", "ZZ exposure", "measured qubits"]


def circuit_features(native, device):
    """Error budget of a compiled program on a (calibrated) device."""
    layers = schedule(native, device)
    T = float(sum(layer_duration(L, device) for L in layers))
    e2 = sum(device.error_2q(*o.qubits) for o in native.ops if len(o.qubits) == 2)
    e1 = sum(device.error_1q(o.qubits[0]) for o in native.ops
             if len(o.qubits) == 1 and o.name != "id" and not (device.virtual_rz and o.name == "rz"))
    decoh = 0.0
    for q in range(native.n_qubits):
        if device.T1 is not None:
            decoh += T / device.T1[q]
        if device.T_phi is not None:
            decoh += T / device.T_phi[q]
    zz = sum((rate * T) ** 2 for _, rate in device.zz_pairs(native.n_qubits))
    return np.array([e2, e1, decoh, zz, len(native.measures)], dtype=float)


def _measured_keys(clbit_qubits, n_clbits):
    cl = sorted(clbit_qubits)
    keys = []
    for bits in product("01", repeat=len(cl)):
        key = ["0"] * n_clbits
        for b, c in zip(bits, cl):
            key[n_clbits - 1 - c] = b
        keys.append("".join(key))
    return keys


def surviving_signal(noisy, ideal, keys):
    """Best f in noisy ~ f*ideal + (1-f)*uniform (least squares over the measured outcomes)."""
    u = 1.0 / len(keys)
    a = np.array([noisy.get(k, 0.0) - u for k in keys])
    b = np.array([ideal.get(k, 0.0) - u for k in keys])
    denom = float(b @ b)
    return 1.0 if denom < 1e-12 else float(np.clip(a @ b / denom, F_MIN, 1.0))


def undo_blur(dist, f, clbit_qubits, n_clbits):
    """Invert the blur: (p - (1-f) u) / f, clipped to >= 0 and renormalized."""
    keys = _measured_keys(clbit_qubits, n_clbits)
    u = 1.0 / len(keys)
    f = float(np.clip(f, F_MIN, 1.0))
    out = {k: max((dist.get(k, 0.0) - (1 - f) * u) / f, 0.0) for k in keys}
    s = sum(out.values())
    return {k: v / s for k, v in out.items() if v > 0} if s > 0 else dict(dist)


def random_training_circuit(rng):
    n = int(rng.integers(2, 6))
    L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
    for _ in range(int(rng.integers(4, 31))):
        if rng.random() < 0.35:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"cx q[{a}], q[{b}];")
        elif rng.random() < 0.5:
            L.append(f"{['h', 'x', 's', 't', 'sx'][int(rng.integers(5))]} q[{int(rng.integers(n))}];")
        else:
            L.append(f"{['rx', 'ry', 'rz'][int(rng.integers(3))]}({rng.uniform(-3, 3):.4f}) q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return " ".join(L)


def training_data(device, n_circuits=120, seed=1000):
    """(features, f*) for random circuits, from exact noisy and exact ideal distributions."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_circuits):
        prog = parse(random_training_circuit(rng))
        native, _ = transpile(prog, device)
        cq = {c: q for q, c in native.measures.items()}
        ncl = max(native.n_clbits, max(native.measures.values()) + 1)
        noisy = readout_mitigate(probabilities(native, device), device, cq, ncl)
        ideal = probabilities(prog, DEVICES["ideal"])
        X.append(circuit_features(native, device))
        y.append(surviving_signal(noisy, ideal, _measured_keys(cq, ncl)))
    return np.array(X), np.array(y)


class LearnedMitigator:
    def __init__(self, kind="linear"):
        if kind not in ("linear", "mlp"):
            raise ValueError("kind must be 'linear' or 'mlp'")
        self.kind = kind

    def fit(self, X, y, seed=0):
        if self.kind == "linear":
            A = np.column_stack([X, np.ones(len(X))])
            self.coef, *_ = np.linalg.lstsq(A, -np.log(y), rcond=None)
        else:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
            Z = (X - self.mu) / self.sd
            r = np.random.default_rng(seed)
            idx = r.permutation(len(y))
            cut = int(0.8 * len(y))
            tr, va = idx[:cut], idx[cut:]
            self.net = train_mlp([X.shape[1], 16, 16, 1], Z[tr], y[tr], Z[va], y[va],
                                 seed=seed, epochs=300, batch=16, verbose=False)
        return self

    def predict_f(self, X):
        X = np.atleast_2d(X)
        if self.kind == "linear":
            f = np.exp(-(np.column_stack([X, np.ones(len(X))]) @ self.coef))
        else:
            f = self.net.predict((X - self.mu) / self.sd)
        return np.clip(f, F_MIN, 1.0)

    def mitigate(self, dist_after_readout, native, device, clbit_qubits, n_clbits):
        f = float(self.predict_f(circuit_features(native, device))[0])
        return undo_blur(dist_after_readout, f, clbit_qubits, n_clbits)


_CACHE = {}


def get_mitigator(device, kind="linear", n_circuits=120, seed=1000):
    """Train once per (device calibration, kind) and reuse."""
    key = (device.name, kind, n_circuits, seed)
    if key not in _CACHE:
        X, y = training_data(device, n_circuits, seed)
        _CACHE[key] = LearnedMitigator(kind).fit(X, y)
    return _CACHE[key]
