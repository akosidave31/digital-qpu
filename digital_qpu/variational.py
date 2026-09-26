"""Trainable Grover circuit (a variational quantum algorithm), trained on the simulated chips.

Idea: take the 3-qubit Grover circuit of algorithms.grover3() and keep its two-qubit structure exactly
(the oracle and the CCZ inside the diffusion stay fixed), but replace every Hadamard/X layer with a
TRAINABLE layer: an arbitrary rotation rz-ry-rz on each qubit. Grover itself is one point in this
parameter space (grover_init), so training can only start from it and look for something better:
on the ideal chip, better single-qubit angles; on a noisy chip, angles that suffer less from its
particular noise.

Training: success = exact probability of the marked answer (no shot noise), gradient by the
parameter-shift rule, Adam optimizer.

Honest limits:
- The reward needs the marked item, so this does NOT search better. It learns the best circuit for
  a known task on a given (simulated) chip - noise-aware circuit optimisation.
- Parameter shift is exact on the ideal chip. On a noisy chip it is approximate: the compiler turns
  single-qubit gates into sx pulses, and how many depends on the angle, so the noise itself changes
  slightly with the parameters.
- Everything runs on a classical simulator; nothing here is quantum speedup."""
import math
import time
import numpy as np
from .qasm import parse
from .compiler import transpile
from .executor import probabilities
from .algorithms import _ccz, _program, grover3
from .device import get_device
from .calibration import calibrate

SHIFT = math.pi / 2


def zyz(U):
    """Angles (lam, theta, phi) with U = e^{i a} rz(phi) ry(theta) rz(lam) (rz = exp(-i angle Z/2)).
    In time order the qubit gets rz(lam), then ry(theta), then rz(phi)."""
    U = np.asarray(U, dtype=complex)
    U = U / np.sqrt(np.linalg.det(U))
    a, c = U[0, 0], U[1, 0]
    theta = 2 * math.atan2(abs(c), abs(a))
    s = -2 * np.angle(a) if abs(a) > 1e-12 else 0.0          # phi + lam
    d = 2 * np.angle(c) if abs(c) > 1e-12 else 0.0           # phi - lam
    return float((s - d) / 2), float(theta), float((s + d) / 2)


def rot(lam, theta, phi):
    """The 2x2 matrix rz(phi) ry(theta) rz(lam) (for tests and init)."""
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return rz(phi) @ np.array([[c, -s], [s, c]], dtype=complex) @ rz(lam)


_H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]])


class VariationalGrover:
    """Grover-shaped circuit on 3 qubits with trainable single-qubit layers.
    rounds: number of oracle + diffusion rounds (Grover's best for 8 items is 2)."""

    def __init__(self, marked=0b101, rounds=2):
        self.n, self.marked, self.rounds = 3, marked, rounds
        self.key = format(marked, "03b")
        self.items = []          # fixed QASM lines (str) or trainable gates (name, qubit, param index)
        self.n_params = 0
        self._layer()                                            # replaces the opening H layer
        flips = [f"x q[{i}];" for i in range(3) if not marked >> i & 1]
        for _ in range(rounds):
            self.items += flips + _ccz(0, 1, 2) + flips          # oracle: fixed (it defines the task)
            self._layer()                                        # replaces H then X
            self.items += _ccz(0, 1, 2)                          # diffusion's CCZ: fixed
            self._layer()                                        # replaces X then H

    def _layer(self):
        for q in range(3):
            for name in ("rz", "ry", "rz"):
                self.items.append((name, q, self.n_params))
                self.n_params += 1

    def grover_init(self):
        """Parameters that reproduce the fixed Grover circuit exactly (up to a global phase)."""
        layers = [_H] + [_X @ _H, _H @ _X] * self.rounds         # operator of each layer (H then X = X @ H)
        return np.array([a for U in layers for _ in range(3) for a in zyz(U)])

    def qasm(self, params, shift=None):
        """shift = (item index, delta): that one gate's angle moved by delta (parameter-shift rule)."""
        body = []
        for i, it in enumerate(self.items):
            if isinstance(it, str):
                body.append(it)
            else:
                name, q, p = it
                ang = float(params[p]) + (shift[1] if shift and shift[0] == i else 0.0)
                body.append(f"{name}({ang!r}) q[{q}];")
        return _program(3, 3, body, [(i, i) for i in range(3)])

    def success(self, params, device, shift=None):
        """Exact probability of the marked answer on the device (compiled when it has native gates)."""
        prog = parse(self.qasm(params, shift))
        if device.native_gates is not None:
            prog, _ = transpile(prog, device)
        return float(probabilities(prog, device).get(self.key, 0.0))

    def gradient(self, params, device):
        """Parameter-shift gradient of success (every gate has exactly one parameter here)."""
        g = np.zeros(self.n_params)
        for i, it in enumerate(self.items):
            if isinstance(it, str):
                continue
            plus = self.success(params, device, (i, SHIFT))
            minus = self.success(params, device, (i, -SHIFT))
            g[it[2]] += (plus - minus) / 2
        return g

    def train(self, device, params=None, epochs=40, lr=0.05, seed=0, log=None):
        """Adam ascent on success. Returns (best params, history of success per epoch)."""
        if params is None:
            params = np.random.default_rng(seed).uniform(-math.pi, math.pi, self.n_params)
        params = np.array(params, dtype=float)
        m, v = np.zeros_like(params), np.zeros_like(params)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best, best_p, hist = -1.0, params.copy(), []
        for t in range(1, epochs + 1):
            s = self.success(params, device)
            hist.append(s)
            if s > best:
                best, best_p = s, params.copy()
            g = self.gradient(params, device)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            params = params + lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            if log:
                log(t, s)
        s = self.success(params, device)
        hist.append(s)
        if s > best:
            best, best_p = s, params.copy()
        return best_p, hist


def fixed_grover_success(device, iterations=2, marked=0b101):
    """The project's fixed grover3() circuit, exact success probability on the device."""
    alg = grover3(marked, iterations)
    prog = parse(alg["qasm"])
    if device.native_gates is not None:
        prog, _ = transpile(prog, device)
    return float(probabilities(prog, device).get(alg["expected"], 0.0))


def experiment(epochs=40, lr=0.05, test_days=range(1, 11), train_device="dq-5", log=print):
    """The whole v0.16.0 experiment (for Colab): train on the ideal chip and on train_device (nominal
    calibration), then test every circuit on calibration days it never saw. Returns a results dict."""
    ideal, chip = get_device("ideal"), get_device(train_device)
    out = {"epochs": epochs, "lr": lr, "train_device": train_device, "runs": {}, "test": {}}
    for rounds in (2, 1):
        vg = VariationalGrover(rounds=rounds)
        for dev_name, dev in (("ideal", ideal), (train_device, chip)):
            t0 = time.time()
            p0 = vg.grover_init()
            start = vg.success(p0, dev)
            log(f"training rounds={rounds} on {dev_name}: start (= fixed Grover) {start:.4f}")
            best, hist = vg.train(dev, p0, epochs=epochs, lr=lr,
                                  log=lambda t, s: log(f"   epoch {t:3d}  success {s:.4f}") if t % 5 == 0 else None)
            out["runs"][f"{dev_name}/rounds{rounds}"] = {"start": start, "best": max(hist), "history": hist,
                                                         "params": best.tolist(), "seconds": time.time() - t0}
            log(f"   best {max(hist):.4f}  ({time.time() - t0:.0f} s)")
    log(f"testing on {train_device} calibration days {list(test_days)} (never seen in training)")
    for d in test_days:
        dev = calibrate(chip, d)
        row = {"fixed_grover/rounds2": fixed_grover_success(dev, 2), "fixed_grover/rounds1": fixed_grover_success(dev, 1)}
        for rounds in (2, 1):
            vg = VariationalGrover(rounds=rounds)
            for src in ("ideal", train_device):
                row[f"{src}-trained/rounds{rounds}"] = vg.success(
                    np.array(out["runs"][f"{src}/rounds{rounds}"]["params"]), dev)
        out["test"][d] = row
        log(f"   day {d:>2}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


def summarize(out, train_device="dq-5"):
    """Plain-text verdict lines for the experiment's pre-set targets (see EXPERIMENTS.md, v0.16.0)."""
    R, T = out["runs"], out["test"]
    lines = []
    ideal_best = max(R["ideal/rounds2"]["best"], R["ideal/rounds1"]["best"])
    lines.append(f"ideal: fixed Grover {R['ideal/rounds2']['start']:.4f} -> best trained {ideal_best:.4f}  "
                 f"target >= 0.99: {'MET' if ideal_best >= 0.99 else 'MISSED'}")
    chip_best = max(R[f"{train_device}/rounds2"]["best"], R[f"{train_device}/rounds1"]["best"])
    fixed = R[f"{train_device}/rounds2"]["start"]
    lines.append(f"{train_device} (training calibration): fixed Grover {fixed:.4f} -> best trained {chip_best:.4f}  "
                 f"target >= 0.45: {'MET' if chip_best >= 0.45 else 'MISSED'}")
    best_key = max((f"{train_device}-trained/rounds2", f"{train_device}-trained/rounds1"),
                   key=lambda k: np.mean([row[k] for row in T.values()]))
    diffs = np.array([row[best_key] - row["fixed_grover/rounds2"] for row in T.values()])
    se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    lines.append(f"unseen days ({len(diffs)}): {best_key} minus fixed Grover = {diffs.mean():+.4f} +/- {se:.4f} "
                 f"(better on {int((diffs > 0).sum())}/{len(diffs)} days)  target >= +0.05 on average: "
                 f"{'MET' if diffs.mean() >= 0.05 else 'MISSED'}")
    return lines
