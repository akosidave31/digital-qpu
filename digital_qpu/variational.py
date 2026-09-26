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
- Trained on ONE marked item (v0.16.0), the circuit learned to output that answer while partly ignoring
  the oracle (memorisation; see EXPERIMENTS.md). Since v0.17.0, JointGrover trains one set of angles on
  ALL 8 marked items at once, and per_item_spread() flags any circuit that favours some answers.
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
                name, q, p = it[:3]
                coef = it[3] if len(it) > 3 else 1.0             # angle = coef * parameter (v0.18.0)
                ang = coef * float(params[p]) + (shift[1] if shift and shift[0] == i else 0.0)
                body.append(f"{name}({ang!r}) q[{q}];")
        return _program(3, 3, body, [(i, i) for i in range(3)])

    def success(self, params, device, shift=None):
        """Exact probability of the marked answer on the device (compiled when it has native gates)."""
        prog = parse(self.qasm(params, shift))
        if device.native_gates is not None:
            prog, _ = transpile(prog, device)
        return float(probabilities(prog, device).get(self.key, 0.0))

    def gradient(self, params, device):
        """Parameter-shift gradient of success; a gate whose angle is coef * parameter adds coef times its
        shift derivative (chain rule), so one parameter may drive several gates."""
        g = np.zeros(self.n_params)
        for i, it in enumerate(self.items):
            if isinstance(it, str):
                continue
            plus = self.success(params, device, (i, SHIFT))
            minus = self.success(params, device, (i, -SHIFT))
            g[it[2]] += (it[3] if len(it) > 3 else 1.0) * (plus - minus) / 2
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
    """FLAWED (kept to reproduce the v0.16.0 record): trains on ONE marked item, so the circuit can learn
    the answer instead of using the oracle. Use joint_experiment instead.
    The whole v0.16.0 experiment (for Colab): train on the ideal chip and on train_device (nominal
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


# ---------------------------------------------------------------------------------------------------
# v0.17.0: joint training over all marked items (no answer can be memorised)
# ---------------------------------------------------------------------------------------------------
SPREAD_LIMIT = 0.10       # a trained circuit counts only if, on the ideal chip, max - min over items <= this


class JointGrover:
    """One shared set of angles, scored on the AVERAGE success over all 8 possible marked items.
    Memorising one answer cannot pay: it helps 1 of 8 oracles and hurts the others."""

    def __init__(self, rounds=2, factory=None):
        self.rounds = rounds
        make = factory or (lambda m: VariationalGrover(marked=m, rounds=rounds))
        self.circuits = [make(m) for m in range(8)]
        self.n_params = self.circuits[0].n_params

    def grover_init(self):
        return self.circuits[0].grover_init()                   # the layers do not depend on the marked item

    def per_item(self, params, device):
        return [c.success(params, device) for c in self.circuits]

    def success(self, params, device):
        return float(np.mean(self.per_item(params, device)))

    def gradient(self, params, device):
        return np.mean([c.gradient(params, device) for c in self.circuits], axis=0)

    train = VariationalGrover.train


def per_item_spread(params, rounds, device=None, factory=None):
    """(per-item successes on the ideal chip, max - min). The memorisation check."""
    dev = device or get_device("ideal")
    make = factory or (lambda m: VariationalGrover(marked=m, rounds=rounds))
    vals = [make(m).success(np.asarray(params), dev) for m in range(8)]
    return vals, max(vals) - min(vals)


def fixed_grover_mean(device, iterations=2):
    return float(np.mean([fixed_grover_success(device, iterations, m) for m in range(8)]))


def joint_experiment(epochs=40, lr=0.05, test_days=range(1, 11), train_device="dq-5", log=print):
    """v0.17.0: train JointGrover (all 8 marked items) on the ideal chip and on train_device, both 1 and 2
    rounds, starting from Grover; check memorisation; test on unseen calibration days. Returns a dict."""
    ideal, chip = get_device("ideal"), get_device(train_device)
    out = {"epochs": epochs, "lr": lr, "train_device": train_device, "runs": {}, "test": {}}
    for rounds in (2, 1):
        jg = JointGrover(rounds=rounds)
        for dev_name, dev in (("ideal", ideal), (train_device, chip)):
            t0 = time.time()
            p0 = jg.grover_init()
            start = jg.success(p0, dev)
            log(f"joint training rounds={rounds} on {dev_name}: start (= fixed Grover, mean of 8 items) {start:.4f}")
            best, hist = jg.train(dev, p0, epochs=epochs, lr=lr,
                                  log=lambda t, s: log(f"   epoch {t:3d}  mean success {s:.4f}") if t % 5 == 0 else None)
            items, spread = per_item_spread(best, rounds)
            out["runs"][f"{dev_name}/rounds{rounds}"] = {
                "start": start, "best": max(hist), "history": hist, "params": best.tolist(),
                "ideal_per_item": items, "ideal_spread": spread, "seconds": time.time() - t0}
            log(f"   best mean {max(hist):.4f}  | ideal per item {' '.join(f'{v:.2f}' for v in items)} "
                f"spread {spread:.3f}  ({time.time() - t0:.0f} s)")
    log(f"testing on {train_device} calibration days {list(test_days)} (never seen), mean over all 8 marked items")
    for d in test_days:
        dev = calibrate(chip, d)
        row = {"fixed/rounds2": fixed_grover_mean(dev, 2), "fixed/rounds1": fixed_grover_mean(dev, 1)}
        for rounds in (2, 1):
            jg = JointGrover(rounds=rounds)
            for src in ("ideal", train_device):
                row[f"{src}-trained/rounds{rounds}"] = jg.success(
                    np.array(out["runs"][f"{src}/rounds{rounds}"]["params"]), dev)
        out["test"][d] = row
        log(f"   day {d:>2}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


def summarize_joint(out, train_device="dq-5"):
    """Verdicts for the v0.17.0 targets (EXPERIMENTS.md). A run only counts if it passes the spread check."""
    R, T = out["runs"], out["test"]
    ok = lambda k: R[k]["ideal_spread"] <= SPREAD_LIMIT
    lines = []
    for k in R:
        lines.append(f"memorisation check {k:15}: ideal spread {R[k]['ideal_spread']:.3f} "
                     f"{'OK' if ok(k) else 'SPECIALISED - does not count'}")
    ideal_valid = [R[k]["best"] for k in ("ideal/rounds2", "ideal/rounds1") if ok(k)]
    ib = max(ideal_valid) if ideal_valid else float("nan")
    lines.append(f"ideal: fixed Grover mean {R['ideal/rounds2']['start']:.4f} -> best valid trained {ib:.4f}  "
                 f"target >= 0.97: {'MET' if ideal_valid and ib >= 0.97 else 'MISSED'}")
    chip_keys = [k for k in (f"{train_device}/rounds2", f"{train_device}/rounds1") if ok(k)]
    fixed_chip = R[f"{train_device}/rounds2"]["start"]
    cb = max((R[k]["best"] for k in chip_keys), default=float("nan"))
    lines.append(f"{train_device} (training calibration): fixed Grover mean {fixed_chip:.4f} -> best valid trained "
                 f"{cb:.4f}  target >= fixed + 0.03: {'MET' if chip_keys and cb >= fixed_chip + 0.03 else 'MISSED'}")
    f2 = np.mean([r["fixed/rounds2"] for r in T.values()])
    f1 = np.mean([r["fixed/rounds1"] for r in T.values()])
    base_key = "fixed/rounds1" if f1 > f2 else "fixed/rounds2"
    lines.append(f"unseen days: fixed Grover mean, 1 round {f1:.4f} vs 2 rounds {f2:.4f} "
                 f"(replication of v0.16.0: shorter wins on the noisy chip: {'YES' if f1 > f2 else 'NO'})")
    cands = [f"{train_device}-trained/rounds{int(k[-1])}" for k in chip_keys]
    if not cands:
        lines.append("unseen days: no valid trained circuit -> target MISSED")
        return lines
    best_key = max(cands, key=lambda k: np.mean([r[k] for r in T.values()]))
    diffs = np.array([r[best_key] - r[base_key] for r in T.values()])
    se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    lines.append(f"unseen days ({len(diffs)}): {best_key} minus best fixed ({base_key}) = {diffs.mean():+.4f} "
                 f"+/- {se:.4f} (better on {int((diffs > 0).sum())}/{len(diffs)} days)  target >= +0.02: "
                 f"{'MET' if diffs.mean() >= 0.02 else 'MISSED'}")
    return lines


# ---------------------------------------------------------------------------------------------------
# v0.18.0: trainable PHASES in the oracle and the diffusion (Long's exact Grover)
# ---------------------------------------------------------------------------------------------------
PHASE_START = math.pi - 0.3      # pi exactly is a symmetric point where the phase gradient is zero


def long_phase(rounds=2, n_items=8):
    """Long (2001): with oracle and diffusion phase phi, `rounds` iterations find the item with
    certainty when phi = 2 arcsin(sin(pi / (4 rounds + 2)) / sin(beta)), beta = arcsin(1/sqrt(N));
    None when that many rounds cannot reach 100%."""
    beta = math.asin(1 / math.sqrt(n_items))
    x = math.sin(math.pi / (4 * rounds + 2)) / math.sin(beta)
    return 2 * math.asin(x) if x <= 1 else None


def _ccp_items(p):
    """Doubly-controlled phase diag(1,...,1, e^{i lambda}) on qubits 0,1,2 with lambda = params[p]:
    cp(l/2)(1,2) cx(0,1) cp(-l/2)(1,2) cx(0,1) cp(l/2)(0,2), each cp(t) = p(t/2) cx p(-t/2) cx p(t/2).
    Exact; at lambda = pi it is CCZ. Uses 8 CNOTs (the fixed Toffoli-based CCZ uses 6)."""
    def cp(sign, x, y):
        k = sign / 4
        return [("p", x, p, k), f"cx q[{x}],q[{y}];", ("p", y, p, -k), f"cx q[{x}],q[{y}];", ("p", y, p, k)]
    return cp(+1, 1, 2) + ["cx q[0],q[1];"] + cp(-1, 1, 2) + ["cx q[0],q[1];"] + cp(+1, 0, 2)


def _ccp6_items(p):
    """v0.19.0: the same doubly-controlled phase with 6 CNOTs - the standard CCZ circuit (as in
    algorithms._ccz, whose H pairs cancel) with every T replaced by p(lambda/4) and every Tdg by
    p(-lambda/4). Its T gates add phases on a, b, c and their XORs; for bits
    a + b + c - (a^b) - (a^c) - (b^c) + (a^b^c) = 4abc, so the total is exactly lambda*abc for ANY lambda."""
    a, b, c = 0, 1, 2
    P = lambda q, k: ("p", q, p, k)
    return [f"cx q[{b}],q[{c}];", P(c, -0.25), f"cx q[{a}],q[{c}];", P(c, 0.25), f"cx q[{b}],q[{c}];",
            P(c, -0.25), f"cx q[{a}],q[{c}];", P(b, 0.25), P(c, 0.25), f"cx q[{a}],q[{b}];", P(a, 0.25),
            P(b, -0.25), f"cx q[{a}],q[{b}];"]


class PhaseGrover(VariationalGrover):
    """Grover with a trainable phase in every oracle call and every diffusion (one parameter each),
    and optionally trainable rz-ry-rz layers as in VariationalGrover (train_layers=True).
    The oracle is still one call per round marking the same item; only its phase is adjustable."""

    def __init__(self, marked=0b101, rounds=2, train_layers=False, form="cnot8"):
        if form not in ("cnot8", "cnot6"):
            raise ValueError("form must be 'cnot8' or 'cnot6'")
        self.n, self.marked, self.rounds, self.train_layers = 3, marked, rounds, train_layers
        self.form = form
        ccp = _ccp_items if form == "cnot8" else _ccp6_items
        self.key = format(marked, "03b")
        self.items, self.n_params = [], 0
        self.slots, self.phase_params = [], []          # (first param, layer matrix), phase param indices
        flips = [f"x q[{i}];" for i in range(3) if not marked >> i & 1]
        self._add_layer(_H, ["h"])
        for _ in range(rounds):
            self.items += flips + ccp(self._new_phase()) + flips                 # oracle with phase
            self._add_layer(_X @ _H, ["h", "x"])
            self.items += ccp(self._new_phase())                                 # diffusion with phase
            self._add_layer(_H @ _X, ["x", "h"])

    def _new_phase(self):
        self.phase_params.append(self.n_params)
        self.n_params += 1
        return self.n_params - 1

    def _add_layer(self, U, fixed_gates):
        if self.train_layers:
            self.slots.append((self.n_params, U))
            self._layer()
        else:
            self.items += [f"{g} q[{q}];" for g in fixed_gates for q in range(3)]

    def grover_init(self, phase=PHASE_START):
        p = np.zeros(self.n_params)
        for start, U in self.slots:
            p[start:start + 9] = [a for _ in range(3) for a in zyz(U)]
        p[self.phase_params] = phase
        return p


def phase_factory(rounds, train_layers, form="cnot8"):
    return lambda m: PhaseGrover(marked=m, rounds=rounds, train_layers=train_layers, form=form)


def fixed_phase_form_mean(device, rounds):
    """Fixed Grover written with the phase-form CCZ (phase = pi): the baseline with the same gate count."""
    jg = JointGrover(rounds, phase_factory(rounds, False))
    return jg.success(jg.circuits[0].grover_init(math.pi), device)


def phase_experiment(epochs=60, lr=0.1, test_days=range(1, 11), train_device="dq-5", log=print):
    """v0.18.0: joint training (all 8 marked items) of the oracle/diffusion phases. Runs:
    ideal phases-only 2 rounds, ideal phases+layers 2 rounds, dq-5 phases-only 2 rounds and 1 round."""
    ideal, chip = get_device("ideal"), get_device(train_device)
    plan = [("ideal", ideal, 2, False), ("ideal", ideal, 2, True),
            (train_device, chip, 2, False), (train_device, chip, 1, False)]
    out = {"epochs": epochs, "lr": lr, "train_device": train_device, "long_phase": long_phase(2),
           "runs": {}, "test": {}}
    for dev_name, dev, rounds, layers in plan:
        key = f"{dev_name}/rounds{rounds}/{'phases+layers' if layers else 'phases'}"
        fac = phase_factory(rounds, layers)
        jg = JointGrover(rounds, fac)
        t0 = time.time()
        p0 = jg.circuits[0].grover_init()
        start = jg.success(p0, dev)
        log(f"{key}: {jg.n_params} params, start (phases {PHASE_START:.3f}) mean {start:.4f}")
        best, hist = jg.train(dev, p0, epochs=epochs, lr=lr,
                              log=lambda t, s: log(f"   epoch {t:3d}  mean success {s:.4f}") if t % 5 == 0 else None)
        items, spread = per_item_spread(best, rounds, factory=fac)
        phases = [float(best[i] % (2 * math.pi)) for i in jg.circuits[0].phase_params]
        out["runs"][key] = {"rounds": rounds, "layers": layers, "start": start, "best": max(hist), "history": hist,
                            "params": best.tolist(), "phases": phases, "ideal_per_item": items,
                            "ideal_spread": spread, "seconds": time.time() - t0}
        log(f"   best mean {max(hist):.4f} | phases {' '.join(f'{x:.3f}' for x in phases)} | ideal per item "
            f"{' '.join(f'{v:.2f}' for v in items)} spread {spread:.3f}  ({time.time() - t0:.0f} s)")
    log(f"testing on {train_device} calibration days {list(test_days)} (never seen), mean over all 8 marked items")
    for d in test_days:
        dev = calibrate(chip, d)
        row = {"fixed/rounds2": fixed_grover_mean(dev, 2), "fixed/rounds1": fixed_grover_mean(dev, 1),
               "fixed-phaseform/rounds2": fixed_phase_form_mean(dev, 2),
               "fixed-phaseform/rounds1": fixed_phase_form_mean(dev, 1)}
        for key, r in out["runs"].items():
            jg = JointGrover(r["rounds"], phase_factory(r["rounds"], r["layers"]))
            row[key] = jg.success(np.array(r["params"]), dev)
        out["test"][d] = row
        log(f"   day {d:>2}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


def summarize_phase(out, train_device="dq-5"):
    """Verdicts for the v0.18.0 targets (EXPERIMENTS.md). Runs failing the spread check do not count."""
    R, T = out["runs"], out["test"]
    ok = lambda k: R[k]["ideal_spread"] <= SPREAD_LIMIT
    lines = [f"memorisation check {k:32}: spread {R[k]['ideal_spread']:.3f} "
             f"{'OK' if ok(k) else 'SPECIALISED - does not count'}" for k in R]
    ideal_keys = [k for k in R if k.startswith("ideal/") and ok(k)]
    ib = max((R[k]["best"] for k in ideal_keys), default=float("nan"))
    lines.append(f"T1 ideal: fixed Grover 0.9453 -> best valid trained {ib:.4f}  target >= 0.99: "
                 f"{'MET' if ideal_keys and ib >= 0.99 else 'MISSED'}")
    lp = out["long_phase"]
    k = "ideal/rounds2/phases"
    ph = R[k]["phases"] if k in R else []
    dist = lambda x: min(abs(x - lp), abs(x - (2 * math.pi - lp)))
    close = bool(ph) and all(dist(x) <= 0.1 for x in ph)
    lines.append(f"T2 learned phases {' '.join(f'{x:.3f}' for x in ph)} vs Long's {lp:.3f} (or {2 * math.pi - lp:.3f}): "
                 f"within 0.1 rad: {'MET' if close else 'MISSED'}")
    fixed_keys = ["fixed/rounds2", "fixed/rounds1", "fixed-phaseform/rounds2", "fixed-phaseform/rounds1"]
    base = max(fixed_keys, key=lambda f: np.mean([r[f] for r in T.values()]))
    chip_keys = [k for k in R if k.startswith(train_device + "/") and ok(k)]
    if not chip_keys:
        lines.append("T3 unseen days: no valid trained circuit -> MISSED")
        return lines
    best_key = max(chip_keys, key=lambda k: np.mean([r[k] for r in T.values()]))
    diffs = np.array([r[best_key] - r[base] for r in T.values()])
    se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    means = "  ".join(f"{f} {np.mean([r[f] for r in T.values()]):.4f}" for f in fixed_keys)
    lines.append(f"unseen-day means of the fixed circuits: {means}")
    lines.append(f"T3 unseen days ({len(diffs)}): {best_key} minus best fixed ({base}) = {diffs.mean():+.4f} "
                 f"+/- {se:.4f} (better on {int((diffs > 0).sum())}/{len(diffs)} days)  target >= +0.02: "
                 f"{'MET' if diffs.mean() >= 0.02 else 'MISSED'}")
    return lines


# ---------------------------------------------------------------------------------------------------
# v0.19.0: exact Grover at the SAME two-qubit cost as standard Grover (6-CNOT phase gate)
# ---------------------------------------------------------------------------------------------------
def two_qubit_count(text, device):
    prog, info = transpile(parse(text), device)
    return info["n_2q"]


def cheap_phase_experiment(epochs=60, lr=0.1, test_days=range(1, 11), train_device="dq-5", log=print):
    """Evaluate exact Grover (Long's phase for every oracle/diffusion) in the 6-CNOT form, untrained, on
    unseen days; also train the phases of the 6-CNOT form on train_device (1 and 2 rounds)."""
    chip = get_device(train_device)
    lp = long_phase(2)
    long6 = JointGrover(2, phase_factory(2, False, "cnot6"))
    long8 = JointGrover(2, phase_factory(2, False, "cnot8"))
    p_long = np.full(4, lp)
    out = {"epochs": epochs, "lr": lr, "train_device": train_device, "long_phase": lp, "runs": {}, "test": {}}
    ideal = get_device("ideal")
    out["ideal"] = {"exact6/rounds2": long6.success(p_long, ideal), "exact8/rounds2": long8.success(p_long, ideal),
                    "fixed/rounds2": fixed_grover_mean(ideal, 2), "fixed/rounds1": fixed_grover_mean(ideal, 1)}
    counts = {"fixed/rounds2": two_qubit_count(grover3(0b101, 2)["qasm"], chip),
              "fixed/rounds1": two_qubit_count(grover3(0b101, 1)["qasm"], chip),
              "exact6/rounds2": two_qubit_count(long6.circuits[5].qasm(p_long), chip),
              "exact8/rounds2": two_qubit_count(long8.circuits[5].qasm(p_long), chip)}
    out["two_qubit_gates"] = counts
    log("ideal chip, mean over 8 marked items: " + "  ".join(f"{k} {v:.4f}" for k, v in out["ideal"].items()))
    log(f"two-qubit gates after compiling for {train_device} (marked 101): " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    for rounds in (2, 1):
        key = f"{train_device}-trained6/rounds{rounds}"
        fac = phase_factory(rounds, False, "cnot6")
        jg = JointGrover(rounds, fac)
        t0 = time.time()
        p0 = jg.circuits[0].grover_init()
        start = jg.success(p0, chip)
        log(f"{key}: {jg.n_params} phases, start (phases {PHASE_START:.3f}) mean {start:.4f}")
        best, hist = jg.train(chip, p0, epochs=epochs, lr=lr,
                              log=lambda t, s: log(f"   epoch {t:3d}  mean success {s:.4f}") if t % 5 == 0 else None)
        items, spread = per_item_spread(best, rounds, factory=fac)
        phases = [float(best[i] % (2 * math.pi)) for i in jg.circuits[0].phase_params]
        out["runs"][key] = {"rounds": rounds, "start": start, "best": max(hist), "history": hist,
                            "params": best.tolist(), "phases": phases, "ideal_per_item": items,
                            "ideal_spread": spread, "seconds": time.time() - t0}
        log(f"   best mean {max(hist):.4f} | phases {' '.join(f'{x:.3f}' for x in phases)} | ideal per item "
            f"{' '.join(f'{v:.2f}' for v in items)} spread {spread:.3f}  ({time.time() - t0:.0f} s)")
    log(f"testing on {train_device} calibration days {list(test_days)} (never seen), mean over all 8 marked items")
    for d in test_days:
        dev = calibrate(chip, d)
        row = {"fixed/rounds2": fixed_grover_mean(dev, 2), "fixed/rounds1": fixed_grover_mean(dev, 1),
               "exact6/rounds2": long6.success(p_long, dev), "exact8/rounds2": long8.success(p_long, dev)}
        for key, r in out["runs"].items():
            row[key] = JointGrover(r["rounds"], phase_factory(r["rounds"], False, "cnot6")).success(np.array(r["params"]), dev)
        out["test"][d] = row
        log(f"   day {d:>2}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return out


def summarize_cheap(out, train_device="dq-5"):
    """Verdicts for the v0.19.0 targets (EXPERIMENTS.md)."""
    T, R, c = out["test"], out["runs"], out["two_qubit_gates"]
    mean = lambda k: float(np.mean([r[k] for r in T.values()]))
    lines = [f"ideal: exact6 {out['ideal']['exact6/rounds2']:.4f}  exact8 {out['ideal']['exact8/rounds2']:.4f}  "
             f"standard 2 rounds {out['ideal']['fixed/rounds2']:.4f}"]
    lines.append(f"T1 two-qubit gates on {train_device}: exact6 {c['exact6/rounds2']} vs standard 2 rounds "
                 f"{c['fixed/rounds2']} (exact8 {c['exact8/rounds2']}): equal: "
                 f"{'MET' if c['exact6/rounds2'] <= c['fixed/rounds2'] else 'MISSED'}")
    diffs = np.array([r["exact6/rounds2"] - r["fixed/rounds2"] for r in T.values()])
    se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    lines.append(f"T2 unseen days: exact6 minus standard (both 2 rounds) = {diffs.mean():+.4f} +/- {se:.4f} "
                 f"(better on {int((diffs > 0).sum())}/{len(diffs)} days)  target >= +0.01: "
                 f"{'MET' if diffs.mean() >= 0.01 else 'MISSED'}")
    lines.append("unseen-day means: " + "  ".join(f"{k} {mean(k):.4f}" for k in next(iter(T.values()))))
    ok = [k for k in R if R[k]["ideal_spread"] <= SPREAD_LIMIT]
    for k in R:
        lines.append(f"memorisation check {k}: spread {R[k]['ideal_spread']:.3f} {'OK' if k in ok else 'SPECIALISED'}")
    fixed_best = max(("fixed/rounds2", "fixed/rounds1", "exact6/rounds2", "exact8/rounds2"), key=mean)
    if ok:
        best = max(ok, key=mean)
        d = np.array([r[best] - r[fixed_best] for r in T.values()])
        lines.append(f"T3 unseen days: best trained ({best}) minus best untrained ({fixed_best}) = {d.mean():+.4f} "
                     f"+/- {d.std(ddof=1) / math.sqrt(len(d)):.4f}  target >= +0.02: {'MET' if d.mean() >= 0.02 else 'MISSED'}")
    else:
        lines.append("T3: no valid trained circuit -> MISSED")
    lines.append(f"reported: best circuit on unseen days overall: {max(T[next(iter(T))], key=mean)} "
                 f"({max(mean(k) for k in next(iter(T.values()))):.4f})")
    return lines
