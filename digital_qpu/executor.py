"""Run a Program on a Device using digital_qubit as the qubits.
Gates are scheduled into time layers (ASAP). Each gate is followed by its depolarizing gate error
(same convention as Qiskit Aer; 'id' is an idle and has none) and, for sx/x pulses, by drive
crosstalk onto wired neighbours. After each layer, for that layer's duration: always-on ZZ between
wired pairs, then T1/T_phi noise on EVERY qubit (busy or idle). Readout error at measurement.
Bit order of results follows Qiskit: classical bit 0 is the RIGHTMOST character."""
import numpy as np
from digital_qubit import NQubit, NDensity, PairNoise, gate_matrix, rx, I2, X, Y, Z, CNOT, SWAP as SWAP_M
from .qasm import QasmError

MAX_NOISY_QUBITS = 10
_ALIASES = {"u1": "p"}


def schedule(program, device):
    """Validate against the device and group ops into layers (as-soon-as-possible)."""
    if program.n_qubits > device.n_qubits:
        raise QasmError(f"program uses {program.n_qubits} qubits; device '{device.name}' has {device.n_qubits}")
    free = [0] * program.n_qubits
    layers = []
    for op in program.ops:
        if len(op.qubits) == 2 and not device.allows(*op.qubits):
            raise QasmError(f"{op.name} q[{op.qubits[0]}],q[{op.qubits[1]}] is not allowed on "
                            f"'{device.name}' (connected pairs: {device.coupling})")
        L = max(free[q] for q in op.qubits)
        while len(layers) <= L:
            layers.append([])
        layers[L].append(op)
        for q in op.qubits:
            free[q] = L + 1
    return layers


SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2


def op_duration(op, device):
    if len(op.qubits) == 2:
        return device.gate_time_2q
    if device.virtual_rz and op.name == "rz":
        return 0.0
    return device.gate_time_1q


def layer_duration(layer, device):
    return max(op_duration(op, device) for op in layer)


def _apply(reg, op):
    if op.name == "id":
        return reg
    if len(op.qubits) == 2:
        return getattr(reg, op.name)(*op.qubits)
    if op.name == "sx":
        return reg.apply1(SX, op.qubits[0])
    if op.name in ("sdg", "tdg"):
        return reg.apply1(gate_matrix(op.name[0]).conj().T, op.qubits[0])
    name = _ALIASES.get(op.name, op.name)
    return reg.apply1(gate_matrix(name, op.params[0] if op.params else None), op.qubits[0])


_PAULI = [I2, X, Y, Z]


def depolarize_1q(reg, q, p):
    """rho -> (1 - p) rho + p I/2 on qubit q."""
    K = [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X, np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]
    return reg.channel1(K, q)


def depolarize_2q(reg, a, b, p):
    """rho -> (1 - p) rho + p I/4 (x) Tr_ab(rho) on qubits a, b."""
    out = (1 - 15 * p / 16) * reg.rho
    for i in range(4):
        for j in range(4):
            if i or j:
                out = out + (p / 16) * reg.apply2(np.kron(_PAULI[i], _PAULI[j]), a, b).rho
    return NDensity(reg.n, out)


def _gate_error(reg, op, device):
    if op.name == "id" or (device.virtual_rz and op.name == "rz"):
        return reg
    if len(op.qubits) == 1:
        p = device.error_1q(op.qubits[0])
        return depolarize_1q(reg, op.qubits[0], p) if p > 0 else reg
    p = device.error_2q(*op.qubits)
    return depolarize_2q(reg, *op.qubits, p) if p > 0 else reg


_PULSE_ANGLE = {"sx": np.pi / 2, "x": np.pi}


def _spillover(reg, op, device, n):
    """Drive crosstalk: a fraction of an sx/x pulse also rotates each wired neighbour."""
    if not device.drive_crosstalk or op.name not in _PULSE_ANGLE:
        return reg
    U = rx(device.drive_crosstalk * _PULSE_ANGLE[op.name])
    for m in device.neighbours(op.qubits[0], n):
        reg = reg.apply1(U, m)
    return reg


def zz_unitary(theta):
    """exp(-i theta/2 Z(x)Z), the same as Qiskit's RZZ(theta)."""
    return np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))


def _zz(reg, d, device, n):
    for (a, b), rate in device.zz_pairs(n):
        reg = reg.apply2(zz_unitary(rate * d / 2), a, b)
    return reg


def final_state_reference(program, device):
    """The original (slow, straightforward) engine, kept as the reference the fast engine must match."""
    layers = schedule(program, device)
    n = program.n_qubits
    density = device.needs_density
    if density and n > MAX_NOISY_QUBITS:
        raise QasmError(f"noisy simulation is limited to {MAX_NOISY_QUBITS} qubits (program has {n})")
    reg = NDensity(n) if density else NQubit(n)
    inf = float("inf")
    for layer in layers:
        for op in layer:
            reg = _apply(reg, op)
            if density:
                reg = _gate_error(reg, op, device)
            reg = _spillover(reg, op, device, n)
        d = layer_duration(layer, device)
        if d > 0:
            reg = _zz(reg, d, device, n)
            if device.has_decoherence:
                for q in range(n):
                    T1 = device.T1[q] if device.T1 is not None else inf
                    Tp = device.T_phi[q] if device.T_phi is not None else None
                    reg = reg.channel1(PairNoise().kraus(d, T1, Tp), q)
    return reg


# ---------------- fast density-matrix engine (v0.9.0) ----------------
# Same math as final_state_reference, far fewer numpy calls:
#  * every single-qubit step (gate, gate error, drive spill-over, decoherence) is a 4x4 channel
#    matrix; consecutive steps on one qubit are multiplied together and applied to the big state
#    once, only when that qubit must interact with another (or at the end);
#  * diagonal operations (cz, all ZZ of a time step) are element-wise multiplications;
#  * the 2-qubit depolarizing error uses (1-p) rho + p I/4 (x) Tr_ab(rho) directly;
#  * small matrices are cached.
_LET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TR = np.array([1, 0, 0, 1], dtype=complex)
_HALF_I = np.array([0.5, 0, 0, 0.5], dtype=complex)
_CACHE = {}


def _cached(key, make):
    v = _CACHE.get(key)
    if v is None:
        v = _CACHE[key] = make()
    return v


def _sup(U):
    return np.kron(U, U.conj())


def _gate_sup(op):
    return _cached(("gate", op.name, op.params), lambda: _sup(_matrix_1q(op)))


def _matrix_1q(op):
    if op.name == "sx":
        return SX
    if op.name in ("sdg", "tdg"):
        return gate_matrix(op.name[0]).conj().T
    return gate_matrix(_ALIASES.get(op.name, op.name), op.params[0] if op.params else None)


def _dep1_sup(p):
    return _cached(("dep1", p), lambda: (1 - p) * np.eye(4) + p * np.outer(_HALF_I, _TR))


def _thermal_sup(d, T1, Tp):
    return _cached(("th", d, T1, Tp), lambda: sum(np.kron(K, K.conj()) for K in PairNoise().kraus(d, T1, Tp)))


def _apply_sup(t, S, q, n):
    out = np.tensordot(S.reshape(2, 2, 2, 2), t, axes=([2, 3], [q, n + q]))
    return np.moveaxis(out, [0, 1], [q, n + q])


def _apply_u2(t, U4, a, b, n):
    U = U4.reshape(2, 2, 2, 2)
    t = np.moveaxis(np.tensordot(U, t, axes=([2, 3], [a, b])), [0, 1], [a, b])
    return np.moveaxis(np.tensordot(U.conj(), t, axes=([2, 3], [n + a, n + b])), [0, 1], [n + a, n + b])


def _cz_mask(a, b, n):
    def make():
        m = np.ones([2 if i in (a, b, n + a, n + b) else 1 for i in range(2 * n)])
        for ra in (0, 1):
            for rb in (0, 1):
                for ca in (0, 1):
                    for cb in (0, 1):
                        idx = [0] * (2 * n)
                        idx[a], idx[b], idx[n + a], idx[n + b] = ra, rb, ca, cb
                        m[tuple(idx)] = (-1) ** (ra * rb) * (-1) ** (ca * cb)
        return m
    return _cached(("cz", a, b, n), make)


def _zz_mask(pairs, d, n):
    def make():
        E = np.zeros((2,) * n)
        z = np.array([1.0, -1.0])
        for (a, b), rate in pairs:
            shape_a = [1] * n
            shape_a[a] = 2
            shape_b = [1] * n
            shape_b[b] = 2
            E = E + (rate * d / 2) * z.reshape(shape_a) * z.reshape(shape_b)
        R = np.exp(-0.5j * E)
        return R.reshape(R.shape + (1,) * n) * R.conj().reshape((1,) * n + R.shape)
    return _cached(("zz", tuple(pairs), d, n), make)


def _dep2(t, a, b, p, n):
    def make():
        rows, cols = _LET[:n], _LET[n:2 * n]
        full = rows + cols
        inp = list(full)
        inp[n + a], inp[n + b] = rows[a], rows[b]
        rest = "".join(c for i, c in enumerate(full) if i not in (a, b, n + a, n + b))
        return "".join(inp) + "->" + rest, f"{rest},{rows[a]}{cols[a]},{rows[b]}{cols[b]}->{full}"
    trace, expand = _cached(("dep2", a, b, n), make)
    r = np.einsum(trace, t)
    return (1 - p) * t + (p / 4) * np.einsum(expand, r, I2, I2)


def _final_density_fast(layers, device, n):
    t = np.zeros((2,) * (2 * n), dtype=complex)
    t[(0,) * (2 * n)] = 1.0
    pend = [None] * n

    def push(q, S):
        pend[q] = S if pend[q] is None else S @ pend[q]

    def flush(q):
        nonlocal t
        if pend[q] is not None:
            t = _apply_sup(t, pend[q], q, n)
            pend[q] = None

    zz_pairs = device.zz_pairs(n)
    zz_qubits = sorted({q for (a, b), _ in zz_pairs for q in (a, b)})
    inf = float("inf")
    for layer in layers:
        for op in layer:
            if len(op.qubits) == 1:
                q = op.qubits[0]
                if op.name != "id":
                    push(q, _gate_sup(op))
                    virtual = device.virtual_rz and op.name == "rz"
                    p = device.error_1q(q)
                    if p > 0 and not virtual:
                        push(q, _dep1_sup(p))
                if device.drive_crosstalk and op.name in _PULSE_ANGLE:
                    ang = device.drive_crosstalk * _PULSE_ANGLE[op.name]
                    S = _cached(("rx", ang), lambda: _sup(rx(ang)))
                    for m in device.neighbours(q, n):
                        push(m, S)
            else:
                a, b = op.qubits
                flush(a)
                flush(b)
                if op.name == "cz":
                    t = t * _cz_mask(a, b, n)
                else:
                    t = _apply_u2(t, {"cx": CNOT, "swap": SWAP_M}.get(op.name, None), a, b, n)
                p = device.error_2q(a, b)
                if p > 0:
                    t = _dep2(t, a, b, p, n)
        d = layer_duration(layer, device)
        if d > 0:
            if zz_pairs:
                for q in zz_qubits:
                    flush(q)
                t = t * _zz_mask(zz_pairs, d, n)
            if device.has_decoherence:
                for q in range(n):
                    T1 = device.T1[q] if device.T1 is not None else inf
                    Tp = device.T_phi[q] if device.T_phi is not None else None
                    push(q, _thermal_sup(d, T1, Tp))
    for q in range(n):
        flush(q)
    D = 2 ** n
    return NDensity(n, t.reshape(D, D))


def final_state(program, device):
    """NQubit (no stochastic noise) or NDensity (decoherence and/or gate errors) after all gates."""
    layers = schedule(program, device)
    n = program.n_qubits
    if not device.needs_density:
        return final_state_reference(program, device)
    if n > MAX_NOISY_QUBITS:
        raise QasmError(f"noisy simulation is limited to {MAX_NOISY_QUBITS} qubits (program has {n})")
    return _final_density_fast(layers, device, n)


def probabilities(program, device):
    """Exact probability of every classical outcome (readout error included), Qiskit bit order."""
    if not program.measures:
        raise QasmError("program has no measurements")
    reg = final_state(program, device)
    n = program.n_qubits
    mq = sorted(program.measures)
    P = reg.probs().reshape((2,) * n)
    others = tuple(q for q in range(n) if q not in mq)
    P = P.sum(axis=others) if others else P
    if device.readout_error is not None:
        for k, q in enumerate(mq):
            p01, p10 = device.readout_error[q]
            M = np.array([[1 - p01, p10], [p01, 1 - p10]])
            P = np.moveaxis(np.tensordot(M, P, axes=([1], [k])), 0, k)
    out = {}
    ncl = max(program.n_clbits, max(program.measures.values()) + 1)
    for idx, p in enumerate(P.reshape(-1)):
        bits = np.unravel_index(idx, (2,) * len(mq))
        key = ["0"] * ncl
        for b, q in zip(bits, mq):
            key[ncl - 1 - program.measures[q]] = str(int(b))
        k = "".join(key)
        out[k] = out.get(k, 0.0) + float(p)
    return out


def sample_counts(probs, shots, rng):
    keys = list(probs)
    p = np.clip(np.array([probs[k] for k in keys]), 0, None)
    counts = rng.multinomial(shots, p / p.sum())
    return {k: int(c) for k, c in zip(keys, counts) if c > 0}
