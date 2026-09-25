"""Run a Program on a Device using digital_qubit as the qubits.
Gates are scheduled into time layers (ASAP). Each gate is followed by its depolarizing gate error
(same convention as Qiskit Aer; 'id' is an idle and has none). After each layer, EVERY qubit
experiences T1/T_phi noise for that layer's duration (busy or idle). Readout error at measurement.
Bit order of results follows Qiskit: classical bit 0 is the RIGHTMOST character."""
import numpy as np
from digital_qubit import NQubit, NDensity, PairNoise, gate_matrix, I2, X, Y, Z
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


def final_state(program, device):
    """NQubit (noise-free gates) or NDensity (decoherence and/or gate errors) after all gates."""
    layers = schedule(program, device)
    n = program.n_qubits
    if not device.needs_density:
        reg = NQubit(n)
        for layer in layers:
            for op in layer:
                reg = _apply(reg, op)
        return reg
    if n > MAX_NOISY_QUBITS:
        raise QasmError(f"noisy simulation is limited to {MAX_NOISY_QUBITS} qubits (program has {n})")
    reg = NDensity(n)
    inf = float("inf")
    for layer in layers:
        for op in layer:
            reg = _gate_error(_apply(reg, op), op, device)
        d = layer_duration(layer, device)
        if d > 0:
            for q in range(n):
                T1 = device.T1[q] if device.T1 is not None else inf
                Tp = device.T_phi[q] if device.T_phi is not None else None
                reg = reg.channel1(PairNoise().kraus(d, T1, Tp), q)
    return reg


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
