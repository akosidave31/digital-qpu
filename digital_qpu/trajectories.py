"""Trajectory (Monte-Carlo wavefunction) engine for noisy circuits with more qubits.
Each trajectory is a pure state (2^n numbers instead of 4^n); noise happens as random events with
exactly the right probabilities:
  * gate error (depolarizing): a random Pauli with probability p/4 each (1 qubit) or p/16 each
    of the 15 non-identity pairs (2 qubits);
  * T1: the qubit jumps from |1> to |0> with probability gamma * P(qubit = 1), otherwise the
    |1> part shrinks by sqrt(1 - gamma) (then renormalize);
  * dephasing: a Z flip with probability (1 - e^{-t/T_phi}) / 2;
  * ZZ and drive spill-over are unitary and applied exactly.
Averaging the probability distributions of many trajectories reproduces the exact noisy result;
the statistical error shrinks as 1/sqrt(number of trajectories). All trajectories of a batch are
processed together, so the number of numpy calls does not grow with the number of trajectories."""
import numpy as np
from digital_qubit import rx, X, Y, Z, I2
from .qasm import QasmError
from .executor import (schedule, layer_duration, _matrix_1q, _cached, _PULSE_ANGLE, _sign_mask,
                       _zz_phase)

MAX_TRAJECTORY_QUBITS = 16
PAULI = [I2, X, Y, Z]


class TrajectoryState:
    """Averaged outcome probabilities of many trajectories (same interface as other engines' probs)."""

    def __init__(self, n, probs, n_traj):
        self.n, self._p, self.n_traj = n, probs, n_traj

    def probs(self):
        return self._p


def _u1(B, U, q):
    return np.moveaxis(np.tensordot(U, B, axes=([1], [q + 1])), 0, q + 1)


def _u1_subset(B, U, q, sel):
    if sel.any():
        B[sel] = _u1(B[sel], U, q)
    return B


def _cx_batch(B, c, t, n):
    B = np.array(B)
    idx = [slice(None)] * (n + 1)
    idx[c + 1] = 1
    sub = B[tuple(idx)]
    tt = t if t < c else t - 1
    sub[...] = np.flip(sub, axis=tt + 1).copy()
    return B


def _thermal(B, q, gamma, p_deph, rng, n):
    T = B.shape[0]
    one = [slice(None)] * (n + 1)
    one[q + 1] = 1
    zero = list(one)
    zero[q + 1] = 0
    one, zero = tuple(one), tuple(zero)
    if gamma > 0:
        B = np.array(B)
        p1 = (np.abs(B[one]) ** 2).reshape(T, -1).sum(1)
        jump = rng.random(T) < gamma * p1
        s1, s0 = B[one], B[zero]
        if jump.any():
            s0[jump] = s1[jump]
            s1[jump] = 0
        s1[~jump] *= np.sqrt(1 - gamma)
        norm = np.sqrt((np.abs(B) ** 2).reshape(T, -1).sum(1))
        B /= norm.reshape((T,) + (1,) * n)
    if p_deph > 0:
        flip = rng.random(T) < p_deph
        if flip.any():
            B = np.array(B)
            s1 = B[one]
            s1[flip] *= -1
    return B


def final_trajectories(program, device, n_traj=300, seed=0, batch=64):
    layers = schedule(program, device)
    n = program.n_qubits
    if n > MAX_TRAJECTORY_QUBITS:
        raise QasmError(f"trajectory simulation is limited to {MAX_TRAJECTORY_QUBITS} qubits (program has {n})")
    batch = max(1, min(batch, 2 ** 22 // 2 ** n))
    rng = np.random.default_rng(seed)
    zz_pairs = device.zz_pairs(n)
    inf = float("inf")
    acc = np.zeros(2 ** n)
    done = 0
    while done < n_traj:
        T = min(batch, n_traj - done)
        B = np.zeros((T,) + (2,) * n, dtype=complex)
        B[(slice(None),) + (0,) * n] = 1.0
        for layer in layers:
            for op in layer:
                if len(op.qubits) == 1:
                    q = op.qubits[0]
                    if op.name != "id":
                        B = _u1(B, _cached(("u", op.name, op.params), lambda: _matrix_1q(op)), q)
                        p = device.error_1q(q)
                        if p > 0 and not (device.virtual_rz and op.name == "rz"):
                            r = rng.random(T)
                            B = np.array(B)
                            for k in (1, 2, 3):
                                B = _u1_subset(B, PAULI[k], q, (r >= (k - 1) * p / 4) & (r < k * p / 4))
                    if device.drive_crosstalk and op.name in _PULSE_ANGLE:
                        U = rx(device.drive_crosstalk * _PULSE_ANGLE[op.name])
                        for m in device.neighbours(q, n):
                            B = _u1(B, U, m)
                else:
                    a, b = op.qubits
                    if op.name == "cz":
                        B = B * _sign_mask(a, b, n)
                    elif op.name == "swap":
                        B = np.swapaxes(B, a + 1, b + 1)
                    elif op.name == "cx":
                        B = _cx_batch(B, a, b, n)
                    else:
                        raise QasmError(f"unsupported two-qubit gate {op.name}")
                    p = device.error_2q(a, b)
                    if p > 0:
                        r = rng.random(T)
                        hit = r < 15 * p / 16
                        if hit.any():
                            B = np.array(B)
                            k = np.minimum((r / (p / 16)).astype(int), 14) + 1      # 1..15
                            for kk in np.unique(k[hit]):
                                sel = hit & (k == kk)
                                i, j = divmod(int(kk), 4)
                                if i:
                                    B = _u1_subset(B, PAULI[i], a, sel)
                                if j:
                                    B = _u1_subset(B, PAULI[j], b, sel)
            d = layer_duration(layer, device)
            if d > 0:
                if zz_pairs:
                    B = B * _zz_phase(zz_pairs, d, n)
                if device.has_decoherence:
                    for q in range(n):
                        T1 = device.T1[q] if device.T1 is not None else inf
                        gamma = 1 - np.exp(-d / T1)
                        p_deph = (1 - np.exp(-d / device.T_phi[q])) / 2 if device.T_phi is not None else 0.0
                        B = _thermal(B, q, gamma, p_deph, rng, n)
        acc += (np.abs(B) ** 2).reshape(T, -1).sum(0)
        done += T
    return TrajectoryState(n, acc / n_traj, n_traj)
