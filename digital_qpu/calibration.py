"""Calibration drift: a device's parameters change from day to day, like a real chip.
Each parameter wanders around its nominal value in log space as an AR(1) process (today resembles
yesterday, correlation RHO). Occasionally a qubit has a bad day: a defect ("TLS") steals its
energy and T1 drops to 20-50% of normal. ZZ comes from the chip's design and barely moves.
The same device + day always gives the same calibration (reproducible)."""
import dataclasses
import zlib
import numpy as np

RHO = 0.7
SIGMA = {"T1": 0.15, "T_phi": 0.20, "gate_1q": 0.20, "gate_2q": 0.25, "readout": 0.20, "zz": 0.03}
TLS_PROB = 0.05
TLS_RANGE = (0.2, 0.5)


def _pairs(x):
    return sorted(x) if isinstance(x, dict) else []


def _series(device, day):
    """Standard-normal AR(1) values for every drifting quantity on the given day, plus TLS draws."""
    nq = device.n_qubits
    k = {"T1": nq, "T_phi": nq, "gate_1q": nq, "gate_2q": len(_pairs(device.gate_error_2q)),
         "readout": nq, "zz": len(_pairs(device.zz))}
    K = sum(k.values())
    seed = zlib.crc32(device.name.encode())
    eps = np.random.default_rng([seed, 1]).standard_normal((day + 1, K))
    uni = np.random.default_rng([seed, 2]).random((day + 1, 2 * nq))
    z = eps[0].copy()
    for d in range(1, day + 1):
        z = RHO * z + np.sqrt(1 - RHO ** 2) * eps[d]
    out, i = {}, 0
    for name, n in k.items():
        out[name] = z[i:i + n]
        i += n
    tls = uni[day, :nq] < TLS_PROB
    factor = TLS_RANGE[0] + (TLS_RANGE[1] - TLS_RANGE[0]) * uni[day, nq:]
    return out, tls, factor


def calibrate(device, day):
    """The device as calibrated on `day` (0, 1, 2, ...). day=None returns the nominal device."""
    if day is None:
        return device
    day = int(day)
    if day < 0:
        raise ValueError("day must be >= 0")
    z, tls, factor = _series(device, day)
    f = lambda nom, key, j: nom * float(np.exp(SIGMA[key] * z[key][j]))
    T1 = None if device.T1 is None else [
        f(t, "T1", q) * (factor[q] if tls[q] else 1.0) for q, t in enumerate(device.T1)]
    T_phi = None if device.T_phi is None else [f(t, "T_phi", q) for q, t in enumerate(device.T_phi)]
    g1 = None if device.gate_error_1q is None else [
        min(0.5, f(e, "gate_1q", q)) for q, e in enumerate(device.gate_error_1q)]
    g2 = device.gate_error_2q
    if isinstance(g2, dict):
        g2 = {p: min(0.5, f(g2[p], "gate_2q", j)) for j, p in enumerate(_pairs(g2))}
    ro = None if device.readout_error is None else [
        (min(0.5, f(a, "readout", q)), min(0.5, f(b, "readout", q))) for q, (a, b) in enumerate(device.readout_error)]
    zz = device.zz
    if isinstance(zz, dict):
        zz = {p: f(zz[p], "zz", j) for j, p in enumerate(_pairs(zz))}
    return dataclasses.replace(device, name=f"{device.name}@day{day}", T1=T1, T_phi=T_phi,
                               gate_error_1q=g1, gate_error_2q=g2, readout_error=ro, zz=zz)


def bad_qubits(device, day):
    """Qubits hit by a TLS defect on this day."""
    _, tls, _ = _series(device, int(day))
    return [q for q in range(device.n_qubits) if tls[q]]


def history(device, qubit, days):
    """Day-by-day T1, T_phi, gate error and readout (0->1) of one qubit."""
    rows = []
    for d in range(days):
        c = calibrate(device, d)
        rows.append({"day": d, "T1": c.T1[qubit] if c.T1 else None,
                     "T_phi": c.T_phi[qubit] if c.T_phi else None,
                     "gate_1q": c.error_1q(qubit),
                     "readout_01": c.readout_error[qubit][0] if c.readout_error else None,
                     "tls": qubit in bad_qubits(device, d)})
    return rows
