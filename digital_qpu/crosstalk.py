"""Measure always-on ZZ crosstalk the way a lab does: a Ramsey experiment.
Put qubit a on the equator, let it wait, and track its phase - once with neighbour b in |0>, once
in |1>. ZZ makes the phase drift in opposite directions; the difference grows at exactly the ZZ rate.
Readout error is corrected with the device's known readout calibration, as labs do.
The spectator's own T1 decay (|1> -> |0>) slows the phase difference over time, so the rate is the
INITIAL slope (quadratic fit), not a straight-line average."""
import numpy as np
from .qasm import Program, Op
from .device import Device
from .executor import probabilities


def pair_device(device, a, b):
    """The part of a device that matters for the pair (a, b), renumbered 0 and 1."""
    pick = lambda lst: None if lst is None else [lst[a], lst[b]]
    rate = dict(device.zz_pairs()).get((a, b), dict(device.zz_pairs()).get((b, a), 0.0))
    return Device(f"{device.name}-{a}{b}", 2, T1=pick(device.T1), T_phi=pick(device.T_phi),
                  gate_time_1q=device.gate_time_1q, gate_time_2q=device.gate_time_2q,
                  readout_error=pick(device.readout_error), gate_error_1q=pick(device.gate_error_1q),
                  coupling=[(0, 1)], zz={(0, 1): rate}, drive_crosstalk=device.drive_crosstalk)


def zz_ramsey(device, a, b, times=(0, 1, 2, 3, 4, 5, 6)):
    """Estimate the ZZ rate between qubits a and b. Returns the fitted rate and the raw phases."""
    dev = pair_device(device, a, b)
    g = dev.gate_time_1q
    p01, p10 = dev.readout_error[0] if dev.readout_error is not None else (0.0, 0.0)
    expect = lambda P: ((P["0"] - P["1"]) - (p10 - p01)) / (1 - p01 - p10)   # readout-corrected <.>
    phases = {}
    for spectator in (0, 1):
        ph = []
        for t in times:
            k = int(round(t / g))
            prep = [Op("h", (0,))] + ([Op("x", (1,))] if spectator else []) + [Op("id", (0,))] * k
            ex = probabilities(Program(2, 1, prep + [Op("h", (0,))], {0: 0}), dev)
            ey = probabilities(Program(2, 1, prep + [Op("sdg", (0,)), Op("h", (0,))], {0: 0}), dev)
            ph.append(np.arctan2(expect(ey), expect(ex)))
        phases[spectator] = np.unwrap(ph)
    diff = phases[1] - phases[0]
    slope = float(np.polyfit(np.asarray(times, float), diff, 2)[1])   # initial slope: spectator T1 decay bends the curve
    true = dict(dev.zz_pairs())[(0, 1)] if dev.zz_pairs() else 0.0
    return {"pair": (a, b), "times": list(times), "phase_diff": diff.tolist(),
            "zz_measured": abs(slope), "zz_configured": true}
