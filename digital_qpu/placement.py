"""v0.21.0 experiment: does noise-aware placement (router="noise-aware") beat the default router?

For every calibration day and every algorithm that fits the chip, compile with router="auto" and with
router="noise-aware", compute the EXACT success probability (no shot noise) on that day's calibrated
chip, and compare. Days with a TLS "bad qubit" are where calibration-aware placement should matter most.

Known bias (against noise-aware): the simulator only simulates physical qubits 0..(highest used). The
default router always uses the lowest-numbered qubits, so crosstalk from the unused qubits above them is
not simulated; a noise-aware placement on higher qubits simulates the idle qubits below it, including
their always-on ZZ crosstalk. So noise-aware is compared under slightly harsher conditions."""
import math
import numpy as np
from .qasm import parse
from .device import get_device
from .calibration import calibrate, bad_qubits
from .compiler import transpile
from .executor import probabilities
from .algorithms import all_algorithms


def _used(native):
    return sorted({q for o in native.ops for q in o.qubits} | set(native.measures))


def placement_experiment(days=range(60), device="dq-5", log=print):
    nominal = get_device(device)
    algs = [a for a in all_algorithms("all") if a["qubits"] <= nominal.n_qubits]
    rows = []
    for d in days:
        dev = calibrate(nominal, d)
        bad = bad_qubits(nominal, d)
        for alg in algs:
            prog = parse(alg["qasm"])
            r = {"day": d, "algorithm": alg["name"], "bad": bad}
            for router in ("auto", "noise-aware"):
                native, _ = transpile(prog, dev, router=router)
                r[router] = float(alg["success"](probabilities(native, dev)))
                r[f"{router}_qubits"] = _used(native)
            r["diff"] = r["noise-aware"] - r["auto"]
            r["auto_uses_bad"] = bool(set(bad) & set(r["auto_qubits"]))
            rows.append(r)
        flag = f" bad qubits {bad}" if bad else ""
        day_rows = [x for x in rows if x["day"] == d]
        log(f"day {d:>2}{flag}: mean change {np.mean([x['diff'] for x in day_rows]) * 100:+.2f} points"
            + "".join(f" | {x['algorithm'].replace('Grover search (3 qubits', 'Grover').replace(')', '')} "
                      f"{x['auto'] * 100:.1f}->{x['noise-aware'] * 100:.1f} {x['auto_qubits']}->{x['noise-aware_qubits']}"
                      for x in day_rows if x["auto_uses_bad"] or abs(x["diff"]) > 0.01))
    return rows


def summarize_placement(rows):
    """Verdicts for the v0.21.0 targets (EXPERIMENTS.md)."""
    diffs = np.array([r["diff"] for r in rows])
    se = lambda x: x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    lines = [f"cases: {len(rows)} (algorithm x day); noise-aware moved the circuit in "
             f"{sum(r['auto_qubits'] != r['noise-aware_qubits'] for r in rows)}"]
    lines.append(f"T1 all cases: noise-aware minus auto = {diffs.mean() * 100:+.2f} +/- {se(diffs) * 100:.2f} points  "
                 f"target >= +0.5: {'MET' if diffs.mean() >= 0.005 else 'MISSED'}")
    bad = np.array([r["diff"] for r in rows if r["auto_uses_bad"]])
    if len(bad) >= 5:
        lines.append(f"T2 cases where auto uses a bad-day qubit ({len(bad)}): {bad.mean() * 100:+.2f} +/- "
                     f"{se(bad) * 100:.2f} points  target >= +3: {'MET' if bad.mean() >= 0.03 else 'MISSED'}")
    else:
        lines.append(f"T2 only {len(bad)} cases where auto uses a bad-day qubit (need >= 5): NOT TESTED")
    harm = float((diffs < -0.01).mean())
    lines.append(f"T3 worse by more than 1 point in {harm * 100:.1f}% of cases (worst {diffs.min() * 100:+.2f})  "
                 f"target <= 10%: {'MET' if harm <= 0.10 else 'MISSED'}")
    for name in sorted({r["algorithm"] for r in rows}):
        x = np.array([r["diff"] for r in rows if r["algorithm"] == name])
        lines.append(f"   {name:36} {x.mean() * 100:+.2f} points (best {x.max() * 100:+.2f}, worst {x.min() * 100:+.2f})")
    return lines
