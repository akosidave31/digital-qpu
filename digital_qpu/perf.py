"""Speed benchmark (measurement only: changes nothing). Answers: where does the time go?
Results can be saved to JSON so every later optimization is compared against the same baseline."""
import cProfile
import io
import json
import platform
import pstats
import time
import numpy as np
from .qasm import parse
from .device import Device, DEVICES
from .executor import final_state, probabilities
from .compiler import transpile


def _best(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def ghz(n):
    body = " ".join(["h q[0];"] + [f"cx q[{i}], q[{i + 1}];" for i in range(n - 1)])
    return parse(f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}]; {body} measure q -> c;")


def line_device(n, features=("gates", "thermal", "crosstalk")):
    """An n-qubit line with dq-5-like values; choose which noise features are switched on."""
    pairs = [(i, i + 1) for i in range(n - 1)]
    kw = dict(coupling=pairs, native_gates=("rz", "sx", "x", "cz"), virtual_rz=True,
              gate_time_1q=0.02, gate_time_2q=0.15)
    if "thermal" in features:
        kw.update(T1=[50.0] * n, T_phi=[40.0] * n)
    if "gates" in features:
        kw.update(gate_error_1q=[7e-4] * n, gate_error_2q={p: 0.01 for p in pairs})
    if "crosstalk" in features:
        kw.update(zz={p: 0.1 for p in pairs}, drive_crosstalk=0.01)
    return Device(f"line{n}-{'+'.join(features) or 'none'}", n, **kw)


def run(quick=False):
    from .algorithms import grover3
    from .learned import training_data
    from .rb import randomized_benchmarking
    res = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()}

    sizes = (10, 14) if quick else (10, 14, 18, 20)
    res["ideal"] = [{"qubits": n, "seconds": _best(lambda: final_state(ghz(n), DEVICES["ideal"]), 1 if n >= 18 else 3),
                     "state_MB": 2 ** n * 16 / 1e6} for n in sizes]

    res["noisy"] = []
    for n in ((2, 4) if quick else (2, 4, 6, 8)):
        dev = line_device(n)
        native, _ = transpile(ghz(n), dev)
        res["noisy"].append({"qubits": n, "seconds": _best(lambda: probabilities(native, dev), 1 if n >= 8 else 3),
                             "state_MB": 4 ** n * 16 / 1e6, "native_ops": len(native.ops)})

    res["trajectories"] = []
    for n, nt in (((10, 50),) if quick else ((10, 300), (12, 300))):
        dev = line_device(n)
        native, _ = transpile(ghz(n), dev)
        t0 = time.perf_counter()
        probabilities(native, dev, method="trajectories", n_traj=nt)
        res["trajectories"].append({"qubits": n, "trajectories": nt, "seconds": time.perf_counter() - t0,
                                    "state_MB_per_trajectory": 2 ** n * 16 / 1e6})

    g = parse(grover3()["qasm"])
    res["features"] = []
    for feats in ((), ("gates",), ("thermal",), ("crosstalk",), ("gates", "thermal", "crosstalk")):
        dev = line_device(5, feats)
        native, _ = transpile(g, dev)
        res["features"].append({"noise": "+".join(feats) or "none",
                                "seconds": _best(lambda: final_state(native, dev), 1 if quick else 2)})

    dq5 = DEVICES["dq-5"]
    n_train = 3 if quick else 10
    res["workloads"] = {
        "compile_grover3_dq5": _best(lambda: transpile(g, dq5)),
        "train_per_circuit": _best(lambda: training_data(dq5, n_train, seed=7), 1) / n_train,
        "rb_qubit0": _best(lambda: randomized_benchmarking(dq5, 0, lengths=(1, 20) if quick else (1, 20, 60, 120, 200)), 1),
    }

    native, _ = transpile(g, dq5)
    pr = cProfile.Profile()
    pr.enable()
    final_state(native, dq5)
    pr.disable()
    stats = pstats.Stats(pr)
    total = stats.total_tt
    rows = []
    for (file, line, fn), (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append({"function": f"{file.split('/')[-1]}:{fn}", "calls": nc, "own_s": tt, "share": tt / total})
    res["profile_total_s"] = total
    res["profile_top"] = sorted(rows, key=lambda r: -r["own_s"])[:10]
    return res


def print_report(res, baseline=None):
    def vs(section, i=None, key=None):
        if baseline is None:
            return ""
        try:
            old = baseline[section][i]["seconds"] if key is None else baseline[section][key]
            new = res[section][i]["seconds"] if key is None else res[section][key]
            return f"   {old / new:6.1f}x faster than baseline" if new > 0 else ""
        except (KeyError, IndexError, TypeError):
            return ""
    print(f"speed benchmark  (python {res['python']}, numpy {res['numpy']}, {res['machine']})")
    print("1. ideal state-vector engine, GHZ circuit")
    for i, r in enumerate(res["ideal"]):
        print(f"   {r['qubits']:>2} qubits  {r['seconds']:8.3f} s   state {r['state_MB']:8.1f} MB{vs('ideal', i)}")
    print("2. noisy engine (density matrix), GHZ on a noisy line")
    for i, r in enumerate(res["noisy"]):
        print(f"   {r['qubits']:>2} qubits  {r['seconds']:8.3f} s   state {r['state_MB']:8.2f} MB   {r['native_ops']} ops{vs('noisy', i)}")
    if res.get("trajectories"):
        print("2b. noisy engine (trajectories), GHZ on a noisy line")
        for r in res["trajectories"]:
            print(f"   {r['qubits']:>2} qubits  {r['seconds']:8.3f} s   {r['trajectories']} trajectories, "
                  f"{r['state_MB_per_trajectory']:.2f} MB each")
    print("3. cost of each noise feature: Grover (3 qubits) compiled on a 5-qubit line")
    base = res["features"][0]["seconds"]
    for i, r in enumerate(res["features"]):
        print(f"   {r['noise']:<24}{r['seconds']:8.3f} s   ({r['seconds'] / base:5.1f}x no-noise){vs('features', i)}")
    w = res["workloads"]
    print("4. workloads on dq-5")
    print(f"   compile Grover            {w['compile_grover3_dq5']:8.3f} s{vs('workloads', key='compile_grover3_dq5')}")
    print(f"   training, per circuit     {w['train_per_circuit']:8.3f} s{vs('workloads', key='train_per_circuit')}")
    print(f"   randomized benchmarking   {w['rb_qubit0']:8.3f} s{vs('workloads', key='rb_qubit0')}")
    print(f"5. where the time goes: one noisy Grover run on dq-5 ({res['profile_total_s']:.2f} s profiled)")
    for r in res["profile_top"]:
        print(f"   {r['share'] * 100:5.1f}%  {r['calls']:>7} calls  {r['function']}")


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
