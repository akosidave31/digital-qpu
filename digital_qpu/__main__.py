"""Command line:  python -m digital_qpu devices
                  python -m digital_qpu run FILE.qasm [--device dq-5] [--shots 1024] [--seed N] [--json]
                  python -m digital_qpu compile FILE.qasm [--device dq-5]  (show what the chip will run)
                  python -m digital_qpu rb [--device dq-5] [--qubit 0]      (randomized benchmarking)
                  python -m digital_qpu zz [--device dq-5] [--pair 0 1]     (measure ZZ crosstalk)
                  python -m digital_qpu calibration [--device dq-5] --day N (that day's data sheet)
                  python -m digital_qpu history [--device dq-5] --qubit Q --days N
                  python -m digital_qpu benchmark [--device dq-5] [--day N] [--shots 4000]
                  python -m digital_qpu algorithms [--device dq-5] [--shots 2000] [--grover standard|exact|1round|all]
                  python -m digital_qpu shor [--shots 2000]                         (factor 15, step by step)
                  python -m digital_qpu speed [--quick] [--save FILE.json] [--compare FILE.json]
                  python -m digital_qpu routing                                     (basic vs look-ahead router)
                  python -m digital_qpu serve [--host 127.0.0.1] [--port 8000]      (web API: submit jobs over HTTP)
                  run / compile / rb / zz also take --day N; run also takes --mitigate readout|learned|learned-linear
                  run / compile / algorithms / shor also take --router auto|lookahead|basic"""
import argparse
import numpy as np
import json
import sys
from .device import DEVICES
from .qpu import QPU
from .rb import randomized_benchmarking
from .crosstalk import zz_ramsey
from .calibration import calibrate, bad_qubits, history
from .algorithms import shor_factors as shor_factors_cli
from .device import get_device
from .qasm import parse
from .compiler import transpile, to_qasm

ROUTERS = ["auto", "lookahead", "basic", "noise-aware"]


def routing_report(device_names):
    """SWAPs and cz gates for every algorithm: basic router vs the default (auto) router."""
    from .algorithms import all_algorithms
    rows = []
    for name in device_names:
        dev = get_device(name)
        for alg in all_algorithms():
            if alg["qubits"] > dev.n_qubits:
                continue
            prog = parse(alg["qasm"])
            _, old = transpile(prog, dev, router="basic")
            _, new = transpile(prog, dev)
            rows.append({"algorithm": alg["name"], "device": name, "swaps_basic": old["swaps"],
                         "swaps_new": new["swaps"], "cz_basic": old["n_2q"], "cz_new": new["n_2q"],
                         "router": new["router"]})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="digital_qpu", description="virtual quantum computer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list devices")
    r = sub.add_parser("run", help="run an OpenQASM 2.0 file")
    r.add_argument("file")
    r.add_argument("--device", default="dq-5")
    r.add_argument("--shots", type=int, default=1024)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--json", action="store_true")
    r.add_argument("--no-compile", action="store_true", help="run the program's gates directly")
    r.add_argument("--day", type=int, default=None)
    r.add_argument("--mitigate", choices=["readout", "learned", "learned-linear"], default=None)
    r.add_argument("--trajectories", type=int, default=300, help="number of trajectories when that engine is used")
    r.add_argument("--router", choices=ROUTERS, default="auto")
    cp = sub.add_parser("compile", help="show the native program the device will run")
    cp.add_argument("file")
    cp.add_argument("--device", default="dq-5")
    cp.add_argument("--day", type=int, default=None)
    cp.add_argument("--router", choices=ROUTERS, default="auto")
    sv = sub.add_parser("serve", help="web API: submit jobs over HTTP (see digital_qpu/server.py)")
    sv.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this device only; 0.0.0.0 = your network")
    sv.add_argument("--port", type=int, default=8000)
    ro = sub.add_parser("routing", help="compare the basic and look-ahead routers on the algorithms")
    ro.add_argument("--devices", nargs="+", default=["dq-5", "dq-12"])
    b = sub.add_parser("rb", help="randomized benchmarking: measure error per gate")
    b.add_argument("--device", default="dq-5")
    b.add_argument("--qubit", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--day", type=int, default=None)
    z = sub.add_parser("zz", help="Ramsey experiment: measure ZZ crosstalk between two qubits")
    z.add_argument("--device", default="dq-5")
    z.add_argument("--pair", type=int, nargs=2, default=[0, 1])
    z.add_argument("--day", type=int, default=None)
    cal = sub.add_parser("calibration", help="the device's calibration data sheet for a day")
    cal.add_argument("--device", default="dq-5")
    cal.add_argument("--day", type=int, required=True)
    be = sub.add_parser("benchmark", help="distance to the exact answers: raw vs readout-mitigated")
    be.add_argument("--device", default="dq-5")
    be.add_argument("--day", type=int, default=None)
    be.add_argument("--shots", type=int, default=4000)
    be.add_argument("--runs", type=int, default=1, help="repeat on this many calibration days (with error bars)")
    al = sub.add_parser("algorithms", help="run famous quantum algorithms and check their answers")
    al.add_argument("--device", default="dq-5")
    al.add_argument("--day", type=int, default=None)
    al.add_argument("--shots", type=int, default=2000)
    al.add_argument("--seed", type=int, default=1)
    al.add_argument("--trajectories", type=int, default=300)
    al.add_argument("--router", choices=ROUTERS, default="auto")
    al.add_argument("--grover", choices=["standard", "exact", "1round", "all"], default="standard",
                    help="which Grover version(s): textbook, exact (Long's phase) or 1 round (best on noisy chips)")
    sh = sub.add_parser("shor", help="Shor's algorithm factoring 15, step by step")
    sh.add_argument("--shots", type=int, default=2000)
    sh.add_argument("--seed", type=int, default=1)
    sh.add_argument("--device", default="ideal")
    sh.add_argument("--day", type=int, default=None)
    sh.add_argument("--trajectories", type=int, default=300)
    sh.add_argument("--router", choices=ROUTERS, default="auto")
    sp = sub.add_parser("speed", help="speed benchmark: where does the time go? (changes nothing)")
    sp.add_argument("--quick", action="store_true")
    sp.add_argument("--save", default=None, help="write results to a JSON file (a baseline to compare against)")
    sp.add_argument("--compare", default=None, help="a saved baseline JSON: show how many times faster")
    hi = sub.add_parser("history", help="how one qubit's calibration drifted over days")
    hi.add_argument("--device", default="dq-5")
    hi.add_argument("--qubit", type=int, default=0)
    hi.add_argument("--days", type=int, default=30)
    a = ap.parse_args(argv)
    if a.cmd == "calibration":
        nom, dev = get_device(a.device), calibrate(get_device(a.device), a.day)
        bad = bad_qubits(nom, a.day)
        print(f"{dev.name}  (nominal values in brackets)")
        print(f"  {'qubit':<6}{'T1':>14}{'T_phi':>14}{'gate err':>18}{'readout 0->1':>18}")
        for q in range(dev.n_qubits):
            print(f"  q{q:<5}{dev.T1[q]:>7.1f} [{nom.T1[q]:>4.0f}]{dev.T_phi[q]:>7.1f} [{nom.T_phi[q]:>4.0f}]"
                  f"{dev.error_1q(q):>10.1e} [{nom.error_1q(q):.0e}]"
                  f"{dev.readout_error[q][0]:>10.3f} [{nom.readout_error[q][0]:.3f}]"
                  f"{'   <- bad day (TLS defect)' if q in bad else ''}")
        if isinstance(dev.gate_error_2q, dict):
            print("  2-qubit gate errors: " + ", ".join(f"q{a}-q{b} {v * 100:.2f}%" for (a, b), v in sorted(dev.gate_error_2q.items())))
        return 0
    if a.cmd == "benchmark" and a.runs > 1:
        from .mitigation import evaluate_many, summarize
        days = list(range(a.runs))
        print(f"{a.runs} runs on calibration days {days}; models retrained each day (~1 min per run on a phone)")
        runs = evaluate_many(a.device, days=days, shots=a.shots)
        cols = [("readout", "readout_tvd"), ("linear", "linear_tvd"), ("MLP", "mlp_tvd"), ("floor", "floor")]
        print(f"  {'day':<6}" + "".join(f"{c:>9}" for c, _ in cols))
        for d, rows in zip(days, runs):
            print(f"  {d:<6}" + "".join(f"{np.mean([r[k] for r in rows]):>9.3f}" for _, k in cols))
        S = summarize(runs)
        print("  " + "-" * 42)
        print(f"  {'mean':<6}" + "".join(f"{S['mean'][k]:>9.3f}" for _, k in cols))
        print(f"  {'+/-':<6}" + "".join(f"{S['se'][k]:>9.3f}" for _, k in cols))
        P = S["paired"]
        print(f"MLP vs linear (same runs, same circuits): linear - MLP = {P['mean_diff']:+.4f} +/- {P['se']:.4f}")
        print("  VERDICT: " + ("MLP is better by more than 2 error bars"
                               if P["b_better_by_2se"] else "no reliable difference -> keep linear as default"))
        for name, k in (("linear", "linear_tvd"), ("MLP", "mlp_tvd")):
            H = S["harm"][k]
            print(f"  harm, {name}: worse than readout alone in {H['rate'] * 100:.0f}% of circuit-runs "
                  f"(by {H['mean_excess']:.3f} on average when worse; worst {H['worst']:+.3f})")
        return 0
    if a.cmd == "benchmark":
        from .mitigation import evaluate
        print("training learned mitigation on random circuits (different seed from the benchmark)...")
        rows = evaluate(a.device, a.day, a.shots)
        print(f"distance to the exact answer (TVD: 0 = perfect), {a.shots} shots per circuit")
        cols = [("raw", "raw_tvd"), ("readout", "readout_tvd"), ("linear", "linear_tvd"), ("MLP", "mlp_tvd"),
                ("floor", "floor")]
        print(f"  {'circuit':<10}" + "".join(f"{c:>9}" for c, _ in cols))
        for r in rows:
            print(f"  {r['circuit']:<10}" + "".join(f"{r[k]:>9.3f}" for _, k in cols))
        avg = {k: float(np.mean([r[k] for r in rows])) for _, k in cols}
        print(f"  {'average':<10}" + "".join(f"{avg[k]:>9.3f}" for _, k in cols))
        gap = avg["readout_tvd"] - avg["floor"]
        for name, k in (("linear", "linear_tvd"), ("MLP", "mlp_tvd")):
            closed = (avg["readout_tvd"] - avg[k]) / gap * 100 if gap > 0 else 0.0
            verdict = "beats" if avg[k] < avg["readout_tvd"] else "does NOT beat"
            print(f"  {name}: {verdict} the readout baseline; closes {closed:.0f}% of the gap to the shot-noise floor")
        return 0
    if a.cmd == "speed":
        from .perf import run, print_report, save
        res = run(quick=a.quick)
        base = None
        if a.compare:
            with open(a.compare) as f:
                base = json.load(f)
        print_report(res, base)
        if a.save:
            save(res, a.save)
            print(f"saved to {a.save}")
        return 0
    if a.cmd == "algorithms":
        from .algorithms import all_algorithms
        dev = calibrate(get_device(a.device), a.day)
        for alg in all_algorithms(grover=a.grover):
            ideal = QPU("ideal").run(alg["qasm"], shots=a.shots, seed=a.seed).result()["counts"]
            ai, si = alg["answer"](ideal), alg["success"](ideal)
            line = f"   expected {alg['expected']} | ideal {ai} ({si * 100:.0f}%) {'OK' if ai == alg['expected'] else 'WRONG'}"
            if alg["qubits"] <= dev.n_qubits:
                noisy = QPU(dev).run(alg["qasm"], shots=a.shots, seed=a.seed, n_traj=a.trajectories,
                                     router=a.router).result()["counts"]
                an, sn = alg["answer"](noisy), alg["success"](noisy)
                line += f" | {dev.name} {an} ({sn * 100:.0f}%) {'OK' if an == alg['expected'] else 'WRONG'}"
            else:
                line += f" | {dev.name}: needs {alg['qubits']} qubits (has {dev.n_qubits})"
            print(f"{alg['name']}: {alg['task']}")
            print(line)
        print("(percent = share of shots that gave the correct answer)")
        return 0
    if a.cmd == "shor":
        from .algorithms import shor15
        from fractions import Fraction
        from math import gcd
        alg = shor15()
        import time as _time
        t0 = _time.time()
        r = QPU(a.device, day=a.day).run(alg["qasm"], shots=a.shots, seed=a.seed, n_traj=a.trajectories,
                                         router=a.router).result()
        counts = r["counts"]
        print(f"Shor's algorithm: factor N = 15 with a = 7 (8 qubits: 4 counting + 4 work) on {r['device']}")
        if r["compiled"]:
            c = r["compiled"]
            print(f"   compiled: {c['n_ops']} native ops ({c['n_2q']} cz), {c['swaps']} swaps ({c['router']} router); "
                  f"circuit time {r['circuit_time']:g}; engine {r['engine']}; {_time.time() - t0:.0f} s")
        useful = (counts.get("0100", 0) + counts.get("1100", 0)) / sum(counts.values())
        print(f"   useful outcomes (y = 4 or 12): {useful * 100:.0f}% (ideal: 50%)")
        print("1. quantum part: measure the 4 counting qubits")
        for k, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"   {k} (y = {int(k, 2):>2})  {c:>5}  {'#' * round(30 * c / a.shots)}")
        print("2. classical part: y / 16 -> fraction -> period r, keep r if 7^r = 1 (mod 15)")
        for k in sorted(counts, key=counts.get, reverse=True):
            y = int(k, 2)
            if y == 0:
                print("   y =  0: no information, skip")
                continue
            fr = Fraction(y, 16).limit_denominator(15)
            r = fr.denominator
            if not (r % 2 == 0 and pow(7, r, 15) == 1):
                verdict = "(not the period)"
            elif 1 < gcd(pow(7, r // 2) - 1, 15) < 15:
                verdict = f"7^{r} mod 15 = 1  -> period found"
            else:
                verdict = f"7^{r} mod 15 = 1, but only trivial factors (a multiple of the period)"
            print(f"   y = {y:>2}: {y}/16 = {fr} -> r = {r}  {verdict}")
        r, f = shor_factors_cli(counts)
        if f:
            print(f"3. factors: gcd(7^{r // 2} - 1, 15) = {gcd(7 ** (r // 2) - 1, 15)}, "
                  f"gcd(7^{r // 2} + 1, 15) = {gcd(7 ** (r // 2) + 1, 15)}")
            print(f"   15 = {f[0]} x {f[1]}")
        return 0
    if a.cmd == "history":
        nom = get_device(a.device)
        print(f"{nom.name} qubit {a.qubit}: T1 by day (nominal {nom.T1[a.qubit]:g})")
        for row in history(nom, a.qubit, a.days):
            bar = "#" * round(30 * row["T1"] / (1.6 * nom.T1[a.qubit]))
            print(f"  day {row['day']:>3}  T1 {row['T1']:6.1f}  {bar}{'  <- TLS' if row['tls'] else ''}")
        return 0
    if a.cmd == "zz":
        res = zz_ramsey(calibrate(get_device(a.device), a.day), *a.pair)
        print(f"ZZ Ramsey on {a.device}, qubits {a.pair[0]}-{a.pair[1]}")
        for t, d in zip(res["times"], res["phase_diff"]):
            print(f"  wait {t:>4g}   phase difference {d:+.4f} rad")
        print(f"measured ZZ rate  {res['zz_measured']:.4f} rad per time unit")
        print(f"built-in ZZ rate  {res['zz_configured']:.4f}")
        return 0
    if a.cmd == "compile":
        dev = calibrate(get_device(a.device), a.day)
        with open(a.file, encoding="utf-8") as f:
            native, info = transpile(parse(f.read()), dev, router=a.router)
        print(to_qasm(native))
        print(f"// {info['n_ops']} native ops: {info['n_2q']} cz, {info['n_sx']} sx/x pulses, "
              f"{info['n_rz']} virtual rz; {info['swaps']} swaps inserted ({info['router']} router)")
        print(f"// layout (logical -> physical): start {info['initial_layout']}, end {info['layout']}")
        return 0
    if a.cmd == "routing":
        rows = routing_report(a.devices)
        print(f"  {'algorithm':<28}{'device':<8}{'swaps basic->new':>18}{'cz basic->new':>16}")
        for r in rows:
            print(f"  {r['algorithm']:<28}{r['device']:<8}{r['swaps_basic']:>9} -> {r['swaps_new']:<5}"
                  f"{r['cz_basic']:>8} -> {r['cz_new']:<5}")
        sb, sn = sum(r["swaps_basic"] for r in rows), sum(r["swaps_new"] for r in rows)
        cb, cn = sum(r["cz_basic"] for r in rows), sum(r["cz_new"] for r in rows)
        print(f"  {'total':<36}{sb:>9} -> {sn:<5}{cb:>8} -> {cn:<5}")
        worse = [r for r in rows if r["swaps_new"] > r["swaps_basic"]]
        print("  new router never needs more SWAPs than the basic one" if not worse
              else f"  WARNING: more SWAPs than basic on {len(worse)} circuit(s)")
        return 0
    if a.cmd == "rb":
        res = randomized_benchmarking(calibrate(get_device(a.device), a.day), a.qubit, seed=a.seed)
        print(f"randomized benchmarking on {a.device}, qubit {a.qubit}")
        for m, y in zip(res["lengths"], res["survival"]):
            print(f"  {m:>4} Cliffords   P(0) = {y:.4f}  {'#' * round(40 * y)}")
        print(f"fit: P(0) = {res['A']:.3f} * {res['p']:.5f}^m + {res['B']:.3f}   (B fixed from readout calibration)")
        print(f"error per Clifford  {res['epc']:.2e}")
        print(f"error per gate      {res['epg']:.2e}   (built-in expectation {res['predicted_epg']:.2e})")
        print(f"free 3-parameter fit would give {res['epg_free']:.2e} (B = {res['B_free']:.3f}; unreliable with short sequences)")
        return 0
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port)
        return 0
    if a.cmd == "devices":
        for d in DEVICES.values():
            print(f"{d.name:<8} {d.n_qubits:>3} qubits  {d.description}")
        return 0
    with open(a.file, encoding="utf-8") as f:
        job = QPU(a.device, day=a.day).run(f.read(), shots=a.shots, seed=a.seed, compile=not a.no_compile,
                                            mitigate=a.mitigate, n_traj=a.trajectories, router=a.router)
    if job.status == "ERROR":
        print(f"ERROR: {job.error}", file=sys.stderr)
        return 1
    res = job.result()
    if a.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"job {res['job_id']} on {res['device']}: {res['shots']} shots, "
          f"depth {res['depth']}, circuit time {res['circuit_time']:g}, engine {res['engine']}")
    if res["compiled"]:
        c = res["compiled"]
        print(f"compiled: {c['n_ops']} native ops ({c['n_2q']} cz, {c['n_sx']} sx/x, {c['n_rz']} virtual rz), "
              f"{c['swaps']} swaps inserted ({c['router']} router)")
    for k, c in res["counts"].items():
        print(f"  {k}  {c:>6}  {'#' * round(40 * c / res['shots'])}")
    if "mitigated" in res:
        print(f"{res['mitigation']}-mitigated probabilities:")
        for k, v in sorted(res["mitigated"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {k}  {v:6.3f}  {'#' * round(40 * v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
