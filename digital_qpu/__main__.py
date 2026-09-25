"""Command line:  python -m digital_qpu devices
                  python -m digital_qpu run FILE.qasm [--device dq-5] [--shots 1024] [--seed N] [--json]
                  python -m digital_qpu compile FILE.qasm [--device dq-5]  (show what the chip will run)
                  python -m digital_qpu rb [--device dq-5] [--qubit 0]      (randomized benchmarking)
                  python -m digital_qpu zz [--device dq-5] [--pair 0 1]     (measure ZZ crosstalk)
                  python -m digital_qpu calibration [--device dq-5] --day N (that day's data sheet)
                  python -m digital_qpu history [--device dq-5] --qubit Q --days N
                  python -m digital_qpu benchmark [--device dq-5] [--day N] [--shots 4000]
                  run / compile / rb / zz also take --day N; run also takes --mitigate readout|learned"""
import argparse
import numpy as np
import json
import sys
from .device import DEVICES
from .qpu import QPU
from .rb import randomized_benchmarking
from .crosstalk import zz_ramsey
from .calibration import calibrate, bad_qubits, history
from .device import get_device
from .qasm import parse
from .compiler import transpile, to_qasm


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
    r.add_argument("--mitigate", choices=["readout", "learned"], default=None)
    cp = sub.add_parser("compile", help="show the native program the device will run")
    cp.add_argument("file")
    cp.add_argument("--device", default="dq-5")
    cp.add_argument("--day", type=int, default=None)
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
            native, info = transpile(parse(f.read()), dev)
        print(to_qasm(native))
        print(f"// {info['n_ops']} native ops: {info['n_2q']} cz, {info['n_sx']} sx/x pulses, "
              f"{info['n_rz']} virtual rz; {info['swaps']} swaps inserted; layout {info['layout']}")
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
    if a.cmd == "devices":
        for d in DEVICES.values():
            print(f"{d.name:<8} {d.n_qubits:>3} qubits  {d.description}")
        return 0
    with open(a.file, encoding="utf-8") as f:
        job = QPU(a.device, day=a.day).run(f.read(), shots=a.shots, seed=a.seed, compile=not a.no_compile,
                                            mitigate=a.mitigate)
    if job.status == "ERROR":
        print(f"ERROR: {job.error}", file=sys.stderr)
        return 1
    res = job.result()
    if a.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"job {res['job_id']} on {res['device']}: {res['shots']} shots, "
          f"depth {res['depth']}, circuit time {res['circuit_time']:g}")
    if res["compiled"]:
        c = res["compiled"]
        print(f"compiled: {c['n_ops']} native ops ({c['n_2q']} cz, {c['n_sx']} sx/x, {c['n_rz']} virtual rz), "
              f"{c['swaps']} swaps inserted")
    for k, c in res["counts"].items():
        print(f"  {k}  {c:>6}  {'#' * round(40 * c / res['shots'])}")
    if "mitigated" in res:
        print("readout-mitigated probabilities:")
        for k, v in sorted(res["mitigated"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {k}  {v:6.3f}  {'#' * round(40 * v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
