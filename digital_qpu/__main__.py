"""Command line:  python -m digital_qpu devices
                  python -m digital_qpu run FILE.qasm [--device dq-5] [--shots 1024] [--seed N] [--json]
                  python -m digital_qpu compile FILE.qasm [--device dq-5]  (show what the chip will run)
                  python -m digital_qpu rb [--device dq-5] [--qubit 0]      (randomized benchmarking)"""
import argparse
import json
import sys
from .device import DEVICES
from .qpu import QPU
from .rb import randomized_benchmarking
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
    cp = sub.add_parser("compile", help="show the native program the device will run")
    cp.add_argument("file")
    cp.add_argument("--device", default="dq-5")
    b = sub.add_parser("rb", help="randomized benchmarking: measure error per gate")
    b.add_argument("--device", default="dq-5")
    b.add_argument("--qubit", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "compile":
        dev = get_device(a.device)
        with open(a.file, encoding="utf-8") as f:
            native, info = transpile(parse(f.read()), dev)
        print(to_qasm(native))
        print(f"// {info['n_ops']} native ops: {info['n_2q']} cz, {info['n_sx']} sx/x pulses, "
              f"{info['n_rz']} virtual rz; {info['swaps']} swaps inserted; layout {info['layout']}")
        return 0
    if a.cmd == "rb":
        res = randomized_benchmarking(get_device(a.device), a.qubit, seed=a.seed)
        print(f"randomized benchmarking on {a.device}, qubit {a.qubit}")
        for m, y in zip(res["lengths"], res["survival"]):
            print(f"  {m:>4} Cliffords   P(0) = {y:.4f}  {'#' * round(40 * y)}")
        print(f"fit: P(0) = {res['A']:.3f} * {res['p']:.5f}^m + {res['B']:.3f}")
        print(f"error per Clifford  {res['epc']:.2e}")
        print(f"error per gate      {res['epg']:.2e}   (built-in expectation {res['predicted_epg']:.2e})")
        return 0
    if a.cmd == "devices":
        for d in DEVICES.values():
            print(f"{d.name:<8} {d.n_qubits:>3} qubits  {d.description}")
        return 0
    with open(a.file, encoding="utf-8") as f:
        job = QPU(a.device).run(f.read(), shots=a.shots, seed=a.seed, compile=not a.no_compile)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
