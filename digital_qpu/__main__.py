"""Command line:  python -m digital_qpu devices
                  python -m digital_qpu run FILE.qasm [--device dq-5] [--shots 1024] [--seed N] [--json]"""
import argparse
import json
import sys
from .device import DEVICES
from .qpu import QPU


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
    a = ap.parse_args(argv)
    if a.cmd == "devices":
        for d in DEVICES.values():
            print(f"{d.name:<8} {d.n_qubits:>3} qubits  {d.description}")
        return 0
    with open(a.file, encoding="utf-8") as f:
        job = QPU(a.device).run(f.read(), shots=a.shots, seed=a.seed)
    if job.status == "ERROR":
        print(f"ERROR: {job.error}", file=sys.stderr)
        return 1
    res = job.result()
    if a.json:
        print(json.dumps(res, indent=2))
        return 0
    print(f"job {res['job_id']} on {res['device']}: {res['shots']} shots, "
          f"depth {res['depth']}, circuit time {res['circuit_time']:g}")
    for k, c in res["counts"].items():
        print(f"  {k}  {c:>6}  {'#' * round(40 * c / res['shots'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
