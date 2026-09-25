import math
import numpy as np
from digital_qpu import parse, probabilities, Device, DEVICES, zz_ramsey, QPU


def prog(n, body):
    return parse(f'OPENQASM 2.0;\nqreg q[{n}];\ncreg c[{n}];\n' + body)


def test_zz_phase_matches_formula():
    zeta = 0.3
    dev = Device("zz", 2, coupling=[(0, 1)], zz=zeta, gate_time_1q=1.0)
    for spectator in ("", "x q[1];"):
        for k in (0, 3, 10):
            P = probabilities(prog(2, "h q[0];" + spectator + " id q[0];" * k + " h q[0]; measure q[0] -> c[0];"), dev)
            p0 = P["00"]                       # c1 is unused, c0 (rightmost) holds qubit 0
            assert math.isclose(p0, (1 + math.cos(zeta * (k + 1) / 2)) / 2, rel_tol=1e-9)


def test_no_zz_without_wiring():
    dev = Device("nozz", 2, zz=0.3, gate_time_1q=1.0)          # float rate but no coupling map
    P = probabilities(prog(2, "h q[0]; x q[1];" + " id q[0];" * 10 + " h q[0]; measure q[0] -> c[0];"), dev)
    assert math.isclose(P["00"], 1.0)


def test_drive_crosstalk_matches_formula():
    eps = 0.02
    dev = Device("drv", 2, coupling=[(0, 1)], drive_crosstalk=eps)
    for N in (2, 10, 20):
        P = probabilities(prog(2, "x q[0];" * N + " measure q -> c;"), dev)
        assert math.isclose(P["10"], math.sin(N * eps * math.pi / 2) ** 2, abs_tol=1e-12)


def test_coherent_errors_grow_quadratically():
    dev = Device("drv", 2, coupling=[(0, 1)], drive_crosstalk=0.005)
    e1 = probabilities(prog(2, "x q[0];" * 10 + " measure q -> c;"), dev)["10"]
    e2 = probabilities(prog(2, "x q[0];" * 20 + " measure q -> c;"), dev)["10"]
    assert 3.8 < e2 / e1 < 4.0          # doubling the circuit ~quadruples a coherent error


def test_virtual_rz_and_idle_do_not_spill():
    dev = Device("drv", 2, coupling=[(0, 1)], drive_crosstalk=0.1, virtual_rz=True)
    P = probabilities(prog(2, "rz(1.0) q[0]; id q[0];" * 10 + " measure q -> c;"), dev)
    assert math.isclose(P["00"], 1.0)


def test_ramsey_measures_the_built_in_zz():
    for a, b in [(0, 1), (1, 2), (3, 4)]:
        res = zz_ramsey(DEVICES["dq-5"], a, b)
        assert abs(res["zz_measured"] - res["zz_configured"]) / res["zz_configured"] < 0.05, res


def test_crosstalk_hurts_ghz_on_dq5():
    ghz = open("examples/ghz5.qasm").read()
    base = {k: v for k, v in DEVICES["dq-5"].__dict__.items() if k not in ("name", "zz", "drive_crosstalk")}
    quiet = Device("dq-5-quiet", **base)
    from digital_qpu import transpile
    native, _ = transpile(parse(ghz), DEVICES["dq-5"])
    P_real = probabilities(native, DEVICES["dq-5"])
    P_quiet = probabilities(native, quiet)
    good = lambda P: P["00000"] + P["11111"]
    assert good(P_real) < good(P_quiet)


def test_neighbours():
    dq5 = DEVICES["dq-5"]
    assert dq5.neighbours(0) == [1] and dq5.neighbours(2) == [1, 3] and dq5.neighbours(2, n=3) == [1]


def test_cli_zz(capsys):
    from digital_qpu.__main__ import main
    assert main(["zz", "--pair", "0", "1"]) == 0
    assert "measured ZZ rate" in capsys.readouterr().out
