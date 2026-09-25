import math
import numpy as np
from digital_qpu import parse, probabilities, Device, DEVICES, QPU
from digital_qpu import randomized_benchmarking, clifford_group, fit_decay
from digital_qubit import I2


def prog(n, body):
    return parse(f'OPENQASM 2.0;\nqreg q[{n}];\ncreg c[{n}];\n' + body)


def test_single_qubit_gate_error_matches_formula():
    lam = 0.01
    dev = Device("dep1", 1, gate_error_1q=[lam])
    for N in (2, 10, 40):
        P = probabilities(prog(1, "x q[0];" * N + " measure q -> c;"), dev)
        assert math.isclose(P["0"], (1 + (1 - lam) ** N) / 2, rel_tol=1e-9)


def test_two_qubit_gate_error_matches_formula():
    lam = 0.02
    dev = Device("dep2", 2, gate_error_2q=lam)
    P = probabilities(prog(2, "h q[0]; cx q[0], q[1]; measure q -> c;"), dev)
    assert math.isclose(P["00"] + P["11"], 1 - lam / 2, rel_tol=1e-9)


def test_idle_has_no_gate_error_and_pairs_have_their_own_rates():
    dev = Device("d", 1, gate_error_1q=[0.05])
    assert math.isclose(probabilities(prog(1, "id q[0];" * 10 + " measure q -> c;"), dev)["0"], 1.0)
    dq5 = DEVICES["dq-5"]
    assert dq5.error_2q(1, 0) == dq5.error_2q(0, 1) == 0.010 and dq5.error_2q(3, 4) == 0.014


def test_gate_errors_make_results_worse_on_dq5():
    grover = open("examples/grover2.qasm").read()
    no_gate_err = Device("dq-5-noerr", **{k: v for k, v in DEVICES["dq-5"].__dict__.items()
                                          if k not in ("name", "gate_error_1q", "gate_error_2q")})
    P_real = probabilities(parse(grover), DEVICES["dq-5"])["11"]
    P_noerr = probabilities(parse(grover), no_gate_err)["11"]
    assert P_real < P_noerr - 0.005


def test_clifford_group():
    C = clifford_group()
    assert len(C) == 24
    same = lambda A, B: abs(abs(np.trace(A.conj().T @ B)) - 2) < 1e-9
    for A, _ in C[:8]:
        for B, _ in C[:8]:
            assert any(same(A @ B, W) for W, _ in C)


def test_fit_decay_recovers_parameters():
    ms = np.array([1, 10, 30, 60, 100])
    p, A, B = fit_decay(ms, 0.45 * 0.99 ** ms + 0.52)
    assert abs(p - 0.99) < 1e-4 and abs(A - 0.45) < 1e-3 and abs(B - 0.52) < 1e-3


def test_rb_measures_pure_gate_error():
    dev = Device("dep", 1, gate_error_1q=[0.004])
    res = randomized_benchmarking(dev, 0, lengths=(1, 20, 60, 120), n_seq=8, seed=1)
    assert abs(res["epg"] - 0.002) / 0.002 < 0.1


def test_rb_measures_what_dq5_is_built_with():
    res = randomized_benchmarking(DEVICES["dq-5"], 0, lengths=(1, 30, 90, 180), n_seq=8, seed=2)
    assert abs(res["epg"] - res["predicted_epg"]) / res["predicted_epg"] < 0.2, res
