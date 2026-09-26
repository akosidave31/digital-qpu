import math
import numpy as np
import pytest
from digital_qpu import parse, probabilities, schedule, QasmError, Device, DEVICES
from digital_qpu.executor import layer_duration

IDEAL = DEVICES["ideal"]


def prog(n, body, ncl=None):
    return parse(f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{ncl or n}];\n' + body)


def close(d, expect, tol=1e-9):
    return all(abs(d.get(k, 0.0) - v) < tol for k, v in expect.items()) and abs(sum(d.values()) - 1) < 1e-9


def test_bell_ideal():
    P = probabilities(prog(2, "h q[0]; cx q[0], q[1]; measure q -> c;"), IDEAL)
    assert close(P, {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0})


def test_bit_order_matches_qiskit():
    P = probabilities(prog(1, "x q[0]; measure q[0] -> c[0];", ncl=2), IDEAL)
    assert close(P, {"01": 1.0})
    P = probabilities(prog(2, "x q[0]; measure q[0] -> c[1]; measure q[1] -> c[0];"), IDEAL)
    assert close(P, {"10": 1.0})


def test_grover_finds_marked_item():
    with open("examples/grover2.qasm") as f:
        P = probabilities(parse(f.read()), IDEAL)
    assert close(P, {"11": 1.0})


def test_ghz_ideal():
    with open("examples/ghz5.qasm") as f:
        P = probabilities(parse(f.read()), IDEAL)
    assert close(P, {"00000": 0.5, "11111": 0.5})


def test_readout_error():
    dev = Device("ro", 1, readout_error=[(0.1, 0.2)])
    assert close(probabilities(prog(1, "measure q -> c;"), dev), {"0": 0.9, "1": 0.1})
    assert close(probabilities(prog(1, "x q[0]; measure q -> c;"), dev), {"0": 0.2, "1": 0.8})


def test_dephasing_during_gates_matches_formula():
    dev = Device("deph", 1, T_phi=[40.0], gate_time_1q=1.0)
    for k in (0, 5, 20):
        P = probabilities(prog(1, "h q[0];" + " id q[0];" * k + " h q[0]; measure q -> c;"), dev)
        assert math.isclose(P["0"], (1 + math.exp(-(k + 1) / 40.0)) / 2, rel_tol=1e-9)


def test_energy_loss_matches_formula():
    dev = Device("t1", 1, T1=[50.0], gate_time_1q=1.0)
    for k in (0, 10, 40):
        P = probabilities(prog(1, "x q[0];" + " id q[0];" * k + " measure q -> c;"), dev)
        assert math.isclose(P["1"], math.exp(-(k + 1) / 50.0), rel_tol=1e-9)


def test_idle_qubits_also_decay():
    dq5 = DEVICES["dq-5"]
    short = probabilities(prog(5, "x q[4]; measure q[4] -> c[0];", ncl=1), dq5)
    long = probabilities(prog(5, "x q[4];" + " id q[0];" * 300 + " measure q[4] -> c[0];", ncl=1), dq5)
    assert long["1"] < short["1"] - 0.05


def test_scheduling_layers_and_time():
    dev = DEVICES["dq-5"]
    layers = schedule(prog(2, "h q[0]; h q[1]; cx q[0], q[1];"), dev)
    assert len(layers) == 2 and len(layers[0]) == 2
    assert math.isclose(sum(layer_duration(L, dev) for L in layers), dev.gate_time_1q + dev.gate_time_2q)


def test_noisy_bell_on_dq5():
    P = probabilities(prog(2, "h q[0]; cx q[0], q[1]; measure q -> c;"), DEVICES["dq-5"])
    assert abs(sum(P.values()) - 1) < 1e-9
    assert P["00"] + P["11"] > 0.9 and P["01"] + P["10"] > 0.005


@pytest.mark.parametrize("body,dev", [
    ("cx q[0], q[2]; measure q -> c;", DEVICES["dq-5"]),              # not connected on the line
    ("h q[0];", IDEAL),                                               # no measurement
])
def test_rejected_programs(body, dev):
    with pytest.raises(QasmError):
        probabilities(prog(3, body), dev)


def test_limits():
    with pytest.raises(QasmError):
        probabilities(prog(6, "h q[0]; measure q -> c;"), DEVICES["dq-5"])     # device too small
    big = Device("big", 12, T1=[50.0] * 12, gate_time_1q=0.1)
    with pytest.raises(QasmError):
        probabilities(prog(12, "h q[0]; measure q -> c;"), big, method="density")   # exact noisy limit
    P12 = probabilities(prog(12, "h q[0]; measure q -> c;"), big, n_traj=50)       # trajectories: fine
    assert abs(sum(P12.values()) - 1) < 1e-9
    huge = Device("huge", 17, T1=[50.0] * 17, gate_time_1q=0.1)
    with pytest.raises(QasmError):
        probabilities(prog(17, "h q[0]; measure q -> c;"), huge)                    # trajectory limit
    P = probabilities(prog(16, "h q[0]; measure q[0] -> c[0];"), IDEAL)        # ideal: fine
    assert math.isclose(P["0" * 16], 0.5)
