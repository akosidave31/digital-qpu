import numpy as np
from digital_qpu import (parse, probabilities, transpile, readout_mitigate, benchmark_suite, evaluate, tvd,
                         distribution, Device, DEVICES, QPU)

LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]
RO = [(0.02, 0.05), (0.03, 0.06), (0.01, 0.04), (0.04, 0.08), (0.02, 0.05)]


def exact_parts(qasm, dev):
    native, _ = transpile(parse(qasm), dev)
    return probabilities(native, dev), {c: q for q, c in native.measures.items()}, native.n_clbits


def test_readout_mitigation_is_exact_without_shot_noise():
    dev = Device("ro", 5, readout_error=RO, coupling=LINE, native_gates=("rz", "sx", "x", "cz"), virtual_rz=True)
    for name, qasm in benchmark_suite():
        noisy, cq, ncl = exact_parts(qasm, dev)
        ideal = probabilities(parse(qasm), DEVICES["ideal"])
        assert tvd(readout_mitigate(noisy, dev, cq, ncl), ideal) < 1e-9, name


def test_mapping_follows_routed_qubits():
    dev = Device("ro", 5, readout_error=RO, coupling=LINE, native_gates=("rz", "sx", "x", "cz"), virtual_rz=True)
    qasm = "OPENQASM 2.0; qreg q[5]; creg c[5]; x q[0]; cx q[0], q[4]; measure q -> c;"
    noisy, cq, ncl = exact_parts(qasm, dev)
    assert cq[0] != 0                                # q0 was moved by the router
    assert abs(readout_mitigate(noisy, dev, cq, ncl)["10001"] - 1) < 1e-9


def test_mitigated_output_is_a_valid_distribution():
    r = QPU("dq-5").run(open("examples/ghz5.qasm").read(), shots=300, seed=2, mitigate="readout").result()
    m = r["mitigated"]
    assert all(v >= 0 for v in m.values()) and abs(sum(m.values()) - 1) < 1e-9


def test_mitigation_helps_with_shots_on_readout_only_device():
    dev = Device("ro", 5, readout_error=RO, coupling=LINE, native_gates=("rz", "sx", "x", "cz"), virtual_rz=True)
    qasm = open("examples/ghz5.qasm").read()
    ideal = {k: v for k, v in probabilities(parse(qasm), DEVICES["ideal"]).items() if v > 0}
    r = QPU(dev).run(qasm, shots=20000, seed=4, mitigate="readout").result()
    raw_tvd = tvd(distribution(r["counts"]), ideal)
    assert tvd(r["mitigated"], ideal) < 0.3 * raw_tvd


def test_benchmark_on_dq5_readout_helps_but_does_not_fix_everything():
    rows = evaluate("dq-5", shots=4000, learned=False)
    raw = np.mean([r["raw_tvd"] for r in rows])
    mit = np.mean([r["readout_tvd"] for r in rows])
    assert len(rows) == 10
    assert mit < raw                     # readout mitigation helps on average
    assert mit > 0.01                    # gate errors, decoherence and crosstalk remain


def test_ideal_device_needs_no_mitigation():
    r = QPU("ideal").run(open("examples/bell.qasm").read(), shots=100, seed=1, mitigate="readout").result()
    assert r["mitigated"] == distribution(r["counts"])


def test_cli(capsys):
    from digital_qpu.__main__ import main
    assert main(["run", "examples/bell.qasm", "--mitigate", "readout", "--shots", "200", "--seed", "1"]) == 0
    assert "readout-mitigated" in capsys.readouterr().out

