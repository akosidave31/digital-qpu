import numpy as np
import pytest
from digital_qpu import parse, probabilities, transpile, native_1q, to_qasm, schedule, Device, DEVICES, QPU
from digital_qpu.compiler import unitary_1q, SX
from digital_qpu.executor import layer_duration
from digital_qubit import H, I2

LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]
NATIVE = ("rz", "sx", "x", "cz")
CLEAN_LINE = Device("line5", 5, coupling=LINE, native_gates=NATIVE, virtual_rz=True)
G1 = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
GP = ["rx", "ry", "rz", "u1"]


def rand_qasm(rng, n, depth):
    L = ['OPENQASM 2.0;', f'qreg q[{n}];', f'creg c[{n}];']
    for _ in range(depth):
        r = rng.random()
        if n > 1 and r < 0.3:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
        elif r < 0.6:
            L.append(f"{GP[int(rng.integers(4))]}({rng.uniform(-3, 3):.6f}) q[{int(rng.integers(n))}];")
        else:
            L.append(f"{G1[int(rng.integers(len(G1)))]} q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return "\n".join(L)


def matrix_of(ops):
    U = I2
    for o in ops:
        U = unitary_1q(o) @ U
    return U


def same(A, B):
    return abs(abs(np.trace(A.conj().T @ B)) - 2) < 1e-7


def test_single_qubit_decomposition(rng):
    for _ in range(300):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        ops = native_1q(q, 0)
        assert same(matrix_of(ops), q)
        assert {o.name for o in ops} <= {"rz", "sx", "x"} and sum(o.name == "sx" for o in ops) <= 2


def test_decomposition_is_minimal_for_special_gates():
    assert native_1q(I2, 0) == []
    assert [o.name for o in native_1q(np.diag([1, 1j]), 0)] == ["rz"]
    assert [o.name for o in native_1q(np.array([[0, 1], [1, 0]]), 0)] == ["x"]
    assert sum(o.name == "sx" for o in native_1q(H, 0)) == 1


def test_compiled_programs_give_identical_results(rng):
    for n in (2, 3, 4, 5):
        for _ in range(15):
            prog = parse(rand_qasm(rng, n, 30))
            native, info = transpile(prog, CLEAN_LINE)
            P_orig = probabilities(prog, DEVICES["ideal"])
            P_nat = probabilities(native, CLEAN_LINE)
            for k in set(P_orig) | set(P_nat):
                assert abs(P_orig.get(k, 0) - P_nat.get(k, 0)) < 1e-9


def test_compiled_output_is_native_and_wired(rng):
    for _ in range(20):
        native, _ = transpile(parse(rand_qasm(rng, 5, 30)), CLEAN_LINE)
        for o in native.ops:
            assert o.name in NATIVE
            if len(o.qubits) == 2:
                assert CLEAN_LINE.allows(*o.qubits)


def test_routing_inserts_swaps_only_when_needed():
    near = parse("OPENQASM 2.0; qreg q[5]; creg c[5]; cx q[1], q[2]; measure q -> c;")
    far = parse("OPENQASM 2.0; qreg q[5]; creg c[5]; x q[0]; cx q[0], q[4]; measure q -> c;")
    assert transpile(near, CLEAN_LINE)[1]["swaps"] == 0
    native, info = transpile(far, CLEAN_LINE)
    assert info["swaps"] == 3
    P = probabilities(native, CLEAN_LINE)
    assert abs(P["10001"] - 1) < 1e-9          # q0 and q4 end up 1, reported on the right bits


def test_virtual_rz_is_free():
    dev = Device("v", 1, T1=[50.0], gate_time_1q=0.1, gate_error_1q=[0.1], virtual_rz=True)
    prog = parse("OPENQASM 2.0; qreg q[1]; creg c[1];" + " rz(0.3) q[0];" * 50 + " measure q -> c;")
    assert sum(layer_duration(L, dev) for L in schedule(prog, dev)) == 0
    assert abs(probabilities(prog, dev)["0"] - 1) < 1e-12


def test_qpu_now_runs_unwired_programs_on_dq5():
    qasm = "OPENQASM 2.0; qreg q[5]; creg c[5]; h q[0]; cx q[0], q[4]; measure q -> c;"
    job = QPU("dq-5").run(qasm, shots=500, seed=3)
    assert job.status == "DONE", job.error
    r = job.result()
    assert r["compiled"]["swaps"] == 3 and set(r["compiled"]) >= {"n_ops", "n_2q", "layout"}
    top = sorted(r["counts"], key=r["counts"].get, reverse=True)[:2]
    assert set(top) == {"00000", "10001"}
    assert QPU("dq-5").run(qasm, compile=False).status == "ERROR"      # raw program is rejected


def test_to_qasm_round_trip(rng):
    native, _ = transpile(parse(rand_qasm(rng, 4, 25)), CLEAN_LINE)
    again = parse(to_qasm(native))
    assert len(again.ops) == len(native.ops)
    P1, P2 = probabilities(native, CLEAN_LINE), probabilities(again, CLEAN_LINE)
    assert all(abs(P1[k] - P2.get(k, 0)) < 1e-12 for k in P1)


def test_cli_compile(capsys):
    from digital_qpu.__main__ import main
    assert main(["compile", "examples/bell.qasm"]) == 0
    out = capsys.readouterr().out
    assert "cz" in out and "native ops" in out
