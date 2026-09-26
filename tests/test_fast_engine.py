"""The fast engine must give the same result as the original reference engine."""
import numpy as np
import pytest
from digital_qpu import parse, transpile, Device, DEVICES, all_algorithms
from digital_qpu.executor import final_state, final_state_reference

G1 = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"]
GP = ["rx", "ry", "rz", "u1"]


def rand_program(rng, n, depth):
    L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
    for _ in range(depth):
        r = rng.random()
        if n > 1 and r < 0.3:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
        elif r < 0.6:
            L.append(f"{GP[int(rng.integers(4))]}({rng.uniform(-3, 3):.5f}) q[{int(rng.integers(n))}];")
        else:
            L.append(f"{G1[int(rng.integers(len(G1)))]} q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return parse(" ".join(L))


def device(n, feats, virtual_rz):
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    kw = dict(coupling=pairs, gate_time_1q=0.03, gate_time_2q=0.2, virtual_rz=virtual_rz)
    if "thermal" in feats:
        kw.update(T1=list(np.linspace(30, 60, n)), T_phi=list(np.linspace(25, 45, n)))
    if "gates" in feats:
        kw.update(gate_error_1q=list(np.linspace(1e-3, 3e-3, n)), gate_error_2q={p: 0.01 + 0.002 * i for i, p in enumerate(pairs)})
    if "zz" in feats:
        kw.update(zz={p: 0.1 + 0.03 * i for i, p in enumerate(pairs)})
    if "drive" in feats:
        kw.update(drive_crosstalk=0.02)
    if "t1only" in feats:
        kw.update(T1=[40.0] * n)
    return Device("t", n, **kw)


CONFIGS = [("gates",), ("thermal",), ("t1only",), ("gates", "zz"), ("thermal", "drive"),
           ("gates", "thermal", "zz", "drive")]


@pytest.mark.parametrize("feats", CONFIGS)
@pytest.mark.parametrize("virtual_rz", [False, True])
def test_fast_engine_matches_reference(feats, virtual_rz):
    rng = np.random.default_rng(hash((feats, virtual_rz)) % 2 ** 32)
    for n in (1, 2, 3, 4):
        dev = device(n, feats, virtual_rz)
        for _ in range(4):
            prog = rand_program(rng, n, 25)
            diff = np.max(np.abs(final_state(prog, dev).rho - final_state_reference(prog, dev).rho))
            assert diff < 1e-12, (feats, n, diff)


def test_fast_engine_matches_reference_on_compiled_algorithms():
    dq5 = DEVICES["dq-5"]
    for alg in all_algorithms():
        if alg["qubits"] > 5:
            continue
        native, _ = transpile(parse(alg["qasm"]), dq5)
        diff = np.max(np.abs(final_state(native, dq5).rho - final_state_reference(native, dq5).rho))
        assert diff < 1e-12, (alg["name"], diff)


PURE_CONFIGS = [(), ("zz",), ("drive",), ("zz", "drive")]


@pytest.mark.parametrize("feats", PURE_CONFIGS)
@pytest.mark.parametrize("virtual_rz", [False, True])
def test_fast_pure_engine_matches_reference(feats, virtual_rz):
    rng = np.random.default_rng(len(feats) * 10 + int(virtual_rz))
    for n in (1, 2, 3, 5, 8):
        dev = device(n, feats, virtual_rz)
        assert not dev.needs_density
        for _ in range(4):
            prog = rand_program(rng, n, 40)
            diff = np.max(np.abs(final_state(prog, dev).psi - final_state_reference(prog, dev).psi))
            assert diff < 1e-12, (feats, n, diff)


def test_fast_pure_engine_on_ideal_algorithms_and_readout_only_device():
    ideal = DEVICES["ideal"]
    for alg in all_algorithms():
        prog = parse(alg["qasm"])
        diff = np.max(np.abs(final_state(prog, ideal).psi - final_state_reference(prog, ideal).psi))
        assert diff < 1e-12, (alg["name"], diff)
    ro = Device("ro", 3, readout_error=[(0.01, 0.02)] * 3)
    prog = parse("OPENQASM 2.0; qreg q[3]; creg c[3]; h q[0]; cx q[0],q[2]; swap q[1],q[2]; measure q -> c;")
    assert np.max(np.abs(final_state(prog, ro).psi - final_state_reference(prog, ro).psi)) < 1e-12
