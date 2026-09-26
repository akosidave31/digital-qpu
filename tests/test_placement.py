"""v0.21.0: noise-aware placement."""
import numpy as np
import pytest
from digital_qpu import parse, probabilities, transpile, Device, DEVICES
from digital_qpu.compiler import _connected_subsets, expected_log_fidelity, ROUTERS
from digital_qpu.placement import summarize_placement

LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]
NATIVE = ("rz", "sx", "x", "cz")
CLEAN = Device("line5", 5, coupling=LINE, native_gates=NATIVE, virtual_rz=True)
GHZ3 = "OPENQASM 2.0; qreg q[3]; creg c[3]; h q[0]; cx q[0], q[1]; cx q[1], q[2]; measure q -> c;"


def used(native):
    return {q for o in native.ops for q in o.qubits} | set(native.measures)


def test_connected_regions_of_a_line():
    adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
    assert _connected_subsets(adj, 3) == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert len(_connected_subsets(adj, 1)) == 5 and _connected_subsets(adj, 5) == [[0, 1, 2, 3, 4]]


def test_noise_aware_results_are_correct_on_a_clean_chip():
    rng = np.random.default_rng(21)
    for n in (2, 3, 4):
        for _ in range(4):
            L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
            for _ in range(20):
                if rng.random() < 0.4:
                    a, b = (int(x) for x in rng.choice(n, 2, replace=False))
                    L.append(f"cx q[{a}],q[{b}];")
                else:
                    L.append(f"{['h', 't', 'sx', 's'][int(rng.integers(4))]} q[{int(rng.integers(n))}];")
            L.append("measure q -> c;")
            prog = parse(" ".join(L))
            native, info = transpile(prog, CLEAN, router="noise-aware")
            P0, P1 = probabilities(prog, DEVICES["ideal"]), probabilities(native, CLEAN)
            assert all(abs(P0.get(k, 0) - P1.get(k, 0)) < 1e-9 for k in set(P0) | set(P1))


def test_noise_aware_avoids_a_bad_qubit():
    bad = Device("badq0", 5, T1=[2.0] + [50.0] * 4, T_phi=[2.0] + [40.0] * 4, gate_time_1q=0.02, gate_time_2q=0.15,
                 readout_error=[(0.3, 0.3)] + [(0.01, 0.03)] * 4, coupling=LINE, native_gates=NATIVE,
                 virtual_rz=True, gate_error_1q=[1e-3] * 5, gate_error_2q=0.01)
    prog = parse(GHZ3)
    auto, _ = transpile(prog, bad)
    smart, info = transpile(prog, bad, router="noise-aware")
    ok = lambda nat: sum(v for k, v in probabilities(nat, bad).items() if k in ("000", "111"))
    assert 0 in used(auto) and 0 not in used(smart) and info["router"] == "noise-aware"
    assert ok(smart) > ok(auto) + 0.1


def test_estimate_prefers_fewer_gates():
    dq5 = DEVICES["dq-5"]
    short, _ = transpile(parse("OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;"), dq5)
    long, _ = transpile(parse("OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0];" + " cx q[0], q[1];" * 5 +
                              " measure q -> c;"), dq5)
    assert expected_log_fidelity(long, dq5) < expected_log_fidelity(short, dq5) < 0


def test_router_is_available_everywhere(capsys):
    from digital_qpu.__main__ import main, ROUTERS as CLI
    from digital_qpu.server import ROUTERS as API
    assert "noise-aware" in ROUTERS and "noise-aware" in CLI and "noise-aware" in API
    assert main(["compile", "examples/bell.qasm", "--router", "noise-aware"]) == 0
    assert "noise-aware router" in capsys.readouterr().out
    with pytest.raises(ValueError):
        transpile(parse(GHZ3), CLEAN, router="magic")


def test_placement_summary():
    rows = [{"diff": d, "auto_uses_bad": b, "algorithm": "X", "auto_qubits": [0, 1], "noise-aware_qubits": q}
            for d, b, q in [(0.05, True, [2, 3])] * 5 + [(0.0, False, [0, 1])] * 15]
    text = "\n".join(summarize_placement(rows))
    assert "target >= +0.5: MET" in text and "target >= +3: MET" in text and "target <= 10%: MET" in text
