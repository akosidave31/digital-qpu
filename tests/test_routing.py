"""v0.11.0: look-ahead router with smart initial placement."""
import numpy as np
import pytest
from digital_qpu import parse, probabilities, transpile, Device, DEVICES, QPU
from digital_qpu.algorithms import all_algorithms, shor15, grover3

NATIVE = ("rz", "sx", "x", "cz")
LINE5 = Device("line5", 5, coupling=[(0, 1), (1, 2), (2, 3), (3, 4)], native_gates=NATIVE, virtual_rz=True)
LADDER = Device("ladder", 12, coupling=list(DEVICES["dq-12"].coupling), native_gates=NATIVE, virtual_rz=True)


def rand_qasm(rng, n, depth, p2=0.5):
    L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
    for _ in range(depth):
        if rng.random() < p2:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
        else:
            L.append(f"{['h', 't', 'sx', 's'][int(rng.integers(4))]} q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return " ".join(L)


def test_never_more_swaps_than_basic_router():
    rng = np.random.default_rng(11)
    for dev, sizes in ((LINE5, (3, 4, 5)), (LADDER, (5, 8, 12))):
        for n in sizes:
            for _ in range(8):
                prog = parse(rand_qasm(rng, n, 30))
                basic = transpile(prog, dev, router="basic")[1]["swaps"]
                assert transpile(prog, dev)[1]["swaps"] <= basic


def test_lookahead_results_are_identical_on_a_clean_chip():
    rng = np.random.default_rng(12)
    for dev, sizes in ((LINE5, (2, 3, 5)), (LADDER, (4, 6))):
        for n in sizes:
            for _ in range(6):
                prog = parse(rand_qasm(rng, n, 25))
                native, info = transpile(prog, dev, router="lookahead")
                assert info["router"] == "lookahead"
                P0, P1 = probabilities(prog, DEVICES["ideal"]), probabilities(native, dev)
                for k in set(P0) | set(P1):
                    assert abs(P0.get(k, 0) - P1.get(k, 0)) < 1e-9
                for o in native.ops:
                    if len(o.qubits) == 2:
                        assert dev.allows(*o.qubits)


def test_program_stays_in_the_smallest_connected_region():
    prog = parse(shor15()["qasm"])
    native, info = transpile(prog, LADDER, router="lookahead")
    assert native.n_qubits == 8 and max(info["layout"]) < 8


def test_famous_algorithms_still_correct_with_new_router():
    for alg in all_algorithms():
        if alg["qubits"] > 5:
            continue
        prog = parse(alg["qasm"])
        P0 = probabilities(prog, DEVICES["ideal"])
        native, _ = transpile(prog, LINE5)
        P1 = probabilities(native, LINE5)
        assert all(abs(P0.get(k, 0) - P1.get(k, 0)) < 1e-9 for k in set(P0) | set(P1)), alg["name"]


def test_grover_and_shor_do_not_get_worse():
    g = parse(grover3()["qasm"])
    assert transpile(g, DEVICES["dq-5"])[1]["swaps"] <= transpile(g, DEVICES["dq-5"], router="basic")[1]["swaps"]
    s = parse(shor15()["qasm"])
    assert transpile(s, DEVICES["dq-12"])[1]["swaps"] <= transpile(s, DEVICES["dq-12"], router="basic")[1]["swaps"]


def test_unknown_router_is_rejected():
    prog = parse("OPENQASM 2.0; qreg q[2]; creg c[2]; cx q[0], q[1]; measure q -> c;")
    with pytest.raises(ValueError):
        transpile(prog, LINE5, router="magic")
    assert QPU("dq-5").run("OPENQASM 2.0; qreg q[2]; creg c[2]; cx q[0], q[1]; measure q -> c;",
                           router="magic").status == "ERROR"


def test_cli_routing_report(capsys):
    from digital_qpu.__main__ import main
    assert main(["routing", "--devices", "dq-5"]) == 0
    out = capsys.readouterr().out
    assert "Grover" in out and "never needs more SWAPs" in out


def test_shor_labels(capsys):
    from digital_qpu.__main__ import main
    assert main(["shor", "--shots", "400"]) == 0
    out = capsys.readouterr().out
    assert "period found" in out and "15 = 3 x 5" in out
    assert "7^2 mod 15 = 1" not in out                  # y = 8 gives r = 2, which is not the period
