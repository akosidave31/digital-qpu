import math
import numpy as np
import pytest
from digital_qpu import (parse, probabilities, final_state, DEVICES, QPU, all_algorithms, bernstein_vazirani,
                         deutsch_jozsa, grover3, phase_estimation, shor15, shor_factors)
from digital_qpu.algorithms import _ccx, _cswap, _cp, HEAD, FILES

IDEAL = DEVICES["ideal"]


def prog(n, body, measure=True):
    text = "\n".join([HEAD, f"qreg q[{n}];", f"creg c[{n}];"] + body)
    if measure:
        text += "\n" + "\n".join(f"measure q[{i}] -> c[{i}];" for i in range(n))
    return parse(text)


def key_of(bits):                    # bits[i] = value of qubit i  ->  Qiskit-order key
    return "".join(str(b) for b in reversed(bits))


def test_toffoli_truth_table():
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                prep = [f"x q[{i}];" for i, v in enumerate((a, b, c)) if v]
                P = probabilities(prog(3, prep + _ccx(0, 1, 2)), IDEAL)
                assert abs(P[key_of((a, b, c ^ (a & b)))] - 1) < 1e-9


def test_controlled_swap_truth_table():
    for c in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                prep = [f"x q[{i}];" for i, v in enumerate((c, a, b)) if v]
                out = (c, b, a) if c else (c, a, b)
                P = probabilities(prog(3, prep + _cswap(0, 1, 2)), IDEAL)
                assert abs(P[key_of(out)] - 1) < 1e-9


def test_controlled_phase_is_exact():
    for theta in (0.3, -1.1, math.pi / 4):
        psi = final_state(prog(2, ["h q[0];", "h q[1];"] + _cp(theta, 0, 1), measure=False), IDEAL).psi
        expect = 0.5 * np.array([1, 1, 1, np.exp(1j * theta)])
        assert abs(abs(np.vdot(expect, psi)) - 1) < 1e-9


def ideal_probs(alg):
    return probabilities(parse(alg["qasm"]), IDEAL)


def test_bernstein_vazirani_finds_every_secret():
    for s in range(8):
        a = bernstein_vazirani(s)
        assert abs(ideal_probs(a)[a["expected"]] - 1) < 1e-9


def test_deutsch_jozsa():
    assert abs(ideal_probs(deutsch_jozsa("constant"))["000"] - 1) < 1e-9
    assert ideal_probs(deutsch_jozsa("balanced")).get("000", 0) < 1e-9


def test_grover_success_probability():
    expect = math.sin(5 * math.asin(1 / math.sqrt(8))) ** 2           # 0.945 after 2 iterations
    for m in range(8):
        a = grover3(m)
        assert abs(ideal_probs(a)[a["expected"]] - expect) < 1e-9


def test_phase_estimation_reads_every_phase():
    for k in range(8):
        a = phase_estimation(k)
        assert abs(ideal_probs(a)[a["expected"]] - 1) < 1e-9


def test_shor_factors_15():
    P = ideal_probs(shor15())
    for k in ("0000", "0100", "1000", "1100"):
        assert abs(P[k] - 0.25) < 1e-9
    r, f = shor_factors({k: int(round(v * 1000)) for k, v in P.items()})
    assert r == 4 and f == [3, 5]
    counts = QPU("ideal").run(shor15()["qasm"], shots=500, seed=2).result()["counts"]
    assert shor15()["answer"](counts) == "3 x 5"


@pytest.mark.parametrize("builder", [bernstein_vazirani, grover3, phase_estimation,
                                     lambda: deutsch_jozsa("balanced"), lambda: deutsch_jozsa("constant")])
def test_noisy_chip_still_gets_the_right_answer(builder):
    a = builder()
    counts = QPU("dq-5").run(a["qasm"], shots=2000, seed=3).result()["counts"]
    assert a["answer"](counts) == a["expected"]
    assert a["success"](counts) < 1.0                                  # but not perfectly: it is noisy


def test_example_files_match_the_generators():
    for name, fn in FILES.items():
        a = fn()
        text = open(f"examples/algorithms/{name}").read()
        assert text.endswith(a["qasm"])


def test_cli(capsys):
    from digital_qpu.__main__ import main
    assert main(["algorithms", "--shots", "500"]) == 0
    out = capsys.readouterr().out
    assert out.count("OK") >= 6 and "needs 8 qubits" in out
    assert main(["shor", "--shots", "500"]) == 0
    assert "15 = 3 x 5" in capsys.readouterr().out
