"""v0.20.0: the three Grover versions from the v0.16-v0.19 research, usable everywhere."""
import math
import pytest
from digital_qpu import (parse, probabilities, transpile, DEVICES, all_algorithms, grover_variant,
                         exact_grover_phase, GROVER_VARIANTS)

IDEAL, DQ5 = DEVICES["ideal"], DEVICES["dq-5"]
TH = math.asin(1 / math.sqrt(8))


def p_marked(alg, dev=IDEAL):
    return probabilities(parse(alg["qasm"]), dev).get(alg["expected"], 0.0)


def test_each_version_has_its_theoretical_success_for_every_marked_item():
    for m in range(8):
        assert abs(p_marked(grover_variant("standard", m)) - math.sin(5 * TH) ** 2) < 1e-9
        assert abs(p_marked(grover_variant("1round", m)) - math.sin(3 * TH) ** 2) < 1e-9
        assert p_marked(grover_variant("exact", m)) > 0.9999


def test_exact_phase_matches_the_research_code():
    from digital_qpu.variational import long_phase
    assert abs(exact_grover_phase(2) - long_phase(2)) < 1e-12 and exact_grover_phase(1) is None


def test_exact_version_costs_the_same_two_qubit_gates():
    n2 = lambda v: transpile(parse(grover_variant(v)["qasm"]), DQ5)[1]["n_2q"]
    assert n2("exact") == n2("standard") and n2("1round") < n2("standard")


def test_algorithm_lists():
    names = lambda g: [a["name"] for a in all_algorithms(g) if a["name"].startswith("Grover")]
    assert names("standard") == ["Grover search (3 qubits)"]
    assert len(names("all")) == 3 and GROVER_VARIANTS == ("standard", "exact", "1round")
    with pytest.raises(ValueError):
        grover_variant("magic")


def test_cli_shows_all_three(capsys):
    from digital_qpu.__main__ import main
    assert main(["algorithms", "--grover", "all", "--shots", "300"]) == 0
    out = capsys.readouterr().out
    assert "Grover search (3 qubits, exact)" in out and "Grover search (3 qubits, 1 round)" in out
    assert "phase 2.1" in out
