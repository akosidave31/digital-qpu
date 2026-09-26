"""v0.16.0: trainable Grover circuit."""
import math
import numpy as np
from digital_qpu import DEVICES
from digital_qpu.variational import VariationalGrover, zyz, rot, fixed_grover_success, summarize

IDEAL, DQ5 = DEVICES["ideal"], DEVICES["dq-5"]


def test_zyz_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(200):
        U, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        assert abs(abs(np.trace(U.conj().T @ rot(*zyz(U)))) - 2) < 1e-9
    for U in (np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1, -1]), np.array([[1, 1], [1, -1]]) / math.sqrt(2)):
        assert abs(abs(np.trace(np.conj(U).T @ rot(*zyz(U)))) - 2) < 1e-9


def test_grover_is_one_point_of_the_trainable_circuit():
    th = math.asin(1 / math.sqrt(8))
    for rounds in (2, 1):
        vg = VariationalGrover(rounds=rounds)
        assert vg.n_params == 9 + 18 * rounds
        s = vg.success(vg.grover_init(), IDEAL)
        assert abs(s - math.sin((2 * rounds + 1) * th) ** 2) < 1e-9
        assert abs(s - fixed_grover_success(IDEAL, rounds)) < 1e-9
    vg = VariationalGrover(rounds=2)
    assert abs(vg.success(vg.grover_init(), DQ5) - fixed_grover_success(DQ5, 2)) < 1e-6    # same on the noisy chip


def test_parameter_shift_matches_finite_differences_on_ideal():
    vg = VariationalGrover(rounds=1)
    p = np.random.default_rng(5).uniform(-3, 3, vg.n_params)
    g = vg.gradient(p, IDEAL)
    eps = 1e-4
    for k in range(vg.n_params):
        e = np.zeros(vg.n_params)
        e[k] = eps
        fd = (vg.success(p + e, IDEAL) - vg.success(p - e, IDEAL)) / (2 * eps)
        assert abs(fd - g[k]) < 1e-6, k


def test_training_improves_from_a_random_start():
    vg = VariationalGrover(rounds=1)
    _, hist = vg.train(IDEAL, epochs=10, lr=0.1, seed=7)
    assert max(hist) > hist[0] + 0.05


def test_summary_reads_the_results():
    runs = {k: {"start": 0.37, "best": b} for k, b in
            (("ideal/rounds2", 0.99), ("ideal/rounds1", 0.8), ("dq-5/rounds2", 0.5), ("dq-5/rounds1", 0.4))}
    test = {d: {"fixed_grover/rounds2": 0.37, "fixed_grover/rounds1": 0.4, "dq-5-trained/rounds2": 0.45 + 0.01 * d,
                "dq-5-trained/rounds1": 0.4, "ideal-trained/rounds2": 0.37, "ideal-trained/rounds1": 0.4}
            for d in range(1, 4)}
    lines = summarize({"runs": runs, "test": test})
    assert "MET" in lines[0] and "MET" in lines[1] and "MET" in lines[2]


# ---- v0.17.0: joint training and the permanent memorisation check ----
from digital_qpu.variational import JointGrover, per_item_spread, fixed_grover_mean, summarize_joint, SPREAD_LIMIT


def test_joint_objective_is_the_mean_over_all_marked_items():
    jg = JointGrover(rounds=2)
    p0 = jg.grover_init()
    items = jg.per_item(p0, IDEAL)
    assert len(items) == 8 and abs(jg.success(p0, IDEAL) - np.mean(items)) < 1e-12
    th = math.asin(1 / math.sqrt(8))
    assert all(abs(v - math.sin(5 * th) ** 2) < 1e-9 for v in items)          # Grover: same for every item
    assert abs(fixed_grover_mean(IDEAL, 2) - math.sin(5 * th) ** 2) < 1e-9
    assert per_item_spread(p0, 2)[1] < 1e-9


def test_joint_gradient_matches_finite_differences():
    jg = JointGrover(rounds=1)
    p = np.random.default_rng(11).uniform(-3, 3, jg.n_params)
    g = jg.gradient(p, IDEAL)
    eps = 1e-4
    for k in range(0, jg.n_params, 3):
        e = np.zeros(jg.n_params)
        e[k] = eps
        fd = (jg.success(p + e, IDEAL) - jg.success(p - e, IDEAL)) / (2 * eps)
        assert abs(fd - g[k]) < 1e-6, k


def test_memorisation_check_catches_single_item_training():
    """The v0.16.0 failure: trained on one marked item, the circuit favours that answer."""
    vg = VariationalGrover(marked=0b101, rounds=1)
    best, _ = vg.train(IDEAL, vg.grover_init(), epochs=10, lr=0.05)
    vals, spread = per_item_spread(best, 1)
    assert vals[0b101] == max(vals) and spread > SPREAD_LIMIT


def test_summary_ignores_specialised_circuits():
    run = lambda start, best, spread: {"start": start, "best": best, "ideal_spread": spread}
    runs = {"ideal/rounds2": run(0.945, 0.98, 0.02), "ideal/rounds1": run(0.78, 0.99, 0.5),
            "dq-5/rounds2": run(0.36, 0.40, 0.05), "dq-5/rounds1": run(0.40, 0.60, 0.6)}
    test = {d: {"fixed/rounds2": 0.34, "fixed/rounds1": 0.42, "ideal-trained/rounds2": 0.35,
                "dq-5-trained/rounds2": 0.45, "ideal-trained/rounds1": 0.5, "dq-5-trained/rounds1": 0.6}
            for d in range(1, 4)}
    lines = summarize_joint({"runs": runs, "test": test})
    text = "\n".join(lines)
    assert text.count("SPECIALISED") == 2
    assert "best valid trained 0.9800" in text                               # the 0.99 run does not count
    assert "dq-5-trained/rounds2 minus best fixed (fixed/rounds1)" in text   # the specialised 0.60 is ignored


# ---- v0.18.0: trainable phases (Long's exact Grover) ----
from digital_qpu.variational import PhaseGrover, long_phase, phase_factory, fixed_phase_form_mean, summarize_phase


def test_phase_form_at_pi_is_exactly_grover():
    th = math.asin(1 / math.sqrt(8))
    for layers in (False, True):
        for m in (0, 5, 7):
            pg = PhaseGrover(marked=m, rounds=2, train_layers=layers)
            assert abs(pg.success(pg.grover_init(math.pi), IDEAL) - math.sin(5 * th) ** 2) < 1e-9
    assert abs(fixed_phase_form_mean(IDEAL, 2) - math.sin(5 * th) ** 2) < 1e-9
    assert PhaseGrover(rounds=2).n_params == 4 and PhaseGrover(rounds=2, train_layers=True).n_params == 49


def test_hundred_percent_is_reachable_with_two_rounds():
    """Scan one common phase for all oracle and diffusion calls: some phase must give ~100%."""
    pg = PhaseGrover(marked=5, rounds=2)
    best = max(pg.success(np.full(4, phi), IDEAL) for phi in np.linspace(0, 2 * math.pi, 721))
    assert best > 0.999


def test_long_phase_formula():
    pg = PhaseGrover(marked=5, rounds=2)
    lp = long_phase(2)
    assert 2.12 < lp < 2.14 and long_phase(1) is None
    assert max(pg.success(np.full(4, lp), IDEAL), pg.success(np.full(4, 2 * math.pi - lp), IDEAL)) > 0.9999


def test_phase_gradient_matches_finite_differences():
    pg = PhaseGrover(marked=3, rounds=1, train_layers=True)
    p = np.random.default_rng(13).uniform(-3, 3, pg.n_params)
    g = pg.gradient(p, IDEAL)
    eps = 1e-4
    for k in list(pg.phase_params) + [0, 4, 11]:
        e = np.zeros(pg.n_params)
        e[k] = eps
        fd = (pg.success(p + e, IDEAL) - pg.success(p - e, IDEAL)) / (2 * eps)
        assert abs(fd - g[k]) < 1e-6, k


def test_phase_summary():
    run = lambda best, spread, phases=None: {"best": best, "ideal_spread": spread, "phases": phases or []}
    runs = {"ideal/rounds2/phases": run(0.999, 0.0, [2.13, 2.12, 2.14, 2.13]),
            "ideal/rounds2/phases+layers": run(0.9995, 0.3), "dq-5/rounds2/phases": run(0.40, 0.01),
            "dq-5/rounds1/phases": run(0.46, 0.01)}
    test = {d: {"fixed/rounds2": 0.35, "fixed/rounds1": 0.44, "fixed-phaseform/rounds2": 0.33,
                "fixed-phaseform/rounds1": 0.42, "ideal/rounds2/phases": 0.3, "ideal/rounds2/phases+layers": 0.3,
                "dq-5/rounds2/phases": 0.37, "dq-5/rounds1/phases": 0.47} for d in range(1, 4)}
    text = "\n".join(summarize_phase({"runs": runs, "test": test, "long_phase": long_phase(2)}))
    assert "T1 ideal: fixed Grover 0.9453 -> best valid trained 0.9990  target >= 0.99: MET" in text
    assert "within 0.1 rad: MET" in text and "SPECIALISED" in text
    assert "dq-5/rounds1/phases minus best fixed (fixed/rounds1)" in text


# ---- v0.19.0: the 6-CNOT phase gate ----
from digital_qpu.variational import _ccp_items, _ccp6_items, two_qubit_count, summarize_cheap, JointGrover
from digital_qpu.algorithms import _program, grover3
from digital_qpu import parse
from digital_qpu.executor import final_state


def _render(items, lam):
    return [it if isinstance(it, str) else f"{it[0]}({it[3] * lam!r}) q[{it[1]}];" for it in items]


def test_six_cnot_phase_gate_equals_eight_cnot_one_exactly():
    prep = ["ry(0.7) q[0];", "ry(1.9) q[1];", "ry(2.6) q[2];", "rz(0.4) q[1];"]
    for lam in (0.0, 0.8, math.pi, 2.13, 4.9, -1.1):
        a = final_state(parse(_program(3, 3, prep + _render(_ccp_items(0), lam), [])), IDEAL).psi
        b = final_state(parse(_program(3, 3, prep + _render(_ccp6_items(0), lam), [])), IDEAL).psi
        assert np.max(np.abs(np.asarray(a) - np.asarray(b))) < 1e-12, lam
    assert sum(isinstance(x, str) and x.startswith("cx") for x in _ccp6_items(0)) == 6
    assert sum(isinstance(x, str) and x.startswith("cx") for x in _ccp_items(0)) == 8


def test_six_cnot_exact_grover():
    th = math.asin(1 / math.sqrt(8))
    lp = long_phase(2)
    for m in range(8):
        pg = PhaseGrover(marked=m, rounds=2, form="cnot6")
        assert abs(pg.success(np.full(4, math.pi), IDEAL) - math.sin(5 * th) ** 2) < 1e-9
        assert pg.success(np.full(4, lp), IDEAL) > 0.9999


def test_six_cnot_form_costs_no_more_than_standard_grover():
    pg = PhaseGrover(marked=5, rounds=2, form="cnot6")
    six = two_qubit_count(pg.qasm(np.full(4, long_phase(2))), DQ5)
    eight = two_qubit_count(PhaseGrover(marked=5, rounds=2).qasm(np.full(4, long_phase(2))), DQ5)
    standard = two_qubit_count(grover3(0b101, 2)["qasm"], DQ5)
    assert six <= standard < eight


def test_six_cnot_gradient_matches_finite_differences():
    pg = PhaseGrover(marked=6, rounds=2, form="cnot6")
    p = np.array([1.1, 2.9, -0.4, 2.2])
    g = pg.gradient(p, IDEAL)
    eps = 1e-4
    for k in range(4):
        e = np.zeros(4)
        e[k] = eps
        fd = (pg.success(p + e, IDEAL) - pg.success(p - e, IDEAL)) / (2 * eps)
        assert abs(fd - g[k]) < 1e-6, k


def test_cheap_summary():
    rows = {d: {"fixed/rounds2": 0.35, "fixed/rounds1": 0.44, "exact6/rounds2": 0.37, "exact8/rounds2": 0.34,
                "dq-5-trained6/rounds2": 0.375, "dq-5-trained6/rounds1": 0.45} for d in range(1, 4)}
    out = {"ideal": {"exact6/rounds2": 1.0, "exact8/rounds2": 1.0, "fixed/rounds2": 0.945, "fixed/rounds1": 0.78},
           "two_qubit_gates": {"fixed/rounds2": 48, "fixed/rounds1": 24, "exact6/rounds2": 48, "exact8/rounds2": 60},
           "runs": {"dq-5-trained6/rounds2": {"ideal_spread": 0.0}, "dq-5-trained6/rounds1": {"ideal_spread": 0.0}},
           "test": rows}
    text = "\n".join(summarize_cheap(out))
    assert "T1" in text and "equal: MET" in text and "target >= +0.01: MET" in text
    assert "best untrained (fixed/rounds1)" in text
