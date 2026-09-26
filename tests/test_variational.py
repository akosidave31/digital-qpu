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
