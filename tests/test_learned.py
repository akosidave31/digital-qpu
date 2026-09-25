import numpy as np
import pytest
from digital_qpu import parse, transpile, DEVICES, QPU, evaluate
from digital_qpu.learned import (circuit_features, surviving_signal, undo_blur, training_data,
                                 LearnedMitigator, _measured_keys)

DQ5 = DEVICES["dq-5"]


@pytest.fixture(scope="module")
def data():
    X, y = training_data(DQ5, 120, seed=1000)
    Xh, yh = training_data(DQ5, 40, seed=2000)            # held out
    return X, y, Xh, yh


@pytest.fixture(scope="module")
def mitigators(data):
    X, y, _, _ = data
    return {k: LearnedMitigator(k).fit(X, y) for k in ("linear", "mlp")}


def test_blur_model_is_exactly_invertible():
    cq, ncl = {0: 0, 1: 1}, 2
    keys = _measured_keys(cq, ncl)
    ideal = {"00": 0.5, "11": 0.5}
    f = 0.7
    noisy = {k: f * ideal.get(k, 0) + (1 - f) / 4 for k in keys}
    assert abs(surviving_signal(noisy, ideal, keys) - f) < 1e-12
    back = undo_blur(noisy, f, cq, ncl)
    assert all(abs(back.get(k, 0) - ideal.get(k, 0)) < 1e-12 for k in keys)


def test_features_grow_with_the_circuit():
    small = transpile(parse("OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;"), DQ5)[0]
    big = transpile(parse(open("examples/ghz5.qasm").read()), DQ5)[0]
    fs, fb = circuit_features(small, DQ5), circuit_features(big, DQ5)
    assert np.all(fb[:4] > fs[:4]) and fb[4] == 5 and fs[4] == 2


def test_signal_shrinks_as_error_budget_grows(data):
    X, y, _, _ = data
    assert np.all((y > 0) & (y <= 1))
    assert np.corrcoef(X[:, 0], y)[0, 1] < -0.5            # more 2-qubit gate error -> less signal


@pytest.mark.parametrize("kind,min_corr", [("linear", 0.8), ("mlp", 0.6)])
def test_models_predict_signal_on_unseen_circuits(data, mitigators, kind, min_corr):
    _, _, Xh, yh = data
    pred = mitigators[kind].predict_f(Xh)
    assert np.corrcoef(pred, yh)[0, 1] > min_corr


def test_release_criterion_learned_beats_readout_baseline(mitigators):
    rows = evaluate("dq-5", shots=4000, mitigators=mitigators)
    avg = lambda k: np.mean([r[k] for r in rows])
    print({k: round(avg(k), 4) for k in ("raw_tvd", "readout_tvd", "linear_tvd", "mlp_tvd", "floor")})
    assert avg("linear_tvd") < avg("readout_tvd")
    assert avg("floor") < avg("readout_tvd")


def test_qpu_learned_mitigation():
    r = QPU("dq-5").run(open("examples/ghz5.qasm").read(), shots=2000, seed=3, mitigate="learned").result()
    m = r["mitigated"]
    assert all(v >= 0 for v in m.values()) and abs(sum(m.values()) - 1) < 1e-9
    assert m.get("00000", 0) + m.get("11111", 0) > 0.9
