import numpy as np
from digital_qpu import evaluate_many, summarize


def fake_runs(diffs, readout=0.05):
    runs = []
    for d in diffs:
        rows = [{"raw_tvd": 0.09, "readout_tvd": readout, "linear_tvd": 0.035, "mlp_tvd": 0.035 - d, "floor": 0.009}
                for _ in range(10)]
        runs.append(rows)
    return runs


def test_summary_statistics():
    S = summarize(fake_runs([0.004, 0.006, 0.005, 0.005, 0.005]))
    assert S["n_runs"] == 5
    assert abs(S["mean"]["linear_tvd"] - 0.035) < 1e-12
    assert abs(S["paired"]["mean_diff"] - 0.005) < 1e-12
    assert S["paired"]["b_better_by_2se"]
    assert S["harm"]["linear_tvd"]["rate"] == 0.0


def test_no_verdict_from_noise():
    S = summarize(fake_runs([0.01, -0.01, 0.012, -0.008, 0.0]))
    assert not S["paired"]["b_better_by_2se"]


def test_harm_is_counted():
    runs = fake_runs([0.0, 0.0], readout=0.030)            # both models 0.035 > readout 0.030
    S = summarize(runs)
    assert S["harm"]["linear_tvd"]["rate"] == 1.0
    assert abs(S["harm"]["linear_tvd"]["mean_excess"] - 0.005) < 1e-12


def test_evaluate_many_small():
    runs = evaluate_many("dq-5", days=(0, 1), shots=500, n_train=30)
    assert len(runs) == 2 and all(len(rows) == 10 for rows in runs)
    S = summarize(runs)
    assert S["mean"]["floor"] < S["mean"]["readout_tvd"] < S["mean"]["raw_tvd"]
