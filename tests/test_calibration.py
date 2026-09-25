import numpy as np
from digital_qpu import calibrate, bad_qubits, history, DEVICES, QPU, randomized_benchmarking
from digital_qpu.calibration import RHO, SIGMA, TLS_PROB

DQ5 = DEVICES["dq-5"]
DAYS = 365


def test_same_day_same_calibration_and_nominal_untouched():
    assert calibrate(DQ5, 12) == calibrate(DQ5, 12)
    assert calibrate(DQ5, 12).T1 != calibrate(DQ5, 13).T1
    assert calibrate(DQ5, None) is DQ5 and DQ5.T1 == [50.0, 45.0, 55.0, 48.0, 52.0]
    assert calibrate(DQ5, 7).name == "dq-5@day7"


def test_day_values_do_not_depend_on_how_far_we_look():
    a = calibrate(DQ5, 40)
    for d in range(0, 41, 5):
        calibrate(DQ5, d)
    assert calibrate(DQ5, 40) == a


def test_every_day_is_physical():
    for d in range(DAYS):
        c = calibrate(DQ5, d)
        assert all(t > 0 for t in c.T1 + c.T_phi)
        assert all(0 <= e <= 0.5 for e in c.gate_error_1q + list(c.gate_error_2q.values()))
        assert all(0 <= a <= 0.5 and 0 <= b <= 0.5 for a, b in c.readout_error)


def test_drift_statistics_are_realistic():
    logT1 = np.array([[np.log(calibrate(DQ5, d).T1[q] / DQ5.T1[q]) for q in range(5)] for d in range(DAYS)])
    ok = np.array([[q not in bad_qubits(DQ5, d) for q in range(5)] for d in range(DAYS)])
    for q in range(5):
        x = logT1[ok[:, q], q]
        assert abs(np.median(x)) < 0.08                       # centred on nominal
        assert 0.10 < np.std(x) < 0.21                         # ~15% spread
    both_ok = ok[:-1, 0] & ok[1:, 0]                         # bad days are independent: leave them out
    r = np.corrcoef(logT1[:-1, 0][both_ok], logT1[1:, 0][both_ok])[0, 1]
    assert 0.5 < r < 0.85                                      # today resembles yesterday (RHO = 0.7)
    rate = 1 - ok.mean()
    assert 0.02 < rate < 0.09                                  # bad days are rare (~5%)


def test_bad_days_really_hurt_T1():
    for d in range(DAYS):
        for q in bad_qubits(DQ5, d):
            assert calibrate(DQ5, d).T1[q] < 0.5 * DQ5.T1[q] * np.exp(4 * SIGMA["T1"])


def test_zz_is_stable():
    for d in range(0, DAYS, 7):
        for p, v in calibrate(DQ5, d).zz.items():
            assert abs(v / DQ5.zz[p] - 1) < 0.15


def test_rb_detects_a_bad_day():
    bad_days = [d for d in range(DAYS) if 0 in bad_qubits(DQ5, d)]
    day = min(bad_days, key=lambda d: calibrate(DQ5, d).T1[0])          # the worst bad day for q0
    good = next(d for d in range(DAYS) if not bad_qubits(DQ5, d) and abs(calibrate(DQ5, d).T1[0] / 50 - 1) < 0.1)
    kw = dict(lengths=(1, 30, 90, 180), n_seq=8, seed=2)
    bad_rb = randomized_benchmarking(calibrate(DQ5, day), 0, **kw)
    good_rb = randomized_benchmarking(calibrate(DQ5, good), 0, **kw)
    assert bad_rb["epg"] > 1.3 * good_rb["epg"]                         # benchmarking notices
    measured_ratio = bad_rb["epg"] / good_rb["epg"]
    expected_ratio = bad_rb["predicted_epg"] / good_rb["predicted_epg"]
    assert abs(measured_ratio / expected_ratio - 1) < 0.25               # and measures how much worse


def test_qpu_and_history():
    r = QPU("dq-5", day=3).run(open("examples/bell.qasm").read(), shots=200, seed=1).result()
    assert r["device"] == "dq-5@day3"
    h = history(DQ5, 0, 10)
    assert len(h) == 10 and all(row["T1"] > 0 for row in h)


def test_cli(capsys):
    from digital_qpu.__main__ import main
    assert main(["calibration", "--day", "5"]) == 0
    assert "dq-5@day5" in capsys.readouterr().out
    assert main(["history", "--qubit", "0", "--days", "5"]) == 0
    assert main(["run", "examples/bell.qasm", "--day", "5", "--shots", "50", "--seed", "1"]) == 0
