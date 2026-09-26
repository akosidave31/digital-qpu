import json
from digital_qpu.perf import run, save, line_device, ghz
from digital_qpu import probabilities, transpile


def test_quick_speed_benchmark_runs(tmp_path):
    res = run(quick=True)
    assert [r["qubits"] for r in res["ideal"]] == [10, 14]
    assert all(r["seconds"] > 0 for r in res["ideal"] + res["noisy"] + res["features"])
    assert res["features"][0]["noise"] == "none" and len(res["features"]) == 5
    assert 0 < sum(r["share"] for r in res["profile_top"]) <= 1.0 + 1e-9
    save(res, tmp_path / "b.json")
    assert json.load(open(tmp_path / "b.json"))["ideal"][0]["qubits"] == 10


def test_benchmark_devices_are_valid():
    dev = line_device(4)
    P = probabilities(transpile(ghz(4), dev)[0], dev)
    assert abs(sum(P.values()) - 1) < 1e-9 and P["0000"] + P["1111"] > 0.8
