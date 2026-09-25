import glob
import json
import pytest
from digital_qpu import QPU
from digital_qpu.__main__ import main

BELL = open("examples/bell.qasm").read()


def test_job_runs():
    job = QPU("ideal").run(BELL, shots=1000, seed=1)
    r = job.result()
    assert job.status == "DONE" and sum(r["counts"].values()) == 1000
    assert set(r["counts"]) <= {"00", "11"}
    assert r["depth"] == 2 and r["device"] == "ideal"


def test_seeded_runs_repeat():
    a = QPU("dq-5").run(BELL, shots=500, seed=7).result()["counts"]
    b = QPU("dq-5").run(BELL, shots=500, seed=7).result()["counts"]
    assert a == b


def test_failed_job_reports_error():
    job = QPU("dq-5").run("OPENQASM 2.0; qreg q[2]; creg c[2]; foo q[0];")
    assert job.status == "ERROR" and "foo" in job.error
    with pytest.raises(RuntimeError):
        job.result()


@pytest.mark.parametrize("path", sorted(glob.glob("examples/*.qasm")))
def test_every_example_runs_on_dq5(path):
    job = QPU("dq-5").run(open(path).read(), shots=200, seed=0)
    assert job.status == "DONE", job.error


def test_cli(capsys):
    assert main(["devices"]) == 0
    assert "dq-5" in capsys.readouterr().out
    assert main(["run", "examples/bell.qasm", "--device", "ideal", "--shots", "100", "--seed", "1", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert sum(res["counts"].values()) == 100
