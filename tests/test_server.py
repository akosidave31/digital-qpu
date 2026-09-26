"""v0.14.0: web API (standard library HTTP server, background job worker)."""
import json
import threading
import time
import urllib.error
import urllib.request
import pytest
import digital_qpu
from digital_qpu import server

BELL = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;"
_open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open     # never use a proxy for localhost


@pytest.fixture(scope="module")
def base():
    httpd = server.make_server("127.0.0.1", 0, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.service.stop()
    httpd.server_close()


def call(base, path, body=None, raw=None, method=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if data is not None else "GET"))
    try:
        with _open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait(base, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, job = call(base, f"/jobs/{jid}")
        if job["status"] in ("DONE", "ERROR"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {jid} did not finish")


def test_info_and_devices(base):
    code, info = call(base, "/")
    assert code == 200 and info["version"] == digital_qpu.__version__ and "POST /jobs" in info["endpoints"]
    code, d = call(base, "/devices")
    names = {x["name"]: x["n_qubits"] for x in d["devices"]}
    assert code == 200 and names["dq-5"] == 5 and names["dq-12"] == 12 and names["ideal"] == 20


def test_submit_and_read_result(base):
    code, job = call(base, "/jobs", {"qasm": BELL, "device": "ideal", "shots": 200, "seed": 1})
    assert code == 202 and job["status"] == "QUEUED" and job["url"] == f"/jobs/{job['job_id']}"
    done = wait(base, job["job_id"])
    assert done["status"] == "DONE" and done["error"] is None
    r = done["result"]
    assert r["job_id"] == job["job_id"] and sum(r["counts"].values()) == 200 and set(r["counts"]) <= {"00", "11"}
    assert done["started"] >= done["submitted"] and done["finished"] >= done["started"]


def test_same_seed_same_counts_on_noisy_chip(base):
    spec = {"qasm": BELL, "device": "dq-5", "shots": 500, "seed": 7, "router": "lookahead"}
    a = wait(base, call(base, "/jobs", spec)[1]["job_id"])
    b = wait(base, call(base, "/jobs", spec)[1]["job_id"])
    assert a["status"] == b["status"] == "DONE" and a["result"]["counts"] == b["result"]["counts"]
    assert a["result"]["compiled"]["router"] == "lookahead"


def test_mitigation_through_the_api(base):
    job = wait(base, call(base, "/jobs", {"qasm": BELL, "device": "dq-5", "shots": 500, "seed": 2,
                                          "mitigate": "readout", "day": 3})[1]["job_id"])
    assert job["status"] == "DONE" and job["result"]["device"] == "dq-5@day3"
    m = job["result"]["mitigated"]
    assert abs(sum(m.values()) - 1) < 1e-9


@pytest.mark.parametrize("body,fragment", [
    ({}, "'qasm'"),
    ({"qasm": "OPENQASM 2.0; qreg q[2]; creg c[2]; foo q[0]; measure q -> c;"}, "invalid program"),
    ({"qasm": BELL, "device": "dq-99"}, "unknown device"),
    ({"qasm": BELL, "shots": 0}, "'shots'"),
    ({"qasm": BELL, "shots": server.MAX_SHOTS + 1}, "'shots'"),
    ({"qasm": BELL, "shots": True}, "'shots'"),
    ({"qasm": BELL, "router": "magic"}, "'router'"),
    ({"qasm": BELL, "mitigate": "magic"}, "'mitigate'"),
    ({"qasm": BELL, "trajectories": 0}, "'trajectories'"),
    ({"qasm": BELL, "seed": -1}, "'seed'"),
    ({"qasm": BELL, "colour": "blue"}, "unknown field"),
    ({"qasm": "OPENQASM 2.0; qreg q[6]; creg c[6]; h q[0]; measure q -> c;", "device": "dq-5"}, "6 qubits"),
    ([1, 2], "JSON object"),
])
def test_bad_submissions_are_rejected_immediately(base, body, fragment):
    code, err = call(base, "/jobs", body)
    assert code == 400 and fragment in err["error"], err


def test_invalid_json_and_unknown_paths(base):
    assert call(base, "/jobs", raw=b"{not json")[0] == 400
    assert call(base, "/jobs/doesnotexist")[0] == 404
    assert call(base, "/nothing")[0] == 404
    assert call(base, "/devices", {"x": 1})[0] == 404


def test_full_queue_answers_429(base, monkeypatch):
    monkeypatch.setattr(server, "MAX_QUEUE", 0)
    code, err = call(base, "/jobs", {"qasm": BELL, "device": "ideal"})
    assert code == 429 and "queue full" in err["error"]


def test_job_list_is_newest_first_without_results(base):
    jid = call(base, "/jobs", {"qasm": BELL, "device": "ideal", "shots": 10})[1]["job_id"]
    wait(base, jid)
    code, lst = call(base, "/jobs")
    assert code == 200 and lst["jobs"][0]["job_id"] == jid and "result" not in lst["jobs"][0]


def test_server_answers_while_a_job_runs(base):
    from digital_qpu import shor15
    code, job = call(base, "/jobs", {"qasm": shor15()["qasm"], "device": "dq-12", "shots": 10, "seed": 1})
    assert code == 202
    time.sleep(0.3)                                   # let the worker start the (several-second) job
    t0 = time.time()
    code, _ = call(base, "/devices")
    assert code == 200 and time.time() - t0 < 2.0
    assert wait(base, job["job_id"])["status"] == "DONE"


def test_old_finished_jobs_are_dropped(monkeypatch):
    monkeypatch.setattr(server, "MAX_KEEP", 3)
    s = server.Service()
    try:
        ids = [s.submit(server.validate({"qasm": BELL, "device": "ideal", "shots": 5}))["job_id"] for _ in range(6)]
        t0 = time.time()
        while any((s.get(i) or {"status": "DONE"})["status"] not in ("DONE", "ERROR") for i in ids):
            assert time.time() - t0 < 60
            time.sleep(0.02)
        s.submit(server.validate({"qasm": BELL, "device": "ideal", "shots": 5}))
        assert len(s.jobs) <= 3 and s.get(ids[0]) is None
    finally:
        s.stop()


def test_algorithms_endpoint_lists_ready_made_circuits(base):
    code, d = call(base, "/algorithms")
    names = [x["name"] for x in d["algorithms"]]
    assert code == 200 and "Grover search (3 qubits, exact)" in names and "Grover search (3 qubits, 1 round)" in names
    exact = next(x for x in d["algorithms"] if x["name"].endswith("exact)"))
    job = wait(base, call(base, "/jobs", {"qasm": exact["qasm"], "device": "ideal", "shots": 300, "seed": 1})[1]["job_id"])
    assert job["status"] == "DONE" and job["result"]["counts"] == {exact["expected"]: 300}
