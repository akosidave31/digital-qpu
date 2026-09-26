"""Web API: submit jobs over HTTP, like a quantum cloud service. Standard library only.

    python -m digital_qpu serve [--host 127.0.0.1] [--port 8000]

Endpoints (JSON in, JSON out):
    GET  /               service info
    GET  /devices        the available chips
    POST /jobs           submit {"qasm": "...", "device": "dq-5", "shots": 1024, ...} -> 202 {"job_id", ...}
    GET  /jobs           recent jobs, newest first (without results)
    GET  /jobs/<id>      a job's status, and its result when DONE

Optional job fields: seed, day (calibration day), mitigate (readout | learned | learned-linear),
router (auto | lookahead | basic), trajectories. Requests are checked when submitted, so a bad
program gets an immediate 400 with the reason instead of a failed job later.

Jobs run one at a time in a background worker thread (the simulator is CPU-bound), so the server
keeps answering while a job runs. There is NO authentication: keep the default host 127.0.0.1
(this device only) or use --host 0.0.0.0 on a trusted network only."""
import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import __version__
from .device import DEVICES
from .qasm import parse
from .qpu import QPU

MAX_BODY = 200_000          # bytes per request
MAX_SHOTS = 100_000
MAX_TRAJECTORIES = 2000
MAX_QUEUE = 50              # jobs waiting to run
MAX_KEEP = 500              # finished jobs kept in memory (oldest dropped first)
ROUTERS = ("auto", "lookahead", "basic")
MITIGATIONS = (None, "readout", "learned", "learned-linear")
FIELDS = {"qasm", "device", "shots", "seed", "day", "mitigate", "router", "trajectories"}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate(body):
    """Check a submission. Returns the job spec, or raises ValueError with a readable reason."""
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    unknown = set(body) - FIELDS
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(unknown)}; allowed: {sorted(FIELDS)}")
    qasm = body.get("qasm")
    if not isinstance(qasm, str) or not qasm.strip():
        raise ValueError("'qasm' (OpenQASM 2.0 program text) is required")
    device = body.get("device", "dq-5")
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; available: {sorted(DEVICES)}")
    shots = body.get("shots", 1024)
    if not _is_int(shots) or not 1 <= shots <= MAX_SHOTS:
        raise ValueError(f"'shots' must be an integer from 1 to {MAX_SHOTS}")
    seed = body.get("seed")
    if seed is not None and (not _is_int(seed) or seed < 0):
        raise ValueError("'seed' must be a non-negative integer or null")
    day = body.get("day")
    if day is not None and (not _is_int(day) or day < 0):
        raise ValueError("'day' must be a non-negative integer or null")
    traj = body.get("trajectories", 300)
    if not _is_int(traj) or not 1 <= traj <= MAX_TRAJECTORIES:
        raise ValueError(f"'trajectories' must be an integer from 1 to {MAX_TRAJECTORIES}")
    mitigate = body.get("mitigate")
    if mitigate not in MITIGATIONS:
        raise ValueError(f"'mitigate' must be one of {list(MITIGATIONS)}")
    router = body.get("router", "auto")
    if router not in ROUTERS:
        raise ValueError(f"'router' must be one of {list(ROUTERS)}")
    try:
        program = parse(qasm)
    except Exception as e:
        raise ValueError(f"invalid program: {e}") from None
    if program.n_qubits > DEVICES[device].n_qubits:
        raise ValueError(f"program uses {program.n_qubits} qubits; {device} has {DEVICES[device].n_qubits}")
    return {"qasm": qasm, "device": device, "shots": shots, "seed": seed, "day": day,
            "mitigate": mitigate, "router": router, "trajectories": traj}


def _plain(o):
    """Make numpy values JSON-friendly."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class Service:
    """Job store plus one background worker."""

    def __init__(self):
        self.jobs = {}                      # job id -> record (insertion order = submission order)
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def submit(self, spec):
        with self.lock:
            waiting = sum(j["status"] == "QUEUED" for j in self.jobs.values())
            if waiting >= MAX_QUEUE:
                raise OverflowError(f"queue full ({MAX_QUEUE} jobs waiting); try again later")
            jid = uuid.uuid4().hex[:12]
            self.jobs[jid] = {"job_id": jid, "status": "QUEUED", "device": spec["device"],
                              "shots": spec["shots"], "submitted": time.time(), "started": None,
                              "finished": None, "error": None, "result": None}
            self._trim()
            record = dict(self.jobs[jid])
        self.queue.put((jid, spec))
        return record

    def _trim(self):
        finished = [k for k, j in self.jobs.items() if j["status"] in ("DONE", "ERROR")]
        while len(self.jobs) > MAX_KEEP and finished:
            del self.jobs[finished.pop(0)]

    def _work(self):
        while True:
            jid, spec = self.queue.get()
            if jid is None:
                return
            with self.lock:
                job = self.jobs.get(jid)
                if job is None:
                    continue
                job["status"], job["started"] = "RUNNING", time.time()
            try:
                result = QPU(spec["device"], day=spec["day"]).run(
                    spec["qasm"], shots=spec["shots"], seed=spec["seed"], mitigate=spec["mitigate"],
                    n_traj=spec["trajectories"], router=spec["router"]).result()
                result = json.loads(json.dumps(result, default=_plain))
                result["job_id"] = jid
                update = {"status": "DONE", "result": result}
            except Exception as e:
                update = {"status": "ERROR", "error": str(e)}
            with self.lock:
                job.update(update)
                job["finished"] = time.time()

    def get(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            return None if job is None else dict(job)

    def recent(self, limit=50):
        with self.lock:
            jobs = list(self.jobs.values())[-limit:]
        return [{k: v for k, v in j.items() if k != "result"} for j in reversed(jobs)]

    def stop(self):
        self.queue.put((None, None))


ENDPOINTS = {"GET /": "service info", "GET /devices": "available chips",
             "POST /jobs": "submit a job: {qasm, device, shots, seed, day, mitigate, router, trajectories}",
             "GET /jobs": "recent jobs, newest first", "GET /jobs/<id>": "status and result"}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"digital-qpu/{__version__}"

        def log_message(self, fmt, *args):
            if not getattr(self.server, "quiet", False):
                super().log_message(fmt, *args)

        def _send(self, code, obj):
            data = json.dumps(obj, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code, message):
            self._send(code, {"error": message})

        def _path(self):
            return self.path.split("?", 1)[0].rstrip("/") or "/"

        def do_GET(self):
            path = self._path()
            if path == "/":
                return self._send(200, {"service": "digital-qpu", "version": __version__, "endpoints": ENDPOINTS})
            if path == "/devices":
                return self._send(200, {"devices": [{"name": d.name, "n_qubits": d.n_qubits,
                                                     "description": d.description} for d in DEVICES.values()]})
            if path == "/jobs":
                return self._send(200, {"jobs": service.recent()})
            if path.startswith("/jobs/"):
                job = service.get(path[len("/jobs/"):])
                return self._send(200, job) if job else self._error(404, "no such job")
            self._error(404, "not found; see GET / for the endpoints")

        def do_POST(self):
            if self._path() != "/jobs":
                return self._error(404, "not found; submit jobs with POST /jobs")
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return self._error(400, "bad Content-Length")
            if n > MAX_BODY:
                return self._error(413, f"request too large (max {MAX_BODY} bytes)")
            try:
                body = json.loads(self.rfile.read(n) or b"null")
            except ValueError:
                return self._error(400, "body must be valid JSON")
            try:
                spec = validate(body)
            except ValueError as e:
                return self._error(400, str(e))
            try:
                job = service.submit(spec)
            except OverflowError as e:
                return self._error(429, str(e))
            job["url"] = f"/jobs/{job['job_id']}"
            self._send(202, job)

    return Handler


def make_server(host="127.0.0.1", port=8000, quiet=False):
    """An HTTP server with its job service attached (port 0 = pick a free port)."""
    service = Service()
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.daemon_threads = True
    httpd.service, httpd.quiet = service, quiet
    return httpd


def serve(host="127.0.0.1", port=8000):
    httpd = make_server(host, port)
    print(f"digital-qpu {__version__} API on http://{host}:{httpd.server_address[1]}  (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost"):
        print("warning: no authentication - anyone who can reach this address can submit jobs")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.service.stop()
        httpd.server_close()
