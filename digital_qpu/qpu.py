"""The virtual quantum computer: submit a program, get a job, read the result."""
import time
import uuid
import numpy as np
from .qasm import parse
from .device import get_device
from .executor import schedule, layer_duration, probabilities, sample_counts


class Job:
    def __init__(self, job_id):
        self.id = job_id
        self.status = "QUEUED"
        self.error = None
        self._result = None

    def result(self):
        if self.status == "ERROR":
            raise RuntimeError(f"job {self.id} failed: {self.error}")
        return self._result


class QPU:
    def __init__(self, device="dq-5"):
        self.device = get_device(device)

    def run(self, qasm, shots=1024, seed=None):
        job = Job(uuid.uuid4().hex[:12])
        t0 = time.time()
        try:
            job.status = "RUNNING"
            program = parse(qasm)
            layers = schedule(program, self.device)
            probs = probabilities(program, self.device)
            counts = sample_counts(probs, shots, np.random.default_rng(seed))
            job._result = {
                "job_id": job.id, "device": self.device.name, "shots": shots,
                "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                "n_qubits": program.n_qubits, "depth": len(layers),
                "circuit_time": float(sum(layer_duration(L, self.device) for L in layers)),
                "elapsed_s": round(time.time() - t0, 4)}
            job.status = "DONE"
        except Exception as e:
            job.status, job.error = "ERROR", str(e)
        return job
