"""The virtual quantum computer: submit a program, get a job, read the result."""
import time
import uuid
import numpy as np
from .qasm import parse
from .device import get_device
from .executor import schedule, layer_duration, probabilities, sample_counts
from .compiler import transpile
from .calibration import calibrate


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
    def __init__(self, device="dq-5", day=None):
        """day: use the device's calibration on that day (None = nominal values)."""
        self.device = calibrate(get_device(device), day)

    def run(self, qasm, shots=1024, seed=None, compile=True, mitigate=None):
        """compile=True (default): devices with native gates get the program transpiled first.
        mitigate="readout": also return readout-mitigated probabilities."""
        job = Job(uuid.uuid4().hex[:12])
        t0 = time.time()
        try:
            job.status = "RUNNING"
            program = parse(qasm)
            info = None
            if compile and self.device.native_gates is not None:
                program, info = transpile(program, self.device)
                bad = {o.name for o in program.ops} - set(self.device.native_gates)
                if bad:
                    raise RuntimeError(f"compiler produced non-native gates: {bad}")
            layers = schedule(program, self.device)
            probs = probabilities(program, self.device)
            counts = sample_counts(probs, shots, np.random.default_rng(seed))
            job._result = {
                "job_id": job.id, "device": self.device.name, "shots": shots,
                "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                "n_qubits": program.n_qubits, "depth": len(layers),
                "circuit_time": float(sum(layer_duration(L, self.device) for L in layers)),
                "compiled": info,
                "clbit_qubits": {c: q for q, c in program.measures.items()},
                "n_clbits": max(program.n_clbits, max(program.measures.values()) + 1),
                "elapsed_s": round(time.time() - t0, 4)}
            if mitigate == "readout":
                from .mitigation import readout_mitigate, distribution
                job._result["mitigated"] = readout_mitigate(distribution(counts), self.device,
                                                            job._result["clbit_qubits"], job._result["n_clbits"])
            elif mitigate is not None:
                raise ValueError(f"unknown mitigation '{mitigate}' (available: 'readout')")
            job.status = "DONE"
        except Exception as e:
            job.status, job.error = "ERROR", str(e)
        return job
