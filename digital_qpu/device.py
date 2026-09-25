"""Device description: like a real chip's calibration sheet.
Times are in the same abstract units as T1/T2 (think microseconds)."""
from dataclasses import dataclass, field


@dataclass
class Device:
    name: str
    n_qubits: int
    T1: list = None                 # per qubit; None = no energy loss
    T_phi: list = None              # per qubit; None = no pure dephasing
    gate_time_1q: float = 0.0
    gate_time_2q: float = 0.0
    readout_error: list = None      # per qubit (P(read 1 | 0), P(read 0 | 1)); None = perfect
    coupling: list = None           # allowed 2-qubit pairs; None = all-to-all
    description: str = ""

    @property
    def has_decoherence(self):
        return self.T1 is not None or self.T_phi is not None

    def allows(self, a, b):
        return self.coupling is None or (a, b) in self.coupling or (b, a) in self.coupling

    def describe(self):
        return {"name": self.name, "n_qubits": self.n_qubits, "T1": self.T1, "T_phi": self.T_phi,
                "gate_time_1q": self.gate_time_1q, "gate_time_2q": self.gate_time_2q,
                "readout_error": self.readout_error, "coupling": self.coupling,
                "description": self.description}


DEVICES = {
    "ideal": Device("ideal", 20, description="perfect 20-qubit machine: no noise, all-to-all"),
    "dq-5": Device("dq-5", 5,
                   T1=[50.0, 45.0, 55.0, 48.0, 52.0],
                   T_phi=[40.0, 35.0, 45.0, 38.0, 42.0],
                   gate_time_1q=0.1, gate_time_2q=0.4,
                   readout_error=[(0.010, 0.030), (0.015, 0.035), (0.010, 0.025), (0.020, 0.040), (0.012, 0.030)],
                   coupling=[(0, 1), (1, 2), (2, 3), (3, 4)],
                   description="noisy 5-qubit line: q0-q1-q2-q3-q4"),
}


def get_device(name):
    if isinstance(name, Device):
        return name
    if name not in DEVICES:
        raise ValueError(f"unknown device '{name}'; available: {', '.join(DEVICES)}")
    return DEVICES[name]
