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
    gate_error_1q: list = None      # per qubit depolarizing parameter (Qiskit Aer convention)
    gate_error_2q: object = None    # dict {(a, b): p} per pair, or one float for all pairs
    native_gates: tuple = None      # e.g. ("rz", "sx", "x", "cz"); None = runs any gate directly
    virtual_rz: bool = False        # rz is a software frame change: zero time, zero error
    zz: object = None               # always-on ZZ rate (rad per time unit): {(a, b): rate} or one float for all wired pairs
    drive_crosstalk: float = 0.0    # fraction of each sx/x pulse that spills onto wired neighbours
    description: str = ""

    @property
    def has_decoherence(self):
        return self.T1 is not None or self.T_phi is not None

    @property
    def needs_density(self):
        """Mixed states are needed for decoherence or gate errors (readout error alone is not enough)."""
        return self.has_decoherence or self.gate_error_1q is not None or self.gate_error_2q is not None

    def error_1q(self, q):
        return 0.0 if self.gate_error_1q is None else self.gate_error_1q[q]

    def error_2q(self, a, b):
        e = self.gate_error_2q
        if e is None:
            return 0.0
        if isinstance(e, dict):
            return e.get((a, b), e.get((b, a), 0.0))
        return float(e)

    def allows(self, a, b):
        return self.coupling is None or (a, b) in self.coupling or (b, a) in self.coupling

    def zz_pairs(self, n=None):
        """[((a, b), rate)] for ZZ-coupled pairs (only pairs inside the first n qubits)."""
        n = self.n_qubits if n is None else n
        if self.zz is None:
            return []
        if isinstance(self.zz, dict):
            items = list(self.zz.items())
        else:
            items = [(pair, float(self.zz)) for pair in (self.coupling or [])]
        return [((a, b), r) for (a, b), r in items if a < n and b < n and r != 0]

    def neighbours(self, q, n=None):
        n = self.n_qubits if n is None else n
        return sorted(({b if a == q else a for a, b in (self.coupling or []) if q in (a, b)} - {q}) & set(range(n)))

    def describe(self):
        return {"name": self.name, "n_qubits": self.n_qubits, "T1": self.T1, "T_phi": self.T_phi,
                "gate_time_1q": self.gate_time_1q, "gate_time_2q": self.gate_time_2q,
                "readout_error": self.readout_error, "coupling": self.coupling,
                "native_gates": self.native_gates, "virtual_rz": self.virtual_rz,
                "zz": ({f"{a}-{b}": v for (a, b), v in self.zz.items()} if isinstance(self.zz, dict) else self.zz),
                "drive_crosstalk": self.drive_crosstalk,
                "gate_error_1q": self.gate_error_1q,
                "gate_error_2q": ({f"{a}-{b}": v for (a, b), v in self.gate_error_2q.items()}
                                  if isinstance(self.gate_error_2q, dict) else self.gate_error_2q),
                "description": self.description}


DEVICES = {
    "ideal": Device("ideal", 20, description="perfect 20-qubit machine: no noise, all-to-all"),
    "dq-5": Device("dq-5", 5,
                   T1=[50.0, 45.0, 55.0, 48.0, 52.0],
                   T_phi=[40.0, 35.0, 45.0, 38.0, 42.0],
                   gate_time_1q=0.02, gate_time_2q=0.15,
                   readout_error=[(0.010, 0.030), (0.015, 0.035), (0.010, 0.025), (0.020, 0.040), (0.012, 0.030)],
                   coupling=[(0, 1), (1, 2), (2, 3), (3, 4)],
                   gate_error_1q=[6e-4, 8e-4, 5e-4, 9e-4, 7e-4],
                   gate_error_2q={(0, 1): 0.010, (1, 2): 0.012, (2, 3): 0.009, (3, 4): 0.014},
                   native_gates=("rz", "sx", "x", "cz"), virtual_rz=True,
                   zz={(0, 1): 0.10, (1, 2): 0.15, (2, 3): 0.08, (3, 4): 0.12}, drive_crosstalk=0.01,
                   description="noisy 5-qubit line: q0-q1-q2-q3-q4 (native rz sx x cz; decoherence, gate, readout errors, crosstalk)"),
}

_LADDER = [(i, i + 1) for i in range(5)] + [(i, i + 1) for i in range(6, 11)] + [(i, i + 6) for i in range(6)]
DEVICES["dq-12"] = Device(
    "dq-12", 12,
    T1=[50.0, 45.0, 55.0, 48.0, 52.0, 47.0, 53.0, 49.0, 51.0, 46.0, 54.0, 50.0],
    T_phi=[40.0, 35.0, 45.0, 38.0, 42.0, 37.0, 43.0, 39.0, 41.0, 36.0, 44.0, 40.0],
    gate_time_1q=0.02, gate_time_2q=0.15,
    readout_error=[(0.010, 0.030), (0.015, 0.035), (0.010, 0.025), (0.020, 0.040), (0.012, 0.030), (0.014, 0.032),
                   (0.011, 0.028), (0.016, 0.036), (0.012, 0.027), (0.018, 0.038), (0.010, 0.029), (0.013, 0.033)],
    coupling=_LADDER,
    gate_error_1q=[6e-4, 8e-4, 5e-4, 9e-4, 7e-4, 6e-4, 7e-4, 8e-4, 5e-4, 9e-4, 6e-4, 7e-4],
    gate_error_2q={p: 0.009 + 0.001 * (i % 6) for i, p in enumerate(_LADDER)},
    native_gates=("rz", "sx", "x", "cz"), virtual_rz=True,
    zz={p: 0.08 + 0.014 * (i % 6) for i, p in enumerate(_LADDER)}, drive_crosstalk=0.01,
    description="noisy 12-qubit ladder: two rows of 6 (q0-q5, q6-q11) with rungs q_i-q_(i+6)")


def get_device(name):
    if isinstance(name, Device):
        return name
    if name not in DEVICES:
        raise ValueError(f"unknown device '{name}'; available: {', '.join(DEVICES)}")
    return DEVICES[name]
