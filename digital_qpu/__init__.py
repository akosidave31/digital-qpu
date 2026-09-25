"""digital_qpu: a virtual quantum computer built on the digital_qubit library.
OpenQASM 2.0 programs -> scheduled on a device with realistic noise -> measurement counts."""
__version__ = "0.1.0"

from .qasm import parse, Program, Op, QasmError
from .device import Device, DEVICES, get_device
from .executor import schedule, final_state, probabilities, sample_counts, MAX_NOISY_QUBITS
from .qpu import QPU, Job
