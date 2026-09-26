"""digital_qpu: a virtual quantum computer built on the digital_qubit library.
OpenQASM 2.0 programs -> scheduled on a device with realistic noise -> measurement counts."""
__version__ = "0.12.0"

from .qasm import parse, Program, Op, QasmError
from .device import Device, DEVICES, get_device
from .executor import schedule, final_state, probabilities, sample_counts, MAX_NOISY_QUBITS, engine_for
from .trajectories import final_trajectories, MAX_TRAJECTORY_QUBITS
from .qpu import QPU, Job
from .compiler import transpile, route, native_1q, to_qasm
from .mitigation import readout_mitigate, benchmark_suite, evaluate, evaluate_many, summarize, tvd, distribution
from .algorithms import all_algorithms, bernstein_vazirani, deutsch_jozsa, grover3, phase_estimation, shor15, shor_factors
from .learned import LearnedMitigator, circuit_features, training_data, get_mitigator
from .calibration import calibrate, bad_qubits, history, tls_days
from .crosstalk import zz_ramsey, pair_device
from .rb import randomized_benchmarking, predicted_epg, clifford_group, CLIFFORDS, fit_decay
