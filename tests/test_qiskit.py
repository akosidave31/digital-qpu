"""Independent cross-check: the same OpenQASM text through Qiskit (its own parser and simulator).
Skipped if qiskit / qiskit-aer are not installed."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer.noise import thermal_relaxation_error
from digital_qubit import Damping
from digital_qpu import parse, probabilities, final_state, schedule, Device, DEVICES
from digital_qpu.executor import layer_duration

G1 = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg"]
GP = ["rx", "ry", "rz", "u1"]


def rand_qasm(rng, n, depth, measure=True):
    L = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];', f'creg c[{n}];']
    for _ in range(depth):
        r = rng.random()
        if n > 1 and r < 0.25:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
        elif r < 0.6:
            L.append(f"{GP[int(rng.integers(4))]}({rng.uniform(-3, 3):.6f}) q[{int(rng.integers(n))}];")
        else:
            L.append(f"{G1[int(rng.integers(9))]} q[{int(rng.integers(n))}];")
    if measure:
        L.append("measure q -> c;")
    return "\n".join(L)


def test_ideal_programs_match_qiskit():
    rng = np.random.default_rng(1)
    for n in range(1, 6):
        for _ in range(20):
            text = rand_qasm(rng, n, 25)
            ours = probabilities(parse(text), DEVICES["ideal"])
            ref = Statevector(QuantumCircuit.from_qasm_str(text.replace("measure q -> c;", ""))).probabilities_dict()
            for k in set(ours) | set(ref):
                assert abs(ours.get(k, 0.0) - ref.get(k, 0.0)) < 1e-9, (text, k)


def _qiskit_gate(qc, op, n):
    q = [n - 1 - x for x in op.qubits]
    name = "p" if op.name == "u1" else op.name
    getattr(qc, name)(*op.params, *q)


def test_noisy_execution_matches_qiskit_layer_by_layer():
    rng = np.random.default_rng(2)
    n = 3
    dev = Device("t", n, T1=[50.0, 40.0, 60.0], T_phi=[40.0, 30.0, 45.0], gate_time_1q=0.5, gate_time_2q=2.0)
    for _ in range(10):
        program = parse(rand_qasm(rng, n, 20))
        ref = DensityMatrix.from_label("0" * n)
        for layer in schedule(program, dev):
            qc = QuantumCircuit(n)
            for op in layer:
                _qiskit_gate(qc, op, n)
            ref = ref.evolve(qc)
            d = layer_duration(layer, dev)
            for q in range(n):
                ch = thermal_relaxation_error(dev.T1[q], Damping.T2(dev.T1[q], dev.T_phi[q]), d).to_quantumchannel()
                ref = ref.evolve(ch, qargs=[n - 1 - q])
        assert np.max(np.abs(final_state(program, dev).rho - ref.data)) < 1e-10


def test_gate_errors_match_qiskit_layer_by_layer():
    from qiskit_aer.noise import depolarizing_error
    rng = np.random.default_rng(3)
    n = 3
    dev = Device("t", n, T1=[50.0, 40.0, 60.0], T_phi=[40.0, 30.0, 45.0], gate_time_1q=0.05, gate_time_2q=0.3,
                 gate_error_1q=[0.002, 0.003, 0.001], gate_error_2q=0.02)
    for _ in range(10):
        program = parse(rand_qasm(rng, n, 20))
        ref = DensityMatrix.from_label("0" * n)
        for layer in schedule(program, dev):
            for op in layer:
                qc = QuantumCircuit(n)
                _qiskit_gate(qc, op, n)
                ref = ref.evolve(qc)
                q = [n - 1 - x for x in op.qubits]
                if op.name != "id":
                    lam = dev.error_1q(op.qubits[0]) if len(q) == 1 else dev.error_2q(*op.qubits)
                    ref = ref.evolve(depolarizing_error(lam, len(q)).to_quantumchannel(), qargs=q)
            d = layer_duration(layer, dev)
            for x in range(n):
                ch = thermal_relaxation_error(dev.T1[x], Damping.T2(dev.T1[x], dev.T_phi[x]), d).to_quantumchannel()
                ref = ref.evolve(ch, qargs=[n - 1 - x])
        assert np.max(np.abs(final_state(program, dev).rho - ref.data)) < 1e-10


def test_compiled_programs_match_qiskit_on_original():
    from digital_qpu import transpile
    rng = np.random.default_rng(4)
    line = Device("line5", 5, coupling=[(0, 1), (1, 2), (2, 3), (3, 4)],
                  native_gates=("rz", "sx", "x", "cz"), virtual_rz=True)
    for _ in range(20):
        text = rand_qasm(rng, 5, 30)
        native, _ = transpile(parse(text), line)
        ours = probabilities(native, line)
        ref = Statevector(QuantumCircuit.from_qasm_str(text.replace("measure q -> c;", ""))).probabilities_dict()
        for k in set(ours) | set(ref):
            assert abs(ours.get(k, 0.0) - ref.get(k, 0.0)) < 1e-9


def test_crosstalk_matches_qiskit_layer_by_layer():
    from qiskit_aer.noise import depolarizing_error
    rng = np.random.default_rng(5)
    n = 3
    pairs = [(0, 1), (1, 2), (0, 2)]
    dev = Device("xt", n, T1=[50.0, 40.0, 60.0], T_phi=[40.0, 30.0, 45.0], gate_time_1q=0.05, gate_time_2q=0.3,
                 gate_error_1q=[0.002, 0.003, 0.001], gate_error_2q=0.02, coupling=pairs,
                 zz={(0, 1): 0.2, (1, 2): 0.3, (0, 2): 0.1}, drive_crosstalk=0.03)
    angle = {"sx": np.pi / 2, "x": np.pi}
    for _ in range(10):
        text = rand_qasm(rng, n, 20).replace("h q[", "sx q[")
        program = parse(text)
        ref = DensityMatrix.from_label("0" * n)
        for layer in schedule(program, dev):
            for op in layer:
                qc = QuantumCircuit(n)
                _qiskit_gate(qc, op, n)
                ref = ref.evolve(qc)
                q = [n - 1 - x for x in op.qubits]
                if op.name != "id":
                    lam = dev.error_1q(op.qubits[0]) if len(q) == 1 else dev.error_2q(*op.qubits)
                    ref = ref.evolve(depolarizing_error(lam, len(q)).to_quantumchannel(), qargs=q)
                if op.name in angle:
                    for m in dev.neighbours(op.qubits[0], n):
                        qc = QuantumCircuit(n)
                        qc.rx(dev.drive_crosstalk * angle[op.name], n - 1 - m)
                        ref = ref.evolve(qc)
            d = layer_duration(layer, dev)
            qc = QuantumCircuit(n)
            for (a, b), rate in dev.zz_pairs(n):
                qc.rzz(rate * d / 2, n - 1 - a, n - 1 - b)
            ref = ref.evolve(qc)
            for x in range(n):
                ch = thermal_relaxation_error(dev.T1[x], Damping.T2(dev.T1[x], dev.T_phi[x]), d).to_quantumchannel()
                ref = ref.evolve(ch, qargs=[n - 1 - x])
        assert np.max(np.abs(final_state(program, dev).rho - ref.data)) < 1e-10
