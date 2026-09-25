# digital_qpu

[![tests](https://github.com/akosidave31/digital-qpu/actions/workflows/tests.yml/badge.svg)](https://github.com/akosidave31/digital-qpu/actions/workflows/tests.yml)

A **virtual quantum computer**. Write a program in OpenQASM 2.0 (the language Qiskit and IBM
hardware use), run it on a simulated device with realistic noise, and get measurement counts back -
the same workflow as a real quantum cloud service. The qubits are provided by
[digital_qubit](https://github.com/akosidave31/digital-qubit).

It is a classical simulation: no quantum speed-up, and memory limits it to ~10 qubits with full
noise (~20 without). It is for learning, teaching, and testing quantum programs.

## Install

    pip install git+https://github.com/akosidave31/digital-qpu

## Use it

Command line:

    digital-qpu devices
    digital-qpu run examples/bell.qasm --device dq-5 --shots 1000

Python:

    from digital_qpu import QPU
    job = QPU("dq-5").run(open("examples/grover2.qasm").read(), shots=1000)
    print(job.result()["counts"])      # mostly '11' - Grover found the marked item

Results use Qiskit's bit order: classical bit 0 is the rightmost character.

## How it works

| Layer | What it does |
|---|---|
| `qasm.py` | parses OpenQASM 2.0 (one qreg/creg; gates id x y z h s sdg t tdg rx ry rz p u1 cx cz swap) |
| `device.py` | a chip's calibration sheet: per-qubit T1/T_phi, gate times, readout error, wiring |
| `executor.py` | schedules gates into time layers; after each layer every qubit (busy or idle) feels noise for that time; readout error at measurement |
| `qpu.py` | jobs: submit a program, get a job id, status and result |

Devices: `ideal` (20 qubits, no noise) and `dq-5` (5 noisy qubits in a line q0-q1-q2-q3-q4).

## Verified

- Noise formulas: dephasing and energy loss during gates match the exact expressions.
- Against Qiskit: the same OpenQASM text through Qiskit's own parser and simulator gives the same
  probabilities (1e-9), and the noisy execution matches a layer-by-layer Qiskit Aer reference (1e-10).

## Not supported yet

Mid-circuit measurement, `if`, `reset`, custom `gate` definitions, automatic qubit routing
(2-qubit gates must follow the device wiring).

## Roadmap

1. Engine + command line (this release)
2. Web API: submit jobs over HTTP, like a quantum cloud service
3. App on top of the API

MIT license.
