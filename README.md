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

## Web API

Submit jobs over HTTP, like a quantum cloud service (standard library only, runs fine in Termux):

    digital-qpu serve                       # http://127.0.0.1:8000, this device only
    digital-qpu serve --host 0.0.0.0        # reachable from your network (no authentication!)

    curl -s localhost:8000/devices
    curl -s -X POST localhost:8000/jobs -d '{"qasm": "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1]; measure q -> c;", "device": "dq-5", "shots": 1000}'
    curl -s localhost:8000/jobs/<job_id>    # status QUEUED -> RUNNING -> DONE, then the result

| Endpoint | What it does |
|---|---|
| `GET /` | service info and endpoints |
| `GET /devices` | available chips |
| `POST /jobs` | submit `{qasm, device, shots, seed, day, mitigate, router, trajectories}`; answers 202 with a job id |
| `GET /jobs` | recent jobs, newest first |
| `GET /jobs/<id>` | status, and the result when done |

Bad requests (invalid program, unknown device, too many shots, ...) are rejected immediately with
400 and the reason. Jobs run one at a time in the background, so the server keeps answering while a
job runs; at most 50 jobs wait in the queue (429 when full). No authentication yet: use it on this
device or a trusted network.

## Famous quantum algorithms

The same computations run on real quantum hardware, as OpenQASM programs in
[`examples/algorithms/`](examples/algorithms):

| algorithm | what it does |
|---|---|
| Bernstein-Vazirani | finds a hidden 3-bit string with ONE query (classical: 3) |
| Deutsch-Jozsa | constant or balanced function? ONE query |
| Grover (3 qubits) | finds 1 marked item among 8 in 2 steps (94.5% ideal success) |
| Phase estimation | reads a hidden phase out as binary digits (core of Shor) |
| Shor | factors 15 = 3 x 5 (8 qubits, ideal device) |

    digital-qpu algorithms          # every algorithm on the ideal machine and on dq-5
    digital-qpu shor                # factor 15 step by step: quantum part + classical post-processing
    digital-qpu shor --device dq-12 # the same on the noisy 12-qubit chip (trajectories)

Same computation, not the same speed: this is a classical simulation (see the top of this README).

## Error mitigation

    digital-qpu run examples/ghz5.qasm --mitigate readout    # counts + readout-mitigated result
    digital-qpu benchmark                                    # 10 circuits: raw vs mitigated error
    digital-qpu benchmark --runs 5                           # 5 calibration days, error bars, harm rate

Readout mitigation removes readout error only; gate errors, decoherence and crosstalk remain.
It is the baseline that any learned mitigation must beat.

Learned mitigation (`--mitigate learned`): after readout mitigation the remaining noise mostly blurs
results toward uniform. A model predicts how much signal survives from the circuit's error budget
(gate errors, decoherence and ZZ exposure from the day's calibration) and undoes the blur. Two
models: an MLP from digital_qubit (default, `--mitigate learned`) and a linear model
(`--mitigate learned-linear`). Trained on random circuits
from a different seed family than the benchmark; `digital-qpu benchmark` compares everything
against the shot-noise floor.

Measured over 5 calibration days (`digital-qpu benchmark --runs 5`), average distance to the exact
answer: readout 0.050, linear 0.035, MLP 0.032 (floor 0.009). The MLP is better on average, but
makes a result worse than readout alone more often (12% vs 6% of circuit-runs; worst +0.032 vs
+0.009). Use `learned-linear` when avoiding occasional bad corrections matters more.

Since v0.13.0 the correction is conservative: only 80% of the predicted correction is applied,
because the model's estimate is imprecise and over-correcting hurts more than under-correcting.
The 80% was chosen on fresh random circuits, not on the benchmark (see EXPERIMENTS.md).

## Speed

    digital-qpu speed --save baseline.json          # where does the time go? (measurement only)
    digital-qpu speed --compare baseline.json       # how many times faster than a saved baseline

Reports the ideal and noisy engines' time and memory as qubits grow, the cost of each noise
feature, real workloads, and a profile of the most expensive functions.

## How it works

| Layer | What it does |
|---|---|
| `qasm.py` | parses OpenQASM 2.0 (one qreg/creg; gates id x y z h s sdg t tdg rx ry rz p u1 cx cz swap) |
| `device.py` | a chip's calibration sheet: per-qubit T1/T_phi, gate times, readout error, wiring |
| `executor.py` | schedules gates into time layers (a qubit's opening gates wait until just before its first 2-qubit gate, so it idles in the safe |0> state); each gate has its own error (depolarizing, as in Qiskit Aer); after each layer every qubit (busy or idle) feels T1/T2 noise for that time; readout error at measurement |
| `compiler.py` | like a real transpiler: routes qubits with SWAPs when they aren't wired together, translates to the chip's native gates (rz, sx, x, cz), merges single-qubit gates and uses as few sx pulses as possible |
| `crosstalk.py` | Ramsey experiment that measures always-on ZZ crosstalk between two qubits, like a lab |
| `calibration.py` | day-to-day calibration drift: parameters wander (today resembles yesterday), occasional bad days when a defect drops a qubit's T1 |
| `learned.py` | learned error mitigation: predicts surviving signal from the circuit's error budget (linear model or digital_qubit's MLP) |
| `mitigation.py` | benchmark suite with exactly known answers; readout-error mitigation (undo the day's readout errors); distance-to-truth metrics |
| `rb.py` | randomized benchmarking: measures the device's error per gate, like a real lab |
| `qpu.py` | jobs: submit a program, get a job id, status and result |
| `variational.py` | trainable Grover circuit (parameter-shift gradients, Adam): noise-aware circuit optimisation |
| `server.py` | web API: the same jobs over HTTP, with a background worker and input checks |

Devices: `ideal` (20 qubits, no noise), `dq-5` (5 noisy qubits in a line q0-q1-q2-q3-q4) and
`dq-12` (12 noisy qubits in a ladder: two rows of 6 with rungs between them).

Engines (chosen automatically): pure state (no noise), exact noisy density matrix (up to 8
qubits automatically, 10 on request) and noisy trajectories (9-16 qubits; an average over many
random noise histories, 300 by default; `--trajectories N`).

`dq-5` calibration (abstract time units; think microseconds):

| qubit | T1 | T_phi | 1-qubit gate error | readout error (0->1, 1->0) |
|---|---|---|---|---|
| q0 | 50 | 40 | 6e-4 | 1.0%, 3.0% |
| q1 | 45 | 35 | 8e-4 | 1.5%, 3.5% |
| q2 | 55 | 45 | 5e-4 | 1.0%, 2.5% |
| q3 | 48 | 38 | 9e-4 | 2.0%, 4.0% |
| q4 | 52 | 42 | 7e-4 | 1.2%, 3.0% |

Two-qubit gate errors: q0-q1 1.0%, q1-q2 1.2%, q2-q3 0.9%, q3-q4 1.4%.
Native gates: rz (virtual: zero time, zero error), sx, x, cz. Programs are compiled automatically.

Crosstalk (coherent errors, like real superconducting chips):
- always-on ZZ between wired neighbours: q0-q1 0.10, q1-q2 0.15, q2-q3 0.08, q3-q4 0.12 rad per
  time unit - a qubit's phase drifts depending on whether its neighbour is 0 or 1
- drive crosstalk: 1% of every sx/x pulse spills onto wired neighbours

Measure the ZZ yourself:

    digital-qpu zz --pair 0 1

Calibration drift: like a real chip, `dq-5` is recalibrated every day. T1, T_phi, gate and readout
errors wander around the values above (~15-25%), and about 5% of qubit-days are "bad days" where a
material defect drops T1 to 20-50% of normal. ZZ comes from the chip design and barely moves.

    digital-qpu calibration --day 12          # that day's data sheet
    digital-qpu history --qubit 0 --days 30   # how q0 drifted over a month
    digital-qpu run examples/bell.qasm --day 12
    digital-qpu rb --qubit 0 --day 12         # benchmarking notices bad days

See what the chip will actually run:

    digital-qpu compile examples/ghz5.qasm

Routing (v0.11.0): when two qubits that must interact aren't wired together, the compiler inserts
SWAPs. It first picks a good starting placement (qubits that talk a lot go next to each other), then
chooses each SWAP by looking at the next 20 two-qubit gates, and keeps the older simple router if
that happens to need fewer SWAPs. Compare them:

    digital-qpu routing                              # SWAPs and cz gates: basic vs new router
    digital-qpu compile examples/algorithms/grover3.qasm --router basic
    digital-qpu run examples/algorithms/grover3.qasm --router lookahead

Programs are placed on the smallest connected block of qubits 0..m-1 that fits them, so the
simulated register stays small (Shor on dq-12 uses qubits 0-7).
Gate times: 0.02 (1-qubit), 0.15 (2-qubit). Error values use Qiskit Aer's depolarizing parameter.

Measure it yourself, like a lab would:

    digital-qpu rb --qubit 0          # randomized benchmarking -> error per gate

## Verified

Quantum-validation suite (`python validation/quantum_validation.py`, report in `validation/REPORT.txt`):
96 checks of the quantum-circuit MATHEMATICS - single-qubit gates and identities, interference, Bell
state (amplitudes, correlations, CHSH value 2*sqrt(2) of the simulated state), Bernstein-Vazirani,
Deutsch-Jozsa, Grover (2-4 qubits), QFT, Shor N=15 - each compared with a hand-derived result AND an
independent reference simulator written separately (own gate matrices, own QASM reader, dense Kronecker
products), plus seeded sampling checks. This shows the classical simulation computes what the
quantum-circuit model predicts; it does not show physical quantum behaviour or any speedup.

- Noise formulas: dephasing and energy loss during gates match the exact expressions.
- Compiled programs give identical results to the originals (random circuits, including routing),
  checked against both the uncompiled program and Qiskit.
- Crosstalk follows the exact formulas; coherent errors grow ~4x when a circuit doubles (random
  errors grow ~2x); the Ramsey experiment measures back the built-in ZZ rates (within 5%).
- Drift statistics are realistic (centred, ~15% spread, day-to-day correlation, ~5% bad days), and
  randomized benchmarking detects a bad day and measures that day's actual error rate.
- Readout mitigation is exact without shot noise (including routed circuits) and helps on real runs.
- Randomized benchmarking measures back the error per gate the device is built with (within 5%),
  and that built-in value agrees with an exact simulator calculation (see EXPERIMENTS.md).
- Against Qiskit: the same OpenQASM text through Qiskit's own parser and simulator gives the same
  probabilities (1e-9), and the noisy execution - decoherence, gate errors and crosstalk - matches a layer-by-layer Qiskit Aer
  reference (1e-10).

## Not supported yet

Mid-circuit measurement, `if`, `reset`, custom `gate` definitions. Routing searches only a few
starting layouts and one prefix block of qubits - not an optimal (exhaustive) search, and it
ignores which qubits have the lowest error today.

## Roadmap

1. Engine + command line (v0.1.0)
2. Gate errors, realistic timing, randomized benchmarking (v0.2.0)
3. Native gates, compilation and automatic qubit routing (v0.3.0)
4. Crosstalk: always-on ZZ and drive spill-over, Ramsey measurement (v0.4.0)
5. Calibration drift, daily data sheets, bad days (v0.5.0); investigation fixes (v0.5.1)
6. Error mitigation: benchmark suite + readout baseline (v0.6.0); learned mitigation (v0.7.0/0.7.1)
7. Famous algorithms: Bernstein-Vazirani, Deutsch-Jozsa, Grover, phase estimation, Shor-15 (v0.8.0)
8. Faster noisy engine: combined channels, diagonal gates as multiplications (v0.9.0);
   faster pure-state engine (v0.9.1)
9. Capacity: trajectory engine (up to 16 noisy qubits), 12-qubit ladder chip dq-12, noisy Shor (v0.10.0)
10. Smarter routing: better initial placement + look-ahead SWAP choice (v0.11.0)
11. Late start scheduling: qubits stay in |0> until needed, like ALAP on real devices (v0.12.0)
12. Reduce learned mitigation's harm: conservative correction (v0.13.0)
13. Web API: submit jobs over HTTP, like a quantum cloud service (v0.14.0)
14. App on top of the API
15. Trainable Grover: variational circuit trained on the noisy chip (v0.16.0: found to memorise the answer;
    v0.17.0: trained on all 8 marked items at once, with a permanent memorisation check)
16. Trainable oracle/diffusion phases: tests Long's exact (100%) Grover (v0.18.0)

See EXPERIMENTS.md for investigations and the decisions they led to.

MIT license.
