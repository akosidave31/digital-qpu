# Experiments and investigations

## v0.5.1 - two open questions from v0.5.0

Script: `tools/investigate.py` (read-only; run from the repo root). Measure first, then fix.

### A. Are bad days (TLS) independent across qubits?
Day 1 of dq-5 had 3 of 5 qubits bad at once (~1 in 1000 by chance).
20000 simulated days: per-qubit rate 4.67-5.06% (built in 5%); pair coincidences within 1.1 sd of
chance; 3+ bad at once 15 times vs 23 expected. **Independent: day 1 was a coincidence.**
Now a permanent test.

### B. Why did RB report +10-20% more error than expected?
| check | measured / exact (free fit) | measured / exact (B fixed) |
|---|---|---|
| standard (5 lengths up to 200, 10 sequences) | 1.192 | 0.998 |
| 3x more sequences | 1.002 | 0.986 |
| 2x longer sequences | 1.025 | 0.989 |
| gate error only | 1.307 | 0.991 |
| T1 only | 1.634 | 1.001 |
| T_phi only | 2.295 | 0.981 |

- The built-in expectation formula is exact: 6.000e-4 vs 5.997e-4 from the simulator (fidelity of
  each noisy gate averaged over the 6 Pauli eigenstates).
- The offset came from the free 3-parameter fit: with short sequences the long-sequence level B is
  poorly determined and a wrong B biases the decay rate.
- **Fix:** B is fixed at its known value (1/2 corrected for readout error); the free fit is still
  reported. Test tolerance tightened from 20% (which had hidden the problem) to 5%.

## v0.7.1 - is the MLP really better than the linear model?

v0.7.0 (one run): linear 0.035, MLP 0.030 average distance to truth; readout baseline 0.046.
One run cannot separate a real difference from run-to-run variation.
- Pre-registered rule: MLP becomes the default only if it beats linear by more than 2 standard
  errors over >= 5 runs (different calibration days, models retrained each day).
- Also measured: harm rate (how often a learned model is worse than readout mitigation alone).
  A strict "never worse" guarantee is impossible without knowing the true answer, so harm is
  measured instead of promised.
Result (5 calibration days, models retrained each day, 10 circuits each):

| | readout | linear | MLP | floor |
|---|---|---|---|---|
| mean distance to truth | 0.050 | 0.035 | 0.032 | 0.009 |
| +/- (standard error) | 0.003 | 0.002 | 0.002 | 0.000 |

- Paired linear - MLP = +0.0035 +/- 0.0003: the MLP is better on every day (> 10 standard errors).
- Harm (worse than readout alone by > 0.002): linear 6% of circuit-runs (mean +0.008, worst
  +0.009); MLP 12% (mean +0.021, worst +0.032). The MLP is better on average but riskier.
- Decision (pre-registered rule): MLP becomes the default. The rule did not include harm; it is
  not changed after seeing the data. Harm is documented, and the linear model stays available
  (`learned-linear`).
- Next: reduce the MLP's harm (v0.8.0). Suspects: the "blur toward uniform" assumption (T1 pulls
  toward 0, not uniform) and over-sharpening results whose true answer is spread out.

## v0.8.1 - performance baseline (measure before optimizing)

Goal: faster runs (raw speed) and more noisy qubits (capacity), in that order, each measured
against this baseline. Suspects before measuring: every gate copies the whole state; the 2-qubit
gate error loops over 15 Pauli terms in Python. To be confirmed or ruled out by the profile.
Result (phone, perf_baseline_phone.json):
- ideal engine healthy: 20 qubits in 1.1 s, ~4x per 2 qubits.
- noisy engine wall: 6 qubits 0.22 s, 8 qubits 3.9 s (~18x per 2 qubits) -> capacity needs a
  different method (trajectories), not just speed.
- feature cost (Grover, 5-qubit line): gate errors 9.3x, decoherence 7.8x, crosstalk 2.2x.
- randomized benchmarking 12.8 s for a 1-qubit experiment.
- profile: >60% of time in numpy bookkeeping (tensordot, reshape, moveaxis, axis checks) over
  8,610 calls for one Grover run: call overhead, not arithmetic.
- suspects: "copying the whole state" mostly WRONG at these sizes; "15-term 2-qubit error loop"
  CONFIRMED (gate errors most expensive; 1,035 kron calls).

## v0.9.0 - fewer numpy calls, same math

Change: single-qubit steps become 4x4 channel matrices, multiplied together and applied once per
qubit when needed; cz and ZZ become element-wise multiplications; direct 2-qubit depolarizing
formula; cached small matrices. The original engine is kept as final_state_reference.
Targets (set before measuring): gate-error cost 9.3x -> < 3x; noisy Grover 0.73 s -> < 0.3 s;
randomized benchmarking 12.8 s -> < 4 s; results identical to the reference (< 1e-12) and all
tests + Qiskit checks unchanged.
Result (phone): all targets met. Gate-error cost 0.4x no-noise; noisy Grover 0.037 s (19.8x);
randomized benchmarking 0.32 s (39.5x); training 9.8x; noisy engine 11-14x at 2-8 qubits;
test suite 3m23s -> 42s. 456 contractions per Grover run instead of 8,610. Identical results.
New finding: the pure-state engine (no noise / crosstalk only) is now the slower path.
