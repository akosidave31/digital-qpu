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

## v0.9.1 - fast pure-state engine

Finding from v0.9.0: noise-free runs (pure-state engine) became the slower path.
Change: pending 2x2 unitaries per qubit applied once; cz and ZZ as element-wise multiplications;
swap as a free relabelling of axes; cx as a flip of the half of the state where the control is 1.
Targets (set before measuring): ideal 20-qubit GHZ 1.05 s -> < 0.5 s; noise-free Grover
0.039 s -> < 0.01 s; crosstalk-only Grover 0.086 s -> < 0.02 s; results identical to the
reference (< 1e-12).
Result (phone): ideal 20-qubit GHZ 0.351 s (3.3x) - met; noise-free Grover 0.009 s (4.6x) - met;
crosstalk-only Grover 0.020 s (4.6x) - at the target (printed value rounded; within measurement noise);
identical to the reference. Test suite 42 s -> 68 s from the new reference-engine equivalence tests.

## v0.10.0 - capacity: trajectories, a 12-qubit chip, noisy Shor

Change: trajectory (Monte-Carlo wavefunction) engine for noisy circuits above 8 qubits (up to 16),
batched so numpy calls do not grow with the number of trajectories; new 12-qubit ladder chip dq-12.
Targets (set before measuring): trajectories agree with the exact engine within statistical error
and get closer with more trajectories (tests); a 12-qubit noisy GHZ with 300 trajectories runs in
< 30 s on the phone; noisy Shor on dq-12 completes. Shor's noisy success rate is an experiment,
not a target.
Result (phone): statistical agreement tests pass. 12-qubit GHZ with 300 trajectories: 30.06 s -
target MISSED by 0.2% (10 qubits: 5.2 s). Noisy Shor on dq-12 completes: routed onto 8 physical qubits, so the
exact engine ran (7 s); 1232 native ops, 194 cz of which 132 come from 44 SWAPs; useful outcomes 31% (ideal
50%); still factors 15 = 3 x 5. Routing overhead dominates -> smarter routing is the next priority.
Fixes found during the run: slow-test skipping was missing in this repo; the Shor test wrongly assumed the
trajectory engine; the 'period found' label also accepts multiples of the period (display only).

Process fix (found while preparing v0.11.0): release packages shipped the maintainer's copy of this file,
which overwrote results recorded after the previous release (v0.9.0 and v0.9.1 were lost from the current
file; they remain in git history). Restored above; the maintainer's copy is now updated at every recording.

## v0.11.0 - smarter routing

Change: look-ahead router. Starting placement chosen from several candidates (identity, a greedy
placement of qubits that interact often, and a "reverse pass" refinement of each); each SWAP must
bring the gate's qubits one step closer, and among those the one that helps the next 20 two-qubit
gates most is chosen. "auto" keeps the basic router whenever that needs fewer SWAPs. Also fixes the
Shor 'period found' label (only an r that gives non-trivial factors is called the period).
Targets (set before measuring): never more SWAPs than the basic router (test, random circuits on the
line and ladder); compiled results identical to the original program on a noise-free chip (test);
fewer SWAPs than v0.10.0 on Grover dq-5 (15) and Shor dq-12 (44); higher noisy success than v0.10.0
for Grover on dq-5 (32%) and Shor useful outcomes on dq-12 (31%).
Result (phone): tests pass (147 passed, 2 slow skipped); never more SWAPs than basic - met; identical
results on a noise-free chip - met. SWAPs: Grover dq-5 15 -> 8 - met; Shor dq-12 44 -> 20 - met; all
algorithms 108 -> 44 SWAPs, 474 -> 282 cz. Grover noisy success on dq-5 32% -> 37% - met. Shor useful
outcomes on dq-12 31% -> 28% - target MISSED, although the circuit is shorter (time 27.55 -> 19.62,
1232 -> 695 native ops; same seed, basic router re-run gives 31%).
Diagnosis: the loss is almost all one output bit. Classical bit 1 comes out flipped ~30% of the time
with the new router vs ~15% with the basic one (e.g. 0010 vs 0000: 137/340 vs 64/396); the other bits
look alike. That bit is read from physical q3, whose calibration is only slightly worse (T1 48, T_phi 38,
readout 2%/4%), which cannot explain a 30% flip rate - so "a weak qubit" is not the cause. Leading
hypothesis (not yet tested): coherent errors that depend on placement - always-on ZZ with the
neighbours of q3 (~0.1 rad per time unit over a ~20-unit circuit) turn into phase errors on a counting
qubit, which the inverse QFT turns into a flipped output bit. Counting SWAPs alone does not capture
this. Next (v0.12.0): test the ZZ hypothesis (same run with ZZ off), then a noise-aware router that
scores layouts by expected error (ZZ exposure, idle time, calibration), not only SWAP count.

## v0.12.0 - late start scheduling

Investigation (after the v0.11.0 Shor miss; exact probabilities, no shot noise, Shor on dq-12):
- ZZ crosstalk off: the gap stays (bit 1 wrong 32.7% -> 30.0% for look-ahead) - ZZ is NOT the cause.
- One noise source at a time: look-ahead is better with gate errors only (+4.6 points useful) and drive
  crosstalk only (+2.7), equal with readout only, worse with T1/T2 only (-4.4; bit 1 wrong 5.3% -> 22.0%).
- "Counting qubits start their superposition too early because of SWAPs" - rejected: the basic router
  touches them earlier (op 2-4) than look-ahead does.
- T1/T2 on one physical qubit at a time: physical q5 alone gives 21.4% bit-1 error with look-ahead
  (all other qubits, both routers: 0-3%). q5's calibration is normal; it holds counting qubit 2, which
  only has H at the start and its inverse-QFT gates at the end.
- Schedule check: q5 gets its first pulse at time 0 but its first cz at 14.67 (q7 similar: 10.17).
  Cause: the compiler delays that H until the qubit's first 2-qubit gate, but the as-soon-as-possible
  scheduler slides it back to time 0, so the qubit waits ~15 time units in a superposition that T2
  dephasing destroys. With the basic router SWAPs happened to use those qubits early, hiding this.
Change: "late start" - a qubit's single-qubit gates before its first 2-qubit gate are scheduled right
before that gate (the qubit waits in |0>, which dephasing cannot harm); like ALAP scheduling on real
devices. Qubits without 2-qubit gates (Ramsey, benchmarking) are scheduled exactly as before.
Targets (set before measuring): all tests pass, including noise-free equivalence and the Qiskit
cross-checks; Shor on dq-12 (seed 1, 2000 shots) useful outcomes > 31% (the missed v0.11.0 target);
Grover on dq-5 not below 35% (v0.11.0: 37%, minus 2 points for shot noise); Shor dq-12 circuit time
not more than 5% above v0.11.0 (19.62).
Result (phone): tests pass (148 passed, 2 slow skipped) - met. Shor on dq-12: useful outcomes 28% -> 32%
(seed 1, 2000 shots) - met; exact (no shot noise) 27.3% -> 32.0%, now above the basic router's 29.8%.
Bit-1 errors (y = 2, 6, 10, 14) 536 -> 254 shots. Circuit time unchanged (19.62) - met. Grover on
dq-5 37% - met (unchanged). Next lead: y = 0 and 8 now exceed y = 4 and 12 (456/375 vs 325/318 shots),
i.e. errors on output bit 2 - a different counting qubit; to investigate the same way.

Follow-up (after v0.12.0, exact probabilities, Shor on dq-12): the remaining excess of y = 0 and 8 over
y = 4 and 12 (imbalance +9.7 points) comes mostly from T1/T2 (+16.0 alone; gates +6.5, drive +7.3,
readout +1.9, ZZ -0.6). Per physical qubit it is T1-dominated and spread over q0-q4 (+2 to +9 each,
q5-q7 zero), where the work register sits. Shor's work register must hold |1>s (1, 7, 4, 13), which
decay with T1 whether busy or not (~e^(-20/50) = 67% kept over the circuit). Conclusion: mostly
physics, not a software flaw; calibration-aware placement could gain perhaps 1-2 points (T1 only
varies 45-55). Not pursued now.

## v0.13.0 - conservative learned mitigation

Baseline (v0.12.0, `benchmark --runs 10`, distance to the exact answer): readout 0.037, linear 0.025,
MLP 0.023 +/- 0.001, floor 0.009. MLP worse than readout alone in 11% of circuit-runs (by 0.009 on
average when worse; worst +0.020).
Investigation:
- Harm per circuit: almost all of it is random2 (harmed on 10 of 10 days, 0.028 vs 0.018), the circuit
  with the least room above the floor (0.005); random4 on 2 of 10 days (+0.003); all others never.
- "Harm happens when the ideal answer is close to uniform" - rejected (random4 is the most uniform and
  barely harmed; bell is similar to random2 and never harmed).
- Best-possible f (fitted with the true answer) vs predicted f, exact, nominal dq-5: for random2 the
  blur model fits (best f 0.985 improves 0.015 -> 0.011) but the MLP predicts 0.951 and over-corrects
  (0.023). Predictions are compressed (0.93-0.97 for all circuits; true 0.93-0.99), and over-correcting
  costs more than under-correcting (clipped negative probabilities).
- Correction fraction s (f_used = 1 - s (1 - f)) on 60 FRESH random circuits (seed 2000, not training,
  not benchmark), mean change vs readout / harmed / worst: s=1.0 -0.0358 / 7% / +0.017;
  0.8 -0.0324 / 2% / +0.012; 0.6 -0.0263 / 2% / +0.008; 0.4 -0.0186 / 2% / +0.003; 0.2 -0.0098 / 0% / +0.001.
Change: s = 0.8 (keeps ~90% of the gain, harm 7% -> 2% on fresh circuits); LearnedMitigator(kind,
shrink=...) still allows 1.0 (old behaviour).
Targets (set before measuring, `benchmark --runs 10`): all tests pass; MLP harm rate <= 5% (was 11%);
MLP mean <= 0.025 (was 0.023; allows ~10% of the gain given up plus one error bar); MLP worst harm
< +0.020.
Result (phone, `benchmark --runs 10`): tests pass (149 passed, 2 slow skipped) - met. MLP harm 11% -> 5%
of circuit-runs - met (exactly at the limit); MLP mean 0.023 -> 0.024 +/- 0.001 - met (<= 0.025); MLP
worst harm +0.020 -> +0.014 - met. Linear also safer: harm 9% -> 5%, worst +0.018 -> +0.012 (mean
0.025 -> 0.027). MLP still better than linear by more than 2 error bars (+0.0027 +/- 0.0005).
