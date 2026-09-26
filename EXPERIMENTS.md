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

## v0.14.0 - web API

Change: `digital-qpu serve` - HTTP API (standard library only): POST /jobs, GET /jobs/<id>, GET /jobs,
GET /devices. Submissions are validated immediately (400 with the reason); jobs run one at a time in a
background worker; queue limit 50 (429); finished jobs kept up to 500. Built for Termux first; the
same code can later run on a cloud host.
Targets (set before measuring, on the phone): all tests pass; a noisy Bell job on dq-5 (1000 shots)
goes from submission to DONE in < 2 s; while noisy Shor runs on dq-12, GET /devices answers in < 0.5 s.
Result (phone): tests pass (171 passed, 2 slow skipped) - met. Noisy Bell on dq-5, 1000 shots,
submission to DONE: 0.121 s - met (< 2 s; counts 11: 471, 00: 470, 01: 36, 10: 23). GET /devices while
noisy Shor ran on dq-12 (status RUNNING): 0.009 s - met (< 0.5 s).

## v0.15.0 - quantum-validation suite

Question: does Digital-QPU reproduce the mathematics of the ideal quantum-circuit model? (Not: is it a
quantum computer - it is a classical simulation, and nothing here tests physical quantum behaviour or speed.)
Method: validation/quantum_validation.py - 96 checks in 8 sections, each against a hand-derived analytic
result (numpy.fft for the QFT) AND an independent reference simulator in the same file (own gate
matrices, own QASM reader, dense 2^n x 2^n Kronecker matrices, opposite qubit ordering; imports nothing
from digital_qpu or digital_qubit), plus seeded sampling checks (20000 shots, seed 20260926).
Tolerances fixed before the first run: exact 1e-10, sampling 5 sigma, zero-probability outcomes never sampled.
First run: 94/96. Both failures were wrong hand-written expectations in the suite (SX|+> is |+>, not
e^{i pi/4}|+>; X.Y = +iZ, not -iZ); Digital-QPU and the reference agreed exactly (difference 0.0) in
both. Expectations corrected, tolerances unchanged.
Result (sandbox run, Python 3, numpy 2.4): 96/96 pass; largest exact error 5.8e-15; largest sampling
deviation 1.08 sigma. Verified as mathematics: gate actions and phases, identities, normalization,
interference, Bell-state amplitudes and correlations (incl. X basis, CHSH 2*sqrt(2) of the simulated
state), BV, DJ, Grover 2-4 qubits, QFT, phase estimation, Shor N=15 incl. post-processing.
Not verified: physical quantum behaviour, speedup, physical Bell nonlocality, true randomness, noisy-chip
realism vs real hardware, Shor beyond N=15/a=7 (and that circuit uses knowledge of the period).
Phone confirmation (Termux): full test suite 270 passed, 2 slow skipped (includes all 96 validation cases).

## v0.16.0 - trainable Grover circuit (variational)

Question: can training the single-qubit angles of the 3-qubit Grover circuit beat fixed Grover on the
noisy dq-5 chip (37% with shots in v0.11.0), while keeping its two-qubit structure (oracle and CCZs fixed)?
Method: digital_qpu/variational.py. Each H/X layer becomes a trainable rz-ry-rz layer per qubit (45
parameters for 2 rounds, 27 for 1). Fixed Grover is one point of this space and is the starting point.
Reward = exact probability of the marked answer; gradient by parameter shift (exact on ideal, approximate
on noisy chips because the compiler's pulse count depends on the angle); Adam, lr 0.05, 40 epochs.
Trained on ideal and on dq-5 (nominal calibration); tested on dq-5 calibration days 1-10, never seen.
Run on Colab: notebooks/train_grover.ipynb.
Limits: the reward needs the marked item, so this is circuit optimisation for a known task, not a better
search; classical simulation, no speedup.
Targets (set before running): tests pass; the trainable circuit at its Grover point reproduces fixed Grover
exactly (94.53% ideal, 2 rounds; test); ideal best trained >= 0.99; dq-5 best trained >= 0.45;
unseen days: best dq-5-trained circuit minus fixed Grover >= +0.05 on average.
Result (phone, Termux, 40 epochs, ~7 min): tests pass (5/5). The script reported all three targets MET
(ideal 0.9453 -> 1.0000; dq-5 0.3722 -> 0.5753 with 1 round; unseen days +0.206 +/- 0.001, 10/10 days) -
but these results are INVALID as a better Grover, because of a design flaw found during the run:
- Warning sign: the 1-round circuit reached 98% on ideal in 5 epochs, above the 78.1% one-oracle-call
  Grover limit for 8 items.
- Check: trained angles evaluated with every marked item 000..111 (ideal chip). Normal Grover 0.95 for all.
  ideal/2 rounds 0.78-1.00 (mean 0.93, below Grover's 0.945); dq-5/2 rounds 0.98 on 101 but 0.03-0.32 on
  the others; 1-round circuits 0.99-1.00 on 101, 0.27-0.86 on the others.
- Cause: every single-qubit layer, including the one before the first oracle, was trainable and the reward
  contained the answer, so training learned to produce 101 while partly ignoring the oracle (memorisation).
  The tests and targets did not check that the circuit still uses the oracle - the design mistake.
Genuine finding (fixed circuits, which do use the oracle, marked 101 only): on the unseen dq-5 days fixed
Grover with 1 round averaged 0.423 vs 0.340 with 2 rounds - on this noisy chip the shorter circuit wins,
although it is worse on the ideal chip (78.1% vs 94.5%).
Next (v0.17.0): train one set of angles on all 8 marked items at once (average success), so memorising an
answer cannot pay; compare with normal Grover's average for every marked item.

## v0.17.0 - joint training on all marked items (fixing v0.16.0's memorisation)

Change: JointGrover - one shared set of angles scored on the AVERAGE success over all 8 possible marked
items (8 oracles), so memorising one answer cannot pay. Same circuit shape as v0.16.0 (oracle and CCZs fixed,
rz-ry-rz trainable layers), same start (fixed Grover), parameter shift + Adam, lr 0.05, 40 epochs, 1 and 2
rounds, trained on ideal and on dq-5 (nominal), tested on dq-5 calibration days 1-10 (unseen).
Permanent memorisation check: every trained circuit is evaluated on the ideal chip for each marked item;
if max - min > 0.10 it is flagged SPECIALISED and does not count toward any target. A test shows the
check catches v0.16.0-style single-item training.
Targets (set before running): tests pass; ideal: best valid trained mean >= 0.97 (fixed Grover 0.9453);
dq-5 (training calibration): best valid trained mean >= fixed Grover mean + 0.03; unseen days: best valid
dq-5-trained circuit minus the BETTER fixed Grover (1 or 2 rounds, whichever averages higher) >= +0.02.
Also reported, not a target: does fixed Grover with 1 round still beat 2 rounds on the noisy chip when
averaged over all 8 marked items (replication of the v0.16.0 finding)?
Result (phone, Termux, 40 epochs, ~45 min): tests pass (9/9). Memorisation check passed for all four
trained circuits (ideal spread 0.059, 0.074, 0.000, 0.029 - all <= 0.10), so every result below counts.
- ideal: fixed Grover mean 0.9453 -> 0.9585 (2 rounds; +1.3 points) - target 0.97 MISSED (still rising at
  epoch 40). 1 round: 0.7812 -> 0.7812, no gain (Grover's 1-round angles already optimal in this circuit).
- dq-5 (training calibration): the script reports MET (0.3854 -> 0.4684), but this target compared against
  2-round fixed Grover. Most of that gain comes from using 1 round, not from training: fixed 1-round Grover
  scores 0.4574 there. Fair comparison: 2 rounds 0.3854 -> 0.3978 (+1.2), 1 round 0.4574 -> 0.4684 (+1.1).
  Lesson: the target should have compared with the best fixed circuit, as the unseen-days target did.
- unseen days (10), mean over all 8 marked items: fixed 2 rounds 0.3555, fixed 1 round 0.4386;
  dq-5-trained 2 rounds 0.3667 (+1.1 vs its fixed), 1 round 0.4497; ideal-trained 2 rounds 0.3573 (+0.2).
  Best trained minus best fixed = +0.0111 +/- 0.0002, better on 10/10 days - target +0.02 MISSED, but the
  gain is small and very consistent.
Conclusions: (1) honest training (no memorisation possible) gives a real but modest gain, about +1 point,
on every unseen day; (2) it is noise-aware: angles trained on the ideal chip give almost nothing on dq-5
(+0.2), angles trained on dq-5 give +1.1; (3) confirmed with all 8 marked items: on the noisy chip fixed
Grover with 1 round beats 2 rounds by 8.3 points (0.4386 vs 0.3555) - circuit length matters far more
than tuning angles.

## v0.18.0 - trainable phases in the oracle and diffusion (Long's exact Grover)

Question: v0.17.0 missed its ideal target (0.9585 < 0.97) by training only single-qubit angles. Theory
(G.L. Long, 2001): if the oracle and diffusion apply a phase phi instead of pi, 2 rounds find the item in
8 with CERTAINTY when phi = 2 arcsin(sin(pi/10) / sin(beta)), beta = arcsin(1/sqrt 8), i.e. phi = about 2.13
(or its mirror 2 pi - phi, about 4.16). Can joint training (all 8 marked items, no memorisation possible)
find that?
Change: PhaseGrover - each oracle call and each diffusion gets one trainable phase (a doubly-controlled
phase gate; at pi it is the usual CCZ). The oracle is still one call per round marking the same item;
only its phase is adjustable. Option: also train the rz-ry-rz layers. Cost: the phase-form CCZ uses 8
CNOTs vs 6 in the Toffoli form, so on noisy chips its fixed version (phase pi) is the fair same-gates baseline.
Start: layers at Grover, phases at pi - 0.3 (at exactly pi the phase gradient is zero by symmetry).
Joint training, parameter shift with chain rule, Adam, lr 0.1, 60 epochs. Runs: ideal phases-only 2 rounds;
ideal phases+layers 2 rounds; dq-5 phases-only 2 rounds and 1 round. Unseen: dq-5 days 1-10.
Memorisation check as in v0.17.0 (spread <= 0.10 on the ideal chip, else it does not count).
Targets (set before running): tests pass (including: some common phase reaches > 99.9% with 2 rounds, and
Long's formula gives > 99.99%); T1 ideal best valid trained mean >= 0.99; T2 the 4 learned phases of the
ideal phases-only run each within 0.1 rad of Long's phi (or its mirror); T3 unseen days: best valid dq-5
trained circuit minus the best fixed circuit (Toffoli or phase form, 1 or 2 rounds) >= +0.02.
Result (phone, Termux, 60 epochs, ~54 min): memorisation check passed for all 4 runs (spread 0.000).
- T1 MET: ideal, phases only (4 parameters): mean over all 8 marked items 0.9547 (start) -> 1.0000 by
  epoch 30; every item 1.00. Phases + layers (49 parameters): 0.9998.
- T2 MET: learned phases (oracle 1, diffusion 1, oracle 2, diffusion 2) 2.103 2.204 2.204 2.103 vs Long's
  2.127; largest deviation 0.077 rad. Not all equal: the equal-phase Long solution is one point of a family
  of exact solutions, and training found a nearby symmetric one.
- T3 MISSED: unseen dq-5 days, best trained (1 round, phases) minus best fixed (Toffoli form, 1 round)
  = -0.0043 +/- 0.0002 (0/10 days). Unseen-day means: fixed Toffoli 2 rounds 0.3555, 1 round 0.4386;
  fixed phase form 2 rounds 0.3307, 1 round 0.4330; dq-5-trained phases 2 rounds 0.3503, 1 round 0.4344;
  ideal-trained (exact, 100% on the ideal chip) 2 rounds 0.3452.
Conclusions: (1) training rediscovered Long's exact Grover: 100% for every marked item with 4 phases,
confirming the theory; (2) on the noisy chip the exact algorithm LOSES: 34.5%, below standard Grover's 35.6%
and far below 1-round Grover's 43.9%, because its phase gates cost 8 CNOTs instead of 6 per CCZ; (3) against
the same-gates baseline, phase training helps a little (2 rounds +2.0 points, 1 round +0.1), not enough to
pay for the extra gates; (4) the best circuit found for dq-5 is still plain 1-round Grover. Across v0.16-v0.18:
on this noisy chip, fewer two-qubit gates matter more than any tuning of angles or phases.
