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
