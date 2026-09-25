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
