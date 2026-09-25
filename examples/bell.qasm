// Bell pair: only 00 and 11 should appear (plus a little noise on dq-5)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
