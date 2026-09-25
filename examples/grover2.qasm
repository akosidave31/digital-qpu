// Grover search over 4 items, marked item = 11. One iteration finds it (ideally with certainty).
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q;
cz q[0], q[1];      // oracle: flip the sign of |11>
h q;                // diffusion
x q;
cz q[0], q[1];
x q;
h q;
measure q -> c;
