import math
import pytest
from digital_qpu import parse, QasmError

HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n'


def test_bell_program():
    p = parse(HEAD + "h q[0];\ncx q[0], q[1];\nmeasure q -> c;")
    assert p.n_qubits == 3 and p.n_clbits == 3
    assert [(o.name, o.qubits) for o in p.ops] == [("h", (0,)), ("cx", (0, 1))]
    assert p.measures == {0: 0, 1: 1, 2: 2}


def test_parameters_with_pi():
    p = parse(HEAD + "rx(pi/2) q[0];\nu1(-pi/4) q[1];\nrz(2*pi - 0.5) q[2];")
    assert math.isclose(p.ops[0].params[0], math.pi / 2)
    assert math.isclose(p.ops[1].params[0], -math.pi / 4)
    assert math.isclose(p.ops[2].params[0], 2 * math.pi - 0.5)


def test_broadcast_comments_and_barrier():
    p = parse(HEAD + "// comment line\nh q; // trailing\nbarrier q;\nmeasure q[1] -> c[0];")
    assert [o.qubits for o in p.ops] == [(0,), (1,), (2,)]
    assert p.measures == {1: 0}


@pytest.mark.parametrize("body", [
    "foo q[0];",                          # unknown gate
    "h q[3];",                            # index out of range
    "qreg r[2];",                         # second qreg
    "gate my(a) x { rx(a) x; }",          # custom gate definition
    "measure q[0] -> c[0];\nx q[0];",     # gate after measurement
    "rx q[0];",                           # missing parameter
    "rx(__import__) q[0];",               # unsafe parameter
    "cx q[1], q[1];",                     # same qubit twice
    "h r[0];",                            # unknown register
    "reset q[0];",                        # unsupported statement
])
def test_errors_are_clear(body):
    with pytest.raises(QasmError):
        parse(HEAD + body)
