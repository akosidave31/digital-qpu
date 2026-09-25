"""Compiler (transpiler): turn any program into what the device can physically do.
1. Route: if a 2-qubit gate acts on unconnected qubits, insert SWAPs along the shortest path
   (qubits move; a logical -> physical layout is tracked and measurements follow it).
2. Translate to native gates: cx/swap -> cz + single-qubit gates.
3. Optimize: merge every run of single-qubit gates on a qubit into one 2x2 unitary, then re-express
   it with as few physical pulses as possible: rz (virtual) and at most two sx, or a single x."""
from collections import deque
import numpy as np
from digital_qubit import gate_matrix, H, S, T, I2
from .qasm import Program, Op

SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
ATOL = 1e-9


def unitary_1q(op):
    """2x2 matrix of a single-qubit program op."""
    name = op.name
    if name == "id":
        return I2
    if name == "sx":
        return SX
    if name in ("sdg", "tdg"):
        return gate_matrix(name[0]).conj().T
    if name == "u1":
        name = "p"
    return gate_matrix(name, op.params[0] if op.params else None)


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def zyz(U):
    """U = e^{i g} Rz(alpha) Ry(beta) Rz(gamma). Returns (alpha, beta, gamma)."""
    V = U / np.sqrt(np.linalg.det(U))
    c, s = abs(V[0, 0]), abs(V[1, 0])
    beta = 2 * np.arctan2(s, c)
    if s < ATOL:
        tot, dif = 2 * np.angle(V[1, 1]), 0.0
    elif c < ATOL:
        tot, dif = 0.0, 2 * np.angle(V[1, 0])
    else:
        tot, dif = 2 * np.angle(V[1, 1]), 2 * np.angle(V[1, 0])
    return (tot + dif) / 2, beta, (tot - dif) / 2


def _equal_up_to_phase(A, B):
    return abs(abs(np.trace(A.conj().T @ B)) - 2) < 1e-7


def native_1q(U, q):
    """Fewest native ops (circuit order) implementing U up to global phase."""
    if _equal_up_to_phase(U, I2):
        return []
    if abs(U[0, 1]) < ATOL and abs(U[1, 0]) < ATOL:                      # diagonal: one virtual rz
        return [Op("rz", (q,), (_wrap(np.angle(U[1, 1]) - np.angle(U[0, 0])),))]
    if _equal_up_to_phase(U, np.array([[0, 1], [1, 0]])):
        return [Op("x", (q,))]
    a, b, g = zyz(U)
    if abs(b - np.pi / 2) < 1e-9:                                          # one sx pulse
        seq = [("rz", g - np.pi / 2), ("sx", None), ("rz", a + np.pi / 2)]
    else:                                                                  # two sx pulses
        seq = [("rz", g), ("sx", None), ("rz", b - np.pi), ("sx", None), ("rz", a + np.pi)]
    out = []
    for name, ang in seq:
        if name == "sx":
            out.append(Op("sx", (q,)))
        elif abs(_wrap(ang)) > 1e-12:
            out.append(Op("rz", (q,), (_wrap(ang),)))
    return out


def _shortest_path(coupling, n, a, b):
    adj = {i: set() for i in range(n)}
    for x, y in coupling:
        adj[x].add(y)
        adj[y].add(x)
    prev, seen, dq = {a: None}, {a}, deque([a])
    while dq:
        u = dq.popleft()
        if u == b:
            break
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                prev[v] = u
                dq.append(v)
    if b not in prev:
        raise ValueError(f"qubits {a} and {b} are not connected on this device")
    path, u = [], b
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


def route(program, device):
    """Map logical qubits to physical ones, inserting SWAPs where needed."""
    layout = list(range(program.n_qubits))                  # logical -> physical
    ops, swaps = [], 0
    for op in program.ops:
        phys = tuple(layout[q] for q in op.qubits)
        if len(phys) == 2 and not device.allows(*phys):
            path = _shortest_path(device.coupling, device.n_qubits, phys[0], phys[1])
            for u, v in zip(path[:-2], path[1:-1]):          # move the first qubit next to the second
                ops.append(Op("swap", (u, v)))
                swaps += 1
                lu, lv = (layout.index(u) if u in layout else None), (layout.index(v) if v in layout else None)
                if lu is not None:
                    layout[lu] = v
                if lv is not None:
                    layout[lv] = u
            phys = tuple(layout[q] for q in op.qubits)
        ops.append(Op(op.name, phys, op.params))
    measures = {layout[q]: c for q, c in program.measures.items()}
    used = [q for o in ops for q in o.qubits] + list(measures) + [program.n_qubits - 1]
    return Program(max(used) + 1, program.n_clbits, ops, measures), layout, swaps


def _to_cz(op):
    """cx / swap / cz as cz + single-qubit gates (still unmerged)."""
    a, b = op.qubits
    if op.name == "cz":
        return [op]
    if op.name == "cx":
        return [Op("h", (b,)), Op("cz", (a, b)), Op("h", (b,))]
    if op.name == "swap":
        return _to_cz(Op("cx", (a, b))) + _to_cz(Op("cx", (b, a))) + _to_cz(Op("cx", (a, b)))
    raise ValueError(f"cannot translate {op.name}")


def transpile(program, device):
    """Full compile. Returns (native program, info dict)."""
    routed, layout, swaps = route(program, device)
    stream = []
    for op in routed.ops:
        stream += _to_cz(op) if len(op.qubits) == 2 else [op]
    pending = {}
    out = []

    def flush(q):
        if q in pending:
            out.extend(native_1q(pending.pop(q), q))

    for op in stream:
        if len(op.qubits) == 1:
            q = op.qubits[0]
            pending[q] = unitary_1q(op) @ pending.get(q, I2)
        else:
            for q in op.qubits:
                flush(q)
            out.append(op)
    for q in sorted(pending):
        flush(q)
    native = Program(routed.n_qubits, routed.n_clbits, out, routed.measures)
    info = {"swaps": swaps, "layout": layout, "n_ops": len(out),
            "n_2q": sum(1 for o in out if len(o.qubits) == 2),
            "n_sx": sum(1 for o in out if o.name in ("sx", "x")),
            "n_rz": sum(1 for o in out if o.name == "rz")}
    return native, info


def to_qasm(program):
    """OpenQASM 2.0 text of a program (e.g. to inspect compiled output)."""
    L = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{program.n_qubits}];']
    if program.measures:
        L.append(f'creg c[{max(program.n_clbits, max(program.measures.values()) + 1)}];')
    for op in program.ops:
        p = f"({', '.join(f'{float(x)!r}' for x in op.params)})" if op.params else ""
        L.append(f"{op.name}{p} " + ",".join(f"q[{q}]" for q in op.qubits) + ";")
    for q, c in sorted(program.measures.items()):
        L.append(f"measure q[{q}] -> c[{c}];")
    return "\n".join(L)
