"""OpenQASM 2.0 subset parser.
Supported: qreg/creg (one of each), include, barrier, measure (at the end of each qubit's use),
gates: id x y z h s sdg t tdg sx rx ry rz p u1 cx cz swap. Parameters may use pi and + - * / ( ).
Not supported yet (clear error): custom gate definitions, if, reset, mid-circuit measurement."""
import math
import re
from dataclasses import dataclass, field

ONE_Q = {"id": 0, "sx": 0, "x": 0, "y": 0, "z": 0, "h": 0, "s": 0, "sdg": 0, "t": 0, "tdg": 0,
         "rx": 1, "ry": 1, "rz": 1, "p": 1, "u1": 1}
TWO_Q = {"cx": 0, "cz": 0, "swap": 0}
UNSUPPORTED = ("gate", "opaque", "if", "reset")


class QasmError(ValueError):
    pass


@dataclass
class Op:
    name: str
    qubits: tuple
    params: tuple = ()


@dataclass
class Program:
    n_qubits: int
    n_clbits: int
    ops: list = field(default_factory=list)
    measures: dict = field(default_factory=dict)   # qubit -> clbit


def _param(expr, stmt):
    e = expr.strip()
    rest = e.replace("pi", "")
    if not e or any(ch not in "0123456789.eE+-*/() " for ch in rest):
        raise QasmError(f"bad parameter '{expr}' in: {stmt}")
    try:
        return float(eval(e, {"__builtins__": {}}, {"pi": math.pi}))
    except Exception:
        raise QasmError(f"bad parameter '{expr}' in: {stmt}")


def parse(text):
    """Parse OpenQASM 2.0 text into a Program."""
    lines = [ln.split("//", 1)[0] for ln in text.splitlines()]
    stmts = [s.strip() for s in " ".join(lines).split(";")]
    stmts = [s for s in stmts if s]
    qreg = creg = None
    ops, measures, measured = [], {}, set()

    def reg_arg(arg, stmt, kind):
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\[\s*(\d+)\s*\])?", arg.strip())
        if not m:
            raise QasmError(f"bad argument '{arg}' in: {stmt}")
        name, idx = m.group(1), m.group(2)
        reg = qreg if kind == "q" else creg
        if reg is None or name != reg[0]:
            raise QasmError(f"unknown {'quantum' if kind == 'q' else 'classical'} register '{name}' in: {stmt}")
        if idx is None:
            return list(range(reg[1]))
        i = int(idx)
        if i >= reg[1]:
            raise QasmError(f"index {i} out of range for {name}[{reg[1]}] in: {stmt}")
        return [i]

    for stmt in stmts:
        head = stmt.split()[0]
        if head == "OPENQASM":
            if not stmt.split()[1].startswith("2"):
                raise QasmError("only OpenQASM 2.x is supported")
            continue
        if head == "include" or head == "barrier":
            continue
        if head in UNSUPPORTED:
            raise QasmError(f"'{head}' is not supported yet: {stmt}")
        if head in ("qreg", "creg"):
            m = re.fullmatch(r"(qreg|creg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]", stmt)
            if not m:
                raise QasmError(f"bad register declaration: {stmt}")
            if head == "qreg":
                if qreg is not None:
                    raise QasmError("only one qreg is supported")
                qreg = (m.group(2), int(m.group(3)))
            else:
                if creg is not None:
                    raise QasmError("only one creg is supported")
                creg = (m.group(2), int(m.group(3)))
            continue
        if head == "measure":
            m = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", stmt)
            if not m:
                raise QasmError(f"bad measure: {stmt}")
            qs, cs = reg_arg(m.group(1), stmt, "q"), reg_arg(m.group(2), stmt, "c")
            if len(qs) != len(cs):
                raise QasmError(f"register sizes differ in: {stmt}")
            for q, c in zip(qs, cs):
                if c in measures.values() and measures.get(q) != c:
                    raise QasmError(f"classical bit {c} written twice: {stmt}")
                measures[q] = c
                measured.add(q)
            continue
        m = re.fullmatch(r"([a-z][a-z0-9_]*)\s*(?:\((.*)\))?\s+(.+)", stmt)
        if not m:
            raise QasmError(f"cannot parse: {stmt}")
        name, ptxt, atxt = m.group(1), m.group(2), m.group(3)
        if name not in ONE_Q and name not in TWO_Q:
            raise QasmError(f"unknown gate '{name}': {stmt}")
        params = tuple(_param(p, stmt) for p in ptxt.split(",")) if ptxt else ()
        need = ONE_Q.get(name, 0) if name in ONE_Q else 0
        if len(params) != need:
            raise QasmError(f"gate '{name}' needs {need} parameter(s): {stmt}")
        args = [reg_arg(a, stmt, "q") for a in atxt.split(",")]
        if name in ONE_Q:
            if len(args) != 1:
                raise QasmError(f"gate '{name}' takes one qubit: {stmt}")
            targets = [(q,) for q in args[0]]
        else:
            if len(args) != 2 or len(args[0]) != 1 or len(args[1]) != 1:
                raise QasmError(f"gate '{name}' takes two single qubits: {stmt}")
            if args[0][0] == args[1][0]:
                raise QasmError(f"gate '{name}' needs two different qubits: {stmt}")
            targets = [(args[0][0], args[1][0])]
        for qs in targets:
            if any(q in measured for q in qs):
                raise QasmError(f"gate after measurement (mid-circuit measurement not supported yet): {stmt}")
            ops.append(Op(name, qs, params))
    if qreg is None:
        raise QasmError("no qreg declared")
    return Program(qreg[1], creg[1] if creg else 0, ops, measures)
