"""Famous quantum algorithms as OpenQASM 2.0 programs, with the classical interpretation of results.
They are the same computations run on real quantum hardware; here they run on the simulated chip.
3-qubit gates (Toffoli, controlled-swap, controlled-phase) are written out in their standard
textbook decompositions into 1- and 2-qubit gates, as compilers do for real devices."""
from fractions import Fraction
from math import gcd, pi, asin, sin, sqrt

HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";'


def _ccx(a, b, c):
    """Toffoli (controls a, b; target c): standard 6-CNOT decomposition."""
    return [f"h q[{c}];", f"cx q[{b}],q[{c}];", f"tdg q[{c}];", f"cx q[{a}],q[{c}];", f"t q[{c}];",
            f"cx q[{b}],q[{c}];", f"tdg q[{c}];", f"cx q[{a}],q[{c}];", f"t q[{b}];", f"t q[{c}];",
            f"h q[{c}];", f"cx q[{a}],q[{b}];", f"t q[{a}];", f"tdg q[{b}];", f"cx q[{a}],q[{b}];"]


def _ccz(a, b, c):
    return [f"h q[{c}];"] + _ccx(a, b, c) + [f"h q[{c}];"]


def _cswap(c, a, b):
    """Fredkin: swap a and b when c is 1."""
    return [f"cx q[{b}],q[{a}];"] + _ccx(c, a, b) + [f"cx q[{b}],q[{a}];"]


def _cp(theta, a, b):
    """Controlled phase: |11> gets e^{i theta}."""
    return [f"u1({theta / 2!r}) q[{a}];", f"cx q[{a}],q[{b}];", f"u1({-theta / 2!r}) q[{b}];",
            f"cx q[{a}],q[{b}];", f"u1({theta / 2!r}) q[{b}];"]


def _iqft(qs):
    """Inverse quantum Fourier transform on qubits qs (qs[0] = least significant)."""
    n, body = len(qs), []
    for i in range(n // 2):
        body.append(f"swap q[{qs[i]}],q[{qs[n - i - 1]}];")
    for j in range(n):
        for m in range(j):
            body += _cp(-pi / 2 ** (j - m), qs[m], qs[j])
        body.append(f"h q[{qs[j]}];")
    return body


def _program(n_qubits, n_clbits, body, measures):
    lines = [HEAD, f"qreg q[{n_qubits}];", f"creg c[{n_clbits}];"] + body
    lines += [f"measure q[{q}] -> c[{c}];" for q, c in measures]
    return "\n".join(lines) + "\n"


def _top(counts):
    return max(counts, key=counts.get)


def bernstein_vazirani(secret=0b101, n=3):
    body = [f"x q[{n}];"] + [f"h q[{i}];" for i in range(n + 1)]
    body += [f"cx q[{i}],q[{n}];" for i in range(n) if secret >> i & 1]
    body += [f"h q[{i}];" for i in range(n)]
    key = format(secret, f"0{n}b")
    return {"name": "Bernstein-Vazirani", "qubits": n + 1,
            "task": f"find a hidden {n}-bit string with ONE query (classical: {n})",
            "qasm": _program(n + 1, n, body, [(i, i) for i in range(n)]),
            "expected": key, "answer": _top,
            "success": lambda c: c.get(key, 0) / sum(c.values())}


def deutsch_jozsa(kind="balanced", n=3):
    body = [f"x q[{n}];"] + [f"h q[{i}];" for i in range(n + 1)]
    if kind == "balanced":
        body += [f"cx q[{i}],q[{n}];" for i in range(n)]
    elif kind == "constant":
        body += [f"x q[{n}];"]
    else:
        raise ValueError("kind must be 'balanced' or 'constant'")
    body += [f"h q[{i}];" for i in range(n)]
    zero = "0" * n
    verdict = lambda c: "constant" if _top(c) == zero else "balanced"
    ok = (lambda c: c.get(zero, 0) / sum(c.values())) if kind == "constant" else \
         (lambda c: 1 - c.get(zero, 0) / sum(c.values()))
    return {"name": f"Deutsch-Jozsa ({kind})", "qubits": n + 1,
            "task": "constant or balanced? ONE query (classical worst case: 5)",
            "qasm": _program(n + 1, n, body, [(i, i) for i in range(n)]),
            "expected": kind, "answer": verdict, "success": ok}


def exact_grover_phase(iterations=2, n_items=8):
    """Long (2001): oracle and diffusion phase that makes Grover find the item with certainty in
    `iterations` rounds; None if that many rounds cannot reach 100%. For 8 items and 2 rounds: about 2.13."""
    beta = asin(1 / sqrt(n_items))
    x = sin(pi / (4 * iterations + 2)) / sin(beta)
    return 2 * asin(x) if x <= 1 else None


def _ccp6(lam, a, b, c):
    """Doubly-controlled phase e^{i lam} on |111>: the 6-CNOT CCZ circuit with T -> u1(lam/4), Tdg -> u1(-lam/4).
    Exact for any lam (the phases add up to lam*(a+b+c-a^b-a^c-b^c+a^b^c)/4 = lam*abc); at lam = pi it is CCZ."""
    q = lam / 4
    return [f"cx q[{b}],q[{c}];", f"u1({-q!r}) q[{c}];", f"cx q[{a}],q[{c}];", f"u1({q!r}) q[{c}];",
            f"cx q[{b}],q[{c}];", f"u1({-q!r}) q[{c}];", f"cx q[{a}],q[{c}];", f"u1({q!r}) q[{b}];",
            f"u1({q!r}) q[{c}];", f"cx q[{a}],q[{b}];", f"u1({q!r}) q[{a}];", f"u1({-q!r}) q[{b}];",
            f"cx q[{a}],q[{b}];"]


def grover3(marked=0b101, iterations=2, phase=None):
    """3-qubit Grover. phase=None: the textbook algorithm (CCZ, phase pi). phase=lam: every oracle call and
    diffusion applies phase lam instead (same 6-CNOT cost); lam = exact_grover_phase(2) gives certainty."""
    ccz = _ccz(0, 1, 2) if phase is None else _ccp6(phase, 0, 1, 2)
    flips = [f"x q[{i}];" for i in range(3) if not marked >> i & 1]
    oracle = flips + ccz + flips
    diffusion = [f"h q[{i}];" for i in range(3)] + [f"x q[{i}];" for i in range(3)] + ccz + \
                [f"x q[{i}];" for i in range(3)] + [f"h q[{i}];" for i in range(3)]
    body = [f"h q[{i}];" for i in range(3)] + (oracle + diffusion) * iterations
    key = format(marked, "03b")
    steps = f"{iterations} step" + ("s" if iterations != 1 else "")
    task = (f"find 1 marked item among 8 in {steps} (classical: up to 8 checks)" if phase is None else
            f"find 1 marked item among 8 in {steps}, phase {phase:.3f} instead of pi (certainty on a perfect machine)")
    return {"name": "Grover search (3 qubits)", "qubits": 3,
            "task": task,
            "qasm": _program(3, 3, body, [(i, i) for i in range(3)]),
            "expected": key, "answer": _top,
            "success": lambda c: c.get(key, 0) / sum(c.values())}


def phase_estimation(k=5, n=3):
    t = n
    body = [f"x q[{t}];"] + [f"h q[{j}];" for j in range(n)]
    for j in range(n):
        body += _cp(2 * pi * (k / 2 ** n) * 2 ** j, j, t)
    body += _iqft(list(range(n)))
    key = format(k, f"0{n}b")
    return {"name": "Phase estimation", "qubits": n + 1,
            "task": f"read a hidden phase {k}/{2 ** n} of a turn as binary digits",
            "qasm": _program(n + 1, n, body, [(j, j) for j in range(n)]),
            "expected": key, "answer": _top,
            "success": lambda c: c.get(key, 0) / sum(c.values())}


def _c_mult7(c, w):
    """Controlled multiplication by 7 mod 15 on work qubits w (textbook construction)."""
    return _cswap(c, w[0], w[1]) + _cswap(c, w[1], w[2]) + _cswap(c, w[2], w[3]) + \
           [f"cx q[{c}],q[{q}];" for q in w]


def _c_mult4(c, w):
    """Controlled multiplication by 4 mod 15 (= 7^2 mod 15)."""
    return _cswap(c, w[1], w[3]) + _cswap(c, w[0], w[2])


def shor_factors(counts, N=15, a=7, n=4):
    """Classical post-processing: measured y -> y/2^n -> continued fraction -> period r -> factors."""
    for key in sorted(counts, key=counts.get, reverse=True):
        y = int(key, 2)
        if y == 0:
            continue
        r = Fraction(y, 2 ** n).limit_denominator(N).denominator
        if r % 2 == 0 and pow(a, r, N) == 1:
            f = gcd(pow(a, r // 2) - 1, N)
            if 1 < f < N:
                return r, sorted({f, N // f})
    return None, None


def shor15():
    """Shor's algorithm factoring N = 15 with a = 7: 4 counting qubits + 4 work qubits.
    a^4 = a^8 = 1 (mod 15), so those controlled multiplications are the identity and are omitted."""
    w = [4, 5, 6, 7]
    body = [f"h q[{j}];" for j in range(4)] + [f"x q[{w[0]}];"]
    body += _c_mult7(0, w) + _c_mult4(1, w)
    body += _iqft([0, 1, 2, 3])

    def answer(c):
        r, f = shor_factors(c)
        return f"{f[0]} x {f[1]}" if f else "no factors"

    return {"name": "Shor (factor 15)", "qubits": 8,
            "task": "factor 15 using period finding (a = 7)",
            "qasm": _program(8, 4, body, [(j, j) for j in range(4)]),
            "expected": "3 x 5", "answer": answer,
            "success": lambda c: (c.get("0100", 0) + c.get("1100", 0)) / sum(c.values())}


GROVER_VARIANTS = ("standard", "exact", "1round")


def grover_variant(variant="standard", marked=0b101):
    """The Grover versions compared in EXPERIMENTS.md (v0.16.0-v0.19.0):
    standard: textbook, 2 rounds (94.5% on a perfect machine); exact: Long's phase, 2 rounds, same gate
    count (100%); 1round: textbook, 1 round (78.1% on a perfect machine, but the best on the noisy dq-5)."""
    if variant == "standard":
        return grover3(marked, 2)
    if variant == "exact":
        alg = grover3(marked, 2, phase=exact_grover_phase(2))
        alg["name"] = "Grover search (3 qubits, exact)"
        return alg
    if variant == "1round":
        alg = grover3(marked, 1)
        alg["name"] = "Grover search (3 qubits, 1 round)"
        return alg
    raise ValueError(f"Grover variant must be one of {GROVER_VARIANTS}")


def all_algorithms(grover="standard"):
    """grover: 'standard' (default), 'exact', '1round' or 'all' (all three Grover versions)."""
    variants = GROVER_VARIANTS if grover == "all" else (grover,)
    return [bernstein_vazirani(), deutsch_jozsa("balanced"), deutsch_jozsa("constant")] + \
           [grover_variant(v) for v in variants] + [phase_estimation(), shor15()]


FILES = {"bernstein_vazirani.qasm": bernstein_vazirani, "deutsch_jozsa_balanced.qasm": lambda: deutsch_jozsa("balanced"),
         "deutsch_jozsa_constant.qasm": lambda: deutsch_jozsa("constant"), "grover3.qasm": grover3,
         "phase_estimation.qasm": phase_estimation, "shor15.qasm": shor15}
