"""Compiler (transpiler): turn any program into what the device can physically do.
1. Route: choose where each logical qubit starts on the chip, and insert SWAPs when a 2-qubit gate
   acts on unconnected qubits, choosing SWAPs that also help the upcoming gates (v0.11.0; the
   original shortest-path router is kept as route_basic). Measurements follow the final layout.
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


def route_basic(program, device):
    """The original router (v0.3.0): logical qubit i starts on physical qubit i; when a 2-qubit gate
    needs it, the first qubit walks along a shortest path next to the second."""
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


# ---------------- look-ahead router (v0.11.0) ----------------
# 1. Region: the program is placed on the smallest prefix of physical qubits 0..m-1 (m >= number of
#    logical qubits) whose wiring is connected. This keeps the simulated register small (the engines
#    simulate physical qubits 0..max used) - a simulation choice, documented as a limitation.
# 2. Initial layout: several candidates are routed and the one needing the fewest SWAPs wins:
#    identity; a greedy placement that puts qubits which interact often next to each other; and, for
#    each of those, the "reverse pass" trick (route forward, route the reversed program from the final
#    layout, start from where that ends).
# 3. SWAP choice: when a 2-qubit gate's qubits are not neighbours, only SWAPs that bring them one step
#    closer are allowed (so a gate never needs more SWAPs than its distance - 1, like the basic router);
#    among those, the one that also helps the next gates most is chosen (distances of the next 20
#    2-qubit gates, weighted 0.8^j).
# 4. Safety: "auto" also runs the basic router and keeps whichever needs fewer SWAPs.
LOOKAHEAD, DECAY = 20, 0.8


def _region(device, k):
    """(m, adjacency) of the smallest connected prefix 0..m-1 with m >= k, or None."""
    if device.coupling is None or k > device.n_qubits:
        return None
    for m in range(max(1, k), device.n_qubits + 1):
        adj = {i: set() for i in range(m)}
        for a, b in device.coupling:
            if a < m and b < m:
                adj[a].add(b)
                adj[b].add(a)
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) == m:
            return m, adj
    return None


def _distances(adj):
    m = len(adj)
    inf = 10 ** 9
    dist = [[inf] * m for _ in range(m)]
    for s in range(m):
        dist[s][s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[s][v] == inf:
                    dist[s][v] = dist[s][u] + 1
                    dq.append(v)
    return dist


def _interactions(program):
    """Weight of each logical pair: how often it interacts (earlier gates count slightly more)."""
    w, n2 = {}, 0
    for op in program.ops:
        if len(op.qubits) == 2:
            a, b = sorted(op.qubits)
            w[(a, b)] = w.get((a, b), 0.0) + 1.0 / (1 + 0.05 * n2)
            n2 += 1
    return w


def _greedy_layout(program, adj, dist):
    """Place qubits that interact often on neighbouring physical qubits."""
    k = program.n_qubits
    w = _interactions(program)
    total = [0.0] * k
    for (a, b), x in w.items():
        total[a] += x
        total[b] += x

    def partner(p, r):
        return w.get((min(p, r), max(p, r)), 0.0)

    layout = [None] * k
    free = set(adj)
    placed = []
    while len(placed) < k:
        rest = [lq for lq in range(k) if layout[lq] is None]
        if placed:
            lq = max(rest, key=lambda c: (sum(partner(c, p) for p in placed), total[c], -c))
        else:
            lq = max(rest, key=lambda c: (total[c], -c))
        linked = [p for p in placed if partner(lq, p) > 0]
        if linked:
            q = min(free, key=lambda s: (sum(partner(lq, p) * dist[s][layout[p]] for p in linked), s))
        elif placed:
            q = min(free, key=lambda s: (min(dist[s][layout[p]] for p in placed), s))
        else:
            q = min(free, key=lambda s: (-len(adj[s]), s))
        layout[lq] = q
        free.discard(q)
        placed.append(lq)
    return layout


def _route_with(program, adj, dist, start):
    """Route from a given initial layout. Returns (ops, final layout, number of swaps)."""
    layout = list(start)                                       # logical -> physical
    where = {p: lq for lq, p in enumerate(layout)}             # physical -> logical
    twoq = [i for i, op in enumerate(program.ops) if len(op.qubits) == 2]
    ops, swaps, t = [], 0, 0
    for i, op in enumerate(program.ops):
        if len(op.qubits) == 2:
            while twoq[t] != i:
                t += 1
            a, b = op.qubits
            upcoming = [program.ops[j].qubits for j in twoq[t + 1:t + 1 + LOOKAHEAD]]
            while dist[layout[a]][layout[b]] > 1:
                pa, pb = layout[a], layout[b]
                d = dist[pa][pb]
                best = None
                for x, other in ((pa, pb), (pb, pa)):
                    for y in sorted(adj[x]):
                        if dist[y][other] != d - 1:
                            continue
                        lx, ly = where.get(x), where.get(y)
                        score = 0.0
                        for j, (c, e) in enumerate(upcoming):
                            pc = y if c == lx else x if c == ly else layout[c]
                            pe = y if e == lx else x if e == ly else layout[e]
                            score += DECAY ** j * dist[pc][pe]
                        key = (score, min(x, y), max(x, y))
                        if best is None or key < best[0]:
                            best = (key, x, y)
                _, x, y = best
                ops.append(Op("swap", (x, y)))
                swaps += 1
                lx, ly = where.pop(x, None), where.pop(y, None)
                if lx is not None:
                    layout[lx] = y
                    where[y] = lx
                if ly is not None:
                    layout[ly] = x
                    where[x] = ly
        ops.append(Op(op.name, tuple(layout[q] for q in op.qubits), op.params))
    return ops, layout, swaps


def route_lookahead(program, device):
    """Look-ahead router. Returns (Program, final layout, swaps, initial layout) or None when the
    device has no wiring constraints or no connected region fits the program."""
    k = program.n_qubits
    reg = _region(device, k)
    if reg is None or k == 0:
        return None
    m, adj = reg
    dist = _distances(adj)
    starts = [list(range(k)), _greedy_layout(program, adj, dist)]
    reverse = Program(k, program.n_clbits, list(reversed(program.ops)), program.measures)
    for s in list(starts):
        _, forward_end, _ = _route_with(program, adj, dist, s)
        _, backward_end, _ = _route_with(reverse, adj, dist, forward_end)
        starts.append(backward_end)
    best = None
    for s in starts:
        ops, final, swaps = _route_with(program, adj, dist, s)
        if best is None or swaps < best[2]:
            best = (ops, final, swaps, s)
    ops, final, swaps, start = best
    measures = {final[q]: c for q, c in program.measures.items()}
    used = [q for o in ops for q in o.qubits] + list(final) + list(measures)
    return Program(max(used) + 1, program.n_clbits, ops, measures), final, swaps, start


ROUTERS = ("auto", "lookahead", "basic", "noise-aware")
MAX_REGIONS = 16          # noise-aware: regions routed and scored (the best by a quick calibration score)


def _connected_subsets(adj_full, k):
    """All connected sets of k physical qubits, as sorted lists."""
    frontier = {frozenset([q]) for q in adj_full}
    for _ in range(k - 1):
        frontier = {S | {v} for S in frontier for q in S for v in adj_full[q] if v not in S}
    return sorted(sorted(S) for S in frontier)


def _distances_n(adj, n):
    """All-pairs hop distances inside adj (keys = physical qubits), as an n x n table."""
    inf = 10 ** 9
    dist = [[inf] * n for _ in range(n)]
    for src in adj:
        dist[src][src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[src][v] == inf:
                    dist[src][v] = dist[src][u] + 1
                    dq.append(v)
    return dist


def _lookahead_in_region(program, region, adj_full, n):
    """Look-ahead routing restricted to the physical qubits in `region` (same starts as route_lookahead)."""
    k = program.n_qubits
    R = set(region)
    adj = {q: adj_full[q] & R for q in region}
    dist = _distances_n(adj, n)
    starts = [list(region[:k]), _greedy_layout(program, adj, dist)]
    reverse = Program(k, program.n_clbits, list(reversed(program.ops)), program.measures)
    for st in list(starts):
        _, forward_end, _ = _route_with(program, adj, dist, st)
        _, backward_end, _ = _route_with(reverse, adj, dist, forward_end)
        starts.append(backward_end)
    best = None
    for st in starts:
        ops, final, swaps = _route_with(program, adj, dist, st)
        if best is None or swaps < best[2]:
            best = (ops, final, swaps, st)
    ops, final, swaps, start = best
    measures = {final[q]: c for q, c in program.measures.items()}
    used = [q for o in ops for q in o.qubits] + list(final) + list(measures)
    return Program(max(used) + 1, program.n_clbits, ops, measures), final, swaps, start


def _decoherence_rate(device, q):
    rate = 0.0
    if device.T1 is not None:
        rate += 1 / (2 * device.T1[q])
    if device.T_phi is not None:
        rate += 1 / device.T_phi[q]
    return rate


def expected_log_fidelity(native, device):
    """A quick estimate from the day's calibration (higher is better): gate errors of every pulse and cz,
    readout error of every measured qubit, and decoherence of every used qubit over the circuit's
    duration. A heuristic for choosing a placement, not a simulation."""
    from .executor import schedule, layer_duration
    log_f = 0.0
    for o in native.ops:
        if len(o.qubits) == 2:
            log_f += np.log1p(-min(0.999, device.error_2q(*o.qubits)))
        elif o.name in ("sx", "x"):
            log_f += np.log1p(-min(0.999, device.error_1q(o.qubits[0])))
    if device.readout_error is not None:
        for q in native.measures:
            p01, p10 = device.readout_error[q]
            log_f += np.log1p(-min(0.999, (p01 + p10) / 2))
    duration = float(sum(layer_duration(L, device) for L in schedule(native, device)))
    used = {q for o in native.ops for q in o.qubits} | set(native.measures)
    return log_f - duration * sum(_decoherence_rate(device, q) for q in used)


def _region_quick_score(region, adj_full, device):
    edges = [(a, b) for a in region for b in adj_full[a] if a < b and b in region]
    s = sum(np.log1p(-min(0.999, device.error_2q(a, b))) for a, b in edges) / max(1, len(edges))
    for q in region:
        if device.readout_error is not None:
            s += np.log1p(-min(0.999, sum(device.readout_error[q]) / 2))
        s -= 10.0 * _decoherence_rate(device, q)
    return s


def route_noise_aware(program, device):
    """v0.21.0: try every connected region of k physical qubits (the best MAX_REGIONS by a quick score),
    route each with the look-ahead router, compile, and keep the placement with the best
    expected_log_fidelity. The auto router's result is always one of the candidates.
    Returns (Program, final layout, swaps, initial layout) or None when there is nothing to choose.
    Note: a region far from qubit 0 makes the simulated register larger (physical qubits 0..max)."""
    k = program.n_qubits
    if device.coupling is None or k == 0 or k > device.n_qubits:
        return None
    adj_full = {q: set() for q in range(device.n_qubits)}
    for a, b in device.coupling:
        adj_full[a].add(b)
        adj_full[b].add(a)
    regions = _connected_subsets(adj_full, k)
    regions.sort(key=lambda R: -_region_quick_score(R, adj_full, device))
    candidates = [_lookahead_in_region(program, R, adj_full, device.n_qubits) for R in regions[:MAX_REGIONS]]
    auto = _route_choice(program, device, "auto")
    candidates.append((auto[0], auto[1], auto[2], auto[4]))
    return max(candidates, key=lambda c: expected_log_fidelity(_native(c[0]), device))


def _route_choice(program, device, router="auto"):
    """Returns (Program, final layout, swaps, router name, initial layout)."""
    if router not in ROUTERS:
        raise ValueError("router must be 'auto', 'lookahead', 'basic' or 'noise-aware'")
    if router == "noise-aware":
        chosen = route_noise_aware(program, device)
        if chosen is not None:
            prog, final, swaps, start = chosen
            return prog, final, swaps, "noise-aware", start
        router = "auto"                                 # no wiring constraints or no calibration: nothing to choose
    if router != "basic":
        smart = route_lookahead(program, device)
        if smart is not None:
            prog, final, swaps, start = smart
            if router == "lookahead":
                return prog, final, swaps, "lookahead", start
            try:
                basic = route_basic(program, device)
            except ValueError:
                return prog, final, swaps, "lookahead", start
            if (swaps, prog.n_qubits) <= (basic[2], basic[0].n_qubits):
                return prog, final, swaps, "lookahead", start
            return basic + ("basic", list(range(program.n_qubits)))
    return route_basic(program, device) + ("basic", list(range(program.n_qubits)))


def route(program, device, router="auto"):
    """Map logical qubits to physical ones, inserting SWAPs where needed.
    router: "auto" (look-ahead, or basic if that needs fewer SWAPs), "lookahead" or "basic".
    Returns (Program on physical qubits, final layout logical -> physical, number of SWAPs)."""
    return _route_choice(program, device, router)[:3]


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


def transpile(program, device, router="auto"):
    """Full compile. Returns (native program, info dict). router: auto | lookahead | basic | noise-aware."""
    routed, layout, swaps, router_used, start = _route_choice(program, device, router)
    native = _native(routed)
    out = native.ops
    info = {"swaps": swaps, "layout": layout, "initial_layout": start, "router": router_used, "n_ops": len(out),
            "n_2q": sum(1 for o in out if len(o.qubits) == 2),
            "n_sx": sum(1 for o in out if o.name in ("sx", "x")),
            "n_rz": sum(1 for o in out if o.name == "rz")}
    return native, info


def _native(routed):
    """Routed program -> native gates (cz + merged single-qubit gates with the fewest pulses)."""
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
    return Program(routed.n_qubits, routed.n_clbits, out, routed.measures)


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
