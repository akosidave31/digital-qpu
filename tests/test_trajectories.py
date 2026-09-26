import math
import numpy as np
import pytest
from digital_qpu import (parse, probabilities, Device, DEVICES, QPU, engine_for, final_trajectories,
                         MAX_TRAJECTORY_QUBITS, calibrate, QasmError)

G1 = ["id", "x", "h", "s", "t", "sx", "sdg"]
GP = ["rx", "ry", "rz", "u1"]


def rand_program(rng, n, depth):
    L = [f"OPENQASM 2.0; qreg q[{n}]; creg c[{n}];"]
    for _ in range(depth):
        r = rng.random()
        if n > 1 and r < 0.3:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            L.append(f"{['cx', 'cz', 'swap'][int(rng.integers(3))]} q[{a}],q[{b}];")
        elif r < 0.6:
            L.append(f"{GP[int(rng.integers(4))]}({rng.uniform(-3, 3):.4f}) q[{int(rng.integers(n))}];")
        else:
            L.append(f"{G1[int(rng.integers(len(G1)))]} q[{int(rng.integers(n))}];")
    L.append("measure q -> c;")
    return parse(" ".join(L))


def dev(n, feats):
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    kw = dict(coupling=pairs, gate_time_1q=0.05, gate_time_2q=0.3, virtual_rz=True)
    if "gates" in feats:
        kw.update(gate_error_1q=[0.01] * n, gate_error_2q=0.04)
    if "thermal" in feats:
        kw.update(T1=[15.0 + 3 * q for q in range(n)], T_phi=[12.0 + 2 * q for q in range(n)])
    if "zz" in feats:
        kw.update(zz={p: 0.3 for p in pairs})
    if "drive" in feats:
        kw.update(drive_crosstalk=0.03)
    return Device("t", n, **kw)


def test_noise_free_trajectories_are_exact():
    """No random events happen without stochastic noise, so trajectories must equal the pure state."""
    from digital_qpu.executor import final_state
    rng = np.random.default_rng(1)
    for n in (2, 3, 5):
        d = dev(n, ("zz", "drive"))
        assert engine_for(n, d) == "pure"
        for _ in range(3):
            prog = rand_program(rng, n, 25)
            pure = final_state(prog, d).probs()
            traj = final_trajectories(prog, d, n_traj=3)
            assert traj.n_traj == 3 and np.max(np.abs(pure - traj.probs())) < 1e-12


@pytest.mark.parametrize("feats", [("gates",), ("thermal",), ("gates", "thermal", "zz", "drive")])
def test_trajectories_match_exact_noisy_results(feats):
    rng = np.random.default_rng(len(feats))
    N = 3000
    for n in (2, 3, 4):
        d = dev(n, feats)
        for s in range(2):
            prog = rand_program(rng, n, 15)
            exact = probabilities(prog, d, method="density")
            traj = probabilities(prog, d, method="trajectories", n_traj=N, seed=s)
            for k, p in exact.items():
                assert abs(traj.get(k, 0) - p) < 5 * math.sqrt(p * (1 - p) / N) + 2e-3, (feats, n, k)


def test_more_trajectories_get_closer():
    rng = np.random.default_rng(9)
    d = dev(3, ("gates", "thermal", "zz", "drive"))
    progs = [rand_program(rng, 3, 20) for _ in range(3)]
    def err(N):
        e = []
        for i, prog in enumerate(progs):
            exact = probabilities(prog, d, method="density")
            traj = probabilities(prog, d, method="trajectories", n_traj=N, seed=100 + i)
            e.append(0.5 * sum(abs(traj.get(k, 0) - v) for k, v in exact.items()))
        return np.mean(e)
    assert err(4000) < err(100)


def test_t1_jumps_have_the_right_rate():
    d = Device("t1", 1, T1=[10.0], gate_time_1q=1.0)
    N = 4000
    for k in (0, 5, 15):
        prog = parse("OPENQASM 2.0; qreg q[1]; creg c[1]; x q[0];" + " id q[0];" * k + " measure q -> c;")
        p = math.exp(-(k + 1) / 10.0)
        got = probabilities(prog, d, method="trajectories", n_traj=N, seed=k)["1"]
        assert abs(got - p) < 5 * math.sqrt(p * (1 - p) / N)


def test_engine_selection_and_limits():
    dq12 = DEVICES["dq-12"]
    assert engine_for(8, dq12) == "density" and engine_for(9, dq12) == "trajectories"
    assert engine_for(12, DEVICES["ideal"]) == "pure"
    assert MAX_TRAJECTORY_QUBITS == 16
    with pytest.raises(ValueError):
        engine_for(3, dq12, "magic")


def test_dq12_ladder():
    dq12 = DEVICES["dq-12"]
    deg = [len(dq12.neighbours(q)) for q in range(12)]
    assert len(dq12.coupling) == 16 and max(deg) == 3 and min(deg) == 2
    assert len(dq12.zz_pairs()) == 16
    assert calibrate(dq12, 3).name == "dq-12@day3"


def test_noisy_12_qubit_ghz_runs():
    snake = [0, 1, 2, 3, 4, 5, 11, 10, 9, 8, 7, 6]            # follows the ladder: no SWAPs needed
    body = " ".join([f"h q[{snake[0]}];"] + [f"cx q[{a}], q[{b}];" for a, b in zip(snake, snake[1:])])
    r = QPU("dq-12").run(f"OPENQASM 2.0; qreg q[12]; creg c[12]; {body} measure q -> c;",
                         shots=500, seed=4, n_traj=40).result()
    assert r["engine"] == "trajectories (40)"
    c = r["counts"]
    assert (c.get("0" * 12, 0) + c.get("1" * 12, 0)) / 500 > 0.3


@pytest.mark.slow
def test_noisy_shor_on_dq12_completes():
    from digital_qpu import shor15
    r = QPU("dq-12").run(shor15()["qasm"], shots=1000, seed=1, n_traj=100).result()
    assert (r["engine"] == "density" or r["engine"].startswith("trajectories")) and sum(r["counts"].values()) == 1000
