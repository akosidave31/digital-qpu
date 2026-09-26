"""Runs validation/quantum_validation.py (independent-reference checks of the quantum-circuit math)."""
import importlib.util
import pathlib
import pytest

_path = pathlib.Path(__file__).resolve().parents[1] / "validation" / "quantum_validation.py"
_spec = importlib.util.spec_from_file_location("quantum_validation", _path)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)
RESULTS = V.run_all()


def test_suite_is_complete():
    sections = {r["section"] for r in RESULTS}
    assert len(sections) == 8 and len(RESULTS) >= 90


def test_tolerances_unchanged():
    assert V.AMP_TOL == 1e-10 and V.PROB_TOL == 1e-10 and V.SIGMAS == 5.0 and V.SEED == 20260926


@pytest.mark.parametrize("r", RESULTS, ids=[f"{r['section']}: {r['name']}" for r in RESULTS])
def test_validation_case(r):
    assert r["passed"], f"{r['name']}: expected {r['expected']}; Digital-QPU {r['dq']}; reference {r['ref']}"


def test_reference_is_independent():
    src = _path.read_text()
    ref_part = src.split("# Digital-QPU side")[0]              # everything before the code under test
    imports = [ln.strip() for ln in ref_part.splitlines() if ln.strip().startswith(("import ", "from "))]
    assert imports and not any("digital" in ln for ln in imports), imports
