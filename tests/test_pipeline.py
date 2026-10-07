"""End-to-end test: simulate a sweep with known reflectors, process it,
and check that every reflector is found at the right position.

Run with:  python -m pytest tests/
"""

import numpy as np

import ofdr.process as proc
import simulation.simulate as sim
from ofdr.sweep_calculator import compute


def _run(tmp_path, *sim_args):
    out = tmp_path / "sim.npz"
    s = sim.simulate(sim.build_argparser().parse_args(
        ["--points=200000", f"--out={out}", *sim_args]))
    r = proc.process(proc.build_argparser().parse_args(
        [str(out), "--dl=4", f"--out={tmp_path / 'sim'}"]))
    return s, r


def test_reflectors_found_within_two_cells(tmp_path):
    _, r = _run(tmp_path, "--reflectors", "0.046:0", "0.30:-25", "1.05:-30")
    assert r["passed"]
    for c in r["comparison"]:
        assert c["status"] == "ok"
        assert abs(c["err_cells"]) < 2


def test_reflector_beyond_nyquist_is_folded_not_lost(tmp_path):
    _, r = _run(tmp_path, "--reflectors", "0.30:0", "8.00:-20")
    statuses = {c["status"] for c in r["comparison"]}
    assert "aliased_as_expected" in statuses
    assert "out_of_range_unaccounted" not in statuses


def test_resolution_matches_sweep_calculator(tmp_path):
    _, r = _run(tmp_path)
    expected = compute(200_000, speed_nms=60.0)
    # same physics, two independent code paths: agree within a few percent
    assert np.isclose(r["z_nyq"], expected["z_max"], rtol=0.05)
    assert np.isclose(r["dz_bin"], expected["dz"], rtol=0.05)
