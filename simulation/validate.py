#!/usr/bin/env python3
"""
Validation harness for the aux-referenced OFDR pipeline.

Runs simulation.simulate.simulate() -> ofdr.process.process() across a
matrix of scenarios chosen to exercise the known risks of the method:
resolution limit, weak-signal floor, Nyquist aliasing, sweep-nonlinearity
robustness, tau_aux calibration error, and the uncorrected chromatic
dispersion.

Each scenario gets its own output subfolder (sim.npz + reflectogram csv/png)
so a flagged case can be re-inspected by hand with the ordinary CLI:
    python -m ofdr.process validation_results/doublet_2cell/sim.npz --dl 4

A summary.csv/summary.md is written with one row per (scenario, truth
reflector), plus a printed pass/fail table.

Usage:
    python -m simulation.validate
    python -m simulation.validate --list
    python -m simulation.validate --scenario doublet_2cell
"""
import argparse
import csv
import datetime
import sys
from pathlib import Path

import numpy as np

from scipy.signal import find_peaks

import ofdr.process as proc_mod
import simulation.simulate as sim_mod

DEFAULT_SIM = dict(points=1_000_000, speed=60.0, lam0=1535.0, dl=4.0,
                    bow_pm=50.0, ripple_mhz=15.0, noise=2e-4, beta2=0.0,
                    seed=7)
DEFAULT_PROC = dict(dl=4.0, window="kaiser", kaiser_beta=12.0,
                     peak_floor_db=-45.0, trim=0.01,
                     aux_a=1, aux_b=2, meas_a=3, meas_b=4)

BASELINE_REFLECTORS = ["0.046:0", "0.30:-25", "1.05:-30", "2.00:-35"]

# A pessimistic fast-ripple level, close to the monotonicity limit; the
# simulator's default of 15 MHz is comfortably below it.
STRESS_RIPPLE_MHZ = 100.0

SCENARIOS = [
    dict(name="baseline_strong_ripple", category="position",
         sim=dict(reflectors=BASELINE_REFLECTORS, ripple_mhz=STRESS_RIPPLE_MHZ),
         notes="Baseline reflector set with a pessimistic 100 MHz tuning "
               "ripple."),
    dict(name="baseline_default", category="position",
         sim=dict(reflectors=BASELINE_REFLECTORS),
         notes="Same reflectors, simulator default 15 MHz ripple -- "
               "regression anchor."),
    dict(name="doublet_2cell", category="resolution",
         sim=dict(reflectors=["0.500000:-30", "0.500028:-33"]),
         truth_zs=[0.500000, 0.500028],
         notes="~2 resolution cells apart (28 um at 13.9 um/cell) -- "
               "expected to resolve."),
    dict(name="doublet_1cell", category="resolution",
         sim=dict(reflectors=["0.500000:-30", "0.500014:-33"]),
         truth_zs=[0.500000, 0.500014],
         notes="~1 resolution cell apart (14 um) -- expected NOT to "
               "resolve; documents the floor rather than a bug."),
    dict(name="weak_near_floor", category="amplitude",
         sim=dict(reflectors=["0.046:0", "1.00:-45", "1.50:-50", "2.00:-55"]),
         notes="Weak reflectors near/under the default -45 dB peak floor."),
    dict(name="weak_near_floor_elevated_noise", category="amplitude",
         sim=dict(reflectors=["0.046:0", "1.00:-45", "1.50:-50", "2.00:-55"],
                   noise=1e-3),
         notes="Same weak reflectors, 5x detector noise -- where does "
               "recovery actually break down?"),
    dict(name="nyquist_edge_inside", category="range",
         sim=dict(reflectors=["6.80:-30"]),
         notes="Just inside z_nyq (~6.95 m at this operating point) -- "
               "edge-of-range behavior (trim, window taper)."),
    dict(name="nyquist_beyond_aliased", category="range",
         sim=dict(reflectors=["8.00:-30", "9.00:-35"]),
         notes="Beyond z_nyq -- must show up at the predicted folded "
               "position, not be silently unaccounted for."),
    dict(name="ripple_stress_near_limit", category="robustness",
         sim=dict(reflectors=BASELINE_REFLECTORS, ripple_mhz=STRESS_RIPPLE_MHZ * 1.1),
         notes="Ripple pushed close to the monotonicity limit -- expect a "
               "documented degradation/warning, not a silent bad number."),
    dict(name="bow_stress_2x", category="robustness",
         sim=dict(reflectors=BASELINE_REFLECTORS, bow_pm=100.0),
         notes="2x the default slow bow -- safety margin of resample_on_aux."),
    dict(name="tau_aux_miscal_0.5pct", category="calibration", tau_pct=0.5,
         sim=dict(reflectors=BASELINE_REFLECTORS),
         notes="tau_aux calibration off by 0.5%."),
    dict(name="tau_aux_miscal_1pct", category="calibration", tau_pct=1.0,
         sim=dict(reflectors=BASELINE_REFLECTORS),
         notes="tau_aux calibration off by 1% -- plausible one-time-"
               "calibration drift budget."),
    dict(name="tau_aux_miscal_2pct", category="calibration", tau_pct=2.0,
         sim=dict(reflectors=BASELINE_REFLECTORS),
         notes="tau_aux calibration off by 2%."),
    dict(name="dispersion_beta2_off", category="known_limitation",
         sim=dict(reflectors=["0.30:-25", "1.50:-30"], beta2=0.0),
         group="dispersion_pair",
         notes="Control run, no dispersion (SMF-28 beta2 = -2.17e-26 s^2/m)."),
    dict(name="dispersion_beta2_on", category="known_limitation",
         sim=dict(reflectors=["0.30:-25", "1.50:-30"], beta2=-2.17e-26),
         group="dispersion_pair", threshold_cells=10.0,
         notes="With SMF-28 dispersion, which the pipeline does not "
               "correct -- quantifies the size of this known limitation. "
               "Compare vs. dispersion_beta2_off."),
]


def _to_argv(overrides):
    argv = []
    for k, v in overrides.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.append(f"{flag}={v}")   # "=" so negative values parse
    return argv


def run_one(scenario, base_outdir):
    name = scenario["name"]
    outdir = base_outdir / name
    outdir.mkdir(parents=True, exist_ok=True)

    sim_overrides = dict(DEFAULT_SIM)
    sim_overrides.update(scenario.get("sim", {}))
    sim_argv = _to_argv(sim_overrides) + ["--out", str(outdir / "sim.npz")]
    sim_args = sim_mod.build_argparser().parse_args(sim_argv)
    sim_result = sim_mod.simulate(sim_args)

    proc_overrides = dict(DEFAULT_PROC)
    proc_overrides.update(scenario.get("proc", {}))
    if scenario.get("tau_pct") is not None:
        true_tau_ns = sim_result["meta"]["truth_tau_aux"] * 1e9
        proc_overrides["tau_aux_ns"] = true_tau_ns * (1 + scenario["tau_pct"] / 100.0)
        proc_overrides.pop("dl", None)
    proc_argv = [str(outdir / "sim.npz")] + _to_argv(proc_overrides) + \
        ["--out", str(outdir / name)]
    proc_args = proc_mod.build_argparser().parse_args(proc_argv)
    proc_result = proc_mod.process(proc_args)

    return grade(scenario, sim_result, proc_result)


def grade_resolution(scenario, proc_result, valley_db=3.0, window_cells=10):
    z, db, dz_bin = proc_result["z"], proc_result["db"], proc_result["dz_bin"]
    truth = scenario["truth_zs"]
    lo, hi = min(truth) - window_cells * dz_bin, max(truth) + window_cells * dz_bin
    mask = (z >= lo) & (z <= hi)
    zz, dd = z[mask], db[mask]
    pk, _ = find_peaks(dd)
    close = sorted(j for j in pk if min(abs(zz[j] - t) for t in truth) < 4 * dz_bin)
    if len(close) < 2:
        return [dict(scenario=scenario["name"], category="resolution",
                      status="not_resolved", pass_=(scenario["name"] == "doublet_1cell"),
                      notes=f"{len(close)} local maxima found near truth positions")]
    p1, p2 = close[0], close[-1]
    valley = dd[p1:p2 + 1].min()
    dip1, dip2 = dd[p1] - valley, dd[p2] - valley
    resolved = dip1 >= valley_db and dip2 >= valley_db
    expect_resolved = scenario["name"] == "doublet_2cell"
    return [dict(scenario=scenario["name"], category="resolution",
                  z_found=f"{zz[p1]:.6f}/{zz[p2]:.6f}",
                  status="resolved" if resolved else "not_resolved",
                  pass_=(resolved == expect_resolved),
                  notes=f"valley {dip1:.1f}/{dip2:.1f} dB below the two peaks")]


def grade_calibration(scenario, sim_result, proc_result):
    """tau_aux miscalibration rescales the WHOLE distance axis (every label,
    including the one used to compute dz_bin itself), by a factor that can
    move a peak by far more than the +-20-cell window process()'s own
    truth-comparison searches around z_true. That window is fine for small
    (sub-resolution) errors but silently returns a meaningless, window-
    bounded number once the true peak has drifted out of it -- so this
    scenario needs its own search, centered on the theoretically PREDICTED
    (mis-scaled) position, not on z_true.
    """
    eps = scenario["tau_pct"] / 100.0
    z, R, db, dz_bin = proc_result["z"], proc_result["R"], proc_result["db"], proc_result["dz_bin"]
    truths, dbs = sim_result["meta"]["truth_z"], sim_result["meta"]["truth_db"]
    farthest = max(truths)
    rows = []
    for zt, dbt in zip(truths, dbs):
        z_expect = zt * (1 + eps)   # derived: reported z scales by tau_aux_used/tau_aux_true
        w = (z > z_expect - 20 * dz_bin) & (z < z_expect + 20 * dz_bin)
        if not w.any():
            rows.append(dict(scenario=scenario["name"], category="calibration",
                              z_true=zt, db_true=dbt, status="not_found_near_predicted",
                              pass_=False,
                              notes=f"tau_pct={scenario['tau_pct']}, predicted {z_expect:.4f} m"))
            continue
        jj = int(np.argmax(np.where(w, R, 0)))
        err_um = (z[jj] - zt) * 1e6              # error vs the TRUE position -- what a user sees
        err_cells = err_um / (dz_bin * 1e6)
        err_vs_predicted_cells = (z[jj] - z_expect) / dz_bin  # resampling sanity check
        gate = 2.0 if zt == farthest else 8.0
        rows.append(dict(scenario=scenario["name"], category="calibration",
                          z_true=zt, db_true=dbt, z_found=z[jj], db_found=db[jj],
                          err_um=err_um, err_cells=err_cells, status="ok",
                          pass_=abs(err_cells) < gate,
                          notes=f"tau_pct={scenario['tau_pct']}, "
                                f"vs.predicted={err_vs_predicted_cells:+.2f} cells"))
    return rows


def grade(scenario, sim_result, proc_result):
    if scenario["category"] == "resolution":
        return grade_resolution(scenario, proc_result)
    if scenario["category"] == "calibration":
        return grade_calibration(scenario, sim_result, proc_result)

    threshold_cells = scenario.get("threshold_cells", 2.0)
    rows = []
    for c in proc_result["comparison"]:
        row = dict(scenario=scenario["name"], category=scenario["category"],
                   z_true=c["z_true"], db_true=c["db_true"],
                   z_found=c["z_found"], db_found=c["db_found"],
                   err_um=c["err_um"], err_cells=c["err_cells"],
                   status=c["status"], warnings=";".join(proc_result["warnings"]))

        if scenario["category"] == "range":
            row["pass_"] = c["status"] in ("ok", "aliased_as_expected") and \
                (c["err_cells"] is None or abs(c["err_cells"]) < threshold_cells)
        elif scenario["category"] == "robustness":
            frac_back = sim_result["diag"]["frac_back"]
            if frac_back > 0:
                row["pass_"] = None  # monotonicity broken -- expected, not gated
                row["notes"] = f"{frac_back*100:.2f}% backward steps (expected at this ripple)"
            else:
                row["pass_"] = c["status"] == "ok" and abs(c["err_cells"]) < 5.0
        elif scenario["category"] == "known_limitation":
            row["pass_"] = c["status"] == "ok" and c["err_cells"] is not None and abs(c["err_cells"]) < threshold_cells
        elif scenario["category"] == "amplitude":
            row["pass_"] = None  # tracked, not gated on the first pass -- see plan
        else:  # position
            amp_err = None if c["db_found"] is None else c["db_found"] - c["db_true"]
            row["amp_err_db"] = amp_err
            row["pass_"] = (c["status"] == "ok" and abs(c["err_cells"]) < threshold_cells
                             and amp_err is not None and abs(amp_err) < 1.5)
        rows.append(row)
    return rows


def write_summary(all_rows, outdir):
    cols = ["scenario", "category", "z_true", "db_true", "z_found", "db_found",
            "err_um", "err_cells", "amp_err_db", "status", "pass_", "warnings", "notes"]
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # NOTE: values come from numpy comparisons (numpy.bool_), so `is True`
    # silently fails (numpy.bool_(True) is not the Python singleton True) --
    # use a truthy check instead.
    n_gated = sum(1 for r in all_rows if r.get("pass_") is not None)
    n_pass = sum(1 for r in all_rows if r.get("pass_"))
    with open(outdir / "summary.md", "w") as f:
        f.write(f"# Aux-pipeline validation -- {datetime.date.today()}\n\n")
        f.write(f"{n_pass}/{n_gated} gated checks passed "
                f"({len(all_rows) - n_gated} rows tracked/informational, not gated).\n\n")
        f.write("| scenario | category | status | pass | err_cells | notes |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in all_rows:
            f.write(f"| {r.get('scenario','')} | {r.get('category','')} | "
                    f"{r.get('status','')} | {r.get('pass_','')} | "
                    f"{r.get('err_cells','')} | {r.get('notes','')} |\n")
    return n_pass, n_gated


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default=None, help="run only this scenario by name")
    p.add_argument("--list", action="store_true", help="list scenario names and exit")
    p.add_argument("--outdir", default="validation_results",
                   help="output folder (default: validation_results/)")
    a = p.parse_args()

    if a.list:
        for s in SCENARIOS:
            print(f"{s['name']:32s} [{s['category']}]  {s['notes']}")
        return

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = [s for s in SCENARIOS if a.scenario is None or s["name"] == a.scenario]
    if not scenarios:
        sys.exit(f"no scenario named {a.scenario!r} -- use --list")

    all_rows = []
    for s in scenarios:
        print(f"\n=== {s['name']} ===")
        rows = run_one(s, outdir)
        all_rows.extend(rows)

    n_pass, n_gated = write_summary(all_rows, outdir)
    print(f"\nwrote: {outdir/'summary.csv'}")
    print(f"wrote: {outdir/'summary.md'}")
    print(f"{n_pass}/{n_gated} gated checks passed "
          f"({len(all_rows) - n_gated} rows tracked/informational, not gated).")


if __name__ == "__main__":
    main()
