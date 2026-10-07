#!/usr/bin/env python3
"""
Synthetic 4-channel free-running dataset with KNOWN ground truth.

Simulates what a free-running DAQ records during one laser sweep:
4 channels, sampled uniformly in TIME (default 1 us), no wavelength
table.

    Ch1 / Ch2 : Aux MZI, the ruler (complementary)
    Ch3 / Ch4 : Measurement interferometer, with the test fiber (complementary)

Realistic imperfections of a swept-laser setup:
  - Sweep nonlinearity: slow bow (default 50 pm p-p, see --bow-pm)
  - fast tuning ripple (default 15 MHz rms, see --ripple-mhz)
  - differing photodiode sensitivity (gain drift 0.84 -> 1.14)
  - DC offset and detector noise

Purpose: validate the pipeline (ofdr/process.py) against known ground
truth before touching hardware. If it doesn't place the reflectors at the
given positions, the software is at fault -- not the bench.

Example:
    python -m simulation.simulate --points 1000000 --speed 60 --dl 4 \
        --reflectors 0.046:0 0.30:-25 1.05:-30 2.00:-35 --out testdata.npz
"""

import argparse
import numpy as np

C = 299_792_458.0
NG = 1.468


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=1_000_000)
    p.add_argument("--step-us", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=60.0, help="nm/s")
    p.add_argument("--lam0", type=float, default=1535.0, help="Start wavelength nm")
    p.add_argument("--dl", type=float, default=4.0, help="Aux arm difference m")
    p.add_argument("--reflectors", nargs="+",
                   default=["0.046:0", "0.30:-25", "1.05:-30", "2.00:-35"],
                   help="List z_in_m:dB")
    p.add_argument("--bow-pm", type=float, default=50.0,
                   help="slow bow p-p in pm")
    p.add_argument("--ripple-mhz", type=float, default=15.0,
                   help="fast ripple rms in MHz. Above the monotonicity "
                        "limit (printed at runtime) nu(t) runs backwards "
                        "at times -- at that point no aux can help anymore.")
    p.add_argument("--noise", type=float, default=2e-4, help="Detector noise")
    p.add_argument("--beta2", type=float, default=0.0,
                   help="GVD parameter in s^2/m (e.g. -2.17e-26 for SMF-28). "
                        "0 (default) = no dispersion, matches prior behavior. "
                        "Not corrected in the pipeline (known limitation).")
    p.add_argument("--internal-reflectors", nargs="+", default=[],
                   help="Fixed reflections that are ALWAYS present regardless "
                        "of the DUT (--reflectors) -- e.g. a connector or "
                        "circulator-port back-reflection inside the "
                        "instrument itself. Same z:dB format as --reflectors. "
                        "Added to ground truth like any other reflector, but "
                        "conceptually separate: they model fixed reflections "
                        "of the instrument that a no-fiber control scan "
                        "reveals.")
    p.add_argument("--ghosts", action="store_true",
                   help="Add multipath/double-bounce ghost tones: for every "
                        "pair of tones present (DUT + internal reflectors + "
                        "the aux delay itself), add a weak term at the SUM of "
                        "their delays, amplitude = product of their field "
                        "amplitudes x --ghost-coupling. This is the standard "
                        "mechanism for 'sum position' ghost peaks in "
                        "OFDR/OTDR (light double-bounces between two partial "
                        "reflectors before returning) -- reproduces "
                        "harmonic/sum peaks (e.g. 2x, main+fiber, aux+main).")
    p.add_argument("--ghost-coupling", type=float, default=1.0,
                   help="Extra multiplicative factor on ghost amplitudes "
                        "beyond the amp_i*amp_j product model (default 1.0 "
                        "= pure product model, no extra loss/gain).")
    p.add_argument("--aux-leak-db", type=float, default=-6.0,
                   help="How strongly the aux tone itself participates in "
                        "ghosting (dB relative to the aux's own unit "
                        "amplitude) -- models partial optical crosstalk "
                        "between the aux and measurement paths. Only matters "
                        "if --ghosts is set.")
    p.add_argument("--saturation-frac", type=float, default=0.0,
                   help="Fraction of the sweep (from the start) where the "
                        "detector is saturated/clipped, e.g. 0.2 for 'first "
                        "20%% of points'. 0 (default) = no saturation. "
                        "Saturation distorts the aux phase and can look "
                        "like a monotonicity failure of the laser.")
    p.add_argument("--saturation-ceiling", type=float, default=0.5,
                   help="Within the saturated region, each channel is "
                        "hard-clipped to this fraction of its own "
                        "(unsaturated) peak-to-peak range. Smaller = worse "
                        "saturation. Only matters if --saturation-frac > 0.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="testdata.npz")
    return p


def main():
    a = build_argparser().parse_args()
    simulate(a)


def simulate(a):
    rng = np.random.default_rng(a.seed)
    n = a.points
    dt = a.step_us * 1e-6
    t = np.arange(n) * dt

    # --- true wavelength vs time: linear + bow + ripple -----------
    lam_lin = a.lam0 * 1e-9 + a.speed * 1e-9 * t
    bow = a.bow_pm * 1e-12 * (np.sin(np.pi * t / t[-1]) - 0.5)
    lam_mid = lam_lin[n // 2]
    ripple_m = a.ripple_mhz * 1e6 * lam_mid**2 / C            # MHz -> m
    # fast ripple: band-limited to periods of 2.5 - 30 pm of the sweep
    dlam_per_sample = a.speed * 1e-9 * dt                     # m per sample
    per_lo_samp = 2.5e-12 / dlam_per_sample
    per_hi_samp = 30e-12 / dlam_per_sample
    spec = np.zeros(n // 2 + 1, complex)
    ks = np.arange(n // 2 + 1)
    per_samp = np.divide(n, np.maximum(ks, 1e-9))
    band = (per_samp > per_lo_samp) & (per_samp < per_hi_samp)
    spec[band] = np.exp(2j * np.pi * rng.random(band.sum()))
    ripple = np.fft.irfft(spec, n)
    ripple *= ripple_m / (ripple.std() + 1e-30)
    lam = lam_lin + bow + ripple
    nu = C / lam

    # --- Monotonicity check -----------------------------------------------
    # If the slope of the ripple exceeds the nominal sweep slope,
    # the laser is momentarily running BACKWARDS in nu. In that case the
    # aux phase is no longer monotonic and no software can rescue it.
    dnu = np.diff(nu)
    frac_back = float((dnu * np.sign(dnu[0]) < 0).mean())
    a_max = C / lam_mid**2 * 2.5e-12 / (2 * np.pi)     # shortest period 2.5 pm
    print(f"  Ripple {a.ripple_mhz:.0f} MHz rms "
          f"(monotonicity limit at this bandwidth: ~{a_max/1e6:.0f} MHz)")
    print(f"  Monotonicity: {frac_back*100:.3f} % of steps run backwards"
          + ("  <-- PROBLEM: no software can rescue this"
             if frac_back > 0 else "  (ok)"))

    # --- Signals ----------------------------------------------------------
    refl = []
    for s in a.reflectors:
        z, db = s.split(":")
        refl.append((float(z), float(db)))

    internal_refl = []
    for s in a.internal_reflectors:
        z, db = s.split(":")
        internal_refl.append((float(z), float(db)))

    all_refl = refl + internal_refl

    nu0 = nu.mean()
    meas = np.zeros(n)
    tones = []   # (tau, amp) of every fundamental, for optional ghosting below
    for z, db in all_refl:
        tau = 2 * NG * z / C                      # round trip
        amp = 10 ** (db / 20.0)
        ph = rng.random() * 2 * np.pi
        # quadratic spectral phase from fiber GVD, round trip 2z (0 by default)
        phi_disp = 0.5 * a.beta2 * (2 * np.pi * (nu - nu0)) ** 2 * (2 * z)
        meas += amp * np.cos(2 * np.pi * tau * nu + ph + phi_disp)
        tones.append((tau, amp))
    meas /= max(1.0, max(10 ** (d / 20.0) for _, d in all_refl))

    tau_aux = NG * a.dl / C                       # transmission: one-way only
    aux = np.cos(2 * np.pi * tau_aux * nu + rng.random() * 2 * np.pi)

    if a.ghosts:
        # Multipath/double-bounce ghosts: light that partially reflects off
        # tone i, continues, partially reflects off tone j, and returns,
        # travels the SUM of the two individual round-trip delays. This is
        # the standard explanation for "sum position" peaks in OFDR/OTDR
        # (and i==j gives the 2nd-harmonic case). The aux tone is included
        # (at a reduced, separately-tunable coupling) to model imperfect
        # isolation between the aux and measurement paths.
        aux_amp = 10 ** (a.aux_leak_db / 20.0)
        ghost_tones = tones + [(tau_aux, aux_amp)]
        n_tones = len(ghost_tones)
        for gi in range(n_tones):
            for gj in range(gi, n_tones):
                tau_i, amp_i = ghost_tones[gi]
                tau_j, amp_j = ghost_tones[gj]
                g_amp = amp_i * amp_j * a.ghost_coupling
                g_ph = rng.random() * 2 * np.pi
                meas += g_amp * np.cos(2 * np.pi * (tau_i + tau_j) * nu + g_ph)

    # --- Detector model: DC offset, complementary, gain drift, noise ---
    gain = np.linspace(0.84, 1.14, n)
    vis_m, vis_a = 0.75, 0.9
    ch1 = 0.5 * (1 + vis_a * aux)
    ch2 = 0.5 * (1 - vis_a * aux) / gain
    ch3 = 0.5 * (1 + vis_m * meas)
    ch4 = 0.5 * (1 - vis_m * meas) / gain
    for arr in (ch1, ch2, ch3, ch4):
        arr += rng.normal(0, a.noise, n)

    if a.saturation_frac > 0:
        # Detector saturation at the start of the sweep. Modeled as a hard
        # ceiling, calibrated
        # from each channel's OWN unsaturated range, applied only to the
        # first --saturation-frac of samples.
        sat_n = int(a.saturation_frac * n)
        for arr in (ch1, ch2, ch3, ch4):
            lo, hi = arr[sat_n:].min(), arr[sat_n:].max()
            ceiling = lo + a.saturation_ceiling * (hi - lo)
            np.clip(arr[:sat_n], lo, ceiling, out=arr[:sat_n])

    span = a.speed * (n * dt)
    is_internal = np.array([False] * len(refl) + [True] * len(internal_refl))
    np.savez_compressed(
        a.out, ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
        step_us=a.step_us, lam0_nm=a.lam0, speed_nms=a.speed,
        truth_z=np.array([z for z, _ in all_refl]),
        truth_db=np.array([d for _, d in all_refl]),
        truth_is_internal=is_internal,
        truth_tau_aux=tau_aux, truth_dl=a.dl)

    print(f"written: {a.out}")
    print(f"  {n:,} points, {a.step_us} us clock -> {n*dt:.3f} s")
    print(f"  Sweep {a.speed} nm/s from {a.lam0} nm -> span {span:.2f} nm")
    print(f"  Aux dL = {a.dl} m -> tau_aux = {tau_aux*1e9:.3f} ns "
          f"(appears at z = {a.dl/2:.3f} m)")
    print("  GROUND TRUTH (DUT reflectors):")
    for z, db in refl:
        print(f"    z = {z:7.4f} m   {db:6.1f} dB   "
              f"tau = {2*NG*z/C*1e9:7.3f} ns")
    if internal_refl:
        print("  GROUND TRUTH (fixed internal reflectors):")
        for z, db in internal_refl:
            print(f"    z = {z:7.4f} m   {db:6.1f} dB   "
                  f"tau = {2*NG*z/C*1e9:7.3f} ns")
    if a.ghosts:
        print(f"  Ghosts enabled: {len(tones)+1} fundamentals -> "
              f"{(len(tones)+1)*(len(tones)+2)//2} ghost tones "
              f"(coupling x{a.ghost_coupling}, aux leak {a.aux_leak_db} dB)")
    if a.saturation_frac > 0:
        print(f"  Detector saturation: first {a.saturation_frac*100:.0f}% "
              f"of samples clipped to {a.saturation_ceiling*100:.0f}% of "
              f"each channel's own range")

    return dict(
        ch1=ch1, ch2=ch2, ch3=ch3, ch4=ch4,
        meta=dict(step_us=a.step_us, lam0_nm=a.lam0, speed_nms=a.speed,
                   truth_z=np.array([z for z, _ in all_refl]),
                   truth_db=np.array([d for _, d in all_refl]),
                   truth_is_internal=is_internal,
                   truth_tau_aux=tau_aux, truth_dl=a.dl),
        diag=dict(frac_back=frac_back, a_max_mhz=a_max / 1e6,
                   span_nm=span))


if __name__ == "__main__":
    main()
