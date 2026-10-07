#!/usr/bin/env python3
"""
OFDR processing with an auxiliary Mach-Zehnder interferometer (4 channels).

The laser sweep is never perfectly linear. Instead of trusting a wavelength
table, the frequency axis is taken from the phase of an auxiliary MZI
("aux", the ruler) that is recorded alongside the measurement:

    Ch1 / Ch2 : aux MZI = ruler          (complementary outputs)
    Ch3 / Ch4 : measurement interferometer with the device under test
                (complementary outputs)

Pipeline:
    balanced subtraction -> resample onto uniform aux phase (= uniform
    optical frequency) -> baseline removal -> window -> FFT -> reflectogram,
    peak list, resolution/width check, noise floor

IMPORTANT: the aux phase is NOT smoothed. It has to carry the fast tuning
ripple of the laser, otherwise the ripple is not corrected away.

Calibration of tau_aux (see check_aux.py):
    tau_aux = total unwrapped aux phase / (2*pi * optical frequency span)
Pass the result via --tau-aux-ns. Without it, tau_aux is estimated from the
arm length difference --dl (tau_aux = n_g * dL / c), good to ~1 %.

Examples:
    python -m ofdr.process testdata.npz --dl 4
    python -m ofdr.process scan.npz --tau-aux-ns 19.587 --zmax 2.5
"""

import argparse
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.signal.windows import kaiser, hann, blackmanharris

from ofdr.io import load_channels

C = 299_792_458.0
NG = 1.468          # group index of SMF-28 at 1550 nm


# ------------------------------------------------- balanced subtraction
def balanced(a, b, nseg=32):
    """P = a - g*b with a slowly varying gain g, estimated segment-wise
    from the ratio of medians. Removes the common-mode intensity (laser
    power drift, DC) and keeps the interference term."""
    n = len(a)
    seg = max(n // nseg, 256)
    g = np.empty(n)
    for i in range(0, n, seg):
        s = slice(i, min(i + seg, n))
        m = np.median(b[s])
        g[s] = np.median(a[s]) / m if m > 0 else 1.0
    g = uniform_filter1d(g, seg)
    return a - g * b, g


def analytic(x):
    """Analytic signal via FFT (one-sided spectrum), like scipy's hilbert."""
    n = len(x)
    X = np.fft.fft(x)
    X[n // 2 + 1:] = 0.0
    X[1:n // 2] *= 2.0
    return np.fft.ifft(X)


# ------------------------------------------------------------- the core
def resample_on_aux(meas, aux, tau_aux, trim=0.01):
    """Resample the measurement signal onto uniform aux phase.

    Core idea: the aux phase is phi(t) = 2*pi*tau_aux*nu(t) + const.
    Uniform steps in phi are therefore uniform steps in optical frequency
    nu -- no matter how non-uniformly the laser was actually swept. That
    makes the axis linear and the FFT valid.

    Returns: (y, dnu, span_nu, diag)
    """
    n = len(aux)
    an = analytic(aux)
    phi = np.unwrap(np.angle(an))
    if phi[-1] < phi[0]:                 # nu decreases when lambda increases
        phi = -phi

    # Trim the edges: the laser is still ramping up there, and the
    # analytic signal has edge artifacts.
    k = max(1, int(trim * n))
    sl = slice(k, n - k)
    phi = phi[sl]
    meas = meas[sl]
    amp = np.abs(an)[sl]

    # Enforce monotonicity (noise can cause tiny backward steps).
    # The step must be large compared to the numerical precision of phi,
    # otherwise duplicates remain -- hence relative to the mean step.
    eps = 1e-6 * (phi[-1] - phi[0]) / len(phi)
    phi = np.maximum.accumulate(phi) + np.arange(len(phi)) * eps

    m = len(phi)
    phi_u = np.linspace(phi[0], phi[-1], m)
    y = PchipInterpolator(phi, meas)(phi_u)

    dphi = phi_u[1] - phi_u[0]
    dnu = dphi / (2 * np.pi * tau_aux)
    span_nu = (phi[-1] - phi[0]) / (2 * np.pi * tau_aux)
    fringes = (phi[-1] - phi[0]) / (2 * np.pi)
    diag = dict(fringes=fringes, pts_per_fringe=m / fringes,
                amp_min=amp.min() / amp.mean(), n_used=m)
    return y, dnu, span_nu, diag


def noise_floor(db):
    """RMS noise floor of a reflectogram, in the same dB scale as `db`.

    `db` is 20*log10(R/Rmax), so the linear power is 10**(db/10) and the
    floor -- the RMS of the background -- is 10*log10(mean power).

    The mean is taken as median/ln2 instead of the plain mean: for the
    Rayleigh-distributed magnitude of a complex-Gaussian background the two
    are the same, but the median ignores the reflector peaks, so no
    peak-free window has to be picked by hand. Over a full trace the plain
    mean comes out far too high because the peaks dominate it.
    """
    p = 10.0 ** (np.asarray(db, float) / 10.0)
    return 10.0 * np.log10(np.median(p) / np.log(2.0))


def window_limit_bins(win, pad):
    """-3 dB width of the window's own main lobe, in unpadded FFT bins.

    Measured, not looked up in a table, and measured on EXACTLY the grid the
    data is measured on (same window length, same pad factor): a peak width
    read off a 1x grid is quantised to whole bins, so it can only be compared
    against a limit that carries the same quantisation. A hard-coded number
    is the true continuous value and silently becomes the wrong reference as
    soon as --pad-factor changes.

    Probe is a cosine sitting half a bin off grid -- the worst case, and the
    normal case for a real reflector, which has no reason to land on a bin.
    """
    n = win.size
    probe = np.cos(2 * np.pi * (n // 8 + 0.5) * np.arange(n) / n)
    s = np.abs(np.fft.rfft(probe * win, n=n * pad))
    i = int(np.argmax(s))
    half = s[i] / np.sqrt(2.0)
    l = r = i
    while l > 0 and s[l] > half:
        l -= 1
    while r < len(s) - 1 and s[r] > half:
        r += 1
    return (r - l) / pad


# ----------------------------------------------------------------- CLI
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="4-channel capture (.npz)")
    p.add_argument("--tau-aux-ns", type=float, default=None,
                   help="calibrated aux delay in ns (preferred)")
    p.add_argument("--dl", type=float, default=None,
                   help="aux arm length difference in m (fallback, if tau_aux "
                        "has not been calibrated yet)")
    p.add_argument("--window", default="kaiser",
                   choices=["hann", "blackmanharris", "kaiser"])
    p.add_argument("--kaiser-beta", type=float, default=12.0)
    p.add_argument("--pad-factor", type=int, default=1,
                   help="zero-pad the FFT to this multiple. It INTERPOLATES "
                        "the peak shape, it does NOT improve resolution -- "
                        "use it when comparing peak widths between window "
                        "settings (4-8 is plenty)")
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--peak-floor-db", type=float, default=-45.0)
    p.add_argument("--trim", type=float, default=0.01,
                   help="fraction discarded at each edge")
    p.add_argument("--aux-a", type=int, default=1, choices=[1, 2, 3, 4],
                   help="first channel of the aux pair (default 1)")
    p.add_argument("--aux-b", type=int, default=2, choices=[1, 2, 3, 4],
                   help="second channel of the aux pair (default 2)")
    p.add_argument("--meas-a", type=int, default=3, choices=[1, 2, 3, 4],
                   help="first channel of the measurement pair (default 3)")
    p.add_argument("--meas-b", type=int, default=4, choices=[1, 2, 3, 4],
                   help="second channel of the measurement pair (default 4)")
    p.add_argument("--single", action="store_true",
                   help="skip balanced subtraction; use only the stronger "
                        "channel of each pair (by median power), high-pass "
                        "filtered on its own. Use this when a pair is very "
                        "unbalanced (one channel near the noise floor) and "
                        "balanced subtraction may inject more noise than it "
                        "cancels.")
    p.add_argument("--zero-m", type=float, default=None,
                   help="put 0 of the PLOT axis here (absolute m), e.g. at a "
                        "connector reflection, so a fiber end reads as its "
                        "length. The CSV stays absolute.")
    p.add_argument("--zero-label", default="connector",
                   help="name of the --zero-m point on the axis label")
    p.add_argument("--out", default=None, help="output prefix")
    return p


def main():
    process(build_argparser().parse_args())


def process(a):
    warnings = []

    chans, meta = load_channels(a.scan)
    missing = [n for n in (a.aux_a, a.aux_b, a.meas_a, a.meas_b) if n not in chans]
    if missing:
        sys.exit(f"Missing channels in file: {['Ch%d' % n for n in missing]}")
    ch1, ch2 = chans[a.aux_a], chans[a.aux_b]
    ch3, ch4 = chans[a.meas_a], chans[a.meas_b]

    if a.tau_aux_ns is not None:
        tau_aux = a.tau_aux_ns * 1e-9
        src = "calibrated"
    elif a.dl is not None:
        tau_aux = NG * a.dl / C
        src = f"estimated from dL = {a.dl} m"
    else:
        sys.exit("specify --tau-aux-ns or --dl")
    n = len(ch1)
    print(f"{a.scan}: {n:,} points x 4 channels")
    print(f"aux pair: Ch{a.aux_a}/Ch{a.aux_b}   "
          f"measurement pair: Ch{a.meas_a}/Ch{a.meas_b}")
    print(f"tau_aux = {tau_aux*1e9:.4f} ns  ({src})"
          f"  -> aux appears at z = {C*tau_aux/(2*NG):.3f} m")

    if a.single:
        aux_ch, aux_name = ((ch2, f"Ch{a.aux_b}") if np.median(ch2) > np.median(ch1)
                            else (ch1, f"Ch{a.aux_a}"))
        meas_ch, meas_name = ((ch4, f"Ch{a.meas_b}") if np.median(ch4) > np.median(ch3)
                              else (ch3, f"Ch{a.meas_a}"))
        print(f"single-channel mode: aux={aux_name}, meas={meas_name} "
              "-- stronger of each pair, high-pass filtered individually")
        aux = aux_ch - uniform_filter1d(aux_ch, max(len(aux_ch) // 32, 256))
        meas = meas_ch - uniform_filter1d(meas_ch, max(len(meas_ch) // 32, 256))
    else:
        aux, ga = balanced(ch1, ch2)
        meas, gm = balanced(ch3, ch4)
        print(f"balanced subtraction: g_meas {gm.min():.3f}..{gm.max():.3f}, "
              f"g_aux {ga.min():.3f}..{ga.max():.3f}")

    y, dnu, span_nu, diag = resample_on_aux(meas, aux, tau_aux, a.trim)
    m = len(y)
    print(f"Aux: {diag['fringes']:,.0f} fringes, "
          f"{diag['pts_per_fringe']:.1f} points/fringe, "
          f"amplitude minimum {diag['amp_min']*100:.0f} % of mean")
    if diag["pts_per_fringe"] < 4:
        print("  WARNING: fewer than 4 points per aux fringe -- sweep slower "
              "or shorten the aux path.")
        warnings.append("low_pts_per_fringe")
    if diag["amp_min"] < 0.2:
        print("  WARNING: aux contrast drops somewhere (polarization "
              "fading?). Phase unreliable there.")
        warnings.append("low_aux_contrast")

    # Remove residual baseline
    t = np.linspace(-1, 1, m)
    y = y - np.polyval(np.polyfit(t, y, 5), t)

    dz_bin = C / (2 * NG * span_nu)
    z_nyq = C / (4 * NG * dnu)
    lam_mid = 1565e-9
    print(f"Span {span_nu/1e12:.3f} THz "
          f"(~{span_nu*lam_mid**2/C*1e9:.1f} nm), point spacing "
          f"{dnu/1e6:.2f} MHz")
    print(f"  RESOLUTION {dz_bin*1e6:.2f} um   RANGE {z_nyq:.3f} m   "
          f"cells {z_nyq/dz_bin:,.0f}")
    print(f"  !! no reflector beyond {z_nyq:.2f} m, otherwise it will alias")

    win = {"hann": hann(m), "blackmanharris": blackmanharris(m),
           "kaiser": kaiser(m, a.kaiser_beta)}[a.window]
    pad = max(1, a.pad_factor)
    m_fft = m * pad
    R = np.abs(np.fft.rfft(y * win, n=m_fft))
    z = np.arange(len(R)) * C / (2 * NG * dnu * m_fft)
    db = 20 * np.log10(R / R.max() + 1e-15)
    if pad > 1:
        print(f"  zero-padded {pad}x: plot grid {(z[1]-z[0])*1e6:.2f} um "
              f"(resolution is still {dz_bin*1e6:.2f} um)")

    i = int(np.argmax(R))
    half = R[i] / np.sqrt(2)
    l = r = i
    while l > 0 and R[l] > half:
        l -= 1
    while r < len(R) - 1 and R[r] > half:
        r += 1
    width = (r - l) * (z[1] - z[0])
    wlim = window_limit_bins(win, pad)
    print(f"\nMain peak {z[i]*1000:.4f} mm, -3 dB width {width*1e6:.1f} um "
          f"(window limit ~{wlim*dz_bin*1e6:.1f} um)")
    if width > 2 * wlim * dz_bin:
        print("  WARNING: main peak > 2x window limit -- frequency axis "
              "suspicious. Investigate before trusting positions.")
        warnings.append("main_peak_too_wide")

    zmax = a.zmax if a.zmax else z[-1]
    # Over the analysis range only, so this matches the exported CSV --
    # otherwise the quiet tail up to Nyquist drags the floor down.
    nf = noise_floor(db[z <= zmax])
    print(f"\nRMS noise floor {nf:.1f} dB   -> dynamic range {-nf:.1f} dB "
          f"(strongest peak is 0 dB by normalisation)")
    pk, _ = find_peaks(db, height=a.peak_floor_db,
                       distance=max(3, int(200e-6 / (z[1] - z[0]))))
    print(f"\nPeaks above {a.peak_floor_db:.0f} dB:")
    for j in pk:
        if z[j] <= zmax:
            h = z[j] / z[i] if z[i] > 0 else 0
            tag = ("   <- harmonic, not a reflector"
                   if 1.5 < h < 6 and abs(h - round(h)) < 0.03 else "")
            print(f"   {z[j]*1000:10.4f} mm  {db[j]:6.1f} dB{tag}")

    # Self-test against known truth (only for synthetic data)
    comparison = []
    passed = None
    if "truth_z" in meta:
        print("\n--- Comparison with known truth ---")
        worst = 0.0
        any_unaccounted = False
        for zt, dbt in zip(np.atleast_1d(meta["truth_z"]), np.atleast_1d(meta["truth_db"])):
            # A reflector beyond the Nyquist range doesn't vanish -- it
            # aliases (folds back into [0, z_nyq]). Compare against the
            # folded position in that case instead of silently skipping it.
            aliased = zt > z_nyq
            if aliased:
                period = 2 * z_nyq
                zt_expect = zt % period
                if zt_expect > z_nyq:
                    zt_expect = period - zt_expect
            else:
                zt_expect = zt

            w = (z > zt_expect - 20 * dz_bin) & (z < zt_expect + 20 * dz_bin)
            if not w.any():
                any_unaccounted = True
                where = f"fold at {zt_expect:.4f} m" if aliased else "on-axis"
                print(f"   z_true {zt:.4f} m  -- unaccounted for "
                      f"(expected {where}, nothing found there)")
                comparison.append(dict(z_true=zt, db_true=dbt, z_found=None,
                                       db_found=None, err_um=None, err_cells=None,
                                       status="out_of_range_unaccounted"))
                continue
            jj = int(np.argmax(np.where(w, R, 0)))
            err = (z[jj] - zt_expect) * 1e6
            worst = max(worst, abs(err))
            status = "aliased_as_expected" if aliased else "ok"
            tag = f"  ({status}, folded from z_true)" if aliased else ""
            print(f"   z_true {zt:7.4f} m ({dbt:5.1f} dB) -> found "
                  f"{z[jj]:7.4f} m, error {err:+7.1f} um "
                  f"({err/(dz_bin*1e6):+.2f} cells), {db[jj]:6.1f} dB{tag}")
            comparison.append(dict(z_true=zt, db_true=dbt, z_found=z[jj],
                                   db_found=db[jj], err_um=err,
                                   err_cells=err / (dz_bin * 1e6), status=status))
        print(f"   largest error: {worst:.1f} um "
              f"(one cell = {dz_bin*1e6:.1f} um)")
        passed = worst < 2 * dz_bin * 1e6 and not any_unaccounted
        print("   -> PASSED" if passed else "   -> FAILED, check pipeline")

    prefix = a.out or a.scan.rsplit(".", 1)[0]
    keep = z <= zmax
    np.savetxt(f"{prefix}_reflectogram.csv",
               np.column_stack([z[keep], db[keep]]), delimiter=",",
               header="distance_m,amplitude_dB", comments="")
    print(f"\nwrote: {prefix}_reflectogram.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # --zero-m only moves the PLOT axis; the CSV stays absolute.
        z0 = a.zero_m or 0.0
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(z[keep] - z0, db[keep], lw=0.6)
        ax.axhline(nf, color="darkviolet", ls="--", lw=1.1, zorder=4)
        ax.annotate(f"RMS noise floor {nf:.1f} dB  (dynamic range {-nf:.1f} dB)",
                    (z[keep][-1] - z0, nf), fontsize=8.5, ha="right", va="bottom",
                    color="darkviolet",
                    bbox=dict(fc="white", ec="none", alpha=.75, pad=1.0))
        if z0:
            ax.axvline(0.0, color="crimson", lw=0.8, ls=":")
            ax.set_xlabel(f"Distance from {a.zero_label} "
                          f"(m, reflection convention; 0 = {z0:.5f} m absolute)")
            top = ax.secondary_xaxis("top", functions=(lambda x: x + z0,
                                                       lambda x: x - z0))
            top.set_xlabel("absolute distance (m)", fontsize=8)
            top.tick_params(labelsize=7)
        else:
            ax.set_xlabel("Distance (m, reflection convention)")
        ax.set_ylabel("Amplitude (dB rel. maximum)")
        ax.set_title(f"{a.scan} | aux-referenced, {a.window}, "
                     f"dz {dz_bin*1e6:.1f} um, Nyquist {z_nyq:.2f} m", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(max(-110, db[keep].min() - 5), 5)
        fig.tight_layout()
        fig.savefig(f"{prefix}_reflectogram.png", dpi=150)
        plt.close(fig)
        print(f"wrote: {prefix}_reflectogram.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    return dict(z=z, db=db, R=R, dz_bin=dz_bin, z_nyq=z_nyq,
                main_peak_m=z[i], peak_width_um=width * 1e6,
                noise_floor_db=nf, warnings=warnings,
                comparison=comparison, passed=passed)


if __name__ == "__main__":
    main()
