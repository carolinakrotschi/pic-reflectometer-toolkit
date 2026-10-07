#!/usr/bin/env python3
"""
Sweep parameter calculator for free-running OFDR acquisition.

Answers: "if I have N points per channel, at what speed do I need to
sweep, what span do I get, what range, what resolution -- and does my
aux MZI still fit within that?"

Two of {N, sweep speed, span} determine everything, because the DAQ
clock step is fixed (default 1 us):

    T        = N * step
    Span     = v * T
    dlambda  = v * step
    dnu      = c * dlambda / lam^2
    z_max    = c / (4 * n_g * dnu)                <- range (Nyquist)
    dz       = c / (2 * n_g * dnu_span)           <- spatial resolution
    z_max/dz = N/2                                <- always, no exception

Aux MZI check (this one is easy to forget):
  The aux MZI is itself a signal on the detector. It appears at
  z_aux = dL/2 and must therefore lie WITHIN the Nyquist range.
  At the same time, its delay should be at least as large as that of
  the farthest reflector you want to measure:

      2 * z_mess  <=  dL  <  2 * z_max

Examples:
    python -m ofdr.sweep_calculator --points 1000000 --span 60 --dl 4
    python -m ofdr.sweep_calculator --points 1000000 --speed 120 --dl 4
    python -m ofdr.sweep_calculator --points 250000 --span 120 --dl 4
    python -m ofdr.sweep_calculator --table
"""

import argparse

C = 299_792_458.0
NG = 1.468
LAM = 1565e-9          # center of the sweep, used for the conversions
STEP_US = 1.0          # DAQ clock step


def compute(points, speed_nms=None, span_nm=None, step_us=STEP_US):
    """Two of (speed, span) -- one may be None."""
    T = points * step_us * 1e-6                       # s
    if speed_nms is None and span_nm is None:
        raise SystemExit("specify --speed or --span")
    if speed_nms is None:
        speed_nms = span_nm / T
    if span_nm is None:
        span_nm = speed_nms * T

    dlam_pm = speed_nms * step_us * 1e-6 * 1e3        # nm/s * s -> nm -> pm
    dnu = C * dlam_pm * 1e-12 / LAM**2                # Hz
    z_max = C / (4 * NG * dnu)                        # m
    dnu_span = C * span_nm * 1e-9 / LAM**2            # Hz
    dz = C / (2 * NG * dnu_span)                      # m
    dnudt = C / LAM**2 * speed_nms * 1e-9             # Hz/s
    f_nyq = 1.0 / (2 * step_us * 1e-6)                # Hz
    return dict(points=points, T=T, speed=speed_nms, span=span_nm,
                dlam_pm=dlam_pm, dnu=dnu, z_max=z_max, dz=dz,
                cells=z_max / dz, dnudt=dnudt, f_nyq=f_nyq)


def aux_check(r, dl_m, z_mess_m, step_us=STEP_US):
    tau_aux = NG * dl_m / C
    z_aux = dl_m / 2
    f_aux = tau_aux * r["dnudt"]
    pts_per_fringe = (1.0 / (step_us * 1e-6)) / f_aux
    tau_mess = 2 * NG * z_mess_m / C
    return dict(tau_aux=tau_aux, z_aux=z_aux, f_aux=f_aux,
                pts_per_fringe=pts_per_fringe, tau_mess=tau_mess,
                dl_min=2 * z_mess_m, dl_max=2 * r["z_max"])


def report(r, dl_m=None, z_mess_m=2.0, step_us=STEP_US):
    print(f"Points/channel      {r['points']:,}")
    print(f"Clock step          {step_us} us  ->  acquisition time {r['T']:.3f} s")
    print(f"Sweep               {r['speed']:.1f} nm/s over {r['span']:.1f} nm")
    print(f"Point spacing       {r['dlam_pm']:.4f} pm  =  {r['dnu']/1e6:.2f} MHz")
    print()
    print(f"  RANGE       z_max = {r['z_max']:.3f} m")
    print(f"  RESOLUTION  dz    = {r['dz']*1e6:.2f} um")
    print(f"  Cells             = {r['cells']:,.0f}   (= N/2 = {r['points']//2:,})")
    print()
    print(f"Time-domain sampling Nyquist: {r['f_nyq']/1e3:.0f} kHz")
    print(f"  Beat frequency of a reflector at z_max: "
          f"{2*NG*r['z_max']/C*r['dnudt']/1e3:.0f} kHz  (must equal Nyquist)")

    if dl_m is not None:
        a = aux_check(r, dl_m, z_mess_m, step_us)
        print()
        print(f"--- Aux MZI with dL = {dl_m:.2f} m ---")
        print(f"tau_aux             {a['tau_aux']*1e9:.2f} ns")
        print(f"appears at          z = {a['z_aux']:.3f} m "
              f"(must be < z_max = {r['z_max']:.3f} m)")
        print(f"Beat frequency      {a['f_aux']/1e3:.1f} kHz  "
              f"= {a['pts_per_fringe']:.1f} points per fringe")
        print(f"Target range for dL at z_mess = {z_mess_m:.1f} m: "
              f"{a['dl_min']:.2f} m ... {a['dl_max']:.2f} m")
        ok = True
        if a["z_aux"] >= r["z_max"]:
            print("  ERROR: Aux lies beyond Nyquist -- it aliases itself. "
                  "Make dL smaller or sweep slower.")
            ok = False
        if a["pts_per_fringe"] < 2:
            print("  ERROR: fewer than 2 points per aux fringe -- phase not "
                  "reconstructible.")
            ok = False
        elif a["pts_per_fringe"] < 4:
            print("  WARNING: fewer than 4 points per aux fringe -- unwrapping "
                  "becomes sensitive. Sweep slower or reduce dL.")
        if dl_m < a["dl_min"]:
            print(f"  WARNING: dL smaller than 2*z_mess ({a['dl_min']:.2f} m) -- "
                  "noise correction for distant peaks will be worse.")
        if ok and a["pts_per_fringe"] >= 4 and dl_m >= a["dl_min"]:
            print("  -> all conditions satisfied.")


def default_table():
    rows = [(1_000_000, 120, None), (1_000_000, 60, None),
            (1_000_000, 30, None), (500_000, 120, None),
            (250_000, None, 120), (250_000, 120, None), (250_000, 60, None)]
    print(f"{'N/channel':>10} {'v(nm/s)':>8} {'T(s)':>6} {'Span(nm)':>9} "
          f"{'dlam(pm)':>9} {'z_max(m)':>9} {'dz(um)':>8} {'f_aux@dL=4m':>12}")
    for n, v, s in rows:
        r = compute(n, v, s)
        a = aux_check(r, 4.0, 2.0)
        if a["z_aux"] >= r["z_max"] or a["pts_per_fringe"] < 2:
            flag = "  <-- Aux does NOT fit"
        elif a["pts_per_fringe"] < 4:
            flag = "  <-- Aux marginal"
        else:
            flag = ""
        print(f"{n:>10,} {r['speed']:>8.0f} {r['T']:>6.2f} {r['span']:>9.1f} "
              f"{r['dlam_pm']:>9.3f} {r['z_max']:>9.2f} {r['dz']*1e6:>8.1f} "
              f"{a['f_aux']/1e3:>9.0f} kHz{flag}")
    print("\n(f_aux = beat frequency of an aux MZI with a 4 m arm difference;")
    print(" sampling Nyquist is 500 kHz, < 250 kHz is comfortable.)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=int, default=1_000_000,
                   help="points per channel in the buffer")
    p.add_argument("--speed", type=float, default=None, help="nm/s")
    p.add_argument("--span", type=float, default=None, help="nm")
    p.add_argument("--dl", type=float, default=None,
                   help="arm difference of the aux MZI in m")
    p.add_argument("--z-mess", type=float, default=2.0,
                   help="farthest reflector you want to measure (m)")
    p.add_argument("--step-us", type=float, default=STEP_US)
    p.add_argument("--table", action="store_true", help="overview table")
    a = p.parse_args()
    if a.table:
        default_table()
        return
    if a.speed is None and a.span is None:
        a.speed = 60.0
    r = compute(a.points, a.speed, a.span, a.step_us)
    report(r, a.dl, a.z_mess, a.step_us)


if __name__ == "__main__":
    main()
