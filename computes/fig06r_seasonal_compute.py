"""
Quantities of the first row of Figure 6, the arrest by half of the year.

The regime of the shelf break barely follows the calendar at 200 m (Figure 2), but
the stratification of the shelf does (Castro 2014, Cerda and Castro 2014), so every
diagnostic of the arrest is repeated for October to March and April to September,
the halves that group spring with summer and autumn with winter.

 1  response     complex gain of the lower on the upper level of each pair
                 (Equation of Figure 3) fitted apart in each half, with intervals
                 from resampling whole states, and the difference between the
                 halves of the upslope gain, of the downslope gain and of the
                 asymmetry, from the same resampled states in both halves
 2  buoyancy     daily buoyancy frequency of the layer between 155 and 194 m by
                 state and half, and the share of unstable or mixed days, with
                 intervals from resampling blocks of five days
 3  times        canonical arrest time, wedge time and duration of the states in
                 each half, and the ratio of the duration to the wedge time

Reads paper_v3/tables/fig02_daily.csv, fig02_states.csv and the low-passed records.
Outputs in paper_v3
    tables/fig06r_seasonal_response.csv     gains and veering by site, pair, state and half
    tables/fig06r_seasonal_difference.csv   October to March minus April to September
    tables/fig06r_seasonal_buoyancy.csv     stratification by state and half
    tables/fig06r_seasonal_times.csv        time scales by half
    logs/fig06r_seasonal_seasonal_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd

import config_v3 as V
import inputs_v3 as I
import fig03_compute as F3

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []


def say(msg=''):
    print(msg)
    LOG.append(msg)


def half_difference(site, pair, d, lv, label, idx, rng):
    """October to March minus April to September for the two gains and their asymmetry,
    each resampling of whole states serving both halves."""
    ku, kl = lv.index(pair[0]), lv.index(pair[1])
    wi = d['u'][:, ku] + 1j * d['v'][:, ku]
    wb = d['u'][:, kl] + 1j * d['v'][:, kl]
    half = I.halves_of(d['time'])
    ok = np.isfinite(wi) & np.isfinite(wb) & (label != 0)
    need = [ok & (half == h) & (label < 0) for h in V.HALVES] + \
           [ok & (half == h) & (label > 0) for h in V.HALVES]
    if min(int(m.sum()) for m in need) < 240:
        return None
    sts = np.unique(idx[ok])
    rows_of = {k: np.flatnonzero((idx == k) & ok) for k in sts}

    def both(j):
        out = {}
        for h in V.HALVES:
            m = j[half[j] == h]
            up = label[m] < 0
            if up.sum() < 24 or (~up).sum() < 24:
                return None
            c = F3.fit(wi[m], wb[m], up)[0]
            out[h] = (abs(c[1]), abs(c[2]), np.degrees(np.angle(c[1])), np.degrees(np.angle(c[2])))
        a, b = out[V.HALVES[0]], out[V.HALVES[1]]
        return np.array([a[0] - b[0], a[1] - b[1], (a[0] - a[1]) - (b[0] - b[1]),
                         a[2] - b[2], a[3] - b[3]])

    est = both(np.flatnonzero(ok))
    boot = []
    for _ in range(V.N_BOOT):
        j = np.concatenate([rows_of[k] for k in rng.choice(sts, sts.size)])
        r = both(j)
        if r is not None:
            boot.append(r)
    boot = np.array(boot)
    names = ('upslope gain', 'downslope gain', 'asymmetry of the gain', 'upslope veering',
             'downslope veering')
    rows = []
    for k, name in enumerate(names):
        rows.append(dict(site=site, pair=f'{pair[0]} to {pair[1]} m', quantity=name,
                         difference=float(est[k]), lo=float(np.percentile(boot[:, k], 2.5)),
                         hi=float(np.percentile(boot[:, k], 97.5))))
    return rows


def block_boot(values, days, rng, stat, n=V.N_BOOT, length=5):
    days = pd.DatetimeIndex(days)
    block = ((days - days.min()).days // length).values
    ub = np.unique(block)
    rows_of = {b: np.flatnonzero(block == b) for b in ub}
    out = []
    for _ in range(n):
        j = np.concatenate([rows_of[b] for b in rng.choice(ub, ub.size)])
        out.append(stat(values[j]))
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    V.make_dirs()
    rng = np.random.default_rng(1101)
    say('figure 6, first row, the arrest by half of the year')
    say('  halves ' + ' and '.join(f'{h} = {V.HALF_SHORT[h]}' for h in V.HALVES))
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    daily = pd.read_csv(V.DIR_TAB / 'fig02_daily.csv', parse_dates=['day'])
    states = pd.read_csv(V.DIR_TAB / 'fig02_states.csv', parse_dates=['start', 'end'])
    resp, diff, buoy, times = [], [], [], []

    for site in V.SITES_200:
        if site not in lp:
            continue
        d = I.load_lp(lp[site])
        lv = list(d['level'])
        if V.CONDITION_LEVEL not in lv:
            continue
        say(f'\n  {site}')
        label, idx = F3.states_of(d, lv)

        # ------------------------------------------------------------ 1 response
        for name, pair in V.PAIRS.items():
            if not (pair[0] in lv and pair[1] in lv):
                continue
            for half in (None,) + V.HALVES:
                out = F3.response(site, pair, d, lv, label, idx, rng, geo, half)
                if out is None:
                    say(f'    {name} pair, {V.HALF_SHORT.get(half, "all year")}: too few hours')
                    continue
                resp += out[0]
                for r in out[0]:
                    say(f'    {name} pair, {V.HALF_SHORT.get(half, "all year")}, {r["state"]}: gain '
                        f'{r["gain"]:.2f} ({r["gain_lo"]:.2f} to {r["gain_hi"]:.2f}), veering '
                        f'{r["veer"]:+.1f} ({r["veer_lo"]:+.1f} to {r["veer_hi"]:+.1f}) deg, '
                        f'{r["n_hours"]} hours in {r["n_states"]} states')
                a = out[1]
                say(f'    {name} pair, {V.HALF_SHORT.get(half, "all year")}, with minus against: gain '
                    f'{a["gain_difference"]:+.2f} ({a["gain_difference_lo"]:+.2f} to '
                    f'{a["gain_difference_hi"]:+.2f}), veering {a["veer_difference"]:+.1f} '
                    f'({a["veer_difference_lo"]:+.1f} to {a["veer_difference_hi"]:+.1f}) deg')
            rows = half_difference(site, pair, d, lv, label, idx, rng)
            if rows is None:
                say(f'    {name} pair: one half lacks a kind of state, no difference between halves')
                continue
            diff += rows
            for r in rows:
                say(f'    {name} pair, October to March minus April to September, {r["quantity"]}: '
                    f'{r["difference"]:+.2f} ({r["lo"]:+.2f} to {r["hi"]:+.2f})')

        # ------------------------------------------------------------ 2 buoyancy
        g = daily[daily.site == site].copy()
        g = g[g.layer == 'lowest pair'] if (g.layer == 'lowest pair').sum() >= 30 else g
        g['kind'] = np.where(g.u_i <= -V.MIN_SPEED, 'with the current',
                             np.where(g.u_i >= V.MIN_SPEED, 'against the current', ''))
        measured = np.isfinite(g.T_top.values) & np.isfinite(g.T_bot.values)
        g = g[measured]
        for half in V.HALVES:
            for kind in ('with the current', 'against the current'):
                h = g[(g.half == half) & (g.kind == kind)]
                if len(h) < 10:
                    continue
                N = h.N.values
                unstable = (~np.isfinite(N)).astype(float)
                q = np.nanpercentile(N, [5, 25, 50, 75, 95])
                lo, hi = block_boot(unstable, h.day.values, rng, lambda x: 100.0 * np.mean(x))
                mlo, mhi = block_boot(N, h.day.values, rng, np.nanmedian)
                buoy.append(dict(site=site, half=half, kind=kind, n_days=len(h), N_p5=q[0], N_q1=q[1],
                                 N_median=q[2], N_median_lo=mlo, N_median_hi=mhi, N_q3=q[3],
                                 N_p95=q[4], unstable=100.0 * unstable.mean(), unstable_lo=lo,
                                 unstable_hi=hi))
                say(f'    {V.HALF_SHORT[half]}, {kind}: median N {q[2]:.2e} s-1 ({mlo:.2e} to '
                    f'{mhi:.2e}), unstable or mixed on {100 * unstable.mean():.0f} % of {len(h)} days '
                    f'({lo:.0f} to {hi:.0f})')

        # --------------------------------------------------------------- 3 times
        st = states[states.site == site].copy()
        mid = st.start + (st.end - st.start) / 2
        st['half'] = I.halves_of(mid)
        for half in V.HALVES:
            gd = daily[(daily.site == site) & (daily.half == half)]
            sh = st[st.half == half]
            if len(sh) < 5 or len(gd) < 10:
                continue
            dur = sh.hours.values / 24.0
            tc, tw = np.nanmedian(gd.tau_c_h), np.nanmedian(gd.tau_w_h)
            times.append(dict(site=site, half=half, n_states=len(sh),
                              s_median=float(np.nanmedian(gd.s)),
                              s_q1=float(np.nanpercentile(gd.s, 25)), s_q3=float(np.nanpercentile(gd.s, 75)),
                              tau_c_h=tc, tau_c_q1=float(np.nanpercentile(gd.tau_c_h, 25)),
                              tau_c_q3=float(np.nanpercentile(gd.tau_c_h, 75)),
                              tau_w_h=tw, tau_w_q1=float(np.nanpercentile(gd.tau_w_h, 25)),
                              tau_w_q3=float(np.nanpercentile(gd.tau_w_h, 75)),
                              state_days=float(np.median(dur)), state_q1=float(np.percentile(dur, 25)),
                              state_q3=float(np.percentile(dur, 75)),
                              state_over_wedge=float(24.0 * np.median(dur) / tw)))
            say(f'    {V.HALF_SHORT[half]}: s {np.nanmedian(gd.s):.2f}, canonical {tc:.1f} h, wedge '
                f'{tw:.1f} h, states {np.median(dur):.1f} days over {len(sh)} states, '
                f'{24.0 * np.median(dur) / tw:.1f} wedge times')

    pd.DataFrame(resp).to_csv(V.DIR_TAB / 'fig06r_seasonal_response.csv', index=False)
    pd.DataFrame(diff).to_csv(V.DIR_TAB / 'fig06r_seasonal_difference.csv', index=False)
    pd.DataFrame(buoy).to_csv(V.DIR_TAB / 'fig06r_seasonal_buoyancy.csv', index=False)
    pd.DataFrame(times).to_csv(V.DIR_TAB / 'fig06r_seasonal_times.csv', index=False)
    (V.DIR_LOG / 'fig06r_seasonal_seasonal_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()