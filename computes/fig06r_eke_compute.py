"""
Quantities of the second row of Figure 6, the mesoscale activity and the arrest.

The meanders and eddies of the Brazil Current reverse the flow at the shelf break
(Calado et al. 2010, Rocha and Simoes-Sousa 2022, Uchoa et al. 2023, da Silveira et
al. 2023), so they could shorten the states of the interior flow below the time the
layer needs to arrest, or change the response of the layer within a state. The daily
eddy kinetic energy over the slope around each 200 m mooring measures that activity
and is set against the states, the response and the supply.

 1  eke          daily eddy kinetic energy of the DUACS surface geostrophic velocity
                 about its mean over the records, averaged over the water between
                 EKE_DEPTHS within EKE_RADIUS_KM of each mooring
 2  states       duration of every state of the flow at 155 m against its mean eddy
                 kinetic energy, Spearman correlation with intervals from resampling
                 states, and the attenuation and veering of the state corrected to the
                 interior speed U_REF as in Figure 4
 3  terciles     complex gains of Figure 3 fitted apart on the states of the lowest
                 and of the highest tercile of eddy kinetic energy, with intervals and
                 the difference from resampling whole states
 4  daily        correlation of the daily eddy kinetic energy with the uplift at 155 m,
                 the position of the Brazil Current axis, the interior speed and the
                 near-bottom flux of the deficit below 20 C carried by the
                 fluctuations, tested against phase randomised surrogates of the eddy
                 kinetic energy inside every campaign

Reads the altimetry, GEBCO, the low-passed records and paper_v3/tables/fig03_response.csv,
fig02_daily.csv and fig05_daily.csv.
Outputs in paper_v3
    tables/fig06r_eke_series.csv    daily eddy kinetic energy at each 200 m mooring
    tables/fig06r_eke_states.csv       every state with its eddy kinetic energy and response
    tables/fig06r_eke_duration.csv     correlation of the duration with the eddy kinetic energy
    tables/fig06r_eke_terciles.csv     gains in the lowest and highest tercile
    tables/fig06r_eke_correlation.csv        daily correlations and their surrogate levels
    logs/fig06r_eke_eke_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P
import fig03_compute as F3

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []

EKE_RADIUS_KM = 100.0        # radius of the slope around each mooring
EKE_DEPTHS = (200.0, 3000.0)  # m, water over which the eddy kinetic energy is averaged
N_SURR = 500
MIN_STATE_HOURS = 24


def say(msg=''):
    print(msg)
    LOG.append(msg)


def slope_eke(alt, gebco, zname, geo):
    """Daily eddy kinetic energy over the slope around every 200 m mooring."""
    sub = alt.sel(time=slice(*V.PERIOD))
    u, v = sub['ugos'].load(), sub['vgos'].load()
    eke = 0.5 * ((u - u.mean('time')) ** 2 + (v - v.mean('time')) ** 2)
    depth = -gebco[zname].interp(lat=sub.lat, lon=sub.lon)
    lat2, lon2 = np.meshgrid(sub.lat.values, sub.lon.values, indexing='ij')
    out = {}
    for site in V.SITES_200:
        if site not in geo.index:
            continue
        dist = I.distance_km(geo.loc[site, 'lat'], geo.loc[site, 'lon'], lat2, lon2)
        mask = (dist <= EKE_RADIUS_KM) & (depth.values >= EKE_DEPTHS[0]) & (depth.values <= EKE_DEPTHS[1])
        if mask.sum() < 3:
            say(f'  {site}: fewer than three altimetric points over the slope, left out')
            continue
        w = np.cos(np.deg2rad(lat2)) * mask
        series = (eke * w).sum(('lat', 'lon')) / w.sum()
        out[site] = pd.Series(series.values, index=pd.DatetimeIndex(series.time.values).normalize())
        say(f'  {site}: {int(mask.sum())} points, median eddy kinetic energy '
            f'{np.nanmedian(series):.4f} m2 s-2, quartiles {np.nanpercentile(series, 25):.4f} to '
            f'{np.nanpercentile(series, 75):.4f}')
    return out


def per_state(site, d, lv, label, idx, eke, offset):
    ki, kb = lv.index(V.PAIRS['bottom'][0]), lv.index(V.PAIRS['bottom'][1])
    wi = d['u'][:, ki] + 1j * d['v'][:, ki]
    wb = d['u'][:, kb] + 1j * d['v'][:, kb] - offset
    speed = np.abs(wi)
    ratio = wb / wi
    power = {}
    for kind, sel in (('with the current', label < 0), ('against the current', label > 0)):
        use = sel & np.isfinite(ratio) & (speed >= V.MIN_SPEED) & (np.abs(ratio) > 0)
        power[kind] = np.polyfit(np.log(speed[use]), np.log(np.abs(ratio[use])), 1)[0] \
            if use.sum() > 100 else 0.0
    day = d['time'].normalize()
    rows = []
    for k in range(idx.max() + 1):
        m = np.flatnonzero(idx == k)
        if m.size < MIN_STATE_HOURS:
            continue
        kind = 'with the current' if label[m[0]] < 0 else 'against the current'
        e = eke.reindex(day[m].unique()).values
        good = m[np.isfinite(ratio[m]) & (speed[m] >= V.MIN_SPEED)]
        g_ref = np.abs(ratio[good]) * (V.U_REF / speed[good]) ** power[kind]
        rows.append(dict(site=site, state=k, kind=kind, start=d['time'][m[0]], hours=m.size,
                         eke=float(np.nanmean(e)) if np.isfinite(e).any() else np.nan,
                         interior_speed=float(np.nanmedian(speed[m])),
                         n_ratio=good.size,
                         attenuation=float(np.median(g_ref)) if good.size >= 12 else np.nan,
                         veer=float(np.degrees(np.angle(np.sum(ratio[good] / np.abs(ratio[good])))))
                         if good.size >= 12 else np.nan,
                         power=power[kind]))
    return pd.DataFrame(rows)


def terciles(site, d, lv, label, idx, st, rng):
    ki, kb = lv.index(V.PAIRS['bottom'][0]), lv.index(V.PAIRS['bottom'][1])
    wi = d['u'][:, ki] + 1j * d['v'][:, ki]
    wb = d['u'][:, kb] + 1j * d['v'][:, kb]
    ok = np.isfinite(wi) & np.isfinite(wb) & (label != 0)
    s = st.dropna(subset=['eke'])
    if len(s) < 30:
        return []
    lo_cut, hi_cut = np.percentile(s.eke, [100.0 / 3.0, 200.0 / 3.0])
    groups = {'lowest tercile': s.state[s.eke <= lo_cut].values,
              'highest tercile': s.state[s.eke >= hi_cut].values}

    def gains(states_list):
        m = np.concatenate([np.flatnonzero((idx == k) & ok) for k in states_list])
        up = label[m] < 0
        if up.sum() < 120 or (~up).sum() < 120:
            return None
        c = F3.fit(wi[m], wb[m], up)[0]
        return np.array([abs(c[1]), abs(c[2]), np.degrees(np.angle(c[1])), np.degrees(np.angle(c[2]))])

    est = {g: gains(v) for g, v in groups.items()}
    if any(e is None for e in est.values()):
        say('    too few hours of one kind in a tercile, no tercile gains')
        return []
    boot = {g: [] for g in groups}
    diff = []
    for _ in range(V.N_BOOT):
        b = {g: gains(rng.choice(v, v.size)) for g, v in groups.items()}
        if any(x is None for x in b.values()):
            continue
        for g in groups:
            boot[g].append(b[g])
        diff.append(b['highest tercile'] - b['lowest tercile'])
    names = ('upslope gain', 'downslope gain', 'upslope veering', 'downslope veering')
    rows = []
    for g in groups:
        bb = np.array(boot[g])
        for j, name in enumerate(names):
            rows.append(dict(site=site, group=g, quantity=name, value=float(est[g][j]),
                             lo=float(np.percentile(bb[:, j], 2.5)), hi=float(np.percentile(bb[:, j], 97.5)),
                             eke_cut=float(lo_cut if g == 'lowest tercile' else hi_cut),
                             n_states=int(groups[g].size)))
    dd = np.array(diff)
    for j, name in enumerate(names):
        rows.append(dict(site=site, group='highest minus lowest', quantity=name,
                         value=float(est['highest tercile'][j] - est['lowest tercile'][j]),
                         lo=float(np.percentile(dd[:, j], 2.5)), hi=float(np.percentile(dd[:, j], 97.5)),
                         eke_cut=np.nan, n_states=np.nan))
    for r in rows:
        say(f'    {r["group"]}, {r["quantity"]}: {r["value"]:+.2f} ({r["lo"]:+.2f} to {r["hi"]:+.2f})')
    return rows


def segments(days, camp):
    """Runs of consecutive days inside one campaign."""
    days = pd.DatetimeIndex(days)
    brk = np.r_[True, (np.diff(days.values).astype('timedelta64[D]').astype(int) != 1)
                | (camp[1:] != camp[:-1])]
    return np.cumsum(brk)


def daily_correlation(x, y, seg, rng):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    null = []
    for _ in range(N_SURR):
        xs = np.full(x.size, np.nan)
        for s_ in np.unique(seg):
            m = np.flatnonzero((seg == s_) & np.isfinite(x))
            if m.size >= 8:
                xs[m] = P.surrogate(x[m], rng) + np.nanmean(x[m])
        o = np.isfinite(xs) & np.isfinite(y)
        if o.sum() >= 30:
            null.append(abs(np.corrcoef(xs[o], y[o])[0, 1]))
    return float(r), float(np.percentile(null, 95)), int(ok.sum())


def daily_flux(site, d, lv):
    """Daily near-bottom flux of the deficit below T_c carried by the fluctuations."""
    kb = lv.index(V.BOTTOM_LEVEL[site])
    theta = np.minimum(d['T'][:, kb] - 20.0, 0.0)
    df = pd.DataFrame({'v': d['v'][:, kb], 'theta': theta, 'campaign': d['campaign']}, index=d['time'])
    df = df[df.campaign != '']
    for c in ('v', 'theta'):
        df[c] = df[c] - df.groupby('campaign')[c].transform('mean')
    df['vt'] = -df['v'] * df['theta']
    day = df.groupby([df.campaign, df.index.floor('D')])['vt'].mean()
    return pd.Series(day.values, index=day.index.get_level_values(1))


def main():
    V.make_dirs()
    rng = np.random.default_rng(1202)
    say('figure 6, second row, the mesoscale activity and the arrest')
    say(f'  eddy kinetic energy over {EKE_DEPTHS[0]:.0f} to {EKE_DEPTHS[1]:.0f} m within '
        f'{EKE_RADIUS_KM:.0f} km of each mooring')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    gebco, zname = I.open_gebco()
    eke = slope_eke(I.open_altimetry(), gebco, zname, geo)
    pd.concat({s: e for s, e in eke.items()}, names=['site', 'day']).rename('eke').reset_index() \
        .to_csv(V.DIR_TAB / 'fig06r_eke_series.csv', index=False)
    resp = pd.read_csv(V.DIR_TAB / 'fig03_response.csv')
    f5 = pd.read_csv(V.DIR_TAB / 'fig05_daily.csv', parse_dates=['day'])
    f2 = pd.read_csv(V.DIR_TAB / 'fig02_daily.csv', parse_dates=['day'])
    all_states, durations, terc, daily_rows = [], [], [], []

    for site in V.SITES_200:
        if site not in lp or site not in eke:
            continue
        d = I.load_lp(lp[site])
        lv = list(d['level'])
        if V.CONDITION_LEVEL not in lv or not all(z in lv for z in V.PAIRS['bottom']):
            continue
        say(f'\n  {site}')
        label, idx = F3.states_of(d, lv)
        r = resp[(resp.site == site) & (resp.lower == V.PAIRS['bottom'][1]) & (resp.half == 'all')]
        offset = complex(r.offset_u.iloc[0], r.offset_v.iloc[0]) if len(r) else 0j
        st = per_state(site, d, lv, label, idx, eke[site], offset)
        all_states.append(st)

        # ----------------------------------------------------- duration and eke
        s = st.dropna(subset=['eke'])
        for kind in ('all', 'with the current', 'against the current'):
            g = s if kind == 'all' else s[s.kind == kind]
            if len(g) < 15:
                continue
            rho = spearmanr(g.eke, g.hours).correlation
            boot = []
            for _ in range(V.N_BOOT):
                j = rng.integers(0, len(g), len(g))
                boot.append(spearmanr(g.eke.values[j], g.hours.values[j]).correlation)
            lo_cut, hi_cut = np.percentile(g.eke, [100.0 / 3.0, 200.0 / 3.0])
            dl = g.hours[g.eke <= lo_cut].median() / 24.0
            dh = g.hours[g.eke >= hi_cut].median() / 24.0
            durations.append(dict(site=site, kind=kind, n_states=len(g), rho=float(rho),
                                  rho_lo=float(np.nanpercentile(boot, 2.5)),
                                  rho_hi=float(np.nanpercentile(boot, 97.5)),
                                  days_low_eke=float(dl), days_high_eke=float(dh)))
            say(f'    {kind}: duration against eddy kinetic energy, Spearman {rho:+.2f} '
                f'({np.nanpercentile(boot, 2.5):+.2f} to {np.nanpercentile(boot, 97.5):+.2f}) over '
                f'{len(g)} states, median {dl:.1f} days in the lowest and {dh:.1f} days in the '
                f'highest tercile')
        for kind in ('with the current', 'against the current'):
            g = s[(s.kind == kind)].dropna(subset=['attenuation'])
            if len(g) >= 15:
                rho = spearmanr(g.eke, g.attenuation).correlation
                say(f'    {kind}: attenuation of the state against eddy kinetic energy, Spearman '
                    f'{rho:+.2f} over {len(g)} states')
                durations.append(dict(site=site, kind=f'attenuation {kind}', n_states=len(g),
                                      rho=float(rho), rho_lo=np.nan, rho_hi=np.nan,
                                      days_low_eke=np.nan, days_high_eke=np.nan))

        # ------------------------------------------------------------ terciles
        terc += terciles(site, d, lv, label, idx, st, rng)

        # --------------------------------------------------------------- daily
        g5 = f5[f5.site == site].set_index('day')
        g2 = f2[f2.site == site].drop_duplicates('day').set_index('day')
        flux = daily_flux(site, d, lv).groupby(level=0).mean()
        days = g5.index
        base = pd.DataFrame({'eke': eke[site].reindex(days).values,
                             'uplift at 155 m': g5.eta.values,
                             'position of the axis': g5.axis.values,
                             'interior speed': np.abs(g2.u_i.reindex(days).values),
                             'flux by the fluctuations': flux.reindex(days).values,
                             'campaign': g5.campaign.astype(str).values}, index=days)
        seg = segments(base.index, base.campaign.values)
        for name in ('uplift at 155 m', 'position of the axis', 'interior speed', 'flux by the fluctuations'):
            rr, lev, n = daily_correlation(base.eke.values, base[name].values, seg, rng)
            daily_rows.append(dict(site=site, variable=name, r=rr, level=lev, n_days=n))
            say(f'    daily eddy kinetic energy with the {name}: r {rr:+.2f} against a 95 % level of '
                f'{lev:.2f} over {n} days')

    pd.concat(all_states, ignore_index=True).to_csv(V.DIR_TAB / 'fig06r_eke_states.csv', index=False)
    pd.DataFrame(durations).to_csv(V.DIR_TAB / 'fig06r_eke_duration.csv', index=False)
    pd.DataFrame(terc).to_csv(V.DIR_TAB / 'fig06r_eke_terciles.csv', index=False)
    pd.DataFrame(daily_rows).to_csv(V.DIR_TAB / 'fig06r_eke_correlation.csv', index=False)
    (V.DIR_LOG / 'fig06r_eke_eke_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()