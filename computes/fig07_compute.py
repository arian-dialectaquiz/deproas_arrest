"""
Quantities of Figure 6, the path of the central water from the shelf break to the
coast and the transport that supplies it.

Figure 5 found that a coastal trapped wave lifts the thermocline at the 200 m
isobath and that the layer adds a variability of its own. An uplift moves the
isotherms without supplying water, so two questions remain. Does the cold water
that arrives at 200 m reach the mid shelf, the inner shelf and the coast? And
how much central water does the bottom layer carry across the isobaths, and in
what form? The two processes predict different timings across a line. A coastal
trapped wave lifts the whole section almost at once, the moorings of a line being
a few kilometres apart along the isobaths, whereas advection by the layer needs
the cross-shelf distance over the bottom velocity (Lentz and Chapman 2004, Lentz
2001, Austin and Lentz 2002).

 1  events       drops of the near-bottom temperature at each 200 m site larger
                 than EVENT_DROP in EVENT_WINDOW_H, separated by EVENT_SEP_H, and
                 the composite of the near-bottom temperature at the three
                 isobaths of the line, of the coastal upwelling index and of the
                 coastal trapped wave of Figure 5 around their onsets, each
                 referred to its mean over the days BASELINE_DAYS, with 95 %
                 intervals from resampling the events, and the share of the
                 events after which each inner site cools by REACH_DROP within
                 REACH_DAYS
 2  index        the coastal upwelling index of each line, the OISST v2.1 sea
                 surface temperature (Huang et al. 2021) at the pixel nearest to
                 the 50 m site minus that nearest to the 200 m site
 3  lags         lag of the largest correlation between the near-bottom
                 temperature at 200 m and at each inner site, and with the
                 coastal index, with intervals from resampling blocks of five days
                 and the 95 % level of phase randomised surrogates (Ebisuzaki
                 1997), set against the cross-shelf distance
 4  chain        standardised composites around the same onsets of the coastal
                 trapped wave, of the uplift of the isotherms at 155 m of Figure 5
                 and of the cross-isobath velocity at the lowest meter of the 200 m
                 mooring, which follow the event from the forcing to the heave and
                 to the flow near the bottom
 5  flux         near-bottom flux of central water across the isobaths at every
                 mooring from its lowest meter, split inside every campaign into
                 the term of the mean flow and the term of the correlated
                 fluctuations (the events),

                     -<v_b theta> = -<v_b><theta> - <v_b' theta'>,   theta = min(T_b - T_c, 0),

                 theta being the cold deficit below the upper temperature of the
                 central water, zero in warmer water, over the rotations of the frame of Figure 1 and with moving block
                 intervals, a site entering the figure with MIN_FLUX_DAYS of record
 6  transport    transport of central water across the isobaths in the bottom
                 layer of the moorings that hold two levels, kept as a table, between the bottom and the interior meter
                 with v falling linearly to zero at the bottom (Lentz 2001,
                 Kirincich et al. 2005),

                     M = z_b v_b / 2 + (z_i - z_b)(v_b + v_i) / 2,   Q = M 1(T_b < T_c),

                 with T_c the upper temperature of the central water, split into
                 the term of the mean flow and the term of the correlated
                 fluctuations, Q_mean = <M><1> and Q_fluct = <M'1'>, over the
                 rotations of the frame of Figure 1 and with moving block
                 intervals

Outputs in paper_v3
    tables/fig06_events.csv       one row per event
    tables/fig06_composite.csv    composites around the onsets
    tables/fig06_reach.csv        share of the events that reach each inner site
    tables/fig06_lags.csv         lags and correlations against distance
    tables/fig06_flux.csv         near-bottom flux of central water by site, term and rotation
    tables/fig06_transport.csv    transport of the layer by site and rotation, a table only
    tables/fig06_index.csv        daily coastal index of each line
    logs/fig06_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []


def say(msg=''):
    print(msg)
    LOG.append(msg)


def detrended_anomaly(s, camp):
    """Departure from a linear trend inside each campaign, which leaves the synoptic and
    mesoscale variability and removes the drift of a deployment."""
    out = np.full(len(s), np.nan)
    v = np.asarray(s, float)
    t = np.arange(len(s), dtype=float)
    for c in np.unique(camp[camp != '']):
        k = np.flatnonzero((camp == c) & np.isfinite(v))
        if k.size > 10:
            b = np.polyfit(t[k], v[k], 1)
            out[k] = v[k] - np.polyval(b, t[k])
    return out


# ------------------------------------------------------------- the records
def hourly_bottom(site, lp):
    d = I.load_lp(lp[site])
    lv = list(d['level'])
    kb = lv.index(V.BOTTOM_LEVEL[site]) if V.BOTTOM_LEVEL[site] in lv else None
    ki = lv.index(V.INTERIOR_LEVEL[site]) if V.INTERIOR_LEVEL[site] in lv else None
    return d, kb, ki


def daily_bottom(site, lp):
    d, kb, _ = hourly_bottom(site, lp)
    if kb is None:
        return pd.DataFrame()
    df = pd.DataFrame({'T': d['T'][:, kb], 'campaign': d['campaign']}, index=d['time'])
    df = df[df.campaign != '']
    g = df.groupby([df.campaign, df.index.floor('D')])
    day = g.mean(numeric_only=True).where(g.count() >= V.MIN_DAY_HOURS)
    day = day.reset_index(names=['campaign', 'day']).set_index('day')
    day['anom'] = detrended_anomaly(day['T'].values, day.campaign.values)
    return day


# ---------------------------------------------------------- 2 coastal index
def coastal_index(geo):
    files = sorted(V.SST_DIR.glob('oisst_sbb_*.nc'))
    if not files:
        say('  no OISST files, the coastal index is left out')
        return {}
    ds = xr.concat([I.standardise(xr.open_dataset(f)) for f in files], dim='time')
    ds = ds.sortby('time')
    name = next((n for n in ('sst', 'analysed_sst', 'SST') if n in ds.data_vars), list(ds.data_vars)[0])
    sst = ds[name]
    for extra in [d_ for d_ in sst.dims if d_ not in ('time', 'lat', 'lon')]:
        sst = sst.isel({extra: 0})
    sst = sst.sel(time=slice(*V.PERIOD)).load()
    valid = sst.notnull().mean('time').values > 0.8
    LO, LA = np.meshgrid(sst.lon.values, sst.lat.values)

    def pixel(lat, lon):
        dist = np.where(valid, P.distance_km(lat, lon, LA, LO), np.inf)
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        return j, i, float(dist[j, i])

    out = {}
    for line, members in V.LINES.items():
        inner, outer = members[0], members[-1]
        if inner not in geo.index or outer not in geo.index:
            continue
        jc, ic, dc = pixel(geo.loc[inner, 'lat'], geo.loc[inner, 'lon'])
        jo, io, do = pixel(geo.loc[outer, 'lat'], geo.loc[outer, 'lon'])
        s = (sst.isel(lat=jc, lon=ic) - sst.isel(lat=jo, lon=io)).to_pandas()
        s.index = pd.DatetimeIndex(s.index).normalize()
        out[line] = s
        say(f'  {line}: coastal index from the pixel {dc:.0f} km from {inner} minus the pixel {do:.0f} km '
            f'from {outer}, median {s.median():+.2f} degrees, below -1 degree on '
            f'{100 * np.mean(s < -1):.0f} % of the days')
    return out


# ---------------------------------------------------------------- 1 events
def onsets(site, lp, drop):
    d, kb, _ = hourly_bottom(site, lp)
    if kb is None:
        return [], None
    T = d['T'][:, kb]
    camp = d['campaign']
    n = int(V.EVENT_WINDOW_H / V.DT_H)
    dT = np.full(T.size, np.nan)
    dT[:-n] = T[n:] - T[:-n]
    same = np.r_[camp[n:] == camp[:-n], np.zeros(n, bool)] & (camp != '')
    cand = np.flatnonzero((dT <= -drop) & same)
    out, last = [], -10 ** 9
    for i in cand:
        if (i - last) * V.DT_H >= V.EVENT_SEP_H:
            out.append(i)
            last = i
    return [(d['time'][i], camp[i], float(dT[i])) for i in out], d


def composite(series, days, rng):
    """Rows of (lag, mean, lo, hi, n) for a daily series around the given onset days."""
    lags = np.arange(V.COMPOSITE_DAYS[0], V.COMPOSITE_DAYS[1] + 1)
    M = []
    for d0 in days:
        idx = d0 + pd.to_timedelta(lags, unit='D')
        v = series.reindex(idx).values.astype(float)
        base = series.reindex(d0 + pd.to_timedelta(np.arange(V.BASELINE_DAYS[0], V.BASELINE_DAYS[1] + 1),
                                                   unit='D')).values.astype(float)
        if np.isfinite(base).sum() == 0 or np.isfinite(v).mean() < 0.6:
            continue
        M.append(v - np.nanmean(base))
    if len(M) < 5:
        return [], 0
    M = np.array(M)
    boot = np.array([np.nanmean(M[rng.integers(0, len(M), len(M))], axis=0) for _ in range(V.N_BOOT)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return [dict(lag=int(L), mean=float(np.nanmean(M[:, k])), lo=float(lo[k]), hi=float(hi[k]),
                 n=int(np.isfinite(M[:, k]).sum())) for k, L in enumerate(lags)], len(M)


# ------------------------------------------------------------------ 3 lags
def lagged(a, b, camp, rng):
    """Lag of the largest correlation of b on a, positive when b follows a. The three series are
    daily and indexed by day, every pair (a_t, b_t+L) is formed in real time and inside one
    campaign, the interval of the lag comes from resampling the base days t in blocks of five
    days inside each campaign, the same resampled days serving every lag so that the lagged
    structure is kept, and the level from phase randomised copies of b."""
    lags = np.arange(V.LAG_DAYS[0], V.LAG_DAYS[1] + 1)
    idx = a.index
    av = a.values.astype(float)
    cv = camp.reindex(idx).values

    def pairs(bser):
        out = []
        for L in lags:
            t2 = idx + pd.Timedelta(days=int(L))
            y = bser.reindex(t2).values.astype(float)
            same = camp.reindex(t2).values == cv
            out.append((y, np.isfinite(av) & np.isfinite(y) & same))
        return out

    def corr(pr, w):
        rs = []
        for y, ok in pr:
            m = ok & (w > 0)
            if w[m].sum() < 30:
                rs.append(np.nan)
                continue
            ww, x_, y_ = w[m], av[m], y[m]
            mx, my = np.average(x_, weights=ww), np.average(y_, weights=ww)
            cxy = np.average((x_ - mx) * (y_ - my), weights=ww)
            rs.append(cxy / np.sqrt(np.average((x_ - mx) ** 2, weights=ww) * np.average((y_ - my) ** 2, weights=ww)))
        return np.array(rs)

    pr = pairs(b)
    ones = np.ones(idx.size)
    r = corr(pr, ones)
    if not np.isfinite(r).any() or np.nanmax([ok.sum() for _, ok in pr]) < 60:
        return None
    k = int(np.nanargmax(r))
    blocks = []
    for c in pd.unique(cv[pd.notna(cv)]):
        pos = np.flatnonzero(cv == c)
        blocks += [pos[s_:s_ + 5] for s_ in range(0, pos.size, 5)]
    boot_lag = []
    for _ in range(300):
        w = np.zeros(idx.size)
        for j in rng.integers(0, len(blocks), len(blocks)):
            w[blocks[j]] += 1.0
        rb = corr(pr, w)
        if np.isfinite(rb).any():
            boot_lag.append(lags[int(np.nanargmax(rb))])
    sur = []
    b0 = b.reindex(idx)
    for _ in range(200):
        bs = b0.copy()
        for c in pd.unique(cv[pd.notna(cv)]):
            pos = np.flatnonzero((cv == c) & np.isfinite(b0.values))
            if pos.size > 8:
                bs.iloc[pos] = P.surrogate(b0.values[pos], rng)
        sur.append(np.nanmax(corr(pairs(bs), ones)))
    return dict(lag=int(lags[k]), r=float(r[k]), lag_lo=float(np.percentile(boot_lag, 2.5)),
                lag_hi=float(np.percentile(boot_lag, 97.5)), level=float(np.percentile(sur, 95)),
                n_days=int(pr[list(lags).index(0)][1].sum()), r_zero=float(r[list(lags).index(0)]))


# ------------------------------------------------------------- 4 transport
def transport(site, lp, frame_dev, rng):
    d, kb, ki = hourly_bottom(site, lp)
    if kb is None or ki is None:
        return []
    zb = V.WATER_DEPTH_M[site] - V.BOTTOM_LEVEL[site]
    zi = V.WATER_DEPTH_M[site] - V.INTERIOR_LEVEL[site]
    camp = d['campaign']
    rows = []
    for Tc in V.SACW_T_TESTS:
        for rot in (-frame_dev, 0.0, frame_dev):
            c_, s_ = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
            vb = d['v'][:, kb] * c_ - d['u'][:, kb] * s_
            vi = d['v'][:, ki] * c_ - d['u'][:, ki] * s_
            M = zb * vb / 2.0 + (zi - zb) * (vb + vi) / 2.0
            ind = (d['T'][:, kb] < Tc).astype(float)
            ok = np.isfinite(M) & np.isfinite(d['T'][:, kb]) & (camp != '')
            if ok.sum() < 240:
                continue
            m, p = M[ok], ind[ok]
            q_mean = m.mean() * p.mean()
            q_fl = np.mean((m - m.mean()) * (p - p.mean()))
            # moving blocks of five days for the interval of the total and of its terms
            idx = np.flatnonzero(ok)
            nb = 24 * V.BOOT_BLOCK_DAYS
            starts = np.arange(0, idx.size, nb)
            boot = []
            for _ in range(V.N_BOOT):
                j = np.concatenate([np.arange(s, min(s + nb, idx.size))
                                    for s in rng.choice(starts, starts.size)])
                mb, pb = m[j], p[j]
                boot.append((mb.mean() * pb.mean(), np.mean((mb - mb.mean()) * (pb - pb.mean()))))
            boot = np.array(boot)
            rows.append(dict(site=site, T_c=Tc, rotation=rot, n_hours=int(ok.sum()), share_central=p.mean(),
                             M_mean=m.mean(), Q_total=q_mean + q_fl, Q_mean=q_mean, Q_fluct=q_fl,
                             Q_mean_lo=np.percentile(boot[:, 0], 2.5), Q_mean_hi=np.percentile(boot[:, 0], 97.5),
                             Q_fluct_lo=np.percentile(boot[:, 1], 2.5), Q_fluct_hi=np.percentile(boot[:, 1], 97.5)))
    g = [r for r in rows if r['T_c'] == V.SACW_T]
    if g:
        r0 = [r for r in g if r['rotation'] == 0.0][0]
        tot = [r['Q_total'] for r in g]
        say(f'    transport of central water below {V.SACW_T:.0f} degrees over {r0["n_hours"]} hours: '
            f'{r0["Q_total"]:+.3f} m2 s-1 ({min(tot):+.3f} to {max(tot):+.3f} over the frame), the mean flow '
            f'{r0["Q_mean"]:+.3f} and the fluctuations {r0["Q_fluct"]:+.3f} ({r0["Q_fluct_lo"]:+.3f} to '
            f'{r0["Q_fluct_hi"]:+.3f}), central water present on {100 * r0["share_central"]:.0f} % of the hours')
    return rows


def flux(site, lp, frame_dev, rng):
    d, kb, _ = hourly_bottom(site, lp)
    if kb is None:
        return []
    camp = d['campaign']
    T = d['T'][:, kb]
    rows = []
    for Tc in V.SACW_T_TESTS:
        for rot in (-frame_dev, 0.0, frame_dev):
            c_, s_ = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
            v = d['v'][:, kb] * c_ - d['u'][:, kb] * s_
            ok = np.isfinite(v) & np.isfinite(T) & (camp != '')
            if ok.sum() < 240:
                continue
            groups = [np.flatnonzero(ok & (camp == c)) for c in np.unique(camp[ok])]
            groups = [g for g in groups if g.size >= 48]

            def terms(sel):
                fm, fe, n = 0.0, 0.0, 0
                for g in sel:
                    vv, tt = v[g], np.minimum(T[g] - Tc, 0.0)    # the cold deficit, zero in warmer water
                    fm += -vv.mean() * tt.mean() * g.size
                    fe += -np.mean((vv - vv.mean()) * (tt - tt.mean())) * g.size
                    n += g.size
                return fm / n, fe / n

            fm, fe = terms(groups)
            nb = 24 * V.BOOT_BLOCK_DAYS
            boot = []
            for _ in range(V.N_BOOT):
                sel = []
                for g in groups:
                    starts = np.arange(0, g.size, nb)
                    j = np.concatenate([np.arange(s0, min(s0 + nb, g.size)) for s0 in rng.choice(starts, starts.size)])
                    sel.append(g[j])
                boot.append(terms(sel))
            boot = np.array(boot)
            n_days = sum(g.size for g in groups) * V.DT_H / 24.0
            rows.append(dict(site=site, T_c=Tc, rotation=rot, n_days=n_days, flux_total=fm + fe,
                             flux_mean=fm, flux_eddy=fe,
                             flux_mean_lo=np.percentile(boot[:, 0], 2.5), flux_mean_hi=np.percentile(boot[:, 0], 97.5),
                             flux_eddy_lo=np.percentile(boot[:, 1], 2.5), flux_eddy_hi=np.percentile(boot[:, 1], 97.5),
                             short=int(n_days < V.MIN_FLUX_DAYS)))
    g = [r for r in rows if r['T_c'] == V.SACW_T]
    if g:
        r0 = [r for r in g if r['rotation'] == 0.0][0]
        tot = [r['flux_total'] for r in g]
        say(f'    near-bottom flux of the deficit below {V.SACW_T:.0f} degrees over {r0["n_days"]:.0f} days: '
            f'{r0["flux_total"]:+.4f} degree m s-1 ({min(tot):+.4f} to {max(tot):+.4f} over the frame), mean flow '
            f'{r0["flux_mean"]:+.4f} ({r0["flux_mean_lo"]:+.4f} to {r0["flux_mean_hi"]:+.4f}) and correlated '
            f'fluctuations {r0["flux_eddy"]:+.4f} ({r0["flux_eddy_lo"]:+.4f} to {r0["flux_eddy_hi"]:+.4f})'
            + (', too short for the figure' if r0['short'] else ''))
    return rows


def main():
    V.make_dirs()
    rng = np.random.default_rng(606)
    say('figure 6, the path of the central water from the shelf break to the coast')
    geo, _ = I.geometry(verbose=False)
    ln, _ = I.lines(geo, verbose=False)
    lp = I.lp_files(verbose=False)
    fr = pd.read_csv(V.DIR_TAB / 'fig01_frame.csv').set_index('site')
    f5 = V.DIR_TAB / 'fig05_daily.csv'
    d5 = pd.read_csv(f5, parse_dates=['day']) if f5.exists() else pd.DataFrame()
    index = coastal_index(geo)
    pd.DataFrame(index).to_csv(V.DIR_TAB / 'fig06_index.csv')
    days = {s: daily_bottom(s, lp) for s in V.SITES if s in lp}
    events, comp, reach, lags, trans, fluxes = [], [], [], [], [], []

    for line, members in V.LINES.items():
        outer = members[-1]
        if outer not in lp:
            continue
        say(f'\n  {line}')
        for drop in V.EVENT_DROP_TESTS:
            ev, _ = onsets(outer, lp, drop)
            say(f'    {len(ev)} cold events at {outer} for a drop of {drop:g} degree in {V.EVENT_WINDOW_H} h')
            if drop != V.EVENT_DROP:
                continue
            for t, c, dT in ev:
                events.append(dict(line=line, site=outer, time=t, campaign=c, drop=dT))
            onset_days = pd.DatetimeIndex([t for t, _, _ in ev]).normalize()
            # the near-bottom temperature of the line, the coastal index and the wave
            series = {s: days[s]['T'] for s in members if s in days and len(days[s])}
            if line in index:
                series['coastal index'] = index[line]
            if len(d5) and outer in set(d5.site):
                g5 = d5[d5.site == outer].set_index('day')
                for col, name in (('wave', 'coastal trapped wave'), ('eta', 'uplift at 155 m')):
                    w = g5[col]
                    series[name] = (w - w.mean()) / w.std()
            dd, kb_, _ = hourly_bottom(outer, lp)
            if kb_ is not None:
                vb = pd.DataFrame({'v': dd['v'][:, kb_], 'campaign': dd['campaign']}, index=dd['time'])
                vb = vb[vb.campaign != '']
                gv = vb.groupby([vb.campaign, vb.index.floor('D')])
                vday = gv.mean(numeric_only=True).where(gv.count() >= V.MIN_DAY_HOURS)['v']
                vday = vday - vday.groupby(level=0).transform('mean')
                vday = vday.droplevel(0)
                series['cross-isobath velocity near the bottom'] = vday / vday.std()
            for name, s in series.items():
                rows, n = composite(s.groupby(level=0).mean(), onset_days, rng)
                comp += [dict(line=line, variable=name, n_events=n, **r) for r in rows]
                if rows:
                    lo = min(r['mean'] for r in rows if r['lag'] >= 0)
                    say(f'    composite of {name} over {n} events: {lo:+.2f} at its lowest after the onset')
            # the share of the events that reach each inner site
            for s in members[:-1]:
                if s not in days or not len(days[s]):
                    continue
                T = days[s]['T'].groupby(level=0).mean()
                hit, seen = 0, 0
                for d0 in onset_days:
                    base = T.reindex(d0 + pd.to_timedelta(np.arange(V.BASELINE_DAYS[0], V.BASELINE_DAYS[1] + 1),
                                                          unit='D'))
                    after = T.reindex(d0 + pd.to_timedelta(np.arange(0, V.REACH_DAYS + 1), unit='D'))
                    if base.notna().sum() == 0 or after.notna().sum() < V.REACH_DAYS // 2:
                        continue
                    seen += 1
                    hit += int((after - base.mean()).min() <= -V.REACH_DROP)
                reach.append(dict(line=line, site=s, events_seen=seen, events_reached=hit,
                                  share=hit / seen if seen else np.nan))
                say(f'    {s}: {hit} of the {seen} events it recorded cool it by {V.REACH_DROP} degree within '
                    f'{V.REACH_DAYS} days')
        # lags against the cross-shelf distance
        a_all = days[outer]
        a = a_all['anom'].groupby(level=0).mean()
        camp_a = a_all['campaign'].groupby(level=0).first()
        x_outer = float(ln.set_index('site').loc[outer, 'along_km'])
        for s in members[:-1] + (['coastal index'] if line in index else []):
            if s == 'coastal index':
                raw = index[line].groupby(level=0).mean()
                span = raw.reindex(a.index)
                b = pd.Series(detrended_anomaly(span.values, camp_a.values), index=a.index)
                x = x_outer
            else:
                if s not in days or not len(days[s]):
                    continue
                b = days[s]['anom'].groupby(level=0).mean()
                x = x_outer - float(ln.set_index('site').loc[s, 'along_km'])
            res = lagged(a, b, camp_a, rng)
            if res is None:
                say(f'    {s}: fewer than 60 days in common with {outer}')
                continue
            lags.append(dict(line=line, site=s, distance_km=x, **res))
            say(f'    {s} at {x:.0f} km from {outer}: largest correlation {res["r"]:+.2f} at a lag of {res["lag"]:+d} '
                f'days ({res["lag_lo"]:+.0f} to {res["lag_hi"]:+.0f}), {res["r_zero"]:+.2f} at no lag, level '
                f'{res["level"]:.2f}, over {res["n_days"]} days')

    say('\n  central water across the isobaths near the bottom and in the layer')
    for site in V.SITES:
        if site not in lp:
            continue
        say(f'  {site}')
        dev = float(fr.loc[site, 'bearing_dev_max']) if site in fr.index else 0.0
        fluxes += flux(site, lp, dev, rng)
        trans += transport(site, lp, dev, rng)

    for name, rows in (('fig06_events', events), ('fig06_composite', comp), ('fig06_reach', reach),
                       ('fig06_lags', lags), ('fig06_flux', fluxes), ('fig06_transport', trans)):
        pd.DataFrame(rows).to_csv(V.DIR_TAB / f'{name}.csv', index=False)
    (V.DIR_LOG / 'fig06_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()