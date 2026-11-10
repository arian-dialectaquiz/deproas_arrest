"""
Quantities of Figure 4, the buoyancy of the layer and the time of the arrest.

The layer arrests because its Ekman transport carries the stratification of the
interior along the slope, stabilising the layer when the transport is upslope
and mixing it when it is downslope (Trowbridge and Lentz 1991, MacCready and
Rhines 1993, Brink and Lentz 2010a). The records hold the temperature at four
heights, so the figure tests that mechanism through the buoyancy of the layer
and sets it against the time the flow holds one state.

 1  stratification  buoyancy frequency of the layer between 155 and 194 m and the
                    share of unstable or mixed days, with the current (upslope
                    transport) and against it (downslope), intervals from
                    resampling whole states
 2  drift           change of the temperature difference D = T_194 - T_155 across
                    the layer since each reversal of the flow at 155 m, against
                    the time since the reversal over the arrest time of Figure 2
 3  layer model     the buoyancy balance of the layer forced by the Ekman
                    transport that the interior flow would carry,

                        dD/dt = -c_up F+ - c_dn F- - (D - D0) / tau_r,
                        F = alpha dT/dz (-u_i |u_i|) / |f|,

                    where u_i is the along-isobath velocity at 155 m, positive
                    against the current, so that F is positive for upslope
                    transport, F+ and F- its upslope and downslope parts, dT/dz
                    the interior gradient between 85 and 155 m, D0 the level of
                    the campaign and tau_r the time over which mixing and the
                    instabilities of the layer restore it. Each coefficient is
                    C_d / h, so h = C_d / c is the thickness over which the layer
                    spreads the transported water. For a given tau_r the model is
                    linear in c_up and c_dn and is solved exactly from the start
                    of every state with an exponential filter, tau_r is found on a
                    grid, intervals come from resampling whole states, and the
                    skill is the reduction of the squared error against
                    persistence in campaigns left out of the fit. The arrested
                    thickness over h is the share of the unarrested Ekman
                    transport that builds the buoyancy of the layer. The model is
                    run again with u_i replaced by the along-isobath velocity at
                    194 m, which measures the stress the layer actually carries,
                    and the two skills say which transport the buoyancy follows
 4  time            attenuation of the near-bottom flow, with the offset flow of
                    Figure 3 removed and every hour corrected to the interior
                    speed U_REF with the power law of the gain on the interior
                    speed, and its veering, against the time since the reversal,
                    with an exponential decay fitted to the attenuation and
                    resampled over whole states, whose e-folding time is the
                    observed arrest time
 5  regime          the complex gains of Figure 3 in terciles of the daily slope
                    Burger number of each site, kept as a table

Outputs in paper_v3
    tables/fig04_mixing.csv        stratification by state
    tables/fig04_drift.csv         observed and modelled change across the layer
    tables/fig04_model.csv         parameters of the layer model by site
    tables/fig04_model_tau.csv     squared error against tau_r
    tables/fig04_model_states.csv  observed and modelled change over each state
    tables/fig04_time.csv          attenuation and veering against time
    tables/fig04_decay.csv         fitted decay of the attenuation
    tables/fig04_regime.csv        gains by tercile of s
    logs/fig04_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []
DAY = 86400.0


def say(msg=''):
    print(msg)
    LOG.append(msg)


def states_of(d, lv):
    u = d['u'][:, lv.index(V.CONDITION_LEVEL)]
    label, idx = np.zeros(u.size), np.full(u.size, -1)
    for k, (a, b, sgn) in enumerate(P.runs_of_sign(u, d['campaign'])):
        label[a:b], idx[a:b] = sgn, k
    return label, idx


def block_ids(days, campaign, length=5):
    """Blocks of consecutive days inside each campaign, the units of the resampling."""
    out, bid, last = np.zeros(len(days), int), 0, None
    for k, (dd, c) in enumerate(zip(days, campaign)):
        if last is None or c != last[1] or (dd - last[0]).days > 1 or (k and (out[k - 1] == bid) and
                                                                      (dd - start).days >= length):
            bid += 1
            start = dd
        out[k] = bid
        last = (dd, c)
    return out


def ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(x[ok], y[ok], 1)
    return b, a, np.corrcoef(x[ok], y[ok])[0, 1]


# ---------------------------------------------------------- 3 layer model
def _segments(idx, ok, label):
    segs = []
    for k in range(idx.max() + 1):
        m = np.flatnonzero(idx == k)
        if m.size * V.DT_H < V.MIN_STATE_H or ok[m].mean() < V.MIN_STATE_COMPLETE:
            continue
        segs.append((k, m, float(label[m[0]])))
    return segs


def _design(segs, D, Fp, Fn, camp, tau_h, levels):
    """Target and columns of the layer model for one restoring time, stacked over states.
    D(t) = D_s e + D0 (1 - e) - c_up G+ - c_dn G-, with e = exp(-t / tau_r), G the
    exponentially weighted integral of the forcing since the start of the state and D0
    the level of each campaign, all of which enter linearly."""
    from scipy.signal import lfilter
    dt = V.DT_H * 3600.0
    a = np.exp(-V.DT_H / tau_h)
    ys, Xs, starts = [], [], []
    for k, m, _ in segs:
        t = np.arange(m.size) * V.DT_H
        e = np.exp(-t / tau_h)
        Ds = D[m][0]
        Gp = lfilter([dt], [1.0, -a], np.r_[0.0, Fp[m][1:]])
        Gn = lfilter([dt], [1.0, -a], np.r_[0.0, Fn[m][1:]])
        dummies = np.zeros((m.size, len(levels)))
        dummies[:, levels.index(camp[m[0]])] = 1.0 - e
        ys.append(D[m] - Ds * e)
        Xs.append(np.c_[-Gp, -Gn, dummies])
        starts.append(np.full(m.size, Ds))
    return np.concatenate(ys), np.vstack(Xs), np.concatenate(starts)


def _fit(segs, D, Fp, Fn, camp, taus):
    levels = sorted({camp[m[0]] for _, m, _ in segs})
    best, prof = None, []
    for tau in taus:
        y, X, _ = _design(segs, D, Fp, Fn, camp, tau, levels)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(np.sum((y - X @ coef) ** 2))
        prof.append((tau, sse))
        if best is None or sse < best[1]:
            best = (tau, sse, coef, levels)
    return best, prof


def layer_model(site, d, lv, idx, label, geo, daily, rng, forcing='interior'):
    up, zi_lev, bot = V.BUDGET_LEVELS
    if not all(z in lv for z in V.BUDGET_LEVELS):
        return None, None, None, None
    ku, ki, kb = lv.index(up), lv.index(zi_lev), lv.index(bot)
    lat = float(geo.loc[site, 'lat'])
    f = abs(P.coriolis(lat))
    alpha = float(geo.loc[site, 'slope'])
    s_ = pd.Series
    D = s_(d['T'][:, kb] - d['T'][:, ki]).interpolate(limit=6, limit_area='inside').to_numpy(copy=True)
    Tz = s_((d['T'][:, ku] - d['T'][:, ki]) / (zi_lev - up)).interpolate(limit=6, limit_area='inside') \
        .to_numpy(copy=True)
    kf = ki if forcing == 'interior' else kb
    u = s_(d['u'][:, kf]).interpolate(limit=6, limit_area='inside').to_numpy(copy=True)
    F = alpha * Tz * (-u * np.abs(u)) / f
    Fp, Fn = np.where(F > 0, F, 0.0), np.where(F < 0, F, 0.0)
    camp = d['campaign']
    ok = np.isfinite(D) & np.isfinite(F) & (camp != '')
    segs = _segments(idx, ok, label)
    if len(segs) < 12:
        say(f'    layer model left out, {len(segs)} complete states')
        return None, None, None, None
    for _, m, _ in segs:                       # the few gaps inside a complete state
        for arr in (D, Fp, Fn):
            arr[m] = s_(arr[m]).interpolate(limit_direction='both').to_numpy(copy=True)
    taus = np.geomspace(*V.TAU_GRID_H[:2], int(V.TAU_GRID_H[2]))
    (tau, sse, coef, levels), prof = _fit(segs, D, Fp, Fn, camp, taus)
    y, X, Ds_all = _design(segs, D, Fp, Fn, camp, tau, levels)
    e_all = np.concatenate([np.exp(-np.arange(m.size) * V.DT_H / tau) for _, m, _ in segs])
    pred = Ds_all * e_all + X @ coef
    obs = np.concatenate([D[m] for _, m, _ in segs])
    skill_in = 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - Ds_all) ** 2)

    # skill in campaigns left out of the fit
    camps_of = np.array([camp[m[0]] for _, m, _ in segs])
    se_m, se_p = 0.0, 0.0
    per_camp = {}
    for c in np.unique(camps_of):
        train = [sg for sg, cc in zip(segs, camps_of) if cc != c]
        test = [sg for sg, cc in zip(segs, camps_of) if cc == c]
        if len(train) < 8 or not test:
            continue
        (t_, _, c_, _), _ = _fit(train, D, Fp, Fn, camp, taus)
        # the coefficients and the restoring time come from the other campaigns, and only the
        # level of the campaign left out is fitted, in closed form
        yt, Xt, st = _design(test, D, Fp, Fn, camp, t_, [c])
        r_ = yt - Xt[:, :2] @ c_[:2]
        lev = np.sum(Xt[:, 2] * r_) / np.sum(Xt[:, 2] ** 2)
        et = np.concatenate([np.exp(-np.arange(m.size) * V.DT_H / t_) for _, m, _ in test])
        ot = np.concatenate([D[m] for _, m, _ in test])
        e_m = np.sum((ot - (st * et + Xt[:, :2] @ c_[:2] + Xt[:, 2] * lev)) ** 2)
        e_p = np.sum((ot - st) ** 2)
        se_m += e_m
        se_p += e_p
        per_camp[c] = (1.0 - e_m / e_p if e_p > 0 else np.nan, len(test))
    skill_cv = 1.0 - se_m / se_p if se_p > 0 else np.nan

    # intervals from resampling whole states
    boot = []
    for _ in range(min(V.N_BOOT, 200)):
        pick = [segs[j] for j in rng.integers(0, len(segs), len(segs))]
        (t_, _, c_, _), _ = _fit(pick, D, Fp, Fn, camp, taus)
        boot.append((t_, c_[0], c_[1]))
    boot = np.array(boot)
    q = lambda x: np.nanpercentile(x, [2.5, 97.5])
    g2 = daily[daily.site == site]
    N = float(np.nanmedian(g2.N)) if len(g2) else np.nan
    u_med = float(np.nanmedian(np.abs(u[ok])))
    h_a = f * u_med / (N ** 2 * alpha) if np.isfinite(N) else np.nan
    row = dict(site=site, forcing=forcing, n_states=len(segs), tau_h=tau, tau_lo=q(boot[:, 0])[0], tau_hi=q(boot[:, 0])[1],
               c_up=coef[0], c_up_lo=q(boot[:, 1])[0], c_up_hi=q(boot[:, 1])[1],
               c_dn=coef[1], c_dn_lo=q(boot[:, 2])[0], c_dn_hi=q(boot[:, 2])[1],
               ratio_up_dn=coef[0] / coef[1] if coef[1] else np.nan,
               ratio_lo=q(boot[:, 1] / boot[:, 2])[0], ratio_hi=q(boot[:, 1] / boot[:, 2])[1],
               skill_in=skill_in, skill_cv=skill_cv, h_arrested=h_a, u_median=u_med, N_median=N)
    for name, cd in (('', V.CD), ('_cd_lo', V.CD_RANGE[0]), ('_cd_hi', V.CD_RANGE[1])):
        row[f'h_up{name}'] = cd / coef[0] if coef[0] > 0 else np.nan
        row[f'h_dn{name}'] = cd / coef[1] if coef[1] > 0 else np.nan
    # share of the unarrested Ekman transport that builds the buoyancy of the layer, the
    # arrested thickness over the thickness C_d / c, with the interval of c
    for key in ('up', 'dn'):
        cc, lo_, hi_ = row[f'c_{key}'], row[f'c_{key}_lo'], row[f'c_{key}_hi']
        row[f'eff_{key}'] = h_a * cc / V.CD
        row[f'eff_{key}_lo'] = h_a * max(lo_, 0.0) / V.CD
        row[f'eff_{key}_hi'] = h_a * hi_ / V.CD
    for c, (sk, n) in per_camp.items():
        row[f'skill_campaign_{c}'] = sk
    say(f'    layer model forced by the {forcing} flow, over {len(segs)} states: restoring time {tau:.0f} h ({row["tau_lo"]:.0f} to '
        f'{row["tau_hi"]:.0f}), c_up {coef[0]:.2e} ({row["c_up_lo"]:.2e} to {row["c_up_hi"]:.2e}) and c_dn '
        f'{coef[1]:.2e} ({row["c_dn_lo"]:.2e} to {row["c_dn_hi"]:.2e}) m-1, upslope over downslope '
        f'{row["ratio_up_dn"]:.2f} ({row["ratio_lo"]:.2f} to {row["ratio_hi"]:.2f})')
    say(f'    thickness C_d / c of {row["h_up"]:.0f} m upslope and {row["h_dn"]:.0f} m downslope for C_d '
        f'{V.CD:.0e}, against an arrested thickness of {h_a:.0f} m, skill against persistence {skill_in:.2f} '
        f'in the fit and {skill_cv:.2f} in campaigns left out, share of the Ekman transport '
        f'{row["eff_up"]:.2f} ({row["eff_up_lo"]:.2f} to {row["eff_up_hi"]:.2f}) upslope and {row["eff_dn"]:.2f} '
        f'({row["eff_dn_lo"]:.2f} to {row["eff_dn_hi"]:.2f}) downslope')
    say('    skill of each campaign left out ' + ', '.join(f'{c} {sk:+.2f} over {n} states'
                                                          for c, (sk, n) in per_camp.items()))
    # observed and modelled change over each state, and the hourly modelled series
    per_state = []
    D_model = np.full(D.size, np.nan)
    pos = 0
    for n, (k, m, sgn) in enumerate(segs):
        pm = pred[pos:pos + m.size]
        D_model[m] = pm
        e = int(V.STATE_EDGE_H / V.DT_H)
        per_state.append(dict(site=site, state=k, kind='with the current' if sgn < 0 else 'against the current',
                              hours=m.size * V.DT_H, observed=np.mean(D[m][-e:]) - np.mean(D[m][:e]),
                              modelled=np.mean(pm[-e:]) - np.mean(pm[:e])))
        pos += m.size
    prof = pd.DataFrame(prof, columns=['tau_h', 'sse'])
    prof.insert(0, 'site', site)
    return pd.DataFrame([row]), prof, pd.DataFrame(per_state), D_model


# --------------------------------------------------------------- 2 time
def _decay(x, g, rng=None):
    """g = g_inf + (g_0 - g_inf) exp(-x / x_e), fitted to the composite."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(x) & np.isfinite(g)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan
    try:
        p, _ = curve_fit(lambda xx, gi, g0, xe: gi + (g0 - gi) * np.exp(-xx / xe), x[ok], g[ok],
                         p0=[np.nanmin(g), np.nanmax(g), 3.0], bounds=([0, 0, 0.01], [3, 3, 1000]),
                         maxfev=8000)
        return p[0], p[1], p[2]
    except Exception:                                  # noqa: BLE001
        return np.nan, np.nan, np.nan


def time_composites(site, d, lv, label, idx, daily, offset, D_model, rng):
    """Change across the layer since the reversal, observed and modelled, and the attenuation
    of the near-bottom flow corrected to the interior speed U_REF, with its veering."""
    if not all(z in lv for z in V.PAIRS['bottom']):
        return [], [], []
    ki, kb = lv.index(V.PAIRS['bottom'][0]), lv.index(V.PAIRS['bottom'][1])
    wi = d['u'][:, ki] + 1j * d['v'][:, ki]
    wb = d['u'][:, kb] + 1j * d['v'][:, kb] - offset      # the flow of its own of Figure 3 removed
    dT = d['T'][:, kb] - d['T'][:, ki]
    key = 'tau_c_h' if V.TIME_SCALE == 'canonical' else 'tau_w_h'
    g0 = daily[daily.site == site]
    tau = pd.Series(g0[key].values, index=pd.DatetimeIndex(g0.day)).groupby(level=0).median() \
        .reindex(d['time'].normalize()).values
    e = int(V.STATE_EDGE_H / V.DT_H)
    t_since = np.full(idx.size, np.nan)
    anom_o, anom_m = np.full(idx.size, np.nan), np.full(idx.size, np.nan)
    for k in range(idx.max() + 1):
        m = np.flatnonzero(idx == k)
        if m.size == 0:
            continue
        t_since[m] = np.arange(m.size) * V.DT_H + 0.5 * V.DT_H
        if np.isfinite(dT[m][:e]).any():
            anom_o[m] = dT[m] - np.nanmean(dT[m][:e])
        if D_model is not None and np.isfinite(D_model[m][:e]).any():
            anom_m[m] = D_model[m] - np.nanmean(D_model[m][:e])
    x = t_since / tau
    edges = np.logspace(*V.TIME_EDGES)
    centre = np.sqrt(edges[1:] * edges[:-1])
    speed = np.abs(wi)
    ratio = wb / wi
    drift, times, decays = [], [], []
    for kind, sel in (('with the current', label < 0), ('against the current', label > 0)):
        # the power law of the gain on the interior speed, from every hour of the kind
        use = sel & np.isfinite(ratio) & (speed >= V.MIN_SPEED) & (np.abs(ratio) > 0)
        beta = np.polyfit(np.log(speed[use]), np.log(np.abs(ratio[use])), 1)[0] if use.sum() > 100 else 0.0
        gain_ref = np.abs(ratio) * (V.U_REF / speed) ** beta
        veer = np.degrees(np.angle(ratio))
        bin_of = np.digitize(x, edges) - 1
        sts_all = np.unique(idx[sel & (idx >= 0)])
        rows_of = {s_: np.flatnonzero(idx == s_) for s_ in sts_all}

        def composite(hours):
            b_ = bin_of[hours]
            out = np.full((centre.size, 4), np.nan)
            for j in range(centre.size):
                h = hours[b_ == j]
                if h.size == 0:
                    continue
                ok_v = h[use[h]]
                out[j] = [np.nanmedian(anom_o[h]), np.nanmedian(anom_m[h]),
                          np.nanmedian(gain_ref[ok_v]) if ok_v.size else np.nan,
                          np.nanmedian(veer[ok_v]) if ok_v.size else np.nan]
            return out

        hours_all = np.concatenate([rows_of[s_] for s_ in sts_all]) if len(sts_all) else np.array([], int)
        base = composite(hours_all)
        n_st = np.array([np.unique(idx[hours_all][bin_of[hours_all] == j]).size for j in range(centre.size)])
        boots = []
        for _ in range(V.N_BOOT):
            pick = rng.choice(sts_all, sts_all.size)
            boots.append(composite(np.concatenate([rows_of[s_] for s_ in pick])))
        boots = np.array(boots)
        lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
        good = n_st >= V.MIN_BIN_STATES
        for j in np.flatnonzero(good):
            drift.append(dict(site=site, kind=kind, x=centre[j], n_states=int(n_st[j]),
                              dT=base[j, 0], dT_lo=lo[j, 0], dT_hi=hi[j, 0], dT_model=base[j, 1]))
            times.append(dict(site=site, kind=kind, scale=V.TIME_SCALE, x=centre[j], n_states=int(n_st[j]),
                              gain=base[j, 2], gain_lo=lo[j, 2], gain_hi=hi[j, 2],
                              veer=base[j, 3], veer_lo=lo[j, 3], veer_hi=hi[j, 3], beta=beta))
        gi, g0_, xe = _decay(centre[good], base[good, 2])
        xb = np.array([_decay(centre[good], b[good, 2])[2] for b in boots[:200]])
        tau_med = float(np.nanmedian(tau[sel]))
        decays.append(dict(site=site, kind=kind, beta=beta, g_inf=gi, g_0=g0_, x_e=xe,
                           x_e_lo=float(np.nanpercentile(xb, 2.5)), x_e_hi=float(np.nanpercentile(xb, 97.5)),
                           tau_median_h=tau_med, t_e_h=xe * tau_med))
        g = [r for r in times if r['kind'] == kind]
        h_ = [r for r in drift if r['kind'] == kind]
        if g:
            say(f'    {kind}: gain on the interior speed as its power {beta:+.2f}, attenuation at '
                f'{V.U_REF} m s-1 of {g[0]["gain"]:.2f} at {g[0]["x"]:.2f} and {g[-1]["gain"]:.2f} at '
                f'{g[-1]["x"]:.1f} arrest times, fitted decay from {g0_:.2f} to {gi:.2f} with an e-folding of '
                f'{xe:.1f} arrest times ({decays[-1]["x_e_lo"]:.1f} to {decays[-1]["x_e_hi"]:.1f}), '
                f'{xe * tau_med:.0f} h, veering {g[0]["veer"]:+.0f} to {g[-1]["veer"]:+.0f} deg')
        if h_:
            say(f'    {kind}: difference across the layer changed by {h_[-1]["dT"]:+.2f} degrees observed '
                f'and {h_[-1]["dT_model"]:+.2f} modelled at {h_[-1]["x"]:.1f} arrest times')
    return drift, times, decays


# ------------------------------------------------------------- 3 regime
def regime(site, d, lv, label, idx, daily, rng):
    if not all(z in lv for z in V.PAIRS['bottom']):
        return []
    ki, kb = lv.index(V.PAIRS['bottom'][0]), lv.index(V.PAIRS['bottom'][1])
    wi = d['u'][:, ki] + 1j * d['v'][:, ki]
    wb = d['u'][:, kb] + 1j * d['v'][:, kb]
    g = daily[daily.site == site]
    g = g[g.layer == 'lowest pair'] if (g.layer == 'lowest pair').sum() >= 30 else g
    s = pd.Series(g.s.values, index=pd.DatetimeIndex(g.day)).reindex(d['time'].normalize()).values
    ok = np.isfinite(wi) & np.isfinite(wb) & (label != 0) & np.isfinite(s)
    if ok.sum() < 720:
        return []
    edges = np.nanquantile(s[ok], [0, 1 / 3, 2 / 3, 1])
    rows = []
    for t_ in range(3):
        m = ok & (s >= edges[t_]) & (s <= edges[t_ + 1])
        up = label[m] < 0
        if up.sum() < 120 or (~up).sum() < 120:
            continue
        coef = P.complex_gains(wi[m], wb[m], up)
        ids = idx[m]
        sts = np.unique(ids)
        rows_of = {s_: np.flatnonzero(ids == s_) for s_ in sts}
        boot = []
        for _ in range(V.N_BOOT):
            j = np.concatenate([rows_of[s_] for s_ in rng.choice(sts, sts.size)])
            if up[j].sum() < 24 or (~up[j]).sum() < 24:
                continue
            boot.append(P.complex_gains(wi[m][j], wb[m][j], up[j]))
        boot = np.asarray(boot)
        for kind, k in (('with the current', 1), ('against the current', 2)):
            rows.append(dict(site=site, tercile=t_ + 1, s_median=float(np.nanmedian(s[m])),
                             s_lo=float(edges[t_]), s_hi=float(edges[t_ + 1]), kind=kind,
                             gain=float(abs(coef[k])),
                             gain_lo=float(np.percentile(np.abs(boot[:, k]), 2.5)),
                             gain_hi=float(np.percentile(np.abs(boot[:, k]), 97.5)),
                             veer=float(np.degrees(np.angle(coef[k]))), n_hours=int((up if k == 1 else ~up).sum())))
    for r in rows:
        say(f'    tercile {r["tercile"]} (s {r["s_median"]:.2f}) {r["kind"]}: gain {r["gain"]:.2f} '
            f'({r["gain_lo"]:.2f} to {r["gain_hi"]:.2f}), veering {r["veer"]:+.0f} deg')
    return rows


# --------------------------------------------------------------- 4 mixing
DAILY_N = []                  # daily N of the layer by kind of state, for the distributions of Figure 4a


def mixing(site, d, label, idx, daily, rng):
    g = daily[daily.site == site].copy()
    g = g[g.layer == 'lowest pair'] if (g.layer == 'lowest pair').sum() >= 30 else g
    if g.empty:
        return []
    hr = pd.DataFrame({'state': label, 'idx': idx}, index=d['time'])
    per_day = hr.groupby(hr.index.floor('D')).agg(state=('state', 'mean'),
                                                  idx=('idx', lambda x: np.bincount(x[x >= 0]).argmax()
                                                       if (x >= 0).any() else -1))
    g = g.join(per_day, on=pd.DatetimeIndex(g.day))
    g['kind'] = np.where(g.state < -0.5, 'with the current',
                         np.where(g.state > 0.5, 'against the current', ''))
    DAILY_N.append(g.loc[g.kind != '', ['site', 'day', 'kind', 'N']])     # the days of Figure 4a
    rows = []
    for kind in ('with the current', 'against the current'):
        h = g[g.kind == kind]
        if len(h) < 10:
            continue
        unstable = ~np.isfinite(h.N.values)
        sts = np.unique(h.idx.values)
        rows_of = {s_: np.flatnonzero(h.idx.values == s_) for s_ in sts}
        boot = []
        for _ in range(V.N_BOOT):
            j = np.concatenate([rows_of[s_] for s_ in rng.choice(sts, sts.size)])
            boot.append(100.0 * np.mean(unstable[j]))
        q = np.nanpercentile(h.N, [5, 25, 50, 75, 95])
        rows.append(dict(site=site, kind=kind, n_days=len(h), N_p5=q[0], N_q1=q[1], N_median=q[2],
                         N_q3=q[3], N_p95=q[4], unstable=100.0 * float(np.mean(unstable)),
                         unstable_lo=float(np.percentile(boot, 2.5)),
                         unstable_hi=float(np.percentile(boot, 97.5))))
        say(f'    {kind}: median N {q[2]:.2e} s-1, unstable or mixed on {rows[-1]["unstable"]:.0f} % '
            f'of {len(h)} days ({rows[-1]["unstable_lo"]:.0f} to {rows[-1]["unstable_hi"]:.0f})')
    return rows


def main():
    V.make_dirs()
    rng = np.random.default_rng(404)
    say('figure 4, the buoyancy of the layer and the time of the arrest')
    say(f'  time scaled by the {V.TIME_SCALE} arrest time')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    fr = pd.read_csv(V.DIR_TAB / 'fig01_frame.csv').set_index('site')
    daily = pd.read_csv(V.DIR_TAB / 'fig02_daily.csv', parse_dates=['day'])
    resp = pd.read_csv(V.DIR_TAB / 'fig03_response.csv')
    models, profs, per_states, drifts, times, decays, reg, mix = [], [], [], [], [], [], [], []
    for site in V.SITES_200:
        if site not in lp:
            continue
        d = I.load_lp(lp[site])
        lv = list(d['level'])
        if V.CONDITION_LEVEL not in lv:
            continue
        say(f'\n  {site}')
        label, idx = states_of(d, lv)
        mix += mixing(site, d, label, idx, daily, rng)
        mod, prof, pst, D_model = layer_model(site, d, lv, idx, label, geo, daily, rng, 'interior')
        if mod is not None:
            models.append(mod)
            profs.append(prof.assign(forcing='interior'))
            per_states.append(pst.assign(forcing='interior'))
        mod_b, prof_b, pst_b, _ = layer_model(site, d, lv, idx, label, geo, daily, rng, 'near-bottom')
        if mod_b is not None:
            models.append(mod_b)
            profs.append(prof_b.assign(forcing='near-bottom'))
            per_states.append(pst_b.assign(forcing='near-bottom'))
        r = resp[(resp.site == site) & (resp.lower == V.PAIRS['bottom'][1]) & (resp.half == 'all')]
        offset = complex(r.offset_u.iloc[0], r.offset_v.iloc[0]) if len(r) else 0j
        dr, tm, dc = time_composites(site, d, lv, label, idx, daily, offset, D_model, rng)
        drifts += dr
        times += tm
        decays += dc
        reg += regime(site, d, lv, label, idx, daily, rng)
    for name, parts in (('fig04_model', models), ('fig04_model_tau', profs), ('fig04_model_states', per_states)):
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(V.DIR_TAB / f'{name}.csv',
                                                                                 index=False)
    pd.DataFrame(drifts).to_csv(V.DIR_TAB / 'fig04_drift.csv', index=False)
    pd.DataFrame(times).to_csv(V.DIR_TAB / 'fig04_time.csv', index=False)
    pd.DataFrame(decays).to_csv(V.DIR_TAB / 'fig04_decay.csv', index=False)
    pd.DataFrame(reg).to_csv(V.DIR_TAB / 'fig04_regime.csv', index=False)
    pd.DataFrame(mix).to_csv(V.DIR_TAB / 'fig04_mixing.csv', index=False)
    (pd.concat(DAILY_N, ignore_index=True) if DAILY_N else pd.DataFrame(columns=['site', 'day', 'kind', 'N'])) \
        .to_csv(V.DIR_TAB / 'fig04_daily.csv', index=False)
    (V.DIR_LOG / 'fig04_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()
