"""
Quantities of Figure 5, what brings the central water to the 200 m isobath.

The near-bottom temperature at the 200 m sites is the temperature of the interior
just above the layer plus the anomaly of the layer, T_194 = T_155 + D. The layer
anomaly D was modelled in Figure 4 as the buoyancy that the Ekman transport of
the layer carries. This figure splits the variance of the near-bottom temperature
between the two parts and attributes the interior part, an adiabatic uplift of
the thermocline, to the two processes proposed for it, the coastal trapped waves
forced by the remote wind (Castro and Lee 1995, Dottori and Castro 2018) and the
cross-isobath position of the Brazil Current axis over the slope (Campos et al.
2000, Roughan and Middleton 2004, Palóczy et al. 2014).

 1  modes        empirical orthogonal functions of the daily temperature at the
                 four levels (Kundu et al. 1975), whose first mode of one sign
                 over the column is a heave of the thermocline
 2  partition    T'_194 = beta T'_155 + r from the daily departures from the campaign
                 means, beta the share of the interior heave that reaches the
                 lowest meter, beta^2 var(T_155) the variance the heave carries to
                 it and var(r) the variability of the layer itself, uncorrelated
                 with the interior
 3  uplift       eta = -(T_155 - <T_155>) / <dT/dz>, the vertical displacement of
                 the isotherms at 155 m in metres, positive upward, with the
                 gradient between 85 and 155 m averaged over the campaign
 4  wave         a first-mode coastal trapped wave forced by the ERA5 along-shore
                 wind stress along the 100 m isobath (Gill and Schumann 1974,
                 Clarke 1977, Battisti and Hickey 1984),

                     dY/dt + c dY/dx + Y / T_f = tau_x / rho_0,

                 solved along its characteristics,

                     Y(x_s, t) = (1 / rho_0 c) integral tau_x(x, t - (x_s - x) / c)
                                 exp(-(x_s - x) / (c T_f)) dx,

                 over the isobath south of each mooring, with c and T_f chosen on a
                 grid by the correlation with eta and tested in campaigns left out
 5  axis         the daily distance of the axis of the Brazil Current from each
                 mooring along a cross-isobath section through it, from the DUACS
                 geostrophic velocity, taken with a negative sign so that a positive
                 value is an axis closer to the mooring
 6  attribution  the variance of eta explained by the coastal trapped wave, by
                 the position of the axis and by both, with moving block intervals, correlations tested
                 against phase randomised surrogates (Ebisuzaki 1997), and the
                 ordinary, partial and multiple coherence of eta on the two
                 (Bendat and Piersol 2010) in the synoptic and the mesoscale band

Outputs in paper_v3
    tables/fig05_modes.csv         modes by site and level
    tables/fig05_partition.csv     split of the variance of the near-bottom temperature
    tables/fig05_ctw_grid.csv      correlation over the phase speed and the decay time
    tables/fig05_ctw.csv           chosen parameters and skill in campaigns left out
    tables/fig05_attribution.csv   variance of eta explained, with intervals
    tables/fig05_coherence.csv     coherence spectra
    tables/fig05_daily.csv         daily eta, wave hindcast and position of the axis
    logs/fig05_compute.txt

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


# --------------------------------------------------------- daily records
def daily_records(site, lp):
    d = I.load_lp(lp[site])
    lv = list(d['level'])
    cols = {f'T{int(z)}': d['T'][:, lv.index(z)] for z in V.LEVELS[site] if z in lv}
    df = pd.DataFrame(cols, index=d['time'])
    df['campaign'] = d['campaign']
    df = df[df.campaign != '']
    g = df.groupby([df.campaign, df.index.floor('D')])
    day = g.mean(numeric_only=True)
    cnt = g.count()
    day = day.where(cnt[day.columns] >= V.MIN_DAY_HOURS)
    return day.reset_index(names=['campaign', 'day'])


def anomalies(day, cols):
    out = day.copy()
    for c in cols:
        out[c] = day[c] - day.groupby('campaign')[c].transform('mean')
    return out


# ------------------------------------------------------------- 1 modes
def modes(site, day):
    """Empirical orthogonal functions of the daily temperature over the levels that leave the
    most complete days, the sparsest level being dropped while fewer than 60 days remain."""
    cols = [c for c in (f'T{z}' for z in V.LEVELS[site]) if c in day and day[c].notna().mean() > 0.2]
    an = anomalies(day, cols)
    while len(cols) > 3 and len(an[cols].dropna()) < 60:
        cols.remove(min(cols, key=lambda c: an[c].notna().sum()))
    a = an[cols].dropna()
    if len(a) < 60 or len(cols) < 3:
        say(f'    modes left out, {len(a)} complete days over {len(cols)} levels')
        return []
    C = np.cov(a.values.T)
    val, vec = np.linalg.eigh(C)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    frac = val / val.sum()
    rows = []
    for m in range(2):
        v = vec[:, m] * np.sqrt(val[m])
        if v.sum() < 0:
            v = -v
        for c, amp in zip(cols, v):
            rows.append(dict(site=site, mode=m + 1, level=int(c[1:]), amplitude=float(amp),
                             fraction=float(frac[m]), n_days=len(a)))
    one_sign = np.all(vec[:, 0] * np.sign(vec[:, 0].sum()) > 0)
    say(f'    modes of the temperature over {len(a)} days at {" ".join(c[1:] for c in cols)} m: the first carries '
        f'{100 * frac[0]:.0f} % with {"one sign" if one_sign else "a change of sign"} over the column, the '
        f'second {100 * frac[1]:.0f} %')
    return rows


# --------------------------------------------------------- 2 partition
def partition(site, day):
    """The near-bottom temperature split into the part of the interior variability that reaches
    it and the variability of the layer itself, T'_194 = beta T'_155 + r, with r uncorrelated
    with the interior, so that beta^2 var(T_155) + var(r) = var(T_194) and beta is the share of
    the interior heave that reaches the lowest meter."""
    if not {'T155', 'T194'}.issubset(day.columns):
        return None
    a = anomalies(day, ['T155', 'T194'])[['T155', 'T194']].dropna()
    if len(a) < 60:
        say(f'    partition left out, {len(a)} days with both levels')
        return None
    vb = float(np.var(a.T194))
    beta = float(np.cov(a.T155, a.T194)[0, 1] / np.var(a.T155, ddof=1))
    resid = a.T194 - beta * a.T155
    D = a.T194 - a.T155
    row = dict(site=site, n_days=len(a), var_bottom=vb, beta=beta,
               transmitted=float(beta ** 2 * np.var(a.T155) / vb), own=float(np.var(resid) / vb),
               interior=float(np.var(a.T155) / vb), layer=float(np.var(D) / vb),
               covariance=float(2 * np.cov(a.T155, D)[0, 1] / vb))
    say(f'    near-bottom temperature over {len(a)} days: {100 * beta:.0f} % of the interior heave reaches the '
        f'lowest meter, the heave it carries is {100 * row["transmitted"]:.0f} % of its variance and the '
        f'variability of the layer itself {100 * row["own"]:.0f} %')
    return row


# ----------------------------------------------------- 4 the forcing path
def forcing_path(gebco, zname):
    field = P.slope_field(gebco, zname, V.CTW_EXTENT, margin=0.2)
    lo, la, along = P.isobath_path(field, V.CTW_ISOBATH, V.CTW_EXTENT,
                                   (V.CTW_EXTENT[0], V.CTW_EXTENT[2]))
    if la[0] > la[-1]:
        lo, la = lo[::-1], la[::-1]
    k = np.arange(0, lo.size, max(1, int(round(V.CTW_STEP_KM / V.PATH_STEP_KM))))
    lo, la = lo[k], la[k]
    c = np.cos(np.deg2rad(la))
    dx = np.diff(lo) * c[:-1] * P.R_EARTH_KM
    dy = np.diff(la) * P.R_EARTH_KM
    s = np.r_[0.0, np.cumsum(np.hypot(dx, dy))]
    t_e = np.r_[dx, dx[-1]]
    t_n = np.r_[dy, dy[-1]]
    nrm = np.hypot(t_e, t_n)
    say(f'  the {V.CTW_ISOBATH:.0f} m isobath from {la[0]:.1f} to {la[-1]:.1f} over {s[-1]:.0f} km, '
        f'{lo.size} points')
    return pd.DataFrame(dict(lon=lo, lat=la, along_km=s, t_east=t_e / nrm, t_north=t_n / nrm))


def path_stress(path):
    """Three-hourly along-path wind stress at every point of the path, positive in the
    direction in which the coastal trapped waves propagate, with the coast on their left."""
    fu = {f.name.split('_')[-1][:4]: f for f in I.era5_files('u10')}
    fv = {f.name.split('_')[-1][:4]: f for f in I.era5_files('v10')}
    parts = []
    for y in sorted(set(fu) & set(fv)):
        du = I.standardise(xr.open_dataset(fu[y]))
        dv = I.standardise(xr.open_dataset(fv[y]))
        nu = 'u10' if 'u10' in du else next(k for k in du.data_vars if k.lower().startswith('u'))
        nv = 'v10' if 'v10' in dv else next(k for k in dv.data_vars if k.lower().startswith('v'))
        la_i = xr.DataArray(path.lat.values, dims='p')
        lo_i = xr.DataArray(path.lon.values, dims='p')
        u = du[nu].sel(lat=la_i, lon=lo_i, method='nearest').sel(time=slice(*V.PERIOD)).load()
        v = dv[nv].sel(lat=la_i, lon=lo_i, method='nearest').sel(time=slice(*V.PERIOD)).load()
        u, v = xr.align(u, v, join='inner')
        spd = np.hypot(u, v)
        cd = P.drag_large_pond(spd)
        tx, ty = V.RHO_AIR * cd * spd * u, V.RHO_AIR * cd * spd * v
        tau = tx * path.t_east.values[None, :] + ty * path.t_north.values[None, :]
        parts.append(tau.to_pandas())
        du.close()
        dv.close()
    tau = pd.concat(parts).sort_index()
    tau = tau[~tau.index.duplicated()].resample(f'{V.CTW_DT_H}h').mean()
    tau = tau.interpolate(limit=4, limit_area='inside')      # short gaps of the forcing
    say(f'  along-path wind stress on {tau.shape[0]} steps of {V.CTW_DT_H} h at {tau.shape[1]} points')
    return tau


def hindcasts(tau, path, x_site):
    """Wave amplitude at x_site for every phase speed and decay time, daily means."""
    t = tau.index
    tt = (t - t[0]) / pd.Timedelta('1h')
    up = np.flatnonzero(path.along_km.values <= x_site)
    L = (x_site - path.along_km.values[up]) * 1e3
    dx = np.r_[np.diff(path.along_km.values), V.CTW_STEP_KM][up] * 1e3
    T = tau.values[:, up]
    out = {}
    for c in V.CTW_C:
        lag_h = L / c / 3600.0
        Sh = np.empty_like(T)
        for j in range(up.size):
            Sh[:, j] = np.interp(tt - lag_h[j], tt, T[:, j], left=np.nan, right=np.nan)
        for tf in V.CTW_TF_DAYS:
            w = np.exp(-L / (c * tf * 86400.0)) * dx / (c * V.RHO0)
            y = np.nansum(Sh * w[None, :], axis=1)
            y[np.isnan(Sh).all(axis=1)] = np.nan
            out[(c, tf)] = pd.Series(y, index=t).resample('1D').mean()
    return out


# -------------------------------------------------- 5 position of the axis
def axis_position(site, geo, alt, gebco, zname):
    r = geo.loc[site]
    off = np.deg2rad((r.bearing + 90.0) % 360.0)          # offshore, to the right of the isobath
    along = np.deg2rad(r.bearing)
    x = np.arange(V.ENC_SECTION_KM[0], V.ENC_SECTION_KM[1] + 1e-6, 2.0)
    c = np.cos(np.deg2rad(r.lat))
    la = r.lat + x * np.cos(off) / P.R_EARTH_KM
    lo = r.lon + x * np.sin(off) / (P.R_EARTH_KM * c)
    pts = dict(lat=xr.DataArray(la, dims='s'), lon=xr.DataArray(lo, dims='s'))
    depth = -gebco[zname].interp(**pts).values
    sub = alt.sel(time=slice(*V.PERIOD))
    ua = (sub['ugos'].interp(**pts) * np.sin(along) + sub['vgos'].interp(**pts) * np.cos(along)).load().values
    search = (depth >= V.SHELF_BREAK) & np.isfinite(ua).any(axis=0)
    dist = np.full(ua.shape[0], np.nan)
    for k in range(ua.shape[0]):
        w = np.where(search, ua[k], np.nan)
        if np.isfinite(w).any():
            j = int(np.nanargmin(w))
            if w[j] <= V.ENC_AXIS_MIN:
                dist[k] = x[j]
    t = pd.DatetimeIndex(sub.time.values)
    say(f'    axis of the current found on {np.isfinite(dist).sum()} of {dist.size} days, median distance '
        f'{np.nanmedian(dist):.0f} km offshore of the mooring ({np.nanpercentile(dist, 25):.0f} to '
        f'{np.nanpercentile(dist, 75):.0f})')
    return pd.Series(-dist, index=t.normalize())


# --------------------------------------------------------- 6 statistics
def r2(y, X):
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 30:
        return np.nan
    A = np.c_[np.ones(ok.sum()), X[ok]]
    b = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    return 1.0 - np.var(y[ok] - A @ b) / np.var(y[ok])


def block_index(camp, n, rng, length=5):
    blocks = []
    for c in np.unique(camp):
        k = np.flatnonzero(camp == c)
        for a in range(0, k.size, length):
            blocks.append(k[a:a + length])
    pick = rng.integers(0, len(blocks), len(blocks))
    return np.concatenate([blocks[j] for j in pick])


def surrogate_level(y, x, camp, rng):
    """95th percentile of |r| between x and phase randomised copies of y built campaign by
    campaign, which keeps the spectrum of each record."""
    ok = np.isfinite(y) & np.isfinite(x)
    rs = []
    for _ in range(V.N_SURROGATE):
        ys = np.full(y.size, np.nan)
        for c in np.unique(camp[ok]):
            k = np.flatnonzero(ok & (camp == c))
            if k.size > 8:
                ys[k] = P.surrogate(y[k], rng)
        m = np.isfinite(ys) & ok
        rs.append(abs(np.corrcoef(ys[m], x[m])[0, 1]))
    return float(np.percentile(rs, 95))


def main():
    V.make_dirs()
    rng = np.random.default_rng(505)
    say('figure 5, what brings the central water to the 200 m isobath')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    gebco, zname = I.open_gebco()
    alt = I.open_altimetry()
    path = forcing_path(gebco, zname)
    tau = path_stress(path)
    mod4 = pd.read_csv(V.DIR_TAB / 'fig04_model.csv') if (V.DIR_TAB / 'fig04_model.csv').exists() else pd.DataFrame()
    modes_rows, parts, grid_rows, ctw_rows, attr, coh, daily_out = [], [], [], [], [], [], []
    for site in V.SITES_200:
        if site not in lp or site not in geo.index:
            continue
        say(f'\n  {site}')
        day = daily_records(site, lp)
        modes_rows += modes(site, day)
        p = partition(site, day)
        if p is not None:
            parts.append(p)
        if not {'T85', 'T155'}.issubset(day.columns):
            say('    no uplift, the levels of 85 and 155 m are not both held')
            continue
        grad = ((day.T85 - day.T155) / 70.0).groupby(day.campaign).transform('mean')
        eta = -(day.T155 - day.groupby('campaign').T155.transform('mean')) / grad
        frame = pd.DataFrame({'campaign': day.campaign.values, 'eta': eta.values},
                             index=pd.DatetimeIndex(day.day))

        # ---------------------------------------------- the coastal trapped wave
        d_path = P.distance_km(geo.loc[site, 'lat'], geo.loc[site, 'lon'], path.lat.values, path.lon.values)
        x_site = float(path.along_km.values[int(np.argmin(d_path))])
        say(f'    wave evaluated {x_site:.0f} km along the isobath, {d_path.min():.0f} km from the mooring')
        H = hindcasts(tau, path, x_site)
        camp = frame.campaign.values
        y = frame.eta.values

        def anom(s):
            v = s.reindex(frame.index).values
            out = np.full(v.size, np.nan)
            for c in np.unique(camp):
                k = camp == c
                out[k] = v[k] - np.nanmean(v[k])
            return out

        Ha = {k: anom(v) for k, v in H.items()}
        for (c, tf), h in Ha.items():
            ok = np.isfinite(h) & np.isfinite(y)
            r = np.corrcoef(h[ok], y[ok])[0, 1] if ok.sum() > 30 else np.nan
            grid_rows.append(dict(site=site, c=c, tf_days=tf, r=r))
        g = pd.DataFrame([x for x in grid_rows if x['site'] == site])
        best = g.loc[g.r.abs().idxmax()]
        # the choice of c and T_f and its skill in campaigns left out
        cv_true, cv_pred, chosen = [], [], []
        for c_out in np.unique(camp):
            tr, te = camp != c_out, camp == c_out
            if te.sum() < 20:
                continue
            sc = {k: (np.corrcoef(h[tr & np.isfinite(h) & np.isfinite(y)],
                                  y[tr & np.isfinite(h) & np.isfinite(y)])[0, 1]) for k, h in Ha.items()}
            k_best = max(sc, key=lambda k: abs(sc[k]) if np.isfinite(sc[k]) else -1)
            chosen.append(k_best)
            h = Ha[k_best]
            m = te & np.isfinite(h) & np.isfinite(y)
            A = np.c_[np.ones((tr & np.isfinite(h) & np.isfinite(y)).sum()), h[tr & np.isfinite(h) & np.isfinite(y)]]
            b = np.linalg.lstsq(A, y[tr & np.isfinite(h) & np.isfinite(y)], rcond=None)[0]
            cv_true.append(y[m])
            cv_pred.append(b[0] + b[1] * h[m])
        cv_true, cv_pred = np.concatenate(cv_true), np.concatenate(cv_pred)
        r_cv = np.corrcoef(cv_true, cv_pred)[0, 1]
        skill_cv = 1.0 - np.mean((cv_true - cv_pred) ** 2) / np.var(cv_true)
        h_best = Ha[(best.c, best.tf_days)]
        lvl = surrogate_level(y, h_best, camp, rng)
        cs = [k[0] for k in chosen]
        ctw_rows.append(dict(site=site, c=best.c, tf_days=best.tf_days, r=best.r, r_level=lvl,
                             r_cv=r_cv, skill_cv=skill_cv, c_min_cv=min(cs), c_max_cv=max(cs),
                             x_site_km=x_site))
        say(f'    wave: correlation {best.r:+.2f} for c {best.c:.0f} m s-1 and T_f {best.tf_days:g} days, level '
            f'of the surrogates {lvl:.2f}, in campaigns left out correlation {r_cv:+.2f} and skill '
            f'{skill_cv:.2f}, c chosen between {min(cs):.0f} and {max(cs):.0f} m s-1')

        # --------------------------------------------------- position of the axis
        E = anom(axis_position(site, geo, alt, gebco, zname))
        okE = np.isfinite(E) & np.isfinite(y)
        rE = np.corrcoef(E[okE], y[okE])[0, 1] if okE.sum() > 30 else np.nan
        lvlE = surrogate_level(y, E, camp, rng) if okE.sum() > 30 else np.nan
        say(f'    position of the axis: correlation {rE:+.2f} with the uplift, level of the surrogates {lvlE:.2f}')

        # ------------------------------------------------------------ attribution
        useE = okE.sum() > 30 and np.nanstd(E) > 0
        if not useE:
            say('    no usable position of the axis at this site, the attribution carries the wave alone')
            E = np.zeros(y.size)
        X = {'wave': h_best[:, None], 'axis': E[:, None], 'both': np.c_[h_best, E]}
        if not useE:
            X = {'wave': h_best[:, None]}
        full = np.isfinite(y) & np.isfinite(h_best) & np.isfinite(E)
        for name, Xk in X.items():
            val = r2(np.where(full, y, np.nan), Xk)
            boot = []
            idx_full = np.flatnonzero(full)
            for _ in range(V.N_BOOT):
                j = idx_full[block_index(camp[full], full.sum(), rng)]
                boot.append(r2(y[j], Xk[j]))
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            attr.append(dict(site=site, target='uplift', predictor=name, explained=val, lo=lo, hi=hi,
                             n_days=int(full.sum())))
            say(f'    uplift explained by {name}: {100 * val:.0f} % ({100 * lo:.0f} to {100 * hi:.0f}) over '
                f'{int(full.sum())} days')
        m4 = mod4[mod4.site == site] if len(mod4) else pd.DataFrame()
        for _, r4 in m4.iterrows():
            attr.append(dict(site=site, target='layer', predictor=f'layer model forced by the {r4.forcing} flow',
                             explained=r4.skill_cv, lo=np.nan, hi=np.nan, n_days=np.nan))

        # --------------------------------------------------------------- coherence
        blocks = []
        for c in np.unique(camp):
            k = np.flatnonzero((camp == c))
            M = np.c_[h_best[k], E[k], y[k]]
            M = pd.DataFrame(M).interpolate(limit=2, limit_area='inside').values
            good = np.isfinite(M).all(axis=1)
            # the longest runs of complete days inside the campaign
            edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a >= V.COH_SEG_DAYS:
                    blocks.append(M[a:b])
        f, S, nseg = P.cross_spectra(blocks, V.COH_SEG_DAYS)
        flat = S is not None and (np.nanmin(np.abs(S[1:, 1, 1])) == 0 or np.nanmin(np.abs(S[1:, 0, 0])) == 0)
        if S is not None and nseg >= 3 and not flat:
            g1, g2, p1, p2, mult = P.coherences(S)
            for k in range(1, f.size):
                coh.append(dict(site=site, freq_cpd=f[k], period_days=1.0 / f[k], wave=g1[k],
                                axis=g2[k], wave_partial=p1[k], axis_partial=p2[k],
                                multiple=mult[k], n_segments=nseg, level=P.coherence_level(nseg),
                                level_partial=P.coherence_level(nseg, 1)))
            for band, (a, b) in (('synoptic', V.SYNOPTIC_DAYS), ('mesoscale', V.MESOSCALE_DAYS)):
                sel = (1.0 / f[1:] >= a) & (1.0 / f[1:] <= b)
                if sel.any():
                    say(f'    {band} band over {nseg} segments: partial coherence {np.mean(p1[1:][sel]):.2f} with the '
                        f'wave and {np.mean(p2[1:][sel]):.2f} with the position of the axis, multiple '
                        f'{np.mean(mult[1:][sel]):.2f}, levels {P.coherence_level(nseg):.2f} and '
                        f'{P.coherence_level(nseg, 1):.2f}')
        else:
            say(f'    {nseg} segments of {V.COH_SEG_DAYS} days, too few for the coherence')
        A = np.c_[np.ones(full.sum()), h_best[full], E[full]]
        b = np.linalg.lstsq(A, y[full], rcond=None)[0]
        Aw = np.c_[np.ones(full.sum()), h_best[full]]
        bw = np.linalg.lstsq(Aw, y[full], rcond=None)[0]
        pred_w = np.full(y.size, np.nan)
        pred_b = np.full(y.size, np.nan)
        pred_w[full] = Aw @ bw
        pred_b[full] = A @ b
        daily_out.append(pd.DataFrame(dict(site=site, day=frame.index, campaign=camp, eta=y, wave=h_best,
                                           axis=E, eta_wave=pred_w, eta_both=pred_b)))
    for name, rows in (('fig05_modes', modes_rows), ('fig05_partition', parts), ('fig05_ctw_grid', grid_rows),
                       ('fig05_ctw', ctw_rows), ('fig05_attribution', attr), ('fig05_coherence', coh)):
        pd.DataFrame(rows).to_csv(V.DIR_TAB / f'{name}.csv', index=False)
    if daily_out:
        pd.concat(daily_out, ignore_index=True).to_csv(V.DIR_TAB / 'fig05_daily.csv', index=False)
    (V.DIR_LOG / 'fig05_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()