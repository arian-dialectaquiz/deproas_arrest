"""
Quantities of Figure 5, what brings the SACW to the 200 m isobath.

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
                 over the column is an uplift of the thermocline, with intervals from
                 blocks of days resampled inside the campaigns
 2  partition    T'_194 = beta T'_155 + r from the daily departures from the campaign
                 means, beta the share of the interior uplift that reaches the
                 lowest meter, beta^2 var(T_155) the variance the uplift carries to
                 it and var(r) the variability of the layer itself, uncorrelated
                 with the interior, with intervals from the same blocks
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
                 grid by the correlation with eta and tested in campaigns left out,
                 each correlation of the grid with its interval from the blocks
 5  axis         the daily distance of the axis of the Brazil Current from each
                 mooring along a cross-isobath section through it, from the DUACS
                 geostrophic velocity, taken with a negative sign so that a positive
                 value is an axis closer to the mooring
 6  attribution  the variance of eta explained by the coastal trapped wave, by
                 the position of the axis and by both, with moving block intervals, correlations tested
                 against phase randomised surrogates (Ebisuzaki 1997), and the
                 ordinary, partial and multiple coherence of eta on the two
                 (Bendat and Piersol 2010) in the synoptic and the mesoscale band,
                 each partial coherence with the interval of its Fisher transform
 7  parts        the uplift split inside each campaign into a slow part, its Lanczos
                 low-pass at the long limit of the synoptic band (10 days), and the
                 synoptic part left. Each part is correlated with the same part of
                 the wave, and the slow part also with the position of the axis,
                 against the surrogates
 8  response     the median uplift in quintiles of the coastal trapped wave (signed
                 so that its upwelling phase is positive) and of the position of
                 the axis, a dose and response that does not rest on the straight
                 line of a correlation. Each median is set against the range it
                 takes when the uplift is replaced by phase randomised surrogates
                 built campaign by campaign, which keep the persistence of the
                 uplift and of the predictor, and the rank correlation is set
                 against the level of the same surrogates. Days that share one
                 value of a predictor fall in one class, so a predictor that moves
                 by steps, as the axis found on the altimetry grid, gets fewer
                 classes.

Everything added to the earlier version (the intervals of the modes, the partition,
the grid, the reconstruction and the coherence, the parts and the response) draws
from a random stream of its own and only adds columns, so every number of the
earlier tables is reproduced unchanged.

Outputs in paper_v3
    tables/fig05_modes.csv         modes by site and level, with intervals
    tables/fig05_partition.csv     split of the variance of the near-bottom temperature, with intervals
    tables/fig05_ctw_grid.csv      correlation over the phase speed and the decay time, with intervals
    tables/fig05_ctw.csv           chosen parameters, skill in campaigns left out and the correlations
                                   of the synoptic and of the slow part
    tables/fig05_attribution.csv   variance of eta explained, with intervals
    tables/fig05_coherence.csv     coherence spectra, with the intervals of the partial coherences
    tables/fig05_response.csv      median uplift in quintiles of the wave and of the axis, with the range
                                   of the surrogates and the rank correlation
    tables/fig05_daily.csv         daily eta with its slow and synoptic parts, wave hindcast with its
                                   interval and position of the axis
    logs/fig05_compute.txt

Created by Arian Dialectaquiz.
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

import config_v3 as V
import inputs_v3 as I
import physics_v3 as P

warnings.simplefilter('ignore', RuntimeWarning)
LOG = []
RESPONSE_BINS = 5          # classes of equal size of each predictor in the dose and response
RESPONSE_MIN_DAYS = 15     # days a class needs to be kept
SLOW_CUTOFF_DAYS = V.SYNOPTIC_DAYS[1]              # the slow part lies beyond the synoptic band
SLOW_HALF_WINDOW = int(round(1.5 * SLOW_CUTOFF_DAYS))   # half window of its low-pass, days
SLOW_MIN_WEIGHT = 0.8      # share of the filter weight that must fall on days with data


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
def modes(site, day, rng):
    """Empirical orthogonal functions of the daily temperature over the levels that leave the
    most complete days, the sparsest level being dropped while fewer than 60 days remain. The
    intervals come from blocks of days resampled inside the campaigns, every resampled mode
    being given the sign of the mode of the record."""
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
    load = vec[:, :2] * np.sqrt(val[:2])
    load = load * np.where(load.sum(axis=0) < 0, -1.0, 1.0)
    camp = an.loc[a.index, 'campaign'].values
    b_load, b_frac = [], []
    for _ in range(V.N_BOOT):
        j = block_index(camp, camp.size, rng)
        vb, eb = np.linalg.eigh(np.cov(a.values[j].T))
        ob = np.argsort(vb)[::-1]
        vb, eb = vb[ob], eb[:, ob]
        lb = eb[:, :2] * np.sqrt(vb[:2])
        b_load.append(lb * np.where((lb * load).sum(axis=0) < 0, -1.0, 1.0))
        b_frac.append(vb[:2] / vb.sum())
    l_lo, l_hi = np.percentile(b_load, [2.5, 97.5], axis=0)
    f_lo, f_hi = np.percentile(b_frac, [2.5, 97.5], axis=0)
    rows = []
    for m in range(2):
        for i, c in enumerate(cols):
            rows.append(dict(site=site, mode=m + 1, level=int(c[1:]), amplitude=float(load[i, m]),
                             amp_lo=float(l_lo[i, m]), amp_hi=float(l_hi[i, m]),
                             fraction=float(frac[m]), fraction_lo=float(f_lo[m]), fraction_hi=float(f_hi[m]),
                             n_days=len(a)))
    one_sign = np.all(vec[:, 0] * np.sign(vec[:, 0].sum()) > 0)
    say(f'    modes of the temperature over {len(a)} days at {" ".join(c[1:] for c in cols)} m: the first carries '
        f'{100 * frac[0]:.0f} % ({100 * f_lo[0]:.0f} to {100 * f_hi[0]:.0f}) with '
        f'{"one sign" if one_sign else "a change of sign"} over the column, the second {100 * frac[1]:.0f} %')
    return rows


# --------------------------------------------------------- 2 partition
def partition(site, day, rng):
    """The near-bottom temperature split into the part of the interior variability that reaches
    it and the variability of the layer itself, T'_194 = beta T'_155 + r, with r uncorrelated
    with the interior, so that beta^2 var(T_155) + var(r) = var(T_194) and beta is the share of
    the interior uplift that reaches the lowest meter. The intervals come from blocks of days
    resampled inside the campaigns."""
    if not {'T155', 'T194'}.issubset(day.columns):
        return None
    an = anomalies(day, ['T155', 'T194'])
    a = an[['T155', 'T194']].dropna()
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
    camp = an.loc[a.index, 'campaign'].values
    xi, xb = a.T155.values, a.T194.values
    boot = []
    for _ in range(V.N_BOOT):
        j = block_index(camp, camp.size, rng)
        bj = np.cov(xi[j], xb[j])[0, 1] / np.var(xi[j], ddof=1)
        boot.append((bj, bj ** 2 * np.var(xi[j]) / np.var(xb[j])))
    (row['beta_lo'], row['transmitted_lo']), (row['beta_hi'], row['transmitted_hi']) = \
        (tuple(float(v) for v in q) for q in np.percentile(boot, [2.5, 97.5], axis=0))
    say(f'    near-bottom temperature over {len(a)} days: {100 * beta:.0f} % of the interior uplift reaches the '
        f'lowest meter, the uplift it carries is {100 * row["transmitted"]:.0f} % '
        f'({100 * row["transmitted_lo"]:.0f} to {100 * row["transmitted_hi"]:.0f}) of its variance and the '
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


def boot_correlation(y, x, sets):
    """95 % interval of the correlation between y and x over resampled sets of days."""
    rs = []
    for j in sets:
        m = np.isfinite(x[j]) & np.isfinite(y[j])
        rs.append(np.corrcoef(x[j][m], y[j][m])[0, 1] if m.sum() > 30 else np.nan)
    return np.nanpercentile(rs, [2.5, 97.5]) if np.isfinite(rs).any() else (np.nan, np.nan)


def slow_part(y, camp):
    """Lanczos low-pass of the daily uplift at SLOW_CUTOFF_DAYS (Duchon 1979), campaign by
    campaign. The weights are renormalised over the days with data, and a day is left
    undefined where those days hold less than SLOW_MIN_WEIGHT of the weight, which leaves
    out the ends of the campaigns and the neighbourhood of long gaps."""
    w = P.lanczos_weights(24.0 * SLOW_CUTOFF_DAYS, SLOW_HALF_WINDOW, dt_h=24.0)
    out = np.full(y.size, np.nan)
    for c in np.unique(camp):
        k = np.flatnonzero(camp == c)
        good = np.isfinite(y[k])
        num = np.convolve(np.where(good, y[k], 0.0), w, mode='same')
        den = np.convolve(good.astype(float), w, mode='same')
        out[k] = np.where(den >= SLOW_MIN_WEIGHT, num / np.where(den == 0, np.nan, den), np.nan)
    return out


def rank_correlation(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def correlation_and_level(y, x, camp, rng):
    """Correlation between y and x and the 95 % level of the surrogates of y."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() <= 30 or np.std(x[ok]) == 0:
        return np.nan, np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), surrogate_level(y, x, camp, rng)


def response(site, name, y, x, camp, rng):
    """Dose and response of the uplift y to the predictor x. The days are split into classes
    of equal size of x, days that share one value falling in one class, and the median
    uplift of each class is set against the range it takes when the uplift is replaced by
    phase randomised surrogates built campaign by campaign (Ebisuzaki 1997). The surrogates
    keep the spectrum of the uplift and leave the predictor as it is, so the range holds
    the persistence of both. The rank correlation is set against the same surrogates."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 60 or np.std(x[ok]) == 0:
        return []
    idx = np.flatnonzero(ok)
    xs = (x - np.mean(x[idx])) / np.std(x[idx])
    edges = np.unique(np.quantile(xs[idx], np.linspace(0.0, 1.0, RESPONSE_BINS + 1)))
    nb = edges.size - 1
    if nb < 2:
        return []
    cls = np.clip(np.searchsorted(edges, xs, side='right') - 1, 0, nb - 1)

    def medians(v, k):
        return np.array([np.median(v[k == b]) if (k == b).sum() >= 3 else np.nan for b in range(nb)])

    base, centre, raw = medians(y[idx], cls[idx]), medians(xs[idx], cls[idx]), medians(x[idx], cls[idx])
    count = np.array([(cls[idx] == b).sum() for b in range(nb)])
    rho = rank_correlation(xs[idx], y[idx])
    null, rho_null = [], []
    for _ in range(V.N_SURROGATE):
        ys = np.full(y.size, np.nan)
        for c in np.unique(camp[idx]):
            k = idx[camp[idx] == c]
            if k.size > 8:
                ys[k] = P.surrogate(y[k], rng)
        m = idx[np.isfinite(ys[idx])]
        null.append(medians(ys[m], cls[m]))
        rho_null.append(abs(rank_correlation(xs[m], ys[m])))
    n_lo, n_hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
    rho_level = float(np.percentile(rho_null, 95))
    keep = [b for b in range(nb) if count[b] >= RESPONSE_MIN_DAYS and np.isfinite(base[b])]
    rows = [dict(site=site, predictor=name, x=float(centre[b]), x_raw=float(raw[b]), eta=float(base[b]),
                 null_lo=float(n_lo[b]), null_hi=float(n_hi[b]), n_days=int(count[b]), rho=rho,
                 rho_level=rho_level) for b in keep]
    if rows:
        outside = sum(1 for r_ in rows if not r_['null_lo'] <= r_['eta'] <= r_['null_hi'])
        say(f'    response to the {name}: median uplift from {rows[0]["eta"]:+.0f} m in the lowest of {len(rows)} '
            f'classes (surrogates {rows[0]["null_lo"]:+.0f} to {rows[0]["null_hi"]:+.0f}) to {rows[-1]["eta"]:+.0f} m '
            f'in the highest ({rows[-1]["null_lo"]:+.0f} to {rows[-1]["null_hi"]:+.0f}), {outside} of {len(rows)} '
            f'classes outside the range of the surrogates, rank correlation {rho:+.2f} against a level of '
            f'{rho_level:.2f}')
    return rows


def coherence_interval(g2, n_seg, conditioned=0):
    """95 % interval of a squared coherence from n_seg segments, reduced by one degree of freedom
    for every conditioning input. The inverse hyperbolic tangent of the coherence is close to
    normal, with a variance and a bias of 1 / (2 n - 2) (Bendat and Piersol 2010, Thomson and
    Emery 2014). With independent segments the interval holds the true value 95 % of the times
    for squared coherences above 0.1 and about 90 % of the times for no coherence at all."""
    n = max(n_seg - conditioned, 2)
    w = np.arctanh(np.sqrt(np.clip(np.asarray(g2, float), 0.0, 0.999999)))
    bias, sd = 1.0 / (2 * n - 2), 1.0 / np.sqrt(2 * n - 2)
    return np.tanh(np.maximum(w - bias - 1.96 * sd, 0.0)) ** 2, np.tanh(w - bias + 1.96 * sd) ** 2


def main():
    V.make_dirs()
    rng = np.random.default_rng(505)
    rng_i = np.random.default_rng(506)        # stream of the added intervals, the earlier numbers stay as they were
    say('figure 5, what brings the SACW to the 200 m isobath')
    geo, _ = I.geometry(verbose=False)
    lp = I.lp_files(verbose=False)
    gebco, zname = I.open_gebco()
    alt = I.open_altimetry()
    path = forcing_path(gebco, zname)
    tau = path_stress(path)
    mod4 = pd.read_csv(V.DIR_TAB / 'fig04_model.csv') if (V.DIR_TAB / 'fig04_model.csv').exists() else pd.DataFrame()
    modes_rows, parts, grid_rows, ctw_rows, attr, coh, daily_out, resp = [], [], [], [], [], [], [], []
    for site in V.SITES_200:
        if site not in lp or site not in geo.index:
            continue
        say(f'\n  {site}')
        day = daily_records(site, lp)
        modes_rows += modes(site, day, rng_i)
        p = partition(site, day, rng_i)
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
        fin = np.flatnonzero(np.isfinite(y))
        sets = [fin[block_index(camp[fin], fin.size, rng_i)] for _ in range(V.N_BOOT)]
        for (c, tf), h in Ha.items():
            ok = np.isfinite(h) & np.isfinite(y)
            r = np.corrcoef(h[ok], y[ok])[0, 1] if ok.sum() > 30 else np.nan
            r_lo, r_hi = boot_correlation(y, h, sets) if ok.sum() > 30 else (np.nan, np.nan)
            grid_rows.append(dict(site=site, c=c, tf_days=tf, r=r, r_lo=r_lo, r_hi=r_hi))
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
        # the slow and the synoptic part of the uplift against the wave
        slow = slow_part(y, camp)
        syn = y - slow
        h_slow = slow_part(h_best, camp)
        h_syn = h_best - h_slow                    # the wave split as the uplift, like against like
        r_syn, lvl_syn = correlation_and_level(syn, h_syn, camp, rng_i)
        r_slw, lvl_slw = correlation_and_level(slow, h_slow, camp, rng_i)
        both_parts = np.isfinite(syn)
        share = float(np.var(syn[both_parts]) / np.var(y[both_parts])) if both_parts.sum() > 30 else np.nan
        ctw_rows[-1].update(r_syn=r_syn, r_syn_level=lvl_syn, r_slow=r_slw, r_slow_level=lvl_slw, syn_share=share)
        say(f'    synoptic part (periods under {SLOW_CUTOFF_DAYS:g} days, {100 * share:.0f} % of the variance of the '
            f'uplift): correlation {r_syn:+.2f} with the synoptic part of the wave, level {lvl_syn:.2f}. Slow part: '
            f'correlation {r_slw:+.2f} with the slow part of the wave, level {lvl_slw:.2f}')
        say(f'    wave: correlation {best.r:+.2f} for c {best.c:.0f} m s-1 and T_f {best.tf_days:g} days, level '
            f'of the surrogates {lvl:.2f}, in campaigns left out correlation {r_cv:+.2f} and skill '
            f'{skill_cv:.2f}, c chosen between {min(cs):.0f} and {max(cs):.0f} m s-1')

        # --------------------------------------------------- position of the axis
        E = anom(axis_position(site, geo, alt, gebco, zname))
        okE = np.isfinite(E) & np.isfinite(y)
        rE = np.corrcoef(E[okE], y[okE])[0, 1] if okE.sum() > 30 else np.nan
        lvlE = surrogate_level(y, E, camp, rng) if okE.sum() > 30 else np.nan
        say(f'    position of the axis: correlation {rE:+.2f} with the uplift, level of the surrogates {lvlE:.2f}')
        r_sa, lvl_sa = correlation_and_level(slow, E, camp, rng_i)
        ctw_rows[-1].update(r_slow_axis=r_sa, r_slow_axis_level=lvl_sa)
        say(f'    slow part of the uplift: correlation {r_sa:+.2f} with the position of the axis, level {lvl_sa:.2f}')

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
        # ------------------------------------------------------ dose and response
        resp += response(site, 'wave', y, np.sign(best.r) * h_best, camp, rng_i)
        if useE:
            resp += response(site, 'axis', y, E, camp, rng_i)
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
            p1_lo, p1_hi = coherence_interval(p1, nseg, 1)
            p2_lo, p2_hi = coherence_interval(p2, nseg, 1)
            for k in range(1, f.size):
                coh.append(dict(site=site, freq_cpd=f[k], period_days=1.0 / f[k], wave=g1[k],
                                axis=g2[k], wave_partial=p1[k], axis_partial=p2[k],
                                wave_partial_lo=p1_lo[k], wave_partial_hi=p1_hi[k],
                                axis_partial_lo=p2_lo[k], axis_partial_hi=p2_hi[k],
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
        pred_lo = np.full(y.size, np.nan)
        pred_hi = np.full(y.size, np.nan)
        pred_w[full] = Aw @ bw
        pred_b[full] = A @ b
        # interval of the reconstruction from the wave, the fit repeated over resampled blocks
        y_full, camp_full, fits = y[full], camp[full], []
        for _ in range(V.N_BOOT):
            j = block_index(camp_full, y_full.size, rng_i)
            fits.append(Aw @ np.linalg.lstsq(Aw[j], y_full[j], rcond=None)[0])
        pred_lo[full], pred_hi[full] = np.percentile(fits, [2.5, 97.5], axis=0)
        # the synoptic part of the reconstruction, each resampled fit filtered as the uplift
        def synoptic(v):
            z = np.full(y.size, np.nan)
            z[full] = v
            return z - slow_part(z, camp)

        pred_syn = synoptic(Aw @ bw)
        syn_lo, syn_hi = np.nanpercentile([synoptic(v) for v in fits], [2.5, 97.5], axis=0)
        daily_out.append(pd.DataFrame(dict(site=site, day=frame.index, campaign=camp, eta=y, wave=h_best,
                                           axis=E, eta_wave=pred_w, eta_wave_lo=pred_lo, eta_wave_hi=pred_hi,
                                           eta_both=pred_b, eta_slow=slow, eta_syn=syn, eta_wave_syn=pred_syn,
                                           eta_wave_syn_lo=syn_lo, eta_wave_syn_hi=syn_hi)))
    for name, rows in (('fig05_modes', modes_rows), ('fig05_partition', parts), ('fig05_ctw_grid', grid_rows),
                       ('fig05_ctw', ctw_rows), ('fig05_attribution', attr), ('fig05_coherence', coh),
                       ('fig05_response', resp)):
        pd.DataFrame(rows).to_csv(V.DIR_TAB / f'{name}.csv', index=False)
    if daily_out:
        pd.concat(daily_out, ignore_index=True).to_csv(V.DIR_TAB / 'fig05_daily.csv', index=False)
    (V.DIR_LOG / 'fig05_compute.txt').write_text('\n'.join(LOG) + '\n')
    say(f'\ntables in {V.DIR_TAB}')


if __name__ == '__main__':
    main()
